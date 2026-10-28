"""Run CAD-only grasp selection on the 96 frozen observed NIST placements.

No simulator object poses, new perception, VLM, robot, physics or insertion
execution is used. Known final transforms remain the previously declared
nominal CAD task goals, with the task board placed at nominal workspace XY=0.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from experiments.cad_grasp_geometry_20261026.gallery import read_stl
from experiments.cad_grasp_geometry_20261026.inputs import local
from .pipeline import select_grasp
from .task_geometry import derive_task


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'outputs/cad_grasp_geometry_20261026/inputs.json'
HAND = ROOT / 'outputs/cad_grasp_geometry_20261026/hand_model.json'
BASE = ROOT / 'outputs/cad_grasp_selection_20261028/runs'
SETTINGS = dict(max_poses=100, surface_budget=200, orientation_samples=10,
                pre_opening_clearance_m=.001)
BASE_SEED = 20261028


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def prepare_target(part, vertices, faces):
    """Retain native scale; record the previously used gear-only seam weld.

    Original CAD files never change. Mean-cluster vertex welding uses 50 nm
    only for Gear_Large, preserves triangle order, and records actual movement.
    Every other target passes through without geometric changes.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    tolerance = 5e-8 if part == 'Gear_Large' else 0.
    mapping = np.arange(len(vertices), dtype=int)
    moved = vertices.copy()
    pair_count = 0
    if tolerance:
        pairs = cKDTree(vertices).query_pairs(tolerance, output_type='ndarray')
        pair_count = len(pairs)
        if len(pairs):
            graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(vertices), len(vertices)))
            count, mapping = connected_components(graph, directed=False)
            moved = np.column_stack([
                np.bincount(mapping, weights=vertices[:, axis], minlength=count)
                / np.bincount(mapping, minlength=count) for axis in range(3)])
    displacement = np.linalg.norm(moved[mapping] - vertices, axis=1)
    receipt = dict(part=part, method='mean of proximity-connected seam vertices'
                   if tolerance else 'unchanged native vertices',
                   proximity_tolerance_m=tolerance, merged_close_pair_count=pair_count,
                   original_vertex_count=len(vertices), validation_vertex_count=len(moved),
                   triangle_count=len(faces), triangle_order_preserved=True,
                   maximum_vertex_displacement_m=float(displacement.max()),
                   original_CAD_file_modified=False, native_metric_scale_preserved=True,
                   earlier_protocol='experiments/cad_grasp_geometry_20261026/run_trial.py:load_target')
    return moved, mapping[faces], mapping, receipt


def protected_paths():
    names = ['outputs/cad_grasp_geometry_20261026/inputs.json',
             'outputs/cad_grasp_geometry_20261026/hand_model.json',
             'experiments/runtime_antipodal_assembly_20261023/goal_audit.json',
             'experiments/grconvnet_vlm_pickup_20261024/vlm_client.py',
             'experiments/grconvnet_vlm_pickup_20261024/grasp_rank_prompt.txt',
             'docs/cad_view_prompt.txt',
             'simulation/nist_task_board_1.py',
             'outputs/simulation/nist_task_board_1/assets/provenance.json',
             'outputs/fabrica_original_20261027/runs/20261027T193122Z/results.json',
             'outputs/fabrica_original_20261027/runs/20261027T193122Z/summary.json',
             'outputs/fabrica_goal_geometry_20261027/runs/20261027T213516Z/results.json']
    return [ROOT / name for name in names if (ROOT / name).is_file()]


def unavailable(message):
    return dict(status='unavailable_inputs', selected_grasp=None,
                proposed_count=0, valid_count=0, message=message,
                missing_or_invalid_inputs=[message])


def run_part(part, rows, seed, output):
    start = time.perf_counter()
    mesh_path = local(rows[0]['mesh_path'])
    original_vertices, faces = read_stl(mesh_path)
    vertices, faces, mapping, weld = prepare_target(part, original_vertices, faces)
    task = derive_task(part, vertices, faces)
    task_record = serial(task)
    task_record.update(mesh_path=str(mesh_path.relative_to(ROOT)), mesh_sha256=sha(mesh_path),
                       validation_mesh=weld,
                       final_goal_frame='nominal task-board center XY=(0,0), original workspace Z',
                       original_static_scene_board_position_used=False,
                       simulator_object_poses_read=False)
    asset_dir = Path(output) / 'derived_tasks'
    dump(asset_dir / f'{part}.json', task_record)
    np.savez_compressed(asset_dir / f'{part}_validation_mesh.npz',
                        vertices=vertices, faces=faces, original_to_validation_vertex=mapping)
    hand = json.loads(HAND.read_text())
    task_ready = str(task.get('status', '')).startswith('available')
    spec = task.get('task_spec', {})
    receivers = []
    receiver_sources = []
    if task_ready:
        for item in spec['receivers']:
            path = local(item['mesh_path'])
            receiver_vertices, receiver_faces = read_stl(path)
            receivers.append(dict(name=item['name'], vertices=receiver_vertices,
                                  faces=receiver_faces,
                                  world_T_mesh=np.asarray(item['board_T_mesh'])))
            receiver_sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    records = []
    for row in sorted(rows, key=lambda item: item['scene']):
        case_start = time.perf_counter()
        diagnostics = []
        if row.get('input_status') != 'available' or row.get('world_T_cad') is None:
            selection = unavailable('Saved observed CAD alignment is unavailable.')
        elif not task_ready:
            selection = unavailable('CAD task geometry unresolved: ' + str(task.get('message', task.get('status'))))
        else:
            request = dict(vertices_cad_m=vertices, faces=faces, hand=hand,
                world_T_cad_pickup=row['world_T_cad'],
                world_T_cad_final=spec['goal_T_cad_board'],
                axis_point_cad_m=task['axis_point_cad_m'],
                axis_direction_cad=task['axis_direction_cad'],
                table_plane_world=row['table_plane_world'],
                protected_contact_check=task['protected_contact_check'],
                receivers=receivers, simulator_object_poses_read=False)
            selection = select_grasp(request, seed=seed, diagnostics=diagnostics, **SETTINGS)
        rejection_counts = Counter()
        for candidate in diagnostics:
            if candidate['validity']['status'] != 'valid':
                rejection_counts.update(candidate['validity'].get('reasons', ['unspecified rejection']))
        case = dict(scene=row['scene'], part=part,
            mesh_path=str(mesh_path.relative_to(ROOT)), world_T_cad=row['world_T_cad'],
            selection=selection, wall_seconds=time.perf_counter()-case_start,
            rejection_counts=dict(rejection_counts), seed=seed,
            pose_source=row['pose_source'], observed_alignment_status=row['input_status'],
            table_plane_world=row['table_plane_world'],
            derived_task_path=f'derived_tasks/{part}.json',
            final_pose_status=spec.get('goal_status', 'unavailable'),
            final_pose_frame='nominal task-board center XY=(0,0), original workspace Z',
            no_VLM=True, no_pickup_or_assembly_execution=True,
            no_physics_steps=True, no_new_pose_estimation=True,
            no_insertion_or_release_checks=True, simulator_object_poses_read=False)
        folder = Path(output) / 'cases' / row['scene'] / part
        dump(folder / 'selection.json', case)
        dump(folder / 'candidate_diagnostics.json', diagnostics)
        records.append(case)
        print(json.dumps(dict(scene=row['scene'], part=part,
            status=selection['status'], proposed=selection['proposed_count'],
            valid=selection['valid_count'], seconds=round(case['wall_seconds'], 3))), flush=True)
    dump(asset_dir / f'{part}_sources.json', receiver_sources)
    return records, dict(part=part, seconds=time.perf_counter()-start,
                         task_status=task.get('status'), validation_mesh=weld)


def main(output=None, workers=4):
    started = time.perf_counter()
    rows = json.loads(INPUTS.read_text())['records']
    scenes = sorted({row['scene'] for row in rows})
    parts = sorted({row['part'] for row in rows})
    if len(rows) != 96 or scenes != [f'N{i:02d}' for i in range(8)] or len(parts) != 12:
        raise ValueError('Expected exactly twelve parts across all eight observed NIST scenes')
    output = Path(output) if output else BASE / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=False)
    before = {str(path.relative_to(ROOT)): sha(path) for path in protected_paths()}
    sources = [*Path(__file__).parent.glob('*.py'),
               ROOT / 'experiments/cad_grasp_geometry_20261026/geometry_kernel.py',
               ROOT / 'experiments/fabrica_candidates_20261027/sampler.py',
               ROOT / 'experiments/fabrica_goal_geometry_20261027/task_inputs.py']
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    dump(output / 'launch_manifest.json', dict(settings=SETTINGS, workers=workers,
        base_seed=BASE_SEED, seed_policy='one seed per CAD reused across scenes; runtime generation called for each case',
        source_sha256=source_hashes, protected_sha256=before,
        observed_manifest=str(INPUTS.relative_to(ROOT)),
        scope='candidate generation, modeled geometry checks and priority selection only',
        final_goal_scope='unchanged nominal board-centered goals; not verified mating transforms',
        VLM_selection=False, pickup_execution=False, assembly_execution=False,
        simulator_object_poses_read=False))
    results, part_times = [], []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        jobs = {executor.submit(run_part, part, [row for row in rows if row['part'] == part],
                                BASE_SEED+index, output): part for index, part in enumerate(parts)}
        for job in as_completed(jobs):
            part = jobs[job]
            records, timing = job.result()
            results.extend(records)
            part_times.append(timing)
            dump(output / 'progress.json', dict(completed_parts=len(part_times),
                completed_cases=len(results), last_part=part,
                selected=sum(row['selection']['status']=='selected' for row in results)))
    results.sort(key=lambda item: (item['scene'], item['part']))
    after = {str(path.relative_to(ROOT)): sha(path) for path in protected_paths()}
    source_after = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    summary = dict(placements=len(results), scenes=scenes, parts=parts,
        outcome_counts=dict(Counter(row['selection']['status'] for row in results)),
        proposed_count=sum(row['selection']['proposed_count'] for row in results),
        valid_count=sum(row['selection']['valid_count'] for row in results),
        per_part={part:dict(Counter(row['selection']['status'] for row in results if row['part']==part)) for part in parts},
        per_scene={scene:dict(Counter(row['selection']['status'] for row in results if row['scene']==scene)) for scene in scenes},
        elapsed_seconds=time.perf_counter()-started,
        sum_case_seconds=sum(row['wall_seconds'] for row in results), workers=workers,
        part_times=part_times, protected_inputs_unchanged=before==after,
        launched_source_unchanged=source_hashes==source_after,
        no_VLM_selection=True, no_pickup_or_assembly_execution=True,
        no_physics_steps=True, no_new_pose_estimation=True,
        no_insertion_or_release_checks=True, simulator_object_poses_read=False)
    dump(output / 'results.json', results)
    dump(output / 'summary.json', summary)
    dump(output / 'protected_hashes_after.json', after)
    with (output/'placement_counts.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['scene','part','status','proposed','valid','candidate_id',
                  'q_tau_m','q_surface_m','r_tau','r_surface','theta_down_rad','seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            selection = item['selection']
            grasp = selection.get('selected_grasp') or {}
            criteria = grasp.get('criteria', {})
            writer.writerow(dict(scene=item['scene'], part=item['part'], status=selection['status'],
                proposed=selection['proposed_count'], valid=selection['valid_count'],
                candidate_id=grasp.get('candidate_id'), seconds=item['wall_seconds'],
                **{key:criteria.get(key) for key in fields[6:11]}))
    if before != after or source_hashes != source_after:
        raise AssertionError('Protected input or launched source changed during run; preserve and audit this run')
    print('RUN_DIRECTORY=' + str(output), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    arguments = parser.parse_args()
    main(arguments.output, arguments.workers)
