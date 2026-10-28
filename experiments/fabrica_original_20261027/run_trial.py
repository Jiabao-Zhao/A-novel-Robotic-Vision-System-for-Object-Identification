"""Original Fabrica candidate stage on frozen CAD fits; object-only context."""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import signal
import time
import traceback

from . import native
import numpy as np

from experiments.cad_grasp_geometry_20261026.inputs import MANIFEST, local, read


ROOT = native.ROOT
BASE = ROOT/'outputs/fabrica_original_20261027'
CONFIG = dict(surface_budget=200, angle_budget=10, antipodal_threshold=0.95,
    clearance_cm=1., sparsity_cm=.05, native_offset_delta_cm=0., ground_z='-infinity',
    workers=4, case_timeout_seconds=90, seed=20261027)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def timed_out(_signum, _frame):
    raise TimeoutError('Original Fabrica did not finish within the external 90 second case limit')


def initialize():
    native.load()


def run_case(row, index, output):
    start = time.perf_counter()
    folder = Path(output)/'cases'/row['scene']/row['part']
    result = dict(row, status='started', native_pair_count=0, generated_pose_count=0,
        retained_pose_count=0, retained_pair_count=0, contact_pairs_world_m=[],
        retained_pair_indices=[], retained_poses=[], seed=CONFIG['seed']+index)
    try:
        if row['input_status'] != 'available':
            raise ValueError('Frozen observed CAD alignment unavailable; no pose fit will run')
        up = native.load()
        mesh = up.trimesh.load_mesh(local(row['mesh_path']), process=False, maintain_order=True)
        if not isinstance(mesh, up.trimesh.Trimesh):
            raise TypeError('Expected native CAD triangle mesh')
        transform = np.asarray(row['world_T_cad'], dtype=float)
        mesh.vertices = (np.asarray(mesh.vertices) @ transform[:3,:3].T + transform[:3,3])*100.
        ground_cm = 0.0
        highest_cm = float(np.max(mesh.vertices[:,2]))
        default_empty = highest_cm <= ground_cm+CONFIG['clearance_cm']
        result['native_default_ground_preflight'] = dict(
            ground_z_cm=ground_cm, highest_CAD_z_cm=highest_cm,
            saved_observed_table_z_cm=float(row['table_z_m'])*100.,
            required_contact_z_greater_than_cm=ground_cm+CONFIG['clearance_cm'],
            status='analytically_no_point_above_native_ground_gate' if default_empty
                else 'not_evaluated_with_native_default_ground',
            no_default_sampler_run=True,
            note='Original sampler has an unbounded loop; this is a vertex-bound preflight of literal ground_z=0, not a default-ground success claim. The observed table height is recorded separately; no fitted tilted ground plane is substituted.')
        old_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(CONFIG['case_timeout_seconds'])
        try:
            trace = native.generate(mesh, seed=result['seed'], surface_budget=CONFIG['surface_budget'],
                angle_budget=CONFIG['angle_budget'], ground_z=-np.inf)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        pairs_m = trace['pairs_cm']/100.
        generated, retained = [], []
        for pose in trace['generated']:
            record = dict(id=f"G{pose['grasp_id']:05d}", grasp_id=pose['grasp_id'],
                pair_index=pose['pair_index'], open_ratio=pose['open_ratio'],
                opening_m=.08*pose['open_ratio'], quat_wxyz=pose['quat_wxyz'].tolist(),
                world_T_fabrica_panda_hand=native.pose_metres(pose).tolist(),
                contacts_world_m=pairs_m[pose['pair_index']].tolist(),
                retained=pose['retained'], native_message=pose['native_message'])
            generated.append(record)
            if pose['retained']:
                retained.append(record)
        retained_indices = sorted({pose['pair_index'] for pose in retained})
        rejection = Counter(p['native_message'] for p in generated if not p['retained'])
        result.update(status='native_candidates_available' if retained else 'no_pose_retained_by_native_FCL',
            mesh_info=dict(vertices=len(mesh.vertices), triangles=len(mesh.faces),
                preprocessing='trimesh.load_mesh(process=False, maintain_order=True); only supplied metric alignment and m-to-cm conversion'),
            native_pair_count=len(pairs_m), generated_pose_count=len(generated),
            retained_pose_count=len(retained), retained_pair_count=len(retained_indices),
            contact_pairs_world_m=pairs_m[retained_indices].tolist(),
            retained_pair_indices=retained_indices, retained_poses=retained,
            all_sampled_pairs_world_m=pairs_m.tolist(),
            native_rejection_counts=dict(rejection),
            candidate_trace_path=str((folder/'candidate_trace.json').relative_to(ROOT)),
            all_retained_pose_path=str((folder/'retained_poses.json').relative_to(ROOT)),
            source_functions=trace['source_functions'])
        dump(folder/'candidate_trace.json', generated)
        dump(folder/'retained_poses.json', retained)
        np.savez_compressed(folder/'sampled_pairs.npz', pairs_world_m=pairs_m,
            retained_pair_indices=np.asarray(retained_indices, dtype=int))
    except TimeoutError as error:
        result.update(status='native_case_timeout_no_partial_result', error=str(error))
    except Exception as error:
        result.update(status='native_generation_error', error=str(error), traceback=traceback.format_exc())
    result['timing_seconds'] = time.perf_counter()-start
    result['no_vlm'] = result['no_robot_execution'] = result['no_physics_steps'] = True
    dump(folder/'result.json', result)
    return result


def audit(records):
    if len(records) != 96 or len({(r['scene'],r['part']) for r in records}) != 96:
        raise AssertionError('Expected all 96 unique placements')
    for row in records:
        pairs = np.asarray(row['contact_pairs_world_m'], dtype=float).reshape(-1,2,3)
        if len(pairs) != row['retained_pair_count']:
            raise AssertionError('Displayed pair count mismatch')
        if len(row['retained_poses']) != row['retained_pose_count']:
            raise AssertionError('Retained pose count mismatch')
        indices = {pose['pair_index'] for pose in row['retained_poses']}
        if indices != set(row['retained_pair_indices']):
            raise AssertionError('Pair display loses or adds a retained pair')
        for pose in row['retained_poses']:
            t = np.asarray(pose['world_T_fabrica_panda_hand'])
            if not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1]):
                raise AssertionError('Invalid native hand transform')
            r = t[:3,:3]
            if not np.allclose(r.T@r, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(r),1.):
                raise AssertionError('Invalid native rotation')
            if not (0 <= pose['open_ratio'] <= .95):
                raise AssertionError('Native Panda opening limit violated')
    return dict(records=len(records), all_retained_poses_saved=True,
        all_retained_pair_projections_saved=True, source_bodies_unchanged=True,
        no_VLM=True, no_robot_execution=True, no_physics_steps=True,
        limitation='Record/frame/count audit, not independent FCL or physical verification')


def main():
    start = time.perf_counter()
    inputs = read(MANIFEST)['records']
    if len(inputs) != 96:
        raise ValueError('Expected twelve frozen CAD parts in eight scenes')
    initialize()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/'runs'/stamp
    output.mkdir(parents=True, exist_ok=False)
    protected_paths = [MANIFEST,
        ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json',
        ROOT/'experiments/grconvnet_vlm_pickup_20261024/vlm_client.py',
        ROOT/'experiments/grconvnet_vlm_pickup_20261024/grasp_rank_prompt.txt',
        ROOT/'docs/cad_view_prompt.txt',
        ROOT/'outputs/fabrica_candidates_20261027/runs/20261027T185418Z/results.json',
        ROOT/'outputs/cad_grasp_method_comparison_20261026/aligned_contacts/20261027T152238Z/results.json']
    before = {str(p.relative_to(ROOT)):native.sha(p) for p in protected_paths if p.exists()}
    sources = [Path(__file__), Path(native.__file__), *sorted(native.VENDOR.rglob('*.py')),
        *sorted(native.VENDOR.rglob('*.obj')), native.VENDOR/'LICENSE']
    versions = {name:importlib.metadata.version(name) for name in ('numpy','scipy','trimesh','python-fcl','Rtree','tqdm')}
    manifest = dict(created_utc=stamp, method='Original Fabrica candidate functions and class methods in an object-only context',
        config=CONFIG, placements=96, parts=12, scenes=8,
        repository='https://github.com/yunshengtian/Fabrica',
        commit='215a30fe51e59299588a5b5e417a9cb934fb393e',
        dependency_versions=versions, protected_sha256=before,
        source_sha256={str(p.relative_to(ROOT)):native.sha(p) for p in sources},
        input_manifest=str(MANIFEST.relative_to(ROOT)),
        native_length_units='centimeters; exports use meters',
        pose_frame='world_T_fabrica_panda_hand; localY closes fingers; quaternion wxyz',
        context_changes=['Original constructor replaced by an explicit one-part identity-path context.',
            'Public ground_z=-infinity disables the native workspace-height gate.',
            'Ground collision manager is empty; no fixture, table, other loose parts or robot arm are supplied.',
            'Original target-clearance spheres, native roll sampling, Panda width and target/gripper FCL checks remain active.',
            'Trace wrappers record native candidates/decisions; verbose logging only is enabled inside native check.'],
        custom_downward_only=False, custom_one_degree_contact_rule=False,
        custom_pad_depth_search=False, custom_contact_centering=False,
        custom_pair_grouping=False, custom_quality_score=False,
        original_class_constructor_executed=False, original_generation_and_FCL_methods_executed=True,
        full_Fabrica_assembly_pipeline_executed=False,
        no_VLM=True, no_VLM_prompt_change=True, no_pickup_or_assembly_execution=True,
        no_physics_steps=True, no_new_pose_estimation=True, simulator_object_state_read=False,
        weights_downloaded=False)
    dump(output/'manifest.json', manifest)
    print('OUTPUT='+str(output), flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=CONFIG['workers'], initializer=initialize) as pool:
        jobs = [pool.submit(run_case, row, index, str(output)) for index,row in enumerate(inputs)]
        for job in as_completed(jobs):
            record = job.result()
            records.append(record)
            print(json.dumps({k:record[k] for k in ('scene','part','status','native_pair_count',
                'generated_pose_count','retained_pose_count','retained_pair_count','timing_seconds')}), flush=True)
    records.sort(key=lambda r:(r['scene'],r['part']))
    dump(output/'results.json', records)
    verification = audit(records)
    after = {str(p.relative_to(ROOT)):native.sha(p) for p in protected_paths if p.exists()}
    verification['protected_unchanged'] = before == after
    if before != after:
        raise AssertionError('Protected input/prior result/VLM file changed')
    dump(output/'audit.json', verification)
    by_part = {}
    for part in sorted({r['part'] for r in records}):
        rows = [r for r in records if r['part']==part]
        by_part[part] = dict(available=sum(r['retained_pose_count']>0 for r in rows), placements=len(rows),
            native_pairs=sum(r['native_pair_count'] for r in rows),
            generated_poses=sum(r['generated_pose_count'] for r in rows),
            retained_poses=sum(r['retained_pose_count'] for r in rows),
            retained_pairs=sum(r['retained_pair_count'] for r in rows),
            min_display_pairs=min(r['retained_pair_count'] for r in rows),
            max_display_pairs=max(r['retained_pair_count'] for r in rows))
    summary = dict(placements=len(records), available=sum(r['retained_pose_count']>0 for r in records),
        status_counts=dict(Counter(r['status'] for r in records)), by_part=by_part,
        native_pairs=sum(r['native_pair_count'] for r in records),
        generated_poses=sum(r['generated_pose_count'] for r in records),
        retained_poses=sum(r['retained_pose_count'] for r in records),
        retained_pairs=sum(r['retained_pair_count'] for r in records),
        elapsed_seconds=time.perf_counter()-start,
        native_default_ground_analytically_empty=sum(r.get('native_default_ground_preflight',{}).get('status')=='analytically_no_point_above_native_ground_gate' for r in records),
        audit=verification, no_VLM=True, no_execution=True,
        timing_note='Elapsed includes four-worker generation, native FCL checks, serialization and audit; rendering is separate.')
    dump(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
