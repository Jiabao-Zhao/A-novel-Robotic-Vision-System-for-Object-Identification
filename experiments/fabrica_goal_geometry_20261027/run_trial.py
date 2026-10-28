"""Filter the frozen Fabrica bank; never regenerate or move its grasps."""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from experiments.fabrica_original_20261027 import native
import numpy as np
from experiments.cad_grasp_geometry_20261026.inputs import local, read
from .endpoint import EndpointChecker, transform_points, welded_mesh
from .task_inputs import geometry_spec, evaluate_contact_pairs


ROOT = native.ROOT
SOURCE = ROOT/'outputs/fabrica_original_20261027/runs/20261027T193122Z'
BASE = ROOT/'outputs/fabrica_goal_geometry_20261027'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value.relative_to(ROOT))
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def run_case(scene, part, output):
    start = time.perf_counter()
    source_path = SOURCE/'cases'/scene/part/'result.json'
    source = read(source_path)
    poses = source['retained_poses']
    spec = geometry_spec(part)
    mesh = welded_mesh(local(source['mesh_path']))
    pick_T_cad = np.asarray(source['world_T_cad'])
    cad_T_pick = np.linalg.inv(pick_T_cad)
    pairs_world = np.asarray(source['all_sampled_pairs_world_m'])
    pairs_cad = transform_points(pairs_world.reshape(-1, 3), cad_T_pick).reshape(-1, 2, 3)
    regions = evaluate_contact_pairs(mesh, pairs_cad, spec)
    goal_hands = np.asarray([spec['goal_T_cad_board'] @ cad_T_pick @
                            np.asarray(p['world_T_fabrica_panda_hand']) for p in poses])
    checker = EndpointChecker(spec['receivers'])
    endpoint, endpoint_reasons = checker.check(goal_hands, [p['open_ratio'] for p in poses])
    region_ok = np.asarray([regions['eligible'][p['pair_index']] for p in poses])
    retained = []
    decisions = []
    for i, pose in enumerate(poses):
        keep = bool(region_ok[i] and endpoint[i])
        decisions.append(dict(id=pose['id'], pair_index=pose['pair_index'],
            mating_region_eligible=bool(region_ok[i]),
            mating_region_status=regions['status'],
            mating_region_reason=regions['reasons'][pose['pair_index']],
            endpoint_pass=bool(endpoint[i]), endpoint_reason=endpoint_reasons[i], retained=keep))
        if keep:
            retained.append(dict(pose, board_T_fabrica_panda_hand_at_goal=goal_hands[i].tolist()))
    pair_indices = sorted({p['pair_index'] for p in retained})
    region_available = not regions['status'].startswith('unavailable')
    status = ('geometry_candidates_available' if retained else 'no_candidate_passed_goal_geometry')
    if not region_available:
        status = 'endpoint_candidates_available' if retained else 'no_candidate_passed_endpoint_geometry'
    excluded = {'retained_poses', 'contact_pairs_world_m', 'all_sampled_pairs_world_m',
                'retained_pair_indices', 'candidate_trace_path', 'all_retained_pose_path'}
    result = {k: v for k, v in source.items() if k not in excluded}
    result.update(status=status, source_case_path=str(source_path.relative_to(ROOT)),
        source_case_sha256=native.sha(source_path),
        source_retained_pose_count=len(poses), source_retained_pair_count=source['retained_pair_count'],
        retained_pose_count=len(retained), retained_pair_count=len(pair_indices),
        retained_poses=retained, retained_pair_indices=pair_indices,
        contact_pairs_world_m=pairs_world[pair_indices].tolist(),
        mating_region_check_status=regions['status'],
        goal_geometry_note=regions['note'], geometry_spec=serial(spec),
        region_eligible_poses=int(region_ok.sum()), endpoint_pass_poses=int(endpoint.sum()),
        region_rejected_poses=int((~region_ok).sum()),
        endpoint_rejected_poses=int((~endpoint).sum()),
        rejected_by_both=int((~region_ok & ~endpoint).sum()),
        endpoint_rejection_counts=dict(Counter(r for ok, r in zip(endpoint, endpoint_reasons) if not ok)),
        region_rejection_counts=dict(Counter(regions['reasons'][p['pair_index']]
            for p, ok in zip(poses, region_ok) if not ok)),
        no_insertion_check=True, no_release_check=True, no_VLM=True,
        no_robot_execution=True, no_physics_steps=True,
        timing_seconds=time.perf_counter()-start)
    folder = Path(output)/'cases'/scene/part
    dump(folder/'decisions.json', decisions)
    dump(folder/'result.json', result)
    return {k: result[k] for k in ('scene', 'part', 'status', 'source_retained_pose_count',
        'retained_pose_count', 'retained_pair_count', 'region_eligible_poses', 'endpoint_pass_poses',
        'region_rejected_poses', 'endpoint_rejected_poses', 'rejected_by_both', 'timing_seconds')}


def protected_paths():
    return [SOURCE/'results.json', SOURCE/'summary.json', SOURCE/'gallery/index.html',
        ROOT/'outputs/cad_grasp_geometry_20261026/inputs.json',
        ROOT/'experiments/runtime_antipodal_assembly_20261023/goal_audit.json',
        ROOT/'experiments/grconvnet_vlm_pickup_20261024/vlm_client.py',
        ROOT/'experiments/grconvnet_vlm_pickup_20261024/grasp_rank_prompt.txt',
        ROOT/'docs/cad_view_prompt.txt',
        ROOT/'simulation/nist_task_board_1.py',
        ROOT/'outputs/simulation/nist_task_board_1/assets/provenance.json']


def audit(records):
    if len(records) != 96 or len({(r['scene'], r['part']) for r in records}) != 96:
        raise AssertionError('Expected 96 unique placements')
    for record in records:
        source = read(ROOT/record['source_case_path'])
        if native.sha(ROOT/record['source_case_path']) != record['source_case_sha256']:
            raise AssertionError('Source case changed during filtering')
        by_id = {p['id']: p for p in source['retained_poses']}
        indices = sorted({p['pair_index'] for p in record['retained_poses']})
        if indices != record['retained_pair_indices']:
            raise AssertionError('Contact-pair support mismatch')
        if len(record['retained_poses']) != record['retained_pose_count']:
            raise AssertionError('Retained pose count mismatch')
        if not np.array_equal(np.asarray(record['contact_pairs_world_m']).reshape(-1, 2, 3),
                np.asarray(source['all_sampled_pairs_world_m'])[indices].reshape(-1, 2, 3)):
            raise AssertionError('Contact pairs changed')
        goal = np.asarray(record['geometry_spec']['goal_T_cad_board'])
        inverse_pick = np.linalg.inv(record['world_T_cad'])
        for pose in record['retained_poses']:
            original = by_id[pose['id']]
            for key, value in original.items():
                if pose[key] != value:
                    raise AssertionError('Source grasp changed: ' + key)
            expected = goal @ inverse_pick @ np.asarray(original['world_T_fabrica_panda_hand'])
            if not np.allclose(expected, pose['board_T_fabrica_panda_hand_at_goal'], atol=1e-12):
                raise AssertionError('Final hand transform changed part-to-hand relation')
    return dict(placements=96, source_grasps_unchanged=True, no_new_grasps=True,
                pair_support_complete=True, fixed_part_to_hand_transport_verified=True)


def main():
    start = time.perf_counter()
    before = {str(p.relative_to(ROOT)): native.sha(p) for p in protected_paths() if p.exists()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/'runs'/stamp
    output.mkdir(parents=True, exist_ok=False)
    inputs = read(SOURCE/'results.json')
    keys = [(r['scene'], r['part']) for r in inputs]
    if len(keys) != 96 or len(set(keys)) != 96:
        raise ValueError('Source bank is not complete')
    del inputs
    receiver_assets = {item['mesh_path'] for _, part in keys
                       for item in geometry_spec(part)['receivers']}
    manifest = dict(created_utc=stamp, source_run=str(SOURCE.relative_to(ROOT)),
        method='Existing native Fabrica bank plus nominal mating-region and seated-hand CAD checks',
        units='metres', goal_frame='board-centered XY; source world Z preserved',
        workers=4, endpoint_opening='exact saved Fabrica holding open_ratio; no extra opening',
        added_clearance_margin_m=0.,
        goal_authority='Previously user-authorized nominal simulation targets, not verified assembly transforms',
        no_sampler_run=True, no_insertion_check=True, no_release_check=True,
        no_VLM=True, no_pickup_or_assembly_execution=True, no_physics_steps=True,
        no_pose_estimation_run=True, no_simulator_state_read=True,
        protected_sha256=before,
        receiver_sha256={str(p.relative_to(ROOT)): native.sha(p) for p in sorted(receiver_assets)},
        source_sha256={str(p.relative_to(ROOT)): native.sha(p)
                       for p in Path(__file__).parent.glob('*.py')})
    dump(output/'manifest.json', manifest)
    print('OUTPUT=' + str(output), flush=True)
    case_summaries = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run_case, scene, part, str(output)) for scene, part in keys]
        for job in as_completed(jobs):
            record = job.result()
            case_summaries.append(record)
            print(json.dumps(record), flush=True)
    records = [read(output/'cases'/scene/part/'result.json') for scene, part in sorted(keys)]
    dump(output/'results.json', records)
    checks = audit(records)
    after = {str(p.relative_to(ROOT)): native.sha(p) for p in protected_paths() if p.exists()}
    checks['protected_files_unchanged'] = before == after
    if before != after:
        raise AssertionError('Protected evidence or VLM source changed')
    dump(output/'audit.json', checks)
    by_part = {}
    for part in sorted({r['part'] for r in records}):
        rows = [r for r in records if r['part'] == part]
        by_part[part] = dict(placements=len(rows), available=sum(r['retained_pose_count'] > 0 for r in rows),
            source_poses=sum(r['source_retained_pose_count'] for r in rows),
            retained_poses=sum(r['retained_pose_count'] for r in rows),
            retained_pairs=sum(r['retained_pair_count'] for r in rows),
            region_rejected=sum(r['region_rejected_poses'] for r in rows),
            endpoint_rejected=sum(r['endpoint_rejected_poses'] for r in rows),
            region_statuses=sorted({r['mating_region_check_status'] for r in rows}))
    summary = dict(placements=96, parts=12, scenes=8,
        available=sum(r['retained_pose_count'] > 0 for r in records),
        input_poses=sum(r['source_retained_pose_count'] for r in records),
        retained_poses=sum(r['retained_pose_count'] for r in records),
        retained_pairs=sum(r['retained_pair_count'] for r in records), by_part=by_part,
        elapsed_seconds=time.perf_counter()-start,
        timing_scope='Four-worker CAD filtering, serialization and audits; gallery rendering excluded',
        region_unavailable_placements=sum(r['mating_region_check_status'].startswith('unavailable') for r in records),
        no_insertion_check=True, no_release_check=True, no_VLM=True, no_execution=True,
        audit=checks)
    dump(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return output


if __name__ == '__main__':
    main()
