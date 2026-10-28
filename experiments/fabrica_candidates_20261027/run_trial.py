"""Fabrica-derived CAD candidates only; no VLM, physics or robot commands."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.cad_grasp_geometry_20261026 import run_trial as core
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_geometry_20261026.inputs import MANIFEST, local, read
from experiments.cad_grasp_regions_20261026 import run as placement
from experiments.cad_surface_grasps_20261026.centering import center_from_first_contacts
from . import sampler


ROOT = core.ROOT
BASE = ROOT / 'outputs/fabrica_candidates_20261027'
CONFIG = dict(
    maximum_closing_projection_deg=15., actual_contact_normal_limit_deg=1.,
    pad_depth_fractions=[0.5, 0.25, 0.75], extra_opening_m=0.012,
    both_finger_assignments=True, workers=4,
    display_group_angle_deg=10., display_group_width_m=0.002,
    display_group_xy_pad_width_fraction=0.5,
    display_group_z_pad_height_fraction=0.125,
)


def initialize():
    core.initialize()
    core.CONFIG = dict(core.CONFIG, normal_angle_deg=CONFIG['actual_contact_normal_limit_deg'])


def canonical_pair(pair):
    return pair[::-1].copy() if tuple(pair[1]) < tuple(pair[0]) else pair.copy()


def group_for_display(grasps, pads):
    """Group checked nearby poses for browsing; preserve all candidates for later selection."""
    xy = min(float(np.ptp(np.asarray(p['vertices_zero_m'])[:, 1])) for p in pads)
    z = min(float(np.ptp(np.asarray(p['vertices_zero_m'])[:, 2])) for p in pads)
    xy *= CONFIG['display_group_xy_pad_width_fraction']
    z *= CONFIG['display_group_z_pad_height_fraction']
    cosine = np.cos(np.deg2rad(CONFIG['display_group_angle_deg']))
    groups = []
    for grasp in grasps:
        pose = np.asarray(grasp['world_T_grip'])
        pair = canonical_pair(np.asarray(grasp['contacts_world_m']))
        match = None
        for group in groups:
            old = np.asarray(group['world_T_grip'])
            other_pair = canonical_pair(np.asarray(group['contacts_world_m']))
            if abs(float(pose[:3, 0] @ old[:3, 0])) < cosine:
                continue
            if np.linalg.norm(pose[:2, 3] - old[:2, 3]) > xy:
                continue
            if abs(float(pose[2, 3] - old[2, 3])) > z:
                continue
            if abs(grasp['contact_opening_m'] - group['contact_opening_m']) > CONFIG['display_group_width_m']:
                continue
            if np.max(np.linalg.norm(pair - other_pair, axis=1)) > xy:
                continue
            match = group
            break
        if match is None:
            groups.append(dict(grasp, group_members=[grasp['id']],
                representative_rule='first_valid_in_Fabrica_centroid_order_then_middle_pad_depth; display_only'))
        else:
            match['group_members'].append(grasp['id'])
    for index, group in enumerate(groups, 1):
        group['family_id'] = f'F{index:03d}'
    return groups, dict(xy_distance_m=xy, z_distance_m=z,
        angle_deg=CONFIG['display_group_angle_deg'], width_difference_m=CONFIG['display_group_width_m'],
        no_candidate_deleted=True, no_quality_ranking=True,
        note='Nearby-pose display groups, not proven semantic surface or cylinder-symmetry families.')


def run_case(row, cache, output):
    started = time.perf_counter()
    folder = Path(output) / 'cases' / row['scene'] / row['part']
    result = dict(row, status='started', grasps=[], all_valid_grasps=[],
                  generated_count=0, valid_count=0, grouped_count=0)
    try:
        if row['input_status'] != 'available':
            raise ValueError('The frozen observed CAD fit is unavailable; no fit will be recomputed')
        components, pads, gap_offset = core.hand_geometry()
        vertices, faces, mesh_info = core.load_target(row)
        collider = MeshCollider(vertices, faces)
        if collider.topology['component_count'] != collider.topology['closed_component_count']:
            raise ValueError('Native CAD has an open component')
        data = np.load(local(cache['npz_path']))
        pose = np.asarray(row['world_T_cad'])
        pairs = data['pairs_m'] @ pose[:3, :3].T + pose[:3, 3]
        pairs = np.asarray([canonical_pair(p) for p in pairs]).reshape(-1, 2, 3)
        plane = np.asarray(row['table_plane_world'])
        pad_low = max(float(np.asarray(p['vertices_zero_m'])[:, 2].min()) for p in pads)
        pad_high = min(float(np.asarray(p['vertices_zero_m'])[:, 2].max()) for p in pads)
        margin = core.CONFIG['pad_edge_margin_m']
        usable_low, usable_high = pad_low + margin, pad_high - margin
        depths = [usable_low + fraction * (usable_high - usable_low)
                  for fraction in CONFIG['pad_depth_fractions']]
        checks, rejected_pairs = [], Counter()
        for pair_index, pair in enumerate(pairs, 1):
            difference = pair[1] - pair[0]
            width = float(np.linalg.norm(difference))
            opening = width + gap_offset + CONFIG['extra_opening_m']
            if not (width > 1e-6 and opening <= core.HAND['opening_limits_m'][1]):
                rejected_pairs['opening_outside_common_Panda_margin'] += 1
                continue
            tilt = float(np.degrees(np.arcsin(np.clip(abs(difference[2]) / width, 0, 1))))
            if tilt > CONFIG['maximum_closing_projection_deg']:
                rejected_pairs['contact_line_not_near_horizontal'] += 1
                continue
            axis = difference.copy()
            axis[2] = 0.
            axis /= np.linalg.norm(axis)
            midpoint = pair.mean(axis=0)
            table_height = placement.minimum_table_height(midpoint, axis, opening, components, plane)
            seen_heights = set()
            for depth_index, desired_depth in enumerate(depths, 1):
                height = max(float(midpoint[2] + desired_depth), table_height)
                height_key = round(height, 9)
                if height_key in seen_heights:
                    continue
                seen_heights.add(height_key)
                positive = np.eye(4)
                positive[:3, :3] = placement.hand_rotation(axis)
                positive[:3, 3] = midpoint
                positive[2, 3] = height
                positive, centering = center_from_first_contacts(
                    positive, opening, collider, pads, core.CONFIG['contact_search_fraction_tolerance'])
                for sign in (1., -1.):
                    hand = positive.copy()
                    hand[:3, :3] = placement.hand_rotation(sign * axis)
                    candidate = dict(
                        id=f'G{len(checks)+1:05d}', family_id=f'P{pair_index:04d}D{depth_index}',
                        pair_index=pair_index, depth_index=depth_index,
                        desired_pad_depth_m=float(desired_depth), pad_depth_m=float(height-midpoint[2]),
                        jaw_sign=sign, world_T_grip=hand.tolist(), contacts_world_m=pair.tolist(),
                        width_m=width, opening_m=float(opening), native_pair_projection_deg=tilt,
                        center_projection=centering, table_height_adjustment_m=float(height-midpoint[2]-desired_depth))
                    reason, evidence = core.validate(candidate, collider, components, pads, plane)
                    candidate.update(valid=reason is None, rejection_reason=reason, **evidence)
                    checks.append(candidate)
        valid = [g for g in checks if g['valid']]
        groups, grouping = group_for_display(valid, pads)
        result.update(
            mesh_info=mesh_info, target_solid_topology=collider.topology,
            intrinsic_record=cache, native_pairs=len(pairs), pair_rejection_counts=dict(rejected_pairs),
            generated_count=len(checks), valid_count=len(valid), grouped_count=len(groups),
            grasps=groups, all_valid_grasps=valid, grouping=grouping,
            pad_depths_requested_m=depths,
            rejection_counts=dict(Counter(g['rejection_reason'] for g in checks if not g['valid'])),
            status='candidates_available' if valid else 'no_candidate_passed_in_bounded_search',
            all_candidate_checks=str((folder/'candidate_checks.json').relative_to(ROOT)),
            no_vlm=True, no_assembly_score=True, no_robot_execution=True, no_physics_steps=True)
        core.dump(folder/'candidate_checks.json', checks)
    except Exception as error:
        result.update(status='generation_error', error=str(error), traceback=traceback.format_exc())
    result['timing_seconds'] = time.perf_counter() - started
    core.dump(folder/'result.json', result)
    return result


def audit(records):
    seen = set()
    candidates = 0
    groups = 0
    largest_normal_error = 0.
    for record in records:
        key = (record['scene'], record['part'])
        if key in seen:
            raise AssertionError('Repeated scene/part record')
        seen.add(key)
        if record['status'] == 'generation_error':
            raise AssertionError(record['error'])
        good = record['all_valid_grasps']
        ids = {g['id'] for g in good}
        assigned = [x for group in record['grasps'] for x in group['group_members']]
        if len(assigned) != len(set(assigned)) or set(assigned) != ids:
            raise AssertionError('Display grouping omitted or duplicated candidates')
        for group in record['grasps']:
            if group['id'] not in ids:
                raise AssertionError('Representative is not an actual checked candidate')
        for candidate in good:
            transform = np.asarray(candidate['world_T_grip'])
            rotation = transform[:3, :3]
            if not np.isfinite(transform).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-9):
                raise AssertionError('Invalid hand transform')
            if not np.allclose(rotation[:, 2], [0., 0., -1.], atol=1e-9):
                raise AssertionError('Hand approach is not straight down')
            if len(candidate['actual_contacts']) != 2:
                raise AssertionError('Missing a pad contact')
            for contact in candidate['actual_contacts']:
                angle = float(np.degrees(np.arccos(np.clip(contact['normal_alignment'], -1, 1))))
                largest_normal_error = max(largest_normal_error, angle)
                if angle > CONFIG['actual_contact_normal_limit_deg'] + 1e-7:
                    raise AssertionError('Actual contact exceeds declared normal limit')
            if candidate['contact_opening_difference_m'] > core.CONFIG['contact_opening_difference_m'] + 1e-12:
                raise AssertionError('First contact mismatch exceeds existing limit')
            candidates += 1
        groups += len(record['grasps'])
    return dict(records=len(records), validated_candidates=candidates, display_groups=groups,
        maximum_actual_normal_error_deg=largest_normal_error, all_checks_passed=True,
        all_valid_candidates_preserved=True, no_vlm=True, no_execution=True,
        limitation='Structural/contact-evidence audit; not a second contact-kernel implementation or physical validation.')


def main():
    if not MANIFEST.exists() or not (core.BASE/'hand_model.json').exists():
        raise FileNotFoundError('Frozen CAD inputs and hand geometry must already exist')
    inputs = read(MANIFEST)['records']
    if len(inputs) != 96:
        raise ValueError('Expected frozen twelve-part, eight-scene dataset')
    initialize()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/'runs'/stamp
    output.mkdir(parents=True, exist_ok=False)
    protected = [MANIFEST, core.BASE/'hand_model.json', Path(core.__file__),
        ROOT/'experiments/cad_grasp_geometry_20261026/geometry_kernel.py',
        ROOT/'experiments/grconvnet_vlm_pickup_20261024/vlm_client.py',
        ROOT/'experiments/grconvnet_vlm_pickup_20261024/grasp_rank_prompt.txt',
        ROOT/'docs/cad_view_prompt.txt',
        ROOT/'outputs/cad_grasp_method_comparison_20261026/aligned_contacts/20261027T152238Z/results.json']
    before = {str(p.relative_to(ROOT)): core.sha(p) for p in protected}
    manifest = dict(created_utc=stamp, method='Fabrica-derived CAD contact generation + common downward Panda checker',
        scope='candidate generation only; user explicitly leaves VLM selection untouched',
        placements=96, parts=12, scenes=8, config=CONFIG, sampler_config=sampler.CONFIG,
        validation_config=core.CONFIG, inputs=str(MANIFEST.relative_to(ROOT)),
        protected_sha256=before, source_sha256={str(p.relative_to(ROOT)):core.sha(p) for p in (
            Path(__file__), Path(sampler.__file__), sampler.SOURCE)},
        no_vlm=True, no_vlm_prompt_change=True, no_assembly_score=True,
        no_robot_execution=True, no_physics_steps=True, no_new_pose_estimation=True,
        simulator_object_poses_read=False, learned_models_loaded=False,
        original_full_Fabrica_framework_executed=False,
        adaptations=['Bounded metre-scale contact kernel port; trimesh and FCL are absent.',
            'Original clearance spheres replaced by existing finite-hand/target/table checks.',
            'Existing Panda grip-site frame and straight-down conversion replace Fabrica base-frame/roll expansion.',
            'Three pad-relative depth positions are geometric variants added here; not a native Fabrica depth search.',
            'Post-validation pad-sized nearby-pose groups are display-only; every valid candidate is preserved.',
            'No mating task score, VLM ranking, selected final grasp or assembly view is generated.'])
    core.dump(output/'manifest.json', manifest)
    print('OUTPUT='+str(output), flush=True)
    caches = {}
    for index, part in enumerate(sorted({r['part'] for r in inputs})):
        row = next(r for r in inputs if r['part'] == part)
        native = dict(row, world_T_cad=np.eye(4).tolist())
        vertices, faces, mesh_info = core.load_target(native)
        sampled = sampler.generate_pairs(vertices, faces, seed=20261027+index)
        path = output/'CAD_contacts'/f'{part}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k:v for k,v in sampled.items() if k != 'metadata'})
        metadata = dict(sampled['metadata'], part=part, mesh_info=mesh_info,
                        npz_path=str(path.relative_to(ROOT)))
        core.dump(path.with_suffix('.json'), metadata)
        caches[part] = metadata
        print(json.dumps(dict(part=part, sparse_pairs=len(sampled['pairs_m']),
            raw_pairs=len(sampled['raw_pairs_m']), seconds=metadata['sampling_seconds'])), flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=CONFIG['workers'], initializer=initialize) as pool:
        jobs = [pool.submit(run_case, row, caches[row['part']], str(output)) for row in inputs]
        for job in as_completed(jobs):
            record = job.result()
            records.append(record)
            records.sort(key=lambda r:(r['scene'],r['part']))
            core.dump(output/'results.json', records)
            print(json.dumps({k:record[k] for k in (
                'scene','part','status','generated_count','valid_count','grouped_count','timing_seconds')}), flush=True)
    verification = audit(records)
    after = {str(p.relative_to(ROOT)):core.sha(p) for p in protected}
    verification['protected_unchanged'] = before == after
    if before != after:
        raise AssertionError('A protected input, prior result or VLM file changed')
    core.dump(output/'audit.json', verification)
    by_part = {}
    for part in sorted(caches):
        rows = [r for r in records if r['part'] == part]
        by_part[part] = dict(available=sum(bool(r['grasps']) for r in rows), placements=len(rows),
            checked=sum(r['generated_count'] for r in rows), valid=sum(r['valid_count'] for r in rows),
            display_groups=sum(r['grouped_count'] for r in rows),
            generation_seconds=caches[part]['sampling_seconds'],
            validation_seconds=sum(r['timing_seconds'] for r in rows))
    summary = dict(placements=len(records), parts=12, scenes=8,
        available=sum(bool(r['grasps']) for r in records),
        generated=sum(r['generated_count'] for r in records), valid=verification['validated_candidates'],
        display_groups=verification['display_groups'], by_part=by_part, audit=verification,
        no_vlm=True, no_physics_steps=True, no_pickup_or_assembly=True,
        timing_note='CAD contacts sampled once per part; validation task times are sums inside four-worker run, not isolated latency.')
    core.dump(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
