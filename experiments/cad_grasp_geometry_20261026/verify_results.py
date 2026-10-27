"""Independent saved-record arithmetic, pad geometry and CAD-surface audit.

Usage: python -m experiments.cad_grasp_geometry_20261026.verify_results RUN_PATH
This does not rerun generation, validate pickup dynamics, or call a VLM.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import open3d as o3d

from .inputs import OUTPUT, REPO, local, read, sha256


SURFACE_DIAGNOSTIC_REFERENCE_M = 2e-6
WITNESS_TRIANGLE_NUMERICAL_TOLERANCE_M = 5e-9


class Checks:
    def __init__(self):
        self.count = 0
        self.failed_count = 0
        self.failures = []

    def check(self, condition, message):
        self.count += 1
        if not bool(condition):
            self.failed_count += 1
            if len(self.failures) < 200:
                self.failures.append(message)


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def moved_vertices(component, opening):
    return np.asarray(component['vertices_zero_m']) + opening * np.asarray(component['translation_per_opening'])


def triangle_distances(points, triangles):
    """Double-precision distance to the queried triangle, including its edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = b-a, c-a
    normal = np.cross(u, v)
    squared = np.einsum('ij,ij->i', normal, normal)
    dot = np.einsum('ij,ij->i', points-a, normal)
    safe = squared > 0
    factor = np.divide(dot, squared, out=np.zeros_like(dot), where=safe)
    projection = points-factor[:, None]*normal
    coordinate = projection-a
    bu = np.divide(np.einsum('ij,ij->i', np.cross(coordinate, v), normal), squared,
                   out=np.zeros_like(dot), where=safe)
    bv = np.divide(np.einsum('ij,ij->i', np.cross(u, coordinate), normal), squared,
                   out=np.zeros_like(dot), where=safe)
    inside = safe & (bu >= -1e-10) & (bv >= -1e-10) & (bu+bv <= 1+1e-10)
    distance = np.where(inside, np.linalg.norm(points-projection, axis=1), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        direction = end-start
        length = np.einsum('ij,ij->i', direction, direction)
        t = np.divide(np.einsum('ij,ij->i', points-start, direction), length,
                      out=np.zeros_like(length), where=length > 0)
        closest = start+np.clip(t, 0, 1)[:, None]*direction
        distance = np.minimum(distance, np.linalg.norm(points-closest, axis=1))
    return distance


def surface_distances(row, nominal, actual, actual_triangle_ids):
    """Query the native CAD independently in CAD coordinates, preserving holes."""
    mesh = o3d.io.read_triangle_mesh(str(local(row['mesh_path'])))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if row['part'] == 'Gear_Large':
        mesh.merge_close_vertices(5e-8)
    query = o3d.t.geometry.RaycastingScene()
    query.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    pose = np.asarray(row['world_T_cad'])
    result = {}
    for label, values in [('nominal', nominal), ('actual_first_contact', actual)]:
        points = np.unique(np.asarray(values, float).reshape(-1, 3), axis=0) if len(values) else np.empty((0, 3))
        if not len(points):
            result[label] = {'unique_points': 0, 'maximum_distance_m': None}
            continue
        cad_points = (points - pose[:3, 3]) @ pose[:3, :3]
        response = query.compute_closest_points(o3d.core.Tensor(cad_points.astype(np.float32)))
        closest = response['points'].numpy()
        raw_distances = np.linalg.norm(cad_points - closest, axis=1)
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[response['primitive_ids'].numpy()]]
        distances = triangle_distances(cad_points, triangles)
        result[label] = {'unique_points': len(points), 'maximum_distance_m': float(distances.max()),
                         'mean_distance_m': float(distances.mean()),
                         'points_above_diagnostic_reference': int((distances > SURFACE_DIAGNOSTIC_REFERENCE_M).sum()),
                         'raw_float32_query_maximum_distance_m': float(raw_distances.max())}
        if label == 'actual_first_contact':
            source_points = (np.asarray(actual)-pose[:3, 3])@pose[:3, :3]
            source_triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[actual_triangle_ids]]
            witness_distances = triangle_distances(source_points, source_triangles)
            result[label]['recorded_native_triangle_maximum_distance_m'] = float(witness_distances.max())
    return result


def audit_case(row, input_row, hand, config, checks):
    label = f"{row['scene']}/{row['part']}"
    before = checks.failed_count
    for field in ('mesh_path', 'mesh_sha256', 'world_T_cad', 'table_plane_world', 'pose_source', 'candidate_id'):
        checks.check(row[field] == input_row[field], f'{label}: input field changed: {field}')
    checks.check(row.get('simulator_object_poses_read') is False, f'{label}: simulator pose flag')
    checks.check(row.get('associations_assumed') is True, f'{label}: association assumption missing')
    checks.check(row['status'] in ('feasible_grasps_available', 'no_feasible_grasp_in_finite_search'),
                 f"{label}: result status {row['status']}")
    path = local(row.get('all_candidate_checks', ''))
    if not path.is_file():
        checks.check(False, f'{label}: complete candidate-check file unavailable')
        return {'scene': row['scene'], 'part': row['part'], 'audit_failures': checks.failed_count-before}
    candidates = read(path)
    by_id = {candidate['id']: candidate for candidate in candidates}
    valid = [candidate for candidate in candidates if candidate['valid']]
    rejected = [candidate for candidate in candidates if not candidate['valid']]
    groups = row['grasps']
    checks.check(len(candidates) == len(by_id), f'{label}: repeated candidate IDs')
    checks.check(row['generated_count'] == len(candidates), f'{label}: generated count')
    checks.check(row['valid_count'] == len(valid), f'{label}: valid count')
    checks.check(row['grouped_count'] == len(groups), f'{label}: grouped count')
    checks.check(row['generated_count'] == len(valid)+len(rejected), f'{label}: pass/reject partition')
    checks.check(Counter(candidate['rejection_reason'] for candidate in rejected) == row['rejection_counts'],
                 f'{label}: rejection reason counts')
    checks.check(all(candidate['rejection_reason'] is None for candidate in valid), f'{label}: valid candidate rejection flag')
    checks.check(all(isinstance(candidate['rejection_reason'], str) for candidate in rejected),
                 f'{label}: rejected candidate lacks reason')
    checks.check(len(candidates) == row['unique_horizontal_pairs']*2*len(config['finger_depths_m']),
                 f'{label}: every pair/jaw/depth combination was not checked')
    members = [member for group in groups for member in group.get('grouped_candidate_ids', [])]
    checks.check(len(members) == len(set(members)) == len(valid), f'{label}: grouping duplicated or lost valid candidates')
    checks.check(set(members) == {candidate['id'] for candidate in valid}, f'{label}: groups do not partition all valid IDs')
    checks.check(len({group['id'] for group in groups}) == len(groups), f'{label}: duplicate representative IDs')
    checks.check({group['pad_depth_m'] for group in groups} == {candidate['pad_depth_m'] for candidate in valid},
                 f'{label}: depth alternative lost')
    for group in groups:
        name = f"{label}/{group['id']}"
        ids = group.get('grouped_candidate_ids', [])
        checks.check(bool(ids) and group['id'] in ids, f'{name}: representative not in its group')
        checks.check(group['id'] in by_id and by_id[group['id']]['valid'], f'{name}: representative not checked valid')
        if group['id'] in by_id:
            expected = dict(by_id[group['id']], grouped_candidate_ids=ids)
            checks.check(group == expected, f'{name}: representative differs from checked record')
        for member in ids:
            checks.check(member in by_id and by_id[member]['pad_depth_m'] == group['pad_depth_m'],
                         f'{name}: grouped a different finger depth')

    components = hand['components']
    pads = sorted((value for value in components if value['is_pad']),
                  key=lambda value: np.mean(value['vertices_zero_m'], axis=0)[0])
    pad_by_name = {pad['name']: pad for pad in pads}
    gap_offset = float(np.max(np.asarray(pads[0]['vertices_zero_m'])[:, 0])
                       - np.min(np.asarray(pads[1]['vertices_zero_m'])[:, 0]))
    plane = np.asarray(row['table_plane_world'])
    nominal_points, actual_points, actual_triangle_ids = [], [], []
    largest_rotation_error = 0.
    maximum_opening_error = 0.
    maximum_width_difference = 0.
    maximum_table_replay_error = 0.
    maximum_inner_face_error = 0.
    minimum_table_clearance = None
    for candidate in candidates:
        name = f"{label}/{candidate['id']}"
        pose = np.asarray(candidate['world_T_grip'], float)
        rotation = pose[:3, :3]
        contacts = np.asarray(candidate['contacts_world_m'], float)
        nominal_points.extend(contacts)
        checks.check(pose.shape == (4, 4) and np.isfinite(pose).all(), f'{name}: pose is not finite 4x4')
        checks.check(np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12), f'{name}: homogeneous pose row')
        rotation_error = float(np.max(np.abs(rotation.T@rotation-np.eye(3))))
        largest_rotation_error = max(largest_rotation_error, rotation_error)
        checks.check(rotation_error <= 5e-7 and abs(np.linalg.det(rotation)-1) <= 5e-7,
                     f'{name}: orientation is not a proper rotation')
        checks.check(np.allclose(rotation[:, 2], [0, 0, -1], atol=1e-12), f'{name}: approach is not down')
        checks.check(contacts.shape == (2, 3) and np.isfinite(contacts).all(), f'{name}: nominal contacts')
        checks.check(abs(contacts[0, 2]-contacts[1, 2]) <= 1e-10, f'{name}: nominal contacts differ in height')
        width = float(np.linalg.norm(contacts[1]-contacts[0]))
        checks.check(abs(width-candidate['width_m']) <= 5e-8, f'{name}: contact width arithmetic')
        opening_error = abs(candidate['opening_m']-(width+gap_offset+config['extra_opening_m']))
        maximum_opening_error = max(maximum_opening_error, opening_error)
        checks.check(opening_error <= 5e-8, f'{name}: opening must include actual pad gap offset and extra opening')
        checks.check(0 <= candidate['opening_m'] <= hand['opening_limits_m'][1]+1e-12, f'{name}: opening outside Panda limits')
        if not candidate['valid']:
            continue
        checks.check(candidate['pad_depth_m'] in config['finger_depths_m'], f'{name}: unconfigured depth')
        for pad in pads:
            pad_vertices = np.asarray(pad['vertices_zero_m'])
            checks.check(pad_vertices[:, 2].min()+config['pad_edge_margin_m'] <= candidate['pad_depth_m']
                         <= pad_vertices[:, 2].max()-config['pad_edge_margin_m'], f'{name}: nominal point outside finite inner pad')
        table_values = []
        for component in components:
            for opening in (0., candidate['opening_m']):
                world = transform(moved_vertices(component, opening), pose)
                table_values.append(float((world@plane[:3]+plane[3]).min()))
        independently_computed = min(table_values)
        if minimum_table_clearance is None or independently_computed < minimum_table_clearance:
            minimum_table_clearance = independently_computed
        maximum_table_replay_error = max(maximum_table_replay_error, abs(independently_computed-candidate['table_clearance_m']))
        checks.check(independently_computed >= config['table_clearance_m']-1e-10, f'{name}: hand/table clearance')
        checks.check(abs(independently_computed-candidate['table_clearance_m']) <= 1e-9,
                     f'{name}: stored table clearance differs from independent full-vertex result')
        openings = candidate['first_contact_openings_m']
        mismatch = abs(openings[0]-openings[1])
        maximum_width_difference = max(maximum_width_difference, mismatch)
        checks.check(mismatch <= config['contact_opening_difference_m']+1e-12, f'{name}: contact opening disagreement')
        checks.check(abs(mismatch-candidate['contact_opening_difference_m']) <= 1e-12, f'{name}: opening difference arithmetic')
        checks.check(abs(max(openings)-candidate['contact_opening_m']) <= 1e-12, f'{name}: stop opening is not earlier first contact')
        witnesses = candidate['actual_contacts']
        checks.check(len(witnesses) == 2 and {value['pad'] for value in witnesses} == set(pad_by_name),
                     f'{name}: missing or duplicate pad witness')
        for witness in witnesses:
            pad = pad_by_name[witness['pad']]
            vertices = moved_vertices(pad, witness['opening_m'])
            point = (np.asarray(witness['point_world_m'])-pose[:3, 3])@rotation
            actual_points.append(witness['point_world_m'])
            actual_triangle_ids.append(witness['first_contact']['triangle_index'])
            checks.check(np.isfinite(point).all(), f'{name}: nonfinite actual contact witness')
            inner_x = vertices[:, 0].max() if pad['translation_per_opening'][0] < 0 else vertices[:, 0].min()
            inner_error = abs(point[0]-inner_x)
            maximum_inner_face_error = max(maximum_inner_face_error, inner_error)
            checks.check(inner_error <= config['contact_inner_face_tolerance_m']+2e-8, f'{name}: witness not on inner pad face')
            tolerance = config['contact_inner_face_tolerance_m']+2e-8
            checks.check(np.all(point >= vertices.min(0)-tolerance) and np.all(point <= vertices.max(0)+tolerance),
                         f'{name}: witness outside finite pad')
            checks.check(witness['normal_alignment'] >= np.cos(np.deg2rad(config['normal_angle_deg']))-1e-12,
                         f'{name}: first-contact normal misaligned')
            first = witness['first_contact']
            checks.check(first['lower_fraction'] <= first['fraction'] <= first['upper_fraction'], f'{name}: invalid contact bracket')
            checks.check(first['upper_fraction']-first['lower_fraction'] <= config['contact_search_fraction_tolerance']+1e-12,
                         f'{name}: first-contact bracket too wide')
            checks.check(abs(witness['opening_m']-candidate['opening_m']*(1-first['fraction'])) <= 1e-12,
                         f'{name}: first-contact opening/fraction arithmetic')
    distances = surface_distances(row, nominal_points, actual_points, actual_triangle_ids)
    for kind, result in distances.items():
        if result['unique_points']:
            checks.check(np.isfinite(result['maximum_distance_m']), f'{label}: nonfinite {kind} CAD distance')
    actual_result = distances['actual_first_contact']
    if actual_result['unique_points']:
        checks.check(actual_result['recorded_native_triangle_maximum_distance_m'] <= WITNESS_TRIANGLE_NUMERICAL_TOLERANCE_M,
                     f'{label}: actual witness is not on its saved native CAD triangle')
    return {'scene': row['scene'], 'part': row['part'], 'generated': len(candidates), 'valid': len(valid),
            'groups': len(groups), 'all_valid_candidates_partitioned': len(members) == len(valid),
            'maximum_rotation_orthogonality_error': largest_rotation_error,
            'maximum_opening_arithmetic_error_m': maximum_opening_error,
            'maximum_first_contact_opening_difference_m': maximum_width_difference,
            'minimum_recomputed_table_clearance_m': minimum_table_clearance,
            'maximum_table_clearance_replay_error_m': maximum_table_replay_error,
            'maximum_actual_contact_inner_face_error_m': maximum_inner_face_error,
            'surface_distance_audit': distances, 'audit_failures': checks.failed_count-before}


def verify(run_path):
    started = perf_counter()
    run = local(run_path)
    manifest = read(run/'manifest.json')
    results = read(run/'results.json')
    base_run = local(manifest['base_run']) if manifest.get('base_run') else None
    base_manifest = read(base_run/'manifest.json') if base_run else None
    base_results = read(base_run/'results.json') if base_run else None
    effective_config = manifest.get('config') or base_manifest['config']
    input_path = manifest.get('input_manifest') or base_manifest['input_manifest']
    input_manifest = read(local(input_path))
    input_rows = {(row['scene'], row['part']): row for row in input_manifest['records']}
    hand = read(OUTPUT/'hand_model.json')
    checks = Checks()
    checks.check(len(input_rows) == 96, 'input manifest does not contain 96 unique scene/part pairs')
    checks.check(set(input_manifest['scenes']) == {f'N{i:02d}' for i in range(8)}, 'input scene inventory')
    checks.check(len(input_manifest['parts']) == 12, 'input part inventory')
    checks.check(set(input_rows) == {(scene, part) for scene in input_manifest['scenes'] for part in input_manifest['parts']},
                 'input inventory is not the full eight-by-twelve cross product')
    keys = [(row['scene'], row['part']) for row in results]
    checks.check(len(keys) == len(set(keys)) == manifest['cases'], 'run result count/uniqueness')
    checks.check(set(keys) <= set(input_rows), 'run contains an unknown input case')
    full_run = manifest['cases'] == 96
    if full_run:
        checks.check(set(keys) == set(input_rows), 'full run omits input scene/part pairs')
    for flag in ('no_vlm', 'no_robot_execution', 'no_physics_steps', 'no_other_part_or_robot_self_checks', 'no_candidate_cap'):
        checks.check(manifest.get(flag) is True, f'manifest flag missing/false: {flag}')
    if base_run:
        checks.check(sha256(base_run/'manifest.json') == manifest['base_manifest_sha256'], 'base manifest hash changed')
        checks.check(sha256(base_run/'results.json') == manifest['base_results_sha256'], 'base result hash changed')
        base_by_id = {(row['scene'], row['part']): row for row in base_results}
        targets = {(row['scene'], row['part']) for row in base_results if row['valid_count'] == 0}
        declared = {(row['scene'], row['part']) for row in manifest['followup_cases']}
        checks.check(declared == targets, 'coverage follow-up did not declare every zero-valid baseline case')
        fallback_rows = read(run/'fallback_results.json')
        checks.check({(row['scene'], row['part']) for row in fallback_rows} == targets,
                     'coverage follow-up did not complete every declared zero-valid case')
        completion = read(run/'completion.json')
        checks.check(completion['complete'] is True and completion['combined_cases'] == 96
                     and completion['completed_followup_cases'] == len(targets), 'coverage follow-up completion record')
        for row in results:
            original = base_by_id[row['scene'], row['part']]
            label = f"{row['scene']}/{row['part']}"
            phase = row.get('search_stage')
            if phase in ('height_followup', 'surface_coverage'):
                checks.check(original['valid_count'] == 0, f'{label}: coverage follow-up was used despite base valid grasps')
                for name in ('generated_count', 'valid_count', 'grouped_count', 'timing_seconds', 'sampling_seconds', 'validation_seconds'):
                    checks.check(row.get('base_'+name) == original[name], f'{label}: saved base {name} differs')
                checks.check(row.get('base_rejection_counts') == original['rejection_counts'],
                             f'{label}: base rejection counts changed')
                checks.check(abs(row['timing_seconds']-row['base_timing_seconds']-row[phase+'_seconds']) <= 1e-8,
                             f'{label}: combined base/follow-up time arithmetic')
                checks.check(row['total_proposals_checked_including_base'] == original['generated_count']+row['generated_count'],
                             f'{label}: combined proposal-count arithmetic')
                if phase == 'height_followup':
                    checks.check(row['height_grid_count'] == len(row['height_grid_world_m']), f'{label}: height-grid count')
                    checks.check(row['horizontal_lines_cast'] == row['source_xy_line_count']*row['height_grid_count'],
                                 f'{label}: not every XY line/height combination was cast')
            else:
                checks.check(row == original, f'{label}: base success record was changed in combined run')
    preserved = {}
    for path, digest in manifest['protected_sha256'].items():
        actual = sha256(local(path))
        preserved[path] = actual == digest
        checks.check(actual == digest, f'protected file changed: {path}')
    critical_generation_sources = {}
    for filename in ('run_trial.py', 'geometry_kernel.py', 'hand_model.py'):
        source_path = Path(__file__).with_name(filename)
        relative_source = source_path.relative_to(REPO).as_posix()
        expected = manifest['source_sha256'].get(relative_source)
        actual = sha256(source_path)
        critical_generation_sources[relative_source] = actual == expected
        checks.check(actual == expected, f'critical generation source differs from frozen run: {relative_source}')
    preservation = read(run/'preservation.json')
    checks.check(preservation['unchanged'] is True, 'run preservation record reports changed protected source')
    checks.check(preservation['sha256_after'] == manifest['protected_sha256'], 'run preservation digest arithmetic')
    frozen_inputs_unchanged = True
    for path, digest in input_manifest['source_sha256'].items():
        same = sha256(local(path)) == digest
        checks.check(same, f'observed input source changed: {path}')
        frozen_inputs_unchanged &= same
    cases = []
    for number, row in enumerate(results, 1):
        cases.append(audit_case(row, input_rows[row['scene'], row['part']], hand, effective_config, checks))
        print(json.dumps({'audited_cases': number, 'total': len(results), 'scene': row['scene'], 'part': row['part'],
                          'failures': checks.failed_count}), flush=True)
    totals = {field: sum(case.get(field, 0) for case in cases) for field in ('generated', 'valid', 'groups')}
    totals['proposals_checked_including_base'] = sum(row.get('total_proposals_checked_including_base', row['generated_count'])
                                                   for row in results)
    surface_summary = {}
    for kind in ('nominal', 'actual_first_contact'):
        measurements = [case['surface_distance_audit'][kind] for case in cases
                        if case.get('surface_distance_audit', {}).get(kind, {}).get('unique_points')]
        surface_summary[kind] = {
            'unique_points_across_cases': sum(value['unique_points'] for value in measurements),
            'maximum_returned_triangle_distance_m': max((value['maximum_distance_m'] for value in measurements), default=None),
            'maximum_raw_float32_query_distance_m': max((value['raw_float32_query_maximum_distance_m'] for value in measurements), default=None),
            'points_above_diagnostic_reference': sum(value['points_above_diagnostic_reference'] for value in measurements),
        }
        if kind == 'actual_first_contact':
            surface_summary[kind]['maximum_recorded_native_triangle_distance_m'] = max(
                (value['recorded_native_triangle_maximum_distance_m'] for value in measurements), default=None)
    report = {'status': ('failed' if checks.failed_count else 'passed_full_96' if full_run else 'passed_development_subset'),
              'run_path': str(run.relative_to(REPO)), 'full_96_case_inventory_verified': full_run and set(keys) == set(input_rows),
              'audited_cases': len(cases), 'checks': checks.count, 'failed_checks': checks.failed_count,
              'failures': checks.failures, 'failure_messages_truncated': checks.failed_count > len(checks.failures),
              'totals': totals, 'cases': cases, 'contact_surface_distance_summary': surface_summary,
              'placements_with_retained_grasps': sum(row['grouped_count'] > 0 for row in results),
              'group_count_distribution': dict(sorted(Counter(row['grouped_count'] for row in results).items())),
              'protected_sources_unchanged': preserved,
              'critical_generation_sources_match_frozen_run': critical_generation_sources,
              'observed_input_sources_unchanged': bool(frozen_inputs_unchanged),
              'coverage_followup_combination_checked': bool(base_run),
              'hand_model_sha256': sha256(OUTPUT/'hand_model.json'),
              'surface_diagnostic_reference_m': SURFACE_DIAGNOSTIC_REFERENCE_M,
              'surface_diagnostic_reference_is_acceptance_gate': False,
              'witness_triangle_numerical_tolerance_m': WITNESS_TRIANGLE_NUMERICAL_TOLERANCE_M,
              'surface_audit_method': 'Independent Open3D closest-triangle queries in native CAD coordinates, with double-precision distance refinement on returned triangles; raw float32 coordinate distances are also reported. Returned-triangle distances are upper bounds on true nearest-surface distance. Actual first-contact witnesses are additionally checked against their recorded native triangle IDs. Gear uses declared 50nm numerical seam weld.',
              'table_audit_method': 'Every original exported hand vertex, full upward observed table plane, opening=0 and approach opening endpoints; independently reconstruct translation from hand JSON.',
              'grouping_audit_method': 'Group members must partition every checked-valid candidate once; all depths retained; representatives must equal actual checked records. No invented candidates or padding.',
              'limitations': ['Does not rerun the full triangle collision/first-contact kernel or validate friction, forces, pickup or assembly.',
                              'Actual first-contact witnesses can differ in height; nominal antipodal points must be horizontal.',
                              'Inner-face witness check verifies finite pad geometry, not support area or a physical contact patch.',
                              'No-VLM/execution statements are checked manifest declarations, supported by frozen experiment source; no hardware trace is inferred.'],
              'audit_source_sha256': sha256(Path(__file__)), 'seconds': perf_counter()-started}
    (run/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'audited_cases', 'checks', 'failed_checks', 'totals', 'seconds')}), flush=True)
    return report


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_geometry_20261026.verify_results RUN_PATH')
    outcome = verify(sys.argv[1])
    raise SystemExit(1 if outcome['failed_checks'] else 0)
