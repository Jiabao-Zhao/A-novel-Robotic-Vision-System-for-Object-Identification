"""Post-launch streaming variant of the frozen comparison report.

Copied from report.py after the initial full trial launched. Category totals
accumulate per case instead of retaining candidate dictionaries across cases.
A later post-processing repair compares each saved component table minimum to
that component, and the saved whole-hand minimum only to the whole hand. The
pre-repair bytes are preserved against the supplemental launch hash. This file
is not represented as unchanged original or supplemental launch source.

Only saved candidates are inspected. No candidate generation, collision-engine
query, robot motion, physics, learned model or new pose estimation is run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments.cad_grasp_selection_20261028.criteria import select_priority
from experiments.cad_grasp_selection_20261028.geometry import rigid_transform
from experiments.cad_grasp_selection_20261029.hand_frame import (
    FINGERTIP_FRAME, fingertip_calibration,
)
from experiments.cad_grasp_selection_20261029.task_inputs import build_request, HAND, REGISTRY, ROOT


VARIANTS = ('previous', 'refined')
HISTORICAL = ROOT/'outputs/cad_grasp_selection_20261028/runs/20261028T190844Z'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-9):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, atol=tolerance, rtol=0), message)


def transform(points, pose):
    return np.asarray(points, float)@pose[:3, :3].T+pose[:3, 3]


def reference_from_request(request):
    """Recompute fixed mesh quantities directly, without the scoring helper."""
    vertices = np.asarray(request['vertices_cad_m'], float)
    triangles = vertices[np.asarray(request['faces'], int)]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)/2
    require(np.isfinite(area).all() and area.sum() > 0, 'Invalid CAD area for audit')
    centroid = np.sum(area[:, None]*triangles.mean(axis=1), axis=0)/area.sum()
    point, direction = np.asarray(request['axis_point_cad_m']), np.asarray(request['axis_direction_cad'])
    return dict(surface_centroid_cad_m=centroid, axis_point_cad_m=point,
        axis_direction_cad=direction,
        D_tau_m=float(np.linalg.norm(np.cross(vertices-point, direction), axis=1).max()),
        D_surface_m=float(np.linalg.norm(vertices-centroid, axis=1).max()))


def candidate_table_clearance(candidate, pickup, plane, hand, *, return_component_minima=False):
    """Exact minimum for affine component motion against a fixed plane.

    Every saved component vertex is checked at both opening endpoints. A plane
    distance is linear along this motion, so the endpoint minimum also bounds
    every intermediate opening. This does not check hand/part/receiver overlap.
    """
    grip = rigid_transform(candidate['cad_T_grip_site'], 'saved grip-site pose')
    world_pose = pickup@grip
    minimum, witness, component_minima = np.inf, None, {}
    for component in hand['components']:
        component_min = np.inf
        for label, opening in (('holding', candidate['holding_opening_m']),
                               ('pre_opening', candidate['pre_opening_m'])):
            points = (np.asarray(component['vertices_zero_m'], float)
                      + float(opening)*np.asarray(component['translation_per_opening'], float))
            value = float(np.min(transform(points, world_pose)@plane[:3]+plane[3]))
            component_min = min(component_min, value)
            if value < minimum:
                minimum = value
                witness = dict(component=component['name'], opening_state=label)
        component_minima[component['name']] = float(component_min)
    require(np.isfinite(minimum), 'Complete hand table audit has no finite vertices')
    if return_component_minima:
        return float(minimum), witness, component_minima
    return float(minimum), witness


def raw_criteria(candidate, reference, pickup, plane):
    midpoint = np.asarray(candidate['contacts_cad_m'], float).mean(axis=0)
    q_tau = float(np.linalg.norm(np.cross(midpoint-reference['axis_point_cad_m'],
                                         reference['axis_direction_cad'])))
    q_surface = float(np.linalg.norm(midpoint-reference['surface_centroid_cad_m']))
    approach = pickup[:3, :3]@np.asarray(candidate['cad_T_hand'], float)[:3, 2]
    theta = float(np.arccos(np.clip(approach@(-plane[:3]), -1, 1)))
    return dict(q_tau_m=q_tau, q_surface_m=q_surface,
        r_tau=q_tau/reference['D_tau_m'] if reference['D_tau_m'] else 0.,
        r_surface=q_surface/reference['D_surface_m'] if reference['D_surface_m'] else 0.,
        theta_down_rad=theta)


def diagnostic_counts(candidates):
    counts = defaultdict(Counter)
    for candidate in candidates:
        for category, detail in candidate['validity'].get('diagnostics', {}).items():
            status = detail.get('status', 'unavailable') if isinstance(detail, dict) else str(detail)
            counts[category][str(status)] += 1
    return {category:dict(value) for category, value in sorted(counts.items())}


def _saved_table_values(value):
    """Read whole-hand values only; component reports have their own scope."""
    if not isinstance(value, dict):
        return []
    checks = value.get('checks', value)
    if not isinstance(checks, dict):
        return []
    return [float(checks[key]) for key in
            ('table_minimum_clearance_m', 'minimum_table_clearance_m') if key in checks]


def compare_saved_table_clearances(validity, whole_hand_minimum, component_minima):
    """Compare saved minima with independent minima in the same geometry scope."""
    whole_count = 0
    for saved in _saved_table_values(validity):
        close(saved, whole_hand_minimum,
              'Saved table minimum differs from complete-hand endpoint audit')
        whole_count += 1
    component_count = 0
    for detail in validity.get('checks', {}).get('component_geometry', []):
        if 'table_minimum_clearance_m' not in detail:
            continue
        name = detail.get('component')
        require(name in component_minima, f'Saved table report names an unknown hand component: {name}')
        close(detail['table_minimum_clearance_m'], component_minima[name],
              f'Saved table minimum differs from endpoint audit for component {name}')
        component_count += 1
    return whole_count, component_count


def audit_variant(folder, hand, registry, request_cache):
    rows, summary = read(folder/'results.json'), read(folder/'summary.json')
    require(len(rows) == 96, f'{folder.name}: expected 96 placements')
    keys = {(row['scene'], row['part']) for row in rows}
    scenes = sorted({row['scene'] for row in rows})
    parts = sorted({row['part'] for row in rows})
    require(len(keys) == 96 and scenes == [f'N{i:02d}' for i in range(8)] and len(parts) == 12,
            f'{folder.name}: repeated or missing scene/part')
    totals = Counter()
    per_part = {}
    category_totals = defaultdict(Counter)
    case_audits = []
    tip_cache = {}
    max_anchor_error, max_criterion_error = 0., 0.
    files = {str(folder/'results.json'):sha(folder/'results.json'),
             str(folder/'summary.json'):sha(folder/'summary.json')}
    for row in rows:
        part = row['part']
        selection = row['selection']
        path = folder/'cases'/row['scene']/part/'candidate_diagnostics.json'
        candidates = read(path)
        files[str(path)] = sha(path)
        require(len(candidates) == selection['proposed_count'] and len(candidates) <= 100,
                f'{folder.name}/{row["scene"]}/{part}: candidate count or 100-pose cap mismatch')
        require(len({c['candidate_id'] for c in candidates}) == len(candidates), 'Repeated candidate ID')
        valid = [c for c in candidates if c.get('valid') and c['validity']['status'] == 'valid']
        require(len(valid) == selection['valid_count'], 'Saved valid count differs from diagnostics')
        require(not any(c.get('valid') and c['validity']['status'] != 'valid' for c in candidates),
                'A candidate is flagged valid while geometry status is invalid')
        decision = select_priority(valid, selection.get('tie_tolerances'))
        selected = selection.get('selected_grasp')
        if valid:
            require(selection['status'] == 'selected' and isinstance(selected, dict),
                    'Valid candidates exist without a selected grasp')
            require(selected.get('valid') and selected['validity']['status'] == 'valid',
                    'Selected grasp is not geometrically valid')
            require(selected['candidate_id'] == decision['selected_candidate']['candidate_id'],
                    'Selected candidate differs from saved-candidate priority audit')
        else:
            require(selected is None and selection['status'] != 'selected',
                    'Failed or unavailable case contains a selected fallback')
        p = per_part.setdefault(part, dict(placements=0, selected=0, proposed=0, valid=0,
            unavailable_placements=0, table_clear_proposals=0, table_blocked_proposals=0,
            table_minima_m=[], selected_table_minima_m=[], blocking_components=Counter(),
            wall_seconds=0.))
        p['placements'] += 1
        p['selected'] += int(selected is not None)
        p['proposed'] += len(candidates)
        p['valid'] += len(valid)
        p['unavailable_placements'] += int('unavailable' in selection['status'])
        p['wall_seconds'] += float(row['wall_seconds'])
        totals.update(placements=1, proposed=len(candidates), valid=len(valid), selected=int(selected is not None))
        if not candidates:
            case_audits.append(dict(scene=row['scene'], part=part, candidates_checked=0,
                                    selected=False, status=selection['status']))
            continue
        pickup = rigid_transform(row['world_T_cad'], 'observed pickup CAD pose')
        plane = np.asarray(row['table_plane_world'], float)
        require(plane.shape == (4,) and np.isfinite(plane).all()
                and abs(np.linalg.norm(plane[:3])-1) <= 1e-8, 'Invalid saved table plane')
        if part not in request_cache:
            prepared = build_request(part, pickup, plane, registry=registry, hand=hand)
            require(prepared['status'] == 'available', f'Cannot reconstruct inputs for audit: {part}')
            request_cache[part] = prepared['request']
        request = request_cache[part]
        reference = reference_from_request(request)
        final = rigid_transform(request['world_T_cad_final'], 'corrected final CAD pose')
        for name, expected in reference.items():
            close(selection['reference'][name], expected, 'Saved fixed CAD criterion reference differs', 1e-10)
        selected_table = None
        for candidate in candidates:
            require(candidate.get('hand_pose_reference') == FINGERTIP_FRAME,
                    'Candidate does not identify the physical fingertip frame')
            inset = float(candidate['contact_inset_from_pad_tip_m'])
            if inset not in tip_cache:
                tip_cache[inset] = fingertip_calibration(hand, inset)
            tip = tip_cache[inset]
            tip_pose = rigid_transform(candidate['cad_T_hand'], 'saved fingertip pose')
            grip = rigid_transform(candidate['cad_T_grip_site'], 'saved grip-site pose')
            close(tip_pose, grip@tip['grip_site_T_fingertip'], 'Fingertip/grip-site transforms disagree')
            anchors = (tip['contact_anchors_zero_grip_site_m']
                       + float(candidate['holding_opening_m'])*tip['contact_anchors_per_opening'])
            actual_contacts = transform(anchors, grip)
            contacts = np.asarray(candidate['contacts_cad_m'], float)
            close(actual_contacts, contacts, 'Near-tip anchors do not reconstruct saved contacts')
            max_anchor_error = max(max_anchor_error, float(np.linalg.norm(actual_contacts-contacts, axis=1).max()))
            table_min, witness, component_minima = candidate_table_clearance(
                candidate, pickup, plane, hand, return_component_minima=True)
            p['table_minima_m'].append(table_min)
            tolerance = float(candidate['validity'].get('tolerances', {}).get('table_clearance_m', .0001))
            is_table_clear = table_min >= tolerance
            p['table_clear_proposals'] += int(is_table_clear)
            p['table_blocked_proposals'] += int(not is_table_clear)
            if not is_table_clear:
                p['blocking_components'][witness['component']] += 1
            whole_count, component_count = compare_saved_table_clearances(
                candidate['validity'], table_min, component_minima)
            totals['saved_table_values_compared'] += whole_count
            totals['saved_component_table_values_compared'] += component_count
            totals['all_candidate_table_minima_recomputed'] += 1
            totals['all_candidate_anchor_pairs_reconstructed'] += 1
            if candidate in valid:
                calculated = raw_criteria(candidate, reference, pickup, plane)
                for name, expected in calculated.items():
                    close(candidate['criteria'][name], expected, 'Saved candidate criterion differs', 1e-9)
                    max_criterion_error = max(max_criterion_error, abs(float(candidate['criteria'][name])-expected))
                for name in ('r_tau', 'r_surface'):
                    require(-1e-10 <= candidate['criteria'][name] <= 1+1e-10, 'Normalized criterion outside [0,1]')
                require(0 <= candidate['criteria']['theta_down_rad'] <= np.pi+1e-10,
                        'Downward angle outside [0,pi]')
                for field, expected in (
                    ('world_T_hand_pickup', pickup@tip_pose), ('world_T_hand_final', final@tip_pose),
                    ('world_T_grip_site_pickup', pickup@grip), ('world_T_grip_site_final', final@grip)):
                    rigid_transform(candidate[field], field)
                    close(candidate[field], expected, 'Saved pickup/final world hand transform differs')
                require(is_table_clear, 'A geometrically valid candidate fails the independent table audit')
                totals['valid_candidate_criteria_and_world_poses_recomputed'] += 1
            if selected is not None and candidate['candidate_id'] == selected['candidate_id']:
                selected_table = table_min
                p['selected_table_minima_m'].append(table_min)
                for field in ('cad_T_hand', 'cad_T_grip_site', 'contacts_cad_m', 'world_T_hand_pickup', 'world_T_hand_final'):
                    close(selected[field], candidate[field], 'Selected payload differs from diagnostic candidate')
        counts = diagnostic_counts(candidates)
        for category, statuses in counts.items():
            category_totals[category].update(statuses)
        require(counts == row.get('independent_check_counts'), 'Case diagnostic category totals differ')
        case_audits.append(dict(scene=row['scene'], part=part, candidates_checked=len(candidates),
            selected=selected is not None, status=selection['status'],
            selected_candidate_id=selected['candidate_id'] if selected is not None else None,
            selected_table_minimum_clearance_m=selected_table,
            independent_check_counts=counts))
    for part, data in per_part.items():
        data['blocking_components'] = dict(data['blocking_components'])
        for key in ('table_minima_m', 'selected_table_minima_m'):
            values = data.pop(key)
            data[key[:-2]+'_range_m'] = [min(values), max(values)] if values else None
    require(totals['selected'] == summary['selected_placements'], 'Summary selected count differs')
    require(totals['proposed'] == summary['proposed_poses'], 'Summary proposed count differs')
    require(totals['valid'] == summary['valid_poses'], 'Summary valid count differs')
    return dict(variant=folder.name, totals=dict(totals), per_part=per_part,
        status_counts=dict(Counter(row['selection']['status'] for row in rows)),
        independent_check_counts={key:dict(value) for key,value in sorted(category_totals.items())},
        maximum_anchor_reconstruction_error_m=max_anchor_error,
        maximum_raw_or_normalized_criterion_error=max_criterion_error,
        case_audits=case_audits), rows, summary, files


def report(comparison_root):
    folder = Path(comparison_root)
    manifest = read(folder/'launch_manifest.json')
    comparison = read(folder/'comparison.json')
    require(manifest['settings']['max_poses'] == 100, 'The declared whole-pose cap must be 100')
    for group in ('source_sha256', 'protected_sha256'):
        for name, expected in manifest[group].items():
            require(sha(ROOT/name) == expected, f'Launch hash changed: {name}')
    hand, registry = read(HAND), read(REGISTRY)
    require(sha(HAND) == manifest['hand_model_sha256'], 'Hand model changed since launch')
    require(sha(REGISTRY) == manifest['corrected_registry_sha256'], 'Corrected goal registry changed since launch')
    require(sha(ROOT/manifest['observed_manifest']) == manifest['observed_manifest_sha256'],
            'Observed CAD input manifest changed since launch')
    audits, records, summaries, watched = {}, {}, {}, {}
    requests = {}
    for variant in VARIANTS:
        audits[variant], records[variant], summaries[variant], files = audit_variant(
            folder/variant, hand, registry, requests)
        watched.update(files)
    previous = {(row['scene'], row['part']):row for row in records['previous']}
    refined = {(row['scene'], row['part']):row for row in records['refined']}
    require(previous.keys() == refined.keys(), 'Variants use different placements')
    for key, old in previous.items():
        new = refined[key]
        for field in ('seed', 'mesh_path', 'world_T_cad', 'table_plane_world'):
            require(old[field] == new[field], f'Controlled input differs: {key}, {field}')
    for part in audits['previous']['per_part']:
        old = read(folder/'previous'/'task_inputs'/f'{part}.json')['source']
        new = read(folder/'refined'/'task_inputs'/f'{part}.json')['source']
        require(old == new, 'Variants use different task, hand, region or receiver input provenance')
    require(summaries['previous']['workers'] == summaries['refined']['workers'], 'Worker counts differ')
    for variant in VARIANTS:
        require(comparison['variants'][variant] == summaries[variant], 'Comparison summary differs from saved variant')
    historical_summary = read(HISTORICAL/'summary.json')
    historical = dict(path=str(HISTORICAL),
        selected_placements=historical_summary['outcome_counts'].get('selected', 0),
        placements=historical_summary['placements'],
        elapsed_seconds=historical_summary['elapsed_seconds'],
        controlled_comparison=False,
        reason='Different assembly goals, sampling region and pad-contact convention; numerical improvement cannot be attributed to one change.')
    verification = dict(status='passed', placements_per_variant=96, variants=audits,
        post_launch_audit_helper=dict(path=str(Path(__file__).resolve()),
            sha256=sha(__file__), added_after_original_launch=True,
            preserved_frozen_report_sha256=sha(Path(__file__).with_name('report.py')),
            difference='Per-case diagnostic counters replace retained cross-case dictionaries. Post-processing table comparisons use matching whole-hand or component scope; grasp data and clearance calculations are unchanged.'),
        controlled_inputs_equal=True, source_and_protected_launch_hashes_match=True,
        hand_registry_and_observed_manifest_hashes_match=True,
        same_worker_count=True, whole_pose_cap=100,
        saved_candidate_priority_recomputed=True,
        criteria_recomputed_from_fixed_CAD=True,
        original_near_tip_anchors_and_world_transforms_reconstructed=True,
        all_candidate_full_hand_table_clearance_recomputed=True,
        table_audit_method='All complete hand vertices at holding and pre-opening endpoints; exact for declared affine opening motion against a fixed table plane.',
        collision_engine_rerun=False,
        geometry_validity_limit='Saved target/receiver/contact/protected-region check outcomes are counted; this audit does not independently re-run their collision or contact engines.',
        category_counts_nonexclusive=True, unavailable_is_success=False,
        pickup_or_assembly_success_measured=False, VLM_selection_excluded=True,
        historical_reference=historical)
    paths = dict(report=folder/'report.md', verification=folder/'verification.json',
                 placement_counts=folder/'placement_counts.csv')
    with paths['placement_counts'].open('w', newline='', encoding='utf-8') as stream:
        fields = ['variant','scene','part','status','proposed','valid','selected_candidate_id',
                  'r_tau','r_surface','q_tau_m','q_surface_m','theta_down_rad','wall_seconds',
                  'selected_table_minimum_clearance_m']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            case_lookup = {(c['scene'], c['part']):c for c in audits[variant]['case_audits']}
            for row in records[variant]:
                selected = row['selection'].get('selected_grasp')
                values = dict(variant=variant, scene=row['scene'], part=row['part'],
                    status=row['selection']['status'], proposed=row['selection']['proposed_count'],
                    valid=row['selection']['valid_count'], wall_seconds=row['wall_seconds'],
                    selected_candidate_id=selected['candidate_id'] if selected else None,
                    selected_table_minimum_clearance_m=case_lookup[(row['scene'], row['part'])].get('selected_table_minimum_clearance_m'))
                values.update({key:selected['criteria'][key] if selected else None
                               for key in ['r_tau','r_surface','q_tau_m','q_surface_m','theta_down_rad']})
                writer.writerow(values)
    before, after = summaries['previous'], summaries['refined']
    lines = [
        '# Saved CAD grasp comparison', '',
        f"The refined construction selected a geometrically valid grasp in {after['selected_placements']}/96 placements; the controlled previous construction selected one in {before['selected_placements']}/96. These are geometry outcomes, not pickup or assembly success.", '',
        '| Trial | Selected placements | Proposed poses | Valid poses | Total wall time | Summed case time |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for variant in VARIANTS:
        s = summaries[variant]
        lines.append(f"| {variant.capitalize()} | {s['selected_placements']}/96 | {s['proposed_poses']} | {s['valid_poses']} | {s['elapsed_wall_seconds']:.2f} s | {s['summed_case_seconds']:.2f} s |")
    lines += ['', 'Both controlled trials use the same 96 saved alignments, corrected assembly goals, protected/allowed contact regions, complete hand model, seeds, 100-pose cap, worker count and exhaustive checker framework. The refined trial uses the revised surface contact policy in that framework. Candidate sets differ, so the same seed does not imply identical contacts.', '',
        'Total wall time includes input/model preparation, sampling, geometry checks, selection, worker startup and saved-case writes. It excludes galleries and this audit. Summed case time is not elapsed wall time when workers run concurrently.', '',
        f"Historical October 28 reference: {historical['selected_placements']}/{historical['placements']} placements selected; {historical['elapsed_seconds']:.2f} s. It used different goals, sampling regions and pad-contact conventions. This is not a controlled baseline, and the changes cannot be assigned a single causal improvement from that comparison.", '',
        '## Per-part outcomes', '',
        '| Part | Previous selected / 8 | Previous proposed | Previous valid | Refined selected / 8 | Refined proposed | Refined valid |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for part in sorted(audits['previous']['per_part']):
        a, b = (audits[v]['per_part'][part] for v in VARIANTS)
        lines.append(f"| {part.replace('_',' ')} | {a['selected']}/8 | {a['proposed']} | {a['valid']} | {b['selected']}/8 | {b['proposed']} | {b['valid']} |")
    lines += ['', '## Independent check outcomes', '',
        'Each candidate receives several checks. The counts below are nonexclusive: one pose can fail several categories. An unavailable or unperformed check is not a passed check; a case without a valid selected grasp is not a successful case.', '',
        '| Trial | Check category | Saved status | Number of poses |', '|---|---|---|---:|']
    for variant in VARIANTS:
        for category, statuses in audits[variant]['independent_check_counts'].items():
            for status, count in sorted(statuses.items()):
                lines.append(f"| {variant} | {category.replace('_',' ')} | {status.replace('_',' ')} | {count} |")
    lines += ['', '## The 8 mm round pin and the complete hand', '']
    small = 'RGOCG8-50_8mm'
    for variant in VARIANTS:
        p = audits[variant]['per_part'].get(small)
        if p is None:
            continue
        limits = p['table_minima_range_m']
        interval = f"{limits[0]*1000:.3f} to {limits[1]*1000:.3f} mm" if limits else 'unavailable (no proposed pose)'
        lines.append(f"- {variant.capitalize()}: {p['proposed']} proposals; {p['table_clear_proposals']} pass the saved table-clearance threshold when independently checked; {p['table_blocked_proposals']} do not. The minimum complete-hand distance above the table ranges from {interval}. Blocking-component counts: {p['blocking_components'] or 'none'}.")
    lines += ['', 'The table audit keeps every saved palm, finger and pad vertex. It uses both the holding and pre-opening positions, with the saved grip-site transform; the physical fingertip is a separate reference. A table-clear proposal may still fail contact alignment, part overlap, protected-surface or receiver checks.', '',
        '## Verification and limits', '',
        'The audit checks 96 distinct placements per variant, the pose cap, valid selected payloads, deterministic priority among saved valid candidates, normalized criterion bounds, fixed-CAD criterion values, rigid pickup/final transforms, near-tip contact-anchor reconstruction, complete-hand table minima, and launch hashes. It does not generate candidates or rerun the target/receiver collision engine.', '',
        'The first criterion is a geometric assembly-axis moment arm, not measured torque. The second uses the area-weighted CAD surface centroid, not measured center of mass. Neither predicts closing-induced movement or verifies physical insertion. VLM selection is excluded; pickup, assembly, insertion-path and release execution remain paused.', '',
        'Machine-readable case audits are in verification.json; placement counts and selected measurements are in placement_counts.csv. The selected-contact and pickup-side galleries are saved separately.', '',
    ]
    paths['report'].write_text('\n'.join(lines), encoding='utf-8')
    require(all(sha(path) == expected for path, expected in watched.items()),
            'Saved experiment evidence changed during reporting')
    verification['saved_results_and_diagnostics_unchanged'] = True
    dump(paths['verification'], verification)
    print('REPORT='+str(paths['report']), flush=True)
    return paths


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_refinement_20261029.report COMPARISON_ROOT')
    report(sys.argv[1])
