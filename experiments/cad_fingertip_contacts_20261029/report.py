"""Lightweight saved-result comparison and selected finger-contact height audit.

No candidates, contact surfaces or collision decisions are generated here.
Contact coordinates are read from the selected candidates and independently
transformed back into the unchanged original grip-site frame for comparison.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import rigid_transform


ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'
CALIBRATION = ROOT/'outputs/cad_fingertip_contacts_20261029/contact_audit.json'
BASELINE = ROOT/'outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def hand_digest(hand):
    fields = {key:hand[key] for key in ('frame','opening_limits_m','components')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def selected_contact_measurements(row, hand_distal_z, *, require_actual_field=True):
    """Z is along finger +Z in the original grip-site frame, not world height."""
    selection = row['selection']
    pose = selection.get('selected_grasp')
    if selection['status'] != 'selected':
        require(pose is None, 'A failed case contains a selected fallback')
        return None
    require(isinstance(pose,dict) and pose.get('valid')
            and pose.get('validity',{}).get('status') == 'valid',
            'Selected candidate must have passed its saved geometry checks')
    grip = rigid_transform(pose['cad_T_grip_site'], 'selected CAD-to-grip-site pose')
    pair = np.asarray(pose['contacts_cad_m'],float)
    require(pair.shape == (2,3) and np.isfinite(pair).all(), 'Missing selected CAD contact pair')
    reconstructed = (pair-grip[:3,3])@grip[:3,:3]
    actual = pose.get('actual_contacts_hand_m')
    if require_actual_field:
        require(actual is not None, 'Selected candidate lacks actual_contacts_hand_m in original grip-site frame')
    actual = np.asarray(reconstructed if actual is None else actual,float)
    require(actual.shape == (2,3) and np.isfinite(actual).all(), 'Invalid actual hand-frame contact coordinates')
    error = float(np.linalg.norm(actual-reconstructed,axis=1).max())
    require(error <= 1e-8, 'actual_contacts_hand_m disagrees with saved CAD/grip-site contact transform')
    pickup = rigid_transform(row['world_T_cad'], 'saved observed CAD pose')
    world = pair@pickup[:3,:3].T+pickup[:3,3]
    plane = np.asarray(row['table_plane_world'],float)
    require(plane.shape == (4,) and np.isfinite(plane).all()
            and abs(np.linalg.norm(plane[:3])-1) <= 1e-8, 'Invalid saved table plane')
    return dict(scene=row['scene'],part=row['part'],candidate_id=pose['candidate_id'],
        contacts_grip_site_m=actual.tolist(), contact_z_grip_site_m=actual[:,2].tolist(),
        contact_height_above_table_m=(world@plane[:3]+plane[3]).tolist(),
        full_hand_distal_extent_beyond_contact_m=(hand_distal_z-actual[:,2]).tolist(),
        actual_field_recorded=pose.get('actual_contacts_hand_m') is not None,
        hand_frame_reconstruction_error_m=error,
        contact_calibration_source=pose.get('contact_calibration_source',row.get('contact_calibration_source')))


def _check_counts(rows, summary):
    require(len(rows) == 96 and len({(r['scene'],r['part']) for r in rows}) == 96,
            'Expected 96 distinct NIST placements')
    require(sorted({r['scene'] for r in rows}) == [f'N{i:02d}' for i in range(8)]
            and len({r['part'] for r in rows}) == 12, 'Expected eight scenes and twelve part types')
    require(all(0 <= r['selection']['valid_count'] <= r['selection']['proposed_count'] <= 100 for r in rows),
            'Invalid candidate counts or 100-pose cap exceeded')
    totals = dict(selected_placements=sum(r['selection']['status'] == 'selected' for r in rows),
        valid_poses=sum(r['selection']['valid_count'] for r in rows),
        proposed_poses=sum(r['selection']['proposed_count'] for r in rows))
    for key,value in totals.items():
        require(value == summary[key], 'Summary differs from saved results: '+key)
    return totals


def report(run_dir, *, baseline_dir=BASELINE, calibration_path=None, hand_path=HAND):
    run_dir, baseline_dir, hand_path = Path(run_dir).resolve(), Path(baseline_dir).resolve(), Path(hand_path).resolve()
    calibration_path = Path(calibration_path) if calibration_path else next(
        (run_dir/name for name in ('contact_calibration_source.json','contact_calibration.json','contact_audit.json')
         if (run_dir/name).is_file()), CALIBRATION)
    evidence = [run_dir/'results.json',run_dir/'summary.json',baseline_dir/'results.json',
                baseline_dir/'summary.json',hand_path,calibration_path]
    before = {str(path):sha(path) for path in evidence}
    rows, summary = read(run_dir/'results.json'),read(run_dir/'summary.json')
    old_rows, old_summary = read(baseline_dir/'results.json'),read(baseline_dir/'summary.json')
    new_totals, old_totals = _check_counts(rows,summary),_check_counts(old_rows,old_summary)
    lookup = {(r['scene'],r['part']):r for r in old_rows}
    require(set(lookup) == {(r['scene'],r['part']) for r in rows}, 'Baseline placement identities differ')
    for row in rows:
        for key in ('world_T_cad','table_plane_world','mesh_path'):
            require(row[key] == lookup[(row['scene'],row['part'])][key],
                    'Observed alignment, table or CAD source changed in contact comparison')
    hand, calibration = read(hand_path),read(calibration_path)
    if isinstance(calibration.get('calibration'),dict):
        calibration = calibration['calibration']
    source = calibration.get('provenance',{})
    require(source.get('source_hand_model_sha256') == sha(hand_path),
            'Contact calibration is not bound to the unchanged original hand file')
    require(calibration.get('hand_geometry_sha256') == hand_digest(hand),
            'Original hand component vertices, opening motion or geometry labels changed')
    hand_distal_z = float(max(np.asarray(c['vertices_zero_m'])[:,2].max() for c in hand['components']))
    contacts = [m for row in rows if (m:=selected_contact_measurements(row,hand_distal_z)) is not None]
    old_contacts = [m for row in old_rows if (m:=selected_contact_measurements(
        row,hand_distal_z,require_actual_field=False)) is not None]
    parts = sorted({row['part'] for row in rows})
    per_part, gained, lost = {},[],[]
    for row in rows:
        before_selected = lookup[(row['scene'],row['part'])]['selection']['status'] == 'selected'
        after_selected = row['selection']['status'] == 'selected'
        if after_selected and not before_selected:
            gained.append(dict(scene=row['scene'],part=row['part']))
        if before_selected and not after_selected:
            lost.append(dict(scene=row['scene'],part=row['part']))
    for part in parts:
        per_part[part] = {}
        for name,values in (('baseline',old_rows),('finger_contacts',rows)):
            selected = [r for r in values if r['part'] == part]
            per_part[part][name] = dict(selected=sum(r['selection']['status'] == 'selected' for r in selected),
                valid=sum(r['selection']['valid_count'] for r in selected),
                proposed=sum(r['selection']['proposed_count'] for r in selected))
    def contact_range(values,key):
        flat = [number for row in values for number in row[key]]
        return [min(flat),max(flat)] if flat else None
    numerical = dict(baseline_directory=str(baseline_dir),baseline=old_totals,new=new_totals,
        per_part=per_part,gained_placements=gained,lost_placements=lost,
        selected_contact_z_grip_site_range_m=contact_range(contacts,'contact_z_grip_site_m'),
        baseline_contact_z_grip_site_range_m=contact_range(old_contacts,'contact_z_grip_site_m'),
        selected_remaining_finger_overhang_range_m=contact_range(contacts,'full_hand_distal_extent_beyond_contact_m'),
        original_hand_distal_z_grip_site_m=hand_distal_z,
        selected_contact_measurements=contacts,
        calibration_source=str(calibration_path),calibration_source_sha256=sha(calibration_path),
        calibrated_anchor_positions_zero_grip_site_m=calibration.get('contact_anchors_zero_grip_site_m'),
        calibrated_physical_fingertip_origin_grip_site_m=calibration.get('physical_fingertip_origin_grip_site_m'),
        calibrated_full_finger_extent_beyond_anchor_m=calibration.get('full_finger_extent_beyond_anchor_m'),
        original_hand_path=str(hand_path),original_hand_sha256=sha(hand_path),
        original_complete_hand_geometry_unchanged=True,
        contacts_on_original_saved_collision_hull_faces=True,no_synthetic_pad_extension=True,
        physical_contact_accuracy_validated=False,
        visual_groove_caveat=source.get('visual_geometry_caveat',
            'The detailed visual finger has a central groove bridged by the coarse convex collision hull.'),
        candidate_generation_or_collision_revalidation=False,
        VLM_selection_excluded=True,pickup_or_assembly_success_measured=False,
        source_sha256=before,reporter_sha256=sha(__file__))
    with (run_dir/'selected_contacts.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scene','part','candidate_id','left_contact_z_grip_site_mm',
            'right_contact_z_grip_site_mm','left_contact_above_table_mm','right_contact_above_table_mm',
            'left_remaining_finger_overhang_mm','right_remaining_finger_overhang_mm',
            'hand_frame_reconstruction_error_m'])
        for row in contacts:
            writer.writerow([row['scene'],row['part'],row['candidate_id'],
                *[v*1000 for v in row['contact_z_grip_site_m']],
                *[v*1000 for v in row['contact_height_above_table_m']],
                *[v*1000 for v in row['full_hand_distal_extent_beyond_contact_m']],
                row['hand_frame_reconstruction_error_m']])
    def mm_range(value):
        return f'{value[0]*1000:.3f} to {value[1]*1000:.3f} mm' if value else 'unavailable (no selected grasps)'
    elapsed = summary.get('elapsed_wall_seconds')
    time_text = f'{elapsed:.2f} s' if isinstance(elapsed,(int,float)) else 'not recorded'
    lines = ['# Finger-surface contact trial','',
        f"The saved finger-contact trial selected a geometrically valid grasp in {new_totals['selected_placements']}/96 placements, with {new_totals['valid_poses']} valid poses from {new_totals['proposed_poses']} proposals. The preceding baseline selected {old_totals['selected_placements']}/96, with {old_totals['valid_poses']} valid poses from {old_totals['proposed_poses']} proposals.",'',
        f"{len(gained)} placements gained a selected grasp and {len(lost)} lost one. These counts measure modeled geometric availability, not pickup or assembly success.",'',
        '## What moved and what stayed fixed','',
        'The contact anchors are on existing inward-facing triangles of the saved finger collision hulls. The full original palm, fingers and pad solids remain present. No pad box was lengthened and no hand vertices were removed. The physical fingertip pose reference remains distinct from the actual contact positions.','',
        f"For selected grasps, contact Z in the original grip-site frame ranges from {mm_range(numerical['selected_contact_z_grip_site_range_m'])}; baseline contacts range from {mm_range(numerical['baseline_contact_z_grip_site_range_m'])}. Z follows the finger axis and is not height above the table.",'',
        f"The saved hand's furthest vertex is at grip-site Z = {hand_distal_z*1000:.3f} mm. Remaining full-hand extent beyond each selected contact along +Z ranges from {mm_range(numerical['selected_remaining_finger_overhang_range_m'])}. Actual contacts were read from each selected candidate's actual_contacts_hand_m and checked against its CAD-to-grip-site transform.",'',
        '**Collision-model limitation:** '+numerical['visual_groove_caveat']+' The study does not establish that the physical finger has contact material at these locations. Physical contact calibration, holding and precise assembly remain unverified.','',
        '## Per-part comparison','',
        '| Part | Baseline selected / 8 | Baseline valid | New selected / 8 | New valid | New proposed |',
        '|---|---:|---:|---:|---:|---:|']
    for part,value in per_part.items():
        old,new = value['baseline'],value['finger_contacts']
        lines.append(f"| {part.replace('_',' ')} | {old['selected']}/8 | {old['valid']} | {new['selected']}/8 | {new['valid']} | {new['proposed']} |")
    lines += ['', '## Timing and evidence','',
        f"New trial elapsed wall time: {time_text}. The {old_totals['selected_placements']}/96 baseline was assembled from 72 original and 24 corrected cases and has no standalone complete-run wall time. Its original trial and supplemental correction times must remain separate; this report does not invent a baseline speedup.",'',
        'Saved alignments, table planes and CAD mesh paths match the baseline. The contact calibration is bound to the original hand file hash and a digest of every complete hand component and opening motion. This report verifies selected contact coordinates and counts; it does not rerun candidate generation or collision checking.','',
        'VLM selection is excluded. Pickup and assembly execution remain paused. The gallery contains selected CAD contacts and pickup side views, with no final assembly views.','',
        '[Selected contacts and pickup side views](gallery/index.html) · [Contact measurements](selected_contacts.csv) · [Machine-readable report](contact_report.json)','']
    (run_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    require(all(sha(path) == digest for path,digest in before.items()), 'Saved evidence changed during reporting')
    numerical['saved_evidence_unchanged'] = True
    (run_dir/'contact_report.json').write_text(json.dumps(numerical,indent=2)+'\n',encoding='utf-8')
    print('REPORT='+str(run_dir/'report.md'),flush=True)
    return run_dir/'report.md'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_fingertip_contacts_20261029.report RUN_DIRECTORY')
    report(sys.argv[1])
