"""CAD-only gear/nut goal review; never runs grasping, motion or dynamics."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.fabrica_goal_geometry_20261027.task_inputs import geometry_spec
from .cad_tools import read_native_mesh, conversion_receipt

ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT / 'outputs/simulation/nist_task_board_1/assets/meshes'
OUTPUT = ROOT / 'outputs/nist_mating_goals_20261028'


def paired_export_transform(part, receiver):
    """Recover a relative rigid pose IF paired original exports share a frame.

    Matching origins do not by themselves establish assembly correctness. The
    caller records that assumption and the separate geometry limitations.
    """
    receipts = [conversion_receipt(n, MESHES/f'{n}.STL', MESHES/f'{n}_m.stl')
                for n in (part, receiver)]
    affine = [np.asarray(r['source_mm_to_native_m_affine']) for r in receipts]
    result = affine[1] @ np.linalg.inv(affine[0])
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError('Source conversion chain is not a rigid metre transform')
    return result, receipts


def receivers(spec):
    return [dict(name=r['name'], mesh_path=Path(r['mesh_path']).relative_to(ROOT).as_posix(),
                 board_T_mesh=np.asarray(r['board_T_mesh']).tolist())
            for r in spec['receivers']]


def review():
    records = []
    for part, name in [('Gear_Large', 'Gear_Shaft'), ('M16_Hex_Nut', 'M16_Screw')]:
        spec = geometry_spec(part)
        relative, receipts = paired_export_transform(part, name)
        receiver = next(r for r in spec['receivers'] if name in r['name'])
        world_receiver = np.asarray(receiver['board_T_mesh'])
        recovered = world_receiver @ relative
        old = np.asarray(spec['goal_T_cad_board'])
        v, f = read_native_mesh(MESHES/f'{part}_m.stl')
        source_paths = [MESHES/f'{n}{suffix}' for n in (part, name)
                        for suffix in ('.STL', '_m.stl')]
        evidence = [{'path':p.relative_to(ROOT).as_posix(),
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in source_paths]
        measures = dict(receiver_T_part_from_paired_exports=relative.tolist(),
                        source_conversion_receipts=receipts,
                        recovered_minus_old_translation_m=(recovered[:3, 3]-old[:3, 3]).tolist(),
                        original_shared_frame_is_pair_specific=True,
                        original_source_frame_is_not_the_board_frame=True)
        if part == 'Gear_Large':
            display = recovered
            mating = np.diag([1., -1., -1., 1.])
            mating[2, 3] = float(v[:, 2].min())
            receiver_frame = display @ mating
            status = 'source_CAD_pose_recovered_full_fit_not_certified'
            changes = ['Retain the existing coaxial seated height: it agrees with the original paired exports.',
                       'Recover native relative orientation and translation from the source gear/shaft pair instead of assuming them from bounding boxes.']
            warnings = ['Gear has separate small set-screw geometry and a recorded seam in its native mesh.',
                        'The source CAD pose is static reference evidence, not a validated insertion path or tightened set screw.',
                        'Tooth engagement with other gears is outside this twelve-part review.']
            proposed = recovered.tolist()
            measures.update(seating_face_cad_z_m=float(v[:, 2].min()),
                            bore_mouth_radius_m=.005300000659569425,
                            axis_direction_basis='bore-mouth circles and source coaxial shaft',
                            receiver_axis_direction_board=[0., 0., 1.])
        else:
            # Keep a board-seat preview while explicitly withholding an exact
            # threaded mating goal. Original exports alone do not fix phase.
            display = old.copy()
            mating = np.diag([1., -1., -1., 1.])
            mating[2, 3] = float(v[:, 2].min())
            receiver_frame = display @ mating
            status = 'thread_phase_and_receiver_mount_require_resolution'
            proposed = None
            changes = ['Replace the old zero-yaw seated claim with an unresolved threaded-goal status.',
                       'Record the source nut/screw relative transform as a second reference, without treating it as a validated thread fit.']
            warnings = ['The illustration retains the old board-seat height and zero yaw only to expose the unresolved thread relationship.',
                        'Exact thread phase, axial seating and current screw-to-board mounting must agree before a final goal is enabled.',
                        'A nut requires a rotation/translation thread relationship; downward axial alignment alone is insufficient.']
            measures.update(source_recovered_board_T_part=recovered.tolist(),
                            illustrated_pose_is_not_a_verified_seated_goal=True,
                            seating_face_cad_z_m=float(v[:, 2].min()),
                            receiver_axis_direction_board=[0., 0., 1.])
            audit_file = OUTPUT/'m16_thread_audit.json'
            if audit_file.is_file():
                audit = json.loads(audit_file.read_text())
                measures['thread_and_mount_audit'] = audit
                corrected_receiver = np.asarray(audit['corrected_mount_candidate']['board_T_screw'])
                display = corrected_receiver @ relative
                receiver_frame = display @ mating
                changes.append('Lower the screw reference mount by one measured board thickness so its head shoulder meets the board underside.')
                warnings[0] = ('The illustration uses the corrected screw mount and source axial seat relation; '
                               'source zero-yaw threads still overlap, so this is not an enabled final goal.')
        record = dict(part=part,
            part_mesh=(MESHES/f'{part}_m.stl').relative_to(ROOT).as_posix(),
            receivers=receivers(spec), old_board_T_part=old.tolist(),
            review_board_T_part=display.tolist(), proposed_board_T_part=proposed,
            axis_direction_cad=[0., 0., -1.], axis_direction_board=[0., 0., -1.],
            part_mating_frame_cad=mating.tolist(), receiver_mating_frame_board=receiver_frame.tolist(),
            status=status, measurements=measures, changes=changes, warnings=warnings,
            evidence=evidence)
        if part == 'M16_Hex_Nut' and 'thread_and_mount_audit' in measures:
            record['old_receivers'] = deepcopy(record['receivers'])
            for item in record['receivers']:
                if 'M16_Screw' in item['name']:
                    item['board_T_mesh'] = corrected_receiver.tolist()
            record['receiver_mount_correction_is_reference_only'] = True
            record['simulation_layout_modified'] = False
        records.append(record)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = dict(scope='CAD mating-pose review only', units='metres',
                  frame='board-centered XY; original nominal scene world Z',
                  grasp_generation=False, VLM_selection=False, pickup_execution=False,
                  assembly_execution=False, simulator_object_state_read=False, parts=records)
    (OUTPUT/'gears_nut_review.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    review()
