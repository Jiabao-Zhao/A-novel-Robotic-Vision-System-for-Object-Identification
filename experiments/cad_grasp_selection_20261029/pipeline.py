"""Current geometric selector with an explicit calibrated fingertip reference.

This adapter clips only the explicitly allowed origin surface and changes
pose/contact calibration. Rays and complete-hand checks retain the full CAD;
deterministic criterion priorities are unchanged. It reads no simulator state
and performs no pickup or assembly execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cad_grasp_selection_20261028.criteria import (
    build_reference, evaluate_criteria, select_priority,
)
from experiments.cad_grasp_selection_20261028.pipeline import (
    REQUIRED, UNCHECKED, _transform, _json, protected_triangle_check,
)
from .geometry import GeometryValidator
from .hand_frame import FINGERTIP_FRAME
from .sampler import generate_candidates


def select_grasp(request, *, seed=0, max_poses=100, surface_budget=200,
                 orientation_samples=10, pre_opening_clearance_m=.001,
                 geometry_config=None, tie_tolerances=None, diagnostics=None):
    result = dict(status='unavailable_inputs', selected_grasp=None,
        proposed_count=0, valid_count=0, seed=seed, unchecked=UNCHECKED.copy(),
        robot_or_physics_execution=False, contacts_frame='native_CAD',
        hand_pose_reference=FINGERTIP_FRAME, units='metres/radians',
        pose_input='caller-supplied estimated CAD alignment; no environment state read')
    if not isinstance(request, dict):
        return dict(result, message='Required inputs must be a dictionary.',
                    missing_or_invalid_inputs=['request'])
    missing = [key for key in REQUIRED if request.get(key) is None]
    declared_protected = request.get('protected_contact_region')
    if declared_protected is None and request.get('protected_face_indices') is not None:
        declared_protected = dict(type='face_ids', protected_face_ids=request['protected_face_indices'])
    if declared_protected is not None and 'protected_contact_check' in missing:
        missing.remove('protected_contact_check')
    if missing:
        return dict(result, message='Required grasp-selection inputs are unavailable.',
                    missing_or_invalid_inputs=missing)
    try:
        if request.get('simulator_object_poses_read', False):
            raise ValueError('Simulator object ground-truth poses are not permitted inputs')
        pickup = _transform(request['world_T_cad_pickup'], 'world_T_cad_pickup')
        final = _transform(request['world_T_cad_final'], 'world_T_cad_final')
        plane = np.asarray(request['table_plane_world'], dtype=float)
        if (plane.shape != (4,) or not np.isfinite(plane).all()
                or abs(np.linalg.norm(plane[:3])-1.) > 1e-8):
            raise ValueError('table_plane_world needs a finite upward UNIT normal')
        reference = build_reference(request['vertices_cad_m'], request['faces'],
                                    request['axis_point_cad_m'], request['axis_direction_cad'])
        sampling_mode = request.get('sampling_mode', 'assembly_region_guided')
        sampling_region = request.get('sampling_contact_region')
        if sampling_mode == 'assembly_region_guided' and sampling_region is None:
            from .sampling_regions import compile_sampling_region
            if declared_protected is None:
                raise ValueError('assembly_region_guided sampling requires explicit protected_contact_region labels or a compiled sampling_contact_region')
            sampling_region = compile_sampling_region(request['vertices_cad_m'], request['faces'],
                declared_protected, sampling_region=request.get('sampling_region'))
        protected_check = request.get('protected_contact_check')
        if protected_check is None:
            from .task_inputs import protected_checker
            protected_check = protected_checker(declared_protected, request['vertices_cad_m'], request['faces'])
        validator = GeometryValidator(request['vertices_cad_m'], request['faces'],
            request['hand'], pickup, final, plane, protected_check,
            request['receivers'], config=geometry_config)
        if validator.errors:
            raise ValueError('; '.join(validator.errors))
        candidates, sampling = generate_candidates(
            request['vertices_cad_m'], request['faces'], request['hand'], seed=seed,
            max_poses=max_poses, surface_budget=surface_budget,
            orientation_samples=orientation_samples,
            pre_opening_clearance_m=pre_opening_clearance_m,
            contact_inset_m=validator.tip['contact_inset_m'],
            sampling_contact_region=sampling_region, sampling_mode=sampling_mode)
        select_priority([], tie_tolerances)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return dict(result, message='Required geometry, calibration or configuration is unavailable or invalid.',
                    missing_or_invalid_inputs=[str(exc)])
    result.update(reference=reference, sampling=sampling, proposed_count=len(candidates))
    valid, unavailable = [], []
    for candidate in candidates:
        validation = validator.validate(candidate)
        record = dict(candidate, validity=validation)
        if validation['status'] == 'valid':
            try:
                approach_world = pickup[:3, :3]@np.asarray(candidate['approach_cad'])
                criteria = evaluate_criteria(candidate['contacts_cad_m'], reference,
                                              approach_world, plane[:3])
            except ValueError as exc:
                record['validity'] = dict(status='unavailable_input', valid=False,
                                          reasons=['criteria_input_error:'+str(exc)])
                unavailable.append(record['validity'])
            else:
                tip = np.asarray(candidate['cad_T_hand'])
                grip = np.asarray(candidate['cad_T_grip_site'])
                record.update(criteria=criteria, valid=True,
                    world_T_hand_pickup=(pickup@tip).tolist(),
                    world_T_hand_final=(final@tip).tolist(),
                    world_T_grip_site_pickup=(pickup@grip).tolist(),
                    world_T_grip_site_final=(final@grip).tolist(),
                    approach_world=approach_world.tolist())
                valid.append(record)
        elif validation['status'].startswith('unavailable'):
            unavailable.append(validation)
        if diagnostics is not None:
            diagnostics.append(record)
    decision = select_priority(valid, tie_tolerances)
    result.update(valid_count=len(valid), unavailable_candidate_count=len(unavailable),
        selection_stages=decision['stages'], tie_tolerances=decision['tie_tolerances'],
        final_tied_candidate_ids=decision['final_tied_candidate_ids'])
    if valid:
        result.update(status='selected', selected_grasp=decision['selected_candidate'],
                      message='Selected one geometrically valid grasp by explicit criterion priority.')
    elif unavailable:
        result.update(message='Required candidate geometry or protected surfaces are unavailable.',
                      missing_or_invalid_inputs=[entry.get('reasons', []) for entry in unavailable])
    else:
        result.update(status='no_valid_grasp_within_budget',
                      message='No valid grasp found within the sampling budget.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    diagnostics = [] if args.diagnostics else None
    try:
        request = json.loads(args.request.read_text(encoding='utf-8-sig'))
        if not isinstance(request, dict):
            raise ValueError('Request JSON must contain an object')
        if 'hand_model_file' in request and 'hand' not in request:
            path = Path(request['hand_model_file'])
            request['hand'] = json.loads((path if path.is_absolute() else args.request.parent/path).read_text(encoding='utf-8-sig'))
        if 'protected_face_indices' in request and 'protected_contact_check' not in request:
            request['protected_contact_check'] = protected_triangle_check(
                request['vertices_cad_m'], request['faces'], request['protected_face_indices'])
        result = select_grasp(request, seed=args.seed, diagnostics=diagnostics,
            geometry_config=request.get('geometry_config'), tie_tolerances=request.get('tie_tolerances'))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        result = dict(status='unavailable_inputs', selected_grasp=None,
            proposed_count=0, valid_count=0, seed=args.seed,
            message='Required request geometry or calibration is unavailable or invalid.',
            missing_or_invalid_inputs=[str(exc)], unchecked=UNCHECKED.copy(),
            hand_pose_reference=FINGERTIP_FRAME, robot_or_physics_execution=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=_json, allow_nan=False)+'\n', encoding='utf-8')
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2, default=_json, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], proposed_count=result['proposed_count'],
        valid_count=result['valid_count'], output=str(args.output))))


if __name__ == '__main__':
    main()
