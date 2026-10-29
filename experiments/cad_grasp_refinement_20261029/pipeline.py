"""Region-guided CAD grasps with shared contact rules and independent checks."""
from __future__ import annotations

import time

import numpy as np

from experiments.cad_grasp_selection_20261028.criteria import (
    build_reference, evaluate_criteria, select_priority,
)
from experiments.cad_grasp_selection_20261028.pipeline import REQUIRED, UNCHECKED, _transform
from experiments.cad_grasp_selection_20261029.hand_frame import FINGERTIP_FRAME
from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region
from experiments.cad_grasp_selection_20261029.task_inputs import protected_checker
from experiments.cad_grasp_selection_20261029 import sampler as previous_sampler
from .geometry import GeometryValidator
from .contact_model import build_contact_model
from .sampler import generate_candidates


def select_grasp(request, *, seed=0, max_poses=100, surface_budget=200,
                 orientation_samples=10, pre_opening_clearance_m=.001,
                 geometry_config=None, tie_tolerances=None, diagnostics=None,
                 sampler_variant='refined'):
    started = time.perf_counter()
    result = dict(status='unavailable_inputs', selected_grasp=None,
        proposed_count=0, valid_count=0, seed=seed, unchecked=UNCHECKED.copy(),
        robot_or_physics_execution=False, contacts_frame='native_CAD',
        hand_pose_reference=FINGERTIP_FRAME, units='metres/radians',
        sampler_variant=sampler_variant,
        pose_input='caller-supplied observed CAD alignment; no environment state read')
    if not isinstance(request, dict):
        return dict(result, message='Required inputs must be a dictionary.',
                    missing_or_invalid_inputs=['request'])
    missing = [key for key in REQUIRED if request.get(key) is None]
    protected = request.get('protected_contact_region')
    if protected is not None and 'protected_contact_check' in missing:
        missing.remove('protected_contact_check')
    if missing:
        return dict(result, message='Required grasp-selection inputs are unavailable.',
                    missing_or_invalid_inputs=missing)
    try:
        if sampler_variant not in ('previous', 'refined'):
            raise ValueError('sampler_variant must be previous or refined')
        if request.get('simulator_object_poses_read', False):
            raise ValueError('Simulator object ground-truth poses are not permitted inputs')
        pickup = _transform(request['world_T_cad_pickup'], 'world_T_cad_pickup')
        final = _transform(request['world_T_cad_final'], 'world_T_cad_final')
        plane = np.asarray(request['table_plane_world'], float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or abs(np.linalg.norm(plane[:3])-1) > 1e-8:
            raise ValueError('table_plane_world needs a finite unit normal and metre offset')
        reference = build_reference(request['vertices_cad_m'], request['faces'],
                                    request['axis_point_cad_m'], request['axis_direction_cad'])
        mode = request.get('sampling_mode', 'assembly_region_guided')
        region = request.get('sampling_contact_region')
        if mode == 'assembly_region_guided' and region is None:
            region = compile_sampling_region(request['vertices_cad_m'], request['faces'],
                protected, request.get('sampling_region'))
        protected_check = request.get('protected_contact_check')
        if protected_check is None:
            protected_check = protected_checker(protected, request['vertices_cad_m'], request['faces'])
        model = None
        if sampler_variant == 'refined':
            model = request.get('contact_model')
            if model is None:
                model = build_contact_model(request['vertices_cad_m'], request['faces'], region,
                    part=request.get('part'), axis_point_cad_m=request['axis_point_cad_m'],
                    axis_direction_cad=request['axis_direction_cad'])
        validator = GeometryValidator(request['vertices_cad_m'], request['faces'],
            request['hand'], pickup, final, plane, protected_check,
            request['receivers'], config=geometry_config, contact_model=model)
        if validator.errors:
            raise ValueError('; '.join(validator.errors))
        preparation_seconds = time.perf_counter()-started
        sampling_started = time.perf_counter()
        options = dict(seed=seed, max_poses=max_poses, surface_budget=surface_budget,
            orientation_samples=orientation_samples, pre_opening_clearance_m=pre_opening_clearance_m,
            contact_inset_m=validator.tip['contact_inset_m'],
            sampling_contact_region=region, sampling_mode=mode)
        if sampler_variant == 'refined':
            candidates, sampling = generate_candidates(request['vertices_cad_m'], request['faces'],
                request['hand'], contact_model=model,
                normal_angle_rad=validator.config['normal_angle_rad'], **options)
        else:
            candidates, sampling = previous_sampler.generate_candidates(
                request['vertices_cad_m'], request['faces'], request['hand'], **options)
        sampling_seconds = time.perf_counter()-sampling_started
        select_priority([], tie_tolerances)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return dict(result, message='Required geometry, calibration or configuration is unavailable or invalid.',
                    missing_or_invalid_inputs=[str(exc)])
    result.update(reference=reference, sampling=sampling, proposed_count=len(candidates),
        contact_model_provenance=model.provenance if model is not None else None)
    check_started = time.perf_counter()
    valid, unavailable = [], []
    for candidate in candidates:
        validation = validator.validate(candidate)
        record = dict(candidate, validity=validation)
        if validation['status'] == 'valid':
            try:
                approach_world = pickup[:3, :3]@np.asarray(candidate['approach_cad'])
                criteria = evaluate_criteria(candidate['contacts_cad_m'], reference, approach_world, plane[:3])
            except ValueError as exc:
                record['validity'] = dict(status='unavailable_input', valid=False,
                    reasons=['criteria_input_error:'+str(exc)], checks=validation.get('checks', {}),
                    diagnostics=validation.get('diagnostics', {}))
                unavailable.append(record['validity'])
            else:
                tip = np.asarray(candidate['cad_T_hand'])
                grip = np.asarray(candidate['cad_T_grip_site'])
                record.update(criteria=criteria, valid=True,
                    world_T_hand_pickup=(pickup@tip).tolist(), world_T_hand_final=(final@tip).tolist(),
                    world_T_grip_site_pickup=(pickup@grip).tolist(), world_T_grip_site_final=(final@grip).tolist(),
                    approach_world=approach_world.tolist())
                valid.append(record)
        elif validation['status'].startswith('unavailable'):
            unavailable.append(validation)
        if diagnostics is not None:
            diagnostics.append(record)
    decision = select_priority(valid, tie_tolerances)
    result.update(valid_count=len(valid), unavailable_candidate_count=len(unavailable),
        selection_stages=decision['stages'], tie_tolerances=decision['tie_tolerances'],
        final_tied_candidate_ids=decision['final_tied_candidate_ids'],
        timings_seconds=dict(preparation=preparation_seconds, sampling=sampling_seconds,
            validation_and_selection=time.perf_counter()-check_started,
            total=time.perf_counter()-started))
    if valid:
        result.update(status='selected', selected_grasp=decision['selected_candidate'],
            message='Selected one geometrically valid grasp by the agreed criterion priorities.')
    elif unavailable:
        result.update(message='Some required candidate checks are unavailable.',
            missing_or_invalid_inputs=[entry.get('reasons', []) for entry in unavailable])
    else:
        result.update(status='no_valid_grasp_within_budget', message='No valid grasp found within the sampling budget.')
    return result
