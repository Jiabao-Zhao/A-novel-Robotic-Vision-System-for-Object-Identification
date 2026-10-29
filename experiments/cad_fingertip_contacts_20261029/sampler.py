"""Fit first contacts on complete finger solids, without executing a closing.

The old fixed-budget CAD ray bank supplies only pair/angle proposals. Each jaw's
first contact is computed against the complete CAD by a monotone prefix swept
volume query. This is static geometry, not a simulated or physical pickup.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import (
    _solid_collider, component_vertices, pad_calibration, transform_points,
)
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from experiments.cad_grasp_refinement_20261029.sampler import generate_candidates as ray_candidates
from .contact_audit import make_contact_calibration

_FIT_CACHE = {}
FIT_COMMAND_TOLERANCE_M = 2e-10


def _jaw_contact(target, components, rotation, translation, maximum):
    """First collision while this jaw translates inward from its maximum gap.

    Endpoint overlap is not monotone for concave parts. The prefix sweep grows
    as opening decreases, and therefore supplies a valid bisection predicate.
    """
    starts = [transform_points(component_vertices(c, maximum), _pose(rotation, translation))
              for c in components]
    if any(target.intersects(v) for v in starts):
        return None, dict(status='maximum_opening_not_clear', iterations=0)

    def hits(command):
        pose = _pose(rotation, translation)
        return any(target.intersects(np.concatenate((start, transform_points(component_vertices(c, command), pose))))
                   for c, start in zip(components, starts))

    if not hits(0.):
        return None, dict(status='no_contact_within_opening_limits', iterations=0)
    colliding, clear, iterations = 0., float(maximum), 0
    while clear-colliding > FIT_COMMAND_TOLERANCE_M:
        midpoint = (colliding+clear)/2
        if hits(midpoint):
            colliding = midpoint
        else:
            clear = midpoint
        iterations += 1
    return colliding, dict(status='first_contact_bracketed', iterations=iterations,
        contact_command_lower_m=colliding, contact_command_upper_m=clear,
        command_bracket_width_m=clear-colliding,
        predicate='complete component prefix swept volume, not endpoint overlap')


def _pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose


def fit_candidate(candidate, target, hand, calibration, *, pre_opening_clearance_m=.001, contact_model=None):
    result = deepcopy(candidate)
    result['proposal_contacts_cad_m'] = deepcopy(candidate['contacts_cad_m'])
    result['proposal_contact_normals_cad'] = result.pop('contact_normals_cad', None)
    result['contact_mode'] = 'modeled_distal_inner_finger_surfaces'
    result['contact_calibration_sha256'] = calibration['calibration_sha256']
    base = pad_calibration(hand)
    tip = fingertip_calibration(hand)
    anchors = np.asarray(calibration['contact_anchors_zero_grip_site_m'], float)
    rotation = np.asarray(candidate['cad_T_grip_site'], float)[:3, :3]
    translation = np.asarray(candidate['contacts_cad_m'], float).mean(axis=0)-rotation@anchors.mean(axis=0)
    maximum = float(base['opening_limits_m'][1])
    fingers = [next(c for c in hand['components'] if c['name']==name)
               for name in calibration['finger_components']]
    pads = base['pads']
    commands, reports = [], []
    for finger, pad in zip(fingers, pads):
        command, report = _jaw_contact(target, [finger, pad], rotation, translation, maximum)
        commands.append(command)
        reports.append(report)
    result['first_contact_fit'] = dict(jaws=reports, geometry_only=True,
        collision_geometry_changed=False, physical_execution=False)
    if any(value is None for value in commands):
        result['first_contact_fit']['status'] = 'unavailable_contact'
        return result
    held = float(np.mean(commands))
    # The original jaw motions are [-0.5,0,0] and [+0.5,0,0] times
    # the command. A common command and one hand translation reproduce both
    # independent first-contact translations exactly.
    slopes = np.asarray(base['centers_per_opening'])[:, 0]
    offsets = slopes*(np.asarray(commands)-held)
    if not np.isclose(offsets[0], offsets[1], atol=1e-12, rtol=0):
        raise ValueError('Independent jaw fits require symmetric saved opening motion')
    translation += rotation@np.array([float(offsets.mean()), 0., 0.])
    grip_pose = _pose(rotation, translation)
    tip_pose = grip_pose@tip['grip_site_T_fingertip']
    actual, face_ids, local_points = [], [], []
    witnesses_by_jaw = []
    for finger in fingers:
        vertices = transform_points(component_vertices(finger, held), grip_pose)
        witnesses = target.surface_contacts(vertices)
        witnesses_by_jaw.append(len(witnesses))
        if not witnesses:
            result['first_contact_fit'].update(status='first_contact_not_on_distal_finger',
                finger_witness_counts=witnesses_by_jaw,
                note='The unchanged pad or another surface may have touched first; no fallback contact is accepted.')
            return result
        # Pick a representative point within one actual target triangle.
        # Averaging disconnected witnesses could put it inside a hole.
        def extent(w):
            points = np.asarray(w['polygon_world_m'])
            return float(np.ptp(points, axis=0).max())
        witness = max(witnesses, key=extent)
        point = np.asarray(witness['polygon_world_m'], float).mean(axis=0)
        actual.append(point)
        face_ids.append(int(witness['triangle_index']))
        local_points.append((point-translation)@rotation)
    result.update(cad_T_hand=tip_pose.tolist(), cad_T_grip_site=grip_pose.tolist(),
        contacts_cad_m=np.asarray(actual).tolist(), contact_face_ids=face_ids,
        actual_contacts_hand_m=np.asarray(local_points).tolist(),
        actual_contact_component_names=calibration['finger_components'],
        actual_contacts_hand_frame='original saved grip-site frame',
        holding_opening_m=held,
        pre_opening_m=min(maximum, held+2*float(pre_opening_clearance_m)/base['gap_per_opening']),
        reconstructed_contact_anchors_cad_m=np.asarray(actual).tolist(),
        contact_anchor_convention='actual complete-finger first-contact surface witnesses; not old pad anchors')
    if contact_model is not None:
        normals=np.asarray(contact_model.contact_normals(np.asarray(actual),np.asarray(face_ids)),float)
    else:
        triangles=target.faces[np.asarray(face_ids)]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
    result['contact_normals_cad']=normals.tolist()
    result['first_contact_fit'].update(status='fitted', finger_witness_counts=witnesses_by_jaw,
        actual_contacts_beyond_physical_tip_m=(np.asarray(local_points)[:, 2]-
            float(tip['fingertip_origin_grip_site_m'][2])).tolist())
    return result


def _bank_key(vertices, faces, candidates, calibration, clearance):
    digest=hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices,dtype='<f8').tobytes())
    digest.update(np.ascontiguousarray(faces,dtype='<i8').tobytes())
    digest.update(json.dumps(dict(candidates=candidates,
        calibration=calibration['calibration_sha256'],pre_opening_clearance_m=clearance),sort_keys=True).encode())
    return digest.hexdigest()


def generate_candidates(vertices, faces, hand, **options):
    started = time.perf_counter()
    normal_angle = float(options.get('normal_angle_rad', np.deg2rad(1.)))
    calibration = make_contact_calibration(hand, normal_angle_rad=normal_angle,
        edge_margin_m=float(options.get('contact_inset_m',.00025)))
    candidates, sampling = ray_candidates(vertices, faces, hand, **options)
    clearance=float(options.get('pre_opening_clearance_m',.001))
    digest = _bank_key(vertices, faces, candidates, calibration, clearance)
    cache_hit = digest in _FIT_CACHE
    if cache_hit:
        fitted = deepcopy(_FIT_CACHE[digest])
    else:
        target = _solid_collider(vertices, faces, 'complete target CAD for fingertip fit')
        fitted = [fit_candidate(candidate, target, hand, calibration,
            pre_opening_clearance_m=clearance,contact_model=options.get('contact_model')) for candidate in candidates]
        _FIT_CACHE[digest] = deepcopy(fitted)
    sampling['fingertip_contact_fit'] = dict(
        method='independent complete-jaw first contact; monotone prefix-sweep bisection',
        calibrated_anchor_source='real inward hull faces of the unchanged saved fingers',
        original_ray_bank_count=len(candidates), fitted_count=sum(
            candidate['first_contact_fit']['status']=='fitted' for candidate in fitted),
        cached_CAD_only_fits=cache_hit, cache_key=digest,
        cache_scope='same complete CAD/hand/calibration and generated proposal bank; independent of observed pickup pose',
        additional_surface_draws=0, additional_orientation_samples=0,
        physical_or_simulated_execution=False,
        total_sampling_and_fit_seconds=time.perf_counter()-started)
    return fitted, sampling
