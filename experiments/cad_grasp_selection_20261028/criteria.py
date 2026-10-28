"""Fixed-CAD geometric criteria and sequential grasp selection.

No learned weights, combined score, grouping, collision decisions or simulator
state are used here. All contact coordinates and reference lengths are metres
in the unchanged CAD frame. Approach preference is evaluated at pickup in the
workspace frame. These criteria do not establish physical grasp stability.
"""
from __future__ import annotations

import numpy as np


DEFAULT_TIE_TOLERANCES = dict(r_tau=1e-10, r_surface=1e-10,
                              theta_down_rad=1e-10)


def _points(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite (N,3) coordinates')
    return result


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite three-vector')
    return result


def _unit(value, name, tolerance=1e-8):
    result = _vector(value, name)
    length = float(np.linalg.norm(result))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Unit-vector tolerance must be finite and nonnegative')
    if length == 0 or abs(length - 1.) > tolerance:
        raise ValueError(f'{name} must be unit length within {tolerance}')
    # Remove only the explicitly tolerated floating-point norm error.
    return result / length


def build_reference(vertices, faces, axis_point_cad, axis_direction_cad,
                    unit_tolerance=1e-8):
    """Compute the area-weighted surface centroid and fixed vertex bounds once.

    This uses triangle areas, never the raw vertex average or a volume centroid.
    Degenerate triangles contribute zero area; an entirely degenerate mesh is
    unavailable. Unreferenced supplied vertices still participate in the bounds,
    matching the declared definition over all CAD mesh vertices.
    """
    vertices = _points(vertices, 'vertices')
    faces = np.asarray(faces)
    if (not len(vertices) or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('faces must be nonempty valid integer triangle indices')
    axis_point = _vector(axis_point_cad, 'axis_point_cad')
    direction = _unit(axis_direction_cad, 'axis_direction_cad', unit_tolerance)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    areas = .5 * np.linalg.norm(cross, axis=1)
    if not np.isfinite(areas).all() or not np.isfinite(areas.sum()) or areas.sum() <= 0:
        raise ValueError('CAD surface centroid requires positive finite total area')
    centroid = np.average(triangles.mean(axis=1), axis=0, weights=areas)
    D_tau = float(np.linalg.norm(np.cross(vertices - axis_point, direction), axis=1).max())
    D_surface = float(np.linalg.norm(vertices - centroid, axis=1).max())
    if not np.isfinite([*centroid, D_tau, D_surface]).all():
        raise ValueError('CAD reference computation overflowed')
    return dict(surface_centroid_cad_m=centroid.tolist(),
                axis_point_cad_m=axis_point.tolist(),
                axis_direction_cad=direction.tolist(),
                D_tau_m=D_tau, D_surface_m=D_surface,
                surface_area_m2=float(areas.sum()),
                zero_tau_criterion=D_tau == 0.,
                zero_surface_criterion=D_surface == 0.,
                frame='native_CAD', length_units='metres')


def _normalize(raw, denominator, name, roundoff_tolerance,
               zero_distance_tolerance_m):
    if not np.isfinite(denominator) or denominator < 0:
        raise ValueError(f'{name} denominator must be finite and nonnegative')
    if denominator == 0.:
        if raw > zero_distance_tolerance_m:
            raise ValueError(f'{name} has a zero CAD bound but a nonzero candidate value')
        return 0., True
    value = raw / denominator
    if not np.isfinite(value) or value < -roundoff_tolerance or value > 1. + roundoff_tolerance:
        raise ValueError(f'{name} exceeds its fixed CAD bound; check frames and contacts')
    return float(np.clip(value, 0., 1.)), False


def evaluate_criteria(contacts_cad, reference, approach_world, table_up_world,
                      roundoff_tolerance=1e-10,
                      zero_distance_tolerance_m=1e-12):
    """Evaluate one intended pad-center pair without changing its midpoint.

    A zero fixed denominator contributes zero for every consistent candidate and
    is marked inactive. It is never replaced with a candidate-dependent scale.
    Values outside [0,1] beyond roundoff are errors, not silently clipped scores.
    Contact-on-CAD and pad-center reconstruction are the validator's obligations.
    """
    for value in (roundoff_tolerance, zero_distance_tolerance_m):
        if not np.isfinite(value) or value < 0:
            raise ValueError('Numerical tolerances must be finite and nonnegative')
    contacts = _points(contacts_cad, 'contacts_cad')
    if contacts.shape != (2, 3):
        raise ValueError('contacts_cad must contain exactly two intended pad centers')
    midpoint = contacts.mean(axis=0)
    axis_point = _vector(reference['axis_point_cad_m'], 'axis_point_cad_m')
    direction = _unit(reference['axis_direction_cad'], 'axis_direction_cad')
    centroid = _vector(reference['surface_centroid_cad_m'], 'surface_centroid_cad_m')
    q_tau = float(np.linalg.norm(np.cross(midpoint - axis_point, direction)))
    q_surface = float(np.linalg.norm(midpoint - centroid))
    r_tau, zero_tau = _normalize(q_tau, float(reference['D_tau_m']), 'assembly moment arm',
                                roundoff_tolerance, zero_distance_tolerance_m)
    r_surface, zero_surface = _normalize(q_surface, float(reference['D_surface_m']),
        'surface centrality', roundoff_tolerance, zero_distance_tolerance_m)
    approach = _unit(approach_world, 'approach_world')
    table_up = _unit(table_up_world, 'table_up_world')
    theta = float(np.arccos(np.clip(np.dot(approach, -table_up), -1., 1.)))
    return dict(midpoint_cad_m=midpoint.tolist(), q_tau_m=q_tau,
                q_surface_m=q_surface, r_tau=r_tau, r_surface=r_surface,
                theta_down_rad=theta, zero_tau_criterion=zero_tau,
                zero_surface_criterion=zero_surface)


def select_priority(candidates, tie_tolerances=None):
    """Select valid candidates by stage minima, then deterministic integer ID.

    Callers supply geometrically valid candidates only. This function never
    substitutes an invalid candidate for an empty set. Tolerances are numerical
    equality tolerances, not weights or physical task-preference thresholds.
    Each stage compares every remaining value to that stage's minimum, avoiding
    nontransitive tolerance-based pairwise comparisons.
    """
    tolerances = DEFAULT_TIE_TOLERANCES.copy()
    if tie_tolerances is not None:
        unknown = set(tie_tolerances) - set(tolerances)
        if unknown:
            raise ValueError(f'Unknown tie criteria: {sorted(unknown)}')
        tolerances.update(tie_tolerances)
    for value in tolerances.values():
        if not np.isfinite(value) or value < 0:
            raise ValueError('Tie tolerances must be finite and nonnegative')
    candidates = list(candidates)
    ids = [candidate['candidate_id'] for candidate in candidates]
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('candidate_id must be a unique integer for deterministic ties')
    for candidate in candidates:
        if candidate.get('valid') is False:
            raise ValueError('Priority selection accepts geometrically valid candidates only')
        for criterion in tolerances:
            value = candidate['criteria'][criterion]
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{criterion} must be a finite nonnegative scalar')
        if (candidate['criteria']['r_tau'] > 1. + 1e-10
                or candidate['criteria']['r_surface'] > 1. + 1e-10
                or candidate['criteria']['theta_down_rad'] > np.pi + 1e-10):
            raise ValueError('Candidate criteria are outside their declared ranges')
    if not candidates:
        return dict(status='no_valid_grasp', selected_candidate=None, stages=[],
                    final_tied_candidate_ids=[], tie_tolerances=tolerances,
                    message='No valid grasp found within the sampling budget.')
    remaining = candidates
    stages = []
    for criterion, tolerance in tolerances.items():
        minimum = min(float(candidate['criteria'][criterion]) for candidate in remaining)
        remaining = [candidate for candidate in remaining
                     if float(candidate['criteria'][criterion]) - minimum <= tolerance]
        stages.append(dict(criterion=criterion, minimum=minimum,
                           numerical_tie_tolerance=float(tolerance),
                           remaining_candidate_ids=sorted(int(c['candidate_id']) for c in remaining)))
    selected = min(remaining, key=lambda candidate: candidate['candidate_id'])
    return dict(status='selected', selected_candidate=selected, stages=stages,
                final_tied_candidate_ids=sorted(int(c['candidate_id']) for c in remaining),
                tie_tolerances=tolerances)
