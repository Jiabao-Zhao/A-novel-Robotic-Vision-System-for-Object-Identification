"""Bounded Fabrica-derived ray proposals with calibrated pad-center placement.

Reuse the earlier area sampler and all-forward-hit ray query, but not its
centroid ordering, pair thinning, downward conversion or finger-depth variants.
All lengths are metres in the supplied CAD frame. No collision check, dynamics,
simulator state, learned model or selection score is imported here.
"""
from __future__ import annotations

import numpy as np

from experiments.fabrica_candidates_20261027.sampler import (
    CONFIG as RAY_CONFIG, _surface_samples, _all_ray_hits,
)
from .geometry import pad_calibration


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f'{name} must be an integer in [{low}, {high or "unbounded"}]')
    return int(value)


def pose_from_pair(contacts_cad_m, approach_cad, hand):
    """Place BOTH usable pad centers on the pair, without a depth parameter."""
    calibration = pad_calibration(hand)
    pair = np.asarray(contacts_cad_m, dtype=float)
    approach = np.asarray(approach_cad, dtype=float)
    if pair.shape != (2, 3) or not np.isfinite(pair).all():
        raise ValueError('contacts_cad_m must be finite (2,3) coordinates')
    span = float(np.linalg.norm(pair[1] - pair[0]))
    if span <= 1e-12:
        raise ValueError('Contact pair has zero or numerically unresolved span')
    closing = (pair[1] - pair[0]) / span
    if (approach.shape != (3,) or not np.isfinite(approach).all()
            or abs(np.linalg.norm(approach) - 1.) > 1e-8
            or abs(approach @ closing) > 1e-8):
        raise ValueError('approach_cad must be a unit vector perpendicular to the contact line')
    approach = approach - closing * (approach @ closing)
    approach /= np.linalg.norm(approach)
    command = ((span - calibration['gap_zero_m']) /
               calibration['gap_per_opening'])
    local_closing = np.asarray(calibration['closing_hand'])
    local_approach = np.asarray(calibration['approach_hand'])
    local_basis = np.column_stack((local_closing,
                                  np.cross(local_approach, local_closing),
                                  local_approach))
    cad_basis = np.column_stack((closing, np.cross(approach, closing), approach))
    rotation = cad_basis @ local_basis.T
    midpoint_hand = (np.asarray(calibration['midpoint_zero_m']) +
                    command * np.asarray(calibration['midpoint_per_opening']))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = pair.mean(axis=0) - rotation @ midpoint_hand
    centers_hand = (np.asarray(calibration['centers_zero_m']) +
                    command * np.asarray(calibration['centers_per_opening']))
    reconstructed = centers_hand @ rotation.T + pose[:3, 3]
    if not np.allclose(reconstructed, pair, rtol=0., atol=1e-9):
        raise ValueError('Calibrated pad centers do not reconstruct the intended contact pair')
    return dict(cad_T_hand=pose.tolist(), contacts_cad_m=pair.tolist(),
                reconstructed_pad_centers_cad_m=reconstructed.tolist(),
                contact_span_m=span, holding_opening_m=float(command),
                approach_cad=approach.tolist())


def generate_candidates(vertices, faces, hand, *, seed=0, surface_budget=200,
                        orientation_samples=10, max_poses=100,
                        opposing_normal_cosine=.95, inward_offset_m=.0005,
                        pre_opening_clearance_m=.001):
    """Randomly choose at most 100 pair/orientation combinations BEFORE checks.

    There is no pair grouping or centroid sort. Every forward exit satisfying
    the normal test remains eligible, including outer grasps spanning a hole.
    Angles cover the full circle; no upward/downward half-space is removed.
    One fixed surface budget bounds rays; no hidden resampling seeks a valid
    pickup. A finite proposal bank may miss feasible grasps.
    """
    seed = _integer(seed, 'seed', 0, 2**32 - 1)
    surface_budget = _integer(surface_budget, 'surface_budget', 1)
    orientation_samples = _integer(orientation_samples, 'orientation_samples', 1)
    max_poses = _integer(max_poses, 'max_poses', 0, 100)
    if (not np.isfinite(opposing_normal_cosine) or not 0 <= opposing_normal_cosine <= 1
            or not np.isfinite(inward_offset_m) or inward_offset_m <= 0
            or not np.isfinite(pre_opening_clearance_m) or pre_opening_clearance_m < 0):
        raise ValueError('Invalid normal threshold, ray offset or pre-opening clearance')
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Expected finite CAD vertices and valid integer triangle faces')
    calibration = pad_calibration(hand)  # Fail explicitly if calibration is absent.
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(twice_area).all() or twice_area.sum() <= 0:
        raise ValueError('The CAD mesh has no positive finite surface area')
    normals = np.divide(cross, twice_area[:, None], out=np.zeros_like(cross),
                        where=twice_area[:, None] > 0)
    rng = np.random.RandomState(seed)
    sampled, face1 = _surface_samples(triangles, twice_area / 2., surface_budget, rng)
    first_normals = normals[face1]
    origins = sampled - inward_offset_m * first_normals
    hits, ray_ids, face2, duplicate_hits = _all_ray_hits(origins, -first_normals, triangles)
    pairs = np.stack((sampled[ray_ids], hits), axis=1)
    spans = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    axes = np.divide(pairs[:, 1] - pairs[:, 0], spans[:, None],
                     out=np.zeros_like(pairs[:, 1]), where=spans[:, None] > 1e-12)
    align1 = np.einsum('ij,ij->i', first_normals[ray_ids], -axes)
    align2 = np.einsum('ij,ij->i', normals[face2], axes)
    accepted = ((spans > 1e-12) & (align1 >= opposing_normal_cosine)
                & (align2 >= opposing_normal_cosine))
    pairs, axes = pairs[accepted], axes[accepted]
    pair_faces = np.column_stack((face1[ray_ids], face2))[accepted]
    pair_normals = np.stack((first_normals[ray_ids], normals[face2]), axis=1)[accepted]
    # Randomize each pair's roll phase, then sample combinations uniformly.
    phases = rng.uniform(0., 2 * np.pi, len(pairs))
    pool_size = len(pairs) * orientation_samples
    chosen = rng.choice(pool_size, min(max_poses, pool_size), replace=False)
    candidates = []
    for candidate_id, pool_index in enumerate(chosen):
        pair_id, angle_id = divmod(int(pool_index), orientation_samples)
        axis = axes[pair_id]
        reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
        basis = np.cross(axis, reference)
        basis /= np.linalg.norm(basis)
        angle = float(phases[pair_id] + 2 * np.pi * angle_id / orientation_samples)
        approach = np.cos(angle) * basis + np.sin(angle) * np.cross(axis, basis)
        candidate = pose_from_pair(pairs[pair_id], approach, hand)
        candidate.update(candidate_id=candidate_id, pair_id=pair_id,
                         proposal_pool_index=int(pool_index), orientation_index=angle_id,
                         orientation_angle_rad=angle,
                         contact_face_ids=pair_faces[pair_id].tolist(),
                         contact_normals_cad=pair_normals[pair_id].tolist(),
                         pre_opening_m=candidate['holding_opening_m'] +
                                       2 * pre_opening_clearance_m /
                                       calibration['gap_per_opening'])
        candidates.append(candidate)
    return candidates, dict(
        method='Fabrica-derived area sampling and inward all-hit normal rays',
        seed=seed, surface_sampling_budget=surface_budget,
        orientation_samples_per_pair=orientation_samples,
        orientation_range_rad=[0., float(2 * np.pi)],
        orientation_phase='seeded uniform phase per pair',
        opposing_normal_cosine=float(opposing_normal_cosine),
        inward_origin_offset_m=float(inward_offset_m),
        pre_opening_total_gap_increase_m=float(2 * pre_opening_clearance_m),
        coincident_hit_tolerance_m=RAY_CONFIG['coincident_hit_tolerance_m'],
        ray_batch_size=RAY_CONFIG['ray_batch_size'],
        normal_accepted_pair_count=len(pairs), proposal_pool_size=pool_size,
        proposed_count=len(candidates), max_complete_poses=max_poses,
        coincident_triangle_hits_removed=int(duplicate_hits),
        pair_reduction=False, centroid_sort=False, independent_depth_sampling=False,
        downward_gate=False, units='metres/radians', contacts_frame='native_CAD',
        contact_convention='centers of the two usable inner pads')
