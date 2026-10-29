"""Bounded side-aware contact rays with unchanged complete CAD geometry.

Only origin selection is clipped to the explicitly allowed contact surface.
Every ray still intersects the original complete mesh; its second endpoint must
also be allowed before the bounded pair/orientation bank is formed. Rays start
on the actual CAD surface, avoiding the historical fixed0.5mm interior jump.
"""
from __future__ import annotations

import hashlib
import numpy as np

from experiments.cad_grasp_selection_20261028 import sampler as legacy
from experiments.cad_grasp_selection_20261028.geometry import transform_points
from experiments.cad_grasp_selection_20261029.hand_frame import (
    DEFAULT_CONTACT_INSET_M, FINGERTIP_FRAME, fingertip_calibration)
from .contact_model import mesh_binding


def pose_from_pair(contacts_cad_m, approach_cad, hand,
                   contact_inset_m=DEFAULT_CONTACT_INSET_M):
    """Place near-tip anchors on both contacts and report the fingertip pose."""
    result = legacy.pose_from_pair(contacts_cad_m, approach_cad, hand)
    tip = fingertip_calibration(hand, contact_inset_m)
    pair = np.asarray(contacts_cad_m, float)
    command = result['holding_opening_m']
    anchors = (tip['contact_anchors_zero_grip_site_m']
               + command*tip['contact_anchors_per_opening'])
    grip_pose = np.asarray(result['cad_T_hand'])
    grip_pose[:3, 3] = pair.mean(axis=0)-grip_pose[:3, :3]@anchors.mean(axis=0)
    reconstructed = transform_points(anchors, grip_pose)
    if not np.allclose(reconstructed, pair, atol=1e-9, rtol=0):
        raise ValueError('Near-tip anchors do not reconstruct both CAD contacts')
    pose = grip_pose@tip['grip_site_T_fingertip']
    result.pop('reconstructed_pad_centers_cad_m')
    result.update(cad_T_hand=pose.tolist(), cad_T_grip_site=grip_pose.tolist(),
        hand_pose_reference=FINGERTIP_FRAME,
        reconstructed_contact_anchors_cad_m=reconstructed.tolist(),
        contact_anchor_convention='near distal usable pad edge; full pad/finger geometry retained',
        contact_inset_from_pad_tip_m=tip['contact_inset_m'],
        grip_site_T_fingertip=tip['grip_site_T_fingertip'].tolist())
    return result


def _origin_domain(vertices, faces, triangles, areas, region, mode):
    if mode not in ('assembly_region_guided', 'legacy_unrestricted'):
        raise ValueError('sampling_mode must be assembly_region_guided or explicit legacy_unrestricted')
    if mode == 'legacy_unrestricted':
        if region is not None:
            raise ValueError('legacy_unrestricted must not silently ignore a supplied sampling contact region')
        return triangles, np.arange(len(faces)), areas, None, dict(
            status='explicit_legacy_unrestricted', full_mesh_is_origin_domain=True)
    if not isinstance(region, dict):
        raise ValueError('assembly_region_guided sampling requires explicit compiled sampling_contact_region labels')
    patches = np.asarray(region.get('sampling_triangles_cad_m'), float)
    ids = np.asarray(region.get('source_face_ids'))
    patch_areas = np.asarray(region.get('triangle_areas_m2'), float)
    contains = region.get('endpoint_contains')
    if (patches.ndim != 3 or patches.shape[1:] != (3, 3) or not np.isfinite(patches).all()
            or ids.shape != (len(patches),) or ids.dtype.kind not in 'iu'
            or (len(ids) and (ids.min() < 0 or ids.max() >= len(faces)))
            or patch_areas.shape != (len(patches),) or not np.isfinite(patch_areas).all()
            or np.any(patch_areas <= 0) or not callable(contains)):
        raise ValueError('Compiled sampling contact region has invalid clipped patches, source face IDs, areas or endpoint query')
    measured = np.linalg.norm(np.cross(patches[:, 1]-patches[:, 0],
                                       patches[:, 2]-patches[:, 0]), axis=1)/2
    if not np.allclose(measured, patch_areas, atol=1e-20, rtol=1e-8):
        raise ValueError('Compiled sampling patch areas disagree with their triangles')
    total = float(region.get('eligible_area_m2', np.nan))
    if not np.isfinite(total) or not np.isclose(total, patch_areas.sum(), atol=1e-18, rtol=1e-8):
        raise ValueError('Compiled sampling region needs a consistent finite eligible area')
    # The compiler may provide a source binding. Never reuse a region compiled
    # on a different mesh after target preprocessing or face reordering.
    if (region.get('mesh_vertex_count') != len(vertices) or region.get('mesh_face_count') != len(faces)
            or region.get('mesh_vertices_sha256') != hashlib.sha256(
                np.ascontiguousarray(vertices, dtype='<f8').tobytes()).hexdigest()
            or region.get('mesh_faces_sha256') != hashlib.sha256(
                np.ascontiguousarray(faces, dtype='<i8').tobytes()).hexdigest()):
        raise ValueError('Sampling contact region belongs to a different or unbound complete CAD mesh')
    if total > float(areas.sum()) + 1e-12:
        raise ValueError('Eligible surface area exceeds the complete CAD surface')
    return patches, ids.astype(int, copy=False), patch_areas, contains, dict(region.get('provenance', {}))


def _endpoint_mask(contains, points, face_ids, label):
    mask = np.asarray(contains(points, face_ids))
    if mask.shape != (len(points),) or mask.dtype.kind != 'b':
        raise ValueError(f'Sampling region endpoint query must return one explicit boolean per {label}')
    return mask


def generate_candidates(vertices, faces, hand, *, seed=0, surface_budget=200,
                        orientation_samples=10, max_poses=100,
                        opposing_normal_cosine=.95, inward_offset_m=.0005,
                        pre_opening_clearance_m=.001,
                        contact_inset_m=DEFAULT_CONTACT_INSET_M,
                        sampling_contact_region=None,
                        sampling_mode='assembly_region_guided',
                        contact_model=None, normal_angle_rad=np.deg2rad(1.)):
    """Sample a fixed number of allowed origins, then at most 100 whole poses.

    There are no rejection-resampling loops, grouping or score ordering. Exact
    clipped patches provide an area-uniform origin domain, not a cropped object.
    Protected surfaces and bores remain in all-forward-hit ray intersections and
    downstream full-pad/table/receiver checks.
    """
    seed = legacy._integer(seed, 'seed', 0, 2**32-1)
    surface_budget = legacy._integer(surface_budget, 'surface_budget', 1)
    orientation_samples = legacy._integer(orientation_samples, 'orientation_samples', 1)
    max_poses = legacy._integer(max_poses, 'max_poses', 0, 100)
    if (not np.isfinite(normal_angle_rad) or not 0 <= normal_angle_rad < np.pi/2
            or not np.isfinite(opposing_normal_cosine) or not 0 <= opposing_normal_cosine <= 1
            or not np.isfinite(inward_offset_m) or inward_offset_m <= 0
            or not np.isfinite(pre_opening_clearance_m) or pre_opening_clearance_m < 0):
        raise ValueError('Invalid normal threshold, ray offset or pre-opening clearance')
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Expected finite CAD vertices and valid integer triangle faces')
    calibration = legacy.pad_calibration(hand)
    fingertip_calibration(hand, contact_inset_m)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(twice_area).all() or twice_area.sum() <= 0:
        raise ValueError('The CAD mesh has no positive finite surface area')
    normals = np.divide(cross, twice_area[:, None], out=np.zeros_like(cross),
                        where=twice_area[:, None] > 0)
    if contact_model is not None:
        if (not callable(getattr(contact_model, 'contact_normals', None))
                or any(getattr(contact_model, 'provenance', {}).get(k) != value
                       for k, value in mesh_binding(vertices, faces).items())):
            raise ValueError('Contact model belongs to a different or unbound complete CAD mesh')
    # Proposal and mandatory validation use the same angular tolerance. The old
    # .95 option is recorded for comparison; it cannot weaken this policy.
    effective_cosine = max(float(opposing_normal_cosine), float(np.cos(normal_angle_rad)))
    def contact_normals(points, ids):
        result = np.asarray(normals[ids] if contact_model is None else
                            contact_model.contact_normals(points, ids), float)
        if (result.shape != (len(points), 3) or not np.isfinite(result).all()
                or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-8, rtol=0)):
            raise ValueError('Contact model must return one finite unit normal per CAD contact')
        return result
    patches, source_ids, areas, contains, provenance = _origin_domain(
        vertices, faces, triangles, twice_area/2, sampling_contact_region, sampling_mode)
    metadata = dict(
        method='Fabrica-derived bounded region origins; verified side normals and on-surface complete-mesh rays',
        sampling_mode=sampling_mode, seed=seed, surface_sampling_budget=surface_budget,
        orientation_samples_per_pair=orientation_samples,
        orientation_range_rad=[0., float(2*np.pi)],
        orientation_phase='seeded uniform phase per retained pair',
        opposing_normal_cosine=effective_cosine,
        requested_legacy_opposing_normal_cosine=float(opposing_normal_cosine),
        normal_angle_rad=float(normal_angle_rad),
        inward_origin_offset_m=0.,
        unused_legacy_inward_origin_offset_m=float(inward_offset_m),
        ray_origin_policy='on original CAD surface; forward intersections beyond 1 nm self-hit threshold',
        contact_model_provenance=(dict(contact_model.provenance) if contact_model is not None
                                  else dict(status='original_triangle_normals')),
        cylindrical_origins_snapped_to_source_edges=0,
        pre_opening_total_gap_increase_m=float(2*pre_opening_clearance_m),
        coincident_hit_tolerance_m=legacy.RAY_CONFIG['coincident_hit_tolerance_m'],
        ray_batch_size=legacy.RAY_CONFIG['ray_batch_size'],
        eligible_surface_area_m2=float(areas.sum()),
        full_CAD_surface_area_m2=float(twice_area.sum()/2),
        eligible_surface_fraction=float(areas.sum()/(twice_area.sum()/2)),
        clipped_patch_count=len(patches), clipped_patch_source_face_ids=source_ids.tolist(),
        sampling_region_provenance=provenance,
        ray_intersection_geometry='complete original CAD triangles; no surface crop',
        full_CAD_face_count=len(faces), surface_origins_drawn=0,
        surface_origins_sampled=0, surface_origins_region_rejected=0,
        surface_origins_cad_m=[], inward_ray_origins_cad_m=[],
        surface_origin_patch_indices=[], surface_origin_source_face_ids=[],
        unique_ray_hits=0, second_endpoint_region_rejections=0, pairs_rejected_by_region=0,
        region_accepted_pair_count_before_normals=0, normal_rejected_pair_count=0,
        normal_accepted_pair_count=0, proposal_pool_size=0, proposed_count=0,
        max_complete_poses=max_poses, coincident_triangle_hits_removed=0,
        region_filter_before_pair_orientation_bank=True, rejection_resampling=False,
        pair_reduction=False, centroid_sort=False, independent_depth_sampling=False,
        downward_gate=False, units='metres/radians', contacts_frame='native_CAD',
        hand_pose_reference=FINGERTIP_FRAME,
        contact_convention='near-tip anchors on the two complete usable pads',
        contact_inset_from_pad_tip_m=float(contact_inset_m))
    if not len(patches):
        metadata['status'] = 'empty_allowed_contact_surface'
        return [], metadata
    rng = np.random.RandomState(seed)
    sampled, patch_ids = legacy._surface_samples(patches, areas, surface_budget, rng)
    first_faces = source_ids[patch_ids]
    metadata['surface_origins_drawn'] = len(sampled)
    if contains is not None:
        keep = _endpoint_mask(contains, sampled, first_faces, 'origin contact')
        metadata['surface_origins_region_rejected'] = int((~keep).sum())
        sampled, first_faces, patch_ids = sampled[keep], first_faces[keep], patch_ids[keep]
    metadata['area_draw_origins_cad_m'] = sampled.tolist()
    if contact_model is not None:
        sampled, snapped = contact_model.snap_origins_to_axial_edges(sampled, first_faces)
        if contains is not None:
            keep = _endpoint_mask(contains, sampled, first_faces, 'constructed origin contact')
            metadata['surface_origins_region_rejected'] += int((~keep).sum())
            sampled, first_faces, patch_ids, snapped = sampled[keep], first_faces[keep], patch_ids[keep], snapped[keep]
        metadata['cylindrical_origins_snapped_to_source_edges'] = int(snapped.sum())
    metadata.update(surface_origins_sampled=len(sampled), surface_origins_cad_m=sampled.tolist(),
                    surface_origin_patch_indices=patch_ids.tolist(),
                    surface_origin_source_face_ids=first_faces.tolist())
    if not len(sampled):
        metadata['status'] = 'no_origins_inside_allowed_contact_surface'
        return [], metadata
    first_normals = contact_normals(sampled, first_faces)
    origins = sampled.copy()
    metadata['inward_ray_origins_cad_m'] = origins.tolist()
    hits, ray_ids, face2, duplicates = legacy._all_ray_hits(origins, -first_normals, triangles)
    metadata.update(unique_ray_hits=len(hits), coincident_triangle_hits_removed=int(duplicates))
    if contains is not None:
        keep = _endpoint_mask(contains, hits, face2, 'ray hit')
        metadata['second_endpoint_region_rejections'] = int((~keep).sum())
        metadata['pairs_rejected_by_region'] = int((~keep).sum())
        hits, ray_ids, face2 = hits[keep], ray_ids[keep], face2[keep]
    metadata['region_accepted_pair_count_before_normals'] = len(hits)
    pairs = np.stack((sampled[ray_ids], hits), axis=1)
    spans = np.linalg.norm(pairs[:, 1]-pairs[:, 0], axis=1)
    axes = np.divide(pairs[:, 1]-pairs[:, 0], spans[:, None],
                     out=np.zeros_like(pairs[:, 1]), where=spans[:, None] > 1e-12)
    align1 = np.einsum('ij,ij->i', first_normals[ray_ids], -axes)
    second_normals = contact_normals(hits, face2)
    align2 = np.einsum('ij,ij->i', second_normals, axes)
    accepted = ((spans > legacy.RAY_CONFIG['coincident_hit_tolerance_m'])
                & (align1 >= effective_cosine) & (align2 >= effective_cosine))
    metadata['normal_rejected_pair_count'] = int((~accepted).sum())
    pairs, axes = pairs[accepted], axes[accepted]
    pair_faces = np.column_stack((first_faces[ray_ids], face2))[accepted]
    pair_normals = np.stack((first_normals[ray_ids], second_normals), axis=1)[accepted]
    phases = rng.uniform(0., 2*np.pi, len(pairs))
    pool_size = len(pairs)*orientation_samples
    chosen = rng.choice(pool_size, min(max_poses, pool_size), replace=False)
    candidates = []
    for candidate_id, pool_index in enumerate(chosen):
        pair_id, angle_id = divmod(int(pool_index), orientation_samples)
        axis = axes[pair_id]
        reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
        basis = np.cross(axis, reference)
        basis /= np.linalg.norm(basis)
        angle = float(phases[pair_id]+2*np.pi*angle_id/orientation_samples)
        approach = np.cos(angle)*basis+np.sin(angle)*np.cross(axis, basis)
        candidate = pose_from_pair(pairs[pair_id], approach, hand, contact_inset_m)
        candidate.update(candidate_id=candidate_id, pair_id=pair_id,
            proposal_pool_index=int(pool_index), orientation_index=angle_id,
            orientation_angle_rad=angle, contact_face_ids=pair_faces[pair_id].tolist(),
            contact_normals_cad=pair_normals[pair_id].tolist(),
            pre_opening_m=candidate['holding_opening_m']+2*pre_opening_clearance_m/calibration['gap_per_opening'])
        candidates.append(candidate)
    metadata.update(normal_accepted_pair_count=len(pairs), proposal_pool_size=pool_size,
                    proposed_count=len(candidates), status='bounded_surface_budget_completed')
    return candidates, metadata
