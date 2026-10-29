"""Compile explicit task contact regions into area-weighted CAD sampling patches.

Only sampling triangles are clipped. The source CAD vertices/triangles remain
unchanged for all rays, contact checks and collisions. Boolean regions are
partitioned into disjoint patches so overlapping whitelists are not oversampled.
No grasp, semantic label, pose estimation or dynamics is generated here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np

BOUNDARY_MARGIN_M = 1e-8  # Existing protected-contact guard, not a new fit margin.
MIN_PATCH_AREA_M2 = 1e-24


def mesh_binding(vertices, faces):
    """Bind a compiled region to exact normalized metric arrays and face order."""
    vertices = np.ascontiguousarray(vertices, dtype='<f8')
    faces = np.ascontiguousarray(faces, dtype='<i8')
    return dict(mesh_vertex_count=len(vertices), mesh_face_count=len(faces),
        mesh_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
        mesh_faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest())


def _normalize(region, face_count, margin, *, protected):
    if not isinstance(region, dict):
        raise ValueError('An explicit task contact-region declaration is required')
    kind = region.get('type')
    if kind == 'halfspace':
        normal = np.asarray(region.get('normal_cad'), dtype=float)
        offset = float(region.get('offset_m', np.nan))
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or abs(np.linalg.norm(normal)-1.) > 1e-8 or not np.isfinite(offset)):
            raise ValueError('Sampling halfspace needs a finite unit CAD normal and metre offset')
        if region.get('inequality', 'normal@point+offset>=0') != 'normal@point+offset>=0':
            raise ValueError('Unsupported sampling halfspace inequality')
        # Protected contact is forbidden at/near the boundary. Positive labels
        # include their CAD boundary (e.g. an actual housing wall), with the
        # same numerical guard: n.p+offset >= -margin. Do not erode away the wall.
        # Units remain metres because n is a unit vector.
        return dict(type=kind, normal=normal,
                    offset=offset+margin)
    if kind == 'face_ids':
        values = region.get('protected_face_ids', region.get('face_ids'))
        if values is None:
            raise ValueError('Sampling face region needs explicit face IDs')
        ids = np.asarray(values)
        if (ids.ndim != 1 or (len(ids) and ids.dtype.kind not in 'iu')
                or (len(ids) and (ids.min() < 0 or ids.max() >= face_count))):
            raise ValueError('Sampling face IDs must be valid integers in the original CAD face order')
        return dict(type=kind, ids=np.unique(ids.astype(np.int64)))
    if kind in ('union', 'intersection'):
        children = region.get('regions')
        if not isinstance(children, list) or not children:
            raise ValueError('Boolean sampling regions need explicit nonempty component regions')
        return dict(type=kind, regions=[_normalize(child, face_count, margin, protected=protected)
                                        for child in children])
    raise ValueError(f'Unsupported task contact-region type: {kind}')


def _empty():
    return np.empty((0, 3, 3), dtype=float), np.empty(0, dtype=np.int64)


def _combine(pieces):
    pieces = [(triangles, ids) for triangles, ids in pieces if len(ids)]
    if not pieces:
        return _empty()
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate([x[0] for x in pieces]), np.concatenate([x[1] for x in pieces])


def _clip_polygon(polygon, normal, offset, positive):
    result = []
    old = polygon[-1]
    old_value = float(old@normal+offset)
    old_inside = old_value >= 0 if positive else old_value <= 0
    for point in polygon:
        value = float(point@normal+offset)
        inside = value >= 0 if positive else value <= 0
        if inside != old_inside:
            fraction = old_value/(old_value-value)
            result.append(old+fraction*(point-old))
        if inside:
            result.append(point)
        old, old_value, old_inside = point, value, inside
    if len(result) < 3:
        return []
    # Remove only numerically identical consecutive witnesses, not small features.
    unique = []
    for point in result:
        if not unique or not np.array_equal(point, unique[-1]):
            unique.append(point)
    if len(unique) > 1 and np.array_equal(unique[0], unique[-1]):
        unique.pop()
    return unique


def _triangulate(polygon, source_face):
    if len(polygon) < 3:
        return _empty()
    triangles = np.asarray([[polygon[0], polygon[i], polygon[i+1]]
                             for i in range(1, len(polygon)-1)], dtype=float)
    area = .5*np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                     triangles[:, 2]-triangles[:, 0]), axis=1)
    triangles = triangles[area > MIN_PATCH_AREA_M2]
    return triangles, np.full(len(triangles), source_face, dtype=np.int64)


def _partition(triangles, source_ids, region):
    """Return disjoint inside/outside triangulations, carrying original face IDs."""
    if not len(source_ids):
        return _empty(), _empty()
    kind = region['type']
    if kind == 'face_ids':
        inside = np.isin(source_ids, region['ids'])
        return (triangles[inside], source_ids[inside]), (triangles[~inside], source_ids[~inside])
    if kind == 'halfspace':
        values = triangles@region['normal']+region['offset']
        inside = (values >= 0).all(axis=1)
        # Coplanar triangles belong to the closed declared region, once only.
        outside = (values <= 0).all(axis=1) & ~inside
        crossing = np.flatnonzero(~inside & ~outside)
        positive = [(triangles[inside], source_ids[inside])]
        negative = [(triangles[outside], source_ids[outside])]
        for index in crossing:
            polygon = triangles[index]
            positive.append(_triangulate(_clip_polygon(polygon, region['normal'], region['offset'], True), source_ids[index]))
            negative.append(_triangulate(_clip_polygon(polygon, region['normal'], region['offset'], False), source_ids[index]))
        return _combine(positive), _combine(negative)
    if kind == 'union':
        remaining = triangles, source_ids
        accepted = []
        for child in region['regions']:
            yes, remaining = _partition(*remaining, child)
            accepted.append(yes)
        return _combine(accepted), remaining
    # Complementary first-failure partition keeps overlapping Boolean clauses
    # disjoint. This avoids double-weighting patches in a positive region union.
    remaining = triangles, source_ids
    rejected = []
    for child in region['regions']:
        remaining, no = _partition(*remaining, child)
        rejected.append(no)
    return remaining, _combine(rejected)


def _contains(points, source_ids, region):
    kind = region['type']
    if kind == 'face_ids':
        return np.isin(source_ids, region['ids'])
    if kind == 'halfspace':
        return points@region['normal']+region['offset'] >= 0
    values = [_contains(points, source_ids, child) for child in region['regions']]
    operation = np.logical_or if kind == 'union' else np.logical_and
    return operation.reduce(values)


def compile_sampling_region(vertices, faces, protected_region, sampling_region=None,
                            *, boundary_margin_m=BOUNDARY_MARGIN_M):
    """Compile allowed contact surface patches before any CAD area sampling.

    Allowed = optional explicit positive region minus explicit protected region.
    Regions use original CAD face IDs and/or ``normal_cad @ point + offset_m >= 0``.
    Their Boolean unions/intersections are supported without overlapping patch
    weights. Protected boundaries are forbidden; positive boundaries are closed
    and retained. The default 10 nm guard matches protected-contact checks.
    Positive labels are caller-supplied reviewed geometry, not semantic inference.
    """
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Sampling compilation requires finite CAD vertices and valid integer faces')
    margin = float(boundary_margin_m)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Sampling boundary margin must be finite and nonnegative')
    protected = _normalize(protected_region, len(faces), margin, protected=True)
    positive = (_normalize(sampling_region, len(faces), margin, protected=False)
                if sampling_region is not None else None)
    triangles = vertices[faces]
    original_areas = .5*np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                               triangles[:, 2]-triangles[:, 0]), axis=1)
    if not np.isfinite(original_areas).all() or np.any(original_areas <= 0):
        raise ValueError('Sampling CAD has degenerate or invalid triangles')
    _, allowed = _partition(triangles, np.arange(len(faces), dtype=np.int64), protected)
    if positive is not None:
        allowed, _ = _partition(*allowed, positive)
    patches, source_ids = allowed
    areas = .5*np.linalg.norm(np.cross(patches[:, 1]-patches[:, 0], patches[:, 2]-patches[:, 0]), axis=1)
    original_area, eligible_area = float(original_areas.sum()), float(areas.sum())
    binding = mesh_binding(vertices, faces)
    source_face_tolerance = max(margin, 1e-10)

    def endpoint_contains(points, face_ids):
        points, ids = np.asarray(points, dtype=float), np.asarray(face_ids)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or ids.shape != (len(points),) or (len(ids) and ids.dtype.kind not in 'iu')
                or (len(ids) and (ids.min() < 0 or ids.max() >= len(faces)))):
            raise ValueError('Endpoint eligibility needs finite CAD points and original integer face IDs')
        ids = ids.astype(np.int64, copy=False)
        eligible = ~_contains(points, ids, protected)
        if positive is not None:
            eligible &= _contains(points, ids, positive)
        source_triangles = triangles[ids]
        ab = source_triangles[:, 1]-source_triangles[:, 0]
        ac = source_triangles[:, 2]-source_triangles[:, 0]
        ap = points-source_triangles[:, 0]
        normals = np.cross(ab, ac)
        squared = np.einsum('ij,ij->i', normals, normals)
        distances = np.abs(np.einsum('ij,ij->i', ap, normals))/np.sqrt(squared)
        u = np.einsum('ij,ij->i', np.cross(ap, ac), normals)/squared
        v = np.einsum('ij,ij->i', np.cross(ab, ap), normals)/squared
        eligible &= ((distances <= source_face_tolerance) & (u >= -1e-8)
                     & (v >= -1e-8) & (u+v <= 1+1e-8))
        return eligible

    provenance = dict(
        method='exact task-region triangle clipping and disjoint Boolean partition',
        protected_region=deepcopy(protected_region), positive_sampling_region=deepcopy(sampling_region),
        positive_region_provided=sampling_region is not None,
        boundary_margin_m=margin, min_sampling_patch_area_m2=MIN_PATCH_AREA_M2,
        endpoint_source_face_tolerance_m=source_face_tolerance,
        original_surface_area_m2=original_area, eligible_area_m2=eligible_area,
        excluded_surface_area_m2=max(0., original_area-eligible_area),
        eligible_area_fraction=eligible_area/original_area,
        eligible_source_face_count=int(len(np.unique(source_ids))),
        sampling_patch_triangle_count=len(patches),
        original_collision_and_ray_mesh_unchanged=True,
        semantic_safety_claim='Only explicitly declared regions are used; nonprotected geometry is not automatically certified safe gripping material.',
        **binding)
    return dict(status='available', sampling_triangles_cad_m=patches,
        source_face_ids=source_ids, triangle_areas_m2=areas,
        eligible_area_m2=eligible_area, endpoint_contains=endpoint_contains,
        provenance=provenance, **binding)
