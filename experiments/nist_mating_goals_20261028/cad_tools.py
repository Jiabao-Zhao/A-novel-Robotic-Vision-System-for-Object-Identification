"""Small, read-only mesh measurements for the NIST mating-goal review.

All section lengths follow the supplied vertices' units (metres in this study).
Sections are triangle/plane intersections, not silhouettes or convex hulls.
Coplanar triangles are ignored explicitly. A few sampled sections are not a
certificate that two three-dimensional solids are disjoint.

Area construction assumes each edge-connected mesh component describes one
solid, including any connected cavity walls. Nested section loops within each
component use even/odd fill; separate components are unioned, matching the
existing geometric validator. This does not interpret a disconnected, inward
oriented shell as a subtractive cavity. Request the report to detect unclosed
section linework before making a fit claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiLineString
from shapely.ops import polygonize_full, unary_union

from experiments.runtime_antipodal_assembly_20261023.extract_goal_audit import native_mesh


def read_native_mesh(path):
    """Read binary STL in its existing coordinate frame; no units conversion."""
    return native_mesh(Path(path))


def transform_points(points, transform):
    """Apply a proper rigid 4x4 transform without mutating either argument."""
    points, transform = np.asarray(points, float), np.asarray(transform, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('points must be finite (N,3) coordinates')
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3],
                               np.eye(3), rtol=0, atol=1e-8)
            or abs(np.linalg.det(transform[:3, :3])-1) > 1e-8):
        raise ValueError('transform must be a finite proper rigid transform')
    return points @ transform[:3, :3].T + transform[:3, 3]


def _checked_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError('Expected finite vertices and valid integer triangle faces')
    return vertices, faces


def mesh_component_labels(faces):
    """Return one label per triangle, joining triangles sharing an exact edge."""
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('faces must be integer triangles')
    if not len(faces):
        return np.empty(0, int)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]])), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    rows, cols = owners[:-1][shared], owners[1:][shared]
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                       shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)[1]


def plane_segments(vertices, faces, normal, offset, *, tolerance=1e-9):
    """Intersect triangles with n.p+offset=0, returning (N,2,3) segments.

    The normal/offset pair is normalized together. The tolerance is a length,
    used only for near-plane classification and duplicate endpoint detection.
    Wholly coplanar triangles and point-only tangencies produce no segment.
    """
    vertices, faces = _checked_mesh(vertices, faces)
    normal = np.asarray(normal, float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isfinite(offset) or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Expected a finite plane and positive length tolerance')
    length = np.linalg.norm(normal)
    if length == 0:
        raise ValueError('Plane normal cannot be zero')
    normal, offset = normal/length, float(offset)/length
    if not len(faces):
        return np.empty((0, 2, 3), float)
    triangles = vertices[faces]
    distances = triangles @ normal + offset
    coplanar = np.all(np.abs(distances) <= tolerance, axis=1)
    active = ((distances.min(axis=1) <= tolerance)
              & (distances.max(axis=1) >= -tolerance) & ~coplanar)
    result = []
    for triangle, signed in zip(triangles[active], distances[active]):
        points = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = signed[first], signed[second]
            if abs(a) <= tolerance:
                points.append(triangle[first])
            if (a < -tolerance and b > tolerance) or (a > tolerance and b < -tolerance):
                points.append(triangle[first] + a/(a-b)*(triangle[second]-triangle[first]))
        unique = []
        for point in points:
            if not any(np.linalg.norm(point-old) <= tolerance for old in unique):
                unique.append(point)
        if len(unique) >= 2:
            # Near-plane classification can supply three almost-collinear points.
            # Their extreme pair is the triangle's section, without a hull fill.
            pts = np.asarray(unique)
            squared = np.sum((pts[:, None]-pts[None, :])**2, axis=2)
            first, second = np.unravel_index(np.argmax(squared), squared.shape)
            if squared[first, second] > tolerance**2:
                result.append(pts[[first, second]])
    if not result:
        return np.empty((0, 2, 3), float)
    return np.asarray(result)


def section_segments(vertices, faces, axis, level, *, tolerance=1e-9):
    """Axis-aligned section as (N,2,2), in ascending remaining-axis order."""
    if axis not in (0, 1, 2) or not np.isfinite(level):
        raise ValueError('axis must be 0, 1 or 2 and level must be finite')
    normal = np.eye(3)[axis]
    segments = plane_segments(vertices, faces, normal, -float(level), tolerance=tolerance)
    return segments[:, :, [i for i in range(3) if i != axis]]


def geometry_from_segments(segments, *, tolerance=1e-9, return_report=False):
    """Polygonize one component's section using even/odd nested-loop fill.

    Snap only to the declared numerical grid; no buffering or hole filling.
    An incomplete section may retain closed regions, so its report must be
    consulted before interpreting an empty overlap as evidence of clearance.
    """
    segments = np.asarray(segments, float)
    if (segments.ndim != 3 or segments.shape[1:] != (2, 2)
            or not np.isfinite(segments).all() or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('segments must be finite (N,2,2) with positive tolerance')
    snapped = np.rint(segments/tolerance)*tolerance
    snapped = snapped[np.linalg.norm(snapped[:, 1]-snapped[:, 0], axis=1) > tolerance/2]
    # Duplicate triangles at a shared edge must not cancel a genuine boundary.
    unique = {}
    for segment in snapped:
        ends = [tuple(point) for point in segment]
        key = tuple(sorted(ends))
        unique[key] = segment
    snapped = np.asarray(list(unique.values()), float).reshape(-1, 2, 2)
    if not len(snapped):
        report = dict(segment_count=0, polygon_faces=0, retained_faces=0,
                      open_linework_length=0., invalid_ring_length=0., complete=True,
                      endpoint_grid_tolerance=tolerance)
        geometry = GeometryCollection()
        return (geometry, report) if return_report else geometry
    linework = unary_union(MultiLineString(snapped.tolist()))
    polygons, cuts, dangles, invalid = polygonize_full(linework)
    retained = []
    starts, ends = snapped[:, 0], snapped[:, 1]
    for polygon in polygons.geoms:
        point = polygon.representative_point()
        x, y = point.x, point.y
        crosses = (starts[:, 1] > y) != (ends[:, 1] > y)
        a, b = starts[crosses], ends[crosses]
        ray_x = a[:, 0] + (y-a[:, 1])*(b[:, 0]-a[:, 0])/(b[:, 1]-a[:, 1])
        if np.count_nonzero(ray_x > x) % 2:
            retained.append(polygon)
    geometry = unary_union(retained) if retained else GeometryCollection()
    open_length = float(cuts.length+dangles.length)
    report = dict(segment_count=len(snapped), polygon_faces=len(polygons.geoms),
                  retained_faces=len(retained), open_linework_length=open_length,
                  invalid_ring_length=float(invalid.length),
                  complete=open_length <= tolerance and invalid.length <= tolerance,
                  endpoint_grid_tolerance=tolerance)
    return (geometry, report) if return_report else geometry


def section_geometry(vertices, faces, axis, level, *, tolerance=1e-9,
                     component_labels=None, return_report=False):
    """Section area: parity within components, union across separate solids."""
    vertices, faces = _checked_mesh(vertices, faces)
    labels = mesh_component_labels(faces) if component_labels is None else np.asarray(component_labels)
    if labels.shape != (len(faces),):
        raise ValueError('component_labels must provide one label per triangle')
    geometries, reports = [], []
    for label in np.unique(labels):
        segments = section_segments(vertices, faces[labels == label], axis, level,
                                    tolerance=tolerance)
        geometry, report = geometry_from_segments(segments, tolerance=tolerance, return_report=True)
        report['component'] = int(label)
        reports.append(report)
        geometries.append(geometry)
    result = unary_union(geometries) if geometries else GeometryCollection()
    report = dict(axis=int(axis), level=float(level),
                  complete=all(item['complete'] for item in reports),
                  components=reports,
                  solid_rule='even_odd_within_edge_component_then_union',
                  scope='one_2D_section_not_a_3D_collision_certificate')
    return (result, report) if return_report else result


def conversion_receipt(part_name, original_mm_path, converted_m_path):
    """Measure the declared source-mm to centered-m conversion, without writing.

    Source files are not presumed to share an assembled frame. DSUB is simplified
    in the existing converter, so nearest-vertex differences are measurements,
    not a requirement for pointwise equality or a source-frame registration.
    """
    original, original_faces = read_native_mesh(original_mm_path)
    converted, converted_faces = read_native_mesh(converted_m_path)
    rotation = np.eye(3)
    if part_name != 'GMC_Laser_Plate_Virtual' and not part_name.startswith(('KET', 'RGOCG', 'Waterproof', 'DSUB')):
        rotation[1, 1] = rotation[2, 2] = -1
    rotated = original @ rotation.T
    center = (rotated.min(axis=0)+rotated.max(axis=0))/2
    expected = (rotated-center)*.001
    affine = np.eye(4)
    affine[:3, :3] = rotation*.001
    affine[:3, 3] = -center*.001
    forward = cKDTree(converted).query(expected)[0]
    reverse = cKDTree(expected).query(converted)[0]
    return dict(part=part_name, original_mm_path=str(original_mm_path),
                converted_m_path=str(converted_m_path),
                source_units='millimetres', converted_units='metres',
                source_mm_to_native_m_affine=affine.tolist(),
                original_bounds_mm=[original.min(0).tolist(), original.max(0).tolist()],
                original_bbox_center_mm=((original.min(0)+original.max(0))/2).tolist(),
                converted_bounds_m=[converted.min(0).tolist(), converted.max(0).tolist()],
                original_vertex_count=len(original), converted_vertex_count=len(converted),
                original_triangle_count=len(original_faces), converted_triangle_count=len(converted_faces),
                maximum_source_to_converted_nearest_vertex_distance_m=float(forward.max()),
                maximum_converted_to_source_nearest_vertex_distance_m=float(reverse.max()),
                shared_original_assembly_frame_established=False,
                source_files_modified=False)
