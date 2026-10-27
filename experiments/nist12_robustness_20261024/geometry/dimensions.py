"""Simple metric-size control: footprint long/short side and table-relative height."""
import numpy as np
from scipy.spatial import ConvexHull

ERROR_SCALE_M = .005


def footprint_sides(points_xy):
    points = np.unique(np.asarray(points_xy, dtype=float), axis=0)
    if not len(points):
        raise ValueError('Footprint needs at least one point.')
    if not np.isfinite(points).all() or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Footprint points must be finite N by 2 coordinates.')
    centered = points - points.mean(0)
    if len(points) < 3 or np.linalg.matrix_rank(centered) < 2:
        direction = np.linalg.svd(centered, full_matrices=True)[2][0]
        return np.array([np.ptp(centered @ direction), 0.])
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2))
    choices = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        extent = np.ptp(points @ np.array([[c, -s], [s, c]]), axis=0)
        choices.append((float(np.prod(extent)), -float(extent.max()), extent))
    extent = min(choices, key=lambda item: item[:2])[2]
    return np.sort(extent)[::-1]


def observed_dimensions(points_camera_m, plane_camera_m, native):
    points = np.asarray(points_camera_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        return {'status': 'unavailable_observed_points', 'dimensions_m': None}
    normal, distance, basis = native.table_frame(plane_camera_m, points)
    sides = footprint_sides(points @ basis)
    height = float(np.max(points @ normal + distance))
    return {'status': 'available', 'dimensions_m': [*sides.tolist(), height]}


def cad_dimensions(cad, native):
    rows = []
    for index, support in enumerate(cad.supports):
        rotation = native.rotation_between(support['outward_normal'], [0., 0., -1.])
        points = cad.vertices @ rotation.T
        sides = footprint_sides(points[:, :2])
        rows.append({'support_index': index,
                     'dimensions_m': [*sides.tolist(), float(np.ptp(points[:, 2]))]})
    return rows


def score_dimensions(observed, expected):
    if observed['status'] != 'available':
        return {'status': observed['status'], 'score': None}
    if not expected:
        return {'status': 'unavailable_CAD_supports', 'score': None}
    values = np.asarray(observed['dimensions_m'], dtype=float)
    candidates = []
    for row in expected:
        difference = np.abs(values - np.asarray(row['dimensions_m'], dtype=float))
        score = 1. - float(np.minimum(difference / ERROR_SCALE_M, 1.).mean())
        candidates.append({'support_index': row['support_index'], 'score': score,
                           'absolute_dimension_errors_m': difference.tolist(),
                           'CAD_dimensions_m': row['dimensions_m']})
    best = max(candidates, key=lambda item: item['score'])
    return {'status': 'available', 'score': best['score'], 'best_support': best,
            'observed_dimensions_m': observed['dimensions_m'], 'error_scale_m': ERROR_SCALE_M}
