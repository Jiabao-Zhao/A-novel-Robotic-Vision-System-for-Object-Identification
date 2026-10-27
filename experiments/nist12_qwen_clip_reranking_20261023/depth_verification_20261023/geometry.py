"""Native-scale single-view CAD verification for the frozen N-scene trial.

The support-face search uses CAD geometry only. Fitting never receives object
labels, simulator poses, or a scale parameter. Camera coordinates are metres.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


VOXEL_M = .001
SAMPLE_POINTS = 40000
ITERATIONS = 30
DEPTH_SCALE_M = .005
YAW_DEGREES = tuple(range(0, 360, 10))
MAX_SUPPORT_FACES = 12


def voxel_means(points, size=VOXEL_M):
    """Fixed CAD-frame/observed-camera voxel samples; no pose-dependent resampling."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('Expected a nonempty N by 3 point array.')
    if not np.isfinite(points).all():
        raise ValueError('Point coordinates must be finite.')
    _, inverse = np.unique(np.floor(points / size).astype(np.int64), axis=0,
                           return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack([np.bincount(inverse, weights=points[:, d]) / counts
                            for d in range(3)])


def sample_mesh(vertices, triangles, count=SAMPLE_POINTS):
    """Deterministic area-uniform mesh sampling, seed 0, without changing units."""
    faces = np.asarray(vertices, float)[np.asarray(triangles, int)]
    areas = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0],
                                    faces[:, 2] - faces[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('CAD mesh has no finite surface area.')
    rng = np.random.default_rng(0)
    picked = faces[rng.choice(len(faces), int(count), p=areas / areas.sum())]
    a, b = rng.random((2, int(count)))
    root = np.sqrt(a)
    return ((1 - root)[:, None] * picked[:, 0]
            + (root * (1 - b))[:, None] * picked[:, 1]
            + (root * b)[:, None] * picked[:, 2])


def rotation_between(source, target):
    source = np.asarray(source, float) / np.linalg.norm(source)
    target = np.asarray(target, float) / np.linalg.norm(target)
    dot = float(np.clip(source @ target, -1, 1))
    if dot > 1 - 1e-12:
        return np.eye(3)
    if dot < -1 + 1e-12:
        helper = np.eye(3)[np.argmin(np.abs(source))]
        return axis_rotation(np.cross(source, helper), np.pi)
    cross = np.cross(source, target)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + dot)


def axis_rotation(axis, radians):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(radians) * skew + (1 - np.cos(radians)) * (skew @ skew)


def support_faces(vertices):
    """CAD convex-hull support normals, following the existing rest_options math.

    Unlike capture generation, this function has no identity-specific exclusions
    or scene-selected rest face. Support uses a 0.2 mm band, 0.4 mm COM margin,
    15 square-mm area minimum. Stable directions closest to each signed CAD axis
    are retained first; area-ranked directions at least 25 degrees from them fill
    the remaining slots up to twelve. This protects small stable end faces from
    being displaced by several large side faces, without object-specific rules.
    """
    vertices = np.unique(np.asarray(vertices, float), axis=0)
    hull = ConvexHull(vertices)
    reference = vertices.mean(0)
    tri = vertices[hull.simplices]
    volume = np.abs(np.einsum('ij,ij->i', tri[:, 0] - reference,
                    np.cross(tri[:, 1] - reference, tri[:, 2] - reference))) / 6
    center = np.average((tri.sum(1) + reference) / 4, axis=0, weights=volume)
    candidates, seen = [], []
    for outward in hull.equations[:, :3]:
        if any(outward @ previous > .9999 for previous in seen):
            continue
        seen.append(outward)
        rotation = rotation_between(outward, [0., 0., -1.])
        turned = vertices @ rotation.T
        contact = turned[turned[:, 2] <= turned[:, 2].min() + .0002, :2]
        if len(contact) < 3:
            continue
        try:
            support = ConvexHull(contact)
        except Exception:
            continue
        projected_center = (center @ rotation.T)[:2]
        margin = -np.max(support.equations[:, :2] @ projected_center
                         + support.equations[:, 2])
        if margin < .0004 or support.volume < .000015:
            continue
        candidates.append(dict(outward_normal=outward.tolist(),
                               support_area_m2=float(support.volume),
                               com_margin_m=float(margin)))
    candidates.sort(key=lambda row: (-row['support_area_m2'],
                                    *np.round(row['outward_normal'], 12)))
    selected = []
    for axis in (np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
                 np.array([0., 1., 0.]), np.array([0., -1., 0.]),
                 np.array([0., 0., 1.]), np.array([0., 0., -1.])):
        if not candidates:
            break
        candidate = max(candidates, key=lambda row: np.asarray(row['outward_normal']) @ axis)
        normal = np.asarray(candidate['outward_normal'])
        if any(normal @ np.asarray(row['outward_normal']) > .9999 for row in selected):
            continue
        selected.append(dict(**candidate, selection='nearest_signed_CAD_axis'))
    for candidate in candidates:
        normal = np.asarray(candidate['outward_normal'])
        if any(normal @ np.asarray(row['outward_normal']) > np.cos(np.deg2rad(25))
               for row in selected):
            continue
        selected.append(dict(**candidate, selection='area_ranked_additional_support'))
        if len(selected) == MAX_SUPPORT_FACES:
            break
    if not selected:
        raise ValueError('No CAD-only stable support face passed the fixed checks.')
    return selected


def table_frame(plane, observed):
    plane = np.asarray(plane, float)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError('Expected finite camera-frame plane [a,b,c,d].')
    norm = np.linalg.norm(plane[:3])
    if norm == 0:
        raise ValueError('Plane normal cannot be zero.')
    normal, distance = plane[:3] / norm, float(plane[3] / norm)
    if observed.mean(0) @ normal + distance < 0:
        normal, distance = -normal, -distance
    helper = np.eye(3)[np.argmin(np.abs(normal))]
    x_axis = helper - helper @ normal * normal
    x_axis /= np.linalg.norm(x_axis)
    return normal, distance, np.column_stack([x_axis, np.cross(normal, x_axis)])


def contact_pose(vertices, observed, normal, distance, basis, outward, yaw, clearance):
    rotation = (axis_rotation(normal, np.deg2rad(yaw))
                @ rotation_between(outward, -normal))
    turned = vertices @ rotation.T
    shift = (clearance - (turned @ normal + distance).min()) * normal
    shift += basis @ (basis.T @ (observed.mean(0) - (turned + shift).mean(0)))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, shift
    return pose


def refine_pose(cad, observed, normal, basis, initial, iterations=ITERATIONS):
    """One-sided observed-to-CAD ICP with fixed samples and exact table height."""
    pose = initial.copy()
    local = (observed - pose[:3, 3]) @ pose[:3, :3]
    distances, indices = cad.tree.query(local, workers=1)
    best_rmse = rmse = float(np.sqrt(np.mean(distances ** 2)))
    best_pose, history = pose.copy(), [rmse]
    stopped = 'iteration_limit'
    for _ in range(int(iterations)):
        matched = cad.points[indices] @ pose[:3, :3].T + pose[:3, 3]
        cad_center, observed_center = matched.mean(0), observed.mean(0)
        a, b = (matched - cad_center) @ basis, (observed - observed_center) @ basis
        cross = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        dot = np.sum(a * b)
        yaw = np.arctan2(cross, dot) if cross != 0 or dot != 0 else 0.
        delta = axis_rotation(normal, yaw)
        shift = cad_center - delta @ cad_center + basis @ (basis.T @ (observed_center - cad_center))
        candidate = pose.copy()
        candidate[:3, :3] = delta @ pose[:3, :3]
        candidate[:3, 3] = delta @ pose[:3, 3] + shift
        local = (observed - candidate[:3, 3]) @ candidate[:3, :3]
        distances, new_indices = cad.tree.query(local, workers=1)
        updated = float(np.sqrt(np.mean(distances ** 2)))
        if updated > rmse + 1e-12:
            stopped = 'no_icp_improvement'
            break
        change = rmse - updated
        pose, indices, rmse = candidate, new_indices, updated
        history.append(rmse)
        if rmse < best_rmse:
            best_rmse, best_pose = rmse, pose.copy()
        if change <= 1e-8:
            stopped = 'rmse_tolerance'
            break
    return best_pose, dict(rmse_m=best_rmse, initial_rmse_m=history[0],
                           rmse_history_m=history, iterations=len(history) - 1,
                           stop_reason=stopped)


def depth_points(depth, mask, K, roi_xy=(0, 0)):
    """Pinhole backprojection with the experiment's pixel offset exactly zero."""
    depth, mask = np.asarray(depth), np.asarray(mask, bool)
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    x, y = x + roi_xy[0], y + roi_xy[1]
    z = depth[y - roi_xy[1], x - roi_xy[0]]
    return np.column_stack([(x - K[0, 2]) * z / K[0, 0],
                            (y - K[1, 2]) * z / K[1, 1], z])


def score_projection(depth, observed, rendered, observed_total=None):
    """Full mask union; all visible CAD pixels count, with no occluder carving."""
    depth, observed, rendered = np.asarray(depth), np.asarray(observed, bool), np.asarray(rendered)
    if depth.shape != observed.shape or depth.shape != rendered.shape:
        raise ValueError('Observed/rendered masks and depths must share the image region.')
    cad = np.isfinite(rendered) & (rendered > 0)
    overlap = observed & cad
    valid_overlap = overlap & np.isfinite(depth) & (depth > 0)
    local_observed = int(observed.sum())
    total = local_observed if observed_total is None else int(observed_total)
    if total < local_observed:
        raise ValueError('Full observed count cannot be smaller than the local observed mask.')
    intersection, projected = int(overlap.sum()), int(cad.sum())
    counts = dict(observed_pixels=total, projected_pixels=projected,
                  union_pixels=total + projected - intersection, intersection_pixels=intersection,
                  valid_depth_overlap_pixels=int(valid_overlap.sum()),
                  observed_only_pixels=total - intersection,
                  cad_only_pixels=projected - intersection)
    row = dict(**counts, p_obs=None, p_cad=None, p_depth=None,
               geometry_score=None, mask_iou=None, depth_mae_m=None)
    if total == 0:
        return dict(**row, status='unavailable_empty_observed_mask')
    if not cad.any():
        return dict(**row, status='unavailable_empty_projected_mask')
    row['p_obs'] = counts['observed_only_pixels'] / counts['union_pixels']
    row['p_cad'] = counts['cad_only_pixels'] / counts['union_pixels']
    row['mask_iou'] = counts['intersection_pixels'] / counts['union_pixels']
    if not valid_overlap.any():
        return dict(**row, status='unavailable_no_valid_depth_overlap')
    residual = np.abs(depth[valid_overlap] - rendered[valid_overlap])
    row['depth_mae_m'] = float(residual.mean())
    row['p_depth'] = float(np.minimum(residual / DEPTH_SCALE_M, 1).mean())
    row['geometry_score'] = 1 - (row['p_obs'] + row['p_cad'] + row['p_depth']) / 3
    return dict(**row, status='available')


@dataclass
class PreparedCAD:
    vertices: np.ndarray
    points: np.ndarray
    tree: object
    supports: list
    surface: object


def prepare_cad(mesh):
    """Prepare once per supplied native-metre CAD; create no output files."""
    from scripts.surface_verification import CadSurface
    native_vertices, triangles = np.asarray(mesh.vertices, float), np.asarray(mesh.triangles, int)
    points = voxel_means(sample_mesh(native_vertices, triangles))
    # STL commonly repeats a vertex for each incident triangle. Deduplicating
    # changes neither the native surface nor its exact support/contact height.
    vertices = np.unique(native_vertices, axis=0)
    return PreparedCAD(vertices, points, cKDTree(points), support_faces(vertices), CadSurface(mesh))


def projection_score(cad, pose, K, depth, mask, roi_xy=(0, 0), observed_bbox=None,
                     observed_total=None):
    """Raycast all projected CAD; count the full mask union without extra rays.

    Observation pixels outside the CAD bounding rectangle contribute exactly to
    O minus C. Omitting rays where CAD cannot project changes neither the full
    silhouette union nor valid-overlap depth disagreement.
    """
    # K is the full camera calibration; shifting its principal point expresses
    # exactly the same rays in the caller's ROI coordinates.
    roi_xy = (0, 0) if roi_xy is None else roi_xy
    local_K = np.asarray(K, float).copy()
    local_K[:2, 2] -= roi_xy
    total = int(mask.sum()) if observed_total is None else int(observed_total)
    if total == 0:
        return dict(status='unavailable_empty_observed_mask', geometry_score=None)
    # Never crop the projected CAD to the observed mask/ROI. This is its own
    # complete image-space bounding box, including every unsupported protrusion.
    render = cad.surface.render(pose, local_K, depth.shape, pixel_offset=0.)
    if render is None:
        return dict(status='unavailable_behind_or_outside_camera', geometry_score=None)
    if render['clipped']:
        return dict(status='unavailable_projection_outside_supplied_image', geometry_score=None)
    x1, y1, x2, y2 = render['roi']
    row = score_projection(depth[y1:y2, x1:x2], mask[y1:y2, x1:x2], render['depth'],
                           observed_total=total)
    return dict(**row, projected_cad_roi_xyxy=[int(x1 + roi_xy[0]), int(y1 + roi_xy[1]),
                                            int(x2 + roi_xy[0]), int(y2 + roi_xy[1])])


def fit_pair(mesh, observed_points, plane, K, depth_roi, mask_roi,
             roi_xy=(0, 0), clearances=(0., .001)):
    """Fit one supplied CAD to one candidate; scores never use scene labels.

    Pass PreparedCAD to reuse its native samples and raycaster across candidates.
    Full-frame depth/mask arrays with roi_xy=(0,0) are recommended: a restricted
    ROI makes a projected CAD extending beyond that supplied image unavailable.
    Every support/yaw hypothesis is ray-scored; RMSE does not preselect the face.
    """
    started = perf_counter()
    cad = mesh if isinstance(mesh, PreparedCAD) else prepare_cad(mesh)
    depth, mask = np.asarray(depth_roi), np.asarray(mask_roi, bool)
    if depth.shape != mask.shape:
        raise ValueError('Depth and observed mask must share a camera image region.')
    if not mask.any():
        return dict(status='unavailable_empty_observed_mask', arms=[], seconds=perf_counter()-started)
    yy, xx = np.nonzero(mask)
    observed_bbox = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    observed_total = len(xx)
    observed = np.asarray(observed_points, float)
    if observed.ndim != 2 or observed.shape[1] != 3 or not len(observed):
        return dict(status='unavailable_empty_observed_cloud', arms=[], seconds=perf_counter()-started)
    observed = voxel_means(observed)
    normal, distance, basis = table_frame(plane, observed)
    arms = []
    for clearance in clearances:
        begin, bank = perf_counter(), []
        for face_index, face in enumerate(cad.supports):
            for yaw in YAW_DEGREES:
                initial = contact_pose(cad.vertices, observed, normal, distance, basis,
                                       face['outward_normal'], yaw, float(clearance))
                pose, icp = refine_pose(cad, observed, normal, basis, initial)
                projected = projection_score(cad, pose, K, depth, mask, roi_xy,
                                             observed_bbox, observed_total)
                heights = (cad.vertices @ pose[:3, :3].T + pose[:3, 3]) @ normal + distance
                if not np.isclose(heights.min(), clearance, atol=1e-10):
                    raise AssertionError('Table contact/clearance changed during ICP.')
                bank.append(dict(support_index=face_index, initial_yaw_deg=yaw,
                                 clearance_m=float(clearance), T_camera_from_cad=pose.tolist(),
                                 minimum_table_height_m=float(heights.min()),
                                 fitting=icp, verification=projected))
        available = [i for i, row in enumerate(bank)
                     if row['verification']['geometry_score'] is not None]
        geometry_best = max(available, key=lambda i: bank[i]['verification']['geometry_score']) if available else None
        rmse_best = min(range(len(bank)), key=lambda i: bank[i]['fitting']['rmse_m'])
        arm_name = ('contact_0mm' if clearance == 0 else 'clearance_1mm'
                    if clearance == .001 else f'clearance_{clearance*1000:g}mm')
        arms.append(dict(arm_name=arm_name, clearance_m=float(clearance), status='available' if available else 'unavailable_all_projections',
                         best_geometry_index=geometry_best, best_geometry=bank[geometry_best] if geometry_best is not None else None,
                         best_rmse_index=rmse_best, best_rmse=bank[rmse_best], pose_bank=bank,
                         hypothesis_count=len(bank), available_projections=len(available), seconds=perf_counter()-begin))
    return dict(status='complete', arms=arms, observed_voxel_points=len(observed),
                cad_voxel_points=len(cad.points), support_faces=cad.supports,
                cad_native_extent_m=np.ptp(cad.vertices, axis=0).tolist(),
                seconds=perf_counter()-started)
