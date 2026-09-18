"""Experimental metric, visibility-aware RGB-D verification; no production imports."""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


VOXEL_M = .005
SIGMA_M = .005  # Fixed existing geometry resolution; not fitted to any outcome.
OCCLUSION_M = .005
TRANSLATION_ITERATIONS = 30


def bbox(mask):
    y, x = np.nonzero(mask)
    return None if not len(x) else (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    area = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
    overlap = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    return overlap / (area(a) + area(b) - overlap)


def surface_scores(depth, observed_mask, rendered_depth, K, pixel_offset=.5):
    """Depths are optical-axis Z in meters. Unknown depth supplies no contradiction.

    A closer non-target measurement can externally occlude CAD; a target pixel
    cannot hide its own incorrectly posed CAD. Full rendered support is supplied
    by the caller, including protrusions outside the observation mask.
    """
    valid = np.isfinite(depth) & (depth > 0)
    observed = observed_mask.astype(bool) & valid
    rendered = np.isfinite(rendered_depth) & (rendered_depth > 0)
    occluded = rendered & valid & ~observed & (depth < rendered_depth - OCCLUSION_M)
    assessable = rendered & valid & ~occluded
    intersection = observed & assessable
    union = observed | assessable
    counts = {"observed_pixels": int(observed.sum()), "rendered_pixels": int(rendered.sum()),
              "externally_occluded_pixels": int(occluded.sum()),
              "unknown_rendered_pixels": int((rendered & ~valid).sum()),
              "assessable_rendered_pixels": int(assessable.sum()), "union_pixels": int(union.sum())}
    if not observed.any() or not union.any():
        return {**counts, "status": "unassessable", "box_iou": None, "mask_iou": None, "surface_score": None}
    y, x = np.nonzero(intersection)
    rays = np.column_stack(((x + pixel_offset - K[0, 2]) / K[0, 0],
                            (y + pixel_offset - K[1, 2]) / K[1, 1], np.ones(len(x))))
    residual = np.abs(depth[intersection] - rendered_depth[intersection]) * np.linalg.norm(rays, axis=1)
    agreement = 1 / (1 + (residual / SIGMA_M) ** 2)
    return {**counts, "status": "assessable", "box_iou": box_iou(bbox(observed), bbox(assessable)),
            "mask_iou": float(intersection.sum() / union.sum()),
            "surface_score": float(agreement.sum() / union.sum()),
            "intersection_depth_rmse_m": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else None}


def depth_points(depth, mask, K, pixel_offset=.5):
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[y, x]
    return np.column_stack(((x + pixel_offset - K[0, 2]) * z / K[0, 0],
                            (y + pixel_offset - K[1, 2]) * z / K[1, 1], z))


def downsample(points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.voxel_down_sample(VOXEL_M).points)


class CadSurface:
    """One reusable native CAD raycaster. All transforms map CAD to camera."""

    def __init__(self, mesh):
        self.vertices = np.asarray(mesh.vertices)
        self.center = (self.vertices.min(0) + self.vertices.max(0)) / 2
        self.radius = np.linalg.norm(self.vertices - self.center, axis=1).max()
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def render(self, T, K, shape, observed_bbox=None, pixel_offset=.5):
        """Exact rays over projected CAD/observed union ROI, never crop to mask."""
        vertices = self.vertices @ T[:3, :3].T + T[:3, 3]
        if np.any(vertices[:, 2] <= 0):
            return None
        pixels = vertices @ K.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        lower = np.floor(pixels.min(0) - 1).astype(int)
        upper = np.ceil(pixels.max(0) + 1).astype(int)
        clipped = bool(lower[0] < 0 or lower[1] < 0 or upper[0] >= shape[1] or upper[1] >= shape[0])
        if observed_bbox is not None:
            lower = np.minimum(lower, observed_bbox[:2])
            upper = np.maximum(upper, observed_bbox[2:])
        x1, y1 = np.maximum(lower, 0)
        x2, y2 = np.minimum(upper, [shape[1], shape[0]])
        if x2 <= x1 or y2 <= y1:
            return None
        y, x = np.mgrid[y1:y2, x1:x2]
        directions = np.stack(((x + pixel_offset - K[0, 2]) / K[0, 0],
                                (y + pixel_offset - K[1, 2]) / K[1, 1], np.ones_like(x)), axis=-1)
        native_direction = directions @ T[:3, :3]
        origin = -T[:3, 3] @ T[:3, :3]
        rays = np.concatenate((np.broadcast_to(origin, native_direction.shape), native_direction), axis=-1)
        hit = self.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        local_K = K.copy()
        local_K[:2, 2] -= [x1, y1]
        return {"depth": hit["t_hit"].numpy(), "K": local_K, "roi": (int(x1), int(y1), int(x2), int(y2)),
                "clipped": clipped}

    def visible_points(self, rotation):
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = -rotation @ self.center + [0, 0, 4 * self.radius]
        K = np.array([[224., 0, 112], [0, 224., 112], [0, 0, 1]])
        render = self.render(T, K, (224, 224))
        p = depth_points(render["depth"], np.isfinite(render["depth"]), render["K"])
        return downsample(p - T[:3, 3])  # rotated native-CAD coordinates


def translation_fit(visible, observed):
    """Same 30 translation-only ICP updates for every view; retain its rotation.

    The existing 5 mm sampling and 7.5 mm correspondence radius are fixed.
    This controlled search does not claim unconstrained 6D pose recovery.
    """
    translation = observed.mean(0) - visible.mean(0)
    tree = cKDTree(visible)
    for _ in range(TRANSLATION_ITERATIONS):
        distance, index = tree.query(observed - translation)
        use = distance < 1.5 * VOXEL_M
        if use.any():
            translation = (observed[use] - visible[index[use]]).mean(0)
    distance, _ = tree.query(observed - translation)
    return translation, float(np.mean(distance < 1.5 * VOXEL_M))


def verify_pose(cad, T, depth, mask, K, table_plane=None, pixel_offset=.5):
    render = cad.render(T, K, depth.shape, bbox(mask), pixel_offset)
    reason = None
    if render is None:
        reason = "behind_or_outside_camera"
    elif render["clipped"]:
        reason = "image_clipped"
    if table_plane is not None:
        vertices = cad.vertices @ T[:3, :3].T + T[:3, 3]
        if np.min(vertices @ table_plane[:3] + table_plane[3]) < -VOXEL_M:
            reason = "table_penetration"
    if reason:
        return {"status": reason, "surface_score": 0., "mask_iou": 0., "box_iou": 0.}
    x1, y1, x2, y2 = render["roi"]
    result = surface_scores(depth[y1:y2, x1:x2], mask[y1:y2, x1:x2], render["depth"], render["K"], pixel_offset)
    return {**result, "render_roi_xyxy_exclusive": render["roi"]}
