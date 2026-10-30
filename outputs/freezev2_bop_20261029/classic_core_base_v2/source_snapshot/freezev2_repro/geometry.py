"""Metric camera/CAD geometry; points are rows, T_camera_from_CAD is explicit."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def validate_intrinsics(K):
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Invalid calibrated pinhole intrinsics")
    return K


def transform_points(points, R, t):
    return np.asarray(points) @ np.asarray(R).T + np.asarray(t).reshape(3)


def validate_pose(R, t, atol=1e-5):
    R, t = np.asarray(R), np.asarray(t)
    if R.shape != (3, 3) or t.size != 3 or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Pose must be finite 3x3 SO(3) plus three translations")
    if not np.allclose(R.T @ R, np.eye(3), atol=atol) or not np.isclose(np.linalg.det(R), 1, atol=atol):
        raise ValueError("Rotation is not in SO(3)")


def project(points_camera_m, K):
    K = validate_intrinsics(K)
    p = np.asarray(points_camera_m, dtype=np.float64)
    uv = np.full((len(p), 2), np.nan)
    valid = np.isfinite(p).all(1) & (p[:, 2] > 0)
    h = p[valid] @ K.T
    uv[valid] = h[:, :2] / h[:, 2:3]
    return uv, valid


def backproject(uv, depth_m, K):
    K = validate_intrinsics(K)
    uv, z = np.asarray(uv, dtype=np.float64), np.asarray(depth_m, dtype=np.float64).reshape(-1)
    if uv.shape != (len(z), 2) or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Backprojection needs finite positive metric depths")
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K).T
    return rays * z[:, None]


def depth_to_metres(raw, depth_scale_mm):
    if not np.isfinite(depth_scale_mm) or depth_scale_mm <= 0:
        raise ValueError("BOP depth_scale is positive millimetres per raw unit")
    return np.asarray(raw, dtype=np.float64) * float(depth_scale_mm) * 1e-3


@dataclass(frozen=True)
class SquareCrop:
    """Edges in full-image pixel coordinates (integer coordinates are centres)."""
    left: float
    top: float
    side: float
    output_size: int = 224

    @classmethod
    def from_mask(cls, mask, output_size=224):
        y, x = np.nonzero(mask)
        if not len(x):
            raise ValueError("Empty external candidate mask")
        side = float(max(x.max() - x.min() + 1, y.max() - y.min() + 1))
        return cls((x.min() + x.max()) / 2 - side / 2,
                   (y.min() + y.max()) / 2 - side / 2, side, output_size)

    def to_full(self, uv_crop):
        return (np.asarray(uv_crop) + .5) * self.side / self.output_size + [self.left, self.top]

    def from_full(self, uv_full):
        return (np.asarray(uv_full) - [self.left, self.top]) * self.output_size / self.side - .5

    def patch_centres(self, grid=16):
        y, x = np.mgrid[:grid, :grid]
        uv = np.column_stack([x.ravel(), y.ravel()])
        return self.to_full((uv + .5) * self.output_size / grid - .5)

    def extract_rgb(self, rgb, padding=0):
        import cv2
        y, x = np.mgrid[:self.output_size, :self.output_size]
        uv = self.to_full(np.stack([x, y], -1))
        return cv2.remap(np.asarray(rgb), uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
                         interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(padding, padding, padding))


def observed_clouds(depth_m, mask, K, *, grid=16, dense_count=3000, seed=0):
    """Sparse patch centres inside mask, dense valid masked depth, no GT pose."""
    mask = np.asarray(mask, bool)
    if mask.shape != depth_m.shape:
        raise ValueError("Mask/depth dimensions disagree")
    crop = SquareCrop.from_mask(mask)
    uv = crop.patch_centres(grid)
    xy = np.floor(uv + .5).astype(int)
    h, w = mask.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    indices = np.flatnonzero(inside)
    q = xy[inside]
    good = mask[q[:, 1], q[:, 0]] & np.isfinite(depth_m[q[:, 1], q[:, 0]]) & (depth_m[q[:, 1], q[:, 0]] > 0)
    indices, q = indices[good], q[good]
    # Backproject actual selected pixel centres, not fractional coordinates paired
    # with a different pixel's depth; retain continuous grid coordinates for DINO.
    sparse = backproject(q, depth_m[q[:, 1], q[:, 0]], K)
    yy, xx = np.nonzero(mask & np.isfinite(depth_m) & (depth_m > 0))
    if len(xx) < 3 or len(sparse) < 3:
        raise ValueError("Insufficient valid observed RGB-D points")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(xx), min(dense_count, len(xx)), replace=False)
    dense = backproject(np.column_stack([xx[selected], yy[selected]]), depth_m[yy[selected], xx[selected]], K)
    return dict(sparse=sparse, dense=dense, patch_indices=indices, pixel_uv=q,
                feature_uv=crop.from_full(uv[indices]), crop=crop,
                valid_dense_before_subsampling=len(xx))


def visible_surface(points_cad_m, normals_cad, R, t, K, depth_m, tolerance_m):
    camera = transform_points(points_cad_m, R, t)
    uv, valid = project(camera, K)
    h, w = depth_m.shape
    xy = np.floor(np.nan_to_num(uv, nan=-1000) + .5).astype(int)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if normals_cad is not None:
        normals = np.asarray(normals_cad) @ np.asarray(R).T
        valid &= np.einsum('ij,ij->i', normals, camera) < 0
    selected = np.flatnonzero(valid)
    zbuffer = depth_m[xy[selected, 1], xy[selected, 0]]
    valid[selected] &= (zbuffer > 0) & (np.abs(zbuffer - camera[selected, 2]) <= tolerance_m)
    return uv, valid


def surface_sample(vertices, triangles, count, seed):
    rng = np.random.default_rng(seed)
    tri = np.asarray(vertices)[np.asarray(triangles)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if area2.sum() <= 0:
        raise ValueError("CAD has no surface area")
    faces = rng.choice(len(tri), count, p=area2 / area2.sum())
    u, v = rng.random((2, count))
    u = np.sqrt(u)
    bary = np.column_stack([1 - u, u * (1 - v), u * v])
    points = np.einsum('ni,nij->nj', bary, tri[faces])
    normals = cross[faces] / area2[faces, None]
    return points, normals, faces
