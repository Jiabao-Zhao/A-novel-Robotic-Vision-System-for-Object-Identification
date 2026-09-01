from pathlib import Path

import numpy as np


def create_open3d_pointcloud(rgb, depth_m, intrinsics, depth_trunc_m=10.0):
    """Reconstruct a colored point cloud from rendered RGB-D and calibration."""
    try:
        import open3d as o3d
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Open3D is unavailable. Install it in the WSL environment with "
            "`python -m pip install open3d`."
        ) from error

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB; received {rgb.shape}.")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB shape {rgb.shape[:2]} and depth shape {depth.shape} do not match.")
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics; received {K.shape}.")

    valid_mask = np.isfinite(depth) & (depth > 0.0) & (depth < float(depth_trunc_m))
    clean_depth = np.where(valid_mask, depth, 0.0).astype(np.float32)
    height, width = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(np.ascontiguousarray(clean_depth)),
        depth_scale=1.0,
        depth_trunc=float(depth_trunc_m),
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return cloud, int(valid_mask.sum())


def save_pointcloud(cloud, output_path):
    try:
        import open3d as o3d
    except ModuleNotFoundError as error:
        raise RuntimeError("Open3D is required to save the point cloud.") from error
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.is_empty():
        raise RuntimeError("RGB-D reconstruction generated an empty point cloud.")
    if not o3d.io.write_point_cloud(str(output_path), cloud):
        raise RuntimeError(f"Open3D failed to save {output_path}.")
    points = np.asarray(cloud.points)
    return {
        "path": output_path,
        "point_count": int(points.shape[0]),
        "minimum_xyz_m": points.min(axis=0),
        "maximum_xyz_m": points.max(axis=0),
    }
