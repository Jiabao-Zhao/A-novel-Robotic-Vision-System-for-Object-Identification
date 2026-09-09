import json
from pathlib import Path

import numpy as np


LIBERO_WORKSPACE_MIN_XYZ_M = (-0.25, -0.35, -0.02)
LIBERO_WORKSPACE_MAX_XYZ_M = (0.35, 0.35, 0.23)


def pointcloud_localization_inputs(observation, rgb_path, depth_path, depth_m=None):
    """Map a LIBERO observation to PointCloudLocalization.run_from_arrays.

    LIBERO depth has already been converted to metres, so depth_scale_m is 1.
    This function deliberately does not instantiate or execute the downstream
    perception pipeline.
    """
    depth = observation.depth_m if depth_m is None else np.asarray(depth_m, dtype=np.float32)
    if depth.shape != observation.depth_m.shape:
        raise ValueError(
            f"Adapted depth shape {depth.shape} does not match RGB-D capture "
            f"shape {observation.depth_m.shape}."
        )
    height, width = depth.shape
    K = observation.intrinsics
    return {
        "rgb": observation.rgb,
        "depth": depth,
        "depth_scale_m": 1.0,
        "camera_intrinsics": {
            "width": int(width),
            "height": int(height),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "rgb_path": Path(rgb_path),
        "depth_path": Path(depth_path),
        "visualize": False,
    }


def mask_depth_to_world_workspace(observation, minimum_xyz_m, maximum_xyz_m, exclude_xy_bounds=None):
    """Zero depth outside an axis-aligned MuJoCo-world workspace.

    The mask is computed only from metric depth, camera intrinsics, and
    world_T_camera. It does not use simulator segmentation or object state.
    """
    minimum = np.asarray(minimum_xyz_m, dtype=np.float64)
    maximum = np.asarray(maximum_xyz_m, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
        raise ValueError("Workspace bounds must be ordered three-element XYZ vectors.")

    depth = np.asarray(observation.depth_m, dtype=np.float32)
    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    K = np.asarray(observation.intrinsics, dtype=np.float64)
    camera_points = np.stack(
        (
            (columns - K[0, 2]) * depth / K[0, 0],
            (rows - K[1, 2]) * depth / K[1, 1],
            depth,
        ),
        axis=-1,
    )
    rotation = np.asarray(observation.world_T_camera[:3, :3], dtype=np.float64)
    translation = np.asarray(observation.world_T_camera[:3, 3], dtype=np.float64)
    world_points = camera_points @ rotation.T + translation
    valid = np.isfinite(depth) & (depth > 0.0)
    inside = valid & np.all(world_points >= minimum, axis=-1) & np.all(
        world_points <= maximum,
        axis=-1,
    )
    if exclude_xy_bounds is not None:
        lower, upper = np.asarray(exclude_xy_bounds, dtype=float)
        if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
            raise ValueError("Excluded fixture bounds must be ordered XY pairs.")
        inside &= ~np.all((world_points[..., :2] >= lower) & (world_points[..., :2] <= upper), axis=-1)
    return np.where(inside, depth, 0.0).astype(np.float32), inside


def run_libero_localization(observation, rgb_path, output_root, workspace_bounds=None):
    """Run the existing localizer; optional bounds are in the MuJoCo world frame.

    Other LIBERO arenas place their tables at different world offsets. Callers
    may translate the fixed crop using arena calibration, never object poses.
    Default LIBERO-Object behavior and all segmentation settings are unchanged.
    """
    from point_cloud_localization import PointCloudConfig, PointCloudLocalization

    output_root = Path(output_root)
    capture_dir = output_root / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    bounds = ((LIBERO_WORKSPACE_MIN_XYZ_M, LIBERO_WORKSPACE_MAX_XYZ_M)
              if workspace_bounds is None else workspace_bounds)
    workspace_depth, workspace_mask = mask_depth_to_world_workspace(observation, *bounds)
    workspace_depth_path = capture_dir / "workspace_depth.npy"
    np.save(workspace_depth_path, workspace_depth)

    config = PointCloudConfig(
        output_dir=output_root / "point_cloud",
        annotation_dir=output_root / "annotation",
        voxel_size_m=0.002,
        plane_distance_threshold_m=0.002,
        dbscan_min_points=5,
        min_cluster_points=50,
        min_roi_width_px=8,
        min_roi_height_px=8,
        raw_cluster_dbscan_min_points=5,
        statistical_outlier_neighbors=10,
        radius_outlier_radius_m=0.008,
        radius_outlier_min_neighbors=2,
    )
    inputs = pointcloud_localization_inputs(
        observation,
        rgb_path=rgb_path,
        depth_path=workspace_depth_path,
        depth_m=workspace_depth,
    )
    paths = PointCloudLocalization(config=config).run_from_arrays(**inputs)
    payload = json.loads(Path(paths["localization"]).read_text(encoding="utf-8"))
    return payload, paths, workspace_mask
