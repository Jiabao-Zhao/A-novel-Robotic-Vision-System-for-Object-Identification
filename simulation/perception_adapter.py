from pathlib import Path


def pointcloud_localization_inputs(observation, rgb_path, depth_path):
    """Map a LIBERO observation to PointCloudLocalization.run_from_arrays.

    LIBERO depth has already been converted to metres, so depth_scale_m is 1.
    This function deliberately does not instantiate or execute the downstream
    perception pipeline.
    """
    height, width = observation.depth_m.shape
    K = observation.intrinsics
    return {
        "rgb": observation.rgb,
        "depth": observation.depth_m,
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
