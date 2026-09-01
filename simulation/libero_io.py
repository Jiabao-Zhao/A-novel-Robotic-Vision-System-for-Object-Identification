import json
from pathlib import Path

import numpy as np

from .libero_sensor import depth_statistics


def save_libero_observation(observation, environment, output_dir):
    """Save calibrated perception inputs without simulator object ground truth."""
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is required to save the LIBERO RGB image.") from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = output_dir / "rgb.png"
    depth_path = output_dir / "depth.npy"
    depth_visualization_path = output_dir / "depth_visualization.png"
    rgb_depth_figure_path = output_dir / "rgb_depth.png"
    intrinsics_path = output_dir / "intrinsics.npy"
    world_T_camera_path = output_dir / "world_T_camera.npy"
    camera_T_world_path = output_dir / "camera_T_world.npy"
    metadata_path = output_dir / "metadata.json"

    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"OpenCV failed to save {rgb_path}.")
    np.save(depth_path, observation.depth_m)
    np.save(intrinsics_path, observation.intrinsics)
    np.save(world_T_camera_path, observation.world_T_camera)
    np.save(camera_T_world_path, observation.camera_T_world)

    stats = depth_statistics(observation.depth_m)
    _save_visualizations(
        observation.rgb,
        observation.depth_m,
        depth_visualization_path,
        rgb_depth_figure_path,
    )

    metadata = {
        "suite": environment.suite_name,
        "task_index": environment.task_index,
        "task": environment.task_name,
        "instruction": observation.instruction,
        "camera": observation.camera_name,
        "rgb_observation_key": observation.rgb_key,
        "depth_observation_key": observation.depth_key,
        "rgb_shape": list(observation.rgb.shape),
        "rgb_dtype": str(observation.rgb.dtype),
        "depth_shape": list(observation.depth_m.shape),
        "depth_dtype": str(observation.depth_m.dtype),
        "depth_unit": "meter",
        "depth_conversion": "robosuite.utils.camera_utils.get_real_depth_map",
        "depth_statistics_m": stats,
        "orientation_correction": observation.orientation_correction,
        "world_T_camera_convention": (
            "Camera pose in the MuJoCo world frame; maps OpenCV-style camera-frame "
            "homogeneous points into world-frame points."
        ),
        "camera_T_world_convention": "Inverse of world_T_camera.",
        "robot_state": {
            key: np.asarray(value).tolist() for key, value in observation.robot_state.items()
        },
        "ground_truth_object_state_saved": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "rgb": rgb_path,
        "depth": depth_path,
        "depth_visualization": depth_visualization_path,
        "rgb_depth_figure": rgb_depth_figure_path,
        "intrinsics": intrinsics_path,
        "world_T_camera": world_T_camera_path,
        "camera_T_world": camera_T_world_path,
        "metadata": metadata_path,
    }


def _save_visualizations(rgb, depth_m, depth_visualization_path, rgb_depth_figure_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("matplotlib is required to save RGB-D visualizations.") from error

    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    valid = depth_m[valid_mask]
    masked_depth = np.ma.masked_where(~valid_mask, depth_m)
    color_map = plt.get_cmap("viridis").with_extremes(bad="black")
    plt.imsave(
        depth_visualization_path,
        masked_depth,
        cmap=color_map,
        vmin=float(valid.min()),
        vmax=float(valid.max()),
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("LIBERO agent-view RGB")
    axes[0].axis("off")
    depth_image = axes[1].imshow(
        masked_depth,
        cmap=color_map,
        vmin=float(valid.min()),
        vmax=float(valid.max()),
    )
    axes[1].set_title("Metric depth (m)")
    axes[1].axis("off")
    figure.colorbar(depth_image, ax=axes[1], label="Depth (m)", fraction=0.046)
    figure.savefig(rgb_depth_figure_path, dpi=160)
    plt.close(figure)
