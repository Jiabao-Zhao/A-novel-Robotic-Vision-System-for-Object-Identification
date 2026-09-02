from pathlib import Path

import numpy as np

from simulation.libero_env import LiberoIntegrationError, LiberoTaskEnvironment
from simulation.libero_io import save_libero_observation
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.perception_adapter import (
    LIBERO_WORKSPACE_MAX_XYZ_M,
    LIBERO_WORKSPACE_MIN_XYZ_M,
    run_libero_localization,
)


SUITE_NAME = "libero_object"
TASK_INDEX = 0
CAMERA_NAME = "agentview"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OUTPUT_ROOT = Path("outputs/simulation/libero_localization")


def main():
    capture_dir = OUTPUT_ROOT / "capture"
    with LiberoTaskEnvironment(
        suite_name=SUITE_NAME,
        task_index=TASK_INDEX,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ) as environment:
        raw_observation = environment.reset()
        observation = LiberoRGBDSensor(environment, CAMERA_NAME).capture(raw_observation)
        capture_paths = save_libero_observation(observation, environment, capture_dir)

    payload, paths, workspace_mask = run_libero_localization(
        observation,
        rgb_path=capture_paths["rgb"],
        output_root=OUTPUT_ROOT,
    )

    print(f"LIBERO suite: {SUITE_NAME}")
    print(f"Task: {environment.task_name}")
    print(f"Instruction: {observation.instruction}")
    print(f"World workspace min XYZ (m): {LIBERO_WORKSPACE_MIN_XYZ_M}")
    print(f"World workspace max XYZ (m): {LIBERO_WORKSPACE_MAX_XYZ_M}")
    print(f"Workspace RGB-D pixels: {int(workspace_mask.sum())}")
    print(f"Localized object candidates: {payload['object_count']}")
    print(f"Plane model in camera frame [a, b, c, d]: {payload['plane_model']}")
    for item in payload["objects"]:
        centroid_camera = np.asarray([*item["centroid_3d_m"], 1.0])
        centroid_world = (observation.world_T_camera @ centroid_camera)[:3]
        print(
            f"  {item['object_id']}: roi={item['roi']}, "
            f"centroid_camera_m={np.round(centroid_camera[:3], 4).tolist()}, "
            f"centroid_world_m={np.round(centroid_world, 4).tolist()}, "
            f"size_camera_aabb_m={np.round(item['size_3d_m'], 4).tolist()}, "
            f"downsampled_points={item['point_count']}, "
            f"saved_points={item['saved_point_count']}"
        )
    print(f"Annotated RGB: {paths['annotated_rgb']}")
    print(f"Localization JSON: {paths['localization']}")
    print("Perception inputs: rendered RGB + metric depth + camera intrinsics only")
    print("VLM, CAD registration, robot control, and simulator ground truth were not used")


if __name__ == "__main__":
    try:
        main()
    except LiberoIntegrationError as error:
        raise SystemExit(f"LIBERO localization test failed: {error}") from error
