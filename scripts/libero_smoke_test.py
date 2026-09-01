import numpy as np

from simulation.libero_env import LiberoIntegrationError, LiberoTaskEnvironment, observation_lines
from simulation.libero_sensor import LiberoRGBDSensor, depth_statistics


SUITE_NAME = "libero_object"
TASK_INDEX = 0
CAMERA_NAME = "agentview"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def main():
    with LiberoTaskEnvironment(
        suite_name=SUITE_NAME,
        task_index=TASK_INDEX,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ) as environment:
        print(f"LIBERO suite: {environment.suite_name}")
        print(f"Task index: {environment.task_index}")
        print(f"Task: {environment.task_name}")
        print(f"Instruction: {environment.instruction}")
        print(f"MuJoCo rendering backend: {environment.rendering_backend}")

        raw_observation = environment.reset()
        print("Observation keys:")
        for line in observation_lines(raw_observation):
            print(line)

        observation = LiberoRGBDSensor(environment, CAMERA_NAME).capture(raw_observation)
        statistics = depth_statistics(observation.depth_m)
        transform_error = float(
            np.max(np.abs(observation.world_T_camera @ observation.camera_T_world - np.eye(4)))
        )

        print(f"Selected RGB key: {observation.rgb_key}")
        print(f"Selected depth key: {observation.depth_key}")
        print(f"RGB: shape={observation.rgb.shape}, dtype={observation.rgb.dtype}")
        print(f"Metric depth: shape={observation.depth_m.shape}, dtype={observation.depth_m.dtype}")
        print(f"Metric depth range (m): {statistics['minimum_m']:.6f} to {statistics['maximum_m']:.6f}")
        print(f"Metric depth median (m): {statistics['median_m']:.6f}")
        print("Camera intrinsics K:")
        print(observation.intrinsics)
        print("world_T_camera (camera pose; camera-frame points -> world frame):")
        print(observation.world_T_camera)
        print("camera_T_world (world-frame points -> camera frame):")
        print(observation.camera_T_world)
        print(f"Transform inverse max error: {transform_error:.3e}")
        print(f"Robot-state keys: {', '.join(sorted(observation.robot_state))}")


if __name__ == "__main__":
    try:
        main()
    except LiberoIntegrationError as error:
        raise SystemExit(f"LIBERO smoke test failed: {error}") from error
