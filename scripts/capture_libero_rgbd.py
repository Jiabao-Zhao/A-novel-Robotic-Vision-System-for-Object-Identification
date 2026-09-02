from pathlib import Path

from simulation.libero_env import LiberoIntegrationError, LiberoTaskEnvironment
from simulation.libero_io import save_libero_observation
from simulation.libero_sensor import LiberoRGBDSensor, depth_statistics
from simulation.perception_adapter import pointcloud_localization_inputs


SUITE_NAME = "libero_object"
TASK_INDEX = 0
CAMERA_NAME = "agentview"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OUTPUT_DIR = Path("outputs/simulation/libero_sample")


def main():
    with LiberoTaskEnvironment(
        suite_name=SUITE_NAME,
        task_index=TASK_INDEX,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ) as environment:
        raw_observation = environment.reset()
        observation = LiberoRGBDSensor(environment, CAMERA_NAME).capture(raw_observation)
        paths = save_libero_observation(observation, environment, OUTPUT_DIR)

    statistics = depth_statistics(observation.depth_m)
    adapter_inputs = pointcloud_localization_inputs(
        observation,
        rgb_path=paths["rgb"],
        depth_path=paths["depth"],
    )
    print(f"Saved LIBERO RGB-D observation to: {OUTPUT_DIR.resolve()}")
    print(f"Instruction: {observation.instruction}")
    print(f"Image resolution: {observation.rgb.shape[1]} x {observation.rgb.shape[0]}")
    print(f"Minimum valid depth: {statistics['minimum_m']:.6f} m")
    print(f"Maximum valid depth: {statistics['maximum_m']:.6f} m")
    print(f"Median valid depth: {statistics['median_m']:.6f} m")
    print("Camera intrinsics K:")
    print(observation.intrinsics)
    print("Saved files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print("Existing perception compatibility:")
    print("  entry point: PointCloudLocalization.run_from_arrays(**inputs)")
    print(f"  depth_scale_m: {adapter_inputs['depth_scale_m']} (depth is already in metres)")
    print(f"  camera_intrinsics: {adapter_inputs['camera_intrinsics']}")
    print("  downstream perception was not executed")


if __name__ == "__main__":
    try:
        main()
    except LiberoIntegrationError as error:
        raise SystemExit(f"LIBERO RGB-D capture failed: {error}") from error
