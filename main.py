from pathlib import Path

from point_cloud_localization import PointCloudLocalization
from vlm_module import classify_from_localization


OPEN_3D_VISUALIZATION = False
USE_SAVED_RAW_CAPTURE = False
USER_TEXT = "find the cable shark device and USB-D connector"
SAVED_RGB_PATH = Path("output/raw/RGB.png")
SAVED_DEPTH_PATH = Path("output/raw/depth_data.npz")


def run_pipeline():
    localizer = PointCloudLocalization()

    if USE_SAVED_RAW_CAPTURE:
        return localizer.run_from_saved_raw(
            rgb_path=SAVED_RGB_PATH,
            depth_path=SAVED_DEPTH_PATH,
            visualize=OPEN_3D_VISUALIZATION,
        )

    return localizer.run(visualize=OPEN_3D_VISUALIZATION)


def main():
    paths = run_pipeline()

    print(f"Saved raw RGB image: {paths['rgb']}")
    print(f"Saved raw depth data: {paths['depth']}")
    print(f"Detected object count: {paths['object_count']}")
    print(f"Saved localization JSON: {paths['localization']}")
    print(f"Saved annotated RGB image: {paths['annotated_rgb']}")
    print(f"Saved workspace point cloud: {paths['workspace_cloud']}")
    print(f"Saved table point cloud: {paths['table_cloud']}")
    print(f"Saved segmented point cloud: {paths['segmented_cloud']}")

    vlm_path = classify_from_localization(
        user_text=USER_TEXT,
        image_path=paths["annotated_rgb"],
        localization_path=paths["localization"],
    )
    print(f"Saved VLM result: {vlm_path}")


if __name__ == "__main__":
    main()
