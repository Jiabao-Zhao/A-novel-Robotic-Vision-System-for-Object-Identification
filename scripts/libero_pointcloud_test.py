import json
from pathlib import Path

import numpy as np

from simulation.pointcloud import create_open3d_pointcloud, save_pointcloud


OUTPUT_DIR = Path("outputs/simulation/libero_sample")
RGB_PATH = OUTPUT_DIR / "rgb.png"
DEPTH_PATH = OUTPUT_DIR / "depth.npy"
INTRINSICS_PATH = OUTPUT_DIR / "intrinsics.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
POINTCLOUD_PATH = OUTPUT_DIR / "pointcloud.ply"
DEPTH_TRUNC_M = 10.0
OPEN3D_VISUALIZATION = False


def main():
    missing = [
        path for path in (RGB_PATH, DEPTH_PATH, INTRINSICS_PATH, METADATA_PATH) if not path.is_file()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(
            f"Missing captured RGB-D inputs: {names}. Run `python -m scripts.capture_libero_rgbd` first."
        )

    try:
        import cv2
    except ModuleNotFoundError as error:
        raise SystemExit("OpenCV is required to load rgb.png.") from error

    bgr = cv2.imread(str(RGB_PATH), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"OpenCV could not read {RGB_PATH}.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_m = np.load(DEPTH_PATH)
    intrinsics = np.load(INTRINSICS_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    cloud, valid_pixels = create_open3d_pointcloud(
        rgb,
        depth_m,
        intrinsics,
        depth_trunc_m=DEPTH_TRUNC_M,
    )
    report = save_pointcloud(cloud, POINTCLOUD_PATH)

    print(f"Instruction: {metadata['instruction']}")
    print(f"Valid RGB-D pixels below {DEPTH_TRUNC_M:g} m: {valid_pixels}")
    print(f"Generated 3D points: {report['point_count']}")
    print("Point-cloud XYZ bounds in the camera frame (m):")
    for index, axis in enumerate("xyz"):
        print(
            f"  {axis}: [{report['minimum_xyz_m'][index]:.6f}, "
            f"{report['maximum_xyz_m'][index]:.6f}]"
        )
    print(f"Saved point cloud: {report['path']}")
    print("Source confirmation: rendered metric depth + RGB + camera intrinsics only")

    if OPEN3D_VISUALIZATION:
        try:
            import open3d as o3d

            o3d.visualization.draw_geometries([cloud])
        except RuntimeError as error:
            print(f"Open3D GUI unavailable; the PLY file is still valid: {error}")


if __name__ == "__main__":
    main()
