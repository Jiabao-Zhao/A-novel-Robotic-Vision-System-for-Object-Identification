import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


CAMERA_SERIAL = "103422070738"
OUTPUT_DIR = Path("output")
WIDTH = 640
HEIGHT = 480
FPS = 30
WARMUP_FRAMES = 30


def capture_rgbd():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(CAMERA_SERIAL)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        frames = None
        for _ in range(WARMUP_FRAMES + 1):
            frames = align.process(pipeline.wait_for_frames())

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not return both color and depth frames.")

        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        return rgb, depth, depth_scale_m
    finally:
        pipeline.stop()


def save_raw_capture(rgb, depth, depth_scale_m, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = output_dir / "rgb.raw"
    depth_path = output_dir / "depth.raw"
    metadata_path = output_dir / "camera_capture_metadata.json"

    np.ascontiguousarray(rgb).tofile(rgb_path)
    np.ascontiguousarray(depth).tofile(depth_path)

    metadata = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "camera_serial": CAMERA_SERIAL,
        "aligned_to": "color",
        "rgb": {
            "path": str(rgb_path),
            "shape": list(rgb.shape),
            "dtype": str(rgb.dtype),
            "format": "rgb8",
        },
        "depth": {
            "path": str(depth_path),
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "format": "z16",
            "scale_m_per_unit": float(depth_scale_m),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return rgb_path, depth_path, metadata_path


def main():
    rgb, depth, depth_scale_m = capture_rgbd()
    rgb_path, depth_path, metadata_path = save_raw_capture(rgb, depth, depth_scale_m)
    print(f"Saved RGB raw data: {rgb_path}")
    print(f"Saved depth raw data: {depth_path}")
    print(f"Saved capture metadata: {metadata_path}")


if __name__ == "__main__":
    main()
