from pathlib import Path

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


class CameraCapture:
    def __init__(self):
        self.camera_serial = None
        self.output_dir = Path("output")
        self.raw_dir = self.output_dir / "raw"
        self.width = 640
        self.height = 480
        self.fps = 30
        self.warmup_frames = 30

    def capture_rgbd(self):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.camera_serial:
            config.enable_device(self.camera_serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            frames = None
            for _ in range(self.warmup_frames + 1):
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

    def save_raw_capture(self, rgb, depth, depth_scale_m):
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = self.raw_dir / "RGB.png"
        depth_path = self.raw_dir / "depth_data.npz"

        o3d.io.write_image(str(rgb_path), o3d.geometry.Image(np.ascontiguousarray(rgb)))
        np.savez(
            depth_path,
            depth_data=np.ascontiguousarray(depth),
            depth_scale_m=float(depth_scale_m),
        )

        return rgb_path, depth_path

    def run(self):
        rgb, depth, depth_scale_m = self.capture_rgbd()
        return self.save_raw_capture(rgb, depth, depth_scale_m)


if __name__ == "__main__":
    camera = CameraCapture()
    rgb_path, depth_path = camera.run()
    print(f"Saved RGB image: {rgb_path}")
    print(f"Saved depth data: {depth_path}")
