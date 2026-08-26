from pathlib import Path

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


class CameraCapture:
    def __init__(self):
        self.camera_serial = None
        self.output_dir = Path("output")
        self.raw_dir = self.output_dir / "raw"
        self.width = 1280
        self.height = 720
        self.fps = 6
        self.warmup_frames = 60
        self.capture_frames = 15
        self.use_spatial_filter = True
        self.use_temporal_filter = True
        self.use_disparity_filter = True
        self.use_hole_filling_filter = True
        self.spatial_filter_magnitude = 2
        self.spatial_filter_smooth_alpha = 0.5
        self.spatial_filter_smooth_delta = 20
        self.temporal_filter_smooth_alpha = 0.4
        self.temporal_filter_smooth_delta = 20
        self.hole_filling_mode = 0

    def capture_rgbd(self):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.camera_serial:
            config.enable_device(self.camera_serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_filters = self.create_depth_filters()

        try:
            for _ in range(self.warmup_frames):
                frames = align.process(pipeline.wait_for_frames())
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    self.filter_depth_frame(depth_frame, depth_filters)

            rgb = None
            unfiltered_depth_frames = []
            filtered_depth_frames = []
            filtered_depth_frame = None
            for _ in range(self.capture_frames):
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError("RealSense did not return both color and depth frames.")

                unfiltered_depth_frames.append(np.asanyarray(depth_frame.get_data()).copy())
                filtered_depth_frame = self.filter_depth_frame(depth_frame, depth_filters)
                rgb = np.asanyarray(color_frame.get_data()).copy()
                filtered_depth_frames.append(np.asanyarray(filtered_depth_frame.get_data()).copy())

            unfiltered_depth = self.average_depth_frames(unfiltered_depth_frames)
            depth = self.average_depth_frames(filtered_depth_frames)
            depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
            intrinsics = self.intrinsics_to_dict(
                filtered_depth_frame.profile.as_video_stream_profile().intrinsics
            )
            return rgb, depth, depth_scale_m, intrinsics, unfiltered_depth
        finally:
            pipeline.stop()

    def save_raw_capture(self, rgb, depth, depth_scale_m, intrinsics=None, unfiltered_depth=None):
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = self.raw_dir / "RGB.png"
        depth_path = self.raw_dir / "depth_data.npz"
        unfiltered_depth_path = self.raw_dir / "depth_unfiltered_data.npz"

        o3d.io.write_image(str(rgb_path), o3d.geometry.Image(np.ascontiguousarray(rgb)))
        if unfiltered_depth is not None:
            np.savez(
                unfiltered_depth_path,
                depth_data=np.ascontiguousarray(unfiltered_depth),
                depth_scale_m=float(depth_scale_m),
                camera_intrinsics=np.asarray(
                    self.intrinsics_to_array(intrinsics), dtype=float
                ),
                depth_frame_count=int(self.capture_frames),
                processing_stage="before_depth_filtering",
            )

        np.savez(
            depth_path,
            depth_data=np.ascontiguousarray(depth),
            depth_scale_m=float(depth_scale_m),
            camera_intrinsics=np.asarray(
                self.intrinsics_to_array(intrinsics), dtype=float
            ),
            depth_frame_count=int(self.capture_frames),
            spatial_filter_enabled=bool(self.use_spatial_filter),
            temporal_filter_enabled=bool(self.use_temporal_filter),
            disparity_filter_enabled=bool(self.use_disparity_filter),
            hole_filling_filter_enabled=bool(self.use_hole_filling_filter),
            processing_stage="after_depth_filtering",
        )

        return rgb_path, depth_path, unfiltered_depth_path

    def run(self):
        rgb, depth, depth_scale_m, intrinsics, unfiltered_depth = self.capture_rgbd()
        return self.save_raw_capture(
            rgb,
            depth,
            depth_scale_m,
            intrinsics,
            unfiltered_depth,
        )

    @staticmethod
    def intrinsics_to_dict(intrinsics):
        return {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
        }

    @staticmethod
    def intrinsics_to_array(intrinsics):
        if intrinsics is None:
            return [0, 0, 0, 0, 0, 0]
        return [
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
        ]

    def create_depth_filters(self):
        filters = {
            "depth_to_disparity": rs.disparity_transform(True)
            if self.use_disparity_filter
            else None,
            "spatial": rs.spatial_filter() if self.use_spatial_filter else None,
            "temporal": rs.temporal_filter() if self.use_temporal_filter else None,
            "hole_filling": rs.hole_filling_filter()
            if self.use_hole_filling_filter
            else None,
            "disparity_to_depth": rs.disparity_transform(False)
            if self.use_disparity_filter
            else None,
        }

        if filters["spatial"] is not None:
            self.set_filter_option(
                filters["spatial"],
                rs.option.filter_magnitude,
                self.spatial_filter_magnitude,
            )
            self.set_filter_option(
                filters["spatial"],
                rs.option.filter_smooth_alpha,
                self.spatial_filter_smooth_alpha,
            )
            self.set_filter_option(
                filters["spatial"],
                rs.option.filter_smooth_delta,
                self.spatial_filter_smooth_delta,
            )

        if filters["temporal"] is not None:
            self.set_filter_option(
                filters["temporal"],
                rs.option.filter_smooth_alpha,
                self.temporal_filter_smooth_alpha,
            )
            self.set_filter_option(
                filters["temporal"],
                rs.option.filter_smooth_delta,
                self.temporal_filter_smooth_delta,
            )

        if filters["hole_filling"] is not None:
            self.set_filter_option(
                filters["hole_filling"],
                rs.option.holes_fill,
                self.hole_filling_mode,
            )

        return filters

    @staticmethod
    def filter_depth_frame(depth_frame, depth_filters):
        filtered = depth_frame
        for filter_name in [
            "depth_to_disparity",
            "spatial",
            "temporal",
            "hole_filling",
            "disparity_to_depth",
        ]:
            depth_filter = depth_filters.get(filter_name)
            if depth_filter is not None:
                filtered = depth_filter.process(filtered)
        return filtered.as_depth_frame()

    @staticmethod
    def average_depth_frames(depth_frames):
        if not depth_frames:
            raise RuntimeError("No depth frames were captured for averaging.")

        stack = np.stack(depth_frames).astype(np.float32)
        valid = stack > 0
        counts = valid.sum(axis=0)
        summed = np.where(valid, stack, 0).sum(axis=0)
        averaged = np.zeros(stack.shape[1:], dtype=np.uint16)
        nonzero = counts > 0
        averaged[nonzero] = np.round(summed[nonzero] / counts[nonzero]).astype(np.uint16)
        return averaged

    @staticmethod
    def set_filter_option(depth_filter, option, value):
        if option in depth_filter.get_supported_options():
            depth_filter.set_option(option, value)


if __name__ == "__main__":
    camera = CameraCapture()
    rgb_path, depth_path, unfiltered_depth_path = camera.run()
    print(f"Saved RGB image: {rgb_path}")
    print(f"Saved unfiltered depth data: {unfiltered_depth_path}")
    print(f"Saved depth data: {depth_path}")
