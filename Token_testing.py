#!/usr/bin/env python3
"""
Standalone real-camera mixed-resolution/foveated VLM token-consumption test.

This file intentionally does NOT import perception.vision_controller or any robot modules.
Everything needed for the test is inside this file:
  - RealSense RGB-D capture
  - depth-plane / histogram object mask localization only
  - candidate extraction and annotation
  - image-pyramid foveated raster generation
  - required OpenAI/Gemini VLM calls
  - token usage printing

Default behavior:
  - captures one aligned RGB-D frame from the RealSense camera
  - creates a binary object mask using depth localization
  - keeps mask pixels high-resolution and makes the background low-resolution/blurred
  - sends original and mixed-resolution images to the selected VLM provider(s)
  - prints token usage in the final summary

Example Windows commands:
  D:\Python_New\python.exe Token_testing_camera_standalone.py --provider gemini
  D:\Python_New\python.exe Token_testing_camera_standalone.py --provider openai --openai-detail high
  D:\Python_New\python.exe Token_testing_camera_standalone.py --provider gemini,openai --scales 1.0,0.5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROBOT_CAMERA_SERIAL = "103422070738"

DEFAULT_DESCRIBE_PROMPT = (
    "Briefly describe this real robot workspace image in 1-2 sentences. "
    "Mention whether you see a small green block or green object. "
    "Mention whether the background appears sharp or low-resolution/blurred."
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_csv_floats(text: str) -> List[float]:
    values: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values or [1.0]


def parse_csv_ints(text: str) -> Tuple[int, ...]:
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Resolution factors must be positive integers.")
        values.append(value)
    values = sorted(set(values))
    if 1 not in values:
        values.insert(0, 1)
    return tuple(values)


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_png(path: Path, image_bgr_or_gray: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr_or_gray)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def object_to_plain(value: Any) -> Any:
    """Best-effort conversion of SDK response objects to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): object_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_plain(v) for v in value]
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return object_to_plain(method())
            except Exception:
                pass
    out: Dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if isinstance(attr, (str, int, float, bool, type(None), list, tuple, dict)):
            out[name] = object_to_plain(attr)
    return out or str(value)


def print_json_block(title: str, payload: Any) -> None:
    print(f"\n{title}")
    print(json.dumps(object_to_plain(payload), indent=2, sort_keys=True))


# -----------------------------------------------------------------------------
# RealSense capture: standalone copy, no external perception module import
# -----------------------------------------------------------------------------


def capture_realsense_rgbd(
    serial: str = ROBOT_CAMERA_SERIAL,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 30,
    use_filters: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Capture one aligned RGB-D frame from RealSense.

    Returns:
      rgb_bgr: HxWx3 uint8 BGR image aligned to color stream
      depth_mm: HxW uint16 depth map in millimeters aligned to color stream
      intrinsics: fx, fy, cx, cy from the color stream
    """
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        raise RuntimeError(
            "pyrealsense2 is required for real camera capture. Install Intel RealSense SDK/python package."
        ) from exc

    pipe = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
    }

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal = rs.temporal_filter()
    hole_filler = rs.hole_filling_filter()
    hole_filler.set_option(rs.option.holes_fill, 0)

    depth_to_disp = rs.disparity_transform(True)
    disp_to_depth = rs.disparity_transform(False)

    try:
        color_frame = None
        depth_frame = None
        warmup_frames = max(1, int(warmup_frames))
        for _ in range(warmup_frames):
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

        if color_frame is None or depth_frame is None:
            raise RuntimeError("RealSense returned no color/depth frame.")

        filtered = depth_frame
        if use_filters:
            filtered = depth_to_disp.process(filtered)
            filtered = spatial.process(filtered)
            filtered = temporal.process(filtered)
            filtered = hole_filler.process(filtered)
            filtered = disp_to_depth.process(filtered)

        rgb_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_mm = np.asanyarray(filtered.get_data()).copy()
    finally:
        pipe.stop()

    return rgb_bgr, depth_mm, intrinsics


# -----------------------------------------------------------------------------
# Depth localization copied into standalone file
# -----------------------------------------------------------------------------


class StandaloneDepthLocalizer:
    def __init__(self, intrinsics: Optional[Dict[str, float]] = None):
        intrinsics = intrinsics or {}
        self.fx = float(intrinsics.get("fx", 623.9816462749620))
        self.fy = float(intrinsics.get("fy", 613.8080113982506))
        self.cx = float(intrinsics.get("cx", 318.5163260449835))
        self.cy = float(intrinsics.get("cy", 237.5512378918142))

        # Same style as the original vision controller settings.
        self.min_object_height_mm = 5.5
        self.bin_size = 3
        self.roi_margin_top = 20
        self.roi_margin_bottom = 50
        self.roi_margin_left = 20
        self.roi_margin_right = 20
        self.min_contour_area = 500
        self.max_contour_area = 80000
        self.max_aspect_ratio = 5.0
        self.morph_kernel_size = 3
        self.fragment_merge_gap_px = 45
        self.nearby_fragment_merge_gap_px = 12
        self.nearby_fragment_area_threshold = 2000
        self.edge_artifact_margin_px = 2
        self.edge_artifact_max_area = 2500

        self.use_plane_localization = True
        self.plane_sample_stride = 4
        self.plane_ransac_iterations = 180
        self.plane_inlier_threshold_mm = 4.0
        self.plane_min_inlier_ratio = 0.35
        self.plane_min_sample_points = 250

        self.grasp_open_clearance_mm = 30.0
        self.grasp_close_margin_mm = 2.0

        self._last_binary_mask: Optional[np.ndarray] = None
        self._last_localization_debug: Dict[str, Any] = {}
        self._last_candidate_debug: List[Dict[str, Any]] = []
        self._last_object_height_map: Optional[np.ndarray] = None
        self._last_plane_depth_map: Optional[np.ndarray] = None

    @staticmethod
    def _valid_depth_mask(depth_data: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        mask = np.isfinite(depth_data) & (depth_data > 0)
        if roi_mask is not None:
            mask = mask & (roi_mask == 1)
        return mask

    def pixel_to_camera(self, u: float, v: float, z: float) -> np.ndarray:
        x = (float(u) - self.cx) * float(z) / self.fx
        y = (float(v) - self.cy) * float(z) / self.fy
        return np.array([x, y, z], dtype=float)

    def localize(self, depth_data: np.ndarray) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
        binary_mask, surface_depth = self._create_binary_mask(depth_data)
        candidates = self._find_candidates(binary_mask, depth_data, surface_depth)
        return candidates, surface_depth, binary_mask

    def _create_binary_mask(self, depth_data: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = depth_data.shape
        self._last_object_height_map = None
        self._last_plane_depth_map = None

        roi_mask = np.zeros_like(depth_data, dtype=np.uint8)
        roi_bottom = height - self.roi_margin_bottom if self.roi_margin_bottom else height
        roi_right = width - self.roi_margin_right if self.roi_margin_right else width
        roi_mask[self.roi_margin_top:roi_bottom, self.roi_margin_left:roi_right] = 1

        valid_roi_mask = self._valid_depth_mask(depth_data, roi_mask=roi_mask)
        valid = depth_data[valid_roi_mask].astype(np.float32)
        if len(valid) == 0:
            binary_mask = np.zeros(depth_data.shape, dtype=np.uint8)
            self._last_binary_mask = binary_mask
            self._last_localization_debug = {
                "valid_depth_count": 0,
                "localization_method": "none",
                "surface_depth_mm": 0.0,
                "object_threshold_mm": 0.0,
                "object_pixel_count": 0,
            }
            return binary_mask, 0.0

        plane_result = None
        if self.use_plane_localization:
            plane_result = self._fit_table_plane_depth_map(
                depth_data=depth_data,
                roi_mask=roi_mask,
                valid_mask=valid_roi_mask,
            )

        if plane_result is not None:
            plane_depth_map, plane_debug = plane_result
            height_map = plane_depth_map - depth_data.astype(np.float32)
            valid_mask = valid_roi_mask & np.isfinite(plane_depth_map)
            object_mask = (height_map >= self.min_object_height_mm) & valid_mask
            binary_mask = object_mask.astype(np.uint8)

            if np.sum(binary_mask) > 0:
                k = self.morph_kernel_size
                kernel = np.ones((k, k), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            surface_depth = float(np.median(plane_depth_map[valid_mask])) if np.any(valid_mask) else 0.0
            self._last_binary_mask = binary_mask
            self._last_object_height_map = height_map
            self._last_plane_depth_map = plane_depth_map
            self._last_localization_debug = {
                "valid_depth_count": int(len(valid)),
                "valid_depth_min_mm": float(np.min(valid)),
                "valid_depth_max_mm": float(np.max(valid)),
                "valid_depth_filter": "finite_positive_depth",
                "localization_method": "ransac_plane_height",
                "surface_depth_mm": surface_depth,
                "object_threshold_mm": float(self.min_object_height_mm),
                "object_pixel_count": int(np.sum(binary_mask)),
                "min_object_height_mm": float(self.min_object_height_mm),
                "min_contour_area": int(self.min_contour_area),
                "max_contour_area": int(self.max_contour_area),
                "max_aspect_ratio": float(self.max_aspect_ratio),
                **plane_debug,
            }
            return binary_mask, surface_depth

        data_range = float(np.max(valid) - np.min(valid))
        num_bins = max(1, int(np.ceil(data_range / self.bin_size)))
        hist, bin_edges = np.histogram(valid, bins=num_bins)

        max_bin = int(np.argmax(hist))
        surface_depth = float(bin_edges[min(max_bin + 1, len(bin_edges) - 1)])
        threshold = surface_depth - self.min_object_height_mm
        valid_mask = self._valid_depth_mask(depth_data, roi_mask=roi_mask)
        object_mask = (depth_data < threshold) & valid_mask
        binary_mask = object_mask.astype(np.uint8)

        if np.sum(binary_mask) > 0:
            k = self.morph_kernel_size
            kernel = np.ones((k, k), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        self._last_binary_mask = binary_mask
        self._last_localization_debug = {
            "valid_depth_count": int(len(valid)),
            "valid_depth_min_mm": float(np.min(valid)),
            "valid_depth_max_mm": float(np.max(valid)),
            "valid_depth_filter": "finite_positive_depth",
            "localization_method": "global_depth_histogram",
            "surface_depth_mm": float(surface_depth),
            "object_threshold_mm": float(threshold),
            "object_pixel_count": int(np.sum(binary_mask)),
            "min_object_height_mm": float(self.min_object_height_mm),
            "min_contour_area": int(self.min_contour_area),
            "max_contour_area": int(self.max_contour_area),
            "max_aspect_ratio": float(self.max_aspect_ratio),
        }
        return binary_mask, surface_depth

    def _fit_table_plane_depth_map(
        self,
        depth_data: np.ndarray,
        roi_mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        ys, xs = np.where(valid_mask)
        min_points = int(self.plane_min_sample_points)
        if xs.size < min_points:
            return None

        stride = max(1, int(self.plane_sample_stride))
        sampled = (ys % stride == 0) & (xs % stride == 0)
        xs_sample = xs[sampled]
        ys_sample = ys[sampled]
        if xs_sample.size < min_points:
            xs_sample = xs
            ys_sample = ys

        z_sample = depth_data[ys_sample, xs_sample].astype(np.float32)
        points = self._pixels_to_camera_points(xs_sample, ys_sample, z_sample)
        if points.shape[0] < min_points:
            return None

        rng = np.random.default_rng(7)
        best_plane = None
        best_inliers = None
        best_count = 0
        iterations = max(1, int(self.plane_ransac_iterations))
        threshold = float(self.plane_inlier_threshold_mm)

        for _ in range(iterations):
            idx = rng.choice(points.shape[0], size=3, replace=False)
            p1, p2, p3 = points[idx]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, p1))
            distances = np.abs(points @ normal + d)
            inliers = distances < threshold
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_plane = (normal, d)

        if best_plane is None or best_inliers is None:
            return None

        inlier_ratio = best_count / float(points.shape[0])
        if inlier_ratio < float(self.plane_min_inlier_ratio):
            return None

        inlier_points = points[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vh[-1]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return None
        normal = normal / normal_norm
        d = -float(np.dot(normal, centroid))

        all_y, all_x = np.indices(depth_data.shape)
        ray_x = (all_x.astype(np.float32) - self.cx) / self.fx
        ray_y = (all_y.astype(np.float32) - self.cy) / self.fy
        denom = normal[0] * ray_x + normal[1] * ray_y + normal[2]

        plane_depth = np.full(depth_data.shape, np.nan, dtype=np.float32)
        usable = np.abs(denom) > 1e-6
        plane_depth[usable] = (-d / denom[usable]).astype(np.float32)
        plane_depth[(roi_mask != 1) | (plane_depth <= 0)] = np.nan

        residuals = np.abs(inlier_points @ normal + d)
        debug = {
            "plane_inlier_count": int(best_count),
            "plane_sample_count": int(points.shape[0]),
            "plane_inlier_ratio": float(inlier_ratio),
            "plane_residual_median_mm": float(np.median(residuals)),
            "plane_residual_p90_mm": float(np.percentile(residuals, 90)),
            "plane_normal": [float(v) for v in normal],
        }
        return plane_depth, debug

    def _pixels_to_camera_points(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = (u.astype(np.float32) - self.cx) * z / self.fx
        y = (v.astype(np.float32) - self.cy) * z / self.fy
        return np.column_stack([x, y, z]).astype(np.float32)

    def _grasp_width_info_from_contour(self, contour: np.ndarray, depth_mm: float) -> Dict[str, Any]:
        short_side, long_side = self._metric_rect_dimensions_mm(contour, depth_mm)
        if short_side is None:
            return {}
        open_width = min(110.0, max(1.0, float(short_side) + self.grasp_open_clearance_mm))
        close_width = min(open_width, max(1.0, float(short_side) - self.grasp_close_margin_mm))
        return {
            "grasp_width_mm": round(float(short_side), 1),
            "object_length_mm": round(float(long_side), 1),
            "recommended_open_width_mm": round(float(open_width), 1),
            "recommended_close_width_mm": round(float(close_width), 1),
            "grasp_width_source": "depth_mask_min_area_rect",
        }

    def _metric_rect_dimensions_mm(self, contour: np.ndarray, depth_mm: float) -> Tuple[Optional[float], Optional[float]]:
        try:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            depth = float(depth_mm)
            if not np.isfinite(depth) or depth <= 0:
                return None, None
            points = np.asarray([self.pixel_to_camera(float(u), float(v), depth) for u, v in box], dtype=float)
            edge_lengths = [float(np.linalg.norm(points[(i + 1) % 4] - points[i])) for i in range(4)]
            side_a = (edge_lengths[0] + edge_lengths[2]) / 2.0
            side_b = (edge_lengths[1] + edge_lengths[3]) / 2.0
            if not all(np.isfinite(value) and value > 0 for value in (side_a, side_b)):
                return None, None
            return min(side_a, side_b), max(side_a, side_b)
        except Exception:
            return None, None

    def _find_candidates(self, binary_mask: np.ndarray, depth_data: np.ndarray, surface_depth: float) -> List[Dict[str, Any]]:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = self._merge_fragmented_contours(contours)
        candidates: List[Dict[str, Any]] = []
        debug_entries: List[Dict[str, Any]] = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            entry: Dict[str, Any] = {
                "id": i,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "area": float(area),
                "aspect": float(aspect),
                "accepted": False,
                "reason": "",
            }
            if area < self.min_contour_area or area > self.max_contour_area:
                entry["reason"] = "area_out_of_range"
                debug_entries.append(entry)
                continue

            if aspect > self.max_aspect_ratio:
                entry["reason"] = "aspect_ratio_too_large"
                debug_entries.append(entry)
                continue

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            obj_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 1, -1)
            valid_obj_mask = (obj_mask == 1) & self._valid_depth_mask(depth_data)
            obj_depths = depth_data[valid_obj_mask]
            if obj_depths.size == 0:
                entry["reason"] = "no_valid_depth"
                debug_entries.append(entry)
                continue

            top_depth = float(np.median(obj_depths))
            height_map = self._last_object_height_map
            if height_map is not None:
                obj_heights = height_map[valid_obj_mask]
                obj_heights = obj_heights[np.isfinite(obj_heights)]
                obj_heights = obj_heights[obj_heights > 0]
                if obj_heights.size == 0:
                    entry["reason"] = "no_valid_height"
                    debug_entries.append(entry)
                    continue
                obj_height = float(np.percentile(obj_heights, 70))
                plane_depth_map = self._last_plane_depth_map
                if plane_depth_map is not None:
                    plane_depths = plane_depth_map[valid_obj_mask]
                    plane_depths = plane_depths[np.isfinite(plane_depths)]
                    table_depth = float(np.median(plane_depths)) if plane_depths.size > 0 else top_depth + obj_height
                else:
                    table_depth = top_depth + obj_height
            else:
                obj_height = float(surface_depth) - top_depth
                table_depth = float(surface_depth)

            center_depth = top_depth + obj_height / 2.0
            z_mm = top_depth
            grasp_width_info = self._grasp_width_info_from_contour(contour, center_depth)

            entry.update(
                {
                    "cx": int(cx),
                    "cy": int(cy),
                    "median_depth_mm": float(top_depth),
                    "top_depth_mm": float(top_depth),
                    "center_depth_mm": float(center_depth),
                    "table_depth_mm": float(table_depth),
                    "object_height_mm": float(obj_height),
                    "z_mm": float(z_mm),
                    **grasp_width_info,
                }
            )

            if obj_height < self.min_object_height_mm:
                entry["reason"] = "object_height_too_low"
                debug_entries.append(entry)
                continue

            if self._is_boundary_artifact(x=x, y=y, w=w, h=h, mask_shape=binary_mask.shape, area=area):
                entry["reason"] = "boundary_artifact"
                debug_entries.append(entry)
                continue

            entry.update({"accepted": True, "reason": "accepted"})
            debug_entries.append(entry)
            candidates.append(
                {
                    "id": i,
                    "cx": int(cx),
                    "cy": int(cy),
                    "z_mm": round(float(z_mm), 1),
                    "top_depth_mm": round(float(top_depth), 1),
                    "center_depth_mm": round(float(center_depth), 1),
                    "table_depth_mm": round(float(table_depth), 1),
                    "object_height": round(float(obj_height), 1),
                    **grasp_width_info,
                    "area": int(area),
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "source": "depth",
                }
            )

        self._last_candidate_debug = debug_entries
        return candidates

    def _is_boundary_artifact(self, x: int, y: int, w: int, h: int, mask_shape: Tuple[int, ...], area: float) -> bool:
        height, width = mask_shape[:2]
        touches_right = x + w >= width - self.edge_artifact_margin_px
        touches_bottom = y + h >= height - self.edge_artifact_margin_px
        return bool((touches_right or touches_bottom) and area <= self.edge_artifact_max_area)

    def _merge_fragmented_contours(self, contours: Iterable[np.ndarray]) -> List[np.ndarray]:
        contours = list(contours)
        if not contours:
            return []

        contour_items = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            contour_items.append({"id": idx, "contour": contour, "bbox": (x, y, w, h), "area": float(area)})

        large_items = [item for item in contour_items if item["area"] >= self.min_contour_area]
        small_items = [item for item in contour_items if item["area"] < self.min_contour_area]
        if not small_items:
            return [item["contour"] for item in large_items]

        groups = []
        used = set()
        for index, _ in enumerate(small_items):
            if index in used:
                continue
            stack = [index]
            group = []
            used.add(index)
            while stack:
                current = stack.pop()
                current_item = small_items[current]
                group.append(current_item)
                for other_index, other_item in enumerate(small_items):
                    if other_index in used:
                        continue
                    if self._bbox_gap(current_item["bbox"], other_item["bbox"]) <= self.fragment_merge_gap_px:
                        used.add(other_index)
                        stack.append(other_index)
            groups.append(group)

        merged = [item["contour"] for item in large_items]
        for group in groups:
            if len(group) == 1:
                merged.append(group[0]["contour"])
                continue
            total_area = sum(item["area"] for item in group)
            if total_area < self.min_contour_area:
                merged.extend(item["contour"] for item in group)
                continue
            points = np.vstack([item["contour"] for item in group])
            merged.append(cv2.convexHull(points))

        return self._merge_nearby_small_contours(merged)

    def _merge_nearby_small_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        if not contours:
            return []

        contour_items = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            contour_items.append({"id": idx, "contour": contour, "bbox": (x, y, w, h), "area": float(cv2.contourArea(contour))})

        groups = []
        used = set()
        for index, _ in enumerate(contour_items):
            if index in used:
                continue
            stack = [index]
            group = []
            used.add(index)
            while stack:
                current = stack.pop()
                current_item = contour_items[current]
                group.append(current_item)
                for other_index, other_item in enumerate(contour_items):
                    if other_index in used:
                        continue
                    if (
                        current_item["area"] <= self.nearby_fragment_area_threshold
                        and other_item["area"] <= self.nearby_fragment_area_threshold
                        and self._bbox_gap(current_item["bbox"], other_item["bbox"]) <= self.nearby_fragment_merge_gap_px
                    ):
                        used.add(other_index)
                        stack.append(other_index)
            groups.append(group)

        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0]["contour"])
                continue
            points = np.vstack([item["contour"] for item in group])
            merged.append(cv2.convexHull(points))
        return merged

    @staticmethod
    def _bbox_gap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> int:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        a_left, a_top, a_right, a_bottom = ax, ay, ax + aw, ay + ah
        b_left, b_top, b_right, b_bottom = bx, by, bx + bw, by + bh
        gap_x = max(0, max(a_left - b_right, b_left - a_right))
        gap_y = max(0, max(a_top - b_bottom, b_top - a_bottom))
        return int(max(gap_x, gap_y))

    def depth_to_debug_image(self, depth_data: np.ndarray) -> np.ndarray:
        depth = depth_data.astype(np.float32)
        valid = depth[self._valid_depth_mask(depth)]
        if valid.size == 0:
            return np.zeros((*depth_data.shape, 3), dtype=np.uint8)
        lo, hi = np.percentile(valid, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(valid))
            hi = float(np.max(valid))
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip(depth, lo, hi)
        normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
        normalized[~self._valid_depth_mask(depth)] = 0
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


# -----------------------------------------------------------------------------
# Foveated mixed-resolution raster generation
# -----------------------------------------------------------------------------


def down_up(image: np.ndarray, factor: int, upsample_interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Downsample by factor, then upsample back to the original canvas size."""
    if factor <= 1:
        return image.copy()
    height, width = image.shape[:2]
    small_w = max(1, width // factor)
    small_h = max(1, height // factor)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (width, height), interpolation=upsample_interpolation)
    return restored


def dilate_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if margin_px <= 0:
        return binary * 255
    ksize = 2 * int(margin_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(binary * 255, kernel, iterations=1)


def make_foveated_raster(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    factors: Sequence[int] = (1, 2, 4, 8, 16),
    roi_margin_px: int = 8,
    distance_scale_px: float = 45.0,
    softness: float = 0.45,
    upsample_interpolation: int = cv2.INTER_LINEAR,
    keep_exact_roi: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create one normal raster image with high-res ROI and low-res background.

    Important: the returned image has the same width/height as image_bgr.
    Closed APIs usually count tokens by that final submitted size, not by visual blur.
    Use --scales to also test smaller final submitted canvases.
    """
    if image_bgr.ndim != 3:
        raise ValueError("image_bgr must be HxWx3.")
    height, width = image_bgr.shape[:2]
    if roi_mask.shape[:2] != (height, width):
        raise ValueError("roi_mask and image must have the same height/width.")

    factors = tuple(sorted(set(int(f) for f in factors if int(f) > 0)))
    if not factors or factors[0] != 1:
        factors = (1,) + factors

    high_res_mask = dilate_mask(roi_mask, roi_margin_px)
    layers = [down_up(image_bgr, factor=f, upsample_interpolation=upsample_interpolation).astype(np.float32) for f in factors]

    background = (high_res_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 5)

    target_level = np.log2(1.0 + dist / max(1e-6, float(distance_scale_px)))
    target_level = np.clip(target_level, 0.0, float(len(factors) - 1))

    weights = []
    sigma = max(1e-6, float(softness))
    for level in range(len(factors)):
        weight = np.exp(-0.5 * ((target_level - level) / sigma) ** 2)
        weights.append(weight)
    weights_np = np.stack(weights, axis=-1)
    weights_np /= weights_np.sum(axis=-1, keepdims=True) + 1e-8

    output = np.zeros_like(layers[0], dtype=np.float32)
    for i, layer in enumerate(layers):
        output += layer * weights_np[..., i : i + 1]

    if keep_exact_roi:
        exact = (roi_mask > 0).astype(np.float32)[..., None]
        output = exact * layers[0] + (1.0 - exact) * output

    return np.clip(output, 0, 255).astype(np.uint8), high_res_mask


def resize_for_api(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return image.copy()
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[:, :, 1] = 255
    m = (mask > 0)[..., None]
    overlay = np.where(m, (alpha * green + (1.0 - alpha) * overlay).astype(np.uint8), overlay)
    return overlay


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_comparison(original: np.ndarray, foveated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_panel = overlay_mask(original, mask)
    panels = [
        put_label(original, "Original camera RGB"),
        put_label(foveated, "Mixed-resolution/foveated"),
        put_label(mask_panel, "High-res ROI mask overlay"),
    ]
    return np.hstack(panels)


# -----------------------------------------------------------------------------
# Candidate payload, annotation, prompt copied into standalone file
# -----------------------------------------------------------------------------


def candidate_vlm_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "id": candidate.get("id"),
        "cx": candidate.get("cx"),
        "cy": candidate.get("cy"),
        "z_mm": candidate.get("z_mm"),
        "object_height": candidate.get("object_height"),
    }
    for key in (
        "bbox",
        "top_depth_mm",
        "center_depth_mm",
        "table_depth_mm",
        "grasp_width_mm",
        "object_length_mm",
        "recommended_open_width_mm",
        "recommended_close_width_mm",
        "source",
        "area",
    ):
        if candidate.get(key) is not None:
            payload[key] = candidate.get(key)
    return payload


def save_vlm_candidate_image(rgb_frame: np.ndarray, candidates: Sequence[Dict[str, Any]], path: Path) -> Path:
    annotated = rgb_frame.copy()
    for candidate in candidates:
        bbox = candidate.get("bbox") or {}
        try:
            x = int(bbox["x"])
            y = int(bbox["y"])
            w = int(bbox["w"])
            h = int(bbox["h"])
            cid = int(candidate["id"])
        except Exception:
            continue

        color = (0, 255, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label = f"id {cid}"
        label_y = max(22, y - 8)
        cv2.putText(annotated, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (int(candidate.get("cx", x + w / 2)), int(candidate.get("cy", y + h / 2))), 5, (0, 0, 255), -1)
    return save_png(path, annotated)


def vlm_pipeline_prompt(user_text: str, detections: Sequence[Dict[str, Any]]) -> str:
    return f"""
You are a perception module for a robot assembly system.

You are given:
1. An image of the workspace.
2. A list of detected object candidates from a depth/color localization module.
3. A user instruction describing the object or objects of interest.

Each detected candidate already has a coordinate.
Your task is NOT to estimate new coordinates.
Your task is to use:
- the image,
- the user instruction,
- and the provided detected candidates

to determine which detected object candidates correspond to the object(s) described by the user.

Candidate field meanings:
- id: candidate identifier
- cx: image x-coordinate of the detected object centroid
- cy: image y-coordinate of the detected object centroid
- z_mm: depth value in millimeters, if available
- bbox: candidate bounding box in image pixels
- object_height: estimated candidate height in millimeters, if available

Rules:
1. Use the user instruction as the semantic target.
2. Use the annotated image to visually interpret the object appearance.
3. Use the provided candidate coordinates only to associate the correct object with its existing coordinate.
4. Do not invent new coordinates.
5. Do not modify the provided coordinates.
6. Only return candidate ids that match the user instruction.
7. If multiple candidates match, return all matching candidates.
8. If no candidate clearly matches, return an empty JSON array.
9. The image is annotated with the same candidate ids as the list below. Prefer those labels over estimating from pixel coordinates alone.

Return only a JSON array in this exact format:
[
  {{
    "id": 0,
    "object_type": "user-described object name",
    "coordinate": {{"cx": 429, "cy": 62, "z_mm": 335}}
  }}
]

User instruction:
{user_text}

Detected object candidates from localization:
{json.dumps(list(detections), indent=2)}
""".strip()


# -----------------------------------------------------------------------------
# VLM connections and token extraction
# -----------------------------------------------------------------------------


def encode_image_data_url(path: Path, mime_type: str = "image/png") -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def call_openai_vlm(
    image_path: Path,
    prompt: str,
    model: str,
    detail: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": "You are a robot vision testing module. Follow the user's output format exactly.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_url(image_path),
                        "detail": detail,
                    },
                },
            ],
        },
    ]

    # max_tokens works with the chat-completions vision path used by the original pipeline.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_output_tokens,
    )
    return {
        "provider": "openai",
        "model": model,
        "detail": detail,
        "image_path": str(image_path),
        "text": response.choices[0].message.content or "",
        "usage": object_to_plain(getattr(response, "usage", None)),
    }


def call_gemini_vlm(
    image_path: Path,
    prompt: str,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    image_bytes = image_path.read_bytes()
    contents = [types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt]

    count_response: Any = None
    try:
        count_response = client.models.count_tokens(model=model, contents=contents)
    except Exception as exc:
        count_response = {"count_tokens_error": str(exc)}

    config = types.GenerateContentConfig(max_output_tokens=max_output_tokens, temperature=0)
    response = client.models.generate_content(model=model, contents=contents, config=config)

    return {
        "provider": "gemini",
        "model": model,
        "image_path": str(image_path),
        "text": getattr(response, "text", "") or "",
        "count_tokens": object_to_plain(count_response),
        "usage": object_to_plain(getattr(response, "usage_metadata", None)),
    }


def extract_usage_numbers(provider_result: Dict[str, Any]) -> Dict[str, Optional[int]]:
    provider = provider_result.get("provider")
    usage = provider_result.get("usage") or {}
    count_tokens = provider_result.get("count_tokens") or {}

    def get_int(d: Dict[str, Any], *names: str) -> Optional[int]:
        for name in names:
            if isinstance(d, dict) and name in d and d[name] is not None:
                try:
                    return int(d[name])
                except Exception:
                    pass
        return None

    if provider == "openai":
        return {
            "input_tokens": get_int(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": get_int(usage, "completion_tokens", "output_tokens"),
            "total_tokens": get_int(usage, "total_tokens"),
        }

    if provider == "gemini":
        input_tokens = get_int(usage, "prompt_token_count")
        if input_tokens is None:
            input_tokens = get_int(count_tokens, "total_tokens", "totalTokens")
        return {
            "input_tokens": input_tokens,
            "output_tokens": get_int(usage, "candidates_token_count"),
            "total_tokens": get_int(usage, "total_token_count"),
        }

    return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def providers_from_arg(provider: str) -> List[str]:
    provider = (provider or "").lower().strip()
    if not provider:
        provider = os.environ.get("VLM_PROVIDER", "gemini,openai").lower().strip()
    if provider in {"all", "both"}:
        return ["gemini", "openai"]
    providers: List[str] = []
    for item in provider.split(","):
        item = item.strip().lower()
        if item in {"gemini", "google"}:
            providers.append("gemini")
        elif item in {"openai", "gpt"}:
            providers.append("openai")
        elif item:
            raise ValueError(f"Unsupported provider: {item}")
    # Preserve order, remove duplicates.
    seen = set()
    out = []
    for p in providers:
        if p not in seen:
            out.append(p)
            seen.add(p)
    if not out:
        raise ValueError("Select at least one provider: gemini, openai, or gemini,openai.")
    return out


# -----------------------------------------------------------------------------
# Final submitted image stats and token summary
# -----------------------------------------------------------------------------


def image_stats(path: Path) -> Dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    return {"path": str(path), "width": int(w), "height": int(h), "file_bytes": int(path.stat().st_size)}


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("\nFINAL TOKEN SUMMARY: no completed VLM calls.")
        return
    headers = ["provider", "prompt_mode", "variant", "scale", "size", "file_kb", "input_tokens", "output_tokens", "total_tokens"]
    formatted_rows: List[Dict[str, str]] = []
    for row in rows:
        formatted_rows.append(
            {
                "provider": str(row.get("provider", "")),
                "prompt_mode": str(row.get("prompt_mode", "")),
                "variant": str(row.get("variant", "")),
                "scale": str(row.get("scale", "")),
                "size": f"{row.get('width')}x{row.get('height')}",
                "file_kb": f"{float(row.get('file_bytes', 0)) / 1024.0:.1f}",
                "input_tokens": str(row.get("input_tokens", "")),
                "output_tokens": str(row.get("output_tokens", "")),
                "total_tokens": str(row.get("total_tokens", "")),
            }
        )
    widths = {h: max(len(h), *(len(r[h]) for r in formatted_rows)) for h in headers}
    print("\nFINAL TOKEN SUMMARY")
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in formatted_rows:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone real-camera mixed-resolution VLM token test.")

    # Real camera input.
    parser.add_argument("--camera-serial", default=os.environ.get("ROBOT_CAMERA_SERIAL", ROBOT_CAMERA_SERIAL))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--no-realsense-filters", action="store_true")

    # Foveated image settings.
    parser.add_argument("--out-dir", default="outputs/foveated_camera_token_test")
    parser.add_argument("--factors", default="1,2,4,8,16", help="Pyramid downsample factors.")
    parser.add_argument("--roi-margin-px", type=int, default=8)
    parser.add_argument("--distance-scale-px", type=float, default=45.0)
    parser.add_argument("--softness", type=float, default=0.45)
    parser.add_argument("--scales", default="1.0,0.5", help="Final submitted canvas scales to test, e.g. 1.0,0.75,0.5")

    # Prompt/VLM settings.
    parser.add_argument("--provider", default=os.environ.get("VLM_PROVIDER", "gemini,openai"), help="gemini, openai, gemini,openai, both, or all")
    parser.add_argument("--prompt-mode", choices=["describe", "pipeline"], default="describe")
    parser.add_argument("--instruction", default="find the green block")
    parser.add_argument("--prompt", default=DEFAULT_DESCRIBE_PROMPT)
    parser.add_argument("--send-annotated", action="store_true", help="Send annotated candidate images to VLM. Automatically enabled for --prompt-mode pipeline.")
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--openai-detail", default=os.environ.get("OPENAI_IMAGE_DETAIL", "high"), choices=["low", "high", "auto", "original"])
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_VLM_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--max-output-tokens", type=int, default=160)

    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    factors = parse_csv_ints(args.factors)
    scales = parse_csv_floats(args.scales)
    providers = providers_from_arg(args.provider)
    send_annotated = bool(args.send_annotated or args.prompt_mode == "pipeline")

    print("\nSELECTED TEST CONFIG")
    print(
        json.dumps(
            {
                "input": "camera",
                "camera_serial": args.camera_serial,
                "mask_source": "depth",
                "provider": providers,
                "prompt_mode": args.prompt_mode,
                "send_annotated": send_annotated,
                "scales": scales,
                "factors": factors,
            },
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Acquire image/depth and localize mask/candidates.
    # ------------------------------------------------------------------
    print("\n[CAMERA] Capturing RealSense RGB-D frame...")
    original, depth_data, intrinsics = capture_realsense_rgbd(
        serial=args.camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        use_filters=not args.no_realsense_filters,
    )

    print(f"[IMAGE] original shape: {original.shape}")
    localizer = StandaloneDepthLocalizer(intrinsics=intrinsics)

    candidates, surface_depth, depth_mask_01 = localizer.localize(depth_data)
    depth_mask = (depth_mask_01 > 0).astype(np.uint8) * 255
    mask = depth_mask
    selected_mask_source = "depth"
    mask = (mask > 0).astype(np.uint8) * 255
    vlm_input = [candidate_vlm_payload(c) for c in candidates]

    # ------------------------------------------------------------------
    # Save localization/debug images.
    # ------------------------------------------------------------------
    original_path = save_png(out_dir / "camera_original.png", original)
    depth_path = save_png(out_dir / "camera_depth_debug.png", localizer.depth_to_debug_image(depth_data))
    depth_mask_path = save_png(out_dir / "depth_binary_mask.png", depth_mask)
    selected_mask_path = save_png(out_dir / "selected_high_res_roi_mask.png", mask)

    candidate_debug = {
        "selected_mask_source": selected_mask_source,
        "surface_depth_mm": float(surface_depth),
        "depth_candidates": candidates,
        "selected_candidates": candidates,
        "localization_debug": localizer._last_localization_debug,
        "contour_debug": localizer._last_candidate_debug,
        "intrinsics": intrinsics,
    }
    (out_dir / "localization_debug.json").write_text(json.dumps(object_to_plain(candidate_debug), indent=2), encoding="utf-8")

    print_json_block(
        "LOCALIZATION RESULT",
        {
            "selected_mask_source": selected_mask_source,
            "surface_depth_mm": surface_depth,
            "selected_candidate_count": len(candidates),
            "selected_candidates": candidates,
            "paths": {
                "camera_original": str(original_path),
                "depth_debug": str(depth_path),
                "depth_binary_mask": str(depth_mask_path),
                "selected_high_res_roi_mask": str(selected_mask_path),
                "localization_debug_json": str(out_dir / "localization_debug.json"),
            },
        },
    )

    # ------------------------------------------------------------------
    # Mixed-resolution image and annotated images.
    # ------------------------------------------------------------------
    foveated, high_res_mask = make_foveated_raster(
        original,
        mask,
        factors=factors,
        roi_margin_px=args.roi_margin_px,
        distance_scale_px=args.distance_scale_px,
        softness=args.softness,
    )

    foveated_path = save_png(out_dir / "camera_mixed_resolution_same_canvas.png", foveated)
    high_res_mask_path = save_png(out_dir / "high_res_roi_mask_with_margin.png", high_res_mask)
    comparison_path = save_png(out_dir / "comparison_real_camera_original_mixed_mask.png", make_comparison(original, foveated, mask))
    annotated_original_path = save_vlm_candidate_image(original, candidates, out_dir / "vlm_original_candidates.png")
    annotated_mixed_path = save_vlm_candidate_image(foveated, candidates, out_dir / "vlm_mixed_resolution_candidates.png")

    metadata = {
        "original_path": str(original_path),
        "mixed_resolution_same_canvas_path": str(foveated_path),
        "comparison_path": str(comparison_path),
        "annotated_original_path": str(annotated_original_path),
        "annotated_mixed_path": str(annotated_mixed_path),
        "high_res_mask_path": str(high_res_mask_path),
        "selected_mask_path": str(selected_mask_path),
        "factors": list(factors),
        "roi_margin_px": args.roi_margin_px,
        "distance_scale_px": args.distance_scale_px,
        "softness": args.softness,
    }
    (out_dir / "foveated_test_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print_json_block("SAVED REAL-CAMERA TEST IMAGES", metadata)

    # ------------------------------------------------------------------
    # Prepare final images sent to API. This is the actual submitted canvas.
    # ------------------------------------------------------------------
    base_original_for_api = cv2.imread(str(annotated_original_path if send_annotated else original_path), cv2.IMREAD_COLOR)
    base_mixed_for_api = cv2.imread(str(annotated_mixed_path if send_annotated else foveated_path), cv2.IMREAD_COLOR)
    if base_original_for_api is None or base_mixed_for_api is None:
        raise RuntimeError("Failed to read generated API source images.")

    api_variants: List[Dict[str, Any]] = []
    for scale in scales:
        for variant_name, image in [("original", base_original_for_api), ("mixed_resolution", base_mixed_for_api)]:
            api_image = resize_for_api(image, scale=scale)
            suffix = safe_name(f"s{scale:g}")
            path = save_png(out_dir / f"api_{variant_name}_{suffix}.png", api_image)
            stats = image_stats(path)
            stats.update(
                {
                    "variant": variant_name,
                    "scale": scale,
                    "send_annotated": send_annotated,
                }
            )
            api_variants.append(stats)

    print_json_block("API IMAGE VARIANTS TO SUBMIT", api_variants)

    if args.prompt_mode == "pipeline":
        prompt = vlm_pipeline_prompt(args.instruction, vlm_input)
    else:
        prompt = args.prompt

    (out_dir / "vlm_prompt.txt").write_text(prompt, encoding="utf-8")

    # ------------------------------------------------------------------
    # Call providers and print real token usage.
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    print(f"\nSELECTED VLM PROVIDERS: {providers}")

    for provider in providers:
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set but OpenAI was selected.")
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set but Gemini was selected.")

        for variant in api_variants:
            image_path = Path(variant["path"])
            print(f"\n[VLM CALL] provider={provider} variant={variant['variant']} scale={variant['scale']} image={image_path}")
            if provider == "openai":
                result = call_openai_vlm(
                    image_path=image_path,
                    prompt=prompt,
                    model=args.openai_model,
                    detail=args.openai_detail,
                    max_output_tokens=args.max_output_tokens,
                )
            elif provider == "gemini":
                result = call_gemini_vlm(
                    image_path=image_path,
                    prompt=prompt,
                    model=args.gemini_model,
                    max_output_tokens=args.max_output_tokens,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            usage_nums = extract_usage_numbers(result)
            result_path = out_dir / f"vlm_{provider}_{args.prompt_mode}_{variant['variant']}_s{safe_name(variant['scale'])}.json"
            result_path.write_text(json.dumps(object_to_plain(result), indent=2), encoding="utf-8")

            print("Response:")
            print(result.get("text", ""))
            print("Usage:")
            print(json.dumps(result.get("usage", {}), indent=2, sort_keys=True))
            if provider == "gemini":
                print("Count tokens before generation:")
                print(json.dumps(result.get("count_tokens", {}), indent=2, sort_keys=True))

            summary_rows.append(
                {
                    "provider": provider,
                    "prompt_mode": args.prompt_mode,
                    "variant": variant["variant"],
                    "scale": variant["scale"],
                    "width": variant["width"],
                    "height": variant["height"],
                    "file_bytes": variant["file_bytes"],
                    **usage_nums,
                }
            )

    print_summary(summary_rows)
    print(f"\nDone. Open comparison image: {comparison_path}")
    print(f"All outputs saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
