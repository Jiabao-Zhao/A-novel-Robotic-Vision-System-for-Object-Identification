"""Inspect saved cleaned object clouds in original RGB pixel coordinates.

Run directly with Python; edit the constants below for another saved capture.
This diagnostic never captures, segments, associates, or registers objects.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from helper_function import AugmentedPointCloudProjection


ROOT = Path(__file__).resolve().parent
LOCALIZATION_PATH = ROOT / "outputs/physical/point_cloud_localization/point_cloud_localization.json"
OUTPUT_DIR = ROOT / "outputs/physical/localization_projection"
MASK_DILATION_PX = 1
MASK_CLOSE_KERNEL_PX = 3
OVERLAY_ALPHA = 0.45
OVERLAY_COLOR_RGB = (0, 190, 255)
COMPARISON_PANEL_SIZE_PX = (320, 240)


def project_with_z_buffer(cloud, intrinsics):
    """Return nearest positive camera Z per pixel, and count points before collisions."""
    # Existing utility applies the pinhole equations and filters nonfinite XYZ,
    # Z <= 0, and floating-point projections outside the image.
    projected = AugmentedPointCloudProjection().project_cloud(cloud, intrinsics)
    width, height = int(intrinsics["width"]), int(intrinsics["height"])
    depth_m = np.full((height, width), np.inf, dtype=float)
    pixels = np.rint(projected[:, :2]).astype(np.int64)
    # A coordinate just below width/height can round beyond the final pixel.
    # Reject it, rather than clipping it onto an unrelated image-edge pixel.
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    u, v = pixels[inside].T
    np.minimum.at(depth_m, (v, u), projected[inside, 2])
    return depth_m, int(np.count_nonzero(inside))


def refine_mask(raw_mask, dilation_px=MASK_DILATION_PX, close_kernel_px=MASK_CLOSE_KERNEL_PX):
    """Only local dilation/closing; never infer a filled object silhouette."""
    if dilation_px < 0 or close_kernel_px < 0 or (close_kernel_px and close_kernel_px % 2 == 0):
        raise ValueError("Use nonnegative dilation and an odd closing kernel (or zero to disable).")
    refined = raw_mask.copy()
    if dilation_px:
        kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
        refined = cv2.dilate(refined, kernel, iterations=1)
    if close_kernel_px:
        kernel = np.ones((close_kernel_px, close_kernel_px), dtype=np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return refined


def project_rgb_on_white(rgb, mask):
    projected = np.full_like(rgb, 255)
    support = mask > 0
    projected[support] = rgb[support]
    return projected


def mask_bbox(mask):
    """Inclusive xyxy pixel bounds, matching the localizer's ROI convention."""
    v, u = np.nonzero(mask)
    if not len(u):
        return None
    return [int(u.min()), int(v.min()), int(u.max()), int(v.max())]


def comparison_sheet(object_id, panels, viewport):
    """Use a shared viewport/scale so out-of-ROI projected support stays visible."""
    panel_width, panel_height = COMPARISON_PANEL_SIZE_PX
    gap, header_height, label_height = 12, 60, 32
    sheet = np.full((header_height + label_height + panel_height + gap,
                     gap + len(panels) * (panel_width + gap), 3), 245, dtype=np.uint8)
    cv2.putText(sheet, f"{object_id} | RGB-D localization projection", (gap, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(sheet, f"Shared pixel viewport: {viewport} | masks: white = supported", (gap, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    x1, y1, x2, y2 = viewport
    scale = min(panel_width / (x2 - x1 + 1), panel_height / (y2 - y1 + 1))
    size = (max(1, round((x2 - x1 + 1) * scale)), max(1, round((y2 - y1 + 1) * scale)))
    for index, (label, image) in enumerate(panels):
        crop = image[y1:y2 + 1, x1:x2 + 1]
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        # Resizing is for this contact sheet only; all other artifacts stay native.
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
        left = gap + index * (panel_width + gap)
        cv2.putText(sheet, label, (left, header_height + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        x, y = left + (panel_width - size[0]) // 2, header_height + label_height + (panel_height - size[1]) // 2
        sheet[y:y + size[1], x:x + size[0]] = crop
    return sheet


def run_diagnostic(localization_path=LOCALIZATION_PATH, output_dir=OUTPUT_DIR):
    localization_path, output_dir = Path(localization_path), Path(output_dir)
    payload = json.loads(localization_path.read_text(encoding="utf-8"))
    if payload.get("frame") != "camera":
        raise ValueError("Projection requires saved object clouds in the camera frame.")
    intrinsics = payload["camera_intrinsics"]
    calibration = np.array([intrinsics[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
    if not np.isfinite(calibration).all() or np.any(calibration[:2] <= 0):
        raise ValueError("Saved camera intrinsics must be finite, with positive fx and fy.")
    rgb_path = ROOT / payload["rgb_path"]  # Saved relative paths are repository-relative.
    rgb = AugmentedPointCloudProjection.load_rgb_image(rgb_path)
    height, width = rgb.shape[:2]
    if (width, height) != (intrinsics["width"], intrinsics["height"]):
        raise ValueError("Saved camera intrinsics and original RGB dimensions differ.")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in payload["objects"]:
        object_id, roi = item["object_id"], item["roi"]
        cloud_path = ROOT / item["pointcloud_path"]
        if cloud_path.stem.endswith("_downsampled"):
            raise ValueError(f"Use the saved raw cleaned object cloud, not {cloud_path.name}.")
        if not cloud_path.is_file():
            raise FileNotFoundError(cloud_path)
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        if cloud.is_empty():
            raise ValueError(f"Saved object cloud is empty or unreadable: {cloud_path}")
        x1, y1, x2, y2 = [int(roi[key]) for key in ("x1", "y1", "x2", "y2")]
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ValueError(f"Object {object_id} has an invalid image ROI: {roi}")
        depth_m, valid_count = project_with_z_buffer(cloud, intrinsics)
        raw_mask = np.isfinite(depth_m).astype(np.uint8) * 255
        refined_mask = refine_mask(raw_mask, MASK_DILATION_PX, MASK_CLOSE_KERNEL_PX)
        raw_projection = project_rgb_on_white(rgb, raw_mask)
        refined_projection = project_rgb_on_white(rgb, refined_mask)
        bbox_crop = rgb[y1:y2 + 1, x1:x2 + 1].copy()
        overlay = rgb.copy()
        support = raw_mask > 0
        overlay[support] = np.rint((1 - OVERLAY_ALPHA) * rgb[support]
                                   + OVERLAY_ALPHA * np.array(OVERLAY_COLOR_RGB)).astype(np.uint8)
        projected_bbox = mask_bbox(raw_mask)
        refined_bbox = mask_bbox(refined_mask)
        viewport = [x1, y1, x2, y2]
        if refined_bbox is not None:
            viewport = [min(x1, refined_bbox[0]), min(y1, refined_bbox[1]),
                        max(x2, refined_bbox[2]), max(y2, refined_bbox[3])]
        # Preserve the actual bbox crop's position when projections extend beyond it.
        bbox_panel = np.full_like(rgb, 255)
        bbox_panel[y1:y2 + 1, x1:x2 + 1] = bbox_crop
        panels = [("1. Original bbox crop", bbox_panel), ("2. Raw projection", raw_projection),
                  ("3. Refined projection", refined_projection), ("4. Raw mask", raw_mask),
                  ("5. Refined mask", refined_mask)]
        images = {"bbox_crop": bbox_crop, "mask_raw": raw_mask, "projection_raw": raw_projection,
                  "mask_refined": refined_mask, "projection_refined": refined_projection,
                  "projection_overlay": overlay, "comparison": comparison_sheet(object_id, panels, viewport)}
        for name, image in images.items():
            AugmentedPointCloudProjection.write_image(output_dir / f"{object_id}_{name}.png", image)
        records.append({
            "object_id": object_id, "pointcloud_path": str(cloud_path),
            "original_point_count": len(cloud.points), "valid_projected_point_count": valid_count,
            "unique_projected_pixel_count": int(np.count_nonzero(raw_mask)), "roi": roi,
            "projected_bbox_2d_xyxy": projected_bbox,
            "raw_mask_pixel_count": int(np.count_nonzero(raw_mask)),
            "refined_mask_pixel_count": int(np.count_nonzero(refined_mask)),
            "image_width": width, "image_height": height, "comparison_bbox_2d_xyxy": viewport,
        })
        print(f"{object_id}: {valid_count}/{len(cloud.points)} points project inside RGB; "
              f"{records[-1]['raw_mask_pixel_count']} raw / {records[-1]['refined_mask_pixel_count']} refined pixels")
    summary = {"localization_path": str(localization_path), "rgb_path": str(rgb_path),
               "frame": "camera", "camera_intrinsics": intrinsics,
               "bbox_convention": "inclusive xyxy pixels", "z_buffer": "minimum positive camera Z per rounded pixel",
               "mask_dilation_px": MASK_DILATION_PX, "mask_close_kernel_px": MASK_CLOSE_KERNEL_PX,
               "objects": records}
    summary_path = output_dir / "projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved projection diagnostics: {output_dir}")
    return summary_path


if __name__ == "__main__":
    run_diagnostic()
