"""SAM-6D ISM projected-bounding-box geometry for fixed CAD/crop matching.

The scoring arithmetic follows JiehongLin/SAM-6D commit
1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2. Inputs use CAD and camera meters,
an explicit CAD-to-OpenCV-camera rotation, and a full-image observed mask.
This module does not estimate rotation or scale, fit a table, or register points.

Adapted from upstream MIT-licensed detector.py, bbox_utils.py, trimesh_utils.py.
Copyright (c) 2023 Van Nguyen Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


POINTCLOUD_SAMPLE_NUM = 2048


def bbox_iou_xyxy(box_a, box_b):
    """Per-pair upstream IoU: max pixel coordinates, without inclusive +1."""
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    intersection_wh = np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])
    if not np.all(intersection_wh > 0):
        return 0.0
    intersection = np.prod(intersection_wh)
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def compute_geometry(cad_points_m, cad_to_camera_R, observed_mask, depth_m, K):
    """Return native-scale SAM-6D bbox IoU and all projection measurements.

    ``cad_points_m`` should be the frozen 2,048 surface samples of the query
    CAD. ``cad_to_camera_R`` is the best visual template's rotation, with camera
    +x right, +y down, +z forward. ``depth_m`` is camera-z depth in meters.
    The observed bbox is the tight min/max box of the supplied full-frame mask.

    Upstream reads millimeters and converts to meters; this interface accepts
    meters directly. Pixels use u/v integer coordinates, with no half-pixel
    offset. Projection is float32, truncates toward zero, then clamps to the
    image. The mean observed surface point is used as the CAD-origin translation
    without subtracting the rotated CAD centroid, as in upstream.

    Structural failures stay unavailable rather than receiving an invented
    score. Nonfinite depths are excluded; nonpositive projected CAD depths are
    unavailable. Neither case occurs in the controlled benchmark inputs.
    """
    points = np.asarray(cad_points_m, dtype=np.float32)
    rotation = np.asarray(cad_to_camera_R, dtype=np.float32)
    mask = np.asarray(observed_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    intrinsic = np.asarray(K, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("CAD samples must be a nonempty N x 3 array in meters.")
    if rotation.shape != (3, 3) or intrinsic.shape != (3, 3):
        raise ValueError("Rotation and camera intrinsics must both be 3 x 3.")
    if mask.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Mask and depth must be matching full-frame H x W arrays.")
    if not np.isfinite(points).all() or not np.isfinite(rotation).all():
        raise ValueError("CAD points and template rotation must be finite.")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=2e-5
    ):
        raise ValueError("Template rotation must preserve native CAD scale and handedness.")
    if not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths.")
    if not np.allclose(intrinsic[2], [0, 0, 1]) or intrinsic[0, 1] != 0:
        raise ValueError("Expected standard zero-skew pinhole camera intrinsics.")

    result = {
        "status": "unavailable",
        "reason": None,
        "geometry_bbox_iou": None,
        "translation_camera_m": None,
        "predicted_bbox_xyxy": None,
        "observed_bbox_xyxy": None,
        "observed_mask_pixels": int(mask.sum()),
        "observed_valid_depth_pixels": 0,
        "cad_sample_count": len(points),
        "projected_cad_positive_depth_count": None,
    }
    rows, cols = np.nonzero(mask)
    if not len(rows):
        result["reason"] = "empty_observed_mask"
        return result
    observed_box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
    result["observed_bbox_xyxy"] = observed_box
    valid = mask & np.isfinite(depth) & (depth > 0)
    valid_rows, valid_cols = np.nonzero(valid)
    count = len(valid_rows)
    result["observed_valid_depth_pixels"] = count
    if not count:
        result["reason"] = "no_positive_finite_observed_depth"
        return result

    # Sum over the full image to retain the upstream float32 reduction policy.
    z = np.where(valid, depth, np.float32(0))
    u = np.arange(depth.shape[1], dtype=np.float32)[None, :]
    v = np.arange(depth.shape[0], dtype=np.float32)[:, None]
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    denominator = np.float32(count) + np.float32(1e-8)
    translation = np.asarray([x.sum(), y.sum(), z.sum()], dtype=np.float32) / denominator
    result["translation_camera_m"] = translation.tolist()

    posed = (rotation @ points.T).T + translation
    result["projected_cad_positive_depth_count"] = int(np.count_nonzero(posed[:, 2] > 0))
    if not np.isfinite(posed).all() or np.any(posed[:, 2] <= 0):
        result["reason"] = "nonpositive_or_nonfinite_projected_cad_depth"
        return result
    homogeneous = (intrinsic @ posed.T).T
    pixels_float = (homogeneous / homogeneous[:, -1, None])[:, :2]
    if not np.isfinite(pixels_float).all() or np.max(np.abs(pixels_float)) >= 2**31:
        result["reason"] = "nonfinite_or_int32_overflow_projection"
        return result
    pixels = pixels_float.astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, depth.shape[1] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, depth.shape[0] - 1)
    predicted_box = np.concatenate((pixels.min(axis=0), pixels.max(axis=0))).tolist()
    result.update({
        "status": "available",
        "predicted_bbox_xyxy": predicted_box,
        "geometry_bbox_iou": bbox_iou_xyxy(predicted_box, observed_box),
    })
    return result
