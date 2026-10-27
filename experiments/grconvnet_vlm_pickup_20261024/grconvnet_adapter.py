"""Pinned pretrained GR-ConvNet proposals from masked, calibrated RGB-D only.

No CAD pose enters this adapter. Returned locations are in the original image's
row/column pixel frame. World-frame conversion belongs to the calibrated host.
"""
import hashlib
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = HERE / "vendor/grconvnet_source"
DEPENDENCIES = ROOT / "outputs/grconvnet_vlm_pickup_20261024/local_dependencies"
MODEL = ROOT / "outputs/grconvnet_vlm_pickup_20261024/model/cornell_rgbd_grconvnet3_ch32_epoch19.pt"
for directory in (DEPENDENCIES, VENDOR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import torch
from skimage.transform import resize

from inference.models.grconvnet3 import GenerativeResnet
from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps


MODEL_SIZE = 224
CROP_MIN_SIDE = 128
CROP_BBOX_MULTIPLIER = 2.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_box(mask):
    """Image-derived square context; fixed policy before any pickup outcome."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Target mask is empty")
    h, w = mask.shape
    bbox_h, bbox_w = rows.max()-rows.min()+1, cols.max()-cols.min()+1
    side = min(h, w, max(CROP_MIN_SIDE, int(np.ceil(CROP_BBOX_MULTIPLIER*max(bbox_h, bbox_w)))))
    center = np.array([(rows.min()+rows.max())/2, (cols.min()+cols.max())/2])
    first = np.floor(center-(side-1)/2).astype(int)
    first = np.maximum(0, np.minimum(first, np.array([h, w])-side))
    return int(first[0]), int(first[1]), int(side)


def rectangle_corners(center, angle, length, finger_extent):
    """The exact upstream Grasp.as_gr convention, avoiding obsolete np.float."""
    row, col = center
    co, si = np.cos(angle), np.sin(angle)
    y1, x1 = row+length/2*si, col-length/2*co
    y2, x2 = row-length/2*si, col+length/2*co
    return np.array([[y1-finger_extent/2*co, x1-finger_extent/2*si],
                     [y2-finger_extent/2*co, x2-finger_extent/2*si],
                     [y2+finger_extent/2*co, x2+finger_extent/2*si],
                     [y1+finger_extent/2*co, x1+finger_extent/2*si]])


class GrConvNetAdapter:
    """One frozen CPU model; up to four unaltered native peak proposals."""
    def __init__(self, device="cpu"):
        started = time.perf_counter()
        if device != "cpu":
            raise ValueError("The frozen comparison uses CPU inference")
        provenance = json.loads((HERE / "vendor/provenance.json").read_text())
        if sha256(MODEL) != provenance["checkpoint"]["sha256"]:
            raise ValueError("Pretrained checkpoint hash differs from the pinned source")
        # Explicit provisioning converts the hash-verified official legacy
        # Module pickle once. Inference only loads the resulting tensor dict.
        safe = provenance["safe_state_dict"]
        safe_path = ROOT / safe["relative_path"]
        if sha256(safe_path) != safe["sha256"]:
            raise ValueError("Converted state dict changed after provisioning")
        self.net = GenerativeResnet(**safe["architecture"])
        self.net.load_state_dict(torch.load(safe_path, map_location="cpu", weights_only=True), strict=True)
        self.net.to("cpu").eval()
        self.net.requires_grad_(False)
        self.model_setup_seconds = time.perf_counter()-started
        self.provenance = provenance

    def predict(self, rgb, depth_m, mask, n_grasps=4):
        started = time.perf_counter()
        rgb, depth_m, mask = np.asarray(rgb), np.asarray(depth_m), np.asarray(mask, dtype=bool)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth_m.shape != rgb.shape[:2] or mask.shape != depth_m.shape:
            raise ValueError("Expected aligned HxWx3 RGB, HxW optical depth and HxW mask")
        if rgb.dtype != np.uint8 or not np.all(np.isfinite(depth_m)) or np.any(depth_m <= 0):
            raise ValueError("RGB must be uint8; optical depth must be finite positive metres")
        if not 1 <= n_grasps <= 4:
            raise ValueError("Frozen maximum is four proposals")
        r0, c0, side = crop_box(mask)
        masked_rgb = np.where(mask[..., None], rgb, np.uint8(255))
        masked_depth = np.where(mask, depth_m, float(depth_m.max())).astype(np.float32)
        rgb_crop = resize(masked_rgb[r0:r0+side, c0:c0+side], (MODEL_SIZE, MODEL_SIZE),
                          preserve_range=True).astype(np.uint8)
        depth_crop = resize(masked_depth[r0:r0+side, c0:c0+side], (MODEL_SIZE, MODEL_SIZE),
                            preserve_range=True).astype(np.float32)
        # OWG's released pre-normalization, applied to calibrated metric depth
        # instead of a simulator's nonlinear graphics-buffer depth values.
        depth_scaled = depth_crop*(255./float(depth_crop.max()))
        depth_scaled = np.clip((depth_scaled-depth_scaled.mean())/175., -1., 1.)
        camera = CameraData(width=MODEL_SIZE, height=MODEL_SIZE, output_size=MODEL_SIZE,
                            include_depth=True, include_rgb=True)
        tensor, _, _ = camera.get_data(rgb=rgb_crop, depth=depth_scaled[..., None])
        preprocess_seconds = time.perf_counter()-started
        stage = time.perf_counter()
        with torch.inference_mode():
            pred = self.net.predict(tensor.to("cpu"))
        inference_seconds = time.perf_counter()-stage
        stage = time.perf_counter()
        quality, angles, widths = post_process_output(pred["pos"], pred["cos"], pred["sin"], pred["width"])
        grasps = detect_grasps(quality, angles, width_img=widths, no_grasps=n_grasps)
        scale = side/MODEL_SIZE
        proposals = []
        for index, grasp in enumerate(grasps):
            center_model = np.asarray(grasp.center, dtype=float)
            center_orig = np.array([r0, c0])+(center_model+.5)*scale-.5
            corners_model = rectangle_corners(center_model, grasp.angle, grasp.length, grasp.width)
            corners_orig = np.array([r0, c0])+(corners_model+.5)*scale-.5
            ir, ic = np.rint(center_orig).astype(int)
            on_mask = bool(mask[np.clip(ir, 0, mask.shape[0]-1), np.clip(ic, 0, mask.shape[1]-1)])
            proposals.append({"network_candidate_id": index+1,
                "center_rc": center_orig.tolist(), "center_rc_original": center_orig.tolist(),
                "center_rc_model": center_model.tolist(), "angle_rad": float(grasp.angle),
                "width_pixels": float(grasp.length*scale), "width_pixels_original": float(grasp.length*scale),
                "finger_extent_pixels": float(grasp.width*scale),
                "jaw_axis_rc_unit": [-float(np.sin(grasp.angle)), float(np.cos(grasp.angle))],
                "rectangle_rc": corners_orig.tolist(), "quality": float(quality[grasp.center]),
                "center_on_target_mask": on_mask})
        postprocess_seconds = time.perf_counter()-stage
        return {"proposals": proposals,
                "maps": {"quality": quality, "angle_rad": angles, "width_pixels_model": widths},
                "input_tensor": tensor.numpy(),
                "preprocessed_rgb": rgb_crop, "preprocessed_depth_scaled": depth_scaled,
                "crop": {"origin_rc": [r0, c0], "side_pixels": side, "model_size": MODEL_SIZE,
                         "source_pixels_per_model_pixel": scale, "pixel_centers": "origin+(model+.5)*scale-.5"},
                "timings": {"preprocess_seconds": preprocess_seconds, "network_seconds": inference_seconds,
                            "postprocess_seconds": postprocess_seconds, "total_seconds": time.perf_counter()-started},
                "method": "Official Cornell RGB-D GR-ConvNet; masked target crop; OWG-style depth pre-normalization; original author CameraData/post_process/detect_grasps; calibrated host mapping",
                "model_sha256": self.provenance["checkpoint"]["sha256"], "device": "cpu"}
