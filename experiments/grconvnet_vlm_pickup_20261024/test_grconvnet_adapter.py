"""Focused model/crop checks; require explicitly provisioned local artifacts."""
import hashlib

import numpy as np
import torch

from experiments.grconvnet_vlm_pickup_20261024.grconvnet_adapter import (
    GrConvNetAdapter, MODEL_SIZE, crop_box, rectangle_corners,
)
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps


def synthetic_input():
    rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
    depth = np.full((480, 640), .65, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:260, 280:360] = True
    depth[mask] = .60
    rgb[mask] = [220, 130, 40]
    return rgb, depth, mask


def test_crop_and_rectangle_axis_mapping():
    _, _, mask = synthetic_input()
    assert crop_box(mask) == (150, 240, 160)
    for angle in (0., np.pi/4, -np.pi/2):
        corners = rectangle_corners([80., 90.], angle, 40., 20.)
        axis = (corners[1]-corners[0])/40.
        np.testing.assert_allclose(axis, [-np.sin(angle), np.cos(angle)], atol=1e-12)
        np.testing.assert_allclose(corners.mean(0), [80., 90.], atol=1e-12)


def test_native_tensor_prediction_and_peak_parity_without_weight_updates():
    model = GrConvNetAdapter()
    rgb, depth, mask = synthetic_input()
    before = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
              for name, value in model.net.state_dict().items()}
    result = model.predict(rgb, depth, mask)
    camera = CameraData(width=MODEL_SIZE, height=MODEL_SIZE, output_size=MODEL_SIZE)
    expected_tensor, _, _ = camera.get_data(
        rgb=result["preprocessed_rgb"], depth=result["preprocessed_depth_scaled"][..., None])
    np.testing.assert_array_equal(expected_tensor.numpy(), result["input_tensor"])
    with torch.inference_mode():
        native = model.net.predict(expected_tensor)
    quality, angle, widths = post_process_output(native["pos"], native["cos"], native["sin"], native["width"])
    for actual, expected in ((result["maps"]["quality"], quality),
                             (result["maps"]["angle_rad"], angle),
                             (result["maps"]["width_pixels_model"], widths)):
        np.testing.assert_array_equal(actual, expected)
    native_grasps = detect_grasps(quality, angle, width_img=widths, no_grasps=4)
    assert len(result["proposals"]) == len(native_grasps)
    scale = result["crop"]["source_pixels_per_model_pixel"]
    origin = np.array(result["crop"]["origin_rc"])
    for actual, expected in zip(result["proposals"], native_grasps):
        np.testing.assert_array_equal(actual["center_rc_model"], expected.center)
        np.testing.assert_allclose(actual["center_rc"], origin+(np.array(expected.center)+.5)*scale-.5)
        assert actual["width_pixels"] == expected.length*scale
        assert actual["angle_rad"] == expected.angle
    after = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
             for name, value in model.net.state_dict().items()}
    assert before == after
    assert not model.net.training and not any(parameter.requires_grad for parameter in model.net.parameters())
