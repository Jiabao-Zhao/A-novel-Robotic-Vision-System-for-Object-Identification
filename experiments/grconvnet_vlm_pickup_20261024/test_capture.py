"""Known-plane camera coordinates and image-derived grip-height checks."""
import numpy as np

from experiments.grconvnet_vlm_pickup_20261024.capture import unproject_pixels, rectangle_geometry


def calibration():
    return {"intrinsics": [[100., 0., 20.], [0., 100., 20.], [0., 0., 1.]],
            "world_T_camera": [[1., 0., 0., .25], [0., -1., 0., 0.],
                               [0., 0., -1., 1.2], [0., 0., 0., 1.]],
            "table_z_world_m": 0.}


def test_known_table_axes_and_metric_depth():
    # A 100-pixel focal length at 1.2 m makes 10 pixels equal 12 cm.
    points = unproject_pixels([[20., 20.], [20., 30.], [30., 20.]], 1.2, calibration())
    np.testing.assert_allclose(points, [[.25, 0., 0.], [.37, 0., 0.], [.25, -.12, 0.]], atol=1e-12)


def test_grasp_height_comes_from_depth_and_table_without_object_pose():
    depth = np.full((41, 41), 1.19)
    mask = np.ones(depth.shape, bool)
    rectangle = {"center_rc": [20., 20.], "jaw_axis_rc_unit": [0., 1.], "width_pixels": 4.}
    geometry = rectangle_geometry(rectangle, depth, mask, calibration())
    np.testing.assert_allclose(geometry["surface_center_world_m"], [.25, 0., .01], atol=1e-12)
    np.testing.assert_allclose(geometry["center_world_m"], [.25, 0., .005], atol=1e-12)
    np.testing.assert_allclose(geometry["jaw_axis_world_unit"], [1., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(geometry["predicted_width_m"], .0476, atol=1e-12)
    mask[20, 20] = False
    fallback = rectangle_geometry(rectangle, depth, mask, calibration())
    np.testing.assert_allclose(fallback["center_world_m"], geometry["center_world_m"], atol=1e-12)
    assert not fallback["mask_center_valid"]
    assert fallback["depth_method"] == "median_visible_target_depth_within_five_pixels"
    assert not fallback["object_pose_used"]
