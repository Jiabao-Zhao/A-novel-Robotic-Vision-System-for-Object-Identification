"""Open3D input compatibility only; no scene labels or score changes."""
import unittest
import numpy as np
from localize import PointCloudLocalization


class DepthDtypeTests(unittest.TestCase):
    def test_float32_is_unchanged_and_float64_has_the_same_cloud(self):
        depth32 = np.linspace(.5, .8, 480, dtype=np.float32).reshape(20, 24)
        depth64 = depth32.astype(np.float64)
        original64 = depth64.copy()
        converted32 = np.ascontiguousarray(depth32, dtype=np.float32)
        self.assertIs(converted32, depth32)
        rgb = np.zeros((20, 24, 3), dtype=np.uint8)
        camera = dict(width=24, height=20, fx=80., fy=80., cx=12., cy=10.)
        localizer = PointCloudLocalization()
        reference = localizer.rgbd_to_pointcloud(rgb, depth32, 1., camera)
        actual = localizer.rgbd_to_pointcloud(rgb, np.ascontiguousarray(depth64, dtype=np.float32), 1., camera)
        np.testing.assert_array_equal(np.asarray(actual.points), np.asarray(reference.points))
        np.testing.assert_array_equal(depth64, original64)
        self.assertEqual(depth64.dtype, np.float64)


if __name__ == '__main__':
    unittest.main()
