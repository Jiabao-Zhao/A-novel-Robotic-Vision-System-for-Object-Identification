"""Checks for the experiment's label handling and actual PCL integration."""

import tempfile
import unittest
from pathlib import Path

from scripts.run_bop_cleanup_lccp import components, lccp_labels, mask_hash, HELPER
import numpy as np
import open3d as o3d


class CleanupLccpTests(unittest.TestCase):
    def test_components_preserve_membership_and_exclude_background(self):
        points = np.column_stack((np.arange(6), np.zeros(6), np.ones(6)))
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        result = components(cloud, np.array([0, 9, 4, 9, 9, 4]), background=0)
        self.assertEqual([label for label, _ in result], [9, 4])
        np.testing.assert_array_equal(np.asarray(result[0][1].points)[:, 0], [1, 3, 4])
        np.testing.assert_array_equal(np.asarray(result[1][1].points)[:, 0], [2, 5])

    def test_exact_mask_cache_ignores_memory_layout_but_detects_pixel_change(self):
        mask = np.eye(10, dtype=bool)
        self.assertEqual(mask_hash(mask), mask_hash(np.asfortranarray(mask)))
        changed = mask.copy()
        changed[0, 1] = True
        self.assertNotEqual(mask_hash(mask), mask_hash(changed))

    @unittest.skipUnless(HELPER.exists(), "Compile PCL helper first")
    def test_pcl_keeps_two_disconnected_surfaces_separate(self):
        x, y = np.meshgrid(np.linspace(-.025, .025, 35), np.linspace(-.025, .025, 35))
        z = .6 - np.sqrt(.04 ** 2 - x ** 2 - y ** 2)
        a = np.column_stack((x.ravel() - .08, y.ravel(), z.ravel()))
        b = np.column_stack((x.ravel() + .08, y.ravel(), z.ravel()))
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((a, b))))
        with tempfile.TemporaryDirectory() as directory:
            labels = lccp_labels(cloud, Path(directory), "disconnected")
        self.assertEqual(len(labels), len(a) + len(b))
        self.assertTrue(np.all(labels > 0))
        self.assertFalse(set(labels[:len(a)]) & set(labels[len(a):]))


if __name__ == "__main__":
    unittest.main()
