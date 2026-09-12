import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

from helper_function import AugmentedPointCloudProjection
import visualize_localization_projection as projection
from visualize_localization_projection import (
    mask_bbox, project_rgb_on_white, project_with_z_buffer, refine_mask, run_diagnostic,
)


class ProjectionTests(unittest.TestCase):
    def test_nearest_z_wins_and_rounded_edge_pixels_are_rejected(self):
        intrinsics = {"fx": 2, "fy": 2, "cx": 1, "cy": 1, "width": 4, "height": 3}
        points = np.array([
            [0, 0, 2], [0, 0, 1], [0.5, 0, 1], [0, 0.7, 1],
            [np.nan, 0, 1], [0, np.inf, 1], [0, 0, 0], [0, 0, -1],
            [1.4, 0, 1], [-0.75, 0, 1], [0, 0.9, 1], [2, 0, 1],
        ])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        depth, count = project_with_z_buffer(cloud, intrinsics)
        self.assertEqual(count, 4)
        self.assertEqual(np.isfinite(depth).sum(), 3)
        self.assertEqual(depth[1, 1], 1)
        self.assertTrue(np.isinf(depth[1, 3]))  # Never clamp rounded u=4 to column 3.
        reversed_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[::-1].copy()))
        np.testing.assert_array_equal(project_with_z_buffer(reversed_cloud, intrinsics)[0], depth)

    def test_no_valid_points_produce_an_empty_mask(self):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, -1]]))
        depth, count = project_with_z_buffer(cloud, {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 3, "height": 2})
        self.assertEqual(count, 0)
        self.assertTrue(np.isinf(depth).all())
        self.assertIsNone(mask_bbox(np.isfinite(depth)))

    def test_raw_projection_copies_only_original_rgb_pixels(self):
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 2] = 255
        output = project_rgb_on_white(rgb, mask)
        np.testing.assert_array_equal(output[1, 2], rgb[1, 2])
        self.assertTrue(np.all(output[mask == 0] == 255))
        np.testing.assert_array_equal(rgb, np.arange(36, dtype=np.uint8).reshape(3, 4, 3))

    def test_refinement_fills_a_small_hole_without_filling_large_gaps(self):
        raw = np.zeros((20, 20), dtype=np.uint8)
        raw[4:11, 4:11] = 255
        raw[7, 7] = 0
        original = raw.copy()
        refined = refine_mask(raw)
        self.assertEqual(refined[7, 7], 255)
        self.assertEqual(refined[15, 15], 0)
        self.assertTrue(set(np.unique(refined)) <= {0, 255})
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(refine_mask(raw, dilation_px=0, close_kernel_px=0), raw)


class DiagnosticOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.rgb_path = self.root / "original.png"
        AugmentedPointCloudProjection.write_image(self.rgb_path, self.rgb)
        self.cloud_path = self.root / "object_cluster_1.ply"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[2, 2, 1], [4, 4, 2], [5, 3, 1]]))
        cloud.paint_uniform_color([1, 0, 0])  # Must not replace the original image colors.
        o3d.io.write_point_cloud(str(self.cloud_path), cloud)
        self.payload = {
            "frame": "camera", "rgb_path": str(self.rgb_path),
            "camera_intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 8, "height": 6},
            "objects": [{"object_id": "object_001", "roi": {"x1": 1, "y1": 1, "x2": 3, "y2": 3},
                         "pointcloud_path": str(self.cloud_path)}],
        }
        self.localization_path = self.root / "localization.json"
        self.output_dir = self.root / "diagnostic"

    def run_fixture(self):
        self.localization_path.write_text(json.dumps(self.payload), encoding="utf-8")
        return run_diagnostic(self.localization_path, self.output_dir)

    def test_all_outputs_preserve_native_pixels_and_report_observed_counts(self):
        before_rgb, before_cloud = self.rgb_path.read_bytes(), self.cloud_path.read_bytes()
        summary = json.loads(self.run_fixture().read_text())
        self.assertEqual(len(list(self.output_dir.glob("*.png"))), 7)
        load = lambda name: np.asarray(o3d.io.read_image(str(self.output_dir / f"object_001_{name}.png")))
        raw_mask = load("mask_raw")
        np.testing.assert_array_equal(load("bbox_crop"), self.rgb[1:4, 1:4])
        self.assertEqual(raw_mask.shape, (6, 8))
        self.assertEqual(np.count_nonzero(raw_mask), 2)
        self.assertEqual(raw_mask[3, 5], 255)  # Outside ROI; do not hide projection mismatch.
        np.testing.assert_array_equal(load("projection_raw")[raw_mask > 0], self.rgb[raw_mask > 0])
        self.assertTrue(np.all(load("projection_raw")[raw_mask == 0] == 255))
        np.testing.assert_array_equal(load("projection_overlay")[raw_mask == 0], self.rgb[raw_mask == 0])
        record = summary["objects"][0]
        self.assertEqual(record["original_point_count"], 3)
        self.assertEqual(record["valid_projected_point_count"], 3)
        self.assertEqual(record["unique_projected_pixel_count"], 2)
        self.assertEqual(record["projected_bbox_2d_xyxy"], [2, 2, 5, 3])
        self.assertGreater(record["comparison_bbox_2d_xyxy"][2], record["roi"]["x2"])
        self.assertEqual(self.rgb_path.read_bytes(), before_rgb)
        self.assertEqual(self.cloud_path.read_bytes(), before_cloud)

    def test_downsampled_input_is_rejected(self):
        self.payload["objects"][0]["pointcloud_path"] = str(self.root / "object_cluster_1_downsampled.ply")
        with self.assertRaisesRegex(ValueError, "raw cleaned"):
            self.run_fixture()

    def test_run_uses_the_configured_morphology_constants(self):
        with patch.object(projection, "MASK_DILATION_PX", 0), patch.object(projection, "MASK_CLOSE_KERNEL_PX", 0):
            summary = json.loads(self.run_fixture().read_text())
        self.assertEqual(summary["mask_dilation_px"], 0)
        self.assertEqual(summary["mask_close_kernel_px"], 0)
        self.assertEqual(summary["objects"][0]["refined_mask_pixel_count"], 2)

    def test_intrinsics_dimension_mismatch_is_not_silently_rescaled(self):
        self.payload["camera_intrinsics"]["width"] = 9
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.run_fixture()


if __name__ == "__main__":
    unittest.main()
