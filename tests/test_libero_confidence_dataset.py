import unittest

import numpy as np

from scripts.capture_libero_confidence_dataset import (
    compact_localization,
    dataset_partition,
    match_target_to_localized_candidate,
)


class LiberoConfidenceDatasetTests(unittest.TestCase):
    def test_dataset_partitions_are_disjoint_by_initial_state(self):
        self.assertEqual(dataset_partition(0), "calibration")
        self.assertEqual(dataset_partition(29), "calibration")
        self.assertEqual(dataset_partition(30), "validation")
        self.assertEqual(dataset_partition(39), "validation")
        self.assertEqual(dataset_partition(40), "test")
        self.assertEqual(dataset_partition(49), "test")

    def test_ground_truth_match_uses_nearest_world_xy_centroid(self):
        localization = {
            "objects": [
                {"object_id": "object_001", "centroid_3d_m": [0.1, 0.0, 0.2]},
                {"object_id": "object_002", "centroid_3d_m": [0.3, 0.0, 0.2]},
            ]
        }

        object_id, diagnostics = match_target_to_localized_candidate(
            localization,
            np.eye(4),
            [0.28, 0.01, 0.0],
        )

        self.assertEqual(object_id, "object_002")
        self.assertTrue(diagnostics["target_localized"])
        self.assertAlmostEqual(
            diagnostics["nearest_xy_distance_m"],
            np.sqrt(0.02**2 + 0.01**2),
        )

    def test_distant_target_is_not_assigned_to_a_candidate(self):
        localization = {
            "objects": [
                {"object_id": "object_001", "centroid_3d_m": [0.0, 0.0, 0.2]}
            ]
        }

        object_id, diagnostics = match_target_to_localized_candidate(
            localization,
            np.eye(4),
            [0.2, 0.2, 0.0],
            maximum_xy_distance_m=0.05,
        )

        self.assertIsNone(object_id)
        self.assertFalse(diagnostics["target_localized"])
        self.assertEqual(diagnostics["nearest_candidate_id"], "object_001")

    def test_compact_localization_removes_pointcloud_paths(self):
        payload = {
            "frame": "camera",
            "camera_intrinsics": {"width": 768, "height": 768},
            "object_count": 1,
            "plane_model": [0.0, 1.0, 0.0, -1.0],
            "objects": [
                {
                    "object_id": "object_001",
                    "roi": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                    "centroid_3d_m": [0.0, 0.0, 1.0],
                    "size_3d_m": [0.1, 0.1, 0.1],
                    "point_count": 100,
                    "saved_point_count": 200,
                    "pointcloud_path": "temporary.ply",
                }
            ],
        }

        compact = compact_localization(payload, "rgb.png")

        self.assertNotIn("pointcloud_path", compact["objects"][0])
        self.assertNotIn("saved_point_count", compact["objects"][0])
        self.assertEqual(compact["objects"][0]["object_id"], "object_001")


if __name__ == "__main__":
    unittest.main()
