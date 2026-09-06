import unittest

import numpy as np

from simulation.nist_peg_task import WORLD_T_HOLE, insertion_metrics


class PegInsertionMetricsTests(unittest.TestCase):
    def test_centered_peg_inserted_six_mm(self):
        result = insertion_metrics(WORLD_T_HOLE[:3, 3] + [0, 0, .019], np.eye(3))
        self.assertTrue(result["inserted"])
        self.assertAlmostEqual(result["insertion_depth_mm"], 6.)

    def test_approach_is_not_insertion(self):
        result = insertion_metrics(WORLD_T_HOLE[:3, 3] + [0, 0, .050], np.eye(3))
        self.assertFalse(result["inserted"])

    def test_target_below_plane_elsewhere_is_not_inserted(self):
        result = insertion_metrics(WORLD_T_HOLE[:3, 3] + [.1, .1, .019], np.eye(3))
        self.assertFalse(result["inserted"])
        self.assertEqual(result["insertion_depth_mm"], 0.)

    def test_submillimeter_misalignment_exceeds_clearance(self):
        result = insertion_metrics(WORLD_T_HOLE[:3, 3] + [.0003, 0, .019], np.eye(3))
        self.assertFalse(result["inserted"])
        self.assertLess(result["estimated_clearance_mm"], 0.)

    def test_known_hole_frame_is_respected(self):
        pose = WORLD_T_HOLE.copy()
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        position = pose[:3, 3] + pose[:3, :3] @ [0, 0, .019]
        result = insertion_metrics(position, pose[:3, :3], pose)
        self.assertTrue(result["inserted"])
