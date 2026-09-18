import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.run_wrist_layout_ablation import layout_plan, rearranged_pose, evaluate, METHODS


class LayoutAblationTests(unittest.TestCase):
    def test_predeclared_position_and_yaw_factors_are_separate(self):
        plan = layout_plan(list("abcdef"))
        self.assertEqual(plan, layout_plan(list("abcdef")))
        self.assertEqual(len(plan["layouts"]), 21)
        position = [r for r in plan["layouts"] if r["family"] == "position"]
        yaw = [r for r in plan["layouts"] if r["family"] == "yaw"]
        self.assertEqual(len({tuple(r["slots"]) for r in position}), 10)
        for row in position:
            self.assertEqual(sorted(row["slots"]), list(range(6)))
            self.assertTrue(all(i != j for i, j in enumerate(row["slots"])))
            self.assertEqual(row["yaw_delta_deg"], [0.] * 6)
        for row in yaw:
            self.assertEqual(row["slots"], list(range(6)))
        for i in range(6):
            self.assertEqual(sorted(r["yaw_delta_deg"][i] for r in yaw), [-150., -120., -90., -60., -30., 30., 60., 90., 120., 150.])

    def test_world_yaw_preserves_height_and_pin_tilt(self):
        rotation = Rotation.from_euler("zy", [30, 90], degrees=True)
        base = np.r_[.1, .2, .008, rotation.as_quat()[[3, 0, 1, 2]]]
        slot = np.r_[-.2, -.1, .04, 1., 0., 0., 0.]
        pose = rearranged_pose(base, slot, 60)
        np.testing.assert_array_equal(pose[:3], [-.2, -.1, .008])
        actual = Rotation.from_quat(pose[3:][[1, 2, 3, 0]]).as_matrix()
        expected = Rotation.from_euler("z", 60, degrees=True).as_matrix() @ rotation.as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        np.testing.assert_allclose(actual[2], rotation.as_matrix()[2], atol=1e-15)

    def test_global_and_patch_rank_independently_and_misses_fail(self):
        self.assertEqual(set(METHODS), {"global", "patch"})
        rows = [{"cad_id": "pin", "object_id": "a", "semantic_score": .3, "appearance_score": .8},
                {"cad_id": "pin", "object_id": "b", "semantic_score": .7, "appearance_score": .6}]
        audit = [{"status": "matched", "simulator_instance": "pin", "object_id": "a"},
                 {"status": "matched", "simulator_instance": "block", "object_id": "b"}]
        scores = evaluate(rows, audit, ["pin"])
        self.assertFalse(scores[0]["correct"])
        self.assertTrue(scores[1]["correct"])
        self.assertAlmostEqual(scores[0]["margin"], -.4)
        self.assertAlmostEqual(scores[1]["margin"], .2)
        self.assertTrue(all(not r["correct"] and r["margin"] is None for r in evaluate([], [], ["pin"])))


if __name__ == "__main__":
    unittest.main()
