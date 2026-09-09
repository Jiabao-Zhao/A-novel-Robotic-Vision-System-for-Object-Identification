"""Pickup outcomes and scene constraints, independent of VLM/provider responses."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_wrist_cluster import PickupEvaluator, HOLD_STEPS
from simulation.wrist_cluster import (DESCRIPTIONS, SEED, WORKSPACE_MIN, WORKSPACE_MAX,
                                      BOARD_EXCLUSION_XY, MIN_OBJECT_GAP_M, MIN_BOARD_GAP_M, random_placements)


class PickupMetricTests(unittest.TestCase):
    def setUp(self):
        model = SimpleNamespace(ngeom=4, geom_bodyid=np.array([1, 2, 3, 3]),
            body_parentid=np.array([0, 0, 0, 0]),
            body_name2id=lambda name: {"nist_part_target": 1, "nist_part_other": 2}[name],
            geom_name2id=lambda name: {"left": 2, "right": 3}[name])
        self.data = SimpleNamespace(body_xpos=np.zeros((4, 3)), ncon=2,
            contact=[SimpleNamespace(geom1=0, geom2=2), SimpleNamespace(geom1=0, geom2=3)])
        env = SimpleNamespace(sim=SimpleNamespace(model=model, data=self.data),
            robots=[SimpleNamespace(gripper=SimpleNamespace(important_geoms={
                "left_fingerpad": ["left"], "right_fingerpad": ["right"]}))])
        with patch("scripts.run_wrist_cluster.DESCRIPTIONS", {"target": "target", "other": "other"}):
            self.evaluator = PickupEvaluator(env, "target")

    def test_lift_needs_bilateral_contact_and_continuous_hold(self):
        self.data.body_xpos[1, 2] = .025
        for _ in range(HOLD_STEPS-1):
            self.assertFalse(self.evaluator.score()["success"])
        self.assertTrue(self.evaluator.score()["success"])
        for _ in range(19):
            self.evaluator.score()
        self.data.ncon = 1
        final = self.evaluator.score()
        self.assertFalse(final["success"])
        self.assertTrue(final["achieved_one_second_hold_at_any_time"])
        self.assertEqual(final["max_consecutive_hold_steps"], 39)
        self.assertEqual(self.evaluator.hold_steps, 0)

    def test_lifting_a_distractor_invalidates_pickup(self):
        self.data.body_xpos[1:3, 2] = .025
        for _ in range(HOLD_STEPS+1):
            self.assertFalse(self.evaluator.score()["success"])
        self.assertEqual(self.evaluator.wrong_lifted, {"other"})

    def test_contact_without_lift_is_not_success(self):
        for _ in range(HOLD_STEPS+1):
            self.assertFalse(self.evaluator.score()["success"])


class SceneLayoutTests(unittest.TestCase):
    def test_new_trial_seed_changes_every_object_initial_placement(self):
        catalog = {name: {"extent_m": [.030, .030, .030]} for name in DESCRIPTIONS}
        first = random_placements(catalog, seed=SEED)
        second = random_placements(catalog, seed=SEED+1)
        self.assertEqual(second, random_placements(catalog, seed=SEED+1))
        for name in DESCRIPTIONS:
            self.assertNotEqual(first[name]["xy_m"], second[name]["xy_m"])
            self.assertNotEqual(first[name]["yaw_deg"], second[name]["yaw_deg"])

    def test_all_targets_have_reproducible_nonoverlapping_placements(self):
        catalog = {name: {"extent_m": [.030, .030, .030]} for name in DESCRIPTIONS}
        first = random_placements(catalog)
        self.assertEqual(first, random_placements(catalog))
        self.assertEqual(len(first), 14)
        self.assertIn("red_block", first)
        self.assertIn("blue_block", first)
        radius = np.linalg.norm([.030, .030]) / 2
        for name, entry in first.items():
            point = np.asarray(entry["xy_m"])
            self.assertTrue(np.all(point-radius >= WORKSPACE_MIN[:2]))
            self.assertTrue(np.all(point+radius <= WORKSPACE_MAX[:2]))
            self.assertGreaterEqual(np.linalg.norm(point-np.clip(point, *BOARD_EXCLUSION_XY)), radius+MIN_BOARD_GAP_M)
            for other, second in first.items():
                if other != name:
                    self.assertGreaterEqual(np.linalg.norm(point-second["xy_m"]), 2*radius+MIN_OBJECT_GAP_M)
        # Exercise the newly used floor on both sides of the old narrow strip.
        xy = np.array([entry["xy_m"] for entry in first.values()])
        self.assertGreater(np.ptp(xy[:, 1]), .50)
        self.assertTrue(np.all(xy >= -.4) and np.all(xy <= .4))


if __name__ == "__main__":
    unittest.main()
