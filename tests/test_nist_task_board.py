"""Integration checks; require the installed WSL simulator and NIST archive."""

import importlib.util
import unittest

import numpy as np

from simulation.nist_task_board_1 import (
    ASSET_DIR, BOARD_BOTTOM_Z_M, BOARD_CENTER_XY_M, FLOOR_Z_M, NistTaskBoardEnvironment, PARTS,
)


@unittest.skipUnless(importlib.util.find_spec("libero") and (ASSET_DIR / "stl.zip").is_file(),
                     "requires WSL LIBERO runtime and downloaded NIST STL archive")
class NistTaskBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = NistTaskBoardEnvironment(image_size=128)
        cls.environment.reset(seed=1000)
        for _ in range(100):
            cls.environment.step(np.array([0., 0., 0., 0., 0., 0., -1.]))

    @classmethod
    def tearDownClass(cls):
        cls.environment.close()

    def test_all_twenty_free_parts_remain_stable(self):
        sim = self.environment.sim
        self.assertEqual(len(PARTS), 20)
        for name in PARTS:
            body_id = sim.model.body_name2id(f"nist_part_{name}")
            position = sim.data.body_xpos[body_id]
            self.assertTrue(np.isfinite(position).all(), name)
            self.assertGreater(position[2], -0.06, name)
            self.assertLess(position[2], 0.10, name)
            joint_id = sim.model.joint_name2id(f"nist_part_{name}_joint")
            self.assertEqual(sim.model.jnt_type[joint_id], 0, name)  # MuJoCo free joint

    def test_board_standoffs_match_actual_arena_floor(self):
        sim = self.environment.sim
        floor_id = sim.model.geom_name2id("floor")
        self.assertEqual(sim.model.geom_type[floor_id], 0)  # MuJoCo plane
        self.assertAlmostEqual(sim.data.geom_xpos[floor_id, 2], FLOOR_Z_M)
        self.assertAlmostEqual(BOARD_BOTTOM_Z_M - FLOOR_Z_M, .020)

    def test_plate_collision_preserves_peg_opening(self):
        import mujoco

        sim = self.environment.sim
        for label, xy in (("solid", [90., 90.]), ("opening", [191.45, 197.176])):
            origin = np.r_[(np.array(xy) - 192) * 0.001 + BOARD_CENTER_XY_M, 0.15]
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                sim.model._model, sim.data._data, origin, np.array([0., 0., -1.]),
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, hit,
            )
            if label == "solid":
                self.assertGreaterEqual(distance, 0)
                self.assertAlmostEqual(0.15 - distance, BOARD_BOTTOM_Z_M + 0.0089916, places=6)
            else:
                # The arena floor is outside collision group 0 in this scene;
                # a ray through the hole can correctly return no intersection.
                self.assertTrue(distance < 0 or 0.15 - distance < BOARD_BOTTOM_Z_M)

    def test_both_cameras_provide_calibrated_metric_depth(self):
        from simulation.libero_sensor import LiberoRGBDSensor

        for camera in self.environment.camera_names:
            observation = LiberoRGBDSensor(self.environment, camera).capture()
            self.assertEqual(observation.rgb.shape, (128, 128, 3))
            self.assertTrue(np.all(np.isfinite(observation.depth_m)))
            self.assertTrue(np.all(observation.depth_m > 0))
            np.testing.assert_allclose(observation.world_T_camera @ observation.camera_T_world,
                                       np.eye(4), atol=1e-8)
