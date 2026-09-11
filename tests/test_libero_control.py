import unittest
from types import SimpleNamespace

import numpy as np

from simulation.libero_control import (
    FINGER_ENGAGEMENT_DEPTH_M,
    MINIMUM_GRIP_SITE_Z_M,
    OPEN_GRIPPER,
    execute_top_grasp_and_place,
    hold_gripper,
    grasp_position_tolerance_m,
    move_eef_to_pose,
    pick_object,
    top_down_grasp_pose,
)


class _FakeEnvironment:
    action_dim = 7

    def __init__(self):
        controller = SimpleNamespace(output_max=np.full(6, 0.05))
        self.robots = [SimpleNamespace(controller=controller)]
        self.position = np.array([0.0, 0.0, 0.2], dtype=float)
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float).copy())
        self.position += np.asarray(action[:3], dtype=float) * 0.05
        observation = {"robot0_eef_pos": self.position.copy()}
        observation["robot0_eef_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
        return observation, 0.0, False, {}


class LiberoControlTests(unittest.TestCase):
    def test_precision_compensation_removes_static_error_without_relaxing_tolerance(self):
        for compensate in (False, True):
            environment = _FakeEnvironment()
            original_step = environment.step
            def loaded_step(action):
                observation, reward, done, info = original_step(action)
                environment.position[0] -= .0003
                observation['robot0_eef_pos'] = environment.position.copy()
                return observation, reward, done, info
            environment.step = loaded_step
            observation = {'robot0_eef_pos': environment.position.copy(),
                           'robot0_eef_quat': np.array([0., 0., 0., 1.])}
            goal = np.eye(4)
            goal[:3, 3] = [.05, 0., .2]
            if compensate:
                final = move_eef_to_pose(environment, observation, goal, OPEN_GRIPPER, 'precision',
                    tolerance_m=.00015, max_steps=100, compensate_position_bias=True)
                self.assertLessEqual(np.linalg.norm(final['robot0_eef_pos']-goal[:3, 3]), .00015)
            else:
                with self.assertRaises(RuntimeError):
                    move_eef_to_pose(environment, observation, goal, OPEN_GRIPPER, 'precision',
                                    tolerance_m=.00015, max_steps=100)

    def test_pick_waits_for_precise_position_before_closing_on_small_parts(self):
        environment = _FakeEnvironment()
        original_step = environment.step

        def settling_step(action):
            slowed = np.asarray(action).copy()
            slowed[:3] *= .25
            return original_step(slowed)

        environment.step = settling_step
        grasp = np.eye(4)
        grasp[:3, 3] = [.05, -.10, .04]
        closure_errors = []

        def record(phase, step, observation, *_):
            if phase == "close_gripper" and step == 0:
                closure_errors.append(np.linalg.norm(observation["robot0_eef_pos"]-grasp[:3, 3]))

        pick_object(environment, {
            "robot0_eef_pos": environment.position.copy(),
            "robot0_eef_quat": np.array([0., 0., 0., 1.])}, grasp, record,
            position_tolerance_m=grasp_position_tolerance_m(np.eye(4), [.016, .010, .050]))
        self.assertEqual(len(closure_errors), 1)
        self.assertLessEqual(closure_errors[0], .001)

    def test_grasp_precision_uses_cad_dimensions_and_world_up_without_object_names(self):
        self.assertEqual(grasp_position_tolerance_m(np.eye(4), [.016, .010, .050]), .001)
        for extent in ([.022, .019, .010], [.035, .035, .035], [.050, .033, .114]):
            self.assertEqual(grasp_position_tolerance_m(np.eye(4), extent), .008)
        y_up = np.eye(4)
        y_up[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.assertEqual(grasp_position_tolerance_m(y_up, [.016, .050, .010]), .001)
        self.assertEqual(grasp_position_tolerance_m(y_up, [.016, .010, .050]), .008)

    def test_simulation_grip_site_allows_low_profile_tabletop_objects(self):
        self.assertEqual(MINIMUM_GRIP_SITE_Z_M, 0.005)

    def test_move_eef_to_pose_scales_normalized_osc_action(self):
        environment = _FakeEnvironment()
        observation = {
            "robot0_eef_pos": environment.position.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        target_pose = np.eye(4)
        target_pose[:3, 3] = [0.1, 0.05, 0.25]

        result = move_eef_to_pose(
            environment,
            observation,
            world_T_eef_target=target_pose,
            gripper_action=OPEN_GRIPPER,
            phase="test_move",
        )

        np.testing.assert_allclose(result["robot0_eef_pos"], [0.1, 0.05, 0.25])

    def test_move_eef_to_pose_rejects_unsafe_target(self):
        environment = _FakeEnvironment()
        observation = {
            "robot0_eef_pos": environment.position.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        target_pose = np.eye(4)
        target_pose[:3, 3] = [2.0, 0.0, 0.2]

        with self.assertRaisesRegex(ValueError, "Outside configured translation bounds"):
            move_eef_to_pose(
                environment,
                observation,
                world_T_eef_target=target_pose,
                gripper_action=OPEN_GRIPPER,
                phase="unsafe_move",
            )

    def test_hold_gripper_can_settle_scene_before_capture(self):
        environment = _FakeEnvironment()
        observation = {
            "robot0_eef_pos": environment.position.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }

        result = hold_gripper(
            environment,
            observation,
            OPEN_GRIPPER,
            phase="initial_physics_settle",
            steps=10,
        )

        self.assertEqual(len(environment.actions), 10)
        for action in environment.actions:
            np.testing.assert_array_equal(action[:6], np.zeros(6))
            self.assertEqual(action[-1], OPEN_GRIPPER)
        np.testing.assert_allclose(result["robot0_eef_pos"], environment.position)

    def test_pick_and_place_phase_names_are_product_generic(self):
        environment = _FakeEnvironment()
        observation = {
            "robot0_eef_pos": environment.position.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        phases = []
        grasp_pose = np.eye(4)
        grasp_pose[:3, 3] = [0.05, -0.1, 0.04]

        execute_top_grasp_and_place(
            environment,
            observation,
            world_T_grasp=grasp_pose,
            place_xyz_m=(0.05, 0.2, 0.08),
            callback=lambda phase, *_: phases.append(phase),
        )

        self.assertIn("move_above_target", phases)
        self.assertIn("descend_to_target", phases)
        self.assertIn("lift_target", phases)
        self.assertIn("release_target", phases)
        self.assertFalse(any("milk" in phase for phase in phases))

    def test_hold_gripper_resists_drift_instead_of_following_each_new_position(self):
        environment = _FakeEnvironment()
        initial = environment.position.copy()
        original_step = environment.step

        def disturbed_step(action):
            environment.position[0] += .001
            return original_step(action)

        environment.step = disturbed_step
        result = hold_gripper(environment, {
            "robot0_eef_pos": initial.copy(), "robot0_eef_quat": np.array([0., 0., 0., 1.])},
            OPEN_GRIPPER, "hold_under_disturbance", 20)
        self.assertLess(np.linalg.norm(result["robot0_eef_pos"]-initial), .0011)
        self.assertTrue(any(action[0] < 0 for action in environment.actions))

    def test_top_down_grasp_pose_uses_center_for_low_profile_z_up_cad(self):
        world_T_cad = np.eye(4)
        world_T_cad[:3, 3] = [0.1, -0.2, 0.01]
        reference_rotation = np.diag([1.0, -1.0, -1.0])
        grasp_pose = top_down_grasp_pose(
            world_T_cad,
            [0.0, 0.0, 0.035],
            [0.08, 0.04, 0.02],
            "Z",
            reference_rotation,
        )

        np.testing.assert_allclose(grasp_pose[:3, 3], [0.1, -0.2, 0.045])
        np.testing.assert_allclose(grasp_pose[:3, 2], [0.0, 0.0, -1.0])

    def test_top_down_grasp_pose_handles_y_up_cad_and_raises_tall_grasp(self):
        world_T_cad = np.eye(4)
        world_T_cad[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        world_T_cad[:3, 3] = [0.05, -0.1, 0.075]
        reference_rotation = np.diag([1.0, -1.0, -1.0])

        grasp_pose = top_down_grasp_pose(
            world_T_cad,
            [0.0, 0.0, 0.0],
            [0.05, 0.15, 0.04],
            "Y",
            reference_rotation,
        )

        expected_z = 0.075 + 0.075 - FINGER_ENGAGEMENT_DEPTH_M
        np.testing.assert_allclose(grasp_pose[:3, 3], [0.05, -0.1, expected_z])
        np.testing.assert_allclose(grasp_pose[:3, :3], reference_rotation)

    def test_floor_nut_estimate_has_margin_above_controller_minimum(self):
        pose = np.eye(4)
        pose[:3, 3] = [-.283, .137, .004939077619726746]
        grasp = top_down_grasp_pose(pose, [0., 0., 0.], [.022, .019, .010],
                                    "Z", np.diag([1., -1., -1.]))
        np.testing.assert_allclose(grasp[:2, 3], pose[:2, 3])
        self.assertAlmostEqual(grasp[2, 3], .0051)

    def test_large_height_error_still_rejected_by_controller(self):
        pose = np.eye(4)
        pose[2, 3] = .002
        grasp = top_down_grasp_pose(pose, [0., 0., 0.], [.022, .019, .010],
                                    "Z", np.diag([1., -1., -1.]))
        with self.assertRaisesRegex(ValueError, "Outside configured translation bounds"):
            move_eef_to_pose(_FakeEnvironment(), {}, grasp, OPEN_GRIPPER, "invalid_grasp")

    def test_top_down_grasp_pose_selects_equivalent_yaw_nearest_to_robot(self):
        world_T_cad = np.eye(4)
        world_T_cad[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        reference_rotation = np.diag([1.0, -1.0, -1.0])

        grasp_pose = top_down_grasp_pose(
            world_T_cad,
            [0.0, 0.0, 0.01],
            [0.08, 0.04, 0.02],
            "Z",
            reference_rotation,
        )

        np.testing.assert_allclose(grasp_pose[:3, :3], reference_rotation)

    def test_top_down_grasp_pose_raises_medium_height_can_above_center(self):
        reference_rotation = np.diag([1.0, -1.0, -1.0])

        grasp_pose = top_down_grasp_pose(
            np.eye(4),
            [0.0, 0.0, 0.035],
            [0.066, 0.083, 0.070],
            "Z",
            reference_rotation,
        )

        np.testing.assert_allclose(grasp_pose[:3, 3], [0.0, 0.0, 0.045])

    def test_free_grasp_yaw_uses_robot_heading_instead_of_gear_phase(self):
        pose = np.eye(4)
        angle = np.deg2rad(-74.)
        pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        reference = np.diag([1., -1., -1.])
        grasp = top_down_grasp_pose(pose, [0., 0., .01], [.062, .062, .02],
                                    "Z", reference, free_yaw=True)
        np.testing.assert_allclose(grasp[:3, :3], reference)
        np.testing.assert_allclose(grasp[:3, 3], [0., 0., .01])


if __name__ == "__main__":
    unittest.main()
