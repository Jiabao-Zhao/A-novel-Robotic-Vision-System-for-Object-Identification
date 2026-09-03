import unittest
from types import SimpleNamespace

import numpy as np

from simulation.libero_control import (
    OPEN_GRIPPER,
    execute_top_grasp_and_place,
    hold_gripper,
    move_eef_to,
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
        return observation, 0.0, False, {}


class LiberoControlTests(unittest.TestCase):
    def test_move_eef_to_scales_normalized_osc_action(self):
        environment = _FakeEnvironment()
        observation = {"robot0_eef_pos": environment.position.copy()}

        result = move_eef_to(
            environment,
            observation,
            target_xyz_m=(0.1, 0.05, 0.25),
            gripper_action=OPEN_GRIPPER,
            phase="test_move",
        )

        np.testing.assert_allclose(result["robot0_eef_pos"], [0.1, 0.05, 0.25])

    def test_move_eef_to_rejects_unsafe_target(self):
        environment = _FakeEnvironment()
        observation = {"robot0_eef_pos": environment.position.copy()}

        with self.assertRaisesRegex(ValueError, "Unsafe or unreachable"):
            move_eef_to(
                environment,
                observation,
                target_xyz_m=(2.0, 0.0, 0.2),
                gripper_action=OPEN_GRIPPER,
                phase="unsafe_move",
            )

    def test_hold_gripper_can_settle_scene_before_capture(self):
        environment = _FakeEnvironment()
        observation = {"robot0_eef_pos": environment.position.copy()}

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
        observation = {"robot0_eef_pos": environment.position.copy()}
        phases = []

        execute_top_grasp_and_place(
            environment,
            observation,
            pick_xyz_m=(0.05, -0.1, 0.04),
            place_xyz_m=(0.05, 0.2, 0.08),
            callback=lambda phase, *_: phases.append(phase),
        )

        self.assertIn("move_above_target", phases)
        self.assertIn("descend_to_target", phases)
        self.assertIn("lift_target", phases)
        self.assertIn("release_target", phases)
        self.assertFalse(any("milk" in phase for phase in phases))


if __name__ == "__main__":
    unittest.main()
