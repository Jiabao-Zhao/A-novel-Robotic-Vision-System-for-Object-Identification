import unittest
from types import SimpleNamespace

import numpy as np

from simulation.libero_env import LiberoIntegrationError, LiberoTaskEnvironment


class _FakeSuite:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.float64)
        self.load_calls = 0

    def get_task_init_states(self, task_index):
        self.load_calls += 1
        if task_index != 7:
            raise AssertionError(f"Unexpected task index: {task_index}")
        return self.states.copy()


class _FakeLiberoEnv:
    def __init__(self):
        self.calls = []
        self.selected_state = None

    def seed(self, seed):
        self.calls.append(("seed", seed))

    def reset(self):
        self.calls.append(("reset", None))
        return {"source": "ordinary_reset"}

    def set_init_state(self, state):
        self.selected_state = np.asarray(state).copy()
        self.calls.append(("set_init_state", self.selected_state.copy()))
        return {"source": "fixed_state", "state": self.selected_state.copy()}


def _environment(states):
    environment = LiberoTaskEnvironment.__new__(LiberoTaskEnvironment)
    environment.suite_name = "libero_object"
    environment.task_index = 7
    environment.suite = _FakeSuite(states)
    environment.env = _FakeLiberoEnv()
    environment.last_observation = None
    environment.last_reset_seed = None
    environment.last_init_state_index = None
    environment.last_init_state_sha256 = None
    environment.control_mode = None
    environment._task_initial_states = None
    return environment


class LiberoEnvironmentTests(unittest.TestCase):
    def test_reset_selects_and_records_official_fixed_state(self):
        states = np.arange(15, dtype=np.float64).reshape(3, 5)
        environment = _environment(states)

        observation = environment.reset(seed=1000, init_state_index=1)

        np.testing.assert_array_equal(observation["state"], states[1])
        np.testing.assert_array_equal(environment.env.selected_state, states[1])
        self.assertEqual(environment.last_reset_seed, 1000)
        self.assertEqual(environment.last_init_state_index, 1)
        self.assertEqual(len(environment.last_init_state_sha256), 64)
        self.assertEqual(environment.initial_state_count, 3)
        self.assertEqual(environment.suite.load_calls, 1)
        self.assertEqual([call[0] for call in environment.env.calls], [
            "seed",
            "reset",
            "set_init_state",
        ])

    def test_reset_rejects_out_of_range_fixed_state(self):
        environment = _environment(np.zeros((2, 5)))

        with self.assertRaisesRegex(LiberoIntegrationError, "outside"):
            environment.reset(init_state_index=2)

    def test_reset_without_index_preserves_ordinary_reset(self):
        environment = _environment(np.zeros((2, 5)))

        observation = environment.reset()

        self.assertEqual(observation["source"], "ordinary_reset")
        self.assertIsNone(environment.last_init_state_index)
        self.assertIsNone(environment.last_init_state_sha256)
        self.assertEqual(environment.suite.load_calls, 0)

    def test_set_control_mode_explicitly_updates_robosuite_controller(self):
        environment = _environment(np.zeros((2, 5)))
        controller = SimpleNamespace(use_delta=False)
        environment.env.robots = [SimpleNamespace(controller=controller)]

        environment.set_control_mode("relative")

        self.assertTrue(controller.use_delta)
        self.assertEqual(environment.control_mode, "relative")


if __name__ == "__main__":
    unittest.main()
