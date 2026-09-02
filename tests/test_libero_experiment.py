import unittest

from simulation.libero_experiment import (
    EXPERIMENT_ROOT,
    LIBERO_OBJECT_TASKS,
    PROPOSED_METHOD_FOLDER,
    VLA_METHOD_FOLDER,
    episode_result_dir,
)


class LiberoExperimentTests(unittest.TestCase):
    def test_catalog_contains_all_ten_basket_tasks(self):
        self.assertEqual([task[0] for task in LIBERO_OBJECT_TASKS], list(range(10)))
        self.assertEqual(len({task[1] for task in LIBERO_OBJECT_TASKS}), 10)
        for _, _, instruction in LIBERO_OBJECT_TASKS:
            self.assertTrue(instruction.endswith("and place it in the basket"))

    def test_episode_paths_separate_methods(self):
        vla = episode_result_dir(VLA_METHOD_FOLDER, 7, 0)
        proposed = episode_result_dir(PROPOSED_METHOD_FOLDER, 7, 0)

        self.assertEqual(
            vla,
            EXPERIMENT_ROOT / "VLA" / "task_07_milk" / "init_state_00",
        )
        self.assertEqual(
            proposed,
            EXPERIMENT_ROOT
            / "proposed_framework"
            / "task_07_milk"
            / "init_state_00",
        )
        self.assertNotEqual(vla, proposed)

    def test_episode_path_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown experiment method"):
            episode_result_dir("hybrid", 7, 0)


if __name__ == "__main__":
    unittest.main()
