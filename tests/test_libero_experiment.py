import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.setup_libero_experiment import main as setup_experiment
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

    def test_state_zero_episode_paths_separate_every_task_and_method(self):
        paths = set()
        for task_index, object_name, _ in LIBERO_OBJECT_TASKS:
            vla = episode_result_dir(VLA_METHOD_FOLDER, task_index, 0)
            proposed = episode_result_dir(PROPOSED_METHOD_FOLDER, task_index, 0)

            self.assertEqual(
                vla,
                EXPERIMENT_ROOT
                / "VLA"
                / f"task_{task_index:02d}_{object_name}"
                / "init_state_00",
            )
            self.assertEqual(
                proposed,
                EXPERIMENT_ROOT
                / "proposed_framework"
                / f"task_{task_index:02d}_{object_name}"
                / "init_state_00",
            )
            self.assertNotEqual(vla, proposed)
            paths.update((vla, proposed))

        self.assertEqual(len(paths), 20)

    def test_task_index_and_initial_state_index_are_independent(self):
        task_zero_state_nine = episode_result_dir(VLA_METHOD_FOLDER, 0, 9)
        task_nine_state_zero = episode_result_dir(VLA_METHOD_FOLDER, 9, 0)

        self.assertIn("task_00_alphabet_soup", str(task_zero_state_nine))
        self.assertIn("init_state_09", str(task_zero_state_nine))
        self.assertIn("task_09_orange_juice", str(task_nine_state_zero))
        self.assertIn("init_state_00", str(task_nine_state_zero))
        self.assertNotEqual(task_zero_state_nine, task_nine_state_zero)

    def test_resolution_trial_path_preserves_the_completed_method_results(self):
        trial = episode_result_dir(
            PROPOSED_METHOD_FOLDER,
            0,
            0,
            run_variant="768x768_instruction_bound_prompt",
        )

        self.assertEqual(
            trial,
            EXPERIMENT_ROOT
            / "proposed_framework"
            / "_resolution_trials"
            / "768x768_instruction_bound_prompt"
            / "task_00_alphabet_soup"
            / "init_state_00",
        )

    def test_episode_path_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown experiment method"):
            episode_result_dir("hybrid", 7, 0)

    def test_setup_manifest_distinguishes_tasks_from_fixed_states(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "experiment"
            with patch(
                "scripts.setup_libero_experiment.EXPERIMENT_ROOT", root
            ), patch("builtins.print"):
                setup_experiment()
            manifest = json.loads(
                (root / "experiment_manifest.json").read_text(encoding="utf-8")
            )

        protocol = manifest["shared_protocol"]
        self.assertEqual(protocol["task_indices"], list(range(10)))
        self.assertEqual(protocol["initial_state_indices_per_task"], [0])
        self.assertIn("task_index", protocol["index_definition"])
        self.assertIn("initial_state_index", protocol["index_definition"])
        self.assertEqual(
            manifest["primary_metric"]["name"], "binary_task_success_rate"
        )
        self.assertFalse(
            manifest["primary_metric"]["speed_or_action_count_used"]
        )
        self.assertEqual(
            manifest["methods"][PROPOSED_METHOD_FOLDER]["observation_resolution_hw"],
            [512, 512],
        )
        self.assertEqual(
            manifest["methods"][VLA_METHOD_FOLDER]["observation_resolution_hw"],
            [256, 256],
        )
        self.assertNotIn("observation_resolution_hw", protocol)


if __name__ == "__main__":
    unittest.main()
