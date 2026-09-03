import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compare_libero_results import build_comparison
from scripts.libero_vla_eval import (
    ALL_TASK_INDICES,
    CHECKPOINT,
    CHECKPOINT_REVISION,
    build_eval_command,
)
from simulation.libero_experiment import LIBERO_OBJECT_TASKS


PROPOSED_METHOD = "rgbd_vlm_cad_scripted_controller"
VLA_METHOD = "smolvla_official_lerobot_evaluator"


def _state_hash(task_index):
    return hashlib.sha256(f"fixed-state-{task_index}".encode()).hexdigest()


def _proposed_result(task_index, success=True):
    _, _, instruction = LIBERO_OBJECT_TASKS[task_index]
    return {
        "method": PROPOSED_METHOD,
        "schema_version": 1,
        "suite": "libero_object",
        "task_index": task_index,
        "instruction": instruction,
        "seed": 1000,
        "initial_state_index": 0,
        "initial_state_sha256": _state_hash(task_index),
        "episode_horizon_steps": 280,
        "control_frequency_hz": 20,
        "control_mode": "relative",
        "observation_resolution_hw": [256, 256],
        "initial_physics_settle_steps": 10,
        "success_predicate": "LIBERO task environment check_success()",
        "success": success,
        "termination_reason": "success" if success else "episode_horizon",
    }


def _vla_manifest(task_index, success=True, scalar_aliases=False):
    _, _, instruction = LIBERO_OBJECT_TASKS[task_index]
    manifest = {
        "method": VLA_METHOD,
        "schema_version": 1,
        "suite": "libero_object",
        "task_index": task_index,
        "instruction": instruction,
        "seed": 1000,
        "episode_seeds": [1000],
        "initial_state_indices": [0],
        "initial_state_sha256s": [_state_hash(task_index)],
        "episode_horizon_steps": 280,
        "control_frequency_hz": 20,
        "control_mode": "relative",
        "observation_resolution_hw": [256, 256],
        "settle_steps_before_policy": 10,
        "success_predicate": "LIBERO task environment check_success()",
        "return_code": 0,
        "initial_state_hash_error": None,
    }
    if scalar_aliases:
        manifest.update(
            {
                "episode_seed": 1000,
                "initial_state_index": 0,
                "initial_state_sha256": _state_hash(task_index),
                "success": success,
            }
        )
    return manifest


def _vla_eval(task_index, success=True):
    return {
        "per_task": [
            {
                "task_group": "libero_object",
                "task_id": task_index,
                "metrics": {"successes": [success]},
            }
        ],
        "overall": {"n_episodes": 1},
    }


def _all_results(scalar_aliases=False):
    proposed = []
    vla = []
    for task_index, _, _ in LIBERO_OBJECT_TASKS:
        proposed_success = task_index % 2 == 0
        vla_success = task_index in {0, 1, 2, 3}
        proposed.append(_proposed_result(task_index, proposed_success))
        vla.append(
            {
                "manifest": _vla_manifest(
                    task_index,
                    success=vla_success,
                    scalar_aliases=scalar_aliases,
                ),
                "eval": _vla_eval(task_index, vla_success),
            }
        )
    return proposed, vla


class LiberoVlaComparisonTests(unittest.TestCase):
    def test_vla_command_uses_pinned_official_checkpoint_and_no_depth_inputs(self):
        command = build_eval_command(Path("outputs/test_vla"))
        command_text = " ".join(command)

        self.assertIn(f"--policy.path={CHECKPOINT}", command)
        self.assertIn(
            f"--policy.pretrained_revision={CHECKPOINT_REVISION}", command
        )
        self.assertIn("--env.task_ids=[7]", command)
        self.assertIn("--env.control_mode=relative", command)
        self.assertIn("--env.episode_length=280", command)
        self.assertNotIn("depth", command_text.lower())
        self.assertNotIn("cad", command_text.lower())

    def test_vla_command_rejects_nonsequential_initial_state_claim(self):
        with patch("scripts.libero_vla_eval.INITIAL_STATE_INDICES", [3]):
            with self.assertRaisesRegex(ValueError, "selects sequential initial states"):
                build_eval_command(Path("outputs/test_vla"))

    def test_vla_command_can_request_one_state_for_all_ten_tasks(self):
        command = build_eval_command(
            Path("outputs/test_vla_all_products"),
            task_indices=ALL_TASK_INDICES,
        )

        self.assertIn("--env.task_ids=[0,1,2,3,4,5,6,7,8,9]", command)
        self.assertIn("--eval.n_episodes=1", command)

    def test_build_comparison_aggregates_all_ten_tasks_by_success_only(self):
        proposed, vla = _all_results()

        comparison = build_comparison(proposed, vla)

        self.assertTrue(comparison["comparison_valid"])
        self.assertEqual(comparison["coverage"]["matched_task_count"], 10)
        self.assertEqual(comparison["coverage"]["episodes_per_method"], 10)
        self.assertEqual(len(comparison["tasks"]), 10)
        self.assertEqual(len(comparison["episodes"]), 20)
        self.assertEqual(comparison["methods"][PROPOSED_METHOD]["successes"], 5)
        self.assertEqual(comparison["methods"][PROPOSED_METHOD]["success_rate"], 0.5)
        self.assertEqual(comparison["methods"][VLA_METHOD]["successes"], 4)
        self.assertEqual(comparison["methods"][VLA_METHOD]["success_rate"], 0.4)
        self.assertFalse(
            comparison["primary_metric"]["speed_or_action_count_used"]
        )
        for row in comparison["episodes"]:
            self.assertNotIn("action_steps", row)
            self.assertNotIn("first_success_step", row)
            artifact_parts = Path(row["artifact"]).parts
            expected_folder = (
                "proposed_framework" if row["method"] == PROPOSED_METHOD else "VLA"
            )
            self.assertIn(expected_folder, artifact_parts)

    def test_build_comparison_accepts_vla_scalar_convenience_aliases(self):
        proposed, vla = _all_results(scalar_aliases=True)

        comparison = build_comparison(proposed, vla)

        self.assertTrue(comparison["comparison_valid"])

    def test_build_comparison_counts_persisted_pipeline_error_as_failure(self):
        proposed, vla = _all_results()
        proposed[6].update(
            {
                "run_status": "pipeline_error",
                "success": False,
                "termination_reason": "pipeline_error",
                "execution_error": "RuntimeError: grounding failed",
            }
        )

        comparison = build_comparison(proposed, vla)

        self.assertEqual(comparison["methods"][PROPOSED_METHOD]["successes"], 4)
        task_six_row = next(
            row
            for row in comparison["episodes"]
            if row["method"] == PROPOSED_METHOD and row["task_index"] == 6
        )
        self.assertEqual(task_six_row["run_status"], "pipeline_error")
        self.assertFalse(task_six_row["success"])

    def test_build_comparison_rejects_incomplete_task_coverage(self):
        proposed, vla = _all_results()
        proposed.pop()

        with self.assertRaisesRegex(ValueError, r"coverage mismatch.*missing=\[9\]"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_duplicate_task_result(self):
        proposed, vla = _all_results()
        proposed.append(dict(proposed[0]))

        with self.assertRaisesRegex(ValueError, "Duplicate proposed method result"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_nonzero_initial_state(self):
        proposed, vla = _all_results()
        vla[3]["manifest"]["initial_state_indices"] = [1]

        with self.assertRaisesRegex(ValueError, "initial_state_index_is_zero"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_initial_state_hash_mismatch(self):
        proposed, vla = _all_results()
        vla[4]["manifest"]["initial_state_sha256s"] = ["a" * 64]

        with self.assertRaisesRegex(ValueError, "initial_state_bytes"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_instruction_mismatch(self):
        proposed, vla = _all_results()
        vla[8]["manifest"]["instruction"] = "pick up the wrong product"

        with self.assertRaisesRegex(ValueError, "instruction"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_protocol_mismatch(self):
        proposed, vla = _all_results()
        vla[2]["manifest"]["observation_resolution_hw"] = [128, 128]

        with self.assertRaisesRegex(ValueError, "observation_resolution_hw"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_mutually_equal_but_wrong_protocol(self):
        proposed, vla = _all_results()
        proposed[6]["episode_horizon_steps"] = 999
        vla[6]["manifest"]["episode_horizon_steps"] = 999

        with self.assertRaisesRegex(ValueError, "episode_horizon_steps"):
            build_comparison(proposed, vla)

    def test_build_comparison_rejects_manifest_eval_success_disagreement(self):
        proposed, vla = _all_results(scalar_aliases=True)
        vla[5]["manifest"]["success"] = not vla[5]["manifest"]["success"]

        with self.assertRaisesRegex(ValueError, "success disagrees"):
            build_comparison(proposed, vla)


if __name__ == "__main__":
    unittest.main()
