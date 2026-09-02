import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compare_libero_results import build_comparison
from scripts.libero_vla_eval import (
    CHECKPOINT,
    CHECKPOINT_REVISION,
    build_eval_command,
)


def _proposed_result():
    return {
        "method": "rgbd_vlm_cad_scripted_controller",
        "schema_version": 1,
        "suite": "libero_object",
        "task_index": 7,
        "instruction": "pick up the milk and place it in the basket",
        "seed": 1000,
        "initial_state_index": 0,
        "initial_state_sha256": "a" * 64,
        "episode_horizon_steps": 280,
        "control_frequency_hz": 20,
        "control_mode": "relative",
        "observation_resolution_hw": [256, 256],
        "initial_physics_settle_steps": 10,
        "success_predicate": "LIBERO task environment check_success()",
        "success": True,
        "termination_reason": "success",
        "first_success_step": 173,
        "action_steps": 173,
    }


def _vla_manifest():
    return {
        "method": "smolvla_official_lerobot_evaluator",
        "schema_version": 1,
        "suite": "libero_object",
        "task_index": 7,
        "instruction": "pick up the milk and place it in the basket",
        "seed": 1000,
        "episode_seeds": [1000],
        "initial_state_indices": [0],
        "initial_state_sha256s": ["a" * 64],
        "episode_horizon_steps": 280,
        "control_frequency_hz": 20,
        "control_mode": "relative",
        "observation_resolution_hw": [256, 256],
        "settle_steps_before_policy": 10,
        "success_predicate": "LIBERO task environment check_success()",
        "return_code": 0,
        "rollout_step_counts": [132],
        "first_success_steps": [None],
        "initial_state_hash_error": None,
        "rollout_metrics_error": None,
    }


def _vla_eval():
    return {
        "per_task": [
            {
                "task_group": "libero_object",
                "task_id": 7,
                "metrics": {"successes": [False]},
            }
        ],
        "overall": {"n_episodes": 1},
    }


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

    def test_build_comparison_accepts_matched_episode(self):
        comparison = build_comparison(
            _proposed_result(), _vla_manifest(), _vla_eval()
        )

        self.assertTrue(comparison["comparison_valid"])
        self.assertEqual(len(comparison["episodes"]), 2)
        self.assertEqual(
            comparison["methods"]["rgbd_vlm_cad_scripted_controller"]["successes"],
            1,
        )
        self.assertEqual(
            comparison["methods"]["smolvla_official_lerobot_evaluator"]["successes"],
            0,
        )
        self.assertEqual(comparison["episodes"][1]["action_steps"], 132)

    def test_build_comparison_rejects_initial_state_mismatch(self):
        manifest = _vla_manifest()
        manifest["initial_state_indices"] = [1]

        with self.assertRaisesRegex(ValueError, "initial_state"):
            build_comparison(_proposed_result(), manifest, _vla_eval())

    def test_build_comparison_rejects_success_step_on_failed_episode(self):
        manifest = _vla_manifest()
        manifest["first_success_steps"] = [132]

        with self.assertRaisesRegex(ValueError, "failed but records"):
            build_comparison(_proposed_result(), manifest, _vla_eval())


if __name__ == "__main__":
    unittest.main()
