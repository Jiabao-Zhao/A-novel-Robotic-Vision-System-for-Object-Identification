import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.libero_vla_eval import (
    ALL_TASK_INDICES,
    CHECKPOINT,
    CHECKPOINT_REVISION,
    METHOD_INPUTS,
    _ensure_publish_targets_are_empty,
    _manifest_matches_current_protocol,
    _normalize_task_indices,
    _parse_batch_results,
    _publish_task_results,
    _task_result_is_complete,
    build_eval_command,
)


class LiberoVlaRunnerTests(unittest.TestCase):
    def test_all_task_command_runs_one_state_zero_episode_per_task(self):
        command = build_eval_command(
            Path("outputs/test_vla_batch"), task_indices=list(reversed(ALL_TASK_INDICES))
        )

        self.assertIn("--env.task_ids=[0,1,2,3,4,5,6,7,8,9]", command)
        self.assertIn("--eval.n_episodes=1", command)
        self.assertIn("--eval.batch_size=1", command)
        self.assertIn("--eval.use_async_envs=false", command)
        self.assertIn("--env.max_parallel_tasks=1", command)
        self.assertIn("--env.init_states=true", command)

    def test_default_command_remains_the_task_seven_pilot(self):
        command = build_eval_command(Path("outputs/test_vla_task7"))
        self.assertIn("--env.task_ids=[7]", command)

    def test_task_indices_reject_duplicates_and_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _normalize_task_indices([0, 0])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            _normalize_task_indices([10])
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            _normalize_task_indices({0, 1})

    def test_completion_manifest_requires_exact_checkpoint_task_and_protocol(self):
        official_hash = "a" * 64
        manifest = {
            "schema_version": 1,
            "method": "smolvla_official_lerobot_evaluator",
            "controller_branch": "vla_baseline",
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_is_libero_finetuned": True,
            "local_finetuning_performed": False,
            "suite": "libero_object",
            "task_index": 0,
            "instruction": "pick up the alphabet soup and place it in the basket",
            "success_predicate": "LIBERO task environment check_success()",
            "seed": 1000,
            "episode_seeds": [1000],
            "initial_state_indices": [0],
            "initial_state_sha256s": [official_hash],
            "initial_state_source": "official LIBERO task .pruned_init array",
            "episode_horizon_steps": 280,
            "control_frequency_hz": 20,
            "control_mode": "relative",
            "observation_resolution_hw": [256, 256],
            "settle_steps_before_policy": 10,
            "method_inputs": list(METHOD_INPUTS),
            "depth_or_cad_inputs_used": False,
            "return_code": 0,
            "initial_state_hash_error": None,
            "rollout_metrics_error": None,
        }
        self.assertTrue(_manifest_matches_current_protocol(manifest, 0, official_hash))

        mismatches = {
            "checkpoint": "different/checkpoint",
            "checkpoint_revision": "different-revision",
            "instruction": "different instruction",
            "seed": 999,
            "episode_seeds": [999],
            "initial_state_indices": [1],
            "initial_state_sha256s": ["b" * 64],
            "episode_horizon_steps": 281,
            "control_frequency_hz": 19,
            "control_mode": "absolute",
            "observation_resolution_hw": [128, 128],
            "settle_steps_before_policy": 9,
        }
        for field, mismatched_value in mismatches.items():
            with self.subTest(field=field):
                candidate = dict(manifest)
                candidate[field] = mismatched_value
                self.assertFalse(
                    _manifest_matches_current_protocol(candidate, 0, official_hash)
                )

    def test_batch_parser_requires_one_result_per_task(self):
        eval_info = {
            "per_task": [
                {
                    "task_group": "libero_object",
                    "task_id": 0,
                    "metrics": {
                        "sum_rewards": [1.0],
                        "max_rewards": [1.0],
                        "successes": [True],
                        "video_paths": ["task0.mp4"],
                        "predicted_video_paths": [],
                    },
                },
                {
                    "task_group": "libero_object",
                    "task_id": 1,
                    "metrics": {
                        "sum_rewards": [0.0],
                        "max_rewards": [0.0],
                        "successes": [False],
                        "video_paths": ["task1.mp4"],
                        "predicted_video_paths": [],
                    },
                },
            ],
            "overall": {"n_episodes": 2},
        }

        def fake_steps(_, task_index):
            return ([20], [20]) if task_index == 0 else ([280], [None])

        with patch(
            "scripts.libero_vla_eval._rollout_step_metrics", side_effect=fake_steps
        ):
            records = _parse_batch_results(eval_info, (0, 1))

        self.assertTrue(records[0]["success"])
        self.assertEqual(records[0]["first_success_step"], 20)
        self.assertFalse(records[1]["success"])
        self.assertIsNone(records[1]["first_success_step"])

        eval_info["overall"]["n_episodes"] = 1
        with self.assertRaisesRegex(ValueError, "overall episode count"):
            _parse_batch_results(eval_info, (0, 1))

    def test_publish_creates_task_package_and_then_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_video = root / "staging" / "videos" / "eval_episode_0.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"test-video")

            def result_dir(_, task_index, initial_state_index):
                self.assertEqual(initial_state_index, 0)
                return root / "published" / f"task_{task_index}"

            records = {
                0: {
                    "metrics": {
                        "sum_rewards": [1.0],
                        "max_rewards": [1.0],
                        "successes": [True],
                        "video_paths": [str(source_video)],
                        "predicted_video_paths": [],
                    },
                    "success": True,
                    "action_steps": 42,
                    "first_success_step": 42,
                }
            }
            batch_manifest = {
                "method_inputs": list(METHOD_INPUTS),
                "rollout_step_count_source": "test frame count",
                "command": ["lerobot-eval"],
                "task_indices": [0],
                "mujoco_gl": "egl",
                "lerobot_version": "test",
                "lerobot_source": {"path": "test"},
                "torch_version": "test",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "elapsed_seconds": 1.0,
                "return_code": 0,
            }

            with patch(
                "scripts.libero_vla_eval.episode_result_dir", side_effect=result_dir
            ), patch(
                "scripts.libero_vla_eval._official_state_zero_sha256",
                return_value="a" * 64,
            ):
                published = _publish_task_results(
                    output_root=root / "staging",
                    batch_manifest=batch_manifest,
                    task_records=records,
                    initial_state_hashes={0: "a" * 64},
                )
                task_output = published[0]
                manifest = json.loads(
                    (task_output / "experiment_manifest.json").read_text(encoding="utf-8")
                )
                eval_info = json.loads(
                    (task_output / "eval_info.json").read_text(encoding="utf-8")
                )

                self.assertEqual(manifest["initial_state_index"], 0)
                self.assertEqual(manifest["initial_state_indices"], [0])
                self.assertEqual(manifest["initial_state_sha256"], "a" * 64)
                self.assertEqual(manifest["episode_seed"], 1000)
                self.assertTrue(manifest["success"])
                self.assertEqual(eval_info["overall"]["n_episodes"], 1)
                self.assertTrue(_task_result_is_complete(0))

                with patch(
                    "scripts.libero_vla_eval._official_state_zero_sha256",
                    return_value="b" * 64,
                ):
                    self.assertFalse(_task_result_is_complete(0))
                with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                    _ensure_publish_targets_are_empty((0,))


if __name__ == "__main__":
    unittest.main()
