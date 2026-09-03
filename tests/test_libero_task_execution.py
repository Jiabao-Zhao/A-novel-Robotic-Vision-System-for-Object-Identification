import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.libero_task_execution import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VLM_CONTACT_SHEET_TILE_SIZE_PX,
    _write_pipeline_failure,
    main,
    task_roles_from_vlm_result,
)


class LiberoTaskExecutionTests(unittest.TestCase):
    def test_proposed_resolution_ablation_uses_high_resolution_input(self):
        self.assertEqual((IMAGE_HEIGHT, IMAGE_WIDTH), (512, 512))
        self.assertEqual(VLM_CONTACT_SHEET_TILE_SIZE_PX, 320)

    def test_task_roles_require_one_target_and_basket_role(self):
        result = {
            "provider": "gemini",
            "normalized_result": {
                "needs_human_clarification": False,
                "selected_objects": [
                    {
                        "object_id": "object_001",
                        "object_type": "basket",
                        "instruction_role": "reference_object",
                    },
                    {
                        "object_id": "object_003",
                        "object_type": "milk carton",
                        "instruction_role": "moved_object",
                    },
                ],
                "object_evaluations": [],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            grounding = task_roles_from_vlm_result(result_path, "milk")

        self.assertEqual(grounding["provider"], "gemini")
        self.assertEqual(grounding["target_name"], "milk")
        self.assertEqual(grounding["target_object_id"], "object_003")
        self.assertEqual(grounding["target_object_type"], "milk carton")
        self.assertEqual(grounding["basket_object_id"], "object_001")

    def test_task_roles_support_non_milk_product(self):
        result = {
            "provider": "gemini",
            "normalized_result": {
                "needs_human_clarification": False,
                "selected_objects": [
                    {
                        "object_id": "object_006",
                        "object_type": "box of butter",
                        "instruction_role": "moved_object",
                    },
                    {
                        "object_id": "object_001",
                        "object_type": "woven basket",
                        "instruction_role": "reference_object",
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            grounding = task_roles_from_vlm_result(result_path, "butter")

        self.assertEqual(grounding["target_object_id"], "object_006")
        self.assertEqual(grounding["target_object_type"], "box of butter")
        self.assertEqual(grounding["basket_object_id"], "object_001")

    def test_task_roles_reject_vlm_type_that_is_not_configured_target(self):
        result = {
            "provider": "gemini",
            "normalized_result": {
                "needs_human_clarification": False,
                "selected_objects": [
                    {
                        "object_id": "object_006",
                        "object_type": "cream cheese package",
                        "instruction_role": "moved_object",
                    },
                    {
                        "object_id": "object_001",
                        "object_type": "woven basket",
                        "instruction_role": "reference_object",
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "does not match the configured target",
            ):
                task_roles_from_vlm_result(result_path, "butter")

    def test_task_roles_reject_missing_moved_object(self):
        result = {
            "provider": "gemini",
            "normalized_result": {
                "needs_human_clarification": False,
                "selected_objects": [
                    {
                        "object_id": "object_001",
                        "object_type": "basket",
                        "instruction_role": "reference_object",
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly one milk"):
                task_roles_from_vlm_result(result_path, "milk")

    def test_task_roles_reject_same_object_for_both_roles(self):
        result = {
            "provider": "gemini",
            "normalized_result": {
                "needs_human_clarification": False,
                "selected_objects": [
                    {
                        "object_id": "object_002",
                        "object_type": "butter",
                        "instruction_role": "moved_object",
                    },
                    {
                        "object_id": "object_002",
                        "object_type": "basket",
                        "instruction_role": "reference_object",
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "same localized object"):
                task_roles_from_vlm_result(result_path, "butter")

    def test_pre_control_failure_is_preserved_as_unsuccessful_episode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with patch(
                "scripts.libero_task_execution._official_initial_state_sha256",
                return_value="a" * 64,
            ):
                _write_pipeline_failure(
                    output_root=output_root,
                    task_index=6,
                    target_slug="butter",
                    instruction="pick up the butter and place it in the basket",
                    error=RuntimeError("grounding failed"),
                )
            payload = json.loads(
                (output_root / "episode.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["run_status"], "pipeline_error")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["termination_reason"], "pipeline_error")
        self.assertEqual(payload["initial_state_sha256"], "a" * 64)
        self.assertIn("grounding failed", payload["execution_error"])

    def test_existing_task_result_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "init_state_00"
            output_root.mkdir()
            episode_path = output_root / "episode.json"
            episode_path.write_text('{"success": true}', encoding="utf-8")
            with patch(
                "scripts.libero_task_execution.episode_result_dir",
                return_value=output_root,
            ), patch("scripts.libero_task_execution._run_task") as run_task:
                with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                    main(7)

            run_task.assert_not_called()
            self.assertEqual(
                episode_path.read_text(encoding="utf-8"),
                '{"success": true}',
            )

    def test_main_persists_pre_control_failure_for_comparison(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "init_state_00"
            with patch(
                "scripts.libero_task_execution.episode_result_dir",
                return_value=output_root,
            ), patch(
                "scripts.libero_task_execution._run_task",
                side_effect=RuntimeError("CAD registration failed"),
            ), patch(
                "scripts.libero_task_execution._official_initial_state_sha256",
                return_value="b" * 64,
            ):
                with self.assertRaisesRegex(RuntimeError, "CAD registration failed"):
                    main(6)

            payload = json.loads(
                (output_root / "episode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["task_index"], 6)
            self.assertEqual(payload["run_status"], "pipeline_error")
            self.assertFalse(payload["success"])
            self.assertEqual(payload["initial_state_sha256"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
