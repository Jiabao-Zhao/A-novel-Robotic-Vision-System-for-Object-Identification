import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.libero_task_execution import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RUN_VARIANT,
    VLM_CONTACT_SHEET_TILE_SIZE_PX,
    _write_pipeline_failure,
    main,
    task_associations_from_vlm_result,
)


def association(target_description, object_id, **updates):
    result = {
        "target_description": target_description,
        "provider": "openai",
        "final_object_id": object_id,
        "requires_human_clarification": False,
        "resolution": "vlm_accepted",
    }
    result.update(updates)
    return result


class LiberoTaskExecutionTests(unittest.TestCase):
    def test_proposed_resolution_ablation_uses_semantic_association(self):
        self.assertEqual((IMAGE_HEIGHT, IMAGE_WIDTH), (768, 768))
        self.assertEqual(VLM_CONTACT_SHEET_TILE_SIZE_PX, 448)
        self.assertEqual(
            RUN_VARIANT,
            "768x768_semantic_association_grasp_pose_v5",
        )

    def test_task_uses_two_independent_semantic_associations(self):
        payload = {
            "associations": [
                association("basket", "object_001"),
                association("milk", "object_003"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            result = task_associations_from_vlm_result(result_path, "milk")

        self.assertEqual(result["target_object_id"], "object_003")
        self.assertEqual(result["basket_object_id"], "object_001")
        self.assertNotIn("instruction_role", json.dumps(result))

    def test_task_rejects_missing_semantic_target(self):
        payload = {"associations": [association("basket", "object_001")]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "independent associations"):
                task_associations_from_vlm_result(result_path, "milk")

    def test_task_rejects_unresolved_association(self):
        payload = {
            "associations": [
                association(
                    "milk",
                    None,
                    requires_human_clarification=True,
                    resolution="confidence_unavailable",
                ),
                association("basket", "object_001"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "human clarification"):
                task_associations_from_vlm_result(result_path, "milk")

    def test_task_rejects_target_not_present(self):
        payload = {
            "associations": [
                association("milk", None, resolution="target_not_present"),
                association("basket", "object_001"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Target not present"):
                task_associations_from_vlm_result(result_path, "milk")

    def test_task_rejects_same_object_for_both_associations(self):
        payload = {
            "associations": [
                association("butter", "object_002"),
                association("basket", "object_002"),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "vlm_result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "same localized object"):
                task_associations_from_vlm_result(result_path, "butter")

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
                    error=RuntimeError("association failed"),
                )
            payload = json.loads(
                (output_root / "episode.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["run_status"], "pipeline_error")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["termination_reason"], "pipeline_error")
        self.assertEqual(payload["initial_state_sha256"], "a" * 64)
        self.assertIn("association failed", payload["execution_error"])

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
