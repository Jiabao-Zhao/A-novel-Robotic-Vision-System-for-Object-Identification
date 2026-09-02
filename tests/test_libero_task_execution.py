import json
import tempfile
import unittest
from pathlib import Path

from scripts.libero_task_execution import task_roles_from_vlm_result


class LiberoTaskExecutionTests(unittest.TestCase):
    def test_task_roles_require_milk_and_basket_roles(self):
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
            grounding = task_roles_from_vlm_result(result_path)

        self.assertEqual(grounding["provider"], "gemini")
        self.assertEqual(grounding["milk_object_id"], "object_003")
        self.assertEqual(grounding["basket_object_id"], "object_001")

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
                task_roles_from_vlm_result(result_path)


if __name__ == "__main__":
    unittest.main()
