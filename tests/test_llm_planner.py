import json
import tempfile
import unittest
from pathlib import Path

from LLM_planner import load_perception_context
from prompt import _llm_planner_prompt


class LLMPlannerBoundaryTests(unittest.TestCase):
    def test_planner_receives_original_instruction_and_resolved_identities(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            associations_path = root / "associations.json"
            registration_path = root / "registration.json"
            pose_path = root / "pose.json"
            associations_path.write_text(
                json.dumps(
                    {
                        "associations": [
                            {
                                "target_description": "white gear",
                                "final_object_id": "object_002",
                                "resolution": "human_corrected",
                            },
                            {
                                "target_description": "red block",
                                "final_object_id": "object_004",
                                "resolution": "vlm_accepted",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registration_path.write_text("{}", encoding="utf-8")
            pose_path.write_text("{}", encoding="utf-8")
            context = load_perception_context(
                associations_path,
                registration_path,
                pose_path,
            )

        self.assertEqual(
            context["resolved_objects"],
            [
                {
                    "target_description": "white gear",
                    "object_id": "object_002",
                    "resolution": "human_corrected",
                },
                {
                    "target_description": "red block",
                    "object_id": "object_004",
                    "resolution": "vlm_accepted",
                },
            ],
        )
        prompt = _llm_planner_prompt(
            "put the white gear on top of the red block",
            context,
            [],
        )
        self.assertIn("put the white gear on top of the red block", prompt)
        self.assertIn("object_002", prompt)
        self.assertIn("object_004", prompt)


if __name__ == "__main__":
    unittest.main()
