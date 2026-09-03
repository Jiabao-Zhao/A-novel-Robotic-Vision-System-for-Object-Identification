import json
import tempfile
import unittest
from pathlib import Path

from prompt import _vlm_prompt
from vlm_module import load_localized_objects


class VLMModuleTests(unittest.TestCase):
    def test_prompt_binds_identity_labels_to_instruction_terms(self):
        prompt = _vlm_prompt(
            "pick up the alphabet soup and place it in the basket",
            [{"object_id": "object_005", "visual_label": "005"}],
        )

        self.assertIn("Instruction-bound identity-label rule (strict)", prompt)
        self.assertIn('predicted_type must be "alphabet soup"', prompt)
        self.assertIn('never "canned soup"', prompt)
        self.assertIn("Do not classify unmentioned scene objects", prompt)
        self.assertIn(
            '"predicted_type": "exact object name from the instruction or null"',
            prompt,
        )

    def test_localized_prompt_keeps_depth_derived_geometry(self):
        localization = {
            "frame": "camera",
            "camera_intrinsics": {"width": 256, "height": 256},
            "objects": [
                {
                    "object_id": "object_003",
                    "roi": {"x1": 10, "y1": 20, "x2": 30, "y2": 50},
                    "centroid_3d_m": [0.1, -0.2, 1.0],
                    "size_3d_m": [0.05, 0.08, 0.12],
                    "point_count": 420,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "localization.json"
            path.write_text(json.dumps(localization), encoding="utf-8")
            detections = load_localized_objects(path)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["centroid_3d_m"], [0.1, -0.2, 1.0])
        self.assertEqual(detections[0]["size_3d_m"], [0.05, 0.08, 0.12])
        self.assertEqual(detections[0]["point_count"], 420)
        self.assertEqual(detections[0]["geometry_frame"], "camera")


if __name__ == "__main__":
    unittest.main()
