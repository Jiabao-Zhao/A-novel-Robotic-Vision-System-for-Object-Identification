import ast
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_libero_new_object_verification import (
    TARGETS, OLD_TARGETS, prepare_scene, selected_tasks, table_workspace_bounds,
    target_ground_truth,
)
from scripts.run_vlm_output_format_ablation import digest


class NewObjectVerificationTests(unittest.TestCase):
    def test_one_new_category_per_scene(self):
        categories = [row[0] for row in TARGETS]
        self.assertEqual(len(categories), 13)
        self.assertEqual(len(set(categories)), len(categories))
        self.assertFalse(set(categories) & OLD_TARGETS)

    def test_fresh_reset_never_requests_saved_state(self):
        # Source-level guard in addition to the runtime assertion on every capture.
        tree = ast.parse(Path("scripts/run_libero_new_object_verification.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([arg.arg for arg in calls[0].keywords], ["seed"])
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in
                             ("set_init_state", "initial_state_count", "get_task_init_states")
                             for node in ast.walk(tree)))

    def test_arena_offset_translates_crop_not_object_geometry(self):
        lower, upper = table_workspace_bounds([0, 0, .9])
        self.assertEqual(lower[:2], [-.25, -.35])
        self.assertEqual(upper[:2], [.35, .35])
        self.assertAlmostEqual(lower[2], .88)
        self.assertAlmostEqual(upper[2], 1.13)
        lower, upper = table_workspace_bounds([0, 0, 0])
        self.assertEqual(lower, [-.25, -.35, -.02])
        self.assertEqual(upper, [.35, .35, .23])

    def test_only_unique_unambiguous_ground_truth_is_accepted(self):
        row = {"object_id": "object_001", "simulator_instance": "bowl_1", "status": "matched"}
        self.assertEqual(target_ground_truth([row], "bowl_1"), "object_001")
        self.assertIsNone(target_ground_truth([row], "mug_1"))
        self.assertIsNone(target_ground_truth([{**row, "status": "ambiguous_possible_split"}], "bowl_1"))
        self.assertIsNone(target_ground_truth([row, {**row, "object_id": "object_002"}], "bowl_1"))

    def test_unresolved_localization_is_not_an_absent_vlm_target(self):
        with self.assertRaisesRegex(ValueError, "Unresolved localization"):
            prepare_scene({"status": "target_localization_unresolved", "ground_truth_object_id": None})

    def test_settings_and_task_mapping_do_not_use_state_indices(self):
        inventory = {category: [{"suite": suite, "task_index": index}]
                     for category, _, suite, index in TARGETS}
        tasks = selected_tasks(inventory)
        self.assertEqual([t["reset_seed"] for t in tasks], list(range(5000, 5013)))
        self.assertFalse(any("initial_state_index" in task for task in tasks))

    def test_prompt_excludes_simulator_truth_and_capture_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.png"
            image.write_bytes(b"frozen test image")
            localization = root / "localization.json"
            localization.write_text(json.dumps({"objects": [{
                "object_id": "object_001", "roi": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
                "centroid_3d_m": [0, 0, 1], "size_3d_m": [.1, .1, .1],
            }]}))
            sample = {"status": "ready", "ground_truth_object_id": "object_001",
                      "target_instance": "secret_simulator_body", "target_description": "white mug",
                      "image_path": str(image), "localization_path": str(localization),
                      "files_sha256": {p.name: digest(p.read_bytes()) for p in (image, localization)}}
            scene, image_bytes = prepare_scene(sample)
            self.assertEqual(image_bytes, b"frozen test image")
            self.assertNotIn("secret_simulator_body", scene["prompt"])
            self.assertNotIn("ground_truth", scene["prompt"])
            self.assertEqual(scene["candidate_map"], {"A": "object_001", "N": None})
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen capture changed"):
                prepare_scene(sample)


if __name__ == "__main__":
    unittest.main()
