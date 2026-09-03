import unittest

import numpy as np

from scripts.evaluate_libero_vlm_labels import (
    evaluate_candidate,
    exact_type_matches,
    match_candidates_to_ground_truth,
)


class TestLiberoVLMLabelAudit(unittest.TestCase):
    def test_one_to_one_position_matching_uses_world_xy(self):
        candidates = np.array([[1.01, 1.0, 9.0], [0.01, 0.0, -3.0]])
        truth = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        matches = match_candidates_to_ground_truth(candidates, truth)

        self.assertEqual([(row, column) for row, column, _ in matches], [(0, 1), (1, 0)])
        self.assertTrue(all(distance < 0.02 for _, _, distance in matches))

    def test_exact_type_matching_uses_canonical_cad_aliases(self):
        self.assertTrue(exact_type_matches("orange juice carton", "orange_juice"))
        self.assertTrue(exact_type_matches("woven basket", "basket"))
        self.assertFalse(exact_type_matches("box", "butter"))
        self.assertFalse(exact_type_matches(None, "milk"))

    def test_task_grounding_is_separate_from_exact_distractor_name(self):
        result = evaluate_candidate(
            task_index=0,
            target_type="alphabet_soup",
            candidate={"object_id": "object_004"},
            evaluation={
                "target_match": "not_match",
                "instruction_role": None,
                "predicted_type": "bottle",
                "bbox_2d_xyxy": [1, 2, 3, 4],
            },
            ground_truth={
                "instance_name": "salad_dressing_1",
                "category": "salad_dressing",
            },
            xy_distance_m=0.01,
        )

        self.assertTrue(result["grounding_correct"])
        self.assertFalse(result["exact_type_correct"])

    def test_false_positive_moved_object_is_grounding_error(self):
        result = evaluate_candidate(
            task_index=0,
            target_type="alphabet_soup",
            candidate={"object_id": "object_003"},
            evaluation={
                "target_match": "plausible_match",
                "instruction_role": "moved_object",
                "predicted_type": "canned soup",
            },
            ground_truth={
                "instance_name": "tomato_sauce_1",
                "category": "tomato_sauce",
            },
            xy_distance_m=0.01,
        )

        self.assertFalse(result["grounding_correct"])


if __name__ == "__main__":
    unittest.main()
