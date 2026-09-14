import unittest

import numpy as np

from scripts.run_wrist_sam_visual_comparison import evaluate, prepare_masked_input, score_pairs


class SamVisualComparisonTests(unittest.TestCase):
    def test_missing_truth_is_an_error_and_has_no_fabricated_score(self):
        cad = np.zeros((14, 384)); cad[:, 0] = 1
        observed = np.zeros(384); observed[0] = 1
        pairs = score_pairs({"pulley": cad}, {"present": observed}, ["missing", "present"])
        self.assertIsNone(pairs[0]["raw_cosine"])
        self.assertEqual(pairs[0]["view_cosines"], [None] * 14)
        result = evaluate(pairs, {"pulley": "missing"}, ["missing", "present"])
        self.assertEqual(result["accuracy"], {"correct": 0, "total": 1, "fraction": 0.0})
        self.assertIsNone(result["targets"][0]["margin"])
        self.assertEqual(result["targets"][0]["prediction"], "present")

    def test_common_candidate_evaluation_removes_same_distractor(self):
        pairs = [{"cad_id": "gear", "object_id": oid, "available": True, "raw_cosine": score}
                 for oid, score in (("correct", .8), ("missing_from_sam", .9), ("other", .4))]
        full = evaluate(pairs, {"gear": "correct"}, [p["object_id"] for p in pairs])
        common = evaluate(pairs, {"gear": "correct"}, ["correct", "other"])
        self.assertEqual(full["accuracy"]["correct"], 0)
        self.assertAlmostEqual(full["targets"][0]["margin"], -.1)
        self.assertEqual(common["accuracy"]["correct"], 1)
        self.assertAlmostEqual(common["targets"][0]["margin"], .4)

    def test_mask_preserves_holes_and_inclusive_edges(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[2, 3] = [100, 50, 25]
        mask = np.zeros((8, 8), dtype=bool)
        mask[2, 3] = mask[4, 5] = True
        canvas, crop, bbox = prepare_masked_input(rgb, mask)
        self.assertEqual(bbox, [3, 2, 5, 4])
        self.assertEqual(crop.shape, (3, 3, 3))
        np.testing.assert_array_equal(crop[0, 0], rgb[2, 3])
        np.testing.assert_array_equal(crop[-1, -1], rgb[4, 5])
        self.assertTrue(np.all(crop[1] == 255))
        self.assertEqual(canvas.shape, (224, 224, 3))
        with self.assertRaises(ValueError):
            prepare_masked_input(rgb, np.zeros((8, 8), dtype=bool))


if __name__ == "__main__":
    unittest.main()
