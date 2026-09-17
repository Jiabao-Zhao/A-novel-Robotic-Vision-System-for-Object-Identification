import unittest

import numpy as np

import cad_object_association as association
from scripts.run_wrist_top5_semantic_ablation import score_view_similarities, template_rotation


class TopFiveSemanticTests(unittest.TestCase):
    def test_top_five_mean_and_separate_best_template(self):
        scores = [-.8, .4, .9, -.2, .6, .7, .1, .8, -.3, .2, 0, -.1, -.4, -.5]
        result = score_view_similarities("pin", scores)
        self.assertAlmostEqual(result["semantic_score"], .68)
        self.assertEqual(result["max_view_score"], .9)
        self.assertEqual(result["top5_view_indices_1based"], [3, 8, 6, 5, 2])
        self.assertEqual(result["best_template_id"], "pin/view_03")
        self.assertEqual(result["per_view_cosines"], scores)
        np.testing.assert_array_equal(result["best_template_rotation_camera_from_cad"], np.diag([1, -1, -1]))

    def test_negative_cosines_and_ties_are_not_clamped_or_deduplicated(self):
        scores = [-.2] * 6 + [-.8] * 8
        result = score_view_similarities("gear", scores)
        self.assertAlmostEqual(result["semantic_score"], -.2)
        self.assertEqual(result["top5_view_indices_1based"], [1, 2, 3, 4, 5])
        self.assertEqual(result["best_template_id"], "gear/view_01")

    def test_rotation_matches_renderer_axes_for_all_views(self):
        for direction in association.VIEW_DIRECTIONS:
            direction = direction / np.linalg.norm(direction)
            right, up = association.CADPointCloudRegistration.table_basis(direction)
            rotation = template_rotation(direction)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            # An image ray advances along +camera z; increasing image rows is -up.
            np.testing.assert_allclose(rotation @ right, [1, 0, 0], atol=1e-12)
            np.testing.assert_allclose(rotation @ -up, [0, 1, 0], atol=1e-12)
            np.testing.assert_allclose(rotation @ -direction, [0, 0, 1], atol=1e-12)

    def test_invalid_scores_fail_instead_of_silently_changing_support(self):
        for scores in ([.5] * 4, [.5] * 13, [float("nan")] * 14, [1.01] * 14):
            with self.assertRaises(ValueError):
                score_view_similarities("pin", scores)


if __name__ == "__main__":
    unittest.main()
