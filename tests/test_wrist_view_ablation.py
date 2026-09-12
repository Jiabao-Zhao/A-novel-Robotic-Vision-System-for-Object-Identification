import unittest

import numpy as np

from scripts.run_wrist_patch_ablation import score_pair
from scripts.run_wrist_view_ablation import score_all_views, select_views, view_disagreements


class ViewSelectionTests(unittest.TestCase):
    def test_combined_can_select_a_third_view_and_does_not_average_independent_maxima(self):
        cls, patch = np.full(14, -1.), np.full(14, -1.)
        cls[:3], patch[:3] = [.9, .1, .65], [.1, .9, .65]
        scores, winners, _ = select_views(cls, patch)
        self.assertEqual(winners, {"cls_only": 1, "patch_only": 2, "cls_patch": 3})
        self.assertEqual(scores, {"cls_only": .9, "patch_only": .9, "cls_patch": .65})
        self.assertNotEqual(scores["cls_patch"], .5 * cls.max() + .5 * patch.max())

    def test_exact_ties_choose_first_view_and_negative_scores_are_retained(self):
        scores, winners, _ = select_views(np.full(14, -.2), np.full(14, -.6))
        self.assertTrue(all(index == 1 for index in winners.values()))
        self.assertEqual(scores, {"cls_only": -.2, "patch_only": -.6, "cls_patch": -.4})

    def test_all_view_patch_matching_escapes_cls_choice_and_keeps_original_score(self):
        observed_cls = np.array([1., 0.])
        observed = np.zeros((16, 16, 2))
        observed[0, :2] = [[1, 0], [0, 1]]
        weights = np.zeros((16, 16))
        weights[0, :2] = [.25, .75]
        cad_cls = np.tile([0., 1.], (14, 1))
        cad_cls[0] = [1, 0]
        cad = np.zeros((14, 16, 16, 2))
        cad[:, 0, :2] = [[1, 0], [0, 1]]
        cad[0, 0, 0] = [-1, 0]
        cad_weights = np.zeros((14, 16, 16))
        cad_weights[:, 0, 0] = 1
        cad_weights[1, 0, 1] = 1
        args = (observed_cls, observed, weights, cad_cls, cad, cad_weights)
        original, result = score_pair(*args), score_all_views(*args)
        self.assertEqual(len(result["patch_view_scores"]), 14)
        self.assertEqual(result["patch_view_scores"][0], original["patch_score"])
        np.testing.assert_array_equal(result["patch_view_scores"], [-.25, 1, *([.25] * 12)])
        self.assertEqual(result["winning_view_index_1based"], {"cls_only": 1, "patch_only": 2, "cls_patch": 2})
        self.assertEqual(result["patch_score"], 1)
        self.assertEqual(result["cls_patch_score"], .5)

    def test_view_disagreements_distinguish_pairwise_and_three_way_changes(self):
        pairs = [{"winning_view_index_1based": dict(zip(("cls_only", "patch_only", "cls_patch"), views))}
                 for views in ((1, 1, 1), (1, 2, 3), (1, 2, 1))]
        self.assertEqual(view_disagreements(pairs), {
            "pair_count": 3, "cls_only_vs_patch_only": 2, "cls_only_vs_cls_patch": 1,
            "patch_only_vs_cls_patch": 2, "any_difference": 2, "all_three_different": 1})


if __name__ == "__main__":
    unittest.main()
