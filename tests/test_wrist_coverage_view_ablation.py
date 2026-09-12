import copy
import unittest

from scripts.run_wrist_coverage_view_ablation import score_views, select_views


def record(view, normalized_cls, coverage):
    return {"cad_id": "cad", "object_id": "object", "view_index_1based": view,
        "cls_cosine": 2 * normalized_cls - 1, "normalized_cls": normalized_cls,
        "C_obs": 0. if coverage is None else coverage,
        "C_cad": coverage, "C_f1": coverage, "coverage_status": "no_alignment" if coverage is None else "aligned",
        "geometry_raw": {"T_observed_from_cad": None if coverage is None else
                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}


class CoverageViewTests(unittest.TestCase):
    def test_geometry_and_fusion_select_own_views_without_combining_independent_maxima(self):
        records = [record(1, .95, .1), record(2, .2, 1.), record(3, .8, .9)]
        before = copy.deepcopy(records)
        result = select_views(score_views(records))
        self.assertEqual(result["geometry_winning_view_index_1based"], 2)
        self.assertEqual(result["fused_winning_view_index_1based"], 3)
        self.assertEqual(result["geometry_score"], 1.)
        self.assertAlmostEqual(result["fused_score"], .85)
        self.assertNotEqual(result["fused_score"], .5 * result["visual_score"] + .5 * result["geometry_score"])
        self.assertEqual(records, before)

    def test_null_alignment_remains_null_and_cannot_win_even_with_high_cls(self):
        rows = score_views([record(1, 1., None), record(2, .2, .2)])
        self.assertIsNone(rows[0]["fused_C_f1"])
        result = select_views(rows)
        self.assertEqual(result["geometry_winning_view_index_1based"], 2)
        self.assertEqual(result["fused_winning_view_index_1based"], 2)
        self.assertEqual((result["defined_view_count"], result["undefined_view_count"]), (1, 1))

    def test_exact_ties_choose_lowest_index_regardless_of_input_order(self):
        result = select_views(score_views([record(9, .4, .8), record(3, .4, .8)]))
        self.assertEqual(result["geometry_winning_view_index_1based"], 3)
        self.assertEqual(result["fused_winning_view_index_1based"], 3)
        self.assertEqual(result["geometry_max_tie_count"], 2)
        self.assertEqual(result["fused_max_tie_count"], 2)

    def test_defined_zero_is_retained_but_all_missing_cannot_produce_a_score(self):
        result = select_views(score_views([record(1, .8, 0.)]))
        self.assertEqual(result["geometry_score"], 0.)
        self.assertEqual(result["defined_view_count"], 1)
        with self.assertRaisesRegex(ValueError, "no defined saved C_f1"):
            select_views(score_views([record(1, .8, None)]))


if __name__ == "__main__":
    unittest.main()
