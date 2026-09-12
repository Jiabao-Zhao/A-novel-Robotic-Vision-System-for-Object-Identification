import copy
import unittest

from scripts.run_wrist_coverage_ablation import frozen_view_records, rescore


class FrozenCoverageTests(unittest.TestCase):
    def setUp(self):
        raw = {"T_observed_from_cad": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
               "registration_fitness": 1.}
        self.row = {"cad_id": "cad", "object_id": "object", "view_index_1based": 2,
                    "cls_cosine": .2, "normalized_cls": .6, "geometry_fitness": 1.,
                    "C_obs": 1., "C_cad": .5, "C_f1": 2 / 3, "fused_score": .8,
                    "geometry_raw": raw, "coverage_status": "aligned"}
        pair = {"cad_id": "cad", "object_id": "object", "winning_view_index_1based": 2,
                "visual_score": .99, "geometry_score": 1., "normalized_cls_at_fused_view": .6,
                "fused_score": .8, "geometry_raw_at_fused_view": copy.deepcopy(raw),
                "C_obs": 1., "C_cad": .5, "C_f1": 2 / 3}
        self.prior = {"associations": [{"candidate_ranking": [pair]}]}

    def test_saved_fused_view_and_its_cls_are_frozen_even_if_another_view_scores_higher(self):
        other = {**self.row, "view_index_1based": 1, "normalized_cls": .99, "fused_score": .995, "C_f1": 1.}
        before = copy.deepcopy(self.row)
        records = frozen_view_records(self.prior, [other, self.row])
        a, b = rescore(records, "C_obs")[0], rescore(records, "C_f1")[0]
        self.assertEqual(a["frozen_view_index_1based"], b["frozen_view_index_1based"])
        self.assertEqual(a["frozen_view_index_1based"], 2)
        self.assertEqual(a["visual_score"], b["visual_score"])
        self.assertEqual(a["visual_score"], .6)
        self.assertEqual(a["fused_score"], .8)
        self.assertAlmostEqual(b["geometry_score"], 2 / 3)
        self.assertAlmostEqual(b["fused_score"], .3 + 1 / 3)
        self.assertEqual(self.row, before)

    def test_missing_coverage_is_rejected_instead_of_imputed_or_reselecting_view(self):
        self.row["C_f1"] = None
        with self.assertRaisesRegex(ValueError, "missing/invalid coverage"):
            frozen_view_records(self.prior, [self.row])

    def test_inconsistent_saved_harmonic_coverage_is_rejected(self):
        self.row["C_f1"] = .9
        with self.assertRaisesRegex(ValueError, "harmonic coverage is inconsistent"):
            frozen_view_records(self.prior, [self.row])


if __name__ == "__main__":
    unittest.main()
