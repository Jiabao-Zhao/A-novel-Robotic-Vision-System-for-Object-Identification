import unittest

from scripts.analyze_vlm_uncertainty_scores import (
    risk_coverage_curve,
    select_development_operating_point,
    threshold_metrics,
)


class VLMUncertaintyAnalysisTests(unittest.TestCase):
    def test_unavailable_score_is_operationally_deferred(self):
        records = [
            {"score": 0.9, "correct": True},
            {"score": 0.8, "correct": False},
            {"score": None, "correct": True},
        ]

        result = threshold_metrics(records, "score", 0.85)

        self.assertEqual(result["autonomous_decisions"], 1)
        self.assertEqual(result["deferred_decisions"], 2)
        self.assertAlmostEqual(result["autonomous_coverage"], 1 / 3)
        self.assertEqual(result["autonomous_accuracy"], 1.0)

    def test_operating_point_must_meet_accuracy_on_both_development_splits(self):
        calibration = [
            {"score": 0.9, "correct": True},
            {"score": 0.8, "correct": False},
        ]
        validation = [
            {"score": 0.95, "correct": True},
            {"score": 0.85, "correct": False},
        ]

        result = select_development_operating_point(
            calibration, validation, "score", 0.90
        )

        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["calibration"]["autonomous_accuracy"], 1.0)
        self.assertEqual(result["validation"]["autonomous_accuracy"], 1.0)

    def test_risk_coverage_curve_preserves_fixed_correctness_labels(self):
        records = [
            {"score": 0.9, "correct": True},
            {"score": 0.8, "correct": False},
            {"score": None, "correct": False},
        ]

        result = risk_coverage_curve(records, "score")

        self.assertAlmostEqual(result["attainable_coverage"], 2 / 3)
        self.assertEqual(result["points"][0]["autonomous_accuracy"], 1.0)
        self.assertEqual(result["points"][1]["autonomous_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
