import unittest

import numpy as np

from scripts.analyze_vlm_likelihood_calibration import (
    expected_calibration_error,
    probability_metrics,
    reliability_bins,
)


def records(correctness):
    return [
        {
            "raw_log_probability": float(np.log(score)),
            "raw_association_likelihood": score,
            "correct": bool(outcome),
        }
        for score, outcome in zip((0.2, 0.4, 0.6, 0.8), correctness)
    ]


class VLMLikelihoodCalibrationTests(unittest.TestCase):
    def test_quantile_reliability_bins_preserve_all_samples(self):
        bins = reliability_bins(
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            np.asarray([0, 0, 1, 1]),
            bin_count=2,
            strategy="quantile",
        )

        self.assertEqual(sum(item["count"] for item in bins), 4)
        self.assertAlmostEqual(bins[0]["mean_score"], 0.15)
        self.assertAlmostEqual(bins[1]["empirical_accuracy"], 1.0)

    def test_expected_calibration_error_is_count_weighted(self):
        bins = [
            {"count": 3, "mean_score": 0.8, "empirical_accuracy": 1.0},
            {"count": 1, "mean_score": 0.6, "empirical_accuracy": 0.0},
        ]

        self.assertAlmostEqual(expected_calibration_error(bins), 0.3)

    def test_probability_metrics_report_ranking_brier_and_both_eces(self):
        metrics = probability_metrics(
            records((False, False, True, True)),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
        )

        self.assertAlmostEqual(metrics["correctness_ranking_auroc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.025)
        self.assertIn("ece_quantile_10", metrics)
        self.assertIn("ece_equal_width_10", metrics)


if __name__ == "__main__":
    unittest.main()
