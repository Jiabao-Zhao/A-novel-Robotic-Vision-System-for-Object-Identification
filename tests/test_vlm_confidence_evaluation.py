import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_vlm_confidence import (
    DEFAULT_THRESHOLDS,
    evaluate_samples,
    load_evaluation_manifest,
    save_evaluation,
    threshold_sweep,
)


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def associate(self, image_path, prompt):
        self.calls += 1
        return {
            "provider": "test",
            "model": "test-vlm",
            "generated_text": "B",
            "token_logprobs": [
                {
                    "token": "B",
                    "log_probability": math.log(0.8),
                    "top_logprobs": [
                        {"token": "A", "log_probability": math.log(0.1)},
                        {"token": "B", "log_probability": math.log(0.8)},
                        {"token": "N", "log_probability": math.log(0.1)},
                    ],
                }
            ],
            "logprob_error": None,
            "top_logprob_error": "not requested",
        }


class VLMConfidenceEvaluationTests(unittest.TestCase):
    def test_manifest_resolves_saved_scene_paths_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "calibration.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "split": "calibration",
                        "samples": [
                            {
                                "sample_id": "trial_001",
                                "image_path": "scene/rgb.png",
                                "localization_path": "scene/localization.json",
                                "target_description": "white gear",
                                "ground_truth_object_id": "object_002",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset_split, samples = load_evaluation_manifest(manifest_path)
            expected_image_path = (root / "scene/rgb.png").resolve()

        self.assertEqual(dataset_split, "calibration")
        self.assertEqual(samples[0]["sample_id"], "trial_001")
        self.assertEqual(samples[0]["image_path"], expected_image_path)

    def test_evaluation_runs_one_raw_likelihood_association_without_human(self):
        localization = {
            "frame": "camera",
            "objects": [
                {
                    "object_id": "object_001",
                    "roi": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                },
                {
                    "object_id": "object_002",
                    "roi": {"x1": 5, "y1": 6, "x2": 7, "y2": 8},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            localization_path = Path(temporary_directory) / "localization.json"
            localization_path.write_text(json.dumps(localization), encoding="utf-8")
            provider = FakeProvider()
            records = evaluate_samples(
                [
                    {
                        "sample_id": "trial_001",
                        "image_path": Path(temporary_directory) / "rgb.png",
                        "localization_path": localization_path,
                        "target_description": "white gear",
                        "ground_truth_object_id": "object_002",
                    }
                ],
                provider,
            )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(records[0]["predicted_object_id"], "object_002")
        self.assertTrue(records[0]["correct"])
        self.assertAlmostEqual(records[0]["association_score"], 0.8)
        self.assertAlmostEqual(records[0]["raw_log_probability"], math.log(0.8))
        self.assertEqual(records[0]["score_type"], "raw_label_likelihood")

    def test_threshold_metrics_use_only_records_with_numeric_scores(self):
        records = [
            {"association_score": 0.9, "correct": True},
            {"association_score": 0.7, "correct": False},
            {"association_score": 0.4, "correct": True},
            {"association_score": None, "correct": False},
        ]

        metrics = threshold_sweep(records, thresholds=(0.7,))[0]

        self.assertEqual(metrics["total_trials"], 4)
        self.assertEqual(metrics["scored_trials"], 3)
        self.assertEqual(metrics["score_unavailable_trials"], 1)
        self.assertEqual(metrics["autonomous_decisions"], 2)
        self.assertEqual(metrics["threshold_deferred_scored_decisions"], 1)
        self.assertEqual(metrics["total_deferred_decisions"], 2)
        self.assertAlmostEqual(metrics["autonomous_coverage"], 2 / 4)
        self.assertAlmostEqual(metrics["deferral_rate"], 2 / 4)
        self.assertAlmostEqual(metrics["autonomous_coverage_scored_trials"], 2 / 3)
        self.assertAlmostEqual(
            metrics["threshold_deferral_rate_scored_trials"], 1 / 3
        )
        self.assertAlmostEqual(metrics["autonomous_accuracy"], 0.5)
        self.assertEqual(metrics["false_autonomous_acceptance_count"], 1)
        self.assertAlmostEqual(
            metrics["false_autonomous_acceptance_rate_all_trials"],
            1 / 4,
        )
        self.assertAlmostEqual(
            metrics["false_autonomous_acceptance_rate_all_scored_trials"],
            1 / 3,
        )
        self.assertAlmostEqual(
            metrics["false_autonomous_acceptance_rate_autonomous"],
            0.5,
        )

    def test_threshold_sweep_on_saved_scores_never_reruns_vlm(self):
        records = [{"association_score": 0.8, "correct": True}]

        with patch(
            "scripts.evaluate_vlm_confidence.infer_target_association"
        ) as inference:
            metrics = threshold_sweep(records, thresholds=(0.5, 0.9))

        inference.assert_not_called()
        self.assertEqual(metrics[0]["autonomous_decisions"], 1)
        self.assertEqual(metrics[1]["total_deferred_decisions"], 1)

    def test_saved_evaluation_keeps_split_separate_from_threshold_selection(self):
        records = [
            {
                "sample_id": "trial_001",
                "target_description": "white gear",
                "provider": "test",
                "model": "test-vlm",
                "predicted_object_id": "object_002",
                "ground_truth_object_id": "object_002",
                "correct": True,
                "raw_log_probability": math.log(0.9),
                "raw_association_likelihood": 0.9,
                "association_score": 0.9,
                "score_type": "raw_label_likelihood",
                "candidate_scores": None,
                "association_margin": None,
                "score_unavailable_reason": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, _, sweep_path = save_evaluation(
                records, "calibration", temporary_directory
            )
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))

        self.assertEqual(sweep["dataset_split"], "calibration")
        self.assertFalse(sweep["threshold_selection_performed"])
        self.assertEqual(len(sweep["metrics"]), len(DEFAULT_THRESHOLDS))


if __name__ == "__main__":
    unittest.main()
