import unittest
from unittest.mock import patch

from scripts.run_libero_confidence_experiment import (
    _evaluate_with_retry,
    _is_retryable_provider_error,
)


class FakeRateLimitError(RuntimeError):
    status_code = 429


class LiberoConfidenceExperimentTests(unittest.TestCase):
    def test_rate_limit_is_retryable(self):
        self.assertTrue(_is_retryable_provider_error(FakeRateLimitError()))

    @patch("scripts.run_libero_confidence_experiment.time.sleep")
    @patch("scripts.run_libero_confidence_experiment.evaluate_samples")
    def test_transient_failure_retries_same_sample(self, evaluate, sleep):
        expected = {"sample_id": "trial_001"}
        evaluate.side_effect = [FakeRateLimitError(), [expected]]

        result = _evaluate_with_retry(
            {"sample_id": "trial_001"}, object(), max_attempts=2
        )

        self.assertEqual(result, expected)
        self.assertEqual(evaluate.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("scripts.run_libero_confidence_experiment.evaluate_samples")
    def test_nontransient_failure_is_not_retried(self, evaluate):
        evaluate.side_effect = ValueError("invalid sample")

        with self.assertRaisesRegex(ValueError, "invalid sample"):
            _evaluate_with_retry(
                {"sample_id": "trial_001"}, object(), max_attempts=3
            )

        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
