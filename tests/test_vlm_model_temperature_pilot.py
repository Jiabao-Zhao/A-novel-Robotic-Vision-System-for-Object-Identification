import math
import unittest
from unittest.mock import patch

from scripts.run_vlm_model_temperature_pilot import (
    CONDITIONS, MODELS, arguments_for, call_key, condition_order,
    first_position_logprobs, response_record,
)
from scripts.run_vlm_output_format_ablation import request_arguments
from scripts.analyze_vlm_model_temperature_pilot import (
    gap_comparison, same_token_rows, summarize_condition, validate_records,
)


def provider_response(token="A", logprob=-.2, alternatives=()):
    return {"model": MODELS[0], "choices": [{"finish_reason": "stop",
        "message": {"content": token}, "logprobs": {"content": [
            {"token": token, "logprob": logprob, "top_logprobs": list(alternatives)}]}}]}


def scene():
    return {"sample_id": "scene1", "ground_truth_object_id": "object_001",
            "candidate_map": {"A": "object_001", "B": "object_002", "N": None}}


class ModelTemperaturePilotTests(unittest.TestCase):
    def test_only_model_and_temperature_change_in_request(self):
        base = request_arguments(b"frozen bytes", "frozen prompt")
        for model, temperature in CONDITIONS:
            actual = arguments_for(b"frozen bytes", "frozen prompt", model, temperature)
            self.assertEqual(actual, {**base, "model": model, "temperature": temperature})

    def test_every_repeat_has_each_condition_once(self):
        sample = {"task_index": 5, "initial_state_index": 1}
        for repeat in range(3):
            self.assertCountEqual(condition_order(sample, repeat), CONDITIONS)
        self.assertNotEqual(condition_order(sample, 0), condition_order(sample, 1))

    def test_cache_keys_distinguish_repeats_and_conditions(self):
        keys = {call_key("s", m, t, r) for m, t in CONDITIONS for r in range(3)}
        self.assertEqual(len(keys), 18)

    def test_exact_token_variants_remain_separate(self):
        raw = provider_response(alternatives=[{"token": " A", "logprob": -2},
                                              {"token": "B", "logprob": -3}])
        result = first_position_logprobs(raw, scene()["candidate_map"])
        self.assertEqual(result, {"A": -.2, " A": -2, "B": -3})
        self.assertNotIn("N", result)  # Missing alternatives are not zero-filled.

    def test_no_search_into_later_position_after_formatting(self):
        raw = provider_response("\n", -.1)
        raw["choices"][0]["logprobs"]["content"].append({"token": "A", "logprob": -.2})
        self.assertEqual(first_position_logprobs(raw, scene()["candidate_map"]), {})

    def test_prediction_is_generated_choice_and_score_is_raw(self):
        raw = provider_response("B", -1, [{"token": "A", "logprob": -.1}])
        row = response_record(scene(), MODELS[0], .5, 0, 0, raw)
        self.assertEqual(row["predicted_object_id"], "object_002")
        self.assertFalse(row["correct"])
        self.assertAlmostEqual(row["raw_sequence_likelihood"], math.exp(-1))

    def test_no_fake_score_when_logprobs_missing(self):
        raw = provider_response()
        del raw["choices"][0]["logprobs"]
        row = response_record(scene(), MODELS[0], 0, 0, 0, raw)
        self.assertIsNone(row["raw_sequence_likelihood"])
        self.assertEqual(row["first_position_choice_logprobs"], {})

    def test_none_is_a_valid_prediction(self):
        row = response_record({**scene(), "ground_truth_object_id": None},
                              MODELS[0], 0, 0, 0, provider_response("N"))
        self.assertTrue(row["correct"])
        self.assertIsNone(row["predicted_object_id"])

    def test_truncated_response_is_not_scored(self):
        raw = provider_response()
        raw["choices"][0]["finish_reason"] = "length"
        row = response_record(scene(), MODELS[0], 0, 0, 0, raw)
        self.assertIsNone(row["raw_sequence_likelihood"])
        self.assertIn("length", row["score_unavailable_reason"])

    def test_same_token_comparison_uses_intersection_not_zero_fill(self):
        a = {"sample_id": "s", "model": MODELS[0], "temperature": 0, "repeat": 0,
             "first_position_choice_logprobs": {"A": -.2, "B": -2}}
        b = {**a, "temperature": 1, "first_position_choice_logprobs": {"A": -.3, "N": -4}}
        rows = same_token_rows(a, b, "temperature")
        self.assertEqual([r["token"] for r in rows], ["A"])
        self.assertAlmostEqual(rows[0]["logprob_delta_b_minus_a"], -.1)

    def test_logodds_distinguishes_scaled_from_unchanged(self):
        high = {"sample_id": "s", "model": MODELS[0], "repeat": 0,
                "first_position_choice_logprobs": {"A": -.1, "B": -2.1}}
        unchanged = gap_comparison(high, high)
        self.assertEqual(unchanged["ratio"], 1)
        scaled = gap_comparison({**high, "first_position_choice_logprobs": {"A": -.01, "B": -4.01}}, high)
        self.assertAlmostEqual(scaled["ratio"], 2)
        self.assertAlmostEqual(scaled["absolute_error_temperature_scaled"], 0)

    def test_gap_requires_two_distinct_choices_available_in_both(self):
        high = {"sample_id": "s", "model": MODELS[0], "repeat": 0,
                "first_position_choice_logprobs": {"A": -.1, " A": -.2, "B": -2.1}}
        low = {**high, "first_position_choice_logprobs": {"A": -.1, " A": -.2}}
        self.assertIsNone(gap_comparison(low, high))

    def test_analysis_metrics_use_saved_scores_without_inference(self):
        rows = [{"sample_id": f"s{i}", "correct": correct, "choice": "A",
                 "raw_sequence_likelihood": score,
                 "raw_log_probability": math.log(score) if score is not None else None,
                 "decision_token_count": 1}
                for i, (correct, score) in enumerate(((True, .9), (False, .7), (False, None)))]
        with patch("scripts.run_vlm_model_temperature_pilot.paced_completion", side_effect=AssertionError("No API")):
            stats, curve = summarize_condition(rows)
        p = next(p for p in stats["thresholds"] if p["threshold"] == .9)
        self.assertEqual(p["autonomous_decisions"], 1)
        self.assertEqual(p["autonomous_accuracy"], 1)
        self.assertAlmostEqual(p["autonomous_coverage"], 1/3)
        self.assertEqual(p["deferred_decisions"], 2)

    def test_saved_grid_rejects_missing_condition(self):
        s = {**scene(), "partition": "calibration"}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_records({"protocol": {"samples": [s]}, "records": []})

    def test_saved_grid_rejects_modified_input(self):
        s = {**scene(), "partition": "calibration", "image_sha256": "original"}
        row = response_record(s, MODELS[0], 0, 0, 0, provider_response())
        row["image_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Input differs"):
            validate_records({"protocol": {"samples": [s]}, "records": [row]})

    def test_complete_grid_verifies_raw_responses(self):
        s = {**scene(), "partition": "calibration"}
        rows = []
        for m, t in CONDITIONS:
            for r in range(3):
                rows.append(response_record(s, m, t, r, 0, {**provider_response(), "model": m}))
        self.assertEqual(len(validate_records({"protocol": {"samples": [s]}, "records": rows})), 18)


if __name__ == "__main__":
    unittest.main()
