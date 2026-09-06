import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.run_vlm_candidate_verification_pilot import (
    VERIFICATION_SYSTEM, call_arguments, scene_prediction, verification_gate,
    verification_prompt, verification_score, run_scene, MODEL,
)
from scripts.run_vlm_output_format_ablation import request_arguments
from scripts.analyze_vlm_candidate_verification_pilot import gate_metrics, matched_coverage, sweep


def response(answer="Y", ly=-.1, ln=-3):
    return {"choices": [{"finish_reason": "stop", "message": {"content": answer},
                         "logprobs": {"content": [{"token": answer,
                         "logprob": ly if answer == "Y" else ln,
                         "top_logprobs": [{"token": "Y", "logprob": ly},
                                          {"token": "N", "logprob": ln}]}]}}]}


class CandidateVerificationTests(unittest.TestCase):
    def test_confident_no_scores_yes_not_generated_no(self):
        score = verification_score(response("N", -12, -.001))
        self.assertEqual(score["answer_label"], "N")
        self.assertAlmostEqual(score["yes_score"], 1 / (1 + math.exp(11.999)))
        self.assertLess(score["yes_score"], .00001)

    def test_stable_normalization_of_very_small_probabilities(self):
        self.assertAlmostEqual(verification_score(response("Y", -1000, -1001))["yes_score"],
                               1 / (1 + math.exp(-1)))

    def test_missing_alternative_is_unavailable_not_zero(self):
        raw = response()
        raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = []
        self.assertIsNone(verification_score(raw)["yes_score"])

    def test_invalid_or_sentinel_alternative_is_unavailable(self):
        for value in (None, -9999, -10000, float("nan"), float("inf"), .01):
            with self.subTest(value=value):
                self.assertIsNone(verification_score(response("Y", -.1, value))["yes_score"])

    def test_leading_whitespace_uses_same_answer_position(self):
        raw = response("Y", -.1, -2)
        raw["choices"][0]["message"]["content"] = "\nY"
        raw["choices"][0]["logprobs"]["content"].insert(0, {"token": "\n", "logprob": -10,
            "top_logprobs": [{"token": "N", "logprob": -.001}]})
        score = verification_score(raw)
        self.assertEqual(score["answer_token_position"], 1)
        self.assertEqual(score["no_log_probability"], -2)
        self.assertAlmostEqual(score["yes_score"], 1 / (1 + math.exp(-1.9)))

    def test_different_token_spellings_are_not_merged(self):
        raw = response()
        raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] = " N"
        self.assertIsNone(verification_score(raw)["yes_score"])

    def test_matching_whitespace_affixes_are_supported(self):
        raw = response()
        raw["choices"][0]["message"]["content"] = " Y"
        token = raw["choices"][0]["logprobs"]["content"][0]
        token["token"] = " Y"
        for item in token["top_logprobs"]:
            item["token"] = " " + item["token"]
        self.assertIsNotNone(verification_score(raw)["yes_score"])

    def test_invalid_answer_and_truncation_are_unavailable(self):
        for answer in ("Yes", "Y. explanation", "B", ""):
            self.assertIsNone(verification_score(response(answer))["yes_score"])
        raw = response()
        raw["choices"][0]["finish_reason"] = "length"
        self.assertIsNone(verification_score(raw)["yes_score"])

    def test_mismatched_token_text_is_unavailable(self):
        raw = response()
        raw["choices"][0]["message"]["content"] = "N"
        self.assertIsNone(verification_score(raw)["yes_score"])

    def test_one_missing_candidate_defers_whole_scene(self):
        p = scene_prediction({"object_001": .99, "object_002": None})
        self.assertIsNone(p["support"])
        self.assertIsNone(p["predicted_object_id"])
        self.assertEqual(verification_gate(p, 0, 0), "deferred")

    def test_exact_ties_always_defer(self):
        p = scene_prediction({"object_001": 1, "object_002": 1})
        self.assertEqual(p["gap"], 0)
        self.assertTrue(p["top_tie"])
        self.assertIsNone(p["predicted_object_id"])
        self.assertEqual(verification_gate(p, 0, 0), "deferred")

    def test_support_and_gap_both_required_inclusive(self):
        p = scene_prediction({"object_001": .875, "object_002": .625})
        self.assertEqual(p["predicted_object_id"], "object_001")
        self.assertEqual(verification_gate(p, .875, .25), "vlm_accepted")
        self.assertEqual(verification_gate(p, .876, .25), "deferred")
        self.assertEqual(verification_gate(p, .875, .251), "deferred")

    def test_low_support_is_not_proven_absence(self):
        p = scene_prediction({"object_001": .1, "object_002": .01})
        self.assertEqual(p["predicted_object_id"], "object_001")
        self.assertEqual(verification_gate(p, .75), "deferred")

    def test_baseline_request_is_unchanged(self):
        self.assertEqual(call_arguments(b"same image", "same prompt"),
                         request_arguments(b"same image", "same prompt"))

    def test_verification_only_changes_task_instruction(self):
        base = request_arguments(b"same image", "verify prompt")
        actual = call_arguments(b"same image", "verify prompt", True)
        base["messages"][0]["content"] = VERIFICATION_SYSTEM
        self.assertEqual(actual, base)
        self.assertNotIn("logit_bias", actual)

    def test_verification_contains_no_baseline_or_ground_truth(self):
        candidates = [{"choice": "A", "object_id": "object_001", "centroid_3d": [1, 2, 3]}]
        prompt = verification_prompt("milk", "object_001", candidates)
        self.assertIn('"centroid_3d":[1,2,3]', prompt)
        self.assertIn("Multiple candidates may match, or none may match.", prompt)
        self.assertNotIn("ground_truth", prompt)
        self.assertNotIn("baseline", prompt)

    def test_resume_reuses_all_successful_calls(self):
        scene = {"sample_id": "s", "task_index": 0, "initial_state_index": 0,
                 "target_description": "milk", "candidate_metadata": [{"object_id": "object_001"}],
                 "candidate_map": {"A": "object_001", "N": None}, "prompt": "baseline",
                 "input_sha256": "input", "image_sha256": "image",
                 "ground_truth_object_id": "object_001"}
        baseline = response()
        baseline["model"] = MODEL
        baseline["choices"][0]["message"]["content"] = "A"
        baseline["choices"][0]["logprobs"]["content"][0]["token"] = "A"
        verify = {**response(), "model": MODEL}
        completions = [MagicMock(), MagicMock()]
        for completion, raw in zip(completions, (baseline, verify)):
            completion.model_dump.return_value = raw
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "responses").mkdir()
            (root / "scenes").mkdir()
            with patch("scripts.run_vlm_candidate_verification_pilot.OUTPUT_ROOT", root), \
                    patch("openai.OpenAI"), \
                    patch("scripts.run_vlm_candidate_verification_pilot.paced_completion", side_effect=completions) as call:
                first = run_scene(scene, b"image")
                # Fresh-reset pilots have no initial-state index and may save
                # to a different experiment directory without changing requests.
                del scene["initial_state_index"]
                scene["sample_index"] = 0
                second = run_scene(scene, b"image", output_root=root)
                self.assertEqual(first, second)
                self.assertEqual(call.call_count, 2)
                with self.assertRaisesRegex(ValueError, "Resume input/settings mismatch"):
                    run_scene({**scene, "input_sha256": "changed"}, b"image")
                saved = json.loads((root / "responses" / "s_object_001.json").read_text())
                self.assertEqual(saved["provider_response"], verify)


def metric_records():
    records = []
    for index, (correct, baseline_score, candidates) in enumerate((
        (True, .9, {"a": .95, "b": .1}),
        (False, .8, {"a": .95, "b": .94}),
        (True, .7, {"a": .8, "b": .1}),
        (False, None, {"a": .99, "b": None}),
    )):
        records.append({"scene": {"sample_id": str(index)},
                        "baseline": {"correct": correct, "raw_sequence_likelihood": baseline_score},
                        "verification": {**scene_prediction(candidates), "correct": correct}})
    return records


class VerificationAnalysisTests(unittest.TestCase):
    def test_same_coverage_comparison_counts_false_acceptances(self):
        rows = metric_records()
        base = gate_metrics(rows, "baseline", .8)
        dual = gate_metrics(rows, "support_gap", .8, .1)
        self.assertEqual(base["autonomous_coverage"], dual["autonomous_coverage"])
        self.assertEqual(base["false_autonomous_acceptance_count"], 1)
        self.assertEqual(dual["false_autonomous_acceptance_count"], 0)
        self.assertEqual(dual["autonomous_accuracy"], 1)
        self.assertEqual(dual["score_unavailable_trials"], 1)
        self.assertEqual(dual["baseline_errors_deferred"], 2)

    def test_missing_scores_count_as_deferred_for_coverage(self):
        point = gate_metrics(metric_records(), "support", 0)
        self.assertEqual(point["autonomous_coverage"], .75)
        self.assertEqual(point["deferred_decisions"], 1)
        self.assertAlmostEqual(point["false_autonomous_acceptance_rate_all_trials"], .25)
        self.assertAlmostEqual(point["false_autonomous_acceptance_rate_autonomous"], 1 / 3)

    def test_sweep_is_offline_and_does_not_split_ties(self):
        with patch("scripts.run_vlm_candidate_verification_pilot.paced_completion", side_effect=AssertionError("No API")):
            points = sweep(metric_records())
            _, matches = matched_coverage(points)
        for p in points:
            if p["method"] == "support":
                self.assertNotEqual(p["autonomous_decisions"], 1)  # .95/.95 support tie
        for row in matches:
            self.assertEqual({p["autonomous_decisions"] for p in row["methods"].values()}, {row["accepted_count"]})

    def test_exact_top_tie_is_deferred_by_support_only_too(self):
        rows = metric_records()[:1]
        rows[0]["verification"].update(scene_prediction({"a": 1, "b": 1}))
        point = gate_metrics(rows, "support", 0)
        self.assertEqual(point["autonomous_decisions"], 0)
        self.assertEqual(point["score_available_trials"], 1)


if __name__ == "__main__":
    unittest.main()
