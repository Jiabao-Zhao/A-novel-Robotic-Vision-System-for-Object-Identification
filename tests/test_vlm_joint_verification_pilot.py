import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.run_vlm_joint_verification_pilot import (
    JOINT_SYSTEM, MAX_OUTPUT_TOKENS, MODEL, answer_order, joint_arguments,
    joint_prompt, parse_joint_response, run_condition,
)
from scripts.run_vlm_candidate_verification_pilot import scene_prediction, verification_gate
from scripts.run_vlm_output_format_ablation import digest, request_arguments


IDS = ["object_001", "object_002", "object_003"]


def response(answers=("N", "Y", "N"), ids=IDS):
    tokens = []
    for index, (object_id, answer) in enumerate(zip(ids, answers)):
        if index:
            tokens.append({"token": "\n", "logprob": -20, "top_logprobs": []})
        tokens.extend([{"token": object_id, "logprob": -30, "top_logprobs": []},
                       {"token": ":", "logprob": -40, "top_logprobs": []}])
        ly, ln = (-.01, -8) if answer == "Y" else (-8, -.01)
        tokens.append({"token": " " + answer, "logprob": ly if answer == "Y" else ln,
                       "top_logprobs": [{"token": " Y", "logprob": ly}, {"token": " N", "logprob": ln}]})
    return {"model": MODEL, "choices": [{"finish_reason": "stop",
             "message": {"content": "".join(t["token"] for t in tokens)}, "logprobs": {"content": tokens}}]}


def scene():
    return {"sample_id": "s", "target_description": "white mug", "input_sha256": "frozen-input",
            "ground_truth_object_id": "object_002", "target_instance": "evaluation_secret",
            "candidate_metadata": [{"choice": c, "object_id": key, "bbox_2d_xyxy": [1, 2, 3, 4]}
                                   for c, key in zip("ABC", IDS)]}


class JointVerificationTests(unittest.TestCase):
    def test_only_named_answer_token_contributes(self):
        parsed = parse_joint_response(response(), IDS)
        self.assertTrue(parsed["format_valid"])
        b = parsed["candidates"]["object_002"]
        self.assertEqual(b["answer_token_position"], 6)
        self.assertEqual(b["answer_slot"], 1)
        self.assertEqual(b["decision_tokens"], [{"index": 6, "token": " Y", "logprob": -.01}])
        self.assertAlmostEqual(b["yes_score"], 1 / (1 + math.exp(-7.99)))

    def test_confident_no_produces_low_yes_not_high_no_score(self):
        a = parse_joint_response(response(), IDS)["candidates"]["object_001"]
        self.assertEqual(a["answer_label"], "N")
        self.assertLess(a["yes_score"], .001)

    def test_whitespace_tokens_do_not_change_answer_score(self):
        raw = response()
        tokens = raw["choices"][0]["logprobs"]["content"]
        tokens.insert(0, {"token": "\n", "logprob": -100, "top_logprobs": []})
        raw["choices"][0]["message"]["content"] = "\n" + raw["choices"][0]["message"]["content"]
        parsed = parse_joint_response(raw, IDS)
        self.assertEqual(parsed["candidates"]["object_002"]["answer_token_position"], 7)
        self.assertAlmostEqual(parsed["candidates"]["object_002"]["yes_score"], 1 / (1 + math.exp(-7.99)))

    def test_no_probability_borrowed_from_another_candidate(self):
        raw = response()
        raw["choices"][0]["logprobs"]["content"][2]["top_logprobs"] = []
        parsed = parse_joint_response(raw, IDS)
        self.assertIsNone(parsed["candidates"]["object_001"]["yes_score"])
        self.assertIsNotNone(parsed["candidates"]["object_002"]["yes_score"])
        predicted = scene_prediction({k: c["yes_score"] for k, c in parsed["candidates"].items()})
        self.assertEqual(verification_gate(predicted, 0, 0), "deferred")

    def test_sentinel_nan_missing_positive_alternatives_remain_unavailable(self):
        for value in (None, -9999, -10000, float("nan"), float("inf"), .01):
            raw = response()
            raw["choices"][0]["logprobs"]["content"][2]["top_logprobs"][0]["logprob"] = value
            self.assertIsNone(parse_joint_response(raw, IDS)["candidates"]["object_001"]["yes_score"])

    def test_mismatched_alternative_spelling_is_unavailable(self):
        raw = response()
        raw["choices"][0]["logprobs"]["content"][2]["top_logprobs"][0]["token"] = "Y"
        self.assertIsNone(parse_joint_response(raw, IDS)["candidates"]["object_001"]["yes_score"])

    def test_answer_fused_with_colon_is_not_a_separable_binary_token(self):
        raw = response()
        tokens = raw["choices"][0]["logprobs"]["content"]
        tokens[2]["token"] = ": N"
        del tokens[1]
        parsed = parse_joint_response(raw, IDS)
        self.assertTrue(parsed["format_valid"])
        self.assertIsNone(parsed["candidates"]["object_001"]["yes_score"])

    def test_invalid_truncated_missing_duplicate_extra_rows_defer(self):
        for text in ("object_001: Y", "object_001: Y\nobject_001: N\nobject_003: N",
                     "object_001: Yes\nobject_002: N\nobject_003: N",
                     "Explanation\nobject_001: Y\nobject_002: N\nobject_003: N"):
            raw = response()
            raw["choices"][0]["message"]["content"] = text
            self.assertFalse(parse_joint_response(raw, IDS)["format_valid"])
        raw = response()
        raw["choices"][0]["finish_reason"] = "length"
        self.assertFalse(parse_joint_response(raw, IDS)["format_valid"])

    def test_reversed_answers_map_to_persistent_ids(self):
        raw = response(("Y", "N", "N"), IDS[::-1])
        parsed = parse_joint_response(raw, IDS[::-1])
        self.assertTrue(parsed["format_valid"])
        self.assertEqual(parsed["candidates"]["object_003"]["answer_slot"], 0)
        self.assertGreater(parsed["candidates"]["object_003"]["yes_score"], .99)
        self.assertFalse(parse_joint_response(raw, IDS)["format_valid"])

    def test_multiple_yes_and_exact_ties_are_supported_and_deferred(self):
        parsed = parse_joint_response(response(("Y", "Y", "N")), IDS)
        predicted = scene_prediction({k: c["yes_score"] for k, c in parsed["candidates"].items()})
        self.assertTrue(predicted["top_tie"])
        self.assertEqual(verification_gate(predicted, 0, 0), "deferred")

    def test_all_no_never_forces_autonomous_selection(self):
        raw = response(("N", "N", "N"))
        token = raw["choices"][0]["logprobs"]["content"][2]
        token["top_logprobs"][0]["logprob"] = -6
        parsed = parse_joint_response(raw, IDS)
        predicted = scene_prediction({k: c["yes_score"] for k, c in parsed["candidates"].items()})
        self.assertEqual(predicted["predicted_object_id"], "object_001")
        self.assertEqual(verification_gate(predicted, .9, .1), "deferred")

    def test_token_text_mismatch_or_no_logprobs_produces_no_fake_scores(self):
        for tokens in ([], [{"token": "different", "logprob": -.1}]):
            raw = response()
            raw["choices"][0]["logprobs"]["content"] = tokens
            parsed = parse_joint_response(raw, IDS)
            self.assertTrue(parsed["format_valid"])
            self.assertTrue(all(c["yes_score"] is None for c in parsed["candidates"].values()))

    def test_reverse_only_changes_requested_answer_order(self):
        s = scene()
        before = copy.deepcopy(s)
        forward, reverse = joint_prompt(s, "forward"), joint_prompt(s, "reverse")
        self.assertEqual(forward.replace(", ".join(IDS), ", ".join(IDS[::-1])), reverse)
        self.assertEqual(answer_order(s, "reverse"), IDS[::-1])
        self.assertEqual(s, before)
        self.assertNotIn("evaluation_secret", forward)
        self.assertNotIn("ground_truth", forward)
        self.assertNotIn("instruction_role", forward)

    def test_request_preserves_image_snapshot_temperature_detail_and_logprobs(self):
        s = scene()
        expected = request_arguments(b"frozen image", joint_prompt(s, "forward"))
        expected["messages"][0]["content"] = JOINT_SYSTEM
        expected["max_completion_tokens"] = MAX_OUTPUT_TOKENS
        self.assertEqual(joint_arguments(s, b"frozen image", "forward"), expected)
        self.assertNotIn("logit_bias", expected)

    def test_resume_makes_no_new_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "responses").mkdir()
            (root / "scenes").mkdir()
            image = root / "input.png"
            image.write_bytes(b"frozen")
            s = {**scene(), "image_path": str(image), "image_sha256": digest(image.read_bytes())}
            completion = MagicMock()
            completion.model_dump.return_value = response()
            with patch("scripts.run_vlm_joint_verification_pilot.OUTPUT_ROOT", root), \
                    patch("openai.OpenAI"), \
                    patch("scripts.run_vlm_joint_verification_pilot.paced_completion", return_value=completion) as call:
                first = run_condition({"scene": s}, "forward")
                second = run_condition({"scene": s}, "forward")
                self.assertEqual(first, second)
                self.assertEqual(call.call_count, 1)
                saved = json.loads((root / "responses" / "s_forward.json").read_text())
                self.assertEqual(saved["provider_response"], response())
                with self.assertRaisesRegex(ValueError, "Resume input/settings mismatch"):
                    run_condition({"scene": {**s, "input_sha256": "changed"}}, "forward")

    def test_analysis_separates_wrong_selection_from_unavailable(self):
        from scripts.analyze_vlm_joint_verification_pilot import condition_summary

        records = []
        for scores in ({"object_001": .01, "object_002": .99},
                       {"object_001": .99, "object_002": .01},
                       {"object_001": None, "object_002": .99}):
            prediction = scene_prediction(scores)
            prediction["correct"] = prediction["predicted_object_id"] == "object_002"
            records.append({"verification": prediction, "candidates": {
                key: {"yes_score": value, "answer_label": "Y" if value and value > .5 else "N"}
                for key, value in scores.items()}})
        summary = condition_summary(records)
        self.assertEqual((summary["correct"], summary["wrong_selections"], summary["unresolved"]), (1, 1, 1))
        self.assertEqual(summary["score_available"], 2)
        self.assertEqual(summary["accuracy_resolved"], .5)

    def test_order_analysis_pairs_by_scene_and_reports_changes_without_selection(self):
        from scripts.analyze_vlm_joint_verification_pilot import order_sensitivity

        def record(sample_id, scores):
            return {"scene": {"sample_id": sample_id, "target_description": "mug"},
                    "verification": scene_prediction(scores), "candidates": {
                        key: {"yes_score": value, "answer_label": "Y" if value > .5 else "N"}
                        for key, value in scores.items()}}

        forward = [record("a", {"object_001": .95, "object_002": .1}),
                   record("b", {"object_001": .95, "object_002": .1})]
        reverse = [record("b", {"object_001": .2, "object_002": .8}), copy.deepcopy(forward[0])]
        before = copy.deepcopy((forward, reverse))
        rows = order_sensitivity(forward, reverse)
        self.assertFalse(rows[0]["prediction_changed"])
        self.assertEqual(rows[0]["max_abs_yes_change"], 0)
        self.assertTrue(rows[1]["prediction_changed"])
        self.assertEqual(rows[1]["candidate_answer_changes"], 2)
        self.assertAlmostEqual(rows[1]["max_abs_yes_change"], .75)
        self.assertEqual((rows[1]["forward_gate"], rows[1]["reverse_gate"]), ("vlm_accepted", "deferred"))
        self.assertEqual((forward, reverse), before)


if __name__ == "__main__":
    unittest.main()
