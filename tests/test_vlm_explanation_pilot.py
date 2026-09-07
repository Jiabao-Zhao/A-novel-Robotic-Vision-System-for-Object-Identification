from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from prompt import _vlm_prompt
from scripts.run_vlm_output_format_ablation import MODEL, digest
from scripts.run_vlm_explanation_pilot import (
    CONDITIONS, MAX_COMPLETION_TOKENS, ORDERS, SCORE, arguments_for, build_prompt,
    parse_response, report, run_scene, summarize,
)


MAP = {c: f"object_{i:03d}" for i, c in enumerate("ABCDEFG", 1)}


def sample():
    return {"sample_id": "s", "task_index": 0, "initial_state_index": 0,
            "partition": "calibration", "target_description": "milk",
            "ground_truth_object_id": "object_002", "candidate_map": {**MAP, "N": None},
            "prompts": {"new_direct_letters": _vlm_prompt("milk", [
                {"choice": c, "bbox_2d_xyxy": [1, 2, 3, 4]} for c in MAP])}, "input_sha256": "frozen"}


def raw(parts):
    return {"model": MODEL, "choices": [{"finish_reason": "stop",
            "message": {"content": "".join(t for t, _ in parts)},
            "logprobs": {"content": [{"token": t, "logprob": p,
                "top_logprobs": [{"token": "A", "logprob": -.000001}]} for t, p in parts]}}]}


class ExplanationPilotTests(unittest.TestCase):
    def test_prompt_body_and_no_n_policy_fixed(self):
        s = sample()
        before = copy.deepcopy(s)
        prompts = [build_prompt(s, c)[0] for c in CONDITIONS]
        body = prompts[0].rsplit("\n", 1)[0]
        self.assertTrue(all(p.startswith(body + "\n") for p in prompts))
        self.assertTrue(all("guaranteed to correspond" in p for p in prompts))
        for condition, prompt in zip(CONDITIONS, prompts):
            self.assertEqual(build_prompt(s, condition)[1], MAP)
            self.assertNotIn("object_002", prompt)
            self.assertNotIn("N means none", prompt)
        self.assertEqual(prompts[1].replace("First line:", "TEMP:").replace("Second line:", "First line:").replace("TEMP:", "Second line:").splitlines()[-2:][::-1], prompts[2].splitlines()[-2:])
        self.assertEqual(s, before)

    def test_shared_request_settings_and_fresh_baseline_budget(self):
        requests = [arguments_for(b"same image", build_prompt(sample(), c)[0]) for c in CONDITIONS]
        for request in requests:
            self.assertEqual(request["model"], "gpt-4.1-mini-2025-04-14")
            self.assertEqual(request["temperature"], 0)
            self.assertEqual(request["max_completion_tokens"], MAX_COMPLETION_TOKENS)
            self.assertNotIn("logit_bias", request)
            self.assertNotIn("response_format", request)
            self.assertNotIn("Return only", request["messages"][0]["content"])
            request["messages"][1]["content"][0]["text"] = "<format treatment>"
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[1], requests[2])

    def test_all_six_permutations_used(self):
        self.assertEqual(len(set(ORDERS)), 6)
        self.assertTrue(all(set(order) == set(CONDITIONS) for order in ORDERS))

    def test_label_only_generated_not_top_alternative(self):
        result = parse_response(raw([("B", -.2)]), MAP, "label_only")
        self.assertEqual(result["predicted_object_id"], "object_002")
        self.assertEqual(result[SCORE], math.exp(-.2))

    def test_label_before_explanation_only_label_scored(self):
        result = parse_response(raw([("\n", -9), ("B", -.2), (".\n", -4),
                                     ("White carton with a blue cap.", -10)]), MAP, "label_then_evidence")
        self.assertEqual(result["choice"], "B")
        self.assertEqual(result["decision_tokens"], [{"index": 1, "token": "B", "logprob": -.2}])
        self.assertEqual(result[SCORE], math.exp(-.2))

    def test_explanation_before_label_does_not_select_article_or_mentioned_label(self):
        result = parse_response(raw([("A blue carton unlike candidate C.", -15), ("\n", -4),
                                     ("B", -.3), (".\n", -5)]), MAP, "evidence_then_label")
        self.assertEqual(result["choice"], "B")
        self.assertEqual(result["decision_tokens"][0]["index"], 2)
        self.assertEqual(result[SCORE], math.exp(-.3))
        self.assertEqual(result["explanation"], "A blue carton unlike candidate C.")

    def test_merged_whitespace_and_label_token_is_indivisible(self):
        result = parse_response(raw([("A carton.", -10), ("\nB", -.4)]), MAP, "evidence_then_label")
        self.assertEqual(result["decision_tokens"], [{"index": 1, "token": "\nB", "logprob": -.4}])
        self.assertEqual(result[SCORE], math.exp(-.4))

    def test_multitoken_decision_span_sums_only_original_decision_tokens(self):
        result = parse_response(raw([("A carton.", -10), ("\nobject", -.1), ("_", -.2),
                                     ("002", -.3), (".", -20)]), {"object_002": "object_002"}, "evidence_then_label")
        self.assertEqual(result["decision_token_count"], 3)
        self.assertAlmostEqual(result[SCORE], math.exp(-.6))

    def test_explanation_likelihood_is_not_part_of_score(self):
        for value in (None, -9999, -100):
            result = parse_response(raw([("A carton.", value), ("\n", None), ("B", -.2)]), MAP, "evidence_then_label")
            self.assertEqual(result[SCORE], math.exp(-.2))

    def test_unavailable_label_probabilities_remain_null(self):
        for value in (None, -9999, float("nan"), float("inf"), .1):
            result = parse_response(raw([("A carton.\n", -.1), ("B", value)]), MAP, "evidence_then_label")
            self.assertEqual(result["choice"], "B")
            self.assertIsNone(result[SCORE])

    def test_invalid_truncated_or_mismatched_responses_do_not_invent_score(self):
        for parts in ([("A carton.\nN", -.1)], [("B\nC", -.1), ("\nA carton.", -.2)],
                      [("A carton.\nChoice: B", -.1)]):
            result = parse_response(raw(parts), MAP, "evidence_then_label")
            self.assertIsNone(result["choice"])
            self.assertIsNone(result[SCORE])
        value = raw([("B", -.2)])
        value["choices"][0]["finish_reason"] = "length"
        self.assertIsNone(parse_response(value, MAP, "label_only")[SCORE])
        value = raw([("B", -.2)])
        value["choices"][0]["message"]["content"] = "C"
        self.assertIsNone(parse_response(value, MAP, "label_only")[SCORE])

    def test_resume_and_offline_analysis_never_call_api(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / "responses").mkdir()
            image, loc = root / "image.png", root / "loc.json"
            image.write_bytes(b"same frozen image")
            loc.write_text("{}")
            s = {**sample(), "new_image_path": str(image), "localization_path": str(loc),
                 "new_image_sha256": digest(image.read_bytes()), "localization_sha256": digest(loc.read_bytes())}
            raw_values = [raw([("B", -.2)]), raw([("B", -.2), ("\nA carton.", -.1)]),
                          raw([("A carton.\n", -.1), ("B", -.2)])]
            with patch("scripts.run_vlm_explanation_pilot.paced_completion") as api:
                api.side_effect = [MagicMock(model_dump=MagicMock(return_value=v)) for v in raw_values]
                rows = run_scene(s, MagicMock(), root)
                self.assertEqual(api.call_count, 3)
                before = {p: p.read_bytes() for p in (root / "responses").glob("*.json")}
                api.reset_mock()
                self.assertEqual(rows, run_scene(s, MagicMock(), root))
                api.assert_not_called()
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                (root / "records.json").write_text(json.dumps({"records": rows}))
                report(root)
                api.assert_not_called()
                self.assertTrue((root / "associations.csv").exists())
                stats, _, _ = summarize(rows)
                self.assertIsNone(stats["label_only"]["by_correctness"]["correctness_ranking_auc"])
                with self.assertRaisesRegex(ValueError, "complete"):
                    summarize(rows[:-1])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    summarize(rows + rows[:1])
                s["input_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "Resume request/input mismatch"):
                    run_scene(s, MagicMock(), root)

    def test_matched_coverage_retains_ties_and_unavailable_defers(self):
        records = []
        for i, score in enumerate((.99, .99, .6, None)):
            for condition in CONDITIONS:
                records.append({"sample_id": str(i), "condition": condition, "predicted_object_id": "object_002",
                                SCORE: score, "correct": i != 1, "format_valid": True,
                                "decision_token_count": 1, "explanation_word_count": 0})
        stats, curves, matched = summarize(records)
        self.assertEqual(stats["label_only"]["score_availability"], .75)
        self.assertEqual({p["coverage"] for p in matched}, {.5, .75})
        self.assertTrue(all(p["old_false_accepts"] == 1 for p in matched))
        self.assertTrue(all(p["autonomous_coverage"] <= .75 for p in curves))


if __name__ == "__main__":
    unittest.main()
