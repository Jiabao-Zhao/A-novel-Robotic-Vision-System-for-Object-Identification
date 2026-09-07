import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.run_vlm_multi_object_pilot import (
    MODELS, ROLE, autonomous, digest, evaluate_response, make_tasks, parse_response,
    request_arguments, run_trial, select_scenes, token_likelihood,
)
from scripts.analyze_vlm_multi_object_pilot import semantic_evaluation


MAP = {"A": "object_001", "B": "object_002", "C": "object_003", "N": None}
INSTRUCTION = "Put the tomato sauce on top of the milk."
TRUTH = {"tomato sauce": "object_001", "milk": "object_002"}


def response(parts, model=MODELS[0]):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": "".join(t for t, _ in parts)},
            "logprobs": {"content": [{"token": t, "logprob": p, "top_logprobs": [{"token": "Z", "logprob": -.00001}]} for t, p in parts]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": len(parts)}}


def parsed(parts):
    return parse_response(response(parts), MAP, INSTRUCTION)


class JointObjectPilotTests(unittest.TestCase):
    def test_role_and_instruction_only_no_ordered_targets_or_metadata(self):
        args = request_arguments(b"same marked PNG", INSTRUCTION, MODELS[0])
        self.assertEqual(args["messages"][0]["content"], ROLE + "\n\nTask instruction:\n" + INSTRUCTION)
        self.assertEqual(len(args["messages"][1]["content"]), 1)
        image = args["messages"][1]["content"][0]
        self.assertEqual(image["type"], "image_url")
        self.assertEqual(image["image_url"]["detail"], "high")
        for forbidden in ("Targets, in order", "object_001", "centroid", "ground_truth", '"confidence":'):
            self.assertNotIn(forbidden, json.dumps(args))
        self.assertEqual(args["temperature"], 0)
        self.assertEqual(args["max_completion_tokens"], 128)
        newer = request_arguments(b"same marked PNG", INSTRUCTION, MODELS[1])
        self.assertEqual(newer["messages"], args["messages"])
        self.assertEqual(newer["reasoning_effort"], "none")
        self.assertEqual(newer["top_logprobs"], 5)

    def test_full_score_includes_formatting_entry_score_does_not(self):
        row = parsed([("\n", -2), ("A", -.1), (":", -3), (" tomato", -.2), (" sauce", -.3),
                      ("\n", -4), ("B", -.4), (": ", -5), ("milk", -.5), ("\n", -6)])
        self.assertTrue(row["format_valid"])
        self.assertAlmostEqual(row["full_response_likelihood"], math.exp(-21.5))
        self.assertAlmostEqual(row["entries"][0]["raw_association_likelihood"], math.exp(-.6))
        self.assertAlmostEqual(row["entries"][1]["raw_association_likelihood"], math.exp(-.9))
        self.assertEqual(row["entries"][0]["decision_token_count"], 3)
        self.assertTrue(evaluate_response(row, TRUTH)["correct"])

    def test_mixed_tokens_are_indivisible_and_no_length_normalization(self):
        row = parsed([(" A:", -.2), (" tomato sauce", -.4), ("\nB: milk", -.3)])
        self.assertAlmostEqual(row["entries"][0]["raw_association_likelihood"], math.exp(-.6))
        self.assertAlmostEqual(row["entries"][1]["raw_association_likelihood"], math.exp(-.3))
        self.assertAlmostEqual(row["full_response_likelihood"], math.exp(-.9))

    def test_token_crossing_two_entries_keeps_full_score_not_fake_entry_scores(self):
        row = parsed([("A: tomato sauce\nB: milk", -.3)])
        self.assertAlmostEqual(row["full_response_likelihood"], math.exp(-.3))
        self.assertTrue(all(e["raw_association_likelihood"] is None for e in row["entries"]))

    def test_names_and_actual_visual_letters_both_matter(self):
        row = parsed([("B: tomato sauce", -.1), ("\nA: milk", -.2)])
        evaluation = evaluate_response(row, TRUTH)
        self.assertTrue(row["format_valid"])
        self.assertFalse(evaluation["correct"])
        self.assertTrue(all(not a["correct"] for a in evaluation["associations"]))

    def test_response_order_is_not_fixed(self):
        row = parsed([("B: milk", -.1), ("\nA: tomato sauce", -.2)])
        self.assertTrue(evaluate_response(row, TRUTH)["correct"])

    def test_optional_the_is_semantically_accepted_without_changing_likelihoods(self):
        row = parsed([("A: the tomato sauce", -.1), ("\nB: the milk", -.2)])
        original = {**row, **evaluate_response(row, TRUTH)}
        self.assertFalse(original["correct"])
        corrected = semantic_evaluation(original, TRUTH)
        self.assertTrue(corrected["correct"])
        self.assertFalse(corrected["literal_name_correct"])
        for key in ("entries", "full_response_likelihood", "joint_gate_score", "generated_output_text"):
            self.assertEqual(corrected[key], original[key])
        self.assertEqual(corrected["associations"][0]["raw_association_likelihood"], row["entries"][0]["raw_association_likelihood"])

    def test_article_normalization_does_not_repair_wrong_ids_or_missing_modifiers(self):
        for text in ("B: the tomato sauce\nA: the milk", "A: sauce\nB: milk"):
            row = parsed([(text, -.01)])
            self.assertFalse(semantic_evaluation({**row, **evaluate_response(row, TRUTH)}, TRUTH)["correct"])

    def test_none_supported_and_false_absence_is_incorrect(self):
        row = parsed([("N: tomato sauce", -.1), ("\nB: milk", -.2)])
        self.assertIsNone(row["entries"][0]["predicted_object_id"])
        self.assertFalse(evaluate_response(row, TRUTH)["correct"])
        self.assertTrue(evaluate_response(row, {"tomato sauce": None, "milk": "object_002"})["correct"])

    def test_missing_target_counts_wrong_without_oracle_gate(self):
        row = parsed([("B: milk", -.0001)])
        evaluation = evaluate_response(row, TRUTH)
        self.assertEqual(evaluation["missing_targets"], ["tomato sauce"])
        self.assertEqual(len(evaluation["associations"]), 2)
        self.assertFalse(evaluation["correct"])
        self.assertTrue(autonomous(row, .9))  # Important: do not use hidden expected targets to defer.

    def test_extra_duplicate_invalid_labels_and_explanations(self):
        for text in ("A: tomato sauce\nB: milk\nC: ketchup", "A: tomato sauce\nB: milk\nB: milk",
                     "Z: tomato sauce\nB: milk", "A: tomato sauce\nB: milk\nThe cartons look clear."):
            row = parsed([(text, -.01)])
            self.assertFalse(row["format_valid"])
            self.assertFalse(autonomous(row, .5))
            self.assertFalse(evaluate_response(row, TRUTH)["correct"])

    def test_synonyms_not_silently_accepted(self):
        row = parsed([("A: canned soup\nB: milk", -.01)])
        result = evaluate_response(row, TRUTH)
        self.assertEqual(result["missing_targets"], ["tomato sauce"])
        self.assertEqual(result["extra_targets"], ["canned soup"])

    def test_missing_invalid_sentinel_positive_probability_never_faked(self):
        for invalid in (None, -9999, float("nan"), float("inf"), .00006103515625):
            row = parsed([("A: tomato sauce", invalid), ("\nB: milk", -.2)])
            self.assertIsNone(row["full_response_likelihood"])
            self.assertIsNone(row["entries"][0]["raw_association_likelihood"])
            self.assertFalse(autonomous(row, 0))
        self.assertEqual(token_likelihood([]), (None, None))

    def test_unaligned_or_truncated_output_has_no_score(self):
        for field in ("truncated", "unaligned", "no_logprobs"):
            raw = response([("A: tomato sauce\nB: milk", -.3)])
            if field == "truncated":
                raw["choices"][0]["finish_reason"] = "length"
            elif field == "unaligned":
                raw["choices"][0]["message"]["content"] += "\n"
            else:
                raw["choices"][0]["logprobs"] = None
            row = parse_response(raw, MAP, INSTRUCTION)
            self.assertIsNone(row["full_response_likelihood"])
            self.assertFalse(autonomous(row, 0))

    def test_gate_boundary_and_score_only_not_correctness(self):
        row = parsed([("B: tomato sauce", -.1), ("\nA: milk", -.2)])
        score = row["full_response_likelihood"]
        self.assertTrue(autonomous(row, score))
        self.assertTrue(autonomous(row, score - .01))
        self.assertFalse(autonomous(row, score + .01))
        self.assertFalse(evaluate_response(row, TRUTH)["correct"])

    def test_selection_calibration_only_one_state_per_task(self):
        samples = [{"cohort": "additional_states", "initial_state_index": 1, "task_index": i, "partition": "calibration"} for i in range(10)]
        self.assertEqual(len(select_scenes(samples[::-1])), 10)
        bad = copy.deepcopy(samples)
        bad[0]["partition"] = "test"
        with self.assertRaises(ValueError):
            select_scenes(bad)

    def test_tasks_use_audited_identity_not_id_position_or_model_prediction(self):
        sample = {"sample_id": "s", "target_description": "tomato sauce", "ground_truth_object_id": "object_003",
                  "localization_sha256": "loc", "candidate_map": MAP}
        audit = {"evaluation_only": True, "excluded_from_vlm_inputs": True, "sample_id": "s", "target_label_agrees": True,
                 "source_sha256": {"localization": "loc"}, "replay_check": {"rgb_exact_match": True},
                 "candidates": [{"object_id": obj, "semantic_identity": name, "status": "matched"} for obj, name in
                                (("object_001", "alphabet soup"), ("object_002", "milk"), ("object_003", "tomato sauce"))]}
        tasks = make_tasks(sample, audit)
        self.assertEqual(tasks[0]["instruction"], INSTRUCTION)
        self.assertEqual(tasks[0]["evaluation_ground_truth"]["tomato sauce"], "object_003")
        self.assertEqual(len(tasks[1]["evaluation_ground_truth"]), 3)
        audit["candidates"][0]["status"] = "ambiguous"
        with self.assertRaises(ValueError):
            make_tasks(sample, audit)

    def test_checkpoint_resume_uses_no_api_and_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "responses").mkdir()
            sample = {"trial_id": "s_2", "sample_id": "s", "partition": "calibration", "target_count": 2,
                      "instruction": INSTRUCTION, "candidate_map": MAP, "evaluation_ground_truth": TRUTH, "input_sha256": "fixed"}
            for field, hash_field in (("new_image_path", "new_image_sha256"), ("localization_path", "localization_sha256"),
                                      ("identity_audit_path", "identity_audit_sha256")):
                path = root / field
                path.write_bytes(b"same input")
                sample[field], sample[hash_field] = str(path), digest(path.read_bytes())
            with patch("scripts.run_vlm_multi_object_pilot.paced_completion") as api:
                api.return_value = MagicMock(model_dump=MagicMock(return_value=response([("A: tomato sauce", -.1), ("\nB: milk", -.2)])))
                first = run_trial(sample, MODELS[0], MagicMock(), root)
                self.assertEqual(api.call_count, 1)
                api.side_effect = AssertionError("Must not rerun API")
                self.assertEqual(first, run_trial(sample, MODELS[0], MagicMock(), root))
                sample["instruction"] = "different task"
                with self.assertRaisesRegex(ValueError, "Resume"):
                    run_trial(sample, MODELS[0], MagicMock(), root)


if __name__ == "__main__":
    unittest.main()
