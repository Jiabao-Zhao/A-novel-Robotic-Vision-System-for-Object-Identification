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
from scripts.run_vlm_output_format_ablation import MODEL, digest, request_arguments
from scripts.run_vlm_presence_pilot import (
    CONDITIONS, NONE_SENTENCE, PRESENCE_SENTENCE, condition_prompt, report,
    run_pair, select_samples, summarize,
)
from scripts.run_vlm_visual_label_ablation import parse_response


def sample(task=0):
    mapping = {c: f"object_{i:03d}" for i, c in enumerate("ABCDEFG", 1)}
    return {"sample_id": f"s{task}", "task_index": task, "initial_state_index": 0,
            "partition": "calibration", "target_description": "milk", "target_localized": True,
            "ground_truth_object_id": "object_002", "candidate_map": {**mapping, "N": None},
            "prompts": {"new_direct_letters": _vlm_prompt("milk", [
                {"choice": c, "bbox_2d_xyxy": [1, 2, 3, 4]} for c in mapping])},
            "input_sha256": "frozen"}


def response(text="B", logprob=-.2):
    return {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {"content": text},
            "logprobs": {"content": [{"token": text, "logprob": logprob,
                "top_logprobs": [{"token": "A", "logprob": -.00001}]}]}}]}


class PresencePilotTests(unittest.TestCase):
    def test_only_one_state_per_target_selected_mechanically(self):
        states = [{**sample(t), "initial_state_index": s} for t in range(10) for s in range(50)]
        selected = select_samples(states[::-1])
        self.assertEqual([s["task_index"] for s in selected], list(range(10)))
        self.assertEqual({s["initial_state_index"] for s in selected}, {0})

    def test_missing_or_duplicate_state_zero_rejected(self):
        states = [sample(t) for t in range(10)]
        for invalid in (states[:-1], states + [sample()]):
            with self.assertRaises(ValueError):
                select_samples(invalid)

    def test_nonlocalized_null_or_test_target_rejected(self):
        for update in ({"partition": "test"}, {"target_localized": False},
                       {"ground_truth_object_id": None}, {"ground_truth_object_id": "object_099"}):
            states = [sample(t) for t in range(10)]
            states[0].update(update)
            with self.assertRaisesRegex(ValueError, "localized calibration target"):
                select_samples(states)

    def test_only_presence_sentence_and_final_n_change(self):
        s = sample()
        before = copy.deepcopy(s)
        old, old_map = condition_prompt(s, CONDITIONS[0])
        new, new_map = condition_prompt(s, CONDITIONS[1])
        self.assertEqual(new, old[:-3].replace(NONE_SENTENCE, PRESENCE_SENTENCE))
        self.assertEqual(new_map, {k: v for k, v in old_map.items() if k != "N"})
        self.assertTrue(new.endswith("A, B, C, D, E, F, G"))
        self.assertNotIn("object_002", new)  # No ground-truth ID leakage.
        self.assertEqual(s, before)
        s["prompts"]["new_direct_letters"] = "changed prompt"
        with self.assertRaisesRegex(ValueError, "layout changed"):
            condition_prompt(s, CONDITIONS[1])

    def test_same_image_settings_system_prompt_except_user_text(self):
        requests = [request_arguments(b"same frozen image", condition_prompt(sample(), c)[0]) for c in CONDITIONS]
        for req in requests:
            req["messages"][1]["content"][0]["text"] = "<treatment>"
        self.assertEqual(*requests)
        self.assertNotIn("logit_bias", requests[0])
        self.assertNotIn("response_format", requests[0])

    def test_n_is_valid_control_but_not_closed_set(self):
        for condition in CONDITIONS:
            mapping = condition_prompt(sample(), condition)[1]
            result = parse_response(response("N"), mapping)
            if condition == "with_n":
                self.assertEqual(result["choice"], "N")
                self.assertEqual(result["raw_sequence_likelihood"], math.exp(-.2))
            else:
                self.assertIsNone(result["choice"])
                self.assertIsNone(result["raw_sequence_likelihood"])

    def test_generated_label_raw_likelihood_unchanged_by_top_alternative(self):
        for condition in CONDITIONS:
            result = parse_response(response(), condition_prompt(sample(), condition)[1])
            self.assertEqual(result["predicted_object_id"], "object_002")
            self.assertEqual(result["raw_sequence_likelihood"], math.exp(-.2))
            self.assertNotIn("candidate_scores", result)

    def test_separable_formatting_excluded(self):
        raw = response()
        raw["choices"][0]["message"]["content"] = "\nB."
        raw["choices"][0]["logprobs"]["content"] = [
            {"token": t, "logprob": p} for t, p in (("\n", -20), ("B", -.2), (".", -30))]
        result = parse_response(raw, condition_prompt(sample(), CONDITIONS[1])[1])
        self.assertEqual(result["decision_token_count"], 1)
        self.assertEqual(result["raw_sequence_likelihood"], math.exp(-.2))

    def test_missing_invalid_probabilities_never_zero_filled(self):
        for value in (None, -9999, .2, float("inf"), float("nan")):
            result = parse_response(response(logprob=value), sample()["candidate_map"])
            self.assertIsNone(result["raw_sequence_likelihood"])

    def test_paired_resume_does_not_call_api_and_offline_report(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / "responses").mkdir()
            image, loc = root / "image.png", root / "loc.json"
            image.write_bytes(b"frozen image")
            loc.write_text("{}")
            s = {**sample(), "new_image_path": str(image), "localization_path": str(loc),
                 "new_image_sha256": digest(image.read_bytes()), "localization_sha256": digest(loc.read_bytes())}
            with patch("scripts.run_vlm_presence_pilot.paced_completion") as api:
                api.return_value.model_dump.return_value = response()
                rows = run_pair(s, MagicMock(), root)
                self.assertEqual(api.call_count, 2)
                before = {p: p.read_bytes() for p in (root / "responses").glob("*.json")}
                api.reset_mock()
                self.assertEqual(run_pair(s, MagicMock(), root), rows)
                api.assert_not_called()
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                (root / "records.json").write_text(json.dumps({"records": rows}))
                report(root)
                api.assert_not_called()
                self.assertTrue((root / "paired_scores.csv").exists())
                s["input_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "Resume request/input mismatch"):
                    run_pair(s, MagicMock(), root)
                image.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Frozen image/localization"):
                    run_pair(s, MagicMock(), root)

    def test_summary_keeps_missing_score_separate_from_accuracy(self):
        rows = [{"sample_id": "s", "target_description": "milk", "ground_truth_object_id": "object_002",
                 "condition": c, "choice": "B", "predicted_object_id": "object_002", "correct": True,
                 "raw_sequence_likelihood": score, "raw_log_probability": math.log(score) if score else None}
                for c, score in zip(CONDITIONS, (.8, None))]
        result = summarize(rows)
        self.assertEqual(result["conditions"][CONDITIONS[1]]["correct"], 1)
        self.assertEqual(result["conditions"][CONDITIONS[1]]["scores_available"], 0)
        self.assertIsNone(result["pairs"][0]["likelihood_delta_no_n_minus_with_n"])
        rows[1].update(raw_sequence_likelihood=.9, raw_log_probability=math.log(.9))
        self.assertAlmostEqual(summarize(rows)["pairs"][0]["likelihood_delta_no_n_minus_with_n"], .1)
        with self.assertRaisesRegex(ValueError, "incomplete pair"):
            summarize(rows[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate condition"):
            summarize(rows + rows[:1])


if __name__ == "__main__":
    unittest.main()
