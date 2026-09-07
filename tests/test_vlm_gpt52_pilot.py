from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from prompt import _vlm_prompt
from scripts.run_vlm_gpt52_pilot import (
    BASELINE_MODEL, CONDITIONS, MODEL, arguments_for, comparison_records,
    digest, parse_response, report, request_arguments, run_trial, select_samples,
)


def sample(task=0, state=1):
    mapping = {"A": "object_001", "B": "object_002", "N": None}
    return {"sample_id": f"s{task}_{state}", "task_index": task, "initial_state_index": state,
            "pair_index": task, "cohort": "additional_states", "partition": "calibration",
            "target_description": "milk", "ground_truth_object_id": "object_002", "input_sha256": "frozen",
            "candidate_map": mapping, "prompts": {"new_direct_letters": _vlm_prompt("milk", [{"choice": c} for c in "AB"])}}


def completion(_client, args):
    prompt = args["messages"][1]["content"][0]["text"]
    parts = [("\n", -10), ("B", -.2)]
    if "Second line:" in prompt:
        parts.append(("\nA blue carton.", -20))
    response = {"model": MODEL, "choices": [{"finish_reason": "stop",
                "message": {"content": "".join(t for t, _ in parts)},
                "logprobs": {"content": [{"token": t, "logprob": p,
                                         "top_logprobs": [{"token": "A", "logprob": -.01}]} for t, p in parts]}}]}
    return MagicMock(model_dump=MagicMock(return_value=response))


class GPT52PilotTests(unittest.TestCase):
    def test_model_and_verified_compatibility_request_changes_only(self):
        old = arguments_for(b"same image", "same prompt")
        new = request_arguments(b"same image", "same prompt")
        self.assertEqual(new.pop("reasoning_effort"), "none")
        self.assertEqual(new.pop("model"), MODEL)
        old.pop("model")
        self.assertEqual(new.pop("top_logprobs"), 5)
        self.assertEqual(old.pop("top_logprobs"), 20)
        self.assertEqual(new, old)
        self.assertEqual(CONDITIONS, ("label_only", "label_then_evidence"))

    def test_fixed_selection_excludes_other_states_and_test_data(self):
        states = [sample(t, s) for s in range(11) for t in range(10)]
        others = [{**sample(i), "sample_id": f"fresh_{i}", "cohort": "additional_objects",
                   "partition": "exploratory_new_objects", "initial_state_index": None} for i in range(12)]
        rows = select_samples((states + others)[::-1])
        self.assertEqual(len(rows), 32)
        self.assertEqual({s["initial_state_index"] for s in rows[:20]}, {1, 2})
        for change in ({"partition": "test"}, {"ground_truth_object_id": None}):
            bad = [{**r} for r in rows]
            bad[0].update(change)
            with self.assertRaises(ValueError):
                select_samples(bad)
        with self.assertRaises(ValueError):
            select_samples(rows + rows[:1])

    def test_score_ignores_explanation_whitespace_and_top_alternatives(self):
        for condition in CONDITIONS:
            prompt = "Second line:" if condition == "label_then_evidence" else ""
            raw = completion(None, request_arguments(b"", prompt)).model_dump()
            row = parse_response(raw, {"A": "object_001", "B": "object_002"}, condition)
            self.assertEqual(row["choice"], "B")
            self.assertEqual(row["decision_token_count"], 1)
            self.assertAlmostEqual(row["raw_sequence_likelihood"], math.exp(-.2))
            # Actual GPT-5.2 response contained a tiny positive logprob. Never clamp to 1.
            for missing in (None, -9999, 0.00006103515625):
                raw["choices"][0]["logprobs"]["content"][1]["logprob"] = missing
                self.assertIsNone(parse_response(raw, {"A": "object_001", "B": "object_002"}, condition)["raw_sequence_likelihood"])

    def test_checkpoint_resumes_without_api_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "responses").mkdir()
            image, loc = root / "image.png", root / "loc.json"
            image.write_bytes(b"fixed image")
            loc.write_text("{}")
            s = {**sample(), "new_image_path": str(image), "localization_path": str(loc),
                 "new_image_sha256": digest(image.read_bytes()), "localization_sha256": digest(loc.read_bytes())}
            with patch("scripts.run_vlm_gpt52_pilot.paced_completion", side_effect=completion) as api:
                rows = [run_trial(s, c, MagicMock(), root) for c in CONDITIONS]
                self.assertEqual(api.call_count, 2)
                api.reset_mock()
                self.assertEqual(rows, [run_trial(s, c, MagicMock(), root) for c in CONDITIONS])
                api.assert_not_called()
                with self.assertRaises(ValueError):
                    run_trial(s, "evidence_then_label", MagicMock(), root)
                with self.assertRaisesRegex(ValueError, "Resume"):
                    run_trial({**s, "input_sha256": "changed"}, CONDITIONS[0], MagicMock(), root)
                checkpoint = Path(rows[0]["response_path"])
                raw = json.loads(checkpoint.read_text())
                raw["provider_response"]["model"] = BASELINE_MODEL
                checkpoint.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    run_trial(s, CONDITIONS[0], MagicMock(), root)

    def test_offline_report_pairs_only_identical_inputs(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            baseline = root / "baseline"
            baseline.mkdir()
            rows = []
            for cohort in ("additional_states", "additional_objects"):
                for condition in CONDITIONS:
                    raw = completion(None, request_arguments(b"", "Second line:" if condition != "label_only" else "")).model_dump()
                    row = {**sample(), "sample_id": cohort, "cohort": cohort, "condition": condition,
                           "model": MODEL, "image_sha256": "image", "localization_sha256": "loc", "correct": True,
                           **parse_response(raw, {"A": "object_001", "B": "object_002"}, condition)}
                    rows.append(row)
            (baseline / "records.json").write_text(json.dumps({"records": [{**r, "model": BASELINE_MODEL} for r in rows]}))
            payload = {"records": rows, "protocol": {"baseline_records_sha256": digest((baseline / "records.json").read_bytes())}}
            (root / "records.json").write_text(json.dumps(payload))
            with patch("scripts.run_vlm_gpt52_pilot.paced_completion", side_effect=AssertionError("offline")):
                self.assertEqual(len(comparison_records(payload, baseline)), 8)
                report(root, baseline)
            self.assertTrue((root / "REPORT.md").exists())
            self.assertTrue((root / "matched_coverage.csv").exists())
            payload["records"][0]["image_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                comparison_records(payload, baseline)


if __name__ == "__main__":
    unittest.main()
