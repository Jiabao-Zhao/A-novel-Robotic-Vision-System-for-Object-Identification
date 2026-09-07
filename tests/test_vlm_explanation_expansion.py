from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from prompt import _vlm_prompt
from scripts.run_vlm_explanation_expansion import (
    CONDITIONS, STATES, build_prompt, report, run_pair, select_new_objects,
    select_states, validate_pairs,
)
from scripts.run_vlm_output_format_ablation import MODEL, digest


def sample(task=0, state=1):
    mapping = {label: f"object_{index:03d}" for index, label in enumerate("ABCDEFG", 1)}
    return {"sample_id": f"s{task}_{state}", "task_index": task, "initial_state_index": state,
            "pair_index": task, "partition": "calibration", "target_localized": True,
            "cohort": "additional_states", "ground_truth_object_id": "object_002", "input_sha256": "frozen",
            "target_description": "milk", "candidate_map": {**mapping, "N": None},
            "prompts": {"new_direct_letters": _vlm_prompt("milk", [{"choice": c} for c in mapping])}}


def completion(_client, args):
    prompt = args["messages"][1]["content"][0]["text"]
    parts = [("B", -.2)]
    if "Second line:" in prompt:
        parts.append(("\nA blue carton.", -20))
    response = {"model": MODEL, "choices": [{"finish_reason": "stop",
                "message": {"content": "".join(t for t, _ in parts)},
                "logprobs": {"content": [{"token": t, "logprob": p} for t, p in parts]}}]}
    return MagicMock(model_dump=MagicMock(return_value=response))


class ExplanationExpansionTests(unittest.TestCase):
    def test_exactly_one_hundred_additional_calibration_scenes(self):
        all_samples = [{**sample(task, state), "partition": "calibration" if state < 30 else "test"}
                       for state in range(50) for task in range(10)]
        selected = select_states(all_samples[::-1])
        self.assertEqual(len(selected), 100)
        self.assertEqual({s["initial_state_index"] for s in selected}, set(STATES))
        self.assertEqual({s["partition"] for s in selected}, {"calibration"})
        self.assertNotIn(0, STATES)

    def test_duplicate_missing_unlocalized_or_noncali_scene_fails(self):
        rows = [sample(task, state) for state in STATES for task in range(10)]
        for bad in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                select_states(bad)
        for update in ({"partition": "test"}, {"target_localized": False}, {"ground_truth_object_id": None}):
            bad = [{**s} for s in rows]
            bad[0].update(update)
            with self.assertRaises(ValueError):
                select_states(bad)

    def test_unresolved_new_object_excluded_before_calls(self):
        ready = {"sample_id": "book", "target_description": "book", "status": "ready", "ground_truth_object_id": "object_003"}
        missing = {"sample_id": "bowl", "target_description": "white bowl", "status": "target_localization_unresolved", "ground_truth_object_id": None}
        selected, excluded = select_new_objects([ready, missing])
        self.assertEqual(selected, [ready])
        self.assertEqual(excluded[0]["sample_id"], "bowl")
        selected, _ = select_new_objects([{**missing, "status": "ready"}])
        self.assertFalse(selected)

    def test_only_two_permitted_arms_and_variable_candidate_counts(self):
        self.assertEqual(CONDITIONS, ("label_only", "label_then_evidence"))
        s = sample()
        s["candidate_map"] = {k: v for k, v in s["candidate_map"].items() if k in ("A", "B", "C", "N")}
        s["prompts"]["new_direct_letters"] = _vlm_prompt("milk", [{"choice": c} for c in "ABC"])
        for condition in CONDITIONS:
            prompt, mapping = build_prompt(s, condition)
            self.assertEqual(list(mapping), list("ABC"))
            self.assertNotIn("N means none", prompt)
            if condition == "label_then_evidence":
                self.assertIn("First line: one label", prompt)
                self.assertIn("Second line: one short sentence", prompt)

    def test_two_fresh_calls_then_cached_resume_and_offline_report(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / "responses").mkdir()
            image, loc = root / "image.png", root / "loc.json"
            image.write_bytes(b"fixed PNG")
            loc.write_text("{}")
            s = {**sample(), "new_image_path": str(image), "localization_path": str(loc),
                 "new_image_sha256": digest(image.read_bytes()), "localization_sha256": digest(loc.read_bytes())}
            with patch("scripts.run_vlm_explanation_expansion.paced_completion", side_effect=completion) as api:
                rows = run_pair(s, MagicMock(), root)
                self.assertEqual(api.call_count, 2)
                self.assertEqual({r["condition"] for r in rows}, set(CONDITIONS))
                self.assertEqual(rows[0]["raw_sequence_likelihood"], rows[1]["raw_sequence_likelihood"])
                before = {p: p.read_bytes() for p in (root / "responses").glob("*.json")}
                requests = [call.args[1] for call in api.call_args_list]
                for args in requests:
                    args["messages"][1]["content"][0]["text"] = "<only treatment>"
                self.assertEqual(requests[0], requests[1])
                api.reset_mock()
                self.assertEqual(run_pair(s, MagicMock(), root), rows)
                api.assert_not_called()
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                self.assertEqual(len(validate_pairs(rows)), 1)
                (root / "records.json").write_text(json.dumps({"records": rows}))
                report(root)
                api.assert_not_called()
                self.assertTrue((root / "REPORT.md").exists())
                self.assertTrue((root / "threshold_curves.csv").exists())
                for invalid in (rows[:-1], rows + rows[:1], [{**rows[0], "condition": "evidence_then_label"}, rows[1]]):
                    with self.assertRaises(ValueError):
                        validate_pairs(invalid)
                altered = [{**r} for r in rows]
                altered[1]["image_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    validate_pairs(altered)
                s["input_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "Resume request/input mismatch"):
                    run_pair(s, MagicMock(), root)


if __name__ == "__main__":
    unittest.main()
