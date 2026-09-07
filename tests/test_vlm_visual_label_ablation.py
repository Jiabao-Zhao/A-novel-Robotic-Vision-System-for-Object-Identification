import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from scripts.run_vlm_visual_label_ablation import (
    CONDITIONS, MODEL, SCORE, header_mask, parse_response, prompts_and_map,
    render_direct_sheet, run_pair,
)
from scripts.run_vlm_output_format_ablation import build_prompt, digest, request_arguments
from vlm_module import create_roi_contact_sheet


def raw_response(text="B", logprob=-.2):
    return {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {"content": text},
            "logprobs": {"content": [{"token": text, "logprob": logprob,
                "top_logprobs": [{"token": "A", "logprob": -.0001}]}]}}]}


class VisualLabelTests(unittest.TestCase):
    def test_new_metadata_only_removes_persistent_id_mapping(self):
        detections = [{"object_id": "object_002", "bbox_2d_xyxy": [1, 2, 30, 40]},
                      {"object_id": "object_001", "centroid_3d_m": [1, 2, 3]}]
        before = copy.deepcopy(detections)
        prompts, mapping = prompts_and_map("milk", detections)
        old, old_map = build_prompt("milk", detections, "compact")
        self.assertEqual(prompts[CONDITIONS[0]], old)
        self.assertEqual(mapping, old_map)
        expected = old.replace(',"object_id":"object_001"', '').replace(',"object_id":"object_002"', '')
        self.assertEqual(prompts[CONDITIONS[1]], expected)
        self.assertNotIn("object_", prompts[CONDITIONS[1]])
        self.assertTrue(prompts[CONDITIONS[1]].endswith("A, B, N"))
        self.assertEqual(detections, before)

    def test_renderer_changes_only_existing_candidate_header_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rgb, loc, old, new = [root / name for name in ("rgb.png", "loc.json", "old.png", "new.png")]
            pixels = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb), pixels)
            loc.write_text(json.dumps({"objects": [
                {"object_id": "object_001", "roi": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}},
                {"object_id": "object_002", "roi": {"x1": 25, "y1": 25, "x2": 60, "y2": 60}}]}))
            mapping = {"A": "object_001", "B": "object_002", "N": None}
            create_roi_contact_sheet(rgb, loc, old, tile_size_px=448)
            source_bytes = (rgb.read_bytes(), loc.read_bytes(), old.read_bytes())
            audit = render_direct_sheet(rgb, loc, old, new, mapping)
            a, b = cv2.imread(str(old)), cv2.imread(str(new))
            self.assertTrue(audit["unchanged_nonlabel_pixels"])
            self.assertGreater(audit["changed_label_pixels"], 0)
            self.assertTrue(np.array_equal(a[~header_mask(a.shape, 2)], b[~header_mask(a.shape, 2)]))
            self.assertEqual(source_bytes, (rgb.read_bytes(), loc.read_bytes(), old.read_bytes()))
            self.assertEqual(audit, render_direct_sheet(rgb, loc, old, new, mapping))
            with self.assertRaisesRegex(ValueError, "crop order"):
                render_direct_sheet(rgb, loc, old, new, {"A": "object_002", "B": "object_001", "N": None})
            a[100, 100] = 0
            cv2.imwrite(str(old), a)
            with self.assertRaisesRegex(ValueError, "reproduce"):
                render_direct_sheet(rgb, loc, old, new, mapping)

    def test_generated_choice_not_top_alternative_determines_prediction(self):
        decision = parse_response(raw_response(), {"A": "object_001", "B": "object_002", "N": None})
        self.assertEqual(decision["predicted_object_id"], "object_002")
        self.assertEqual(decision[SCORE], math.exp(-.2))
        self.assertNotIn("candidate_scores", decision)

    def test_raw_score_excludes_separable_formatting(self):
        raw = raw_response()
        parts = [("\n", -20), ("B", -.2), (".", -30)]
        raw["choices"][0]["message"]["content"] = "\nB."
        raw["choices"][0]["logprobs"]["content"] = [{"token": t, "logprob": p} for t, p in parts]
        result = parse_response(raw, {"B": "object_002"})
        self.assertEqual(result[SCORE], math.exp(-.2))
        self.assertEqual(result["decision_token_count"], 1)

    def test_none_is_valid_without_fake_identity(self):
        result = parse_response(raw_response("N"), {"A": "object_001", "N": None})
        self.assertEqual(result["choice"], "N")
        self.assertIsNone(result["predicted_object_id"])
        self.assertEqual(result[SCORE], math.exp(-.2))

    def test_missing_invalid_and_sentinel_probabilities_remain_unavailable(self):
        for value in (None, float("nan"), float("inf"), -9999, .1):
            result = parse_response(raw_response(logprob=value), {"B": "object_002"})
            self.assertIsNone(result[SCORE])
        invalid = parse_response(raw_response("object_002"), {"B": "object_002", "N": None})
        self.assertIsNone(invalid["choice"])
        raw = raw_response()
        raw["choices"][0]["finish_reason"] = "length"
        self.assertIsNone(parse_response(raw, {"B": "object_002"})[SCORE])

    def test_pair_settings_are_identical_except_visual_and_text_representation(self):
        prompts, mapping = prompts_and_map("milk", [{"object_id": "object_001"}])
        a, b = (request_arguments(image, prompts[c]) for image, c in zip((b"old", b"new"), CONDITIONS))
        self.assertEqual({k: v for k, v in a.items() if k != "messages"},
                         {k: v for k, v in b.items() if k != "messages"})
        self.assertEqual(a["messages"][0], b["messages"][0])
        self.assertEqual(a["model"], "gpt-4.1-mini-2025-04-14")
        self.assertEqual(a["temperature"], 0)
        self.assertNotIn("logit_bias", a)

    def test_resume_uses_saved_pair_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "responses").mkdir()
            old, new, loc = [root / name for name in ("old.png", "new.png", "loc.json")]
            for path in (old, new, loc):
                path.write_bytes(path.name.encode())
            sample = {"sample_id": "s", "partition": "calibration", "task_index": 0,
                      "initial_state_index": 0, "target_description": "milk", "ground_truth_object_id": "object_002",
                      "input_sha256": "input", "old_image_path": str(old), "new_image_path": str(new),
                      "image_sha256": digest(old.read_bytes()), "new_image_sha256": digest(new.read_bytes()),
                      "localization_path": str(loc), "localization_sha256": digest(loc.read_bytes()),
                      "prompts": dict.fromkeys(CONDITIONS, "target"), "candidate_map": {"B": "object_002", "N": None}}
            completion = MagicMock()
            completion.model_dump.return_value = raw_response()
            with patch("scripts.run_vlm_visual_label_ablation.OUTPUT_ROOT", root), patch("openai.OpenAI"), \
                    patch("scripts.run_vlm_visual_label_ablation.paced_completion", return_value=completion) as call:
                first = run_pair(sample)
                self.assertEqual(first, run_pair(sample))
                self.assertEqual(call.call_count, 2)
                changed = {**sample, "input_sha256": "different"}
                with self.assertRaisesRegex(ValueError, "Resume"):
                    run_pair(changed)
                new.write_bytes(b"changed image")
                with self.assertRaisesRegex(ValueError, "image/localization"):
                    run_pair(sample)

    def test_paired_transitions_keep_old_new_direction(self):
        from scripts.analyze_vlm_visual_label_ablation import paired_transitions

        pairs = [(dict(correct=a, predicted_object_id="object_001"),
                  dict(correct=b, predicted_object_id="object_002"))
                 for a, b in ((False, True), (False, True), (True, False), (True, True))]
        result = paired_transitions(pairs)
        self.assertEqual(result["old_wrong_new_correct"], 2)
        self.assertEqual(result["old_correct_new_wrong"], 1)
        self.assertEqual(result["accuracy_difference_new_minus_old"], .25)

    def test_matched_coverage_keeps_ties_and_missing_scores_deferred(self):
        from scripts.analyze_vlm_visual_label_ablation import matched_coverage

        old = [{SCORE: s, "correct": c} for s, c in ((.9, True), (.9, False), (.1, True), (None, False))]
        new = [{SCORE: s, "correct": c} for s, c in ((.8, True), (.7, False), (.1, True), (None, False))]
        rows = matched_coverage(old, new)
        self.assertEqual([r["accepted_count"] for r in rows], [2, 3])
        self.assertEqual(rows[-1]["coverage"], .75)
        self.assertEqual(rows[-1]["human_intervention_rate"], .25)
        self.assertEqual(rows[-1]["old_false_accepts"], 1)
        self.assertEqual(rows[-1]["old_false_accept_rate_all"], .25)
        self.assertEqual(rows[-1]["old_false_accept_rate_accepted"], 1 / 3)
        with self.assertRaises(ValueError):
            matched_coverage(old, new[:-1])

    def test_development_thresholds_do_not_consult_test_data(self):
        from scripts.analyze_vlm_visual_label_ablation import freeze_development_points

        records = [{"condition": c, "partition": split, SCORE: score, "correct": correct}
                   for c in CONDITIONS for split in ("calibration", "validation", "test")
                   for score, correct in ((.9, True), (.6, False))]
        before = freeze_development_points(records)
        self.assertEqual(before[CONDITIONS[0]]["0.95"]["threshold"], .9)
        for row in records:
            if row["partition"] == "test":
                row[SCORE] = "must not inspect"
                row["correct"] = None
        self.assertEqual(freeze_development_points(records), before)

    def test_threshold_equality_uses_raw_likelihood(self):
        from scripts.analyze_vlm_uncertainty_scores import threshold_metrics

        rows = [{SCORE: .8, "correct": True}, {SCORE: .79, "correct": False},
                {SCORE: None, "correct": True}]
        result = threshold_metrics(rows, SCORE, .8)
        self.assertEqual(result["autonomous_decisions"], 1)
        self.assertEqual(result["autonomous_accuracy"], 1)
        self.assertEqual(result["deferred_decisions"], 2)

    def test_raw_auroc_uses_correctness_ranking_and_ties(self):
        from scripts.analyze_vlm_uncertainty_scores import score_distribution

        result = score_distribution([{SCORE: .8, "correct": True}, {SCORE: .8, "correct": False},
                                     {SCORE: .9, "correct": True}, {SCORE: None, "correct": False}], SCORE)
        self.assertEqual(result["correctness_ranking_auc"], .75)

    def test_offline_pair_audit_rejects_missing_duplicate_or_tampered_records(self):
        from scripts.analyze_vlm_visual_label_ablation import validate_records

        with tempfile.TemporaryDirectory() as folder:
            sample = {"sample_id": "s", "partition": "test", "task_index": 0,
                      "initial_state_index": 40, "target_description": "milk",
                      "ground_truth_object_id": "object_002", "input_sha256": "input",
                      "candidate_map": {"B": "object_002", "N": None}}
            rows = []
            for condition in CONDITIONS:
                path = Path(folder) / f"{condition}.json"
                raw = raw_response()
                path.write_text(json.dumps({"input_sha256": "input", "provider_response": raw}))
                rows.append({**sample, "condition": condition, "response_path": str(path),
                             "response_sha256": digest(path.read_bytes()),
                             **parse_response(raw, sample["candidate_map"]), "correct": True})
            self.assertEqual(len(validate_records(rows, [sample])), 1)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                validate_records(rows[:1], [sample])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_records([*rows, rows[0]], [sample])
            changed = copy.deepcopy(rows)
            changed[0][SCORE] = .99
            with self.assertRaisesRegex(ValueError, "generated decision"):
                validate_records(changed, [sample])
            changed = copy.deepcopy(rows)
            changed[0]["correct"] = False
            with self.assertRaisesRegex(ValueError, "ground-truth"):
                validate_records(changed, [sample])
            changed = copy.deepcopy(rows)
            changed[0]["partition"] = "validation"
            with self.assertRaisesRegex(ValueError, "partition"):
                validate_records(changed, [sample])
            Path(rows[0]["response_path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                validate_records(rows, [sample])

    def test_complete_offline_analysis_writes_figures_tables_and_frozen_thresholds(self):
        from scripts.analyze_vlm_visual_label_ablation import analyze

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples, records = [], []
            for index in range(500):
                state = index % 50
                sample = {"sample_id": str(index), "partition": "calibration" if state < 30 else "validation" if state < 40 else "test",
                          "task_index": index // 50, "initial_state_index": state, "target_description": f"target {index // 50}",
                          "ground_truth_object_id": "object_002", "input_sha256": str(index),
                          "candidate_map": {"A": "object_001", "B": "object_002", "N": None},
                          "visual_audit": {"unchanged_nonlabel_pixels": True}, "old_image_path": "old.png", "new_image_path": "new.png"}
                samples.append(sample)
                for condition in CONDITIONS:
                    text = "A" if state % (5 if condition == CONDITIONS[0] else 10) == 0 else "B"
                    raw = raw_response(text, -.2 if text == "A" else -.01)
                    path = root / f"{index}_{condition}.json"
                    path.write_text(json.dumps({"input_sha256": str(index), "provider_response": raw,
                                                "captured_at": "2026-09-06T12:00:00+00:00"}))
                    records.append({**{k: sample[k] for k in ("sample_id", "partition", "task_index", "initial_state_index",
                                        "target_description", "ground_truth_object_id", "input_sha256")},
                                    "condition": condition, "model": MODEL, "response_path": str(path),
                                    "response_sha256": digest(path.read_bytes()), **parse_response(raw, sample["candidate_map"]),
                                    "correct": text == "B"})
            payload = {"protocol": {"fixture": True}, "records": records}
            (root / "full_records.json").write_text(json.dumps(payload))
            with patch("scripts.analyze_vlm_visual_label_ablation.OUTPUT_ROOT", root), \
                    patch("scripts.analyze_vlm_visual_label_ablation.load_inputs", return_value={"protocol": payload["protocol"], "samples": samples}), \
                    patch("openai.OpenAI", side_effect=AssertionError("Offline analysis must not call API")), redirect_stdout(io.StringIO()):
                analyze()
                frozen_before = (root / "analysis/frozen_operating_points.json").read_bytes()
                analyze()
                self.assertEqual(frozen_before, (root / "analysis/frozen_operating_points.json").read_bytes())
            summary = json.loads((root / "analysis/summary.json").read_text())
            self.assertEqual(summary["conditions"][CONDITIONS[0]]["all"]["correct"], 400)
            self.assertEqual(summary["conditions"][CONDITIONS[1]]["all"]["correct"], 450)
            self.assertEqual(summary["paired_transitions"]["all"]["old_wrong_new_correct"], 50)
            for name in ("REPORT.md", "comparison.png", "per_target.png", "associations.csv", "paired_transitions.csv",
                         "per_target.csv", "threshold_curves.csv", "matched_coverage.csv", "high_confidence_wrong.json"):
                self.assertTrue((root / "analysis" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
