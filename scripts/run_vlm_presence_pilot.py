"""Ten frozen state-0 pairs: existing A-G/N prompt versus known presence, A-G only.

Evaluation only. Run `check`, `run` (resumable 20 calls), or `report` (offline).
Production NONE/HITL behavior is unchanged. No threshold is selected here.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

from scripts.run_vlm_output_format_ablation import (
    MODEL, SCENE_ROOT, SYSTEM_PROMPT, digest, paced_completion, request_arguments,
)
from scripts.run_vlm_visual_label_ablation import load_inputs, parse_response, save_json


OUTPUT_ROOT = SCENE_ROOT.parent / "known_presence_pilot"
CONDITIONS = ("with_n", "known_present_no_n")
NONE_SENTENCE = "N means none of the localized candidates corresponds to the target."
PRESENCE_SENTENCE = "The target is guaranteed to correspond to one of the localized candidates."
SCORE = "raw_sequence_likelihood"


def select_samples(samples):
    selected = sorted((s for s in samples if s["initial_state_index"] == 0),
                      key=lambda s: s["task_index"])
    if [s["task_index"] for s in selected] != list(range(10)):
        raise ValueError("Expected exactly one state-0 scene for each of the ten targets.")
    for sample in selected:
        mapping = sample["candidate_map"]
        if (sample["partition"] != "calibration" or not sample["target_localized"]
                or sample["ground_truth_object_id"] is None
                or sample["ground_truth_object_id"] not in mapping.values()):
            raise ValueError("Known-presence pilot requires a localized calibration target.")
        if list(mapping) != [*"ABCDEFG", "N"] or mapping["N"] is not None:
            raise ValueError("Expected the frozen A-G/N candidate map.")
    return selected


def condition_prompt(sample, condition):
    prompt = sample["prompts"]["new_direct_letters"]
    mapping = sample["candidate_map"]
    if condition == "with_n":
        return prompt, dict(mapping)
    if condition != "known_present_no_n":
        raise ValueError(f"Unknown condition: {condition}")
    if prompt.count(NONE_SENTENCE) != 1 or not prompt.endswith(", N"):
        raise ValueError("Frozen prompt layout changed; cannot isolate the presence edit.")
    return (prompt[:-3].replace(NONE_SENTENCE, PRESENCE_SENTENCE),
            {label: object_id for label, object_id in mapping.items() if label != "N"})


def prepare():
    # Read frozen inputs, not previous prediction/results files; never select by correctness.
    samples = select_samples(load_inputs()["samples"])
    for sample in samples:
        gt = SCENE_ROOT / sample["evaluation_ground_truth_path"]
        for path, expected in (
            (Path(sample["new_image_path"]), sample["new_image_sha256"]),
            (Path(sample["localization_path"]), sample["localization_sha256"]),
            (gt, sample["ground_truth_sha256"]),
        ):
            if digest(path.read_bytes()) != expected:
                raise ValueError(f"Frozen input changed: {path}")
        if json.loads(gt.read_text())["ground_truth_object_id"] != sample["ground_truth_object_id"]:
            raise ValueError("Frozen ground truth and input manifest disagree.")
    args = request_arguments(b"", "")
    protocol = {
        "selection": "initial state 0 for each of ten targets; calibration only; not selected by outcomes",
        "conditions": list(CONDITIONS), "fresh_calls": 20,
        "settings": {k: v for k, v in args.items() if k != "messages"},
        "system_prompt": SYSTEM_PROMPT, "image_detail": "high",
        "visual_input": "same frozen direct-letter contact sheet in both requests; no re-rendering",
        "score": "exp(sum(generated decision-bearing token logprobs)); no candidate/length normalization",
        "pair_order": "with_n first for even task index, no_n first for odd task index",
        "treatment": "combined target-present assertion and removal of N, not two separate interventions",
        "threshold": "not fitted or changed; descriptive ten-scene pilot only",
        "source_sha256": {name: digest(Path(name).read_bytes()) for name in (
            "scripts/run_vlm_presence_pilot.py", "scripts/run_vlm_visual_label_ablation.py",
            "scripts/run_vlm_output_format_ablation.py", "prompt.py", "vlm_module.py")},
    }
    payload = {"protocol": protocol, "samples": [
        {**s, "pilot_prompts": {c: condition_prompt(s, c)[0] for c in CONDITIONS}}
        for s in samples]}
    (OUTPUT_ROOT / "responses").mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Pilot inputs/settings changed; refusing to mix cached runs.")
    if not path.exists():
        save_json(path, payload)
    print("Verified ten state-0 calibration pairs; 20 calls maximum, excluding API retries.", flush=True)
    return payload


def run_pair(sample, client, output_root):
    image_bytes = Path(sample["new_image_path"]).read_bytes()
    if (digest(image_bytes) != sample["new_image_sha256"]
            or digest(Path(sample["localization_path"]).read_bytes()) != sample["localization_sha256"]):
        raise ValueError("Frozen image/localization changed before request.")
    rows = []
    order = CONDITIONS if sample["task_index"] % 2 == 0 else CONDITIONS[::-1]
    for index, condition in enumerate(order):
        prompt, mapping = condition_prompt(sample, condition)
        args = request_arguments(image_bytes, prompt)
        request_hash = digest(json.dumps(args, sort_keys=True).encode())
        path = output_root / "responses" / f"{sample['sample_id']}_{condition}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["request_sha256"] != request_hash
                    or saved["input_sha256"] != sample["input_sha256"]):
                raise ValueError("Resume request/input mismatch.")
        else:
            raw = paced_completion(client, args).model_dump(mode="json", exclude_none=True)
            saved = {"request_sha256": request_hash, "input_sha256": sample["input_sha256"],
                     "condition": condition, "pair_order_index": index,
                     "captured_at": datetime.now(timezone.utc).isoformat(), "provider_response": raw}
            save_json(path, saved)  # Preserve raw tokens/top alternatives before parsing.
        raw = saved["provider_response"]
        if raw["model"] != MODEL:
            raise ValueError("Returned model differs from the frozen snapshot.")
        decision = parse_response(raw, mapping)
        row = {key: sample[key] for key in ("sample_id", "partition", "task_index", "initial_state_index",
                                           "target_description", "ground_truth_object_id", "input_sha256")}
        row.update(condition=condition, provider="openai", model=raw["model"], **decision,
                   correct=decision["choice"] is not None and decision["predicted_object_id"] == sample["ground_truth_object_id"],
                   image_sha256=sample["new_image_sha256"], request_sha256=request_hash,
                   response_path=str(path), response_sha256=digest(path.read_bytes()),
                   score_type="raw_label_likelihood")
        rows.append(row)
        print(f"{sample['target_description']} | {condition}: {decision['choice']} "
              f"correct={row['correct']} p={row[SCORE]}", flush=True)
    return rows


def summarize(records):
    by_sample = {}
    for row in records:
        group = by_sample.setdefault(row["sample_id"], {})
        if row["condition"] in group:
            raise ValueError("Duplicate condition for a scene.")
        group[row["condition"]] = row
    pairs = []
    for sample_id, group in sorted(by_sample.items()):
        if set(group) != set(CONDITIONS):
            raise ValueError("Cannot compare an incomplete pair.")
        a, b = (group[c] for c in CONDITIONS)
        pair = {"sample_id": sample_id, "target_description": a["target_description"],
                "ground_truth_object_id": a["ground_truth_object_id"],
                "prediction_changed": a["predicted_object_id"] != b["predicted_object_id"]}
        for prefix, row in (("with_n", a), ("no_n", b)):
            for key in ("choice", "predicted_object_id", "correct", SCORE, "raw_log_probability"):
                pair[f"{prefix}_{key}"] = row[key]
        pair["likelihood_delta_no_n_minus_with_n"] = b[SCORE] - a[SCORE] if a[SCORE] is not None and b[SCORE] is not None else None
        pairs.append(pair)
    stats = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        scores = [r[SCORE] for r in rows if r[SCORE] is not None]
        stats[condition] = {"n": len(rows), "correct": sum(r["correct"] for r in rows),
                            "scores_available": len(scores),
                            "mean": statistics.mean(scores) if scores else None,
                            "median": statistics.median(scores) if scores else None,
                            "min": min(scores) if scores else None, "max": max(scores) if scores else None,
                            "exactly_one": scores.count(1.),
                            "wrong_ge_099": sum(not r["correct"] and r[SCORE] is not None and r[SCORE] >= .99 for r in rows)}
    return {"conditions": stats, "pairs": pairs}


def report(output_root=OUTPUT_ROOT):
    payload = json.loads((output_root / "records.json").read_text())
    summary = summarize(payload["records"])
    save_json(output_root / "summary.json", summary)
    pairs = summary["pairs"]
    with (output_root / "paired_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = ["# Known-presence / no-N pilot", "",
             "Ten frozen LIBERO-Object calibration scenes, initial state 0 per target, selected by index "
             "rather than accuracy. Twenty fresh requests (checkpoints reused on resume). No validation/test "
             "outcomes read. This pilot changes only the presence/N prompt, not visual labels or images.", "",
             "## Prompt change", "", "Replace:", "", f"> {NONE_SENTENCE}", "", "with:", "",
             f"> {PRESENCE_SENTENCE}", "",
             "The output list changes from A, B, C, D, E, F, G, N to A, B, C, D, E, F, G. "
             "Every target is confirmed in the frozen localized candidate set. The specific ground-truth "
             "ID is never disclosed. Candidate IDs/mapping/order, crops and geometry stay fixed. "
             "This combines an explicit presence assertion with removal of N; it cannot isolate their separate effects.", "",
             f"Model: `{MODEL}`; temperature 0; image detail high; identical direct-letter 1792×896 contact "
             "sheets from frozen 768×768 RGB. Same system prompt, max_completion_tokens=8, logprobs=True, "
             "top_logprobs=20. Pair order alternates by task index. No logit bias or constrained decoding.", "",
             "Score = exp(sum of generated decision-bearing token log probabilities). Separable formatting "
             "is excluded. No sequence-length or candidate normalization. Top alternatives are archived only. "
             "Missing probabilities/invalid outputs remain unavailable, not zero. Production prompt, "
             "HITL, final_object_id, CAD and thresholds are untouched.", "",
             "## Paired results", "",
             "| Target | With N: choice / correct | Raw likelihood | No N: choice / correct | Raw likelihood | Change |",
             "| --- | --- | ---: | --- | ---: | ---: |"]
    fmt = lambda value: "unavailable" if value is None else f"{value:.12g}"
    for p in pairs:
        lines.append(f"| {p['target_description']} | {p['with_n_choice']} / {p['with_n_correct']} | "
                     f"{fmt(p['with_n_raw_sequence_likelihood'])} | {p['no_n_choice']} / {p['no_n_correct']} | "
                     f"{fmt(p['no_n_raw_sequence_likelihood'])} | {fmt(p['likelihood_delta_no_n_minus_with_n'])} |")
    lines += ["", "## Summary", "", "```json", json.dumps(summary["conditions"], indent=2), "```", "",
              "## Interpretation limits", "",
              "Higher generated-label likelihood is not necessarily better recognition or calibrated correctness. "
              "One response per condition cannot separate prompt effects from API repeatability variation. "
              "Ten scenes with one state per target are a quick score-sensitivity check, not a reliable threshold "
              "calibration or generalization experiment. Additional states test pose/occlusion robustness; "
              "they do not add semantic categories. No extra states or absent-target trials were run.", "",
              "inputs.json preserves both exact prompts, mappings and image/localization/GT/settings/source hashes. "
              "responses/ preserves raw provider responses, tokens and alternatives. records.json, summary.json "
              "and paired_scores.csv preserve full numeric precision; this table is display-rounded.", "",
              "```sh", "python -m scripts.run_vlm_presence_pilot check",
              "python -m scripts.run_vlm_presence_pilot run",
              "python -m scripts.run_vlm_presence_pilot report", "```", "",
              "`run` resumes only missing requests; `report` is offline. No automatic full experiment.", "",
              "API reference: [OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create). "
              "The official SDK token logprobs/top alternatives structure was checked; scoring reuses the existing parser.", ""]
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["conditions"], indent=2))
    print(f"Report: {output_root / 'REPORT.md'}")


def run(stage):
    if stage == "report":
        return report()
    if stage not in ("check", "run"):
        raise ValueError("Use check, run, or report; this pilot has no full-dataset mode.")
    payload = prepare()
    if stage == "check":
        return
    from openai import OpenAI

    records = []
    with OpenAI(timeout=120, max_retries=3) as client:
        for sample in payload["samples"]:
            records.extend(run_pair(sample, client, OUTPUT_ROOT))
    records.sort(key=lambda r: (r["sample_id"], r["condition"]))
    save_json(OUTPUT_ROOT / "records.json", {"protocol": payload["protocol"], "records": records})
    report()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
