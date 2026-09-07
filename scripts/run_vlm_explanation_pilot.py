"""Thirty fresh calls: label only, label/evidence, evidence/label on ten state-0 scenes.

No-N policy fixed across conditions; no production changes. Run check/run/report.
Evidence is short visible-feature text, never an input to the decision gate.
"""

from datetime import datetime, timezone
from itertools import permutations
import json
import math
from pathlib import Path
import re
import sys

from scripts.run_vlm_output_format_ablation import MODEL, SCENE_ROOT, digest, paced_completion, request_arguments
from scripts.run_vlm_presence_pilot import OUTPUT_ROOT as PRESENCE_ROOT, condition_prompt, select_samples
from scripts.run_vlm_visual_label_ablation import save_json


OUTPUT_ROOT = SCENE_ROOT.parent / "explanation_pilot"
CONDITIONS = ("label_only", "label_then_evidence", "evidence_then_label")
ORDERS = tuple(permutations(CONDITIONS))
SCORE = "raw_sequence_likelihood"
SYSTEM_PROMPT = "Associate one semantic target with an already-localized candidate. Follow the requested output format."
MAX_COMPLETION_TOKENS = 128  # Same budget for all conditions, including fresh label-only control.


def build_prompt(sample, condition):
    prompt, mapping = condition_prompt(sample, "known_present_no_n")
    if condition == "label_only":
        return prompt, mapping
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    body, _old_format = prompt.rsplit("\n", 1)
    label = "one label, without punctuation, from: " + ", ".join(mapping)
    evidence = "one short sentence (at most 20 words) describing visible evidence for this choice"
    first, second = (label, evidence) if condition == "label_then_evidence" else (evidence, label)
    return f"{body}\nReturn exactly two lines and nothing else.\nFirst line: {first}.\nSecond line: {second}.", mapping


def arguments_for(image_bytes, prompt):
    args = request_arguments(image_bytes, prompt)
    args["messages"][0]["content"] = SYSTEM_PROMPT
    args["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    return args


def parse_response(raw, mapping, condition):
    """Score the designated label span in ORIGINAL tokens, not label mentions in evidence.

    The frozen earlier parser only accepts leading decisions. This experiment needs
    arbitrary output positions without modifying those reproducibility-pinned sources.
    Separable whitespace/punctuation is ignored; mixed label/format tokens are indivisible.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    output = raw["choices"][0]
    text = output["message"].get("content") or ""
    tokens = (output.get("logprobs") or {}).get("content") or []
    result = {"generated_output_text": text, "choice": None, "predicted_object_id": None,
              "explanation": None, "explanation_word_count": 0, "format_valid": False,
              "decision_tokens": [], "decision_token_count": 0, "output_token_count": len(tokens),
              "raw_log_probability": None, SCORE: None}
    if output["finish_reason"] != "stop":
        return {**result, "score_unavailable_reason": "unfinished response"}
    lines = [m for m in re.finditer(r"[^\r\n]+", text) if m[0].strip()]
    expected_lines = 1 if condition == "label_only" else 2
    if len(lines) != expected_lines:
        return {**result, "score_unavailable_reason": "invalid response line count"}
    line = lines[-1] if condition == "evidence_then_label" else lines[0]
    labels = "|".join(re.escape(label) for label in mapping)
    formatting = r"[\s`\"'\[\](){}.,:;*!]"
    match = re.fullmatch(rf"{formatting}*(?P<label>{labels}){formatting}*", line[0])
    if match is None:
        return {**result, "score_unavailable_reason": "invalid designated choice line"}
    choice = match["label"]
    start, end = (line.start() + p for p in match.span("label"))
    result.update(choice=choice, predicted_object_id=mapping[choice], format_valid=True,
                  decision_character_span=[start, end])
    if condition != "label_only":
        evidence = lines[0 if condition == "evidence_then_label" else 1][0].strip()
        result.update(explanation=evidence, explanation_word_count=len(evidence.split()))
    # A word-budget overrun is recorded, not a reason to cherry-pick a new response.
    if "".join(t["token"] for t in tokens) != text:
        return {**result, "score_unavailable_reason": "token text unavailable or mismatches output"}
    selected, offset = [], 0
    for index, token in enumerate(tokens):
        stop = offset + len(token["token"])
        if offset < end and stop > start:
            selected.append({"index": index, "token": token["token"], "logprob": token.get("logprob")})
        offset = stop
    result.update(decision_tokens=selected, decision_token_count=len(selected))
    if not selected or any(t["logprob"] is None or not math.isfinite(t["logprob"])
                           or t["logprob"] > 0 or t["logprob"] <= -9999 for t in selected):
        return {**result, "score_unavailable_reason": "decision-token logprobs unavailable"}
    logprob = math.fsum(t["logprob"] for t in selected)
    result.update(raw_log_probability=logprob, raw_sequence_likelihood=math.exp(logprob))
    return result


def prepare():
    previous = json.loads((PRESENCE_ROOT / "inputs.json").read_text())
    for name, expected in previous["protocol"]["source_sha256"].items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    samples = select_samples(previous["samples"])
    for sample in samples:
        gt = SCENE_ROOT / sample["evaluation_ground_truth_path"]
        for path, expected in ((Path(sample["new_image_path"]), sample["new_image_sha256"]),
                               (Path(sample["localization_path"]), sample["localization_sha256"]),
                               (gt, sample["ground_truth_sha256"])):
            if digest(path.read_bytes()) != expected:
                raise ValueError(f"Frozen input changed: {path}")
        if json.loads(gt.read_text())["ground_truth_object_id"] != sample["ground_truth_object_id"]:
            raise ValueError("Ground truth and manifest disagree.")
    args = arguments_for(b"", "")
    protocol = {
        "selection": "initial state 0 for all ten targets; calibration only, not selected by correctness",
        "conditions": list(CONDITIONS), "fresh_calls": 30, "none_policy": "known present; A-G only",
        "settings": {k: v for k, v in args.items() if k != "messages"},
        "system_prompt": SYSTEM_PROMPT, "image_detail": "high",
        "common_changes_from_previous_pilot": "format-neutral system instruction and 128 output tokens in ALL arms; fresh baseline",
        "order": "cycle all six condition permutations by task index, balanced as closely as ten scenes allow",
        "score": "exp(sum(original generated decision-bearing token logprobs)); no length/candidate normalization",
        "interpretation": "evidence-first label likelihood is conditioned on generated evidence, not marginal identity correctness",
        "analysis": "descriptive paired accuracy, label likelihood distributions/AUC, tie-preserving risk-coverage and matched-coverage false accepts; no threshold selection",
        "source_sha256": {**previous["protocol"]["source_sha256"],
                          **{name: digest(Path(name).read_bytes()) for name in (
                              "scripts/run_vlm_explanation_pilot.py", "scripts/analyze_vlm_uncertainty_scores.py",
                              "scripts/analyze_vlm_output_format_ablation.py", "scripts/analyze_vlm_visual_label_ablation.py")}},
    }
    rows = [{**s, "experiment_prompts": {c: build_prompt(s, c)[0] for c in CONDITIONS}} for s in samples]
    protocol["inputs_sha256"] = digest(json.dumps(rows, sort_keys=True).encode())
    payload = {"protocol": protocol, "samples": rows}
    (OUTPUT_ROOT / "responses").mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Pilot inputs/settings changed; refusing to mix cached runs.")
    if not path.exists():
        save_json(path, payload)
    print("Verified ten state-0 scenes and three fixed-format conditions; 30 calls, excluding API retries.", flush=True)
    return payload


def run_scene(sample, client, output_root):
    image = Path(sample["new_image_path"]).read_bytes()
    if (digest(image) != sample["new_image_sha256"]
            or digest(Path(sample["localization_path"]).read_bytes()) != sample["localization_sha256"]):
        raise ValueError("Frozen image/localization changed before request.")
    rows = []
    for index, condition in enumerate(ORDERS[sample["task_index"] % len(ORDERS)]):
        prompt, mapping = build_prompt(sample, condition)
        args = arguments_for(image, prompt)
        request_hash = digest(json.dumps(args, sort_keys=True).encode())
        path = output_root / "responses" / f"{sample['sample_id']}_{condition}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_sha256"] != request_hash or saved["input_sha256"] != sample["input_sha256"]:
                raise ValueError("Resume request/input mismatch.")
        else:
            raw = paced_completion(client, args).model_dump(mode="json", exclude_none=True)
            saved = {"condition": condition, "request_sha256": request_hash, "input_sha256": sample["input_sha256"],
                     "condition_order_index": index, "captured_at": datetime.now(timezone.utc).isoformat(),
                     "provider_response": raw}
            save_json(path, saved)
        raw = saved["provider_response"]
        if raw["model"] != MODEL:
            raise ValueError("Returned model differs from the frozen snapshot.")
        decision = parse_response(raw, mapping, condition)
        row = {key: sample[key] for key in ("sample_id", "partition", "task_index", "initial_state_index",
                                           "target_description", "ground_truth_object_id", "input_sha256",
                                           "localization_sha256")}
        row.update(condition=condition, provider="openai", model=raw["model"], **decision,
                   correct=decision["choice"] is not None and decision["predicted_object_id"] == sample["ground_truth_object_id"],
                   score_type="raw_label_likelihood", candidate_map=mapping,
                   image_sha256=sample["new_image_sha256"], request_sha256=request_hash,
                   response_path=str(path), response_sha256=digest(path.read_bytes()))
        rows.append(row)
        print(f"{sample['target_description']} | {condition}: {row['choice']} "
              f"correct={row['correct']} p={row[SCORE]}", flush=True)
    return rows


def summarize(records):
    from scripts.analyze_vlm_output_format_ablation import metrics
    from scripts.analyze_vlm_uncertainty_scores import risk_coverage_curve
    from scripts.analyze_vlm_visual_label_ablation import matched_coverage

    groups = {}
    for row in records:
        group = groups.setdefault(row["sample_id"], {})
        if row["condition"] in group:
            raise ValueError("Duplicate scene/condition.")
        group[row["condition"]] = row
    if not groups or any(set(g) != set(CONDITIONS) for g in groups.values()):
        raise ValueError("Expected complete three-condition scene groups.")
    baseline = [g["label_only"] for g in groups.values()]
    summary, curves, matched = {}, [], []
    for condition in CONDITIONS:
        rows = [g[condition] for g in groups.values()]
        summary[condition] = metrics(rows)
        summary[condition]["format_valid_count"] = sum(r["format_valid"] for r in rows)
        summary[condition]["explanation_word_budget_overruns"] = sum(r["explanation_word_count"] > 20 for r in rows)
        summary[condition]["transitions_from_label_only"] = {
            "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(baseline, rows)),
            "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(baseline, rows)),
            "prediction_changed": sum(a["predicted_object_id"] != b["predicted_object_id"] for a, b in zip(baseline, rows)),
        }
        curves.extend({"condition": condition, **p} for p in risk_coverage_curve(rows, SCORE)["points"])
        if condition != "label_only":
            matched.extend({"condition": condition, **p} for p in matched_coverage(baseline, rows))
    return summary, curves, matched


def report(output_root=OUTPUT_ROOT):
    from scripts.analyze_vlm_visual_label_ablation import write_csv

    payload = json.loads((output_root / "records.json").read_text())
    records = payload["records"]
    summary, curves, matched = summarize(records)
    save_json(output_root / "summary.json", summary)
    write_csv(output_root / "associations.csv", records)
    write_csv(output_root / "threshold_curves.csv", curves)
    if matched:
        write_csv(output_root / "matched_coverage.csv", matched)
    fmt = lambda x: "unavailable" if x is None else f"{x:.9g}"
    lines = ["# Short-evidence explanation pilot", "",
             "Ten frozen LIBERO-Object state-0 calibration scenes, 30 fresh responses. "
             "No-N/known-presence policy and direct A-G visual labels fixed across all three conditions. "
             "Scores below include only the generated decision label, not explanation likelihood.", "",
             "## Results", "",
             "| Format | Correct / 10 | Mean label likelihood | Range | AUROC | Wrong >=0.99 | Exact-one scores |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        m, d = summary[c], summary[c]["likelihood"]
        lines.append(f"| {c} | {m['correct']} | {fmt(d['mean'] if d else None)} | "
                     f"{fmt(d['min'] if d else None)} - {fmt(d['max'] if d else None)} | "
                     f"{fmt(m['by_correctness']['correctness_ranking_auc'])} | {m['incorrect_with_score_ge_099']} | {m['exact_one_scores']} |")
    lines += ["", "AUROC is undefined if there are no scored errors or no scored correct predictions. "
              "A lower likelihood or a wider range alone is not evidence of a better deferral signal.", "",
              "## Every target", "",
              "| Target | Ground truth | Label only: choice / correct / score | Label then evidence | Evidence then label |",
              "| --- | --- | --- | --- | --- |"]
    for sample_id in sorted({r["sample_id"] for r in records}):
        rows = [next(r for r in records if r["sample_id"] == sample_id and r["condition"] == c) for c in CONDITIONS]
        cells = [f"{r['choice']} / {r['correct']} / {fmt(r[SCORE])}" for r in rows]
        lines.append(f"| {rows[0]['target_description']} | {rows[0]['ground_truth_object_id']} | " + " | ".join(cells) + " |")
    lines += ["", "## False acceptances at exactly matched coverage", "",
              "Only attainable thresholds; tied scores are not split. No operating threshold is selected. "
              "Unavailable scores count as deferred. These are exploratory calibration-scene comparisons.", "",
              "| Explanation format | Coverage | Label-only false accepts | Explanation false accepts | Label-only accuracy | Explanation accuracy |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in matched:
        lines.append(f"| {p['condition']} | {p['coverage']:.0%} | {p['old_false_accepts']} | {p['new_false_accepts']} | "
                     f"{fmt(p['old_accuracy'])} | {fmt(p['new_accuracy'])} |")
    lines += ["", "## Alphabet soup and tomato sauce: actual output text", "",
              "These are model-generated visual claims, not verified descriptions or ground-truth evidence.", ""]
    for r in records:
        if r["target_description"] in ("alphabet soup", "tomato sauce"):
            lines += [f"### {r['target_description']} — {r['condition']}", "",
                      f"Correct: {r['correct']}; label likelihood: {fmt(r[SCORE])}", "",
                      "```text", r["generated_output_text"], "```", ""]
    lines += ["## Controlled protocol and limitations", "",
              f"Same `{MODEL}`, temperature 0, detail high, frozen RGB/localization/GT/candidate order and crops. "
              "All arms use a shared format-neutral system instruction and max_completion_tokens=128 "
              "to accommodate the explanation; hence the fresh label-only control is the valid comparison, "
              "not a historical 8-token response. No API history from other conditions, logit bias, "
              "constrained decoding, or self-reported confidence. The 20-word evidence request is identical "
              "in both explanation arms, with order reversed. Word-budget overruns are logged, never rerun.", "",
              "Parser locates the designated first/last label line and sums ONLY overlapping original token "
              "logprobs. Standalone whitespace/punctuation is excluded. If a label shares a token with "
              "formatting, that token is indivisible and included. Missing/sentinel probabilities, malformed "
              "output or truncation do not create a score. All provider tokens/alternatives remain in responses/.", "",
              "Evidence-first likelihood is conditioned on the already-generated explanation. It may express "
              "consistency with that explanation rather than correct object identity. Evidence written after "
              "the label cannot retroactively modify the earlier token probability, although asking for it "
              "changes the input prompt. Neither score is calibrated correctness probability.", "",
              "Only ten previously inspected calibration scenes, one response per condition: insufficient "
              "for robust model/threshold selection or causal claims beyond this pilot. Fixed temperature "
              "does not quantify API repeatability. Each condition's own prediction defines correctness; "
              "accuracy changes and score discrimination are reported separately. No validation/test "
              "outcomes read, no production/HITL/CAD/planner edits, no automatic follow-up experiment.", "",
              "inputs.json records exact prompts, settings, image/localization/GT/source hashes. "
              "records.json/associations.csv save explanations and decision tokens. summary.json contains "
              "correct/incorrect distributions and paired transitions. threshold_curves.csv includes "
              "accuracy, coverage, false accepts and risk; AURC uses existing trapezoidal tied-endpoint "
              "convention, not attainable within-tie interpolation.", "",
              "```sh", "python -m scripts.run_vlm_explanation_pilot check",
              "python -m scripts.run_vlm_explanation_pilot run",
              "python -m scripts.run_vlm_explanation_pilot report", "```", "",
              "`run` resumes missing calls; `report` is offline. "
              "[Official output-token logprob API](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).", ""]
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({c: {k: summary[c][k] for k in ("correct", "score_availability", "likelihood", "by_correctness")}
                      for c in CONDITIONS}, indent=2))
    print(f"Report: {output_root / 'REPORT.md'}")


def run(stage):
    if stage == "report":
        return report()
    if stage not in ("check", "run"):
        raise ValueError("Use check, run, or report; no full-dataset mode.")
    payload = prepare()
    if stage == "check":
        return
    from openai import OpenAI

    records = []
    with OpenAI(timeout=120, max_retries=3) as client:
        for sample in payload["samples"]:
            records.extend(run_scene(sample, client, OUTPUT_ROOT))
    records.sort(key=lambda r: (r["sample_id"], r["condition"]))
    save_json(OUTPUT_ROOT / "records.json", {"protocol": payload["protocol"], "records": records})
    report()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
