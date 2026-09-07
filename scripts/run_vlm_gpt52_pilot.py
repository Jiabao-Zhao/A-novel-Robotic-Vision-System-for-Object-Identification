"""GPT-5.2 two-format pilot on 32 frozen scenes; no production model change.

check/probe/run/report: probe is the first trial, cached and included in the pilot.
The comparison reuses GPT-4.1-mini records for exactly the same selected scenes.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from scripts.run_vlm_explanation_expansion import (
    CONDITIONS, OUTPUT_ROOT as BASELINE_ROOT, checked_file, validate_pairs,
)
from scripts.run_vlm_explanation_pilot import (
    MODEL as BASELINE_MODEL, SCORE, SYSTEM_PROMPT, arguments_for, build_prompt, parse_response,
)
from scripts.run_vlm_output_format_ablation import SCENE_ROOT, digest, paced_completion
from scripts.run_vlm_visual_label_ablation import save_json


MODEL = "gpt-5.2-2025-12-11"
OUTPUT_ROOT = BASELINE_ROOT.parent / "gpt52_explanation_pilot"
STATES = (1, 2)
WORKERS = 4


def select_samples(samples):
    states = sorted((s for s in samples if s["cohort"] == "additional_states"
                     and s["initial_state_index"] in STATES),
                    key=lambda s: (s["initial_state_index"], s["task_index"]))
    if [(s["initial_state_index"], s["task_index"]) for s in states] != [(i, t) for i in STATES for t in range(10)]:
        raise ValueError("Expected exactly 20 state-1/2 LIBERO-Object calibration scenes.")
    other = sorted((s for s in samples if s["cohort"] == "additional_objects"), key=lambda s: s["sample_id"])
    if len(other) != 12 or any(s["partition"] != "calibration" for s in states) or any(
            s["partition"] != "exploratory_new_objects" for s in other):
        raise ValueError("Expected 12 exploratory captures; no validation/test scenes permitted.")
    rows = states + other
    if len({s["sample_id"] for s in rows}) != 32 or any(
            s["ground_truth_object_id"] is None or s["ground_truth_object_id"] not in s["candidate_map"].values() for s in rows):
        raise ValueError("Duplicate sample or unresolved target in pilot.")
    return rows


def request_arguments(image, prompt):
    # Actual GPT-5.2 endpoint rejected 20; alternatives are diagnostic-only.
    return {**arguments_for(image, prompt), "model": MODEL, "reasoning_effort": "none", "top_logprobs": 5}


def prepare():
    import inspect
    import openai
    from openai.resources.chat.completions import Completions

    parent = json.loads((BASELINE_ROOT / "inputs.json").read_text())
    for name, expected in parent["protocol"]["source_sha256"].items():
        checked_file(name, expected)
    rows = select_samples(parent["samples"])
    for sample in rows:
        for path, expected in ((sample["new_image_path"], sample["new_image_sha256"]),
                               (sample["localization_path"], sample["localization_sha256"])):
            checked_file(path, expected)
        gt = (SCENE_ROOT / sample["evaluation_ground_truth_path"] if sample["cohort"] == "additional_states"
              else Path(sample["localization_path"]).parent / "evaluation_ground_truth.json")
        checked_file(gt, sample["ground_truth_sha256"])
        if json.loads(gt.read_text())["ground_truth_object_id"] != sample["ground_truth_object_id"]:
            raise ValueError("Frozen evaluation identity mismatch.")
        for condition in CONDITIONS:
            if build_prompt(sample, condition)[0] != sample["experiment_prompts"][condition]:
                raise ValueError("Frozen prompt changed.")
    params = inspect.signature(Completions.create).parameters
    required = ("reasoning_effort", "logprobs", "top_logprobs", "max_completion_tokens")
    if not all(name in params for name in required):
        raise RuntimeError("Installed OpenAI SDK lacks required Chat Completions parameters.")
    protocol = {
        "model": MODEL, "baseline_model": BASELINE_MODEL, "openai_sdk_version": openai.__version__,
        "selection": "states 1,2 x ten targets + all 12 usable other-object captures; 32 scenes / 64 fresh calls",
        "conditions": list(CONDITIONS), "settings": {k: v for k, v in request_arguments(b"", "").items() if k != "messages"},
        "system_prompt": SYSTEM_PROMPT, "pair_order": "same fixed alternating order as parent; four concurrent pairs",
        "none_policy": "known-present/no N, unchanged from both parent conditions",
        "score": "raw generated decision-label likelihood only; explanation and alternatives excluded",
        "compatibility_adjustment": "GPT-5.2 rejected top_logprobs=20 (HTTP 400, maximum 5). Request 5 diagnostic alternatives versus 20 for historical mini; no score-definition change.",
        "baseline_inputs_sha256": digest((BASELINE_ROOT / "inputs.json").read_bytes()),
        "baseline_records_sha256": digest((BASELINE_ROOT / "records.json").read_bytes()),
        "source_sha256": {**parent["protocol"]["source_sha256"],
                          "scripts/run_vlm_gpt52_pilot.py": digest(Path(__file__).read_bytes())},
        "documentation": ["https://developers.openai.com/api/docs/models/gpt-5.2",
                          "https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2"],
    }
    payload = {"protocol": protocol, "samples": rows}
    (OUTPUT_ROOT / "responses").mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Pilot inputs/settings changed; refusing to mix runs.")
    if not path.exists():
        save_json(path, payload)
    print(f"Verified 32 frozen scenes; {MODEL}; SDK {openai.__version__}; exactly two output formats.", flush=True)
    return payload


def run_trial(sample, condition, client, output_root):
    if condition not in CONDITIONS:
        raise ValueError("Only label-only and label-then-explanation are permitted.")
    checked_file(sample["new_image_path"], sample["new_image_sha256"])
    checked_file(sample["localization_path"], sample["localization_sha256"])
    prompt, mapping = build_prompt(sample, condition)
    args = request_arguments(Path(sample["new_image_path"]).read_bytes(), prompt)
    request_hash = digest(json.dumps(args, sort_keys=True).encode())
    path = output_root / "responses" / f"{sample['sample_id']}_{condition}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["request_sha256"] != request_hash or saved["input_sha256"] != sample["input_sha256"]:
            raise ValueError("Resume request/input mismatch.")
    else:
        raw = paced_completion(client, args).model_dump(mode="json", exclude_none=True)
        saved = {"condition": condition, "request_sha256": request_hash, "input_sha256": sample["input_sha256"],
                 "captured_at": datetime.now(timezone.utc).isoformat(), "provider_response": raw}
        save_json(path, saved)
    raw = saved["provider_response"]
    if raw["model"] != MODEL:
        raise ValueError("Provider returned a different model snapshot.")
    decision = parse_response(raw, mapping, condition)
    row = {k: sample[k] for k in ("sample_id", "cohort", "partition", "target_description", "ground_truth_object_id",
                                 "initial_state_index", "input_sha256", "localization_sha256")}
    row.update(condition=condition, model=raw["model"], provider="openai", **decision,
               correct=decision["choice"] is not None and decision["predicted_object_id"] == sample["ground_truth_object_id"],
               candidate_map=mapping, image_sha256=sample["new_image_sha256"], request_sha256=request_hash,
               response_path=str(path), response_sha256=digest(path.read_bytes()), score_type="raw_label_likelihood",
               system_fingerprint=raw.get("system_fingerprint"), finish_reason=raw["choices"][0]["finish_reason"],
               usage=raw.get("usage", {}))
    return row


def run_pair(sample, client, output_root):
    order = CONDITIONS if sample["pair_index"] % 2 == 0 else CONDITIONS[::-1]
    rows = [run_trial(sample, c, client, output_root) for c in order]
    print(sample["sample_id"] + " | " + " | ".join(f"{r['condition']}: {r['choice']} correct={r['correct']} p={r[SCORE]}" for r in rows), flush=True)
    return rows


def comparison_records(payload, baseline_root=BASELINE_ROOT):
    checked_file(baseline_root / "records.json", payload["protocol"]["baseline_records_sha256"])
    samples = {r["sample_id"] for r in payload["records"]}
    baseline = [r for r in json.loads((baseline_root / "records.json").read_text())["records"] if r["sample_id"] in samples]
    records = baseline + payload["records"]
    for model in (BASELINE_MODEL, MODEL):
        pairs = validate_pairs([r for r in records if r["model"] == model])
        if len(pairs) != len(samples):
            raise ValueError("Incomplete paired model comparison.")
    index = {(r["model"], r["sample_id"], r["condition"]): r for r in records}
    for row in payload["records"]:
        old = index[BASELINE_MODEL, row["sample_id"], row["condition"]]
        for key in ("target_description", "ground_truth_object_id", "input_sha256", "image_sha256", "localization_sha256", "candidate_map"):
            if old[key] != row[key]:
                raise ValueError(f"Paired model input mismatch: {key}")
    return records


def report(output_root=OUTPUT_ROOT, baseline_root=BASELINE_ROOT):
    from scripts.analyze_vlm_output_format_ablation import metrics
    from scripts.analyze_vlm_uncertainty_scores import plt, risk_coverage_curve
    from scripts.analyze_vlm_visual_label_ablation import matched_coverage, write_csv

    payload = json.loads((output_root / "records.json").read_text())
    records = comparison_records(payload, baseline_root)
    summary, curves, matched, targets = {}, [], [], []
    lines = ["# GPT-5.2: label-only versus label-then-explanation", "",
             "32 frozen scene-target pairs: states 1–2 across ten LIBERO-Object targets (20 scenes), plus 12 other-object captures. "
             "64 fresh GPT-5.2 calls; comparison reuses only the matching 64 GPT-4.1-mini responses. The first image/logprob probe is included, not an extra trial.", "",
             "| Cohort | Model | Format | Correct | AUROC | AURC | Available | Wrong ≥.99 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: "unavailable" if x is None else f"{x:.6g}"
    for cohort in ("additional_states", "additional_objects", "pooled"):
        for model in (BASELINE_MODEL, MODEL):
            groups = []
            for condition in CONDITIONS:
                rows = sorted((r for r in records if r["model"] == model and r["condition"] == condition
                               and (cohort == "pooled" or r["cohort"] == cohort)), key=lambda r: r["sample_id"])
                groups.append(rows)
                m = metrics(rows)
                summary.setdefault(cohort, {}).setdefault(model, {})[condition] = m
                curves.extend({"cohort": cohort, "model": model, "condition": condition, **p} for p in risk_coverage_curve(rows, SCORE)["points"])
                lines.append(f"| {cohort} | {model} | {condition} | {m['correct']}/{m['n']} | {fmt(m['by_correctness']['correctness_ranking_auc'])} | "
                             f"{fmt(m['aurc'])} | {m['score_availability']:.0%} | {m['incorrect_with_score_ge_099']} |")
            matched.extend({"cohort": cohort, "comparison": model + ": label-only -> explanation", **p} for p in matched_coverage(*groups))
            summary[cohort][model]["transitions"] = {
                "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(*groups)),
                "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(*groups)),
            }
        for condition in CONDITIONS:
            groups = [sorted((r for r in records if r["model"] == m and r["condition"] == condition
                              and (cohort == "pooled" or r["cohort"] == cohort)), key=lambda r: r["sample_id"]) for m in (BASELINE_MODEL, MODEL)]
            matched.extend({"cohort": cohort, "comparison": condition + ": mini -> GPT-5.2", **p} for p in matched_coverage(*groups))
    for target in sorted({r["target_description"] for r in records}):
        for model in (BASELINE_MODEL, MODEL):
            for condition in CONDITIONS:
                rows = [r for r in records if r["target_description"] == target and r["model"] == model and r["condition"] == condition]
                targets.append({"target": target, "model": model, "condition": condition, "n": len(rows), "correct": sum(r["correct"] for r in rows)})
    lines += ["", "## Alphabet soup and tomato sauce", "", "Explanations below are unverified model claims, not ground truth or gate inputs.", ""]
    for r in records:
        if r["model"] == MODEL and r["target_description"] in ("alphabet soup", "tomato sauce"):
            lines += [f"### {r['sample_id']} / {r['condition']}", "",
                      f"Prediction {r['predicted_object_id']}; ground truth {r['ground_truth_object_id']}; correct {r['correct']}; raw likelihood {fmt(r[SCORE])}.", "",
                      "```text", r["generated_output_text"], "```", ""]
    lines += ["## Protocol and limitations", "",
              "Pinned `gpt-5.2-2025-12-11`, reasoning_effort=none (required for temperature/logprobs), temperature=0, image detail=high, "
              "max_completion_tokens=128, logprobs=true, top_logprobs=5. Actual endpoint rejected top_logprobs=20 before generating a response. "
              "Mini retains 20 diagnostic alternatives; GPT-5.2 uses its supported maximum of five. Only the generated token's own logprob is scored. "
              "Existing format-neutral system prompt and both exact user prompts reused. "
              "No image rerendering, new crops, candidate reordering, explanation-first condition, target extraction, or new-model prompt tuning.", "",
              "Raw score = exp(sum of original decision-bearing label-token logprobs). Separable whitespace and explanation tokens are excluded; "
              "mixed tokens are indivisible. Generated label selects the object. Missing/sentinel probabilities and invalid/truncated outputs remain "
              "unavailable and count as deferred, never zero or fabricated confidence. Candidate-normalized probabilities are not gate inputs.", "",
              "Threshold curves and exactly matched attainable coverage are exploratory only; ties are never split. Unavailable scores defer. "
              "No threshold chosen or old threshold reused. AUROC is undefined with no scored errors. AURC uses the existing trapezoidal tied-endpoint "
              "convention; connecting lines between ties are not attainable thresholds. Raw likelihood is not calibrated correctness probability.", "",
              "Selection uses only fixed scene/state indices, not correctness. Calibration scenes were previously inspected; other-object captures "
              "are exploratory, not held-out data. Only two correlated states per original target and one capture per additional object. "
              "Additional objects have 3–6 candidates versus seven in LIBERO-Object, so pooled results are descriptive. "
              "Historical mini calls and fresh GPT-5.2 calls are not interleaved; backend/time effects are not isolated. Requested high image detail "
              "does not guarantee identical internal image preprocessing across model families. No production VLM/HITL/CAD/localization/planner edits.", "",
              "inputs.json freezes images/localization/GT/prompts/settings and source hashes. responses/ preserves full provider token alternatives. "
              "records.json contains new-model trials; comparison.json/associations.csv include the matching mini controls. summary.json includes "
              "correct/incorrect distributions and transitions; per_target.csv and matched_coverage.csv provide target counts and both false-accept denominators.", "",
              "```sh", "python -m scripts.run_vlm_gpt52_pilot probe", "python -m scripts.run_vlm_gpt52_pilot run",
              "python -m scripts.run_vlm_gpt52_pilot report", "```", "",
              "`probe` is one cached image request; `run` resumes missing calls; `report` is offline. No automatic larger experiment.", "",
              "[Official GPT-5.2 logprob compatibility](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2)", "",
              "![Risk–coverage](risk_coverage.png)", ""]
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "comparison.json", {"protocol": payload["protocol"], "records": records})
    save_json(output_root / "error_examples.json", [r for r in records if not r["correct"]])
    for name, rows in (("associations", records), ("threshold_curves", curves), ("matched_coverage", matched), ("per_target", targets)):
        write_csv(output_root / f"{name}.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, cohort in zip(axes, ("additional_states", "additional_objects")):
        for model in (BASELINE_MODEL, MODEL):
            for condition in CONDITIONS:
                points = [r for r in curves if r["cohort"] == cohort and r["model"] == model and r["condition"] == condition]
                ax.plot([p["autonomous_coverage"] for p in points], [p["selective_risk"] for p in points], ".-",
                        label=f"{'mini' if model == BASELINE_MODEL else 'GPT-5.2'} / {'label' if condition == 'label_only' else 'label + explanation'}")
        ax.set(title=cohort.replace("_", " "), xlabel="Autonomous coverage", ylabel="Incorrect / accepted", xlim=(0, 1.02))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.savefig(output_root / "risk_coverage.png", dpi=170)
    plt.close(fig)
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["pooled"], indent=2))
    print(f"Report: {output_root / 'REPORT.md'}")


def run(stage):
    if stage == "report":
        return report()
    if stage not in ("check", "probe", "run"):
        raise ValueError("Use check/probe/run/report; no full-experiment mode.")
    payload = prepare()
    if stage == "check":
        return
    from openai import OpenAI

    with OpenAI(timeout=120, max_retries=3) as client:
        first = run_trial(payload["samples"][0], "label_only", client, OUTPUT_ROOT)
        if first[SCORE] is None:
            raise RuntimeError(f"GPT-5.2 image/logprob probe failed: {first.get('score_unavailable_reason')}; raw response saved; pilot not launched.")
        print(f"Image/logprob probe: {first['model']}, label={first['choice']}, tokens={first['decision_tokens']}, p={first[SCORE]}", flush=True)
        if stage == "probe":
            return
        records = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(run_pair, s, client, OUTPUT_ROOT) for s in payload["samples"]]
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda r: (r["sample_id"], r["condition"]))
    validate_pairs(records)
    save_json(OUTPUT_ROOT / "records.json", {"protocol": payload["protocol"], "records": records})
    report()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
