"""Two-arm extension: 100 additional frozen states + 12 other-object captures.

Only label_only and label_then_evidence run. No state-0 pilot results are pooled,
no validation/test outcomes are read, and no production code is changed.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from scripts.run_vlm_explanation_pilot import (
    MODEL, SCORE, SYSTEM_PROMPT, OUTPUT_ROOT as PILOT_ROOT,
    arguments_for, build_prompt, parse_response,
)
from scripts.run_vlm_output_format_ablation import SCENE_ROOT, digest, paced_completion
from scripts.run_vlm_visual_label_ablation import (
    load_inputs, prompts_and_map, render_direct_sheet, save_json,
)
from vlm_module import load_localized_objects


OUTPUT_ROOT = SCENE_ROOT.parent / "explanation_expansion"
NEW_OBJECT_ROOT = Path("outputs/simulation/experiments/new_objects_verification")
CONDITIONS = ("label_only", "label_then_evidence")
STATES = tuple(range(1, 11))
WORKERS = 4
COHORTS = ("additional_states", "additional_objects")


def select_states(samples):
    rows = sorted((s for s in samples if s["initial_state_index"] in STATES),
                  key=lambda s: (s["initial_state_index"], s["task_index"]))
    expected = [(state, task) for state in STATES for task in range(10)]
    if [(s["initial_state_index"], s["task_index"]) for s in rows] != expected:
        raise ValueError("Expected all 100 state-1..10 / ten-target combinations exactly once.")
    if any(s["partition"] != "calibration" or not s["target_localized"]
           or s["ground_truth_object_id"] is None
           or s["ground_truth_object_id"] not in s["candidate_map"].values() for s in rows):
        raise ValueError("Known-presence expansion requires localized calibration targets.")
    return rows


def select_new_objects(samples):
    ready, excluded = [], []
    for sample in samples:
        if sample["status"] != "ready" or sample["ground_truth_object_id"] is None:
            excluded.append({"sample_id": sample["sample_id"], "target_description": sample["target_description"],
                             "reason": "frozen target localization/evaluation identity unresolved"})
        else:
            ready.append(sample)
    return ready, excluded


def checked_file(path, expected):
    if digest(Path(path).read_bytes()) != expected:
        raise ValueError(f"Frozen input changed: {path}")


def prepare():
    # Preserve the exact previous prompt/parser/API settings; do not read its outcomes.
    prior = json.loads((PILOT_ROOT / "inputs.json").read_text())
    for name, expected in prior["protocol"]["source_sha256"].items():
        checked_file(name, expected)
    for folder in (OUTPUT_ROOT, OUTPUT_ROOT / "inputs", OUTPUT_ROOT / "responses"):
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in select_states(load_inputs()["samples"]):
        gt = SCENE_ROOT / sample["evaluation_ground_truth_path"]
        checked_file(sample["new_image_path"], sample["new_image_sha256"])
        checked_file(sample["localization_path"], sample["localization_sha256"])
        checked_file(gt, sample["ground_truth_sha256"])
        if json.loads(gt.read_text())["ground_truth_object_id"] != sample["ground_truth_object_id"]:
            raise ValueError("Frozen target identity disagrees with manifest.")
        rows.append({**sample, "cohort": "additional_states"})
    manifest_path = NEW_OBJECT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ready, excluded = select_new_objects(manifest["samples"])
    if len(ready) != 12 or len(excluded) != 1:
        raise ValueError("Expected the 12 usable captures and one pre-existing unresolved capture.")
    for sample in ready:
        original, localization = Path(sample["image_path"]), Path(sample["localization_path"])
        for name, expected in sample["files_sha256"].items():
            checked_file(original.parent / name, expected)
        if sample["saved_initial_state_used"] or sample["partition"] != "exploratory_new_objects":
            raise ValueError("Expected frozen fresh-reset new-object captures.")
        gt = original.parent / "evaluation_ground_truth.json"
        if json.loads(gt.read_text())["ground_truth_object_id"] != sample["ground_truth_object_id"]:
            raise ValueError("New-object identity and evaluation manifest disagree.")
        prompts, mapping = prompts_and_map(sample["target_description"], load_localized_objects(localization))
        if sample["ground_truth_object_id"] not in mapping.values() or sample["candidate_count"] != len(mapping) - 1:
            raise ValueError("New target absent from the fixed candidate list.")
        direct = OUTPUT_ROOT / "inputs" / f"{sample['sample_id']}.png"
        audit = render_direct_sheet(original.parent / "rgb.png", localization, original, direct, mapping)
        rows.append({**sample, "cohort": "additional_objects", "initial_state_index": None,
                     "candidate_map": mapping, "prompts": prompts, "visual_audit": audit,
                     "new_image_path": str(direct), "new_image_sha256": digest(direct.read_bytes()),
                     "localization_sha256": digest(localization.read_bytes()),
                     "ground_truth_sha256": digest(gt.read_bytes())})
    for index, sample in enumerate(rows):
        sample["pair_index"] = index
        sample["experiment_prompts"] = {c: build_prompt(sample, c)[0] for c in CONDITIONS}
        sample["input_sha256"] = digest(json.dumps({k: v for k, v in sample.items() if k != "input_sha256"}, sort_keys=True).encode())
    args = arguments_for(b"", "")
    settings = {k: v for k, v in args.items() if k != "messages"}
    if settings != prior["protocol"]["settings"] or SYSTEM_PROMPT != prior["protocol"]["system_prompt"]:
        raise ValueError("API settings differ from the explanation pilot.")
    protocol = {
        "selection": "100 calibration scenes: ten targets x states 1..10; plus all 12 localization-ready new-object captures",
        "conditions": list(CONDITIONS), "fresh_calls": 224, "states": list(STATES),
        "excluded_before_api": excluded, "cohort_sizes": {c: sum(s["cohort"] == c for s in rows) for c in COHORTS},
        "settings": settings, "system_prompt": SYSTEM_PROMPT, "image_detail": "high",
        "none_policy": "known-present, valid candidate letters only; identical within every pair",
        "pair_order": "alternate by fixed pair index; each pair runs sequentially, at most four pairs concurrently",
        "score": "raw generated decision-label likelihood; explanation and alternatives are diagnostic-only",
        "analysis": "separate cohort accuracy/AUC/distributions, paired transitions and exactly matched threshold coverage; pooled descriptive result; no threshold selection or held-out claims",
        "visual_input": "existing direct letters for state scenes; only crop-header labels changed on new-object sheets, with non-label pixel equality verified; both arms receive identical bytes",
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "source_sha256": {**prior["protocol"]["source_sha256"],
                          "scripts/run_vlm_explanation_expansion.py": digest(Path(__file__).read_bytes())},
        "inputs_sha256": digest(json.dumps(rows, sort_keys=True).encode()),
    }
    payload = {"protocol": protocol, "samples": rows}
    path = OUTPUT_ROOT / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Expansion protocol changed; refusing to mix runs.")
    if not path.exists():
        save_json(path, payload)
    print("Verified 112 pairs / 224 calls; explanation-first excluded; one unresolved white-bowl capture excluded.", flush=True)
    return payload


def run_pair(sample, client, output_root):
    image = Path(sample["new_image_path"]).read_bytes()
    if digest(image) != sample["new_image_sha256"]:
        raise ValueError("Frozen visual input changed before request.")
    checked_file(sample["localization_path"], sample["localization_sha256"])
    rows = []
    order = CONDITIONS if sample["pair_index"] % 2 == 0 else CONDITIONS[::-1]
    for condition in order:
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
                     "captured_at": datetime.now(timezone.utc).isoformat(), "provider_response": raw}
            save_json(path, saved)
        raw = saved["provider_response"]
        if raw["model"] != MODEL:
            raise ValueError("Unexpected model snapshot.")
        decision = parse_response(raw, mapping, condition)
        row = {k: sample[k] for k in ("sample_id", "cohort", "partition", "target_description", "ground_truth_object_id",
                                     "initial_state_index", "input_sha256", "localization_sha256")}
        row.update(condition=condition, model=MODEL, provider="openai", **decision, candidate_map=mapping,
                   correct=decision["choice"] is not None and decision["predicted_object_id"] == sample["ground_truth_object_id"],
                   image_sha256=sample["new_image_sha256"], request_sha256=request_hash,
                   response_path=str(path), response_sha256=digest(path.read_bytes()), score_type="raw_label_likelihood")
        rows.append(row)
    return rows


def validate_pairs(records):
    grouped = {}
    for row in records:
        if row["condition"] not in CONDITIONS or (row["sample_id"], row["condition"]) in grouped:
            raise ValueError("Unexpected or duplicate condition.")
        grouped[row["sample_id"], row["condition"]] = row
    pairs = []
    for sample_id in sorted({r["sample_id"] for r in records}):
        if any((sample_id, c) not in grouped for c in CONDITIONS):
            raise ValueError("Incomplete pair.")
        a, b = (grouped[sample_id, c] for c in CONDITIONS)
        for key in ("cohort", "image_sha256", "localization_sha256", "input_sha256", "ground_truth_object_id", "target_description", "model", "candidate_map"):
            if a[key] != b[key]:
                raise ValueError(f"Paired {key} mismatch.")
        pairs.append((a, b))
    return pairs


def report(output_root=OUTPUT_ROOT):
    from scripts.analyze_vlm_output_format_ablation import metrics
    from scripts.analyze_vlm_uncertainty_scores import plt, risk_coverage_curve
    from scripts.analyze_vlm_visual_label_ablation import matched_coverage, write_csv

    payload = json.loads((output_root / "records.json").read_text())
    records = payload["records"]
    pairs = validate_pairs(records)
    summary, curves, matched, transitions, targets = {}, [], [], [], []
    for cohort in (*COHORTS, "pooled"):
        selected = pairs if cohort == "pooled" else [(a, b) for a, b in pairs if a["cohort"] == cohort]
        if not selected:
            continue
        summary[cohort] = {c: metrics([p[i] for p in selected]) for i, c in enumerate(CONDITIONS)}
        summary[cohort]["transitions"] = {
            "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in selected),
            "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in selected),
            "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in selected),
            "both_correct": sum(a["correct"] and b["correct"] for a, b in selected),
        }
        for i, c in enumerate(CONDITIONS):
            curves.extend({"cohort": cohort, "condition": c, **p} for p in risk_coverage_curve([r[i] for r in selected], SCORE)["points"])
        matched.extend({"cohort": cohort, **p} for p in matched_coverage(*zip(*selected)))
    for target in sorted({a["target_description"] for a, _ in pairs}):
        selected = [(a, b) for a, b in pairs if a["target_description"] == target]
        targets.append({"target": target, "cohort": selected[0][0]["cohort"], "n": len(selected),
                        "label_only_correct": sum(a["correct"] for a, b in selected),
                        "label_then_evidence_correct": sum(b["correct"] for a, b in selected),
                        "label_only_wrong_ge_099": sum(not a["correct"] and a[SCORE] is not None and a[SCORE] >= .99 for a, b in selected),
                        "label_then_evidence_wrong_ge_099": sum(not b["correct"] and b[SCORE] is not None and b[SCORE] >= .99 for a, b in selected)})
    for a, b in pairs:
        transitions.append({"sample_id": a["sample_id"], "cohort": a["cohort"], "target": a["target_description"],
                            "ground_truth_object_id": a["ground_truth_object_id"], "label_only_prediction": a["predicted_object_id"],
                            "label_then_evidence_prediction": b["predicted_object_id"], "label_only_correct": a["correct"],
                            "label_then_evidence_correct": b["correct"], "label_only_score": a[SCORE], "label_then_evidence_score": b[SCORE],
                            "explanation": b["explanation"]})
    save_json(output_root / "summary.json", summary)
    for name, rows in (("associations", records), ("paired_results", transitions), ("per_target", targets),
                       ("threshold_curves", curves), ("matched_coverage", matched)):
        if rows:
            write_csv(output_root / f"{name}.csv", rows)
    save_json(output_root / "error_examples.json", [r for r in records if not r["correct"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, cohort in zip(axes, COHORTS):
        for c in CONDITIONS:
            points = [p for p in curves if p["cohort"] == cohort and p["condition"] == c]
            ax.plot([p["autonomous_coverage"] for p in points], [p["selective_risk"] for p in points], ".-", label=c)
        ax.set(title=cohort.replace("_", " "), xlabel="Autonomous coverage", ylabel="Incorrect / accepted", xlim=(0, 1.02))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle("Label-only versus label then explanation | exploratory frozen scenes")
    fig.savefig(output_root / "risk_coverage.png", dpi=170)
    plt.close(fig)
    fmt = lambda x: "unavailable" if x is None else f"{x:.6g}"
    lines = ["# Label-then-explanation: expanded paired experiment", "",
             "112 additional scene-target pairs / 224 fresh calls: 100 states 1–10 across ten LIBERO-Object targets, "
             "and 12 other-object fresh-reset captures. No state-0 pilot results are pooled. Explanation-first "
             "was not run. Results below compare new label-only calls with new label-then-explanation calls.", "",
             "## Results", "",
             "| Cohort | Format | Correct | AUROC | AURC | Mean raw likelihood | Wrong >=.99 | Score availability |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cohort, values in summary.items():
        for c in CONDITIONS:
            m = values[c]
            lines.append(f"| {cohort} | {c} | {m['correct']}/{m['n']} | {fmt(m['by_correctness']['correctness_ranking_auc'])} | "
                         f"{fmt(m['aurc'])} | {fmt(m['likelihood']['mean'] if m['likelihood'] else None)} | "
                         f"{m['incorrect_with_score_ge_099']} | {m['score_availability']:.1%} |")
    for cohort, values in summary.items():
        t = values["transitions"]
        lines += ["", f"{cohort}: wrong→correct {t['wrong_to_correct']}; correct→wrong {t['correct_to_wrong']}; "
                  f"both wrong {t['both_wrong']}; both correct {t['both_correct']}.", ""]
    lines += ["## Every target", "", "| Target | Scenes | Label only correct | Label + explanation correct |", "| --- | ---: | ---: | ---: |"]
    for t in targets:
        lines.append(f"| {t['target']} | {t['n']} | {t['label_only_correct']} | {t['label_then_evidence_correct']} |")
    lines += ["", "## Matched coverage / human deferral", "",
              "Only exactly shared attainable coverage is compared; no optimistic ordering inside ties. "
              "The full table and both false-acceptance-rate denominators are in matched_coverage.csv. "
              "Rows below show the highest shared coverage not exceeding each reference level, not selected deployment thresholds.", "",
              "| Cohort | Requested coverage | Actual coverage | Label-only false accepts | Label + explanation false accepts |", "| --- | ---: | ---: | ---: | ---: |"]
    for cohort in summary:
        for coverage in (.5, .75, .9, 1.):
            eligible = [p for p in matched if p["cohort"] == cohort and p["coverage"] <= coverage]
            if eligible:
                p = max(eligible, key=lambda r: r["coverage"])
                lines.append(f"| {cohort} | {coverage:.0%} | {p['coverage']:.1%} | {p['old_false_accepts']} | {p['new_false_accepts']} |")
            else:
                lines.append(f"| {cohort} | {coverage:.0%} | unavailable | — | — |")
    lines += ["", "## Controls and limitations", "",
              f"Same `{MODEL}`, temperature 0, detail high, shared format-neutral system prompt, "
              "max_completion_tokens=128, top_logprobs=20. Same frozen RGB/localization/candidate metadata/" 
              "order, known-presence assertion, and direct visual letters within each pair. The new-object "
              "sheets only replace crop-header IDs with letters; non-label pixel equality is checked using "
              "the existing renderer. There are 3–6 candidates in the new-object cohort versus seven in "
              "LIBERO-Object, so separate cohort results matter. No objects or poses were recaptured.", "",
              "Score = exp(sum of original generated decision-bearing label-token logprobs), excluding "
              "separable formatting and explanation tokens. Generated choice selects the object, not the "
              "largest top alternative. Missing/sentinel values, invalid formats and unfinished outputs "
              "remain unavailable and defer; no confidence filling or response repair. Raw responses, "
              "explanations, token alternatives and hashes remain saved. Explanations are unverified model "
              "claims, not extra ground-truth evidence or gate inputs.", "",
              "All state scenes are from the existing calibration split; other objects are exploratory "
              "captures, not an unseen external benchmark. One pre-existing white-bowl localization failure "
              "is excluded before API calls, not counted as VLM failure or target absence. Neither scene "
              "selection nor exclusion uses model correctness. No validation/test outcome access or "
              "threshold selection; no claims of calibration. Ten states per category remain correlated; "
              "one call per condition does not quantify API repeatability. Pooled results are descriptive.", "",
              "summary.json includes correct/incorrect likelihood quantiles and paired error transitions. "
              "associations.csv/records.json contain generated text, decision tokens, score, correctness and "
              "provenance. paired_results.csv and per_target.csv expose the semantic failures. AURC follows "
              "the existing trapezoidal tied-endpoint convention; lines between ties are not attainable thresholds.", "",
              "No production VLM/HITL/CAD/localization/planner changes. No explanation-first or Y/N calls, "
              "no candidate-normalized gate. Frozen input/source/request hashes protect resumable checkpoints.", "",
              "```sh", "python -m scripts.run_vlm_explanation_expansion check",
              "python -m scripts.run_vlm_explanation_expansion run",
              "python -m scripts.run_vlm_explanation_expansion report", "```", "",
              "`run` resumes missing calls; `report` is offline. No automatic further expansion.", "",
              "![Risk–coverage](risk_coverage.png)", ""]
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({cohort: {c: {k: values[c][k] for k in ("correct", "n", "score_availability", "by_correctness")}
                               for c in CONDITIONS} for cohort, values in summary.items()}, indent=2))
    print(f"Report: {output_root / 'REPORT.md'}")


def run(stage):
    if stage == "report":
        return report()
    if stage not in ("check", "run"):
        raise ValueError("Use check, run, or report.")
    payload = prepare()
    if stage == "check":
        return
    from openai import OpenAI

    records = []
    with OpenAI(timeout=120, max_retries=3) as client, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_pair, s, client, OUTPUT_ROOT) for s in payload["samples"]]
        for count, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            records.extend(rows)
            print(f"{count}/112 pairs | " + " | ".join(f"{r['sample_id']} {r['condition']}: "
                  f"correct={r['correct']} p={r[SCORE]}" for r in rows), flush=True)
    records.sort(key=lambda r: (r["sample_id"], r["condition"]))
    save_json(OUTPUT_ROOT / "records.json", {"protocol": payload["protocol"], "records": records})
    report()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
