"""Offline joint-prompt pilot analysis. No API calls or chosen deployment gate."""

from collections import Counter
import json
import math
from pathlib import Path

from scripts.run_vlm_multi_object_pilot import (
    MODELS, OUTPUT_ROOT, TARGET_COUNTS, digest, evaluate_response, parse_response, save_json,
)


THRESHOLDS = (.5, .7, .8, .9, .95, .99, .999, .9999, .99999, 1.0)


def semantic_evaluation(row, ground_truth):
    """Ignore only a leading definite article, not identity-changing synonyms.

    The prompt explicitly allows object descriptions copied from 'the milk'.
    Apply equally to every response; never change token spans or likelihoods.
    Original literal-name records remain frozen for auditability.
    """
    canonical = lambda name: name.removeprefix("the ")
    parsed = {**row, "entries": [{**e, "target_description": canonical(e["target_description"])} for e in row["entries"]]}
    truth = {canonical(k): v for k, v in ground_truth.items()}
    return {**row, **evaluate_response(parsed, truth), "literal_name_correct": row["correct"],
            "literal_name_associations_correct": sum(a["correct"] for a in row["associations"])}


def validate_records(records):
    keys = [(r["trial_id"], r["model"]) for r in records]
    if len(keys) != len(set(keys)) or any(r["model"] not in MODELS or r["partition"] != "calibration" for r in records):
        raise ValueError("Duplicate, unknown-model or non-calibration record.")
    for row in records:
        if len(row["associations"]) != row["target_count"]:
            raise ValueError("Missing evaluation rows for required targets.")
        expected = row["format_valid"] and not row["extra_targets"] and all(a["correct"] for a in row["associations"])
        if row["correct"] != expected:
            raise ValueError("Joint correctness must include every required association.")
        for lp, score in [(row["full_raw_log_probability"], row["full_response_likelihood"]),
                          *[(e["raw_log_probability"], e["raw_association_likelihood"]) for e in row["entries"]]]:
            if (lp is None) != (score is None) or (lp is not None and score != math.exp(lp)):
                raise ValueError("Raw likelihood/logprob mismatch.")
        if row["joint_gate_score"] != (row["full_response_likelihood"] if row["format_valid"] else None):
            raise ValueError("Gate score differs from the predeclared full-output likelihood.")
    for trial in {r["trial_id"] for r in records}:
        pair = [r for r in records if r["trial_id"] == trial]
        if len(pair) != len(MODELS):
            raise ValueError("Incomplete paired model comparison.")
        if any(pair[0][k] != pair[1][k] for k in ("instruction", "input_sha256", "candidate_map")):
            raise ValueError("Paired request inputs differ.")


def report(output_root=OUTPUT_ROOT):
    from scripts.analyze_vlm_uncertainty_scores import plt, risk_coverage_curve, score_distribution, threshold_metrics
    from scripts.analyze_vlm_visual_label_ablation import write_csv

    payload = json.loads((output_root / "records.json").read_text())
    original = payload["records"]
    validate_records(original)
    inputs = json.loads((output_root / "inputs.json").read_text())
    if inputs["protocol"] != payload["protocol"]:
        raise ValueError("Report and inference protocol mismatch.")
    samples = {s["trial_id"]: s for s in inputs["samples"]}
    if {(r["trial_id"], r["model"]) for r in original} != {(s, m) for s in samples for m in MODELS}:
        raise ValueError("Missing pilot checkpoints; cannot report partial data as complete.")
    records, provider_issues = [], []
    for row in original:
        raw_path = Path(row["response_path"])
        if digest(raw_path.read_bytes()) != row["response_sha256"]:
            raise ValueError("Raw checkpoint changed.")
        saved = json.loads(raw_path.read_text())
        if saved["request_sha256"] != row["request_sha256"] or saved["input_sha256"] != row["input_sha256"]:
            raise ValueError("Checkpoint provenance mismatch.")
        raw = saved["provider_response"]
        parsed = parse_response(raw, row["candidate_map"], row["instruction"])
        if any(parsed[k] != row[k] for k in parsed):
            raise ValueError("Frozen parsed record no longer matches the raw provider response.")
        for token in (raw["choices"][0].get("logprobs") or {}).get("content", []):
            if token.get("logprob") is not None and token["logprob"] > 0:
                provider_issues.append({"trial_id": row["trial_id"], "model": row["model"], "token": token["token"], "logprob": token["logprob"]})
        records.append(semantic_evaluation(row, samples[row["trial_id"]]["evaluation_ground_truth"]))
    validate_records(records)
    save_json(output_root / "semantic_records.json", {"evaluation_policy": "case/whitespace + optional leading 'the'; all original token likelihoods unchanged",
                                                     "frozen_records_sha256": digest((output_root / "records.json").read_bytes()), "records": records})
    summary, curves, thresholds, associations, per_target, matched = [], [], [], [], [], []
    for row in records:
        associations.extend({"trial_id": row["trial_id"], "sample_id": row["sample_id"], "model": row["model"],
                             "target_count": row["target_count"], **a} for a in row["associations"])
    for model in MODELS:
        for count in TARGET_COUNTS:
            rows = [r for r in records if r["model"] == model and r["target_count"] == count]
            if not rows:
                raise ValueError("Missing model/target-count group.")
            pairs = [a for a in associations if a["model"] == model and a["target_count"] == count]
            curve = risk_coverage_curve(rows, "joint_gate_score")
            mean = lambda field: sum(r["usage"].get(field, 0) for r in rows) / len(rows)
            summary.append({"model": model, "target_count": count, "requests": len(rows),
                            "joint_correct": sum(r["correct"] for r in rows),
                            "joint_accuracy": sum(r["correct"] for r in rows) / len(rows),
                            "associations": len(pairs), "associations_correct": sum(a["correct"] for a in pairs),
                            "association_accuracy": sum(a["correct"] for a in pairs) / len(pairs),
                            "full_score_available": sum(r["full_response_likelihood"] is not None for r in rows),
                            "gate_score_available": sum(r["joint_gate_score"] is not None for r in rows),
                            "entry_score_available": sum(a["raw_association_likelihood"] is not None for a in pairs),
                            "format_invalid": sum(not r["format_valid"] for r in rows),
                            "literal_name_joint_correct": sum(r["literal_name_correct"] for r in rows),
                            "literal_name_associations_correct": sum(r["literal_name_associations_correct"] for r in rows),
                            "full_scene_label_responses": sum("full scene:" in r["generated_output_text"].casefold() for r in rows),
                            "missing_targets": sum(len(r["missing_targets"]) for r in rows),
                            "extra_targets": sum(len(r["extra_targets"]) for r in rows),
                            "full_likelihood_distribution": score_distribution(rows, "full_response_likelihood"),
                            "entry_likelihood_distribution": score_distribution(pairs, "raw_association_likelihood"),
                            "risk_coverage": {k: v for k, v in curve.items() if k != "points"},
                            "wrong_ge_099": sum(not r["correct"] and r["joint_gate_score"] is not None and r["joint_gate_score"] >= .99 for r in rows),
                            "mean_prompt_tokens": mean("prompt_tokens"), "mean_completion_tokens": mean("completion_tokens"),
                            "mean_output_tokens": sum(r["output_token_count"] for r in rows) / len(rows)})
            curves.extend({"model": model, "target_count": count, **p} for p in curve["points"])
            thresholds.extend({"model": model, "target_count": count, **threshold_metrics(rows, "joint_gate_score", t)} for t in THRESHOLDS)
            for target in sorted({a["target_description"] for a in pairs}):
                group = [a for a in pairs if a["target_description"] == target]
                per_target.append({"model": model, "target_count": count, "target": target, "n": len(group),
                                   "correct": sum(a["correct"] for a in group)})
    # Only exact attainable nonzero coverages; never split equal-score ties.
    for count in TARGET_COUNTS:
        indexed = [{p["autonomous_decisions"]: p for p in curves if p["model"] == model and p["target_count"] == count} for model in MODELS]
        for accepted in sorted(set(indexed[0]) & set(indexed[1])):
            a, b = (d[accepted] for d in indexed)
            matched.append({"target_count": count, "accepted_requests": accepted, "coverage": a["autonomous_coverage"],
                            "mini_threshold": a["threshold"], "gpt52_threshold": b["threshold"],
                            "mini_false_accepts": a["false_autonomous_acceptance_count"],
                            "gpt52_false_accepts": b["false_autonomous_acceptance_count"]})
    errors = [r for r in records if not r["correct"]]
    unavailable_reasons = Counter(r["score_unavailable_reason"] for r in records if r["full_response_likelihood"] is None)
    save_json(output_root / "summary.json", {"groups": summary, "unavailable_reasons": dict(unavailable_reasons),
                                           "invalid_positive_logprobs": provider_issues,
                                           "analysis_source_sha256": digest(Path(__file__).read_bytes())})
    save_json(output_root / "error_examples.json", errors)
    for name, rows in (("requests", records), ("associations", associations), ("thresholds", thresholds),
                       ("risk_coverage", curves), ("per_target", per_target), ("matched_coverage", matched)):
        write_csv(output_root / f"{name}.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, count in zip(axes, TARGET_COUNTS):
        for model in MODELS:
            points = [p for p in curves if p["model"] == model and p["target_count"] == count]
            available = next(s["gate_score_available"] for s in summary if s["model"] == model and s["target_count"] == count)
            ax.plot([p["autonomous_coverage"] for p in points], [p["selective_risk"] for p in points], ".-",
                    label=f"{model.split('-2025')[0]} ({available}/10 gate scores)")
        ax.set(title=f"{count} objects per request", xlabel="Autonomous request coverage", ylabel="Incorrect joint responses / accepted", xlim=(0, 1.02), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(output_root / "risk_coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i, model in enumerate(MODELS):
        for j, count in enumerate(TARGET_COUNTS):
            ax = axes[i, j]
            group = [r for r in records if r["model"] == model and r["target_count"] == count]
            available = sum(r["full_raw_log_probability"] is not None for r in group)
            for correct, marker, label in ((True, "o", "all associations correct"), (False, "x", "incorrect/incomplete")):
                rows = [r for r in group if r["correct"] == correct and r["full_raw_log_probability"] is not None]
                if rows:
                    ax.scatter([r["output_token_count"] for r in rows], [r["full_raw_log_probability"] for r in rows], marker=marker, label=label)
            ax.set(title=f"{model.split('-2025')[0]} / {count} objects ({available}/{len(group)} scored)",
                   xlabel="Returned text-token logprob entries", ylabel="Full raw log probability")
            if available:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "No usable full-response scores\nInvalid positive token logprobs", ha="center", va="center", transform=ax.transAxes)
                ax.set(xticks=[], yticks=[])
    fig.savefig(output_root / "logprob_vs_length.png", dpi=160)
    plt.close(fig)
    fmt = lambda x: "unavailable" if x is None else f"{x:.6g}"
    lines = ["# Joint multi-object semantic-association pilot", "",
             "10 frozen LIBERO-Object calibration scenes (state 1, one per original target); two- and three-object instructions per scene. "
             "40 API responses / 100 required associations across two previously tested snapshots. No new images, localization, simulation execution or human correction.", "",
             "| Model | Objects/request | Correct associations | Entire response correct | Full scores available | Correctness AUC | Wrong joint score >=.99 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['target_count']} | {s['associations_correct']}/{s['associations']} | {s['joint_correct']}/{s['requests']} | "
                     f"{s['full_score_available']}/{s['requests']} | {fmt(s['full_likelihood_distribution']['correctness_ranking_auc'])} | {s['wrong_ge_099']} |")
    quantitative = []
    for model in MODELS:
        rows = [r for r in records if r["model"] == model]
        pairs = [a for a in associations if a["model"] == model]
        scores = [r["full_response_likelihood"] for r in rows if r["full_response_likelihood"] is not None]
        span = f"{min(scores):.6g}–{max(scores):.6g}" if scores else "unavailable"
        errors_by_target = dict(Counter(a["target_description"] for a in pairs if not a["correct"]))
        quantitative.append(f"{model}: {sum(a['correct'] for a in pairs)}/{len(pairs)} required associations correct; "
                            f"{sum(r['correct'] for r in rows)}/{len(rows)} entire responses correct; "
                            f"{sum('full scene:' in r['generated_output_text'].casefold() for r in rows)} responses with `full scene`; "
                            f"{len(scores)}/{len(rows)} full likelihoods available, range {span}. "
                            f"Incorrect or missing required associations by target: {json.dumps(errors_by_target, sort_keys=True)}.")
    lines += ["", "## Main findings", "",
              "The requested joint response runs in one call with no explanations. However, the first prompt frequently lets GPT-4.1-mini "
              "treat the contact sheet's `full scene` header as an answer label. Those are genuine output-contract failures, not object IDs; "
              "we retain them and do not retrospectively accept that label. Valid partial candidate-letter entries are scored separately.", "",
              "GPT-5.2 has a different limitation: invalid positive token logprobs make many full-response likelihoods unavailable. "
              "The raw response and invalid_positive_logprobs in summary.json identify every affected token. No values are clamped. "
              "This is insufficient evidence to endorse a joint likelihood threshold. Do not interpret a broader numerical range as better error detection.", "",
              *quantitative, "", "These are descriptive, correlated pilot counts, not evidence of general model superiority or improvement over the prior prompt. "
              "Any apparently perfect AUROC on only two available responses is not meaningful evidence of a reliable gate; consult availability alongside ranking.", "",
              "## Scores and exploratory gate", "",
              "Full-response likelihood = exp(sum of ALL original generated output-token logprobs), including punctuation/whitespace. "
              "The hypothetical joint gate uses this score only when the response format is valid. Entry likelihood = exp(sum of original "
              "tokens overlapping that entry's candidate letter OR object description)); separable punctuation/whitespace is excluded. Mixed tokens are indivisible. "
              "No length normalization, candidate normalization, clamping, self-reported confidence, or probability fabricated for missing/sentinel/positive logprobs. "
              "Full means every returned visible-text token logprob, not unreturned termination/control tokens. API completion-token usage and returned logprob-entry counts "
              "are both preserved; they need not match, particularly for GPT-5.2.", "",
              "Ground-truth completeness is an evaluation outcome, NOT an input to the gate. An omitted required target can therefore be a confidently wrong "
              "joint response; those failures are retained. Unknown IDs, invalid format and unavailable scores defer. No actual HITL sessions occurred.", "",
              "Each threshold in thresholds.csv is applied offline, separately by model and object count. Risk–coverage retains ties; line segments between "
              "tied endpoints do not imply attainable thresholds. AUROC is unavailable without both scored correct and incorrect examples. "
              "No threshold was selected, no old threshold reused, and no test or validation outcomes were read.", "",
              "## Exact request", "", "System message contains only the role specification and the original task instruction. "
              "The user message contains the unchanged marked PNG at high image detail. No pre-extracted targets, target order, numerical geometry, "
              "candidate identity map, explanations, or model-to-model answers are sent.", "", "```text", payload['protocol']['role'],
              "", "Task instruction:", "Put the tomato sauce on top of the milk.", "```", "",
              "Expected syntax is one existing visual letter plus an instruction object phrase per line. Output order is not fixed. "
              "N is allowed, but every requested target in this pilot is present: false-absence predictions count as errors and true-absence behavior is not tested.", "",
              "## Systematic errors / examples", ""]
    for row in records:
        if any(a["target_description"] in ("alphabet soup", "tomato sauce") and not a["correct"] for a in row["associations"]):
            lines += [f"### {row['trial_id']} / {row['model']}", "", row["instruction"], "", "```text", row["generated_output_text"], "```", "",
                      f"Full likelihood: {fmt(row['full_response_likelihood'])}.", "",
                      "; ".join(f"{a['target_description']}: predicted {a['predicted_object_id']}, expected {a['ground_truth_object_id']}, entry likelihood {fmt(a['raw_association_likelihood'])}" for a in row["associations"]), ""]
    lines += ["## Interpretation and limits", "",
              "This is a joint-prompt feasibility/score pilot, not a controlled estimate of improvement over the old single-target prompt. "
              "It also changes target extraction, N policy (relative to the prior known-present explanation runs), output naming and removes geometric metadata. "
              "No fresh separate-target control calls were made. Both new models receive identical image/prompt inputs; GPT-5.2 uses reasoning_effort=none "
              "and five diagnostic alternatives, versus twenty for mini, matching previously verified endpoint limits. Temperature=0 and output budget=128 for both.", "",
              "One shared scene supports both instructions; the two/three-target groups and repeated targets are correlated, not independent scene samples. "
              "Extra targets follow a fixed presence-filtered priority, favoring milk/tomato sauce/alphabet soup. Per-target counts are not balanced. "
              "Audit identities were obtained by the earlier pixel-exact simulation replay and mutual-nearest centroid matching, not manual per-point purity labels. "
              "Only the selected ten calibration audit files are opened. Identity maps never reach the model.", "",
              "Full likelihood depends on output length, syntax, copied object names and autoregressive output order. A low value is not automatically evidence of "
              "semantic error; a high value is not calibrated correctness probability. logprob_vs_length.png separates output-length effects descriptively, not causally. "
              "Evaluation correction: the initial literal-name matcher counted `the milk` versus `milk` as different targets even though the prompt permits copying "
              "object phrases from the instruction. Offline semantic evaluation now ignores only a leading `the`, uniformly across all responses. "
              "This affects correctness matching only: no token spans or scores change and no API calls are repeated. records.json retains the original literal results; "
              "semantic_records.json and the report use the corrected evaluation. summary.json also preserves original strict counts. "
              "Omissions/extras are otherwise scored strictly; case and whitespace normalize, but synonyms and identifying modifiers are not silently rewritten. "
              "Object labels are compared to the frozen candidate IDs independently of likelihood.", "",
              "requests.csv includes actual token usage; associations.csv has every required target including omissions. Raw provider responses/alternatives are "
              "immutable checkpoints in responses/. inputs.json freezes task instructions, images, localization, evaluation labels and source/settings hashes. "
              "Production VLM/HITL/final_object_id/CAD/planner behavior is unchanged. Task instructions are hypothetical perception tests, not feasibility-checked actions.", "",
              "```sh", "python -m scripts.run_vlm_multi_object_pilot check", "python -m scripts.run_vlm_multi_object_pilot probe",
              "python -m scripts.run_vlm_multi_object_pilot run", "python -m scripts.run_vlm_multi_object_pilot report", "```", "",
              "Probe makes one request per model and is included in run; run resumes missing requests only; report is offline. No automatic larger experiment.", "",
              "[OpenAI output-token logprob reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)", "",
              "![Risk–coverage](risk_coverage.png)", "", "![Raw log probability versus output length](logprob_vs_length.png)", ""]
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output_root / 'REPORT.md'}")


if __name__ == "__main__":
    report()
