"""Offline, exploratory analysis of the 50 calibration-scene verification pilot.

Never calls a model or opens the held-out set. The two-threshold envelope uses
these pilot outcomes and is deliberately labeled optimistic, not validated.
"""

import csv
import json
import math
from pathlib import Path

from scripts.analyze_vlm_uncertainty_scores import (
    plt, score_distribution, threshold_metrics,
)
from scripts.run_vlm_candidate_verification_pilot import (
    OUTPUT_ROOT, scene_prediction, verification_gate, verification_score,
)
from scripts.run_vlm_output_format_ablation import digest, extract_decision


ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"
# Descriptive references, fixed before inspecting the pilot's results.
REFERENCE_SUPPORT = .9
REFERENCE_GAP = .1
GAP_SLICES = (0, .05, .1, .2, .3, .5)


def validate_records(payload, expected_count=50, fixed_calibration_states=True):
    records = payload["records"]
    inputs = payload["protocol"]["sample_inputs"]
    if len(records) != expected_count or {r["scene"]["sample_id"] for r in records} != set(inputs):
        raise ValueError("Expected the complete pilot, not a partial result.")
    for record in records:
        scene = record["scene"]
        if fixed_calibration_states and (scene["partition"] != "calibration"
                                         or scene["initial_state_index"] not in range(5)):
            raise ValueError("Analysis only permits calibration states 0--4.")
        if scene["input_sha256"] != inputs[scene["sample_id"]]:
            raise ValueError("Scene input hash differs from protocol.")
        if set(record["candidates"]) != {r["object_id"] for r in scene["candidate_metadata"]}:
            raise ValueError("Candidate set differs from frozen localization.")
        for call_id, saved in [("baseline", record["baseline"]), *record["candidates"].items()]:
            response = json.loads(Path(saved["response_path"]).read_text())
            if (response["input_sha256"] != scene["input_sha256"]
                    or response["settings_sha256"] != saved["settings_sha256"]
                    or response["settings_sha256"] != digest(json.dumps(response["settings"], sort_keys=True).encode())):
                raise ValueError("Response input/settings hash mismatch.")
            raw = response["provider_response"]
            if raw["model"] != payload["protocol"]["model"]:
                raise ValueError("Model snapshot mismatch.")
            if call_id != "baseline":
                parsed = verification_score(raw)
                if any(saved.get(k) != v for k, v in parsed.items()):
                    raise ValueError("Saved verification score differs from raw response.")
            else:
                output = raw["choices"][0]
                decision = extract_decision(output["message"].get("content") or "",
                    (output.get("logprobs") or {}).get("content") or [], scene["candidate_map"])
                score = decision["raw_sequence_likelihood"] if output["finish_reason"] == "stop" else None
                if (saved["raw_sequence_likelihood"] != score
                        or saved["predicted_object_id"] != decision["predicted_object_id"]):
                    raise ValueError("Saved baseline differs from raw response.")
                expected_correct = decision["choice"] is not None and decision["predicted_object_id"] == scene["ground_truth_object_id"]
                if saved["correct"] != expected_correct:
                    raise ValueError("Baseline correctness mismatch.")
        predicted = scene_prediction({key: value["yes_score"] for key, value in record["candidates"].items()})
        if any(record["verification"].get(k) != v for k, v in predicted.items()):
            raise ValueError("Saved scene prediction differs from candidate scores.")
        expected_correct = predicted["predicted_object_id"] is not None and predicted["predicted_object_id"] == scene["ground_truth_object_id"]
        if record["verification"]["correct"] != expected_correct:
            raise ValueError("Verification correctness mismatch.")
    return records


def gate_metrics(records, method, support, gap=0):
    accepted = []
    metric_rows = []
    for r in records:
        if method == "baseline":
            score = r["baseline"]["raw_sequence_likelihood"]
            take = score is not None and score >= support
            available = score is not None
            correct = r["baseline"]["correct"]
        else:
            available = r["verification"]["score_available"]
            take = verification_gate(r["verification"], support, gap) == "vlm_accepted"
            correct = r["verification"]["correct"]
        accepted.append(take)
        metric_rows.append({"correct": correct, "gate": float(take) if available else None})
    # Reuse established count/rate definitions; unavailable scores remain deferred.
    metrics = threshold_metrics(metric_rows, "gate", .5)
    del metrics["threshold"]
    return {"method": method, "tau_support": support, "tau_gap": gap, **metrics,
            "accepted_sample_ids": [r["scene"]["sample_id"] for r, take in zip(records, accepted) if take],
            "baseline_errors_deferred": sum(not r["baseline"]["correct"] and not take for r, take in zip(records, accepted)),
            "own_errors_deferred": sum(not row["correct"] and not take for row, take in zip(metric_rows, accepted))}


def sweep(records):
    baseline = sorted({0., *(r["baseline"]["raw_sequence_likelihood"] for r in records
                             if r["baseline"]["raw_sequence_likelihood"] is not None)})
    support = sorted({0., *(r["verification"]["support"] for r in records
                            if r["verification"]["support"] is not None)})
    gaps = sorted({*GAP_SLICES, *(r["verification"]["gap"] for r in records
                                if r["verification"]["gap"] is not None)})
    return ([gate_metrics(records, "baseline", t) for t in baseline]
            + [gate_metrics(records, "support", t) for t in support]
            + [gate_metrics(records, "support_gap", t, g) for t in support for g in gaps])


def matched_coverage(points):
    """Exact accepted counts only; never split score ties or fill missing scores.

    The two-dimensional frontier chooses the lowest observed risk for each
    count ON THIS PILOT. It is an optimistic exploration, not generalization.
    """
    frontiers = {}
    for method in ("baseline", "support", "support_gap"):
        by_count = {}
        for p in points:
            if p["method"] != method or not p["autonomous_decisions"]:
                continue
            count = p["autonomous_decisions"]
            key = (p["false_autonomous_acceptance_count"], p["tau_gap"], p["tau_support"])
            previous = by_count.get(count)
            if previous is None or key < (previous["false_autonomous_acceptance_count"], previous["tau_gap"], previous["tau_support"]):
                by_count[count] = p
        frontiers[method] = by_count
    counts = sorted(set.intersection(*(set(f) for f in frontiers.values())))
    return frontiers, [{"accepted_count": count, "methods": {m: f[count] for m, f in frontiers.items()}}
                       for count in counts]


def summary(records):
    n = len(records)
    baseline = [{"score": r["baseline"]["raw_sequence_likelihood"], "correct": r["baseline"]["correct"]} for r in records]
    verification = [{"score": r["verification"]["support"], "correct": r["verification"]["correct"]} for r in records]
    return {
        "scenes": n, "candidate_calls": sum(len(r["candidates"]) for r in records),
        "baseline_correct": sum(r["correct"] for r in baseline),
        "verification_correct": sum(r["correct"] for r in verification),
        "baseline_accuracy_all_scenes": sum(r["correct"] for r in baseline) / n,
        "verification_accuracy_all_scenes": sum(r["correct"] for r in verification) / n,
        "baseline_score_available": sum(r["score"] is not None for r in baseline),
        "verification_complete_scenes": sum(r["verification"]["score_available"] for r in records),
        "verification_top_ties": sum(r["verification"]["top_tie"] for r in records),
        "candidate_scores_available": sum(c["yes_score"] is not None for r in records for c in r["candidates"].values()),
        "answer_counts": {label: sum(c["answer_label"] == label for r in records for c in r["candidates"].values()) for label in ("Y", "N", None)},
        "single_answer_token_calls": sum(len(c.get("decision_tokens", [])) == 1 for r in records for c in r["candidates"].values()),
        "answer_token_spellings": sorted({json.dumps(c.get("answer_token_spellings", {}), sort_keys=True)
                                          for r in records for c in r["candidates"].values()}),
        "multiple_candidates_above_half": sum(sum(c["yes_score"] is not None and c["yes_score"] >= .5
                                                  for c in r["candidates"].values()) > 1 for r in records),
        "complete_scenes_with_support_below_half": sum(r["verification"]["support"] is not None
                                                       and r["verification"]["support"] < .5 for r in records),
        "baseline_errors_corrected": sum(not r["baseline"]["correct"] and r["verification"]["correct"] for r in records),
        "baseline_correct_lost": sum(r["baseline"]["correct"] and not r["verification"]["correct"] for r in records),
        "baseline_distribution": score_distribution(baseline, "score"),
        "support_distribution": score_distribution(verification, "score"),
        "reference_gates": [gate_metrics(records, "baseline", REFERENCE_SUPPORT),
                            gate_metrics(records, "support", REFERENCE_SUPPORT),
                            gate_metrics(records, "support_gap", REFERENCE_SUPPORT, REFERENCE_GAP)],
        "classes": {target: {
            "scenes": len(rows), "baseline_correct": sum(r["baseline"]["correct"] for r in rows),
            "verification_correct": sum(r["verification"]["correct"] for r in rows),
            "complete_scenes": sum(r["verification"]["score_available"] for r in rows),
            "reference_gates": [gate_metrics(rows, "baseline", REFERENCE_SUPPORT),
                                gate_metrics(rows, "support_gap", REFERENCE_SUPPORT, REFERENCE_GAP)],
        } for target in sorted({r["scene"]["target_description"] for r in records})
          for rows in [[r for r in records if r["scene"]["target_description"] == target]]},
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def plot_results(records, frontiers, points, output_root=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    names = {"baseline": "Baseline raw likelihood", "support": "Verification support",
             "support_gap": "Support + gap (optimistic pilot envelope)"}
    for method, frontier in frontiers.items():
        p = [frontier[n] for n in sorted(frontier)]
        axes[0].plot([v["autonomous_coverage"] for v in p], [v["autonomous_accuracy"] for v in p],
                     ".-", label=names[method])
    fixed = sorted([p for p in points if p["method"] == "support_gap" and p["tau_gap"] == REFERENCE_GAP],
                   key=lambda p: p["autonomous_coverage"])
    axes[0].plot([p["autonomous_coverage"] for p in fixed], [p["autonomous_accuracy"] for p in fixed],
                 "--", color="gray", label="Support + fixed gap 0.1")
    axes[0].set(xlabel=f"Autonomous coverage (all {len(records)} scored-input scenes)", ylabel="Autonomous accuracy", ylim=(0, 1.03))
    axes[0].legend(fontsize=7)
    for correct, color, label in ((True, "tab:blue", "Correct"), (False, "tab:red", "Incorrect/unresolved")):
        rows = [r for r in records if r["verification"]["correct"] == correct and r["verification"]["score_available"]]
        # Transformation is display-only; gates always use original YES scores.
        odds = [math.log10(max(r["verification"]["support"], 1e-16)
                          / max(1 - r["verification"]["support"], 1e-16)) for r in rows]
        axes[1].scatter(odds, [r["verification"]["gap"] for r in rows],
                        c=color, label=label, alpha=.7)
    axes[1].axhline(REFERENCE_GAP, color="gray", ls=":")
    axes[1].axvline(math.log10(REFERENCE_SUPPORT / (1 - REFERENCE_SUPPORT)), color="gray", ls=":")
    axes[1].set_yscale("symlog", linthresh=1e-10)
    axes[1].set_ylim(0, 1.5)
    axes[1].set(xlabel="log10 YES odds (display only; not calibrated)", ylabel="Top-two YES gap (symlog scale)")
    axes[1].legend(fontsize=8)
    for key, name in (("baseline", "Baseline"), ("verification", "Verification")):
        field = "raw_sequence_likelihood" if key == "baseline" else "support"
        for correct, symbol in ((True, "o"), (False, "x")):
            values = sorted(r[key][field] for r in records if r[key][field] is not None and r[key]["correct"] == correct)
            axes[2].plot(values, [(i + 1) / len(values) for i in range(len(values))], marker=symbol,
                         ms=3, label=f"{name}: {'correct' if correct else 'wrong/unresolved'}")
    axes[2].set(xlabel="Raw baseline likelihood / YES support", ylabel="Within-group empirical CDF")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{len(records)} pilot scenes — exploratory, no held-out evaluation")
    fig.savefig((ANALYSIS_ROOT if output_root is None else Path(output_root)) / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    payload = json.loads((OUTPUT_ROOT / "records.json").read_text())
    records = validate_records(payload)
    points = sweep(records)
    frontiers, matches = matched_coverage(points)
    stats = summary(records)
    stats["matched_coverage"] = matches
    stats["exploratory_operating_points"] = {
        method: {str(accuracy): max(
            (p for p in points if p["method"] == method and p["autonomous_decisions"]
             and p["autonomous_accuracy"] >= accuracy),
            key=lambda p: (p["autonomous_decisions"], -p["tau_gap"], -p["tau_support"]), default=None)
                 for accuracy in (1., .95, .9)} for method in frontiers}
    stats["interpretation"] = "Exploratory calibration-pilot results; two-threshold envelope optimizes observed pilot risk, not held-out performance."
    ANALYSIS_ROOT.mkdir(exist_ok=True)
    (ANALYSIS_ROOT / "summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False))
    write_csv(ANALYSIS_ROOT / "threshold_sweep.csv", points)
    write_csv(ANALYSIS_ROOT / "matched_coverage.csv", [p for row in matches for p in row["methods"].values()])
    scene_rows, candidate_rows = [], []
    for r in records:
        scene = r["scene"]
        common = {k: scene[k] for k in ("sample_id", "target_description", "ground_truth_object_id", "input_sha256")}
        scene_rows.append({**common, "baseline_prediction": r["baseline"]["predicted_object_id"],
            "baseline_correct": r["baseline"]["correct"], "baseline_likelihood": r["baseline"]["raw_sequence_likelihood"],
            **r["verification"], "candidate_yes_scores": {k: c["yes_score"] for k, c in r["candidates"].items()}})
        candidate_rows.extend({**common, "object_id": key, **candidate} for key, candidate in r["candidates"].items())
    write_csv(ANALYSIS_ROOT / "scene_predictions.csv", scene_rows)
    write_csv(ANALYSIS_ROOT / "candidate_scores.csv", candidate_rows)
    plot_results(records, frontiers, points)
    lines = ["# Candidate-wise semantic verification pilot", "",
        "50 frozen LIBERO calibration scenes: initial states 0–4 for every target. "
        "GPT-4.1-mini-2025-04-14, temperature 0, high-detail identical marked images and metadata. "
        "50 fresh baselines + 350 independent Y/N calls. Production and test scenes untouched.", "",
        "## Prediction and availability", "",
        f"- Baseline: {stats['baseline_correct']}/50 correct; {stats['baseline_score_available']}/50 scored.",
        f"- Verification: {stats['verification_correct']}/50 correct; {stats['verification_complete_scenes']}/50 fully scored; "
        f"{stats['verification_top_ties']} exact top-score ties. Missing/tied predictions are unresolved, not target absent.",
        f"- Candidate-level score availability: {stats['candidate_scores_available']}/350. "
        f"Single answer-token calls: {stats['single_answer_token_calls']}/350; generated Y/N counts: {stats['answer_counts']}.",
        f"- Baseline errors corrected: {stats['baseline_errors_corrected']}; baseline correct decisions lost: {stats['baseline_correct_lost']}.",
        f"- Scenes supporting multiple candidates at YES >=0.5: {stats['multiple_candidates_above_half']}; "
        f"fully scored scenes with all YES <0.5: {stats['complete_scenes_with_support_below_half']}.",
        "", "## Error detection at exactly matched coverage", "",
        "All attainable matched counts are in matched_coverage.csv. Never split ties. "
        "The support+gap column is the best observed two-threshold operating point at each count: "
        "an optimistic same-pilot envelope, NOT an independently validated gate. "
        "The plot also shows a fixed gap=0.1 slice chosen before inspecting responses.", "",
        "| Accepted / 50 | Baseline accuracy; false accepts | Support accuracy; false accepts | Support+gap accuracy; false accepts | Dual thresholds (support, gap) |",
        "| --- | --- | --- | --- | --- |"]
    selected = sorted({min((r['accepted_count'] for r in matches), key=lambda n: abs(n - wanted))
                       for wanted in (25, 35, 40, 45, 50)}) if matches else []
    for row in matches:
        if row["accepted_count"] not in selected:
            continue
        cells = [f"{row['accepted_count']}/50"]
        for method in ("baseline", "support", "support_gap"):
            p = row["methods"][method]
            cells.append(f"{p['autonomous_accuracy']:.1%}; {p['false_autonomous_acceptance_count']}")
        p = row["methods"]["support_gap"]
        cells.append(f"({p['tau_support']:.12g}, {p['tau_gap']:.12g})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "False-acceptance CSV fields explicitly report incorrect accepts / all scenes and incorrect accepts / accepted scenes. "
              "The sweep also records how many of the SAME baseline errors each gate defers, separating error detection from changed predictions.",
              "", "### Fixed illustrative gates (not coverage-matched)", "",
              "| Method | Thresholds | Accepted / 50 | Accuracy | False accepts | Baseline errors deferred |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in stats["reference_gates"]:
        accuracy = f"{p['autonomous_accuracy']:.1%}" if p['autonomous_accuracy'] is not None else "undefined"
        lines.append(f"| {p['method']} | {p['tau_support']}, {p['tau_gap']} | {p['autonomous_decisions']}/50 | "
                     f"{accuracy} | {p['false_autonomous_acceptance_count']} | {p['baseline_errors_deferred']} |")
    if matches:
        # Descriptive point near 80% coverage; no interpolation/quota selection.
        comparison = min(matches, key=lambda p: abs(p["accepted_count"] - .8 * len(records)))
        b, v, d = (comparison["methods"][key] for key in ("baseline", "support", "support_gap"))
        lines += ["", "### What this pilot establishes", "",
            f"At the exactly shared coverage nearest 80% ({comparison['accepted_count']}/50), false accepts are "
            f"{b['false_autonomous_acceptance_count']} baseline, {v['false_autonomous_acceptance_count']} support-only, "
            f"and {d['false_autonomous_acceptance_count']} support+gap. On the SAME baseline-error set, these gates "
            f"defer {b['baseline_errors_deferred']}, {v['baseline_errors_deferred']}, and {d['baseline_errors_deferred']} errors respectively. "
            "These are the relevant error-detection comparisons; a wider score range alone establishes nothing. "
            "Even an improvement at an optimized pilot point requires independent validation, especially for the two-parameter gate."]
    lines += ["", "Best observed coverage at minimum accuracy (same-pilot exploration):", "",
              "| Method | 100% accuracy | >=95% accuracy | >=90% accuracy |",
              "| --- | --- | --- | --- |"]
    for method, accuracy_points in stats["exploratory_operating_points"].items():
        cells = [f"{p['autonomous_decisions']}/50" if p is not None else "No nonempty set"
                 for p in accuracy_points.values()]
        lines.append("| " + " | ".join([method, *cells]) + " |")
    lines += [
              "", "## Alphabet soup and tomato sauce", "",
              "| Target / state | GT | Baseline prediction (raw) | Verification prediction | YES support | Gap | YES at GT |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        s, b, v = r["scene"], r["baseline"], r["verification"]
        if s["target_description"] not in ("alphabet soup", "tomato sauce"):
            continue
        lines.append(f"| {s['target_description']} / {s['initial_state_index']} | {s['ground_truth_object_id']} | "
                     f"{b['predicted_object_id']} ({b['raw_sequence_likelihood']}) | {v['predicted_object_id']} | "
                     f"{v['support']} | {v['gap']} | {r['candidates'][s['ground_truth_object_id']]['yes_score']} |")
    for target in ("alphabet soup", "tomato sauce", "chocolate pudding"):
        c = stats["classes"][target]
        low_support = sum(r["scene"]["target_description"] == target and r["verification"]["support"] is not None
                          and r["verification"]["support"] < REFERENCE_SUPPORT for r in records)
        lines += ["", f"{target}: baseline {c['baseline_correct']}/{c['scenes']} correct; verification "
                  f"{c['verification_correct']}/{c['scenes']} correct. {low_support}/{c['scenes']} verification "
                  f"scenes have support below {REFERENCE_SUPPORT}."]
    wrong = [r for r in records if not r["verification"]["correct"] and r["verification"]["score_available"]]
    if wrong:
        r = max(wrong, key=lambda r: r["verification"]["support"])
        lines += ["", f"Highest-support verification error: {r['scene']['sample_id']}, "
                  f"support={r['verification']['support']}, gap={r['verification']['gap']}. "
                  f"The baseline was {'correct' if r['baseline']['correct'] else 'incorrect'} on this scene."]
    support_gate, dual_gate = stats["reference_gates"][1:]
    removed_errors = support_gate["false_autonomous_acceptance_count"] - dual_gate["false_autonomous_acceptance_count"]
    removed_correct = support_gate["autonomous_decisions"] - dual_gate["autonomous_decisions"] - removed_errors
    lines += ["", f"At fixed illustrative support={REFERENCE_SUPPORT}, adding gap={REFERENCE_GAP} removes "
              f"{removed_errors} false accepts AND {removed_correct} correct accepts. "
              "This directly quantifies ambiguity detection and its cost, rather than relying on score spread.", "",
              "**Conclusion:** inspect both matched-coverage false accepts and the corrected/lost predictions above. "
              "An optimized zero-error subset alone does not establish a better general gate; compare the "
              ">=90%/95% coverage trade-offs too. Keep this evaluation-only: these exploratory data are "
              "not sufficient to justify replacing production selection."]
    lines += ["", "## Scoring and limits", "",
        "YES support per candidate is exp(lY − logsumexp(lY,lN)), even when N was generated. "
        "The Y/N tokens must be alternatives at the same answer position with matching whitespace affixes. "
        "Missing, sentinel, invalid or incompatible tokens produce null; any null defers the whole scene. "
        "No logit bias, cross-candidate normalization, or calibrated-correctness claim.", "",
        "Independent binary verification can support several objects strongly; max support alone may hide this. "
        "Assess gap by observed false acceptances, not numeric spread. These 50 scenes contain no truly absent target; "
        "low support is insufficient evidence, not proof of absence. Five correlated scenes per target and "
        "same-pilot threshold optimization prevent held-out conclusions. No full study was launched.", "",
        "Raw provider responses (including all returned token alternatives), settings hashes and frozen input hashes "
        "are under ../responses/, ../scenes/, and ../protocol.json. Both commands are resumable/offline respectively:", "",
        "```sh", "python -m scripts.run_vlm_candidate_verification_pilot run",
        "python -m scripts.analyze_vlm_candidate_verification_pilot", "```", "",
        "![Exploratory comparison](comparison.png)", ""]
    (ANALYSIS_ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("classes", "matched_coverage")}, indent=2))
    print(f"Report: {ANALYSIS_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
