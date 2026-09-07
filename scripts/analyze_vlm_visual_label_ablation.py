"""Offline analysis of direct visual letters; thresholds use development splits only."""

from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vlm_output_format_ablation import metrics, paired_accuracy
from scripts.analyze_vlm_uncertainty_scores import (
    plt, risk_coverage_curve, select_development_operating_point, threshold_metrics,
)
from scripts.run_vlm_visual_label_ablation import (
    CONDITIONS, MODEL, OUTPUT_ROOT, SCORE, load_inputs, parse_response, save_json,
)
from scripts.run_vlm_output_format_ablation import digest


SPLITS = ("calibration", "validation", "test")


def paired_transitions(pairs):
    result = paired_accuracy(pairs)
    for old, new in (("compact_wrong_id_correct", "old_wrong_new_correct"),
                     ("compact_correct_id_wrong", "old_correct_new_wrong"),
                     ("accuracy_difference_id_minus_compact", "accuracy_difference_new_minus_old")):
        result[new] = result.pop(old)
    return result


def matched_coverage(old, new):
    """Compare only exact shared, threshold-attainable coverage; never split ties."""
    if len(old) != len(new):
        raise ValueError("Matched coverage requires equal scene counts.")
    curves = [risk_coverage_curve(rows, SCORE)["points"] for rows in (old, new)]
    by_count = [{p["autonomous_decisions"]: p for p in points} for points in curves]
    rows = []
    for count in sorted(set(by_count[0]) & set(by_count[1])):
        a, b = (points[count] for points in by_count)
        rows.append({"accepted_count": count, "coverage": count / len(old),
                     "human_intervention_rate": 1 - count / len(old),
                     "old_threshold": a["threshold"], "new_threshold": b["threshold"],
                     "old_accuracy": a["autonomous_accuracy"], "new_accuracy": b["autonomous_accuracy"],
                     "old_false_accepts": a["false_autonomous_acceptance_count"],
                     "new_false_accepts": b["false_autonomous_acceptance_count"],
                     "old_false_accept_rate_all": a["false_autonomous_acceptance_rate_all_trials"],
                     "new_false_accept_rate_all": b["false_autonomous_acceptance_rate_all_trials"],
                     "old_false_accept_rate_accepted": a["selective_risk"],
                     "new_false_accept_rate_accepted": b["selective_risk"]})
    return rows


def freeze_development_points(records):
    """This function never reads test outcomes or scores."""
    points = {}
    for condition in CONDITIONS:
        calibration = [r for r in records if r["condition"] == condition and r["partition"] == "calibration"]
        validation = [r for r in records if r["condition"] == condition and r["partition"] == "validation"]
        points[condition] = {str(target): select_development_operating_point(calibration, validation, SCORE, target)
                             for target in (.9, .95)}
    return points


def validate_records(records, samples):
    by_id = {s["sample_id"]: s for s in samples}
    seen, pairs = {}, []
    for row in records:
        sample = by_id[row["sample_id"]]
        condition = row["condition"]
        key = (row["sample_id"], condition)
        if key in seen or condition not in CONDITIONS:
            raise ValueError("Duplicate or unknown condition.")
        seen[key] = row
        for field in ("partition", "task_index", "initial_state_index", "target_description",
                      "ground_truth_object_id", "input_sha256"):
            if row[field] != sample[field]:
                raise ValueError(f"Frozen {field} mismatch.")
        path = Path(row["response_path"])
        if digest(path.read_bytes()) != row["response_sha256"]:
            raise ValueError("Immutable raw response hash changed.")
        response = json.loads(path.read_text())
        raw = response["provider_response"]
        if raw["model"] != MODEL or response["input_sha256"] != sample["input_sha256"]:
            raise ValueError("Model/input mismatch in raw response.")
        parsed = parse_response(raw, sample["candidate_map"])
        if any(row.get(k) != v for k, v in parsed.items()):
            raise ValueError("Saved score/choice differs from actual generated decision tokens.")
        if row["correct"] != (row["choice"] is not None and row["predicted_object_id"] == sample["ground_truth_object_id"]):
            raise ValueError("Incorrect frozen ground-truth comparison.")
    for sample_id in sorted(by_id):
        if any((sample_id, c) not in seen for c in CONDITIONS):
            raise ValueError("Incomplete scene pair.")
        pairs.append(tuple(seen[sample_id, c] for c in CONDITIONS))
    if len(seen) != len(samples) * 2:
        raise ValueError("Unexpected response count.")
    return pairs


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def analyze():
    inputs = load_inputs()
    payload = json.loads((OUTPUT_ROOT / "full_records.json").read_text())
    if payload["protocol"] != inputs["protocol"]:
        raise ValueError("Run protocol differs from input audit.")
    records = payload["records"]
    pairs = validate_records(records, inputs["samples"])
    if len(pairs) != 500:
        raise ValueError("Full analysis requires 500 complete scene pairs.")
    for condition in CONDITIONS:
        counts = Counter(r["partition"] for r in records if r["condition"] == condition)
        if counts != {"calibration": 300, "validation": 100, "test": 100}:
            raise ValueError("Frozen split changed.")
    directory = OUTPUT_ROOT / "analysis"
    directory.mkdir(exist_ok=True)
    # Persist development-selected thresholds before computing any test metrics.
    points = freeze_development_points(records)
    frozen = {"protocol_sha256": digest(json.dumps(inputs["protocol"], sort_keys=True).encode()), "points": points}
    frozen_path = directory / "frozen_operating_points.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Previously frozen operating points changed.")
    if not frozen_path.exists():
        save_json(frozen_path, frozen)
    summary = {"conditions": {}, "paired_transitions": {}, "operating_points": points}
    curves, matched, class_rows, transition_rows = [], [], [], []
    for split in ("all", *SPLITS):
        subset = pairs if split == "all" else [(a, b) for a, b in pairs if a["partition"] == split]
        summary["paired_transitions"][split] = paired_transitions(subset)
        matched.extend({"split": split, **row} for row in matched_coverage(*zip(*subset)))
        for index, condition in enumerate(CONDITIONS):
            rows = [p[index] for p in subset]
            summary["conditions"].setdefault(condition, {})[split] = metrics(rows)
            curves.extend({"condition": condition, "split": split,
                           "human_intervention_rate": 1 - p["autonomous_coverage"], **p}
                          for p in risk_coverage_curve(rows, SCORE)["points"])
    for condition in CONDITIONS:
        test = [r for r in records if r["condition"] == condition and r["partition"] == "test"]
        for point in points[condition].values():
            if point is not None:
                point["test"] = threshold_metrics(test, SCORE, point["threshold"])
    for target in sorted({a["target_description"] for a, b in pairs}):
        for split in ("all", *SPLITS):
            selected = [(a, b) for a, b in pairs if a["target_description"] == target
                        and (split == "all" or a["partition"] == split)]
            transition = paired_transitions(selected)
            for index, condition in enumerate(CONDITIONS):
                rows = [p[index] for p in selected]
                class_rows.append({"target": target, "split": split, "condition": condition, "n": len(rows),
                                   "correct": sum(r["correct"] for r in rows),
                                   "accuracy": sum(r["correct"] for r in rows) / len(rows),
                                   "wrong_ge_099": sum(not r["correct"] and r[SCORE] is not None and r[SCORE] >= .99 for r in rows),
                                   **{k: transition[k] for k in ("old_wrong_new_correct", "old_correct_new_wrong", "both_wrong")}})
    for a, b in pairs:
        transition_rows.append({"sample_id": a["sample_id"], "partition": a["partition"],
                                "target_description": a["target_description"], "ground_truth_object_id": a["ground_truth_object_id"],
                                "old_prediction": a["predicted_object_id"], "new_prediction": b["predicted_object_id"],
                                "old_correct": a["correct"], "new_correct": b["correct"],
                                "transition": f"{'correct' if a['correct'] else 'wrong'}_to_{'correct' if b['correct'] else 'wrong'}",
                                "old_score": a[SCORE], "new_score": b[SCORE]})
    high_wrong = [{**r, "visual_prompt_path": next(s["old_image_path" if r["condition"] == CONDITIONS[0] else "new_image_path"]
                    for s in inputs["samples"] if s["sample_id"] == r["sample_id"])}
                  for r in records if not r["correct"] and r[SCORE] is not None and r[SCORE] >= .99]
    raw = [json.loads(Path(r["response_path"]).read_text()) for r in records]
    times = {r["response_path"]: datetime.fromisoformat(v["captured_at"]) for r, v in zip(records, raw)}
    summary["audit"] = {"pairs": len(pairs), "responses": len(raw),
                        "pixel_invariant_pairs": sum(s["visual_audit"]["unchanged_nonlabel_pixels"] for s in inputs["samples"]),
                        "models": dict(Counter(v["provider_response"]["model"] for v in raw)),
                        "fingerprints": dict(Counter(v["provider_response"].get("system_fingerprint", "unavailable") for v in raw)),
                        "finish_reasons": dict(Counter(v["provider_response"]["choices"][0]["finish_reason"] for v in raw)),
                        "maximum_pair_gap_seconds": max(abs((times[a["response_path"]] - times[b["response_path"]]).total_seconds()) for a, b in pairs),
                        "high_confidence_wrong_counts": {c: {str(t): sum(r["condition"] == c and r[SCORE] >= t for r in high_wrong)
                                                              for t in (.99, .999, .99999, 1.)} for c in CONDITIONS}}
    save_json(directory / "summary.json", summary)
    save_json(directory / "high_confidence_wrong.json", high_wrong)
    for name, rows in (("associations", records), ("paired_transitions", transition_rows),
                       ("per_target", class_rows), ("threshold_curves", curves), ("matched_coverage", matched)):
        write_csv(directory / f"{name}.csv", rows)
    plot_results(records, class_rows, curves, matched, directory)
    write_report(summary, class_rows, matched, directory)
    print(json.dumps({"all": {c: summary["conditions"][c]["all"] for c in CONDITIONS},
                      "transitions": summary["paired_transitions"], "operating_points": points}, indent=2))
    print(f"Report: {directory / 'REPORT.md'}")


def plot_results(records, classes, curves, matched, directory):
    colors = dict(zip(CONDITIONS, ("#0072B2", "#D55E00")))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for index, split in enumerate(("validation", "test")):
        for condition in CONDITIONS:
            rows = [p for p in curves if p["condition"] == condition and p["split"] == split]
            axes[0, index].plot([r["autonomous_coverage"] for r in rows], [r["selective_risk"] for r in rows],
                                ".-", color=colors[condition], label=condition)
        axes[0, index].set(title=f"{split}: risk–coverage (ties intact)", xlabel="Autonomous coverage", ylabel="Incorrect / accepted", xlim=(0, 1.01))
        axes[0, index].set_ylim(bottom=0)
        axes[0, index].legend(fontsize=8)
    for condition in CONDITIONS:
        for correct, style in ((True, "-"), (False, "--")):
            rows = [r for r in records if r["condition"] == condition and r["correct"] == correct and r[SCORE] is not None]
            if not rows:
                continue
            x = -np.log10(np.maximum(1 - np.array([r[SCORE] for r in rows]), 1e-15))
            axes[1, 0].hist(x, bins=np.linspace(0, 15, 31), histtype="step", density=True, color=colors[condition],
                            linestyle=style, label=f"{condition}: {'correct' if correct else 'wrong'}")
    axes[1, 0].set(title="All 500: raw-likelihood distributions", xlabel="−log10(1 − likelihood); exact 1 displayed at 15", ylabel="Density")
    axes[1, 0].legend(fontsize=8)
    points = [r for r in matched if r["split"] == "all"]
    for condition, prefix in zip(CONDITIONS, ("old", "new")):
        axes[1, 1].plot([r["coverage"] for r in points], [r[f"{prefix}_false_accepts"] for r in points],
                        ".-", color=colors[condition], label=condition)
    axes[1, 1].set(title="All 500: exactly matched attainable coverage", xlabel="Autonomous coverage", ylabel="False autonomous acceptances")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Visible object IDs versus direct answer letters | paired frozen LIBERO scenes")
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    targets = sorted({r["target"] for r in classes})
    for index, condition in enumerate(CONDITIONS):
        values = {r["target"]: r["accuracy"] for r in classes if r["condition"] == condition and r["split"] == "all"}
        ax.bar(np.arange(len(targets)) + (index - .5) * .38, [values[t] for t in targets], .38,
               color=colors[condition], label=condition)
    ax.set(xticks=np.arange(len(targets)), xticklabels=targets, ylabel="Association accuracy", ylim=(0, 1.05),
           title="50 frozen scenes per target; no robot execution")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend()
    fig.savefig(directory / "per_target.png", dpi=170)
    plt.close(fig)


def write_report(summary, classes, matched, directory):
    def number(value):
        return "undefined" if value is None else f"{value:.4f}"

    a, b = (summary["conditions"][c] for c in CONDITIONS)
    transition = summary["paired_transitions"]["all"]
    old_auc = a["all"]["by_correctness"]["correctness_ranking_auc"]
    new_auc = b["all"]["by_correctness"]["correctness_ranking_auc"]
    lines = ["# Paired direct-visual-label ablation", "", "## Conclusion", "",
             f"Direct letters changed pooled accuracy from {a['all']['accuracy']:.1%} to {b['all']['accuracy']:.1%} "
             f"({transition['accuracy_difference_new_minus_old'] * 100:+.1f} percentage points; "
             f"{transition['old_wrong_new_correct']} errors corrected and {transition['old_correct_new_wrong']} introduced). "
             f"The test split changed from {a['test']['correct']}/100 to {b['test']['correct']}/100. "
             "The pooled paired statistics below describe this fixed ten-target scene collection, not generalization to new categories.", "",
             f"Correctness-ranking AUROC changed from {number(old_auc)} to {number(new_auc)}; "
             f"AURC changed from {a['all']['aurc']:.4f} to {b['all']['aurc']:.4f} (lower is better). "
             f"Wrong predictions with provider-reported likelihood exactly 1 changed from "
             f"{a['all']['incorrect_with_score_equal_one']} to {b['all']['incorrect_with_score_equal_one']}. "
             "Both formats used one decision token in every real trial, so these differences are not a sequence-length penalty.", ""]
    if old_auc is not None and new_auc is not None and new_auc < old_auc:
        lines += ["Direct labeling improves some semantic selections but does **not** establish a better "
                  "raw-likelihood deferral gate here: score discrimination is worse. At high pooled coverage "
                  "the lower total error count can still produce fewer false acceptances; this is not uniform "
                  "dominance across coverage or evidence of better calibrated scores. The unchanged test "
                  "accuracy and test matched-coverage results below do not support replacing the production "
                  "gate with this condition. Keep this as an accuracy ablation, not a validated confidence upgrade.", ""]
    lines += ["## Controlled comparison", "",
             "500 frozen LIBERO-Object scenes, 1,000 fresh paired responses. OLD displays object_001…object_007 "
             "on the existing crop headers and supplies the letter-to-ID map in the prompt. NEW displays A…G "
             "on those same headers; persistent IDs remain local and are removed from NEW candidate metadata. "
             "Both return exactly one A…G or N. This tests direct visual-to-answer labeling, not output-token length.", "",
             "Every non-header pixel is identical: frozen 768×768 source RGB, localization, crop order/padding/scale, "
             "full-scene tile, and 1792×896 contact-sheet layout are unchanged. Existing rendering code is reused. "
             "The prompt sentences and geometric metadata are unchanged except the ID mapping representation. "
             "GPT-4.1-mini-2025-04-14, temperature 0, detail high, max_completion_tokens=8, logprobs=True, top_logprobs=20.", "",
             "Score = exp(sum of generated decision-bearing token logprobs). Separable surrounding formatting "
             "is excluded; no length or candidate normalization. The generated choice determines the prediction; "
             "top alternatives are archived only. Missing/sentinel likelihoods defer. No Y/N verification, "
             "conformal prediction, production VLM/HITL/CAD/localization/planning changes.", "",
             "Accuracy compares generated object IDs with frozen ground truth. AUROC and risk–coverage use "
             "each condition's own correctness outcomes: changed predictions are part of this representation "
             "ablation, not silently replaced by a common prediction rule.", "",
             "## Accuracy and score discrimination", "",
             "| Condition | Split | Correct | Accuracy | AUROC | AURC | Score availability |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        for split in ("all", *SPLITS):
            m = summary["conditions"][c][split]
            lines.append(f"| {c} | {split} | {m['correct']}/{m['n']} | {m['accuracy']:.4f} | {number(m['by_correctness']['correctness_ranking_auc'])} | {number(m['aurc'])} | {m['score_availability']:.1%} |")
    p = summary["paired_transitions"]["all"]
    lines += ["", f"OLD→NEW: wrong→correct {p['old_wrong_new_correct']}; correct→wrong {p['old_correct_new_wrong']}; "
              f"both correct {p['both_correct']}; both wrong {p['both_wrong']}. "
              f"Accuracy difference NEW−OLD {p['accuracy_difference_new_minus_old']:.4f}; paired scene-bootstrap "
              f"95% interval {p['paired_bootstrap_95_interval']}; exact McNemar p={p['mcnemar_exact_two_sided_p']:.6g}.", "",
              "## Per-target errors", "",
              "| Target | OLD correct / 50 | NEW correct / 50 | Wrong→correct | Correct→wrong | OLD wrong ≥.99 | NEW wrong ≥.99 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for target in sorted({r["target"] for r in classes}):
        a, b = [next(r for r in classes if r["target"] == target and r["split"] == "all" and r["condition"] == c) for c in CONDITIONS]
        lines.append(f"| {target} | {a['correct']} | {b['correct']} | {a['old_wrong_new_correct']} | {a['old_correct_new_wrong']} | {a['wrong_ge_099']} | {b['wrong_ge_099']} |")
    lines += ["", "## Raw likelihood remains uncalibrated", "",
              "| Condition | Correct median | Incorrect median | Wrong ≥.99 | Wrong ≥.99999 | Wrong exactly 1 | Mean decision tokens |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        m = summary["conditions"][c]["all"]
        d = m["by_correctness"]
        high = summary["audit"]["high_confidence_wrong_counts"][c]
        lines.append(f"| {c} | {d['correct']['median'] if d['correct'] else None} | {d['incorrect']['median'] if d['incorrect'] else None} | {high['0.99']} | {high['0.99999']} | {high['1.0']} | {m['mean_decision_token_count']:.3f} |")
    lines += ["", "## Matched coverage / human intervention", "",
              "Exact common attainable coverages only; no tie splitting, missing-score filling, or best-case ordering. "
              "The rows below show the largest shared coverage not above each requested level. The full exact "
              "table is in matched_coverage.csv. Test curves and matched-coverage tables are descriptive, not threshold selection.", "",
              "| Split | Requested coverage | Actual coverage | Human intervention | OLD false accepts | NEW false accepts | OLD accuracy | NEW accuracy |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("all", "validation", "test"):
        for coverage in (.5, .6, .7, .8, .9, 1.):
            eligible = [r for r in matched if r["split"] == split and r["coverage"] <= coverage]
            if not eligible:
                lines.append(f"| {split} | {coverage:.0%} | unavailable | — | — | — | — | — |")
                continue
            r = max(eligible, key=lambda r: r["coverage"])
            lines.append(f"| {split} | {coverage:.0%} | {r['coverage']:.1%} | {r['human_intervention_rate']:.1%} | {r['old_false_accepts']} | {r['new_false_accepts']} | {r['old_accuracy']:.4f} | {r['new_accuracy']:.4f} |")
    lines += ["", "## Development-selected thresholds, frozen before test evaluation", "",
              "300 calibration / 100 validation / 100 test, using the original split. Select each condition's "
              "threshold independently for highest validation coverage meeting ≥90% or ≥95% accuracy in both "
              "development splits. No old threshold is reused and no threshold is chosen from test outcomes.", "",
              "| Condition | Accuracy target | Frozen threshold | Validation accuracy / coverage | Test accuracy / coverage | Test false accepts |",
              "| --- | ---: | ---: | --- | --- | ---: |"]
    for c in CONDITIONS:
        for goal, point in summary["operating_points"][c].items():
            if point is None:
                lines.append(f"| {c} | {goal} | unavailable | — | — | — |")
            else:
                v, t = point["validation"], point["test"]
                lines.append(f"| {c} | {goal} | {point['threshold']!r} | {number(v['autonomous_accuracy'])} / {number(v['autonomous_coverage'])} | {number(t['autonomous_accuracy'])} / {number(t['autonomous_coverage'])} | {t['false_autonomous_acceptance_count']} |")
    lines += ["", "## Limitations and artifacts", "",
              "Raw token likelihood is not a calibrated probability of correct identity. Repeated states share "
              "ten object categories; scene-bootstrap/McNemar calculations do not establish transfer to unseen "
              "categories. The split has appeared in prior investigations, so this is not a pristine external "
              "holdout. No absent-target scenes are present. One response per condition does not quantify API "
              "repeatability; fixed temperature and snapshot do not eliminate server variation. AURC integrates "
              "tied-threshold endpoints trapezoidally from the empty origin; within-tie interpolation is a plotting "
              "convention, not an attainable threshold.", "",
              f"Execution audit: {json.dumps(summary['audit'], separators=(',', ':'))}", "",
              "inputs.json freezes image/localization/ground-truth hashes, prompts, local mapping, source hashes, "
              "settings and the analysis plan. responses/ holds immutable raw API/token records. full_records.json "
              "and associations.csv are compact. paired_transitions.csv, per_target.csv, threshold_curves.csv, "
              "matched_coverage.csv and high_confidence_wrong.json preserve all requested comparisons.", "",
              "```sh", "source /home/jiabao/.venvs/lerobot-libero/bin/activate",
              "python -m scripts.run_vlm_visual_label_ablation check",
              "python -m scripts.run_vlm_visual_label_ablation full",
              "python -m scripts.run_vlm_visual_label_ablation analyze", "```", "",
              "Full resumes only missing responses; analyze is offline. No production change or automatic follow-up experiment.", "",
              "![Comparison](comparison.png)", "", "![Per target](per_target.png)", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    analyze()
