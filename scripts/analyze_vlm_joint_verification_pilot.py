"""Offline paired joint-verification analysis; never chooses an answer-order winner."""

import json
import math
from pathlib import Path

from scripts.analyze_vlm_candidate_verification_pilot import gate_metrics, matched_coverage, sweep, write_csv
from scripts.analyze_vlm_uncertainty_scores import plt, score_distribution
from scripts.run_vlm_candidate_verification_pilot import save_json, scene_prediction, verification_gate
from scripts.run_vlm_joint_verification_pilot import (
    CONDITIONS, OUTPUT_ROOT, REFERENCE_ROOT, answer_order, load_reference, parse_joint_response,
)
from scripts.run_vlm_output_format_ablation import digest


def load_groups():
    reference = load_reference()
    reference_by_id = {r["scene"]["sample_id"]: r for r in reference}
    payload = json.loads((OUTPUT_ROOT / "records.json").read_text())
    protocol = payload["protocol"]
    if protocol != json.loads((OUTPUT_ROOT / "protocol.json").read_text()):
        raise ValueError("Protocol file mismatch.")
    if protocol["reference_sha256"] != digest((REFERENCE_ROOT / "records.json").read_bytes()):
        raise ValueError("Frozen independent reference changed.")
    expected = {(key, c) for key in reference_by_id for c in CONDITIONS}
    actual = [(r["scene"]["sample_id"], r["condition"]) for r in payload["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected every scene in both locked answer orders.")
    groups = {"independent": reference, "joint_forward": [], "joint_reverse": []}
    for r in payload["records"]:
        scene = r["scene"]
        reference_row = reference_by_id[scene["sample_id"]]
        if scene != reference_row["scene"]:
            raise ValueError("Joint/reference input differs.")
        response = json.loads(Path(r["response_path"]).read_text())
        raw = response["provider_response"]
        if (raw["model"] != protocol["model"] or response["input_sha256"] != scene["input_sha256"]
                or response["settings_sha256"] != r["settings_sha256"]
                or response["settings_sha256"] != digest(json.dumps(response["settings"], sort_keys=True).encode())):
            raise ValueError("Raw response settings/input hash mismatch.")
        parsed = parse_joint_response(raw, answer_order(scene, r["condition"]))
        if any(r.get(key) != value for key, value in parsed.items()):
            raise ValueError("Saved parsed score differs from actual provider response.")
        predicted = scene_prediction({key: c["yes_score"] for key, c in parsed["candidates"].items()})
        predicted["correct"] = predicted["predicted_object_id"] == scene["ground_truth_object_id"]
        if predicted != r["verification"]:
            raise ValueError("Saved prediction/correctness differs from computed result.")
        groups[f"joint_{r['condition']}"].append({**r, "baseline": reference_row["baseline"]})
    for group in groups.values():
        group.sort(key=lambda r: r["scene"]["sample_id"])
    return groups


def condition_summary(records, baseline=False):
    predictions = [r["baseline"] if baseline else r["verification"] for r in records]
    scores = [p["raw_sequence_likelihood"] if baseline else p["support"] for p in predictions]
    unresolved = sum(p["choice"] is None if baseline else p["predicted_object_id"] is None for p in predictions)
    correct = sum(p["correct"] for p in predictions)
    n = len(predictions)
    result = {"scenes": n, "correct": correct, "wrong_selections": n - correct - unresolved,
              "unresolved": unresolved, "correct_fraction_all_scenes": correct / n,
              "accuracy_resolved": correct / (n - unresolved) if n > unresolved else None,
              "score_available": sum(s is not None for s in scores),
              "score_distribution": score_distribution([{"score": s, "correct": p["correct"]}
                                                       for s, p in zip(scores, predictions)], "score")}
    if not baseline:
        result.update(candidate_count=sum(len(r["candidates"]) for r in records),
                      candidate_scores_available=sum(c["yes_score"] is not None for r in records for c in r["candidates"].values()),
                      multiple_yes_answers=sum(sum(c["answer_label"] == "Y" for c in r["candidates"].values()) > 1 for r in records),
                      all_no_answers=sum(all(c["answer_label"] == "N" for c in r["candidates"].values()) for r in records),
                      top_ties=sum(p["top_tie"] for p in predictions))
    return result


def order_sensitivity(forward, reverse):
    by_id = {r["scene"]["sample_id"]: r for r in reverse}
    rows = []
    for f in forward:
        b = by_id[f["scene"]["sample_id"]]
        deltas = [abs(c["yes_score"] - b["candidates"][key]["yes_score"])
                  for key, c in f["candidates"].items()
                  if c["yes_score"] is not None and b["candidates"][key]["yes_score"] is not None]
        rows.append({"sample_id": f["scene"]["sample_id"], "target_description": f["scene"]["target_description"],
                     "forward_prediction": f["verification"]["predicted_object_id"],
                     "reverse_prediction": b["verification"]["predicted_object_id"],
                     "prediction_changed": f["verification"]["predicted_object_id"] != b["verification"]["predicted_object_id"],
                     "availability_changed": f["verification"]["score_available"] != b["verification"]["score_available"],
                     "candidate_answer_changes": sum(c["answer_label"] != b["candidates"][k]["answer_label"]
                                                      for k, c in f["candidates"].items()),
                     "scorable_candidate_pairs": len(deltas),
                     "mean_abs_yes_change": math.fsum(deltas) / len(deltas) if deltas else None,
                     "max_abs_yes_change": max(deltas, default=None),
                     "forward_support": f["verification"]["support"], "reverse_support": b["verification"]["support"],
                     "forward_gap": f["verification"]["gap"], "reverse_gap": b["verification"]["gap"],
                     "forward_gate": verification_gate(f["verification"], .9, .1),
                     "reverse_gate": verification_gate(b["verification"], .9, .1)})
    return rows


def analyze():
    groups = load_groups()
    directory = OUTPUT_ROOT / "analysis"
    directory.mkdir(exist_ok=True)
    stats = {"baseline": condition_summary(groups["independent"], baseline=True),
             **{key: condition_summary(rows) for key, rows in groups.items()}}
    all_points, frontiers, fixed_gates = [], {}, []
    fixed_gates.append({**gate_metrics(groups["independent"], "baseline", .9), "method": "baseline"})
    for group, records in groups.items():
        points = sweep(records)
        group_frontiers, _ = matched_coverage(points)
        for method in ("baseline", "support", "support_gap"):
            if method == "baseline" and group != "independent":
                continue
            name = "baseline" if method == "baseline" else f"{group}_{method}"
            frontiers[name] = {n: {**p, "method": name} for n, p in group_frontiers[method].items()}
            all_points.extend({**p, "method": name} for p in points if p["method"] == method)
        fixed_gates.extend({**gate_metrics(records, method, .9, gap), "method": f"{group}_{method}"}
                           for method, gap in (("support", 0), ("support_gap", .1)))
    # Exact attainable coverage only; no interpolation, tie splitting, or
    # filling missing scores. Two-threshold frontiers are optimistic pilot fits.
    matched = [{"accepted_count": n, "methods": {key: f.get(n) for key, f in frontiers.items()}}
               for n in sorted(set.union(*(set(f) for f in frontiers.values())))]
    order_rows = order_sensitivity(groups["joint_forward"], groups["joint_reverse"])
    order_stats = {"prediction_changes": sum(r["prediction_changed"] for r in order_rows),
                   "availability_changes": sum(r["availability_changed"] for r in order_rows),
                   "candidate_answer_changes": sum(r["candidate_answer_changes"] for r in order_rows),
                   "fixed_gate_changes": sum(r["forward_gate"] != r["reverse_gate"] for r in order_rows)}
    save_json(directory / "summary.json", {"conditions": stats, "fixed_gates": fixed_gates,
                                           "order_sensitivity": order_stats, "matched_coverage": matched})
    write_csv(directory / "threshold_sweep.csv", all_points)
    write_csv(directory / "matched_coverage.csv", [p for r in matched for p in r["methods"].values() if p is not None])
    write_csv(directory / "order_sensitivity.csv", order_rows)
    scene_rows, candidate_rows = [], []
    for group, records in groups.items():
        for r in records:
            s = r["scene"]
            common = {"condition": group, **{k: s[k] for k in ("sample_id", "target_description", "ground_truth_object_id")}}
            scene_rows.append({**common, **r["verification"],
                               "baseline_prediction": r["baseline"]["predicted_object_id"],
                               "baseline_correct": r["baseline"]["correct"],
                               "baseline_raw_likelihood": r["baseline"]["raw_sequence_likelihood"]})
            candidate_rows.extend({**common, "object_id": k, **c} for k, c in r["candidates"].items())
    write_csv(directory / "scene_predictions.csv", scene_rows)
    write_csv(directory / "candidate_scores.csv", candidate_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"baseline": "black", "independent": "tab:blue", "joint_forward": "tab:orange", "joint_reverse": "tab:green"}
    for name in ("baseline", "independent_support_gap", "joint_forward_support_gap", "joint_reverse_support_gap"):
        group = name.removesuffix("_support_gap")
        points = sorted([p for p in all_points if p["method"] == name and
                         (name == "baseline" or p["tau_gap"] == .1) and p["autonomous_decisions"]],
                        key=lambda p: p["autonomous_coverage"])
        axes[0].plot([p["autonomous_coverage"] for p in points], [p["autonomous_accuracy"] for p in points],
                     ".-", color=colors[group], label=name.replace("_support_gap", " (gap=0.1)"))
    axes[0].set(xlabel="Autonomous coverage (all 12 scenes)", ylabel="Autonomous accuracy", ylim=(0, 1.05))
    axes[0].legend(fontsize=7)
    for group, records in groups.items():
        for correct, marker in ((True, "o"), (False, "x")):
            rows = [r for r in records if r["verification"]["score_available"] and r["verification"]["correct"] == correct]
            x = [math.log10(max(r["verification"]["support"], 1e-16) / max(1 - r["verification"]["support"], 1e-16)) for r in rows]
            axes[1].scatter(x, [r["verification"]["gap"] for r in rows], c=colors[group], marker=marker,
                            label=f"{group}: {'correct' if correct else 'wrong/tied'}", alpha=.75)
    axes[1].axhline(.1, ls=":", color="gray")
    axes[1].axvline(math.log10(9), ls=":", color="gray")
    axes[1].set_yscale("symlog", linthresh=1e-7)
    axes[1].set(xlabel="log10 YES odds (display only, not calibrated)", ylabel="Support gap", ylim=(0, 1.5))
    axes[1].legend(fontsize=7)
    fig.suptitle("Joint verification — frozen 12-scene pilot; reverse order is a sensitivity control")
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)

    lines = ["# Joint versus independent semantic verification", "",
             "Twelve frozen usable new-object scenes, no new capture or saved-state access. "
             "The white bowl remains excluded for its pre-existing localization failure. "
             "Baseline and independent Y/N responses are reused unchanged; 24 fresh joint calls "
             "cover forward (primary) and reverse (sensitivity control) answer order.", "",
             "## Fundamental difference", "",
             "Both methods already see the full marked image and all candidate metadata. "
             "Independent verification starts a fresh completion for each candidate. Joint verification "
             "requests all judgments in one completion, so each later answer conditions on earlier generated "
             "answers and labels. It may encourage consistency or introduce order-dependent judgments; "
             "a single request does not make per-candidate probabilities independent, directly comparable or calibrated.", "",
             "Each score remains exp(lY-logsumexp(lY,lN)) at its own actual Y/N answer position. "
             "Always score YES, even when N is generated. Whitespace, object-ID and colon tokens do not contribute. "
             "Require a separable Y/N token and matching alternatives at the same position; missing/sentinel "
             "probabilities are null. Any missing candidate score or exact top tie defers the scene. "
             "The API fields were verified in installed openai 3.7.0 and "
             "[official documentation](https://developers.openai.com/api/reference/resources/chat).", "",
             "GPT-4.1-mini-2025-04-14; temperature 0; high-detail identical images; top_logprobs=20. "
             "Only prompt/output protocol and max_completion_tokens (8 to 128 for K lines) change. "
             "Reverse changes only requested answer order, not metadata/image order, IDs or ground truth. "
             "No self-reported confidence, logit bias, cross-candidate normalization, or production gate changes.", "",
             "## Selection before the threshold gate", "",
             "Unresolved is not an incorrect selected object and is not proof of target absence.", "",
             "| Method | Correct | Wrong selections | Unresolved | Complete scene scores |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key, s in stats.items():
        lines.append(f"| {key} | {s['correct']} | {s['wrong_selections']} | {s['unresolved']} | {s['score_available']}/12 |")
    lines += ["", "## Illustrative fixed gates (not calibrated)", "",
              "Raw baseline threshold=0.9; verification support=0.9, optionally gap=0.1. "
              "Equal numeric thresholds do not imply equal confidence semantics. Use matched coverage below.", "",
              "| Gate | Accepted / 12 | Accuracy | False accepts |",
              "| --- | ---: | ---: | ---: |"]
    for p in fixed_gates:
        accuracy = f"{p['autonomous_accuracy']:.1%}" if p["autonomous_accuracy"] is not None else "undefined"
        lines.append(f"| {p['method']} | {p['autonomous_decisions']}/12 | {accuracy} | {p['false_autonomous_acceptance_count']} |")
    lines += ["", "## Matched-coverage false accepts", "",
              "Two-threshold columns use the best observed pilot envelope (optimistic, not held-out). "
              "A dash means that exact coverage is unattainable for the method; no filling or interpolation.", "",
              "| Accepted / 12 | Baseline | Independent support+gap | Joint forward support+gap | Joint reverse support+gap |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in matched:
        cells = [f"{row['accepted_count']}/12"]
        for key in ("baseline", "independent_support_gap", "joint_forward_support_gap", "joint_reverse_support_gap"):
            point = row["methods"][key]
            cells.append(str(point["false_autonomous_acceptance_count"]) if point else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Answer-order sensitivity", "",
              f"Prediction/status changes: {order_stats['prediction_changes']}/12; "
              f"score-availability changes: {order_stats['availability_changes']}/12; "
              f"candidate Y/N answer changes: {order_stats['candidate_answer_changes']}/52; "
              f"fixed support+gap gate changes: {order_stats['fixed_gate_changes']}/12.", "",
              "The control is descriptive: one response per order cannot separate systematic order bias "
              "from API nondeterminism or the changed order instruction. Neither order is selected after seeing outcomes.", "",
              "| Target | Forward ID | Reverse ID | Forward support / gap | Reverse support / gap | Largest candidate score change |",
              "| --- | --- | --- | --- | --- | ---: |"]
    for row in order_rows:
        lines.append(f"| {row['target_description']} | {row['forward_prediction']} | {row['reverse_prediction']} | "
                     f"{row['forward_support']} / {row['forward_gap']} | {row['reverse_support']} / {row['reverse_gap']} | {row['max_abs_yes_change']} |")
    lines += ["", "## Observed changes, not calibration evidence", "",
              "The forward condition is the primary result; reverse is only a sensitivity check. "
              "In this pilot joint verification selected all twelve targets correctly in both orders. "
              "Independent verification had no wrong resolved selections to correct, but deferred four "
              "scenes at support=0.9 and gap=0.1. Joint verification accepted those four correctly. "
              "Thus the observed benefit is coverage and selection consistency, not demonstrated "
              "detection of more errors.", "",
              "The baseline's sole wrong selection was cookies for ramekin; its raw likelihood "
              "was already below 0.9, so the baseline gate already deferred that error. Independent "
              "verification gave nearly equal YES support to ramekin and black bowl. For the black-bowl "
              "target it also strongly endorsed a cabinet distractor. Joint responses rejected those "
              "distractors. The book gained support, and red mug became resolvable because all candidate "
              "Y/N alternatives were available. These are descriptive cases, not evidence of the "
              "model's internal reasoning.", "",
              "Every joint scene generated exactly one YES, even though the prompt permits multiple "
              "matches or none. All twelve scenes have one evaluation target. This pilot therefore "
              "cannot establish that joint verification will preserve multiple-match or absent-target "
              "behavior instead of acting like forced selection.", "",
              "Joint support remains strongly saturated near one (forward minimum 0.985936; "
              "median approximately 0.999999997). It does not solve the score-range/calibration "
              "concern. Since there are no joint errors here, this sample cannot establish whether "
              "low support or a small gap reliably detects joint errors. Do not replace the "
              "production gate based on this result alone."]
    lines += ["", "## Limits and reproduction", "",
              "Twelve targets with one scene each are insufficient for calibration or a general superiority claim. "
              "A method with no scored errors cannot have a meaningful correctness-ranking AUROC on this sample. "
              "Missing scores count as deferred for coverage. Threshold sweeps are exploratory and never access "
              "a held-out test split. Cached independent calls versus fresh joint calls and the changed output "
              "protocol prevent attributing all differences solely to shared context. All raw responses, token "
              "alternatives, input/settings hashes and per-candidate scores are preserved.", "",
              "```sh", "source /home/jiabao/.venvs/lerobot-libero/bin/activate",
              "python -m scripts.run_vlm_joint_verification_pilot check",
              "python -m scripts.run_vlm_joint_verification_pilot run",
              "python -m scripts.run_vlm_joint_verification_pilot analyze", "```", "",
              "Run reuses completed responses; analyze is offline. Production VLM/HITL/CAD/localization/planning unchanged.", "",
              "![Comparison](comparison.png)", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"conditions": stats, "order_sensitivity": order_stats}, indent=2))
    print(f"Report: {directory / 'REPORT.md'}")


if __name__ == "__main__":
    analyze()
