"""Saved-score comparison for Ablation 3b; no geometric computation."""

from scripts.run_wrist_input_ablation import save_csv, save_json
from scripts.wrist_geometry_ablation_report import distribution


def save_report(output_dir, results, records, scoring_times):
    truth = {t["cad_id"]: t["expected_object_id"] for t in results["A_C_obs"]["per_target"]}
    distributions, summary = {}, {}
    for name, result in results.items():
        pairs = [r for t in result["associations"] for r in t["candidate_ranking"]]
        distributions[name] = {}
        for mode, score in (("geometry", "geometry_score"), ("fused", "fused_score")):
            distributions[name][mode] = {}
            for label in ("correct", "incorrect"):
                values = [p[score] for p in pairs if (p["object_id"] == truth[p["cad_id"]]) == (label == "correct")]
                distributions[name][mode][label] = {**distribution(values), "score_exactly_one": sum(v == 1. for v in values)}
        summary[name] = {"accuracy": result["accuracy"], "mean_margins": result["mean_margins"],
            "strict_positive_margin_targets": {mode: sum(t[mode]["correct_minus_best_incorrect"] > 0 for t in result["margins"])
                                               for mode in ("geometry", "fused")},
            "correct_predictions_with_tied_incorrect_score": {mode: sum(
                t["correct"][mode] and m[mode]["correct_minus_best_incorrect"] == 0
                for t, m in zip(result["per_target"], result["margins"], strict=True)) for mode in ("geometry", "fused")}}
    changes = []
    a, b = results["A_C_obs"], results["B_C_f1"]
    for index, before in enumerate(a["per_target"]):
        after = b["per_target"][index]
        for mode in ("geometry", "fused"):
            am, bm = a["margins"][index][mode], b["margins"][index][mode]
            changes.append({"cad_id": before["cad_id"], "expected_object_id": before["expected_object_id"], "mode": mode,
                "A_prediction": before["predictions"][mode], "B_prediction": after["predictions"][mode],
                "prediction_changed": before["predictions"][mode] != after["predictions"][mode],
                "A_correct_score": am["correct_score"], "B_correct_score": bm["correct_score"],
                "A_strongest_incorrect_object_id": am["strongest_incorrect_object_id"],
                "B_strongest_incorrect_object_id": bm["strongest_incorrect_object_id"],
                "A_strongest_incorrect_score": am["strongest_incorrect_score"], "B_strongest_incorrect_score": bm["strongest_incorrect_score"],
                "A_margin": am["correct_minus_best_incorrect"], "B_margin": bm["correct_minus_best_incorrect"],
                "margin_delta": bm["correct_minus_best_incorrect"] - am["correct_minus_best_incorrect"]})
    save_csv(output_dir / "prediction_and_margin_changes.csv", changes)
    save_csv(output_dir / "pair_score_changes.csv", [{"cad_id": r["cad_id"], "object_id": r["object_id"],
        "correct_pair": r["object_id"] == truth[r["cad_id"]], "frozen_view_index_1based": r["view_index_1based"],
        "normalized_cls": r["normalized_cls"], "C_obs": r["C_obs"], "C_cad": r["C_cad"], "C_f1": r["C_f1"],
        "geometry_delta": r["C_f1"] - r["C_obs"], "fused_delta": .5 * (r["C_f1"] - r["C_obs"])} for r in records])
    save_json(output_dir / "score_distributions.json", distributions)
    save_json(output_dir / "comparison.json", {"methods": summary, "per_target_changes": changes,
        "score_exactly_one": {name: {label: d["geometry"][label]["score_exactly_one"] for label in ("correct", "incorrect")}
                              for name, d in distributions.items()}})
    lines = ["# Ablation 3b: bidirectional visible-surface coverage", "",
        "All 36 CAD/candidate pairs retain their Ablation 3 fused-winning view, saved transform, and that view's CLS score. "
        "A uses saved C_obs (registration fitness); B substitutes saved C_f1. Both fuse 0.5 normalized CLS + 0.5 geometry. "
        "No DINO, rendering, localization, FPFH, RANSAC, ICP, coverage recomputation or view reselection is performed.", "",
        "| Method | Geometry accuracy | Fused accuracy | Mean geometry margin | Mean fused margin | Strict geometry wins |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        lines.append(f"| {name} | {result['accuracy']['geometry']['correct']}/6 | {result['accuracy']['fused']['correct']}/6 "
                     f"| {result['mean_margins']['geometry']:+.6f} | {result['mean_margins']['fused']:+.6f} "
                     f"| {summary[name]['strict_positive_margin_targets']['geometry']}/6 |")
    lines += ["", "Margin = correct score minus strongest incorrect score. Exact candidate ties retain ascending object-ID order. "
        "Geometry-only here uses the frozen fused-winning view for both methods; Ablation 3's geometry-only column used "
        "an independent max-fitness view. A's fused scores, rankings and margins exactly reproduce Ablation 3 B.", "",
        "## Per-target predictions and margins", "",
        "IDs: 001 pulley; 002 large gear; 003 blue block; 004 red block; 005 medium gear; 006 rectangular pin.", "",
        "| CAD | Mode | Prediction A → B | Margin A → B | Margin change |", "|---|---|---|---:|---:|"]
    for row in changes:
        lines.append(f"| {row['cad_id']} | {row['mode']} | {row['A_prediction']} → {row['B_prediction']} "
                     f"| {row['A_margin']:+.6f} → {row['B_margin']:+.6f} | {row['margin_delta']:+.6f} |")
    lines += ["", "## Geometry scores exactly 1.0", "",
        "| Method | Correct pairs (6) | Incorrect pairs (30) | Correct geometry predictions depending on ties |",
        "|---|---:|---:|---:|"]
    for name, d in distributions.items():
        lines.append(f"| {name} | {d['geometry']['correct']['score_exactly_one']} "
                     f"| {d['geometry']['incorrect']['score_exactly_one']} "
                     f"| {summary[name]['correct_predictions_with_tied_incorrect_score']['geometry']} |")
    lines += ["", "## Correct versus incorrect score distributions", "",
        "C_f1 = 2 C_obs C_cad / (C_obs + C_cad). All 36 frozen views have valid saved transforms and coverage; "
        "no missing values are imputed. Score 1.0 is a saturation diagnostic, not a calibrated identity confidence.", "",
        "| Method | Score | Identity | Count | Mean | Median | Q25–Q75 | Min–max |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, modes in distributions.items():
        for mode, labels in modes.items():
            for label, d in labels.items():
                lines.append(f"| {name} | {mode} | {label} | {d['count']} | {d['mean']:.6f} | {d['median']:.6f} "
                             f"| {d['q25']:.6f}–{d['q75']:.6f} | {d['min']:.6f}–{d['max']:.6f} |")
    lines += ["", "This is a fixed-view score substitution on one six-object scene. View selection was inherited from "
        "the original C_obs fusion. Interpret saturation together with accuracy and margins; a lower incorrect score "
        "alone does not establish better discrimination.", "", "## Runtime and saved evidence", "",
        *[f"- {name} saved scoring and evaluation: {seconds:.6f} s." for name, seconds in scoring_times.items()],
        "- [Frozen views, transforms and original diagnostics](frozen_view_records.json).",
        "- [A scores](A_C_obs/all_pair_scores.csv), [rankings and all margins](A_C_obs/evaluation.json).",
        "- [B scores](B_C_f1/all_pair_scores.csv), [rankings and all margins](B_C_f1/evaluation.json).",
        "- [Every pair's score change](pair_score_changes.csv), [prediction and margin changes](prediction_and_margin_changes.csv).",
        "- [Full distributions and sorted values](score_distributions.json), [comparison](comparison.json).",
        "- [Protocol and hashes](protocol.json), [validation](validation.json), [runtime scope](runtime.json).", "",
        "Production code, source registrations, candidates, view records and weights are unchanged. No thresholds or tuning.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
