"""Three-way saved-score report for Ablation 3c."""

from scripts.run_wrist_input_ablation import save_csv, save_json
from scripts.wrist_geometry_ablation_report import distribution


VIEW_FIELDS = {
    "3_C_obs": {"geometry": "geometry_winning_view_index_1based", "fused": "winning_view_index_1based"},
    "3b_frozen_C_f1": {"geometry": "frozen_view_index_1based", "fused": "frozen_view_index_1based"},
    "3c_C_f1": {"geometry": "geometry_winning_view_index_1based", "fused": "fused_winning_view_index_1based"},
}
FUSED_GEOMETRY_FIELDS = {"3_C_obs": "geometry_fitness_at_fused_view", "3b_frozen_C_f1": "geometry_score",
                         "3c_C_f1": "geometry_score_at_fused_view"}


def save_report(output_dir, results, grouped, all_views):
    truth = {t["cad_id"]: t["expected_object_id"] for t in results["3_C_obs"]["per_target"]}
    pairs = {name: {(r["cad_id"], r["object_id"]): r for t in result["associations"]
                    for r in t["candidate_ranking"]} for name, result in results.items()}
    distributions, summary = {}, {}
    for name, result in results.items():
        distributions[name] = {}
        fields = {"geometry": "geometry_score", "fused": "fused_score", "geometry_at_fused_view": FUSED_GEOMETRY_FIELDS[name]}
        for mode, field in fields.items():
            distributions[name][mode] = {}
            for identity in ("correct", "incorrect"):
                values = [p[field] for p in pairs[name].values()
                          if (p["object_id"] == truth[p["cad_id"]]) == (identity == "correct")]
                distributions[name][mode][identity] = {**distribution(values), "score_exactly_one": sum(v == 1. for v in values)}
        summary[name] = {"accuracy": {m: result["accuracy"][m] for m in ("geometry", "fused")},
            "mean_margins": {m: result["mean_margins"][m] for m in ("geometry", "fused")},
            "strict_positive_margin_targets": {m: sum(t[m]["correct_minus_best_incorrect"] > 0 for t in result["margins"])
                                               for m in ("geometry", "fused")},
            "correct_predictions_depending_on_ties": {m: sum(t["correct"][m] and margin[m]["correct_minus_best_incorrect"] == 0
                for t, margin in zip(result["per_target"], result["margins"], strict=True)) for m in ("geometry", "fused")}}
    view_changes = []
    for key, views in grouped.items():
        by_view = {r["view_index_1based"]: r for r in views}
        for mode, objective in (("geometry", "C_f1"), ("fused", "fused_C_f1")):
            current = pairs["3c_C_f1"][key][VIEW_FIELDS["3c_C_f1"][mode]]
            for baseline in ("3_C_obs", "3b_frozen_C_f1"):
                previous = pairs[baseline][key][VIEW_FIELDS[baseline][mode]]
                old_score, new_score = by_view[previous][objective], by_view[current][objective]
                view_changes.append({"baseline": baseline, "cad_id": key[0], "object_id": key[1],
                    "correct_pair": key[1] == truth[key[0]], "mode": mode,
                    "previous_view_index_1based": previous, "new_view_index_1based": current,
                    "view_changed": current != previous, "previous_view_C_f1_objective": old_score,
                    "new_view_C_f1_objective": new_score, "C_f1_objective_gain": new_score - old_score,
                    "previous_view_also_optimal": new_score == old_score})
    view_summary = {}
    for baseline in ("3_C_obs", "3b_frozen_C_f1"):
        view_summary[baseline] = {}
        for mode in ("geometry", "fused"):
            view_summary[baseline][mode] = {}
            for identity in ("all", "correct", "incorrect"):
                subset = [r for r in view_changes if r["baseline"] == baseline and r["mode"] == mode
                          and (identity == "all" or r["correct_pair"] == (identity == "correct"))]
                view_summary[baseline][mode][identity] = {"pair_count": len(subset),
                    "changed_views": sum(r["view_changed"] for r in subset),
                    "strict_objective_gains": sum(r["C_f1_objective_gain"] > 0 for r in subset),
                    "changed_only_by_tie": sum(r["view_changed"] and r["previous_view_also_optimal"] for r in subset)}
    changes = []
    for index, cad_id in enumerate(truth):
        for mode in ("geometry", "fused"):
            row = {"cad_id": cad_id, "expected_object_id": truth[cad_id], "mode": mode}
            for name, result in results.items():
                target, margin = result["per_target"][index], result["margins"][index]
                assert target["cad_id"] == margin["cad_id"] == cad_id
                prediction = target["predictions"][mode]
                row.update({f"{name}_prediction": prediction,
                    f"{name}_view": pairs[name][cad_id, prediction][VIEW_FIELDS[name][mode]],
                    f"{name}_margin": margin[mode]["correct_minus_best_incorrect"]})
            for baseline in ("3_C_obs", "3b_frozen_C_f1"):
                row[f"margin_delta_vs_{baseline}"] = row["3c_C_f1_margin"] - row[f"{baseline}_margin"]
                row[f"prediction_changed_vs_{baseline}"] = row["3c_C_f1_prediction"] != row[f"{baseline}_prediction"]
            changes.append(row)
    save_csv(output_dir / "selected_view_changes.csv", view_changes)
    save_csv(output_dir / "prediction_and_margin_changes.csv", changes)
    save_json(output_dir / "score_distributions.json", distributions)
    save_json(output_dir / "comparison.json", {"methods": summary, "selected_view_changes": view_summary,
        "per_target_changes": changes, "score_exactly_one": {name: {mode: {identity: d["score_exactly_one"]
            for identity, d in identities.items()} for mode, identities in modes.items()} for name, modes in distributions.items()}})
    missing = sum(r["C_f1"] is None for r in all_views)
    lines = ["# Ablation 3c: C_f1 selects its own CAD view", "",
        "All calculations use the 504 saved Ablation 3 view records. Geometry maximizes saved C_f1; fusion maximizes "
        "0.5 normalized CLS(v) + 0.5 C_f1(v) within the same view. These two methods select views independently. "
        "No DINO, rendering, localization, FPFH, RANSAC, ICP or coverage is rerun.", "",
        "| Method | Geometry accuracy | Fused accuracy | Mean geometry margin | Mean fused margin | Strict geometry wins |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, d in summary.items():
        lines.append(f"| {name} | {d['accuracy']['geometry']['correct']}/6 | {d['accuracy']['fused']['correct']}/6 "
                     f"| {d['mean_margins']['geometry']:+.6f} | {d['mean_margins']['fused']:+.6f} "
                     f"| {d['strict_positive_margin_targets']['geometry']}/6 |")
    old, new = summary["3b_frozen_C_f1"], summary["3c_C_f1"]
    lines += ["", f"Allowing C_f1 to reselect views changes geometry accuracy from {old['accuracy']['geometry']['correct']}/6 "
        f"to {new['accuracy']['geometry']['correct']}/6 and fused accuracy from {old['accuracy']['fused']['correct']}/6 "
        f"to {new['accuracy']['fused']['correct']}/6. Mean fused margin changes from {old['mean_margins']['fused']:+.6f} "
        f"to {new['mean_margins']['fused']:+.6f}.", "",
        "3 uses the published visible-CAD max-C_obs geometry and same-view C_obs fusion. 3b uses its published "
        "C_obs-fusion view for both C_f1 scores. Margins are correct minus strongest incorrect candidate. "
        "Exact candidate ties use ascending object ID; exact view ties use the lowest 1-based index.", "",
        f"{missing}/504 records have null C_f1 and no saved transform. They remain in the output with null fusion, "
        f"and cannot contribute a numeric maximum. All {504 - missing} defined views are scored; all 36 pairs have defined views. "
        "No missing coverage is imputed and no score threshold is applied.", "",
        "## Per-target predictions, winning views and margins", "",
        "IDs: 001 pulley; 002 large gear; 003 blue block; 004 red block; 005 medium gear; 006 rectangular pin. "
        "Each prediction is shown as object ID @ winning view; view comparisons for the same candidate are reported separately below.", "",
        "| CAD | Mode | 3 prediction @ view | 3b prediction @ view | 3c prediction @ view | Margin 3 / 3b / 3c |",
        "|---|---|---|---|---|---:|"]
    for row in changes:
        predictions = [f"{row[name + '_prediction']} @ {row[name + '_view']}" for name in results]
        margins = " / ".join(f"{row[name + '_margin']:+.6f}" for name in results)
        lines.append(f"| {row['cad_id']} | {row['mode']} | " + " | ".join(predictions) + f" | {margins} |")
    lines += ["", "## View changes for the same CAD/candidate pair", "",
        "Each row compares the relevant old view with the new 3c view. Objective gain uses C_f1 (geometry) or "
        "the same-view C_f1 fusion at both views, including when the baseline originally optimized C_obs.", "",
        "| Baseline | Mode | Pair identity | Pairs | Changed view | Strict objective gain | Changed only by tie |",
        "|---|---|---|---:|---:|---:|---:|"]
    for baseline, modes in view_summary.items():
        for mode, identities in modes.items():
            for identity, d in identities.items():
                lines.append(f"| {baseline} | {mode} | {identity} | {d['pair_count']} | {d['changed_views']} "
                             f"| {d['strict_objective_gains']} | {d['changed_only_by_tie']} |")
    lines += ["", "## Score distributions and exact 1.0 counts", "",
        "Geometry is each method's geometry-only score. Geometry at fused view is the geometry term actually used "
        "by its fused winner. Fusion is the full weighted score. Distributions use 6 correct and 30 incorrect pairs.", "",
        "| Method | Score | Identity | Mean | Median | Q25–Q75 | Min–max | Exactly 1.0 |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, modes in distributions.items():
        for mode, identities in modes.items():
            for identity, d in identities.items():
                lines.append(f"| {name} | {mode} | {identity} | {d['mean']:.6f} | {d['median']:.6f} "
                    f"| {d['q25']:.6f}–{d['q75']:.6f} | {d['min']:.6f}–{d['max']:.6f} | {d['score_exactly_one']}/{d['count']} |")
    lines += ["", "The 3c-versus-3b comparison isolates the frozen-view constraint; 3c-versus-3 compares the coverage "
        "score with view selection available to each. Maximization guarantees nondecreasing pair scores relative to 3b, "
        "but improvement in identity discrimination requires better margins or predictions. These results describe one "
        "fixed scene, not general identity-score reliability.", "", "## Saved evidence", "",
        "- [All 504 view scores, including undefined coverage](all_view_scores.csv).",
        "- [All 36 pair scores and both winning views](all_pair_scores.csv), [rankings and all margins](evaluation.json).",
        "- [Exact selected source records and transforms](winning_view_records.json); nested timings are historical Ablation 3 values.",
        "- [Per-target changes](prediction_and_margin_changes.csv), [every candidate's view changes](selected_view_changes.csv).",
        "- [Full score distributions](score_distributions.json), [three-way comparison](comparison.json).",
        "- [Protocol and source hashes](protocol.json), [validation](validation.json), [saved-scoring runtime](runtime.json).", "",
        "Production code and prior experiments are unchanged. No weights, thresholds, view set or geometric parameters are tuned.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
