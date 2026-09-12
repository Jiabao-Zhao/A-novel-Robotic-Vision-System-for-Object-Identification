"""Evaluation-only coverage distributions and comparisons for Ablation 3."""

import numpy as np

from scripts.run_wrist_input_ablation import save_csv, save_json


def distribution(values):
    valid = [float(v) for v in values if v is not None]
    result = {"count": len(valid), "missing": len(values) - len(valid), "sorted_values": sorted(valid)}
    quantiles = np.quantile(valid, [0, .25, .5, .75, 1]).tolist() if valid else [None] * 5
    result.update(dict(zip(("min", "q25", "median", "q75", "max"), quantiles)))
    result["mean"] = float(np.mean(valid)) if valid else None
    result["std_population"] = float(np.std(valid)) if valid else None
    return result


def save_report(output_dir, results, full_pairs, visible_pairs, all_views, timing):
    truth = {r["cad_id"]: r["expected_object_id"] for r in results["A_full_cad"]["per_target"]}
    by_view = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r for r in all_views}
    fused_views = [by_view[p["cad_id"], p["object_id"], p["winning_view_index_1based"]] for p in visible_pairs]
    geometry_views = [by_view[p["cad_id"], p["object_id"], p["geometry_winning_view_index_1based"]] for p in visible_pairs]
    groups = {"A_full_CAD_pairs": full_pairs, "B_fused_winning_view_pairs": fused_views,
              "B_geometry_winning_view_pairs": geometry_views, "B_all_view_registrations": all_views}
    distributions, saturation, samples = {}, {}, []
    for name, rows in groups.items():
        distributions[name], saturation[name] = {}, {}
        for label in ("correct", "incorrect"):
            subset = [r for r in rows if (r["object_id"] == truth[r["cad_id"]]) == (label == "correct")]
            distributions[name][label] = {key: distribution([r[key] for r in subset]) for key in ("C_obs", "C_cad", "C_f1")}
            saturation[name][label] = {"total": len(subset), "fitness_exactly_one": sum(
                (r["geometry_fitness"] if "geometry_fitness" in r else r["geometry_score"]) == 1. for r in subset)}
            for row in subset:
                samples.append({"subset": name, "identity": label, "cad_id": row["cad_id"], "object_id": row["object_id"],
                                "view_index_1based": row.get("view_index_1based"),
                                **{key: row[key] for key in ("C_obs", "C_cad", "C_f1", "coverage_status")}})
    save_json(output_dir / "coverage_distributions.json", distributions)
    save_csv(output_dir / "coverage_samples.csv", samples)
    save_json(output_dir / "fitness_saturation.json", saturation)
    changes = []
    for index, a in enumerate(results["A_full_cad"]["per_target"]):
        b = results["B_visible_cad"]["per_target"][index]
        row = {"cad_id": a["cad_id"], "expected_object_id": a["expected_object_id"]}
        for mode in ("visual", "geometry", "fused"):
            am = results["A_full_cad"]["margins"][index][mode]["correct_minus_best_incorrect"]
            bm = results["B_visible_cad"]["margins"][index][mode]["correct_minus_best_incorrect"]
            row.update({f"A_{mode}_prediction": a["predictions"][mode], f"B_{mode}_prediction": b["predictions"][mode],
                        f"A_{mode}_margin": am, f"B_{mode}_margin": bm, f"{mode}_margin_delta": bm - am})
        changes.append(row)
    save_csv(output_dir / "margin_changes.csv", changes)
    save_json(output_dir / "comparison.json", {
        "accuracy": {name: r["accuracy"] for name, r in results.items()},
        "mean_margins": {name: r["mean_margins"] for name, r in results.items()},
        "per_target_changes": changes, "fitness_saturation": saturation,
        "view_registrations_matched": sum(r["geometry_raw"]["status"] == "matched" for r in all_views),
        "view_registrations_failed": sum(r["geometry_raw"]["status"] != "matched" for r in all_views),
    })
    lines = ["# Ablation 3: full CAD versus view-consistent visible CAD", "",
             "A reproduces the recorded full-CAD geometric scores and masked CLS baseline. B registers the same "
             "six observed clouds against visible surfaces from all 14 existing RGB views: 504 new registrations. "
             "The RGB pixels, CLS descriptors, CADs, 5-mm preprocessing, FPFH/RANSAC/ICP settings and 0.5/0.5 fusion are fixed.", "",
             "| Method | Visual accuracy | Geometry accuracy | Fused accuracy | Mean geometry margin | Mean fused margin |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        a = result["accuracy"]
        lines.append(f"| {name} | {a['visual']['correct']}/6 | {a['geometry']['correct']}/6 | {a['fused']['correct']}/6 "
                     f"| {result['mean_margins']['geometry']:+.6f} | {result['mean_margins']['fused']:+.6f} |")
    lines += ["", "Margins are correct minus strongest incorrect candidate. Visual-only uses max-view CLS for both methods. "
              "B geometry-only uses max-view visible fitness; its fused score uses the maximum same-view sum. "
              "The independent visual and geometry maxima are never combined to produce B's fused score.", "",
              "## Per-target changes", "",
              "IDs: 001 pulley; 002 large gear; 003 blue block; 004 red block; 005 medium gear; 006 rectangular pin.", "",
              "| CAD | Geometry prediction A → B | Fused prediction A → B | Geometry margin A → B | Fused margin A → B |",
              "|---|---|---|---:|---:|"]
    for row in changes:
        lines.append(f"| {row['cad_id']} | {row['A_geometry_prediction']} → {row['B_geometry_prediction']} "
                     f"| {row['A_fused_prediction']} → {row['B_fused_prediction']} "
                     f"| {row['A_geometry_margin']:+.6f} → {row['B_geometry_margin']:+.6f} "
                     f"| {row['A_fused_margin']:+.6f} → {row['B_fused_margin']:+.6f} |")
    lines += ["", "## Incorrect fits with fitness exactly 1.0", "",
              "| Evaluated subset | Incorrect pairs or view records | Fitness = 1.0 |", "|---|---:|---:|"]
    for name, groups_by_identity in saturation.items():
        row = groups_by_identity["incorrect"]
        lines.append(f"| {name} | {row['total']} | {row['fitness_exactly_one']} |")
    lines += ["", "The primary B pair diagnostic uses its fused-winning view. The geometry-winning subset shows "
              "how many incorrect pairs have at least one view with perfect observed fitness. All-view records "
              "count each view separately and are not independent scenes.", "",
              "## Coverage distributions", "",
              "C_obs: fraction of observed registration points within 7.5 mm of aligned CAD points. "
              "C_cad: reverse fraction of CAD points within 7.5 mm of the observation. "
              "C_f1 = 2 C_obs C_cad / (C_obs + C_cad). A's C_cad denominator is the full CAD; "
              "B's denominator is its visible CAD. These diagnostics do not affect any prediction.", "",
              "Cells below show mean / median [Q25, Q75]. Full sorted samples, minima, maxima, and standard "
              "deviations for all four subsets are in [coverage_distributions.json](coverage_distributions.json).", "",
              "| Subset | Identity | C_obs | C_cad | C_f1 |", "|---|---|---|---|---|"]
    for name in ("A_full_CAD_pairs", "B_fused_winning_view_pairs"):
        for label in ("correct", "incorrect"):
            cells = []
            for key in ("C_obs", "C_cad", "C_f1"):
                d = distributions[name][label][key]
                cells.append(f"{d['mean']:.4f} / {d['median']:.4f} [{d['q25']:.4f}, {d['q75']:.4f}]; n={d['count']}"
                             if d["count"] else "unavailable")
            lines.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
    failures = sum(r["geometry_raw"]["status"] != "matched" for r in all_views)
    lines += ["", f"B has {failures}/504 registrations without a valid alignment. A failed registration retains "
              "zero observed fitness. Reverse coverage and harmonic coverage are null without a transform; "
              "their missing counts are reported and omitted from numeric summaries, rather than filled with zero.", "",
              "## Geometry and view construction", "",
              "- Use the existing renderer's 224×224 orthographic rays and first visible surface hit. "
              "Every generated RGB image must equal the fixed template exactly.",
              "- Back-project each foreground ray: `(origin + t_hit * direction) * CAD_radius + CAD_center`. "
              "The result is in the original CAD frame, in meters. No pinhole camera approximation or scale adaptation is used.",
              "- Interpolate mesh vertex normals at each surface hit, matching the full-CAD sampler's normal source. "
              "Apply the unchanged `compute_fpfh` call: 5-mm voxels, 10-mm normal radius when normals are missing, "
              "25-mm FPFH radius. Observed clouds retain their existing camera-oriented preprocessing.",
              "- Use unchanged rigid FPFH/RANSAC and point-to-point ICP. Coverage uses the actual registration point sets. "
              "The reported observed-to-visible-CAD RMSE retains the existing registrar's additional 5-mm voxel pass after alignment.",
              "- `A = 0.5 * ((max_v CLS(v) + 1) / 2) + 0.5 * full_CAD_fitness`.",
              "- `B = max_v [0.5 * ((CLS(v) + 1) / 2) + 0.5 * visible_CAD_fitness(v)]`.", "",
              "View consistency means that RGB and the visible subset share a view index. Registration remains "
              "the requested unrestricted rigid registration; its rotation is not constrained to the rendering camera. "
              "Exact view-score ties choose the first view; candidate ties use object ID. Independent target conflicts are preserved.", "",
              "## Runtime", "",
              f"B's 504 registrations: **{timing['new_registration_time_s']:.4f} s**. "
              f"Visible-CAD surface/descriptor preparation and cache handling: {timing['cad_visible_preparation_time_s']:.4f} s. "
              f"Observed preprocessing: {timing['observed_preprocessing_time_s']:.4f} s. "
              f"Coverage diagnostics: {timing['visible_coverage_diagnostic_time_s'] + timing['baseline_coverage_diagnostic_time_s']:.4f} s.", "",
              "A's registrations are reused, so this is not an end-to-end speed comparison. Saved CLS descriptors "
              "are reused in both methods. Full measured stages, cache status and the historical full-CAD registration "
              "time are recorded in [runtime.json](runtime.json) and [validation.json](validation.json).", "",
              "## Saved evidence", "",
              "- [All 504 CAD/candidate/view scores](all_view_scores.csv), [full metrics and transforms](all_view_scores.json).",
              "- [A pair scores](A_full_cad/all_pair_scores.csv), [rankings, predictions and margins](A_full_cad/evaluation.json).",
              "- [B pair scores and winning views](B_visible_cad/all_pair_scores.csv), [rankings, predictions and margins](B_visible_cad/evaluation.json).",
              "- [Per-target margin changes](margin_changes.csv), [fitness saturation](fitness_saturation.json), "
              "[coverage samples](coverage_samples.csv), [distribution summaries](coverage_distributions.json).",
              "- [Visible geometry cache index](cad_visible_geometry.json): hashes, ray depths, original metric surface "
              "points, 5-mm point clouds/normals, local FPFH, and view metadata. Observed registration features are in `observed_geometry/`.",
              "- [Protocol and provenance](protocol.json), [validation](validation.json).", "",
              "Production code and previous experiments are unchanged. No pruning, thresholds, new views, SAM, "
              "patch-score changes, weight tuning or adaptive voxel size were introduced. This remains one fixed six-object scene.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
