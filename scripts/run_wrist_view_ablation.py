"""Ablation 2b: all-view patch scoring using exact saved Ablation 2 features.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_view_ablation

No feature extraction, rendering, geometry matching, or production-path changes.
"""

import os
import platform
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, save_csv, save_json, snapshot_hashes
from scripts.run_wrist_patch_ablation import VARIANTS, evaluate_variant, score_pair, verify_artifacts


ABLATION_2 = ROOT / "outputs/ablation_2_wrist_association_2026-09-11/20260912T160053225612Z"
OUTPUT_ROOT = ROOT / "outputs/ablation_2b_wrist_association_2026-09-11"


def select_views(cls, patches):
    """Combine within each view before maximizing; exact ties choose first index."""
    scores, winners, times = {}, {}, {}
    for name in VARIANTS:
        start = perf_counter()
        values = cls if name == "cls_only" else patches if name == "patch_only" else .5 * cls + .5 * patches
        index = int(np.argmax(values))
        scores[name] = float(values[index])
        winners[name] = index + 1
        times[name] = perf_counter() - start
    return scores, winners, times


def score_all_views(observed_cls, observed_patches, observed_weights, cad_cls, cad_patches, cad_weights):
    start = perf_counter()
    cls = np.clip(cad_cls @ observed_cls, -1, 1)
    cls_time = perf_counter() - start
    patches, view_times = [], []
    for index in range(len(cad_cls)):
        start = perf_counter()
        # A singleton view reuses the exact Ablation 2 foreground/occupancy matcher.
        row = score_pair(observed_cls, observed_patches, observed_weights,
                         cad_cls[index:index + 1], cad_patches[index:index + 1], cad_weights[index:index + 1])
        patches.append(row["patch_score"])
        view_times.append(perf_counter() - start)
    patches = np.asarray(patches)
    scores, winners, selection_times = select_views(cls, patches)
    return {"cls_view_cosines": cls.tolist(), "patch_view_scores": patches.tolist(),
            "combined_view_scores": (.5 * cls + .5 * patches).tolist(),
            "winning_view_index_1based": winners,
            "cls_score": scores["cls_only"], "patch_score": scores["patch_only"],
            "cls_patch_score": scores["cls_patch"],
            "runtime": {"cls_comparison_time_s": cls_time, "patch_matching_time_s": sum(view_times),
                        "per_view_patch_time_s": view_times, "view_selection_time_s": selection_times}}


def view_disagreements(pairs):
    views = [p["winning_view_index_1based"] for p in pairs]
    counts = {f"{a}_vs_{b}": sum(v[a] != v[b] for v in views)
              for a, b in (("cls_only", "patch_only"), ("cls_only", "cls_patch"), ("patch_only", "cls_patch"))}
    counts.update(pair_count=len(pairs), any_difference=sum(len(set(v.values())) > 1 for v in views),
                  all_three_different=sum(len(set(v.values())) == 3 for v in views))
    return counts


def save_comparison(output_dir, results, prior, pairs):
    truth = {t["cad_id"]: t["expected_object_id"] for t in results["cls_only"]["per_target"]}
    correct_pairs = [p for p in pairs if p["object_id"] == truth[p["cad_id"]]]
    disagreements = {"all_pairs": view_disagreements(pairs), "correct_pairs": view_disagreements(correct_pairs),
                     "incorrect_pairs": view_disagreements([p for p in pairs if p not in correct_pairs]),
                     "per_target": {cad: view_disagreements([p for p in pairs if p["cad_id"] == cad]) for cad in truth}}
    save_json(output_dir / "view_disagreements.json", disagreements)
    changes = []
    for name, result in results.items():
        for index, target in enumerate(result["per_target"]):
            before, after = prior[name]["margins"][index], result["margins"][index]
            assert before["cad_id"] == after["cad_id"] == target["cad_id"]
            row = {"variant": name, "cad_id": target["cad_id"], "expected_object_id": target["expected_object_id"]}
            for mode, margin_key in (("visual", "visual_raw"), ("fused", "fused")):
                old_margin = before[margin_key]["correct_minus_best_incorrect"]
                new_margin = after[margin_key]["correct_minus_best_incorrect"]
                row.update({f"{mode}_ablation_2_margin": old_margin, f"{mode}_ablation_2b_margin": new_margin,
                            f"{mode}_margin_delta": new_margin - old_margin,
                            f"{mode}_ablation_2_prediction": prior[name]["per_target"][index]["predictions"][mode],
                            f"{mode}_ablation_2b_prediction": target["predictions"][mode]})
            changes.append(row)
    save_csv(output_dir / "margin_changes.csv", changes)
    save_json(output_dir / "comparison.json", {"per_target_changes": changes, "view_disagreements": disagreements,
        "margin_improvements_vs_ablation_2": {name: {mode: sum(r[f"{mode}_margin_delta"] > 0
            for r in changes if r["variant"] == name) for mode in ("visual", "fused")} for name in VARIANTS}})
    lines = ["# Ablation 2b: unrestricted CAD-view selection", "",
             "Same six masked observations, exact saved DINOv2-small features, 14 CAD views, masks, "
             "and recorded geometry scores as Ablation 2. Every CAD/candidate/view combination is evaluated (504).", "",
             "| Method | Visual accuracy | Fused accuracy | Mean visual margin | Mean fused margin | Cached scoring + evaluation (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        lines.append(f"| {VARIANTS[name]} | {result['accuracy']['visual']['correct']}/6 "
                     f"| {result['accuracy']['fused']['correct']}/6 "
                     f"| {result['margin_summary']['visual_raw']['mean_correct_minus_best_incorrect']:+.6f} "
                     f"| {result['margin_summary']['fused']['mean_correct_minus_best_incorrect']:+.6f} "
                     f"| {result['runtime']['cached_scoring_and_evaluation_time_s']:.6f} |")
    lines += ["", "Margin = correct score minus the strongest incorrect score. Runtime assumes the exact saved "
              "features and masks are loaded; shared measurements are reused across methods. Feature loading "
              "and total wall time (including hash verification and result writing) are reported in [runtime.json](runtime.json). "
              "DINOv2, rendering, and geometry are not rerun, so these timings are not end-to-end inference timings.", "",
              "## Changes from Ablation 2", "",
              "Each method is compared with its own Ablation 2 result, where patch evidence was restricted to "
              "the CLS-selected view. Signed margin changes are 2b minus 2; no score-increase criterion is used.", "",
              "| CAD | Method | Visual margin 2 → 2b | Δ visual | Fused margin 2 → 2b | Δ fused | Visual prediction 2 → 2b | Fused prediction 2 → 2b |",
              "|---|---|---:|---:|---:|---:|---|---|"]
    for row in changes:
        if row["variant"] == "cls_only":
            continue
        lines.append(f"| {row['cad_id']} | {row['variant']} "
                     f"| {row['visual_ablation_2_margin']:+.6f} → {row['visual_ablation_2b_margin']:+.6f} "
                     f"| {row['visual_margin_delta']:+.6f} "
                     f"| {row['fused_ablation_2_margin']:+.6f} → {row['fused_ablation_2b_margin']:+.6f} "
                     f"| {row['fused_margin_delta']:+.6f} "
                     f"| {row['visual_ablation_2_prediction']} → {row['visual_ablation_2b_prediction']} "
                     f"| {row['fused_ablation_2_prediction']} → {row['fused_ablation_2b_prediction']} |")
    lines += ["", "CLS scores, rankings and margins reproduce Ablation 2. Object IDs: 001 pulley; 002 large gear; "
              "003 blue block; 004 red block; 005 medium gear; 006 rectangular pin.", "",
              "## CAD-view disagreements", "",
              "| Pair subset | Count | CLS ≠ patch | CLS ≠ combined | Patch ≠ combined | Any difference | All three different |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for subset in ("all_pairs", "correct_pairs", "incorrect_pairs"):
        row = disagreements[subset]
        lines.append(f"| {subset} | {row['pair_count']} | {row['cls_only_vs_patch_only']} "
                     f"| {row['cls_only_vs_cls_patch']} | {row['patch_only_vs_cls_patch']} "
                     f"| {row['any_difference']} | {row['all_three_different']} |")
    lines += ["", "These are view-index differences, not measured viewpoint correctness. The test of usefulness "
              "is the change in association accuracy and margins. Exact ties choose the first saved view.", "",
              "## Fixed scoring", "",
              "- CLS: `max_v CLS(v)`.",
              "- Patch: `max_v Patch(v)`; each Patch(v) uses the unchanged occupancy-weighted mean of maximum "
              "foreground token cosines for that view.",
              "- Per-view combined: `max_v [0.5 * CLS(v) + 0.5 * Patch(v)]`. Both terms must come from the same view.",
              "- Fusion: `0.5 * ((method_score + 1) / 2) + 0.5 * recorded_geometry_score`.", "",
              "Independent candidate rankings and conflict handling are unchanged. No production edits, "
              "new thresholds/views, pruning, geometry changes, or weight tuning were made.", "",
              "## Saved results", "",
              "- [All 504 view rows](per_view_scores.csv): CLS, patch and combined score for every view, plus winner indicators.",
              "- [All pair records and three winning indices](pair_scores.json).",
              "- [CLS scores](cls_only/all_pair_scores.csv), [rankings and margins](cls_only/evaluation.json).",
              "- [Patch scores](patch_only/all_pair_scores.csv), [rankings and margins](patch_only/evaluation.json).",
              "- [Combined scores](cls_patch/all_pair_scores.csv), [rankings and margins](cls_patch/evaluation.json).",
              "- [Per-target margin changes](margin_changes.csv), [view disagreement counts](view_disagreements.json).",
              "- [Protocol and source hashes](protocol.json), [validation](validation.json), [runtime](runtime.json).", "",
              "Features, masks, and RGB remain in the hash-verified Ablation 2 directory recorded in protocol.json. "
              "This single fixed scene is not a benchmark estimate; no settings were optimized from the outcomes.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = perf_counter()
    os.chdir(ROOT)
    source_paths = [ROOT / "scripts/run_wrist_view_ablation.py", ROOT / "scripts/run_wrist_patch_ablation.py",
                    ROOT / "scripts/run_wrist_input_ablation.py", ROOT / "scripts/run_wrist_association.py",
                    ROOT / "cad_object_association.py", ROOT / "CADPointCloudRegistration.py", ROOT / "main.py"]
    source_hashes = {str(p.relative_to(ROOT)): association.content_hash(p) for p in source_paths}
    prior_hashes, fixed_hashes = verify_artifacts(ABLATION_2), snapshot_hashes()
    prior_protocol = read_json(ABLATION_2 / "protocol.json")
    targets = prior_protocol["known_cad_ids"]
    assert len(targets) == 6 and prior_protocol["encoder"] == association.DINO_MODEL == "dinov2_vits14"
    assert (association.FUSION_METHOD, association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT) == ("weighted_mean", .5, .5)
    stage = perf_counter()
    with np.load(ABLATION_2 / "features/observed.npz", allow_pickle=False) as saved:
        observed = dict(saved)
    cads = {}
    for cad_id in targets:
        with np.load(ABLATION_2 / "features" / f"{cad_id}.npz", allow_pickle=False) as saved:
            cads[cad_id] = dict(saved)
    feature_load_time = perf_counter() - stage
    for n, features in [(6, observed), *((14, cad) for cad in cads.values())]:
        assert features["patch_tokens"].shape == (n, 16, 16, 384)
        assert features["cls_features"].shape == (n, 384)
        assert features["patch_occupancy"].shape == (n, 16, 16)
    object_ids = list(observed["object_ids"])
    assert len(set(object_ids)) == 6
    prior_pairs = read_json(ABLATION_2 / "pair_scores.json")
    assert len(prior_pairs) == 36
    pairs, cls_delta, patch_delta = [], 0., 0.
    for old in prior_pairs:
        cad_id, oid = old["cad_id"], old["object_id"]
        index, cad = object_ids.index(oid), cads[cad_id]
        row = score_all_views(observed["cls_features"][index], observed["patch_tokens"][index],
                              observed["patch_occupancy"][index], cad["cls_features"], cad["patch_tokens"], cad["patch_occupancy"])
        cls_delta = max(cls_delta, float(np.max(np.abs(np.asarray(row["cls_view_cosines"]) - old["cls_view_cosines"]))))
        old_view = old["selected_view_index_1based"] - 1
        patch_delta = max(patch_delta, abs(row["patch_view_scores"][old_view] - old["patch_score"]))
        assert row["winning_view_index_1based"]["cls_only"] == old_view + 1
        pairs.append({"cad_id": cad_id, "object_id": oid, **row,
                      "geometry_score": old["geometry_score"], "geometry_raw": old["geometry_raw"]})
    assert len({(p["cad_id"], p["object_id"]) for p in pairs}) == 36
    assert cls_delta < 1e-12 and patch_delta < 1e-12
    prior = {name: read_json(ABLATION_2 / name / "evaluation.json") for name in VARIANTS}
    audit = read_json(DATASET / "identity_audit.json")
    cached = {"observed_image_loading_time_s": 0., "observed_joint_feature_extraction_time_s": 0.,
              "observed_mask_preparation_time_s": 0.}
    results = {name: evaluate_variant(name, pairs, targets, audit, cached) for name in VARIANTS}
    output_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True)
    save_json(output_dir / "pair_scores.json", pairs)
    view_rows = []
    for pair in pairs:
        for index, (cls, patch, combined) in enumerate(zip(pair["cls_view_cosines"], pair["patch_view_scores"], pair["combined_view_scores"], strict=True), 1):
            view_rows.append({"cad_id": pair["cad_id"], "object_id": pair["object_id"], "view_index_1based": index,
                              "cls_cosine": cls, "patch_score": patch, "combined_score": combined,
                              **{f"{name}_winner": view == index for name, view in pair["winning_view_index_1based"].items()}})
    assert len(view_rows) == 504
    save_csv(output_dir / "per_view_scores.csv", view_rows)
    for name, result in results.items():
        runtime = {"cls_comparison_time_s": sum(p["runtime"]["cls_comparison_time_s"] for p in pairs) if name != "patch_only" else 0.,
                   "all_view_patch_scoring_time_s": sum(p["runtime"]["patch_matching_time_s"] for p in pairs) if name != "cls_only" else 0.,
                   "view_selection_time_s": sum(p["runtime"]["view_selection_time_s"][name] for p in pairs),
                   "fusion_ranking_evaluation_time_s": result["runtime"]["variant_fusion_ranking_evaluation_time_s"]}
        runtime["cached_scoring_and_evaluation_time_s"] = sum(runtime.values())
        runtime["scope"] = "Shared measured stages reused per method; features/masks already loaded; no encoder/render/geometry runs"
        result["runtime"] = runtime
        result["view_policy"] = "Each method maximizes its own per-view score; combined uses both terms from one view"
        directory = output_dir / name
        directory.mkdir()
        flat = []
        for target in result["associations"]:
            for row in target["candidate_ranking"]:
                row["selected_view_index_1based"] = row["winning_view_index_1based"][name]
                flat.append({**{k: row[k] for k in ("cad_id", "object_id", "cls_score", "patch_score", "cls_patch_score",
                                                   "visual_raw", "visual_score", "geometry_score", "fused_score", "selected_view_index_1based")},
                             **{f"{mode}_winning_view": v for mode, v in row["winning_view_index_1based"].items()},
                             **{f"{mode}_rank": ids.index(row["object_id"]) + 1 for mode, ids in target["rankings"].items()}})
        save_csv(directory / "all_pair_scores.csv", flat)
        save_json(directory / "evaluation.json", result)
        print(f"{name}: accuracy={result['accuracy']} margins={result['margin_summary']}", flush=True)
    assert all(a["rankings"] == b["rankings"] for a, b in zip(results["cls_only"]["associations"], prior["cls_only"]["associations"], strict=True))
    assert results["cls_only"]["margins"] == prior["cls_only"]["margins"]
    save_comparison(output_dir, results, prior, pairs)
    save_json(output_dir / "protocol.json", {
        "dataset": str(DATASET.relative_to(ROOT)), "ablation_2_source": str(ABLATION_2.relative_to(ROOT)),
        "encoder": prior_protocol["encoder"], "encoder_source": prior_protocol["encoder_source"],
        "checkpoint_sha256": prior_protocol["checkpoint_sha256"], "known_cad_ids": targets,
        "features_and_masks": "Exact saved Ablation 2 NPZ arrays; no renormalization, extraction, rendering or mask resizing",
        "patch_matching": "Reuse unchanged Ablation 2 score_pair with one view at a time; every view evaluated",
        "methods": {"cls_only": "max_v CLS(v)", "patch_only": "max_v Patch(v)",
                    "cls_patch": "max_v [0.5 * CLS(v) + 0.5 * Patch(v)]"},
        "fusion": "0.5 * ((method_raw_score + 1) / 2) + 0.5 * recorded_geometry_score",
        "view_indexing": "1-based saved view_01 through view_14; exact ties select first index",
        "geometry": "All 36 raw metrics and scores reused from verified Ablation 2 results",
        "source_sha256": source_hashes, "ablation_2_sha256": prior_hashes, "dataset_sha256": fixed_hashes,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                        **{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
    })
    assert verify_artifacts(ABLATION_2) == prior_hashes and snapshot_hashes() == fixed_hashes
    assert all(association.content_hash(ROOT / p) == digest for p, digest in source_hashes.items())
    save_json(output_dir / "validation.json", {"max_cls_view_delta_vs_ablation_2": cls_delta,
        "max_patch_delta_at_ablation_2_selected_view": patch_delta, "cls_rankings_and_margins_match_ablation_2": True,
        "pair_count": 36, "scored_pair_view_count": 504, "saved_features_masks_inputs_and_geometry_unchanged": True,
        "source_and_fixed_dataset_hashes_unchanged": True, "production_pipeline_modified": False})
    save_json(output_dir / "runtime.json", {"total_wall_time_after_imports_s": perf_counter() - start,
        "saved_feature_loading_time_s": feature_load_time, "variant_runtime": {name: r["runtime"] for name, r in results.items()},
        "scope": "Total includes integrity checks, saved feature loading, matching, evaluation and writing; excludes final output hashes"})
    (output_dir / "SHA256SUMS").write_text("".join(f"{association.content_hash(p)}  {p.relative_to(output_dir).as_posix()}\n"
        for p in sorted(output_dir.rglob("*")) if p.is_file()), encoding="utf-8")
    print(f"Saved Ablation 2b: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
