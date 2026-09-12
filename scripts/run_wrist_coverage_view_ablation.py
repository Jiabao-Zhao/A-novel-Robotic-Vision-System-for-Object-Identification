"""Ablation 3c: choose C_f1 views using only saved Ablation 3 scores.

python -m scripts.run_wrist_coverage_view_ablation

No DINO, rendering, localization, registration or coverage computation.
"""

from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite
from time import perf_counter

import cad_object_association as association
from scripts.run_wrist_coverage_ablation import ABLATION_3
from scripts.run_wrist_geometry_ablation import evaluate_pairs
from scripts.run_wrist_input_ablation import ROOT, read_json, save_csv, save_json
from scripts.run_wrist_patch_ablation import verify_artifacts
from scripts.wrist_coverage_view_report import save_report


ABLATION_3B = ROOT / "outputs/ablation_3b_wrist_association_2026-09-11/20260912T191623251244Z"
OUTPUT_ROOT = ROOT / "outputs/ablation_3c_wrist_association_2026-09-11"


def score_views(records):
    """Keep undefined coverage null; only the weighted sum is new arithmetic."""
    rows = []
    for record in records:
        row = {k: record[k] for k in ("cad_id", "object_id", "view_index_1based", "cls_cosine",
            "normalized_cls", "C_obs", "C_cad", "C_f1", "coverage_status")}
        if row["C_f1"] is None:
            if row["C_cad"] is not None or record["geometry_raw"]["T_observed_from_cad"] is not None:
                raise ValueError("Null C_f1 must refer to a saved registration without an alignment.")
            fused = None
        else:
            if any(not isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ("C_obs", "C_cad", "C_f1")):
                raise ValueError("Invalid saved coverage.")
            if record["geometry_raw"]["T_observed_from_cad"] is None:
                raise ValueError("Defined C_f1 requires a saved alignment.")
            fused = association.fuse_scores(row["normalized_cls"], row["C_f1"])
        rows.append({**row, "fused_C_f1": fused})
    return rows


def select_views(rows):
    """Independent C_f1 and same-view fusion maxima; ties use first view index."""
    ordered = sorted(rows, key=lambda r: r["view_index_1based"])
    valid = [r for r in ordered if r["C_f1"] is not None]
    if not valid:
        raise ValueError("CAD/candidate pair has no defined saved C_f1; cannot rank it.")
    geometry = max(valid, key=lambda r: r["C_f1"])
    fused = max(valid, key=lambda r: r["fused_C_f1"])
    visual = max(ordered, key=lambda r: r["cls_cosine"])
    return {"cad_id": geometry["cad_id"], "object_id": geometry["object_id"],
        "visual_raw": visual["cls_cosine"], "visual_score": visual["normalized_cls"],
        "geometry_score": geometry["C_f1"], "fused_score": fused["fused_C_f1"],
        "geometry_winning_view_index_1based": geometry["view_index_1based"],
        "fused_winning_view_index_1based": fused["view_index_1based"],
        "normalized_cls_at_fused_view": fused["normalized_cls"],
        "geometry_score_at_fused_view": fused["C_f1"],
        "C_obs_at_geometry_view": geometry["C_obs"], "C_cad_at_geometry_view": geometry["C_cad"],
        "C_obs_at_fused_view": fused["C_obs"], "C_cad_at_fused_view": fused["C_cad"],
        "geometry_max_tie_count": sum(r["C_f1"] == geometry["C_f1"] for r in valid),
        "fused_max_tie_count": sum(r["fused_C_f1"] == fused["fused_C_f1"] for r in valid),
        "defined_view_count": len(valid), "undefined_view_count": len(rows) - len(valid)}


def main():
    start = perf_counter()
    inputs = {"ablation_3": verify_artifacts(ABLATION_3), "ablation_3b": verify_artifacts(ABLATION_3B)}
    prior_protocol = read_json(ABLATION_3B / "protocol.json")
    assert prior_protocol["ablation_3_sha256"] == inputs["ablation_3"]
    sources = [*prior_protocol["source_sha256"], "scripts/run_wrist_coverage_view_ablation.py",
               "scripts/wrist_coverage_view_report.py"]
    source_hashes = {p: association.content_hash(ROOT / p) for p in sources}
    assert (association.FUSION_METHOD, association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT) == ("weighted_mean", .5, .5)
    results = {"3_C_obs": read_json(ABLATION_3 / "B_visible_cad/evaluation.json"),
               "3b_frozen_C_f1": read_json(ABLATION_3B / "B_C_f1/evaluation.json")}
    records = read_json(ABLATION_3 / "all_view_scores.json")
    by_view = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r for r in records}
    assert len(by_view) == len(records) == 504
    stage = perf_counter()
    rows = score_views(records)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cad_id"], row["object_id"]].append(row)
    assert len(grouped) == 36
    assert all(sorted(r["view_index_1based"] for r in views) == list(range(1, 15)) for views in grouped.values())
    pairs = [select_views(views) for views in grouped.values()]
    scoring_time = perf_counter() - stage
    targets = {t["cad_id"]: t["target_description"] for t in results["3_C_obs"]["associations"]}
    audit = [{"object_id": t["expected_object_id"], "simulator_instance": t["cad_id"], "status": "matched"}
             for t in results["3_C_obs"]["per_target"]]
    assert len(targets) == len(audit) == 6
    stage = perf_counter()
    results["3c_C_f1"] = evaluate_pairs("3c_C_f1", pairs, targets, audit)
    evaluation_time = perf_counter() - stage
    prior_pairs = {name: {(r["cad_id"], r["object_id"]): r for t in result["associations"]
                         for r in t["candidate_ranking"]} for name, result in results.items()}
    assert all(set(p) == set(grouped) for p in prior_pairs.values())
    for key, views in grouped.items():
        a, b, c = (prior_pairs[name][key] for name in results)
        assert a["geometry_score"] == max(r["C_obs"] for r in views)
        assert a["fused_score"] == max(by_view[*key, r["view_index_1based"]]["fused_score"] for r in views)
        frozen = next(r for r in views if r["view_index_1based"] == b["frozen_view_index_1based"])
        assert b["frozen_view_index_1based"] == a["winning_view_index_1based"]
        assert b["geometry_score"] == frozen["C_f1"] and b["fused_score"] == frozen["fused_C_f1"]
        assert c["geometry_score"] >= b["geometry_score"] and c["fused_score"] >= b["fused_score"]
    output_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True)
    save_csv(output_dir / "all_view_scores.csv", rows)
    save_csv(output_dir / "all_pair_scores.csv", pairs)
    save_json(output_dir / "evaluation.json", results["3c_C_f1"])
    save_json(output_dir / "winning_view_records.json", [{"cad_id": p["cad_id"], "object_id": p["object_id"],
        **{mode: by_view[p["cad_id"], p["object_id"], p[f"{mode}_winning_view_index_1based"]]
           for mode in ("geometry", "fused")}} for p in pairs])
    save_report(output_dir, results, grouped, rows)
    assert verify_artifacts(ABLATION_3) == inputs["ablation_3"]
    assert verify_artifacts(ABLATION_3B) == inputs["ablation_3b"]
    assert all(association.content_hash(ROOT / p) == digest for p, digest in source_hashes.items())
    save_json(output_dir / "protocol.json", {"ablation_3": str(ABLATION_3.relative_to(ROOT)),
        "ablation_3b": str(ABLATION_3B.relative_to(ROOT)), "input_sha256": inputs, "source_sha256": source_hashes,
        "geometry": "max_v saved C_f1(v)", "fusion": "max_v [0.5 * saved normalized_CLS(v) + 0.5 * saved C_f1(v)]",
        "CLS": "Saved normalized CLS used verbatim; visual-only control uses max CLS over all 14 views",
        "null_coverage": "Retain all records; null C_f1 means no saved alignment and has no numeric maximum contribution. No imputation.",
        "ties": "Lowest 1-based view index for exact view ties; ascending object ID for candidate ties",
        "comparison": "3 uses its independent max-C_obs geometry and view-consistent fusion; 3b uses its frozen C_obs-fusion view for both",
        "winning_records": "Exact original records and transforms; nested runtime fields are historical Ablation 3 timings",
        "ground_truth": "Only saved Ablation 3 evaluation identities; no labels enter scoring or view selection",
        "production_pipeline_modified": False, "new_inference_rendering_localization_registration_or_coverage_calls": 0})
    save_json(output_dir / "validation.json", {"saved_view_records": len(rows), "candidate_pairs": len(pairs),
        "defined_C_f1_views": sum(r["C_f1"] is not None for r in rows),
        "undefined_C_f1_views_retained": sum(r["C_f1"] is None for r in rows),
        "all_pairs_have_defined_views": True, "prior_scores_reproduce_saved_records": True,
        "all_new_pair_scores_at_least_frozen_C_f1_scores": True, "source_and_input_hashes_unchanged": True})
    save_json(output_dir / "runtime.json", {"saved_scoring_and_view_selection_s": scoring_time,
        "evaluation_s": evaluation_time, "total_wall_time_after_imports_s": perf_counter() - start,
        "scope": "Saved-score arithmetic and IO only; total excludes imports and final output hashes"})
    (output_dir / "SHA256SUMS").write_text("".join(
        f"{association.content_hash(p)}  {p.relative_to(output_dir).as_posix()}\n"
        for p in sorted(output_dir.rglob("*")) if p.is_file()), encoding="utf-8")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']} | mean margins {result['mean_margins']}", flush=True)
    print(f"Saved Ablation 3c: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
