"""Ablation 3b: rescore the saved Ablation 3 fused-winning registrations.

python -m scripts.run_wrist_coverage_ablation

Only saved JSON scores, coverage, transforms and identity records are consumed.
No feature extraction, rendering, localization or registration is called.
"""

from datetime import datetime, timezone
from math import isclose, isfinite
from time import perf_counter

import cad_object_association as association
from scripts.run_wrist_geometry_ablation import evaluate_pairs
from scripts.run_wrist_input_ablation import ROOT, read_json, save_csv, save_json
from scripts.run_wrist_patch_ablation import verify_artifacts
from scripts.wrist_coverage_ablation_report import save_report


ABLATION_3 = ROOT / "outputs/ablation_3_wrist_association_2026-09-11/20260912T163935340572Z"
OUTPUT_ROOT = ROOT / "outputs/ablation_3b_wrist_association_2026-09-11"
VARIANTS = {"A_C_obs": "C_obs", "B_C_f1": "C_f1"}


def frozen_view_records(prior, all_views):
    """Join the recorded fused-winning indices; never select another view."""
    by_view = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r for r in all_views}
    if len(by_view) != len(all_views):
        raise ValueError("Duplicate saved CAD/candidate/view record.")
    records = []
    for target in prior["associations"]:
        for pair in target["candidate_ranking"]:
            key = pair["cad_id"], pair["object_id"], pair["winning_view_index_1based"]
            row = by_view[key]
            raw = row["geometry_raw"]
            if raw["T_observed_from_cad"] is None or row["coverage_status"] != "aligned":
                raise ValueError(f"Frozen winning view has no saved alignment: {key}")
            coverage = [row[k] for k in ("C_obs", "C_cad", "C_f1")]
            if any(v is None or not isfinite(v) or not 0 <= v <= 1 for v in coverage):
                raise ValueError(f"Frozen winning view has missing/invalid coverage: {key}")
            c_obs, c_cad, c_f1 = coverage
            expected = 2 * c_obs * c_cad / (c_obs + c_cad) if c_obs + c_cad else 0.
            if not isclose(c_f1, expected, rel_tol=0, abs_tol=1e-15):
                raise ValueError(f"Saved harmonic coverage is inconsistent: {key}")
            assert c_obs == row["geometry_fitness"] == raw["registration_fitness"]
            assert raw == pair["geometry_raw_at_fused_view"]
            assert all(row[k] == pair[k] for k in ("C_obs", "C_cad", "C_f1"))
            assert row["normalized_cls"] == pair["normalized_cls_at_fused_view"]
            assert row["fused_score"] == pair["fused_score"]
            assert row["normalized_cls"] == (row["cls_cosine"] + 1) / 2
            records.append(row)
    return records


def rescore(records, geometry_key):
    return [{"cad_id": r["cad_id"], "object_id": r["object_id"],
             "frozen_view_index_1based": r["view_index_1based"],
             "visual_raw": r["cls_cosine"], "visual_score": r["normalized_cls"],
             "geometry_score": r[geometry_key],
             "fused_score": association.fuse_scores(r["normalized_cls"], r[geometry_key]),
             **{key: r[key] for key in ("C_obs", "C_cad", "C_f1")}} for r in records]


def main():
    start = perf_counter()
    input_hashes = verify_artifacts(ABLATION_3)
    source_protocol = read_json(ABLATION_3 / "protocol.json")
    source_paths = [*source_protocol["source_sha256"], "scripts/run_wrist_coverage_ablation.py",
                    "scripts/wrist_coverage_ablation_report.py"]
    source_hashes = {p: association.content_hash(ROOT / p) for p in source_paths}
    assert (association.FUSION_METHOD, association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT) == ("weighted_mean", .5, .5)
    prior = read_json(ABLATION_3 / "B_visible_cad/evaluation.json")
    records = frozen_view_records(prior, read_json(ABLATION_3 / "all_view_scores.json"))
    targets = {t["cad_id"]: t["target_description"] for t in prior["associations"]}
    object_ids = {r["object_id"] for r in records}
    assert len(targets) == len(object_ids) == 6
    assert len(records) == len({(r["cad_id"], r["object_id"]) for r in records}) == 36
    assert all({r["object_id"] for r in records if r["cad_id"] == cad} == object_ids for cad in targets)
    # Identity labels come only from the completed evaluation, after scores exist.
    audit = [{"object_id": t["expected_object_id"], "simulator_instance": t["cad_id"], "status": "matched"}
             for t in prior["per_target"]]
    output_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True)
    save_json(output_dir / "frozen_view_records.json", records)
    results, scoring_times = {}, {}
    for name, geometry_key in VARIANTS.items():
        stage = perf_counter()
        pairs = rescore(records, geometry_key)
        results[name] = evaluate_pairs(name, pairs, targets, audit)
        scoring_times[name] = perf_counter() - stage
        directory = output_dir / name
        directory.mkdir()
        save_json(directory / "evaluation.json", results[name])
        save_csv(directory / "all_pair_scores.csv", pairs)
        print(f"{name}: {results[name]['accuracy']} | mean margins {results[name]['mean_margins']}", flush=True)
    for current, old, margins, old_margins in zip(results["A_C_obs"]["associations"], prior["associations"],
                                                 results["A_C_obs"]["margins"], prior["margins"], strict=True):
        assert current["rankings"]["fused"] == old["rankings"]["fused"]
        assert margins["fused"] == old_margins["fused"]
        old_pairs = {r["object_id"]: r for r in old["candidate_ranking"]}
        assert all(r["fused_score"] == old_pairs[r["object_id"]]["fused_score"] for r in current["candidate_ranking"])
    save_report(output_dir, results, records, scoring_times)
    assert verify_artifacts(ABLATION_3) == input_hashes
    assert all(association.content_hash(ROOT / p) == digest for p, digest in source_hashes.items())
    save_json(output_dir / "protocol.json", {
        "ablation_3_source": str(ABLATION_3.relative_to(ROOT)), "ablation_3_sha256": input_hashes,
        "source_sha256": source_hashes, "dataset": source_protocol["dataset"], "known_cad_ids": targets,
        "view_selection": "Reuse B_visible_cad winning_view_index_1based for each pair; no view reselection",
        "visual_score": "Saved normalized_cls at the frozen fused-winning view; identical for A and B",
        "geometry_scores": VARIANTS, "C_f1": "Use saved C_f1; verify 2*C_obs*C_cad/(C_obs+C_cad)",
        "fusion": "0.5 * saved normalized_cls + 0.5 * geometry_score",
        "geometry_only_scope": "Both scores use the frozen fused-winning view, not the independently maximal fitness view",
        "saved_transforms": "Exact source records in frozen_view_records.json; no transform or coverage recomputation",
        "frozen_record_runtime": "runtime fields in frozen_view_records.json are historical Ablation 3 measurements",
        "tie_policy": "Ascending object ID, unchanged; no one-to-one assignment or confidence gate",
        "ground_truth": "Reuse only Ablation 3 per_target expected_object_id for evaluation",
        "production_pipeline_modified": False, "new_inference_rendering_localization_or_registration_calls": 0,
    })
    save_json(output_dir / "validation.json", {
        "frozen_pair_count": len(records), "all_frozen_views_have_saved_transform_and_coverage": True,
        "all_frozen_records_exactly_match_ablation_3": True, "all_saved_C_f1_match_harmonic_formula": True,
        "A_fused_scores_rankings_and_margins_exactly_reproduce_ablation_3": True,
        "source_and_input_hashes_unchanged": True, "view_reselections": 0,
    })
    save_json(output_dir / "runtime.json", {"scoring_and_evaluation_time_s": scoring_times,
        "total_wall_time_after_imports_s": perf_counter() - start,
        "scope": "Saved-artifact scoring only; total includes input hash checks and output writing, excludes final output hashes"})
    (output_dir / "SHA256SUMS").write_text("".join(
        f"{association.content_hash(p)}  {p.relative_to(output_dir).as_posix()}\n"
        for p in sorted(output_dir.rglob("*")) if p.is_file()), encoding="utf-8")
    print(f"Saved Ablation 3b: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
