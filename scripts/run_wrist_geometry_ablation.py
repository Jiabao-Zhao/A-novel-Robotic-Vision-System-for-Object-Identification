"""Ablation 3: recorded full CAD versus view-consistent visible CAD geometry.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_geometry_ablation

All 504 observed/CAD/view registrations use unchanged 5-mm FPFH/RANSAC/ICP.
"""

import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import open3d as o3d

import cad_object_association as association
from CADPointCloudRegistration import CADPointCloudRegistration
from main import association_conflicts
from scripts.run_wrist_association import evaluate_rankings
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, read_rgb, save_csv, save_json, snapshot_hashes
from scripts.run_wrist_patch_ablation import score_margins, verify_artifacts
from scripts.wrist_visible_geometry import coverage_diagnostics, geometry_from_arrays, prepare_visible_geometry
from scripts.wrist_geometry_ablation_report import save_report


ABLATION_2 = ROOT / "outputs/ablation_2_wrist_association_2026-09-11/20260912T160053225612Z"
OUTPUT_ROOT = ROOT / "outputs/ablation_3_wrist_association_2026-09-11"


def collapse_views(rows):
    """Independent modality maxima; fused score always uses a single shared view."""
    visual = max(rows, key=lambda r: r["cls_cosine"])
    geometry = max(rows, key=lambda r: r["geometry_fitness"])
    fused = max(rows, key=lambda r: r["fused_score"])
    return {"cad_id": fused["cad_id"], "object_id": fused["object_id"],
            "visual_raw": visual["cls_cosine"], "visual_score": visual["normalized_cls"],
            "geometry_score": geometry["geometry_fitness"], "fused_score": fused["fused_score"],
            "visual_winning_view_index_1based": visual["view_index_1based"],
            "geometry_winning_view_index_1based": geometry["view_index_1based"],
            "winning_view_index_1based": fused["view_index_1based"],
            "normalized_cls_at_fused_view": fused["normalized_cls"],
            "geometry_fitness_at_fused_view": fused["geometry_fitness"],
            "geometry_raw_at_fused_view": fused["geometry_raw"],
            **{key: fused[key] for key in ("C_obs", "C_cad", "C_f1", "coverage_status")}}


def evaluate_pairs(name, pairs, targets, audit):
    start = perf_counter()
    associations = []
    for cad_id, description in targets.items():
        rows = [p for p in pairs if p["cad_id"] == cad_id]
        rankings = {mode: [r["object_id"] for r in association.rank_candidates(rows, key)]
                    for mode, key in (("visual", "visual_score"), ("geometry", "geometry_score"), ("fused", "fused_score"))}
        associations.append({"cad_id": cad_id, "target_description": description,
                             "candidate_ranking": association.rank_candidates(rows), "rankings": rankings,
                             "predictions": {mode: ids[0] for mode, ids in rankings.items()},
                             "selected_object_id": rankings["fused"][0], "resolution": "ranking_only"})
    result = evaluate_rankings(associations, audit)
    conflicts = association_conflicts(associations)
    margins = []
    for target, truth in zip(associations, result["per_target"], strict=True):
        target["final_object_id"] = None if conflicts else target["selected_object_id"]
        margins.append({"cad_id": target["cad_id"], "expected_object_id": truth["expected_object_id"],
            **{mode: score_margins(target["candidate_ranking"], truth["expected_object_id"], key)
               for mode, key in (("visual", "visual_raw"), ("geometry", "geometry_score"), ("fused", "fused_score"))}})
    result.update(method=name, associations=associations, margins=margins, conflicts=conflicts,
        association_set_resolution="conflicting" if conflicts else "ranking_only",
        mean_margins={mode: float(np.mean([m[mode]["correct_minus_best_incorrect"] for m in margins]))
                      for mode in ("visual", "geometry", "fused")},
        evaluation_time_s=perf_counter() - start)
    return result


def main():
    start = perf_counter()
    os.chdir(ROOT)
    source_files = ["scripts/run_wrist_geometry_ablation.py", "scripts/wrist_visible_geometry.py",
                    "scripts/wrist_geometry_ablation_report.py", "scripts/run_wrist_input_ablation.py",
                    "scripts/run_wrist_patch_ablation.py", "scripts/run_wrist_view_ablation.py",
                    "scripts/run_wrist_association.py", "cad_object_association.py", "CADPointCloudRegistration.py", "main.py"]
    source_hashes = {name: association.content_hash(ROOT / name) for name in source_files}
    dataset_hashes, input_hashes = snapshot_hashes(), verify_artifacts(ABLATION_2)
    manifest = read_json(DATASET / "manifest.json")
    targets = manifest["targets"]
    localization = read_json(ROOT / manifest["localization_path"])
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    assert len(targets) == len(localization["objects"]) == 6 and localization["frame"] == "camera"
    assert (association.FUSION_METHOD, association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT) == ("weighted_mean", .5, .5)
    assert association.DINO_MODEL == "dinov2_vits14"
    np.testing.assert_array_equal(association.VIEW_DIRECTIONS, manifest["template_camera_directions_cad_xyz"])
    registrar = CADPointCloudRegistration()
    assert registrar.config.voxel_size_m == .005
    output_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (output_dir / "observed_geometry").mkdir(parents=True)
    timing = {"observed_preprocessing_time_s": 0., "cad_visible_preparation_time_s": 0.,
              "new_registration_time_s": 0., "visible_coverage_diagnostic_time_s": 0.,
              "baseline_coverage_diagnostic_time_s": 0., "cls_scoring_time_s": 0., "fusion_time_s": 0.}
    stage = perf_counter()
    with np.load(ABLATION_2 / "features/observed.npz", allow_pickle=False) as saved:
        observed_cls, object_ids = saved["cls_features"], saved["object_ids"].tolist()
    cad_cls = {}
    for cad_id in targets:
        with np.load(ABLATION_2 / "features" / f"{cad_id}.npz", allow_pickle=False) as saved:
            cad_cls[cad_id] = saved["cls_features"]
    timing["cls_feature_loading_time_s"] = perf_counter() - stage
    assert set(object_ids) == {r["object_id"] for r in localization["objects"]}
    observed = {}
    for item in localization["objects"]:
        stage = perf_counter()
        cloud, fpfh = registrar.compute_fpfh(registrar.load_observed_cloud(ROOT / item["pointcloud_path"]), camera_location=[0, 0, 0])
        timing["observed_preprocessing_time_s"] += perf_counter() - stage
        observed[item["object_id"]] = (cloud, fpfh)
        np.savez_compressed(output_dir / "observed_geometry" / f"{item['object_id']}.npz",
                            points=np.asarray(cloud.points), normals=np.asarray(cloud.normals), fpfh=fpfh.data)
    old_pairs = {(r["cad_id"], r["object_id"]): r for r in read_json(ABLATION_2 / "pair_scores.json")}
    all_views, full_pairs, visible_pairs, surface_metadata, full_cache_hashes = [], [], [], {}, {}
    for target_index, cad_id in enumerate(targets, 1):
        original = read_json(DATASET / "results/colored_cpu/association" / f"target_{target_index:03}.json")
        mesh_path = ROOT / library[cad_id]["file_path"]
        full_key = association.cache_key(association.CACHE_VERSION, association.content_hash(mesh_path), asdict(registrar.config), o3d.__version__)
        assert full_key == original["diagnostics"]["cad_geometry_cache_key"]
        cache_path = association.CACHE_DIR / f"{full_key}_geometry.npz"
        full_cache_hashes[str(cache_path)] = association.content_hash(cache_path)
        with np.load(cache_path, allow_pickle=False) as arrays:
            full_cloud, _ = geometry_from_arrays(arrays)
        templates = [read_rgb(DATASET / "templates" / cad_id / f"view_{i:02}.png") for i in range(1, 15)]
        visible_geometry, surface_metadata[cad_id], elapsed = prepare_visible_geometry(
            mesh_path, library[cad_id]["base_color_rgb"], templates, registrar)
        timing["cad_visible_preparation_time_s"] += elapsed
        original_rows = {r["object_id"]: r for r in original["candidate_ranking"]}
        for oid in object_ids:
            cloud, observed_fpfh = observed[oid]
            stage = perf_counter()
            cosines = np.clip(cad_cls[cad_id] @ observed_cls[object_ids.index(oid)], -1, 1)
            timing["cls_scoring_time_s"] += perf_counter() - stage
            np.testing.assert_array_equal(cosines, old_pairs[cad_id, oid]["cls_view_cosines"])
            old = original_rows[oid]
            assert old["geometry_raw"] == old_pairs[cad_id, oid]["geometry_raw"]
            assert old["geometry_score"] == old_pairs[cad_id, oid]["geometry_score"]
            stage = perf_counter()
            full_coverage = coverage_diagnostics(full_cloud, cloud, old["geometry_raw"])
            timing["baseline_coverage_diagnostic_time_s"] += perf_counter() - stage
            cls_max = float(cosines.max())
            visual_score = association.normalize_visual_score(cls_max)
            full_pairs.append({"cad_id": cad_id, "object_id": oid, "visual_raw": cls_max, "visual_score": visual_score,
                               "geometry_score": old["geometry_score"], "geometry_raw": old["geometry_raw"],
                               "fused_score": association.fuse_scores(visual_score, old["geometry_score"]),
                               "visual_winning_view_index_1based": int(cosines.argmax()) + 1,
                               "winning_view_index_1based": None, **full_coverage})
            rows = []
            for index, ((visible_cloud, visible_fpfh), cosine) in enumerate(zip(visible_geometry, cosines, strict=True), 1):
                stage = perf_counter()
                raw = registrar.match_fpfh(visible_cloud, visible_fpfh, cloud, observed_fpfh)
                registration_time = perf_counter() - stage
                timing["new_registration_time_s"] += registration_time
                stage = perf_counter()
                coverage = coverage_diagnostics(visible_cloud, cloud, raw)
                coverage_time = perf_counter() - stage
                timing["visible_coverage_diagnostic_time_s"] += coverage_time
                stage = perf_counter()
                normalized = association.normalize_visual_score(float(cosine))
                fused = association.fuse_scores(normalized, raw["registration_fitness"])
                timing["fusion_time_s"] += perf_counter() - stage
                rows.append({"cad_id": cad_id, "object_id": oid, "view_index_1based": index,
                             "cls_cosine": float(cosine), "normalized_cls": normalized,
                             "geometry_fitness": raw["registration_fitness"],
                             "observed_to_visible_cad_rmse_m": raw["observed_to_cad_rmse_m"],
                             "correspondence_count": raw["correspondence_count"],
                             "observed_point_count": len(cloud.points), "visible_cad_point_count": len(visible_cloud.points),
                             "geometry_raw": raw, **coverage, "fused_score": fused,
                             "runtime": {"registration_time_s": registration_time, "coverage_time_s": coverage_time}})
            all_views.extend(rows)
            visible_pairs.append(collapse_views(rows))
        print(f"{cad_id}: 84 exhaustive view registrations; visible CAD points "
              f"{min(m['downsampled_point_count'] for m in surface_metadata[cad_id])}–"
              f"{max(m['downsampled_point_count'] for m in surface_metadata[cad_id])}", flush=True)
        save_json(output_dir / "cad_visible_geometry.json", surface_metadata)
    assert len(all_views) == 504 and len(full_pairs) == len(visible_pairs) == 36
    save_json(output_dir / "all_view_scores.json", all_views)
    save_csv(output_dir / "all_view_scores.csv", [{**{k: v for k, v in r.items() if k not in ("geometry_raw", "runtime")},
             "registration_status": r["geometry_raw"]["status"], **r["runtime"]} for r in all_views])
    audit = read_json(DATASET / "identity_audit.json")
    results = {"A_full_cad": evaluate_pairs("A_full_cad", full_pairs, targets, audit),
               "B_visible_cad": evaluate_pairs("B_visible_cad", visible_pairs, targets, audit)}
    previous = read_json(ABLATION_2 / "cls_only/evaluation.json")
    for current, old in zip(results["A_full_cad"]["associations"], previous["associations"], strict=True):
        assert current["rankings"] == old["rankings"]
        old_rows = {r["object_id"]: r for r in old["candidate_ranking"]}
        for row in current["candidate_ranking"]:
            assert all(row[k] == old_rows[row["object_id"]][k] for k in ("visual_raw", "geometry_score", "fused_score"))
    for name, result in results.items():
        directory = output_dir / name
        directory.mkdir()
        save_json(directory / "evaluation.json", result)
        save_csv(directory / "all_pair_scores.csv", [{k: v for k, v in row.items() if not isinstance(v, dict)}
                 for target in result["associations"] for row in target["candidate_ranking"]])
        print(f"{name}: {result['accuracy']} | mean margins {result['mean_margins']}", flush=True)
    save_report(output_dir, results, full_pairs, visible_pairs, all_views, timing)
    prior_protocol = read_json(ABLATION_2 / "protocol.json")
    save_json(output_dir / "protocol.json", {
        "dataset": str(DATASET.relative_to(ROOT)), "cls_source": str(ABLATION_2.relative_to(ROOT)),
        "known_cad_ids": targets, "encoder": association.DINO_MODEL, "encoder_source": association.DINO_REPO,
        "checkpoint_sha256": prior_protocol["checkpoint_sha256"], "input_rgb": "Exact masked white-background Ablation 1/2 inputs",
        "visual_features": "Reuse only saved normalized CLS features; all 14 view cosines reproduce Ablation 2 exactly",
        "surface_generation": "First ray hit per existing orthographic RGB pixel; reverse center/radius fitting to native CAD meters",
        "surface_normals": "Barycentrically interpolated CAD vertex normals, matching existing full-CAD sampling; no orientation override",
        "geometry_config": asdict(registrar.config), "correspondence_radius_m": 1.5 * registrar.config.voxel_size_m,
        "registration": "Unchanged match_fpfh: 10000 RANSAC iterations/confidence 0.999, point-to-point ICP 30 iterations, seed 0",
        "method_A": "0.5 * normalize(max_v CLS(v)) + 0.5 * recorded_full_CAD_fitness",
        "method_B": "max_v [0.5 * normalize(CLS(v)) + 0.5 * visible_CAD_fitness(v)]",
        "modality_rankings": "Visual max_v CLS; B geometry-only max_v visible fitness; B fused max of same-view sum",
        "coverage": "C_obs and C_cad are fractions of actual registration point sets within 7.5 mm after alignment; C_f1 is harmonic mean",
        "failed_alignment_coverage": "C_obs retains zero fitness; C_cad/C_f1 are null when no transform exists",
        "rmse": "Unchanged registrar.observed_to_cad_rmse; retains its additional 5-mm voxel pass after transformation",
        "frames": {"observations": "camera, meters", "CAD_clouds": "native CAD, meters", "T_observed_from_cad": "CAD to observation camera"},
        "view_consistency_scope": "Same rendered-visible surface and RGB view; registration remains unrestricted rigid FPFH/RANSAC/ICP",
        "tie_policy": "First view index for equal view scores; ascending object ID for candidate ties",
        "production_pipeline_modified": False, "coverage_used_for_prediction": False,
        "source_sha256": source_hashes, "dataset_sha256": dataset_hashes, "ablation_2_sha256": input_hashes,
        "baseline_full_geometry_cache_sha256": full_cache_hashes,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "open3d": o3d.__version__,
                        **{key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
    })
    assert snapshot_hashes() == dataset_hashes and verify_artifacts(ABLATION_2) == input_hashes
    assert all(association.content_hash(ROOT / path) == digest for path, digest in source_hashes.items())
    assert all(association.content_hash(path) == digest for path, digest in full_cache_hashes.items())
    save_json(output_dir / "validation.json", {"all_84_rgb_templates_unchanged": True, "all_cls_cosines_match_ablation_2": True,
        "A_scores_and_rankings_reproduce_recorded_baseline": True, "C_obs_matches_registration_fitness_and_correspondences": True,
        "source_and_input_hashes_unchanged": True, "geometry_voxel_size_m": .005, "view_registration_count": len(all_views),
        "failed_registration_count": sum(r["geometry_raw"]["status"] != "matched" for r in all_views),
        "geometry_cache_view_count": sum(len(m) for m in surface_metadata.values()),
        "geometry_descriptor_cache_hits": sum(m["cache_hit"] for rows in surface_metadata.values() for m in rows)})
    save_json(output_dir / "runtime.json", {"total_wall_time_after_imports_s": perf_counter() - start, "measured_stages": timing,
        "evaluation_time_s": {name: r["evaluation_time_s"] for name, r in results.items()},
        "historical_full_CAD_registration_time_s": read_json(DATASET / "results/colored_cpu/evaluation.json")["runtime"]["pairwise_matching_time_s"],
        "scope": "A registrations reused, B registrations newly measured; excludes Python imports and final output hashes; all other preparation/QA/IO included in total"})
    (output_dir / "SHA256SUMS").write_text("".join(f"{association.content_hash(p)}  {p.relative_to(output_dir).as_posix()}\n"
        for p in sorted(output_dir.rglob("*")) if p.is_file()), encoding="utf-8")
    print(f"Saved Ablation 3: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
