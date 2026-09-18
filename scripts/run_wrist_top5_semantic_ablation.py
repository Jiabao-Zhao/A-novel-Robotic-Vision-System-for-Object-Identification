"""Top-five semantic mean versus max-view CLS, using saved six-object features.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_top5_semantic_ablation

Experimental scorer only. No encoder, rendering, geometry or production changes.
"""

import os
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, save_csv, save_json, snapshot_hashes
from scripts.run_wrist_patch_ablation import evaluate_variant, verify_artifacts
from scripts.run_wrist_view_ablation import ABLATION_2


OUTPUT_ROOT = ROOT / "outputs/top5_semantic_wrist_association_2026-09-11"
TOP_K = 5


def template_rotation(direction):
    """CAD vectors -> renderer camera vectors: x right, y down, z forward.

    The orthographic renderer's image rows run opposite its `up` vector,
    and its rays point opposite the CAD-to-camera viewing direction.
    This is a discrete template orientation, not an estimated workspace pose.
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    right, up = association.CADPointCloudRegistration.table_basis(direction)
    return np.stack((right, -up, -direction))


def score_view_similarities(cad_id, similarities, template_rotations=None):
    """Mean the best five individual cosines; retain the single best template."""
    scores = np.asarray(similarities, dtype=float)
    if template_rotations is None:
        template_rotations = [template_rotation(d) for d in association.VIEW_DIRECTIONS]
    rotations = np.asarray(template_rotations, dtype=float)
    if scores.shape != (len(rotations),) or len(scores) < TOP_K or rotations.shape != (len(scores), 3, 3):
        raise ValueError("Expected one similarity and one 3x3 rotation per template, with at least five views.")
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
        raise ValueError("Per-view cosines must be finite and in [-1, 1].")
    order = np.argsort(-scores, kind="stable")  # First saved view wins exact ties.
    top = order[:TOP_K]
    best = int(order[0])
    return {
        "per_view_cosines": scores.tolist(),
        "semantic_score": float(scores[top].mean()),
        "max_view_score": float(scores[best]),
        "top5_view_indices_1based": (top + 1).tolist(),
        "top5_template_ids": [f"{cad_id}/view_{i + 1:02}" for i in top],
        "top5_cosines": scores[top].tolist(),
        "best_view_index_1based": best + 1,
        "best_template_id": f"{cad_id}/view_{best + 1:02}",
        "best_template_rotation_camera_from_cad": rotations[best].tolist(),
    }


def save_report(output, evaluations, changes, targets, runtime):
    before, after = evaluations["max_view"], evaluations["top5_mean"]
    gear_target = next(t for t in after["associations"] if t["cad_id"] == "Gear_Large")
    gear_truth = next(t["expected_object_id"] for t in changes if t["cad_id"] == "Gear_Large")
    gear_pair = next(p for p in gear_target["candidate_ranking"] if p["object_id"] == gear_truth)
    gear_cosines = ", ".join(f"{score:.3f}" for score in gear_pair["top5_cosines"])
    lines = ["# Top-five semantic aggregation: fixed six-object test", "",
             f"Visual accuracy changes from {before['accuracy']['visual']['correct']}/6 to "
             f"{after['accuracy']['visual']['correct']}/6, and fused accuracy from "
             f"{before['accuracy']['fused']['correct']}/6 to {after['accuracy']['fused']['correct']}/6. "
             "The per-target table below distinguishes prediction changes from score shifts.", "",
             "Only aggregation changes: max over 14 raw CLS cosines versus the arithmetic mean of the five largest. "
             "The individual best template remains the raw-cosine argmax. All 36 CAD/candidate pairs are retained.", "",
             "Same saved localization-masked inputs, white 224x224 letterbox, DINOv2-small CLS features, "
             "14 original colored orthographic templates, recorded full-CAD geometry, and 0.5/0.5 fusion. "
             "The preceding exact-pose diagnostic has only one template per pair and is not the source for this test.", "",
             "| Method | Visual top-1 | Fused top-1 | Mean visual margin | Mean fused margin |",
             "|---|---:|---:|---:|---:|"]
    for name, result in evaluations.items():
        lines.append(f"| {name} | {result['accuracy']['visual']['correct']}/6 | "
                     f"{result['accuracy']['fused']['correct']}/6 | "
                     f"{result['margin_summary']['visual_raw']['mean_correct_minus_best_incorrect']:+.6f} | "
                     f"{result['margin_summary']['fused']['mean_correct_minus_best_incorrect']:+.6f} |")
    lines += ["", "Margins are correct minus strongest incorrect candidate. A positive margin means separation. "
              "The top-five mean cannot exceed the maximum; lower absolute scores alone are not evidence of worse discrimination.", "",
              "| Target | Correct score: max → top-five | Visual prediction: max → top-five | Visual margin: max → top-five | Fused prediction: max → top-five | Fused margin: max → top-five |",
              "|---|---:|---|---:|---|---:|"]
    for row in changes:
        lines.append(f"| {targets[row['cad_id']]} | {row['max_view_correct_score']:.6f} → {row['top5_mean_correct_score']:.6f} | "
                     f"{row['max_view_visual_prediction']} → {row['top5_mean_visual_prediction']} | "
                     f"{row['max_view_visual_margin']:+.6f} → {row['top5_mean_visual_margin']:+.6f} | "
                     f"{row['max_view_fused_prediction']} → {row['top5_mean_fused_prediction']} | "
                     f"{row['max_view_fused_margin']:+.6f} → {row['top5_mean_fused_margin']:+.6f} |")
    lines += ["", "Object IDs: 001 pulley; 002 large gear; 003 blue block; 004 red block; 005 medium gear; 006 rectangular pin.", "",
              f"The large gear's correct pair has top-five cosines {gear_cosines}: "
              "its mean includes substantially weaker views. Both cube targets have five equal highest cosines, "
              "so their correct-pair scores remain unchanged. This illustrates how the fixed template set "
              "affects aggregation; no repeated or symmetric views were removed.", "",
              "## Best-template return", "",
              "Every pair saves all 14 cosines, the five contributing template IDs/scores, the top-five mean, "
              "the best-template ID, and its 3x3 `R_template_camera_from_cad`. "
              "This rotation maps CAD vectors into the renderer camera frame (x right, y down, z forward); "
              "its transpose is the inverse. It is a discrete template orientation, without translation, "
              "not a verified pose of the observed object. It does not initialize or modify registration. "
              "Changing aggregation does not change the best template for any fixed CAD/candidate pair. "
              "Each evaluation's per-target `semantic_match` returns the visual winner and its score, "
              "best-template ID and rotation separately from the fused decision.", "",
              "## Protocol and runtime", "",
              "Raw semantic score is the mean of five individual cosines, not cosine to an averaged descriptor. "
              "Fusion remains `0.5 * ((semantic_score + 1) / 2) + 0.5 * recorded_geometry_score`. "
              "Independent rankings, object-ID tie handling and set-conflict handling are unchanged. "
              "No thresholds, view deduplication, pruning, tuning, SAM, new views or production changes.", "",
              f"Saved CLS feature loading: {runtime['feature_loading_time_s']:.6f}s. "
              f"All 504 cosines plus 36 top-five/template returns: {runtime['semantic_scoring_time_s']:.6f}s. "
              f"Both variants' evaluation: {runtime['evaluation_time_s']:.6f}s. "
              "These are CPU rescoring times; encoder, rendering and registration were not run.", "",
              "This implements only the requested score formulation. The encoder and template setup are unchanged; "
              "it is not a faithful reproduction of a paper's full method or evidence of general accuracy.", "",
              "[All 36 pairs, scores and rotations](pair_scores.json), [all 504 view scores](per_view_scores.csv), "
              "[template IDs, images and rotations](templates.json), [prediction/margin changes](target_changes.csv), "
              "[max-view rankings](max_view/evaluation.json), [top-five rankings](top5_mean/evaluation.json), "
              "[protocol](protocol.json), [validation](validation.json), [runtime](runtime.json).", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = perf_counter()
    os.chdir(ROOT)
    fixed_hashes, prior_hashes = snapshot_hashes(), verify_artifacts(ABLATION_2)
    source_paths = ["cad_object_association.py", "CADPointCloudRegistration.py", "main.py",
                    "scripts/run_wrist_input_ablation.py", "scripts/run_wrist_patch_ablation.py",
                    "scripts/run_wrist_association.py", "scripts/run_wrist_top5_semantic_ablation.py",
                    "tests/test_wrist_top5_semantic_ablation.py"]
    source_hashes = {p: association.content_hash(ROOT / p) for p in source_paths}
    protocol = read_json(ABLATION_2 / "protocol.json")
    targets = protocol["known_cad_ids"]
    assert len(targets) == 6 and association.DINO_MODEL == protocol["encoder"] == "dinov2_vits14"
    assert association.DINO_REPO == protocol["encoder_source"] and association.IMAGE_SIZE == 224
    assert (association.FUSION_METHOD, association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT) == ("weighted_mean", .5, .5)
    np.testing.assert_array_equal(association.VIEW_DIRECTIONS,
        read_json(DATASET / "manifest.json")["template_camera_directions_cad_xyz"])
    prior = read_json(ABLATION_2 / "pair_scores.json")
    baseline = read_json(ABLATION_2 / "cls_only/evaluation.json")
    stage = perf_counter()
    with np.load(ABLATION_2 / "features/observed.npz") as saved:
        observed, object_ids = saved["cls_features"], saved["object_ids"].tolist()
    cads = {}
    for cid in targets:
        with np.load(ABLATION_2 / "features" / f"{cid}.npz") as saved:
            cads[cid] = saved["cls_features"]
    runtime = {"feature_loading_time_s": perf_counter() - stage}
    assert observed.shape == (6, 384)
    for features in [observed, *cads.values()]:
        assert features.shape[1] == 384
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-12, rtol=0)
    stage = perf_counter()
    pairs, max_delta = [], 0.
    for old in prior:
        cid, oid = old["cad_id"], old["object_id"]
        cosines = np.clip(cads[cid] @ observed[object_ids.index(oid)], -1, 1)
        max_delta = max(max_delta, float(np.max(np.abs(cosines - old["cls_view_cosines"]))))
        scored = score_view_similarities(cid, cosines)
        assert scored["best_view_index_1based"] == old["selected_view_index_1based"]
        pairs.append({"cad_id": cid, "object_id": oid, **scored,
                      "geometry_raw": old["geometry_raw"], "geometry_score": old["geometry_score"]})
    runtime["semantic_scoring_time_s"] = perf_counter() - stage
    assert max_delta < 1e-12 and len({(p['cad_id'], p['object_id']) for p in pairs}) == 36
    stage = perf_counter()
    evaluations = {}
    for name, key in (("max_view", "max_view_score"), ("top5_mean", "semantic_score")):
        # Reuse the existing CLS evaluation/ranking/conflict code with only its raw score replaced.
        rows = [dict(p, cls_score=p[key], runtime={"cls_comparison_time_s": 0.}) for p in pairs]
        result = evaluate_variant("cls_only", rows, targets, read_json(DATASET / "identity_audit.json"),
            {"observed_image_loading_time_s": 0., "observed_joint_feature_extraction_time_s": 0.,
             "observed_mask_preparation_time_s": 0.})
        result["variant"] = name
        result["runtime"] = {"fusion_ranking_evaluation_time_s": result["runtime"]["variant_fusion_ranking_evaluation_time_s"]}
        for target in result["associations"]:
            winner = next(row for row in target["candidate_ranking"] if row["object_id"] == target["predictions"]["visual"])
            target["semantic_match"] = {"object_id": winner["object_id"], "semantic_score": winner["cls_score"],
                "best_template_id": winner["best_template_id"],
                "best_template_rotation_camera_from_cad": winner["best_template_rotation_camera_from_cad"]}
        evaluations[name] = result
    runtime["evaluation_time_s"] = perf_counter() - stage
    assert evaluations["max_view"]["accuracy"] == baseline["accuracy"]
    for new, old in zip(evaluations["max_view"]["associations"], baseline["associations"], strict=True):
        assert new["rankings"] == old["rankings"]
    assert evaluations["max_view"]["margins"] == baseline["margins"]
    changes = []
    for i, cid in enumerate(targets):
        change = {"cad_id": cid, "expected_object_id": baseline["per_target"][i]["expected_object_id"]}
        for name, result in evaluations.items():
            change[f"{name}_correct_score"] = result["margins"][i]["visual_raw"]["correct_score"]
            for mode, margin_key in (("visual", "visual_raw"), ("fused", "fused")):
                change[f"{name}_{mode}_prediction"] = result["associations"][i]["predictions"][mode]
                change[f"{name}_{mode}_margin"] = result["margins"][i][margin_key]["correct_minus_best_incorrect"]
        for mode in ("visual", "fused"):
            change[f"{mode}_margin_delta"] = change[f"top5_mean_{mode}_margin"] - change[f"max_view_{mode}_margin"]
            change[f"{mode}_prediction_changed"] = change[f"top5_mean_{mode}_prediction"] != change[f"max_view_{mode}_prediction"]
        changes.append(change)
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    save_json(output / "pair_scores.json", pairs)
    save_csv(output / "target_changes.csv", changes)
    save_csv(output / "per_view_scores.csv", [{"cad_id": p["cad_id"], "object_id": p["object_id"],
        "view_index_1based": i, "template_id": f"{p['cad_id']}/view_{i:02}", "cls_cosine": score,
        "contributes_to_top5": i in p["top5_view_indices_1based"], "best_template": i == p["best_view_index_1based"]}
        for p in pairs for i, score in enumerate(p["per_view_cosines"], 1)])
    save_json(output / "templates.json", [{"template_id": f"{cid}/view_{i:02}", "view_index_1based": i,
        "image_path": str((DATASET / "templates" / cid / f"view_{i:02}.png").relative_to(ROOT)),
        "camera_direction_in_cad_frame": (direction / np.linalg.norm(direction)).tolist(),
        "R_template_camera_from_cad": template_rotation(direction).tolist()}
        for cid in targets for i, direction in enumerate(association.VIEW_DIRECTIONS, 1)])
    for name, result in evaluations.items():
        (output / name).mkdir()
        save_json(output / name / "evaluation.json", result)
    save_json(output / "protocol.json", {
        "dataset": str(DATASET.relative_to(ROOT)), "source": str(ABLATION_2.relative_to(ROOT)),
        "encoder": protocol["encoder"], "encoder_source": protocol["encoder_source"],
        "checkpoint_sha256": protocol["checkpoint_sha256"], "top_k": TOP_K,
        "score": "Arithmetic mean of five highest raw per-view CLS cosines; no descriptor averaging",
        "best_template": "Single maximum-cosine template, first index on exact ties; separate from aggregate",
        "rotation": "R_template_camera_from_cad = stack(right, -up, -direction); column vectors; x right/y down/z forward",
        "rotation_scope": "Discrete orthographic template orientation only; no observed-pose guarantee or geometry initialization",
        "inputs": "Exact saved Ablation 2 normalized CLS features; localization masks; white 224x224; original 14 templates",
        "geometry": "All 36 original full-CAD records reused unchanged",
        "fusion": "0.5 * ((semantic_score + 1) / 2) + 0.5 * recorded_geometry_score",
        "faithful_paper_reproduction": False, "production_modified": False,
        "encoder_rendering_localization_geometry_rerun": False,
        "source_sha256": source_hashes, "dataset_sha256": fixed_hashes, "ablation_2_sha256": prior_hashes,
    })
    assert snapshot_hashes() == fixed_hashes and verify_artifacts(ABLATION_2) == prior_hashes
    assert all(association.content_hash(ROOT / p) == digest for p, digest in source_hashes.items())
    save_json(output / "validation.json", {"pair_count": 36, "per_view_score_count": 504,
        "max_cosine_delta_vs_saved_ablation_2": max_delta, "baseline_rankings_margins_reproduced": True,
        "best_template_unchanged_pair_count": 36, "saved_inputs_features_geometry_and_production_unchanged": True})
    runtime["wall_time_before_report_s"] = perf_counter() - start
    save_json(output / "runtime.json", runtime)
    save_report(output, evaluations, changes, targets, runtime)
    (output / "SHA256SUMS").write_text("".join(f"{association.content_hash(p)}  {p.relative_to(output).as_posix()}\n"
        for p in sorted(output.rglob("*")) if p.is_file()), encoding="utf-8")
    print({name: result["accuracy"] for name, result in evaluations.items()}, flush=True)
    print(output, flush=True)
    return output


if __name__ == "__main__":
    main()
