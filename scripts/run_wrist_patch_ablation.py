"""Ablation 2: CLS, foreground patch matching, and their fixed 50/50 mean.

Run from the repository root in the existing association environment:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_patch_ablation

Reads the exact masked inputs from Ablation 1. Production association is untouched.
"""

import os
import platform
import shutil
from datetime import datetime, timezone
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d
import torch

import cad_object_association as association
from main import association_conflicts
from scripts.run_wrist_association import evaluate_rankings
from scripts.run_wrist_input_ablation import (
    ROOT, DATASET, read_json, read_rgb, save_csv, save_image, save_json, snapshot_hashes,
)
from scripts.wrist_patch_ablation_report import save_report


ABLATION_1 = ROOT / "outputs/ablation_1_wrist_association_2026-09-11/20260912T153218768705Z"
OUTPUT_ROOT = ROOT / "outputs/ablation_2_wrist_association_2026-09-11"
PATCH_SIZE = 14
GRID_SIZE = 16
VARIANTS = {"cls_only": "CLS only", "patch_only": "Patch only", "cls_patch": "0.5 CLS + 0.5 patch"}


def verify_artifacts(directory):
    hashes = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        hashes[name] = association.content_hash(directory / name)
        if hashes[name] != expected:
            raise ValueError(f"Artifact hash mismatch: {directory / name}")
    return hashes


def letterbox_mask(mask):
    """Same RGB resize dimensions/offset; binary nearest-neighbor, zero padding."""
    height, width = mask.shape
    scale = association.IMAGE_SIZE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((association.IMAGE_SIZE, association.IMAGE_SIZE), dtype=np.uint8)
    y, x = (association.IMAGE_SIZE - size[1]) // 2, (association.IMAGE_SIZE - size[0]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    return canvas


def patch_occupancy(mask):
    """Row-major 16x16 grid: fraction of object pixels within each 14x14 patch."""
    mask = np.asarray(mask)
    if mask.shape != (224, 224) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Patch occupancy requires a binary 224x224 object mask.")
    return mask.reshape(GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE).mean(axis=(1, 3))


def extract_tokens(images):
    """Final layer-normalized tokens, followed by unit L2 norm per descriptor."""
    model = association.load_dino_encoder()
    tensors = []
    for image in images:
        canvas = association.letterbox_rgb(image)
        normalized = (canvas.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)))
    cls, patches = [], []
    with torch.inference_mode():
        for start in range(0, len(tensors), 8):
            batch = torch.stack(tensors[start:start + 8]).to(next(model.parameters()).device)
            output = model.forward_features(batch)
            cls.append(output["x_norm_clstoken"].cpu().numpy())
            patches.append(output["x_norm_patchtokens"].cpu().numpy())
    cls = np.concatenate(cls).astype(np.float64)
    patches = np.concatenate(patches).astype(np.float64)
    if patches.shape != (len(images), 256, 384) or cls.shape != (len(images), 384):
        raise ValueError(f"Expected DINOv2-S/14 tokens; got {cls.shape}, {patches.shape}.")
    for features in (cls, patches):
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        if not np.isfinite(features).all() or np.any(norms == 0):
            raise ValueError("DINO descriptors must be finite and nonzero.")
        features /= norms
    return cls, patches.reshape(len(images), GRID_SIZE, GRID_SIZE, 384)


def cad_renderer_masks(cad_path):
    """Ray-hit masks using the existing renderer's exact mesh fitting and rays.

    This isolated mask reader repeats only its ray construction because the
    production renderer returns RGB alone and must remain unchanged for Ablation 2.
    The caller verifies all RGB renders and ray masks against the saved templates.
    """
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    center = mesh.get_axis_aligned_bounding_box().get_center()
    radius = np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1).max()
    mesh.translate(-center).scale(1 / radius, center=(0, 0, 0))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    grid_x, grid_y = np.meshgrid(np.linspace(-1.1, 1.1, association.IMAGE_SIZE),
                                np.linspace(1.1, -1.1, association.IMAGE_SIZE))
    masks = []
    for direction in association.VIEW_DIRECTIONS:
        direction = direction / np.linalg.norm(direction)
        right, up = association.CADPointCloudRegistration.table_basis(direction)
        origins = 3 * direction + grid_x[..., None] * right + grid_y[..., None] * up
        rays = np.concatenate((origins, np.broadcast_to(-direction, origins.shape)), axis=-1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        masks.append(np.isfinite(hit["t_hit"].numpy()).astype(np.uint8))
    return np.stack(masks)


def score_pair(observed_cls, observed_patches, observed_weights, cad_cls, cad_patches, cad_weights):
    start = perf_counter()
    cls_cosines = np.clip(cad_cls @ observed_cls, -1, 1)
    view = int(cls_cosines.argmax())  # Exact ties choose the first saved view.
    cls_time = perf_counter() - start
    start = perf_counter()
    observed = observed_patches.reshape(256, -1)
    weights = observed_weights.reshape(256)
    cad = cad_patches[view].reshape(256, -1)
    foreground = cad_weights[view].reshape(256) > 0
    support = weights > 0
    if not support.any() or not foreground.any():
        raise ValueError("Patch matching requires observed and CAD foreground support.")
    similarities = np.clip(observed[support] @ cad[foreground].T, -1, 1)
    maxima = similarities.max(axis=1)
    patch_score = float(np.average(maxima, weights=weights[support]))
    return {"cls_view_cosines": cls_cosines.tolist(), "selected_view_index_1based": view + 1,
            "cls_score": float(cls_cosines[view]), "patch_score": patch_score,
            "cls_patch_score": float(.5 * cls_cosines[view] + .5 * patch_score),
            "observed_foreground_patch_count": int(support.sum()),
            "selected_cad_foreground_patch_count": int(foreground.sum()),
            "observed_occupancy_weight_sum": float(weights.sum()),
            "runtime": {"cls_comparison_time_s": cls_time, "patch_matching_time_s": perf_counter() - start}}


def score_margins(rows, expected, key):
    """Correct minus strongest incorrect, plus all five pairwise differences."""
    correct = next(row for row in rows if row["object_id"] == expected)
    incorrect = [row for row in rows if row["object_id"] != expected]
    strongest = association.rank_candidates(incorrect, key)[0]
    return {"correct_score": correct[key], "strongest_incorrect_object_id": strongest["object_id"],
            "strongest_incorrect_score": strongest[key],
            "correct_minus_best_incorrect": correct[key] - strongest[key],
            "correct_minus_mean_incorrect": correct[key] - float(np.mean([r[key] for r in incorrect])),
            "correct_minus_each_incorrect": {row["object_id"]: correct[key] - row[key] for row in incorrect}}


def evaluate_variant(name, pairs, targets, identity_audit, shared_runtime):
    start = perf_counter()
    key = {"cls_only": "cls_score", "patch_only": "patch_score", "cls_patch": "cls_patch_score"}[name]
    results = []
    for cad_id, description in targets.items():
        rows = []
        for pair in pairs:
            if pair["cad_id"] != cad_id:
                continue
            normalized = association.normalize_visual_score(pair[key])
            rows.append({**pair, "visual_raw": pair[key], "visual_score": normalized,
                         "fused_score": association.fuse_scores(normalized, pair["geometry_score"])})
        rankings = {mode: [r["object_id"] for r in association.rank_candidates(rows, score)]
                    for mode, score in (("visual", "visual_score"), ("geometry", "geometry_score"),
                                        ("fused", "fused_score"))}
        results.append({"cad_id": cad_id, "target_description": description,
                        "candidate_ranking": association.rank_candidates(rows), "rankings": rankings,
                        "predictions": {mode: ids[0] for mode, ids in rankings.items()},
                        "selected_object_id": rankings["fused"][0], "resolution": "ranking_only"})
    evaluation = evaluate_rankings(results, identity_audit)
    conflicts = association_conflicts(results)
    margins = []
    for result, target in zip(results, evaluation["per_target"], strict=True):
        result["final_object_id"] = None if conflicts else result["selected_object_id"]
        expected = target["expected_object_id"]
        margins.append({"cad_id": result["cad_id"], "expected_object_id": expected,
                        **{mode: score_margins(result["candidate_ranking"], expected, score)
                           for mode, score in (("visual_raw", "visual_raw"), ("visual_normalized", "visual_score"),
                                               ("fused", "fused_score"))}})
    summary = {mode: {
        "mean_correct_minus_best_incorrect": float(np.mean([m[mode]["correct_minus_best_incorrect"] for m in margins])),
        "positive_margin_target_count": sum(m[mode]["correct_minus_best_incorrect"] > 0 for m in margins),
        "mean_correct_minus_mean_incorrect": float(np.mean([m[mode]["correct_minus_mean_incorrect"] for m in margins])),
    } for mode in ("visual_raw", "visual_normalized", "fused")}
    uses_patches = name != "cls_only"
    runtime = {"observed_image_loading_time_s": shared_runtime["observed_image_loading_time_s"],
               "observed_joint_feature_extraction_time_s": shared_runtime["observed_joint_feature_extraction_time_s"],
               "observed_mask_preparation_time_s": shared_runtime["observed_mask_preparation_time_s"] if uses_patches else 0.,
               "cls_comparison_time_s": sum(p["runtime"]["cls_comparison_time_s"] for p in pairs),
               "patch_matching_time_s": sum(p["runtime"]["patch_matching_time_s"] for p in pairs) if uses_patches else 0.,
               "variant_fusion_ranking_evaluation_time_s": perf_counter() - start,
               "geometry_recomputation_time_s": 0.}
    runtime["accounted_warm_runtime_s"] = sum(runtime.values())
    runtime["scope"] = ("Sum of measured required stages with CAD features/masks ready; shared stages reused, "
                        "not independent end-to-end reruns. Excludes setup, verification and file saving.")
    evaluation.update(variant=name, associations=results, margins=margins, margin_summary=summary,
                      association_set_resolution="conflicting" if conflicts else "ranking_only",
                      conflicts=conflicts, runtime=runtime)
    return evaluation


def main():
    start = perf_counter()
    os.chdir(ROOT)
    fixed_hashes = snapshot_hashes()
    ablation_1_hashes = verify_artifacts(ABLATION_1)
    source_paths = [ROOT / name for name in (
        "cad_object_association.py", "CADPointCloudRegistration.py", "helper_function.py", "main.py",
        "scripts/run_wrist_input_ablation.py", "scripts/run_wrist_association.py",
        "visualize_localization_projection.py", "scripts/run_wrist_patch_ablation.py",
        "scripts/wrist_patch_ablation_report.py")]
    source_hashes = {str(path.relative_to(ROOT)): association.content_hash(path) for path in source_paths}
    assert association.DINO_MODEL == "dinov2_vits14" and association.IMAGE_SIZE == 224
    assert (association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT, association.FUSION_METHOD) == (.5, .5, "weighted_mean")
    manifest = read_json(DATASET / "manifest.json")
    np.testing.assert_array_equal(association.VIEW_DIRECTIONS, manifest["template_camera_directions_cad_xyz"])
    baseline = read_json(ABLATION_1 / "masked/evaluation.json")
    records = baseline["inputs"]
    object_ids = [r["object_id"] for r in records]
    assert len(set(object_ids)) == 6 and len(manifest["targets"]) == 6
    output_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (output_dir / "features").mkdir(parents=True)
    (output_dir / "inputs").mkdir()
    (output_dir / "masks").mkdir()
    timing = {}
    stage = perf_counter()
    images = [read_rgb(ABLATION_1 / "masked/inputs" / f"{oid}.png") for oid in object_ids]
    assert all(image.shape == (224, 224, 3) for image in images)
    timing["observed_image_loading_time_s"] = perf_counter() - stage
    stage = perf_counter()
    masks = []
    for record in records:
        full = cv2.imread(str(ABLATION_1 / "masked/masks" / f"{record['object_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if full is None:
            raise ValueError(f"Missing Ablation 1 mask: {record['object_id']}")
        assert set(np.unique(full)) <= {0, 255}
        x1, y1, x2, y2 = record["crop_bbox_2d_xyxy"]
        masks.append(letterbox_mask(full[y1:y2 + 1, x1:x2 + 1] // 255))
    observed_weights = np.stack([patch_occupancy(mask) for mask in masks])
    timing["observed_mask_preparation_time_s"] = perf_counter() - stage
    stage = perf_counter()
    model = association.load_dino_encoder()
    assert str(next(model.parameters()).device) == "cpu", "Use CUDA_VISIBLE_DEVICES=-1 for the fixed CPU replay."
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert model.patch_size == 14
    timing["model_load_time_s"] = perf_counter() - stage
    cad_features, cad_weights = {}, {}
    timing.update(cad_joint_feature_extraction_time_s=0., cad_mask_preparation_time_s=0.,
                  cad_rgb_verification_time_s=0., feature_and_input_saving_time_s=0.)
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    for cad_id in manifest["targets"]:
        stage = perf_counter()
        paths = sorted((DATASET / "templates" / cad_id).glob("view_*.png"))
        assert len(paths) == 14
        templates = [read_rgb(path) for path in paths]
        cad_features[cad_id] = extract_tokens(templates)
        timing["cad_joint_feature_extraction_time_s"] += perf_counter() - stage
        stage = perf_counter()
        cad_masks = cad_renderer_masks(ROOT / library[cad_id]["file_path"])
        cad_weights[cad_id] = np.stack([patch_occupancy(mask) for mask in cad_masks])
        timing["cad_mask_preparation_time_s"] += perf_counter() - stage
        stage = perf_counter()
        rerendered = association.render_cad_views(ROOT / library[cad_id]["file_path"], library[cad_id]["base_color_rgb"])
        np.testing.assert_array_equal(rerendered, templates)
        # All fixed CAD materials are below RGB=1, so nonwhite pixels exactly expose ray support.
        np.testing.assert_array_equal(cad_masks, np.any(np.asarray(templates) != 255, axis=-1))
        timing["cad_rgb_verification_time_s"] += perf_counter() - stage
        stage = perf_counter()
        cls, patches = cad_features[cad_id]
        np.savez_compressed(output_dir / "features" / f"{cad_id}.npz", cls_features=cls, patch_tokens=patches,
                            object_masks=cad_masks, patch_occupancy=cad_weights[cad_id])
        timing["feature_and_input_saving_time_s"] += perf_counter() - stage
        print(f"Extracted {cad_id}: {patches.shape}, verified 14 unchanged templates and renderer masks", flush=True)
    stage = perf_counter()
    observed_cls, observed_patches = extract_tokens(images)
    timing["observed_joint_feature_extraction_time_s"] = perf_counter() - stage
    stage = perf_counter()
    np.savez_compressed(output_dir / "features/observed.npz", object_ids=object_ids, cls_features=observed_cls,
                        patch_tokens=observed_patches, object_masks=np.stack(masks), patch_occupancy=observed_weights)
    for oid, mask in zip(object_ids, masks, strict=True):
        shutil.copyfile(ABLATION_1 / "masked/inputs" / f"{oid}.png", output_dir / "inputs" / f"{oid}.png")
        save_image(output_dir / "masks" / f"{oid}.png", mask * 255)
    timing["feature_and_input_saving_time_s"] += perf_counter() - stage
    pairs, max_cls_delta = [], 0.
    for target in baseline["associations"]:
        cad_id = target["cad_id"]
        original = read_json(DATASET / "results/colored_cpu/association" /
                             f"target_{list(manifest['targets']).index(cad_id) + 1:03}.json")
        old_rows = {r["object_id"]: r for r in original["candidate_ranking"]}
        baseline_rows = {r["object_id"]: r for r in target["candidate_ranking"]}
        for index, oid in enumerate(object_ids):
            pair = score_pair(observed_cls[index], observed_patches[index], observed_weights[index],
                              *cad_features[cad_id], cad_weights[cad_id])
            old = baseline_rows[oid]
            max_cls_delta = max(max_cls_delta, float(np.max(np.abs(np.asarray(pair["cls_view_cosines"]) - old["view_cosines"]))))
            assert old["geometry_raw"] == old_rows[oid]["geometry_raw"]
            assert old["geometry_score"] == old_rows[oid]["geometry_score"]
            assert pair["selected_view_index_1based"] == old["best_view_index_1based"]
            pairs.append({"cad_id": cad_id, "object_id": oid, **pair,
                          "geometry_raw": old_rows[oid]["geometry_raw"], "geometry_score": old_rows[oid]["geometry_score"]})
    assert len(pairs) == 36 and max_cls_delta < 1e-6
    save_json(output_dir / "pair_scores.json", pairs)
    audit = read_json(DATASET / "identity_audit.json")  # Evaluation labels never enter feature matching.
    evaluations = {name: evaluate_variant(name, pairs, manifest["targets"], audit, timing) for name in VARIANTS}
    assert all(a["rankings"] == b["rankings"] for a, b in
               zip(evaluations["cls_only"]["associations"], baseline["associations"], strict=True))
    for name, result in evaluations.items():
        variant_dir = output_dir / name
        variant_dir.mkdir()
        save_json(variant_dir / "evaluation.json", result)
        rows = []
        for target in result["associations"]:
            for pair in sorted(target["candidate_ranking"], key=lambda p: p["object_id"]):
                rows.append({**{key: pair[key] for key in (
                    "cad_id", "object_id", "selected_view_index_1based", "cls_score", "patch_score", "cls_patch_score",
                    "visual_raw", "visual_score", "geometry_score", "fused_score")},
                    **{f"cls_view_{i:02d}": score for i, score in enumerate(pair["cls_view_cosines"], 1)},
                    **{f"{mode}_rank": ranking.index(pair["object_id"]) + 1 for mode, ranking in target["rankings"].items()}})
        assert len(rows) == 36
        save_csv(variant_dir / "all_pair_scores.csv", rows)
        save_csv(variant_dir / "margins.csv", [{"cad_id": m["cad_id"], "expected_object_id": m["expected_object_id"],
                 **{f"{mode}_{key}": value for mode in ("visual_raw", "visual_normalized", "fused")
                    for key, value in m[mode].items() if key != "correct_minus_each_incorrect"}} for m in result["margins"]])
        print(f"{name}: {result['accuracy']} | margins {result['margin_summary']}", flush=True)
    save_json(output_dir / "protocol.json", {
        "dataset": str(DATASET.relative_to(ROOT)), "ablation_1_source": str(ABLATION_1.relative_to(ROOT)),
        "known_cad_ids": manifest["targets"], "variants": VARIANTS,
        "encoder": association.DINO_MODEL, "encoder_source": association.DINO_REPO,
        "checkpoint_sha256": association.content_hash(association.DINO_HUB_DIR / "checkpoints/dinov2_vits14_pretrain.pth"),
        "tokens": "Final x_norm_clstoken/x_norm_patchtokens, then float64 L2 normalization per 384-D token",
        "patch_grid": [16, 16], "patch_size_pixels": [14, 14], "grid_order": "row-major (y, x)",
        "observed_rgb": "Byte-identical Ablation 1 masked 224x224 white-letterboxed PNGs",
        "observed_masks": "Ablation 1 projection masks, same tight bbox/resize dimensions/offset; nearest-neighbor resize, zero padding",
        "patch_occupancy": "Object-pixel count / 196. Any positive occupancy defines foreground; no tuned cutoff",
        "cad_masks": "Actual finite ray-hit masks; same mesh fit/rays/views as production; every render verified against fixed PNG",
        "view_selection": "Maximum CLS cosine over exactly 14 views; first view wins exact ties; identical rule for all variants",
        "patch_score": "Occupancy-weighted mean of observed-foreground-to-selected-CAD-foreground maximum token cosine; many-to-one allowed",
        "combined_score": "0.5 * cls_score + 0.5 * patch_score; no tuning",
        "fusion": "0.5 * ((variant_raw_score + 1) / 2) + 0.5 * recorded_geometry_score",
        "geometry": "All 36 original colored_cpu geometry_raw/geometry_score records reused exactly; no recomputation",
        "margin_definition": "Correct candidate score minus highest-scoring incorrect candidate; positive means separation; no decision gate",
        "decision_policy": "All 36 CAD/candidate pairs, independent rankings, ties by object ID; existing conflict handling",
        "sam_used": False, "production_path_modified": False,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
                        "device": str(next(model.parameters()).device), "torch_threads": torch.get_num_threads(),
                        **{key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
        "source_sha256": source_hashes, "fixed_dataset_sha256": fixed_hashes, "ablation_1_sha256": ablation_1_hashes,
    })
    save_report(output_dir, evaluations, baseline, timing, images, masks, observed_weights)
    assert snapshot_hashes() == fixed_hashes and verify_artifacts(ABLATION_1) == ablation_1_hashes
    assert all(association.content_hash(ROOT / path) == expected for path, expected in source_hashes.items())
    save_json(output_dir / "validation.json", {
        "fixed_dataset_and_ablation_1_hashes_unchanged": True, "production_and_input_source_hashes_unchanged": True,
        "all_84_cad_rgb_renders_equal_saved_templates": True, "all_84_ray_masks_verified": True,
        "max_cls_view_cosine_delta_from_ablation_1": max_cls_delta,
        "cls_rankings_and_predictions_match_ablation_1": True, "all_36_selected_views_match_ablation_1": True,
        "all_36_recorded_geometry_results_unchanged": True,
        "observed_and_cad_token_shapes": [[6, 16, 16, 384], [14, 16, 16, 384]],
        "saved_observed_rgb_byte_identical_to_ablation_1": all(association.content_hash(output_dir / "inputs" / f"{oid}.png")
            == ablation_1_hashes[f"masked/inputs/{oid}.png"] for oid in object_ids),
        "pair_count_per_variant": 36, "cls_view_cosine_count": 504,
    })
    save_json(output_dir / "runtime.json", {"total_wall_time_after_imports_s": perf_counter() - start,
              "measured_shared_stages": timing, "variant_runtime": {name: r["runtime"] for name, r in evaluations.items()},
              "scope": "Includes integrity checks, feature saving, evaluations and report; excludes final output hashes"})
    (output_dir / "SHA256SUMS").write_text("".join(
        f"{association.content_hash(path)}  {path.relative_to(output_dir).as_posix()}\n"
        for path in sorted(output_dir.rglob("*")) if path.is_file()), encoding="utf-8")
    print(f"Saved Ablation 2: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
