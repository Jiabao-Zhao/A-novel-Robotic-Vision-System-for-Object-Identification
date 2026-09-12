"""Ablation 1: fixed bbox versus projected localization-mask DINO inputs.

From the repository root, in the existing association environment:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_input_ablation

The published colored templates and all 36 recorded geometry results are frozen.
Only observed RGB representation changes. No simulation or registration is rerun.
"""

import csv
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d
import torch

import cad_object_association as association
from main import association_conflicts
from scripts.run_wrist_association import evaluate_rankings
from visualize_localization_projection import mask_bbox, project_rgb_on_white, project_with_z_buffer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/wrist_association_2026-09-11"
OUTPUT_ROOT = ROOT / "outputs/ablation_1_wrist_association_2026-09-11"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path, image):
    pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"Could not save image: {path}")


def snapshot_hashes():
    """Verify the complete published snapshot without rewriting any source data."""
    hashes = {}
    for line in (DATASET / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        actual = association.content_hash(DATASET / name)
        if actual != expected:
            raise ValueError(f"Fixed dataset hash mismatch: {name}")
        hashes[name] = actual
    return hashes


def masked_crop(rgb, cloud, intrinsics):
    """Nearest-pixel projection only; keep holes and never expand point support."""
    depth, projected_count = project_with_z_buffer(cloud, intrinsics)
    mask = np.isfinite(depth)
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("Cleaned cloud has no projected support in the RGB image.")
    white = project_rgb_on_white(rgb, mask)
    np.testing.assert_array_equal(white[mask], rgb[mask])
    assert np.all(white[~mask] == 255)
    x1, y1, x2, y2 = bbox
    return white[y1:y2 + 1, x1:x2 + 1], mask, {
        "crop_bbox_2d_xyxy": bbox, "cloud_point_count": len(cloud.points),
        "projected_point_count": projected_count, "mask_pixel_count": int(mask.sum()),
    }


def run_variant(name, output_dir, rgb, localization, cad_features, recorded):
    start = perf_counter()
    inputs, crops, masks, input_records = [], [], [], []
    for item in localization["objects"]:
        stage = perf_counter()
        object_id = item["object_id"]
        bbox = [item["roi"][key] for key in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = bbox
        if name == "bbox":
            crop = rgb[y1:y2 + 1, x1:x2 + 1].copy()
            np.testing.assert_array_equal(crop, read_rgb(DATASET / "scene/crops" / f"{object_id}.png"))
            details = {"crop_bbox_2d_xyxy": bbox}
        else:
            cloud = o3d.io.read_point_cloud(str(ROOT / item["pointcloud_path"]))
            assert len(cloud.points) == item["saved_point_count"]
            crop, mask, details = masked_crop(rgb, cloud, localization["camera_intrinsics"])
            masks.append(mask.astype(np.uint8) * 255)
        canvas = association.letterbox_rgb(crop)
        # Encoding the saved canvas must be pixel-identical to encoding the raw crop.
        np.testing.assert_array_equal(association.letterbox_rgb(canvas), canvas)
        crops.append(crop)
        inputs.append(canvas)
        input_records.append({"object_id": object_id, "original_bbox_2d_xyxy": bbox,
                              "pointcloud_path": item["pointcloud_path"], **details,
                              "native_crop_shape_hwc": list(crop.shape),
                              "dino_input_shape_hwc": list(canvas.shape),
                              "input_preparation_time_s": perf_counter() - stage})
    preparation_time = perf_counter() - start
    stage = perf_counter()
    features = association.extract_dino_features(inputs)
    dino_time = perf_counter() - stage
    results = []
    stage = perf_counter()
    for baseline in recorded:
        cad_id = baseline["cad_id"]
        fixed_rows = {row["object_id"]: row for row in baseline["candidate_ranking"]}
        rows = []
        for item, feature in zip(localization["objects"], features, strict=True):
            fixed = fixed_rows[item["object_id"]]
            raw = association.cosine_similarity(feature, cad_features[cad_id])
            visual = association.normalize_visual_score(raw)
            # Retain every geometric metric/transform exactly, including failed fits.
            geometry = fixed["geometry_score"]
            views = np.asarray(cad_features[cad_id], dtype=float)
            candidate = np.asarray(feature, dtype=float)
            cosines = np.clip(views @ candidate / (np.linalg.norm(views, axis=1)
                                                 * np.linalg.norm(candidate)), -1, 1)
            assert raw == float(cosines.max())
            rows.append({"object_id": item["object_id"], "visual_raw": raw,
                         "visual_score": visual, "view_cosines": cosines.tolist(),
                         "best_view_index_1based": int(cosines.argmax()) + 1,
                         "geometry_raw": fixed["geometry_raw"], "geometry_score": geometry,
                         "fused_score": association.fuse_scores(visual, geometry)})
        rankings = {mode: [row["object_id"] for row in association.rank_candidates(rows, key)]
                    for mode, key in (("visual", "visual_score"), ("geometry", "geometry_score"),
                                      ("fused", "fused_score"))}
        results.append({"cad_id": cad_id, "target_description": baseline["target_description"],
                        "candidate_ranking": association.rank_candidates(rows),
                        "rankings": rankings, "predictions": {mode: ids[0] for mode, ids in rankings.items()},
                        "selected_object_id": rankings["fused"][0], "resolution": "ranking_only"})
    scoring_time = perf_counter() - stage
    # Ground truth enters only after all independent rankings have been computed.
    evaluation = evaluate_rankings(results, read_json(DATASET / "identity_audit.json"))
    conflicts = association_conflicts(results)
    for result in results:
        result["final_object_id"] = None if conflicts else result["selected_object_id"]
    compute_time = perf_counter() - start
    variant_dir = output_dir / name
    (variant_dir / "inputs").mkdir(parents=True)
    (variant_dir / "crops").mkdir()
    if masks:
        (variant_dir / "masks").mkdir()
    for index, record in enumerate(input_records):
        filename = f"{record['object_id']}.png"
        save_image(variant_dir / "inputs" / filename, inputs[index])
        save_image(variant_dir / "crops" / filename, crops[index])
        if masks:
            save_image(variant_dir / "masks" / filename, masks[index])
        np.testing.assert_array_equal(read_rgb(variant_dir / "inputs" / filename), inputs[index])
    evaluation.update(variant=name, association_set_resolution="conflicting" if conflicts else "ranking_only",
                      conflicts=conflicts, associations=results, inputs=input_records,
                      geometry_source="Frozen results/colored_cpu/association/target_*.json",
                      runtime={"input_preparation_time_s": preparation_time,
                               "workspace_dino_feature_time_s": dino_time,
                               "all_pair_scoring_and_ranking_time_s": scoring_time,
                               "compute_and_evaluation_time_s": compute_time,
                               "image_saving_and_verification_time_s": perf_counter() - start - compute_time,
                               "geometry_recomputation_time_s": 0.0,
                               "shared_setup_included": False})
    flat_rows = []
    truth = {row["cad_id"]: row["expected_object_id"] for row in evaluation["per_target"]}
    for result in results:
        for row in sorted(result["candidate_ranking"], key=lambda row: row["object_id"]):
            flat_rows.append({"cad_id": result["cad_id"], "object_id": row["object_id"],
                              "expected_object_id": truth[result["cad_id"]],
                              **{key: row[key] for key in ("visual_raw", "visual_score", "geometry_score", "fused_score")},
                              **{f"{mode}_rank": ids.index(row["object_id"]) + 1
                                 for mode, ids in result["rankings"].items()},
                              "best_view_index_1based": row["best_view_index_1based"]})
    assert len(flat_rows) == 36
    save_csv(variant_dir / "all_pair_scores.csv", flat_rows)
    evaluation["runtime"]["wall_time_before_result_json_s"] = perf_counter() - start
    save_json(variant_dir / "evaluation.json", evaluation)
    print(f"{name}: {evaluation['accuracy']} | {compute_time:.4f} s compute", flush=True)
    return evaluation, flat_rows, inputs


def save_comparison(output_dir, bbox, masked, shared_runtime):
    a_eval, a_rows, a_inputs = bbox
    b_eval, b_rows, b_inputs = masked
    deltas = []
    for a, b in zip(a_rows, b_rows, strict=True):
        assert (a["cad_id"], a["object_id"]) == (b["cad_id"], b["object_id"])
        assert a["geometry_score"] == b["geometry_score"]
        deltas.append({"cad_id": a["cad_id"], "object_id": a["object_id"],
                       **{f"{mode}_{key}": row[key] for mode, row in (("bbox", a), ("masked", b))
                          for key in ("visual_raw", "fused_score", "visual_rank", "fused_rank")},
                       "raw_cosine_delta": b["visual_raw"] - a["visual_raw"],
                       "fused_score_delta": b["fused_score"] - a["fused_score"],
                       "geometry_score_delta": 0.0})
    save_csv(output_dir / "score_changes.csv", deltas)
    prediction_changes = [{"cad_id": a["cad_id"], "expected_object_id": a["expected_object_id"],
                           "bbox": a["predictions"], "masked": b["predictions"],
                           "changed": {mode: a["predictions"][mode] != b["predictions"][mode]
                                       for mode in a["predictions"]}}
                          for a, b in zip(a_eval["per_target"], b_eval["per_target"], strict=True)]
    save_json(output_dir / "comparison.json", {
        "prediction_changes": prediction_changes,
        "raw_cosine_changed_pair_count": sum(row["raw_cosine_delta"] != 0 for row in deltas),
        "max_absolute_raw_cosine_delta": max(abs(row["raw_cosine_delta"]) for row in deltas),
        "geometry_scores_and_raw_metrics_unchanged": all(
            {r["object_id"]: r["geometry_raw"] for r in a["candidate_ranking"]}
            == {r["object_id"]: r["geometry_raw"] for r in b["candidate_ranking"]}
            for a, b in zip(a_eval["associations"], b_eval["associations"], strict=True)),
    })
    lines = ["# Ablation 1: bbox versus localization-masked RGB", "",
             "Same fixed six-object scene, known CAD IDs, 84 saved colored templates (14 per CAD), "
             "frozen DINOv2-small, and all 36 recorded colored-CPU FPFH/RANSAC/ICP results. "
             "Every candidate and all CAD views are evaluated. Only observed RGB representation changes.", "",
             "Mask: camera-frame cleaned PLY points projected with stored RGB intrinsics; nearest-pixel "
             "rounding and nearest positive Z. Unsupported pixels become white; tight inclusive crop, "
             "aspect-preserving bicubic resize and centered white 224×224 letterbox. "
             "No SAM, dilation, closing, hole filling, pruning, thresholds, or parameter tuning.", "",
             "Raw visual score is the maximum of 14 cosine similarities. Fusion remains "
             "`0.5 * ((cosine + 1) / 2) + 0.5 * recorded_geometry_fitness`.", "",
             "| Input | Visual top-1 | Fused top-1 | Compute + evaluation (s) | Observed DINO batch (s) |",
             "|---|---:|---:|---:|---:|"]
    for result in (a_eval, b_eval):
        accuracy, runtime = result["accuracy"], result["runtime"]
        lines.append(f"| {result['variant']} | {accuracy['visual']['correct']}/6 | {accuracy['fused']['correct']}/6 "
                     f"| {runtime['compute_and_evaluation_time_s']:.4f} | {runtime['workspace_dino_feature_time_s']:.4f} |")
    lines += ["", f"Shared setup: {shared_runtime['total_time_s']:.4f} s, including model loading, "
              "encoding the saved CAD templates once, and reading fixed data. Shared costs and image/result "
              "saving are excluded from the compute column. Geometry is reused, so these are replay timings, "
              "not full registration-pipeline timings. CPU, one thread; bbox runs first after CAD encoding, "
              "then masked. Single-run timings are descriptive, not a speed benchmark.", "",
              "| CAD target | Truth | Visual bbox → masked | Fused bbox → masked |",
              "|---|---|---|---|"]
    for change in prediction_changes:
        lines.append(f"| {change['cad_id']} | {change['expected_object_id']} | "
                     f"{change['bbox']['visual']} → {change['masked']['visual']} | "
                     f"{change['bbox']['fused']} → {change['masked']['fused']} |")
    lines += ["", "Object IDs: 001 pulley; 002 large gear; 003 blue block; 004 red block; "
              "005 medium gear; 006 rectangular pin. Duplicate selections remain independent predictions "
              "for accuracy, and the existing conflict rule keeps downstream final IDs null.", "",
              "## Raw cosine changes (masked minus bbox)", "",
              "| CAD | 001 | 002 | 003 | 004 | 005 | 006 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for index in range(0, len(deltas), 6):
        group = deltas[index:index + 6]
        lines.append(f"| {group[0]['cad_id']} | " + " | ".join(f"{r['raw_cosine_delta']:+.6f}" for r in group) + " |")
    lines += ["", "All geometry-score deltas are exactly zero. Each fused-score delta equals one quarter "
              "of its raw-cosine delta. [Full-precision pair changes](score_changes.csv).", "",
              "Separate results: [bbox scores](bbox/all_pair_scores.csv), [bbox rankings/runtime](bbox/evaluation.json), "
              "[masked scores](masked/all_pair_scores.csv), [masked rankings/runtime](masked/evaluation.json). "
              "Evaluation JSON also contains all 14 per-view cosine scores for every pair.", "",
              "[Protocol and provenance](protocol.json). [Integrity and baseline reproduction](validation.json). "
              "Exact encoder canvases are in `bbox/inputs/` and `masked/inputs/`; native crops and "
              "full-resolution masks are saved alongside them.", "",
              "![Exact 224×224 inputs: bbox left, masked right](input_comparison.png)", "",
              "This is one fixed six-object ablation. No parameters were optimized using its results.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    sheet = np.full((6 * 260 + 40, 480, 3), 255, dtype=np.uint8)
    for x, label in ((8, "A: bbox"), (248, "B: localization mask")):
        cv2.putText(sheet, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    for index, (a, b, record) in enumerate(zip(a_inputs, b_inputs, a_eval["inputs"], strict=True)):
        y = 40 + index * 260
        cv2.putText(sheet, record["object_id"], (8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        sheet[y + 28:y + 252, 8:232] = a
        sheet[y + 28:y + 252, 248:472] = b
    save_image(output_dir / "input_comparison.png", sheet)


def main():
    start = perf_counter()
    os.chdir(ROOT)
    assert association.DINO_MODEL == "dinov2_vits14" and association.IMAGE_SIZE == 224
    assert association.FUSION_METHOD == "weighted_mean"
    assert (association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT) == (.5, .5)
    hashes = snapshot_hashes()
    manifest = read_json(DATASET / "manifest.json")
    localization = read_json(ROOT / manifest["localization_path"])
    assert localization["frame"] == "camera" and len(localization["objects"]) == 6
    intrinsics = localization["camera_intrinsics"]
    np.testing.assert_array_equal(np.load(ROOT / manifest["capture"]["intrinsics"]),
                                  [[intrinsics["fx"], 0, intrinsics["cx"]],
                                   [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]])
    rgb = read_rgb(ROOT / manifest["capture"]["rgb"])
    assert rgb.shape == (intrinsics["height"], intrinsics["width"], 3)
    recorded = [read_json(path) for path in sorted((DATASET / "results/colored_cpu/association").glob("target_*.json"))]
    assert [r["cad_id"] for r in recorded] == list(manifest["targets"])
    expected_ids = {item["object_id"] for item in localization["objects"]}
    for result in recorded:
        assert len(result["candidate_ranking"]) == 6
        assert {row["object_id"] for row in result["candidate_ranking"]} == expected_ids
    stage = perf_counter()
    model = association.load_dino_encoder()
    assert str(next(model.parameters()).device) == "cpu", "Use CUDA_VISIBLE_DEVICES=-1 for this CPU replay."
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    model_time = perf_counter() - stage
    cad_features, cad_times = {}, {}
    for cad_id in manifest["targets"]:
        stage = perf_counter()
        views = sorted((DATASET / "templates" / cad_id).glob("view_*.png"))
        assert len(views) == 14
        cad_features[cad_id] = association.extract_dino_features([read_rgb(path) for path in views])
        cad_times[cad_id] = perf_counter() - stage
    shared_runtime = {"total_time_s": perf_counter() - start, "model_load_time_s": model_time,
                      "cad_template_read_and_encode_time_s": cad_times}
    output_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True)
    bbox = run_variant("bbox", output_dir, rgb, localization, cad_features, recorded)
    masked = run_variant("masked", output_dir, rgb, localization, cad_features, recorded)
    source_paths = [Path(__file__), ROOT / "cad_object_association.py", ROOT / "CADPointCloudRegistration.py",
                    ROOT / "helper_function.py", ROOT / "visualize_localization_projection.py",
                    ROOT / "main.py", ROOT / "scripts/run_wrist_association.py"]
    save_json(output_dir / "protocol.json", {
        "dataset": str(DATASET.relative_to(ROOT)), "known_cad_ids": manifest["targets"],
        "only_changed_factor": "Observed RGB representation: original bbox versus raw localization mask on white",
        "geometry_policy": "Reuse every recorded colored_cpu geometry_raw and geometry_score exactly; no rerun",
        "template_policy": "Encode all 14 published colored PNGs per CAD once; share identical descriptors",
        "mask_policy": "Pinhole camera projection, np.rint, nearest positive Z; no morphology or silhouette filling",
        "mask_dilation_px": 0, "mask_close_kernel_px": 0, "sam_used": False,
        "preprocessing": "Existing bicubic aspect-preserving white letterbox to 224x224; existing ImageNet normalization",
        "encoder": association.DINO_MODEL, "encoder_source": association.DINO_REPO,
        "checkpoint_sha256": association.content_hash(association.DINO_HUB_DIR / "checkpoints/dinov2_vits14_pretrain.pth"),
        "fusion": "0.5 * ((raw_cosine + 1) / 2) + 0.5 * geometry_fitness",
        "candidates_per_target": 6, "targets_per_variant": 6, "views_per_cad": 14,
        "candidate_selection": "Exhaustive independent argmax; exact ties by object ID; no thresholds or tuning",
        "run_order": ["bbox", "masked"], "shared_runtime": shared_runtime,
        "runtime_scope": "Measured after Python imports; shared setup reported once; no geometry recomputation",
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
                        "numpy": np.__version__, "device": "cpu", "torch_num_threads": torch.get_num_threads(),
                        **{key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
        "source_sha256": {str(path.relative_to(ROOT)): association.content_hash(path) for path in source_paths},
        "fixed_dataset_sha256": hashes,
    })
    baseline_delta = max(abs(row["visual_raw"] - next(r["visual_raw"] for r in old["candidate_ranking"]
                         if r["object_id"] == row["object_id"]))
                         for result, old in zip(bbox[0]["associations"], recorded, strict=True)
                         for row in result["candidate_ranking"])
    assert baseline_delta < 1e-6, f"BBox replay differs from fixed baseline: {baseline_delta}"
    assert all(new["predictions"] == old["predictions"]
               for new, old in zip(bbox[0]["associations"], recorded, strict=True))
    assert snapshot_hashes() == hashes
    save_json(output_dir / "validation.json", {
        "fixed_snapshot_files_verified_before_and_after": len(hashes), "baseline_max_raw_cosine_delta": baseline_delta,
        "bbox_predictions_match_recorded": True, "exhaustive_pairs_per_variant": 36,
        "view_cosines_per_variant": 504, "saved_224x224_inputs_equal_encoder_canvases": True,
        "bbox_crops_equal_published_crops": True, "masked_native_pixels_outside_support_are_white": True,
        "masked_native_pixels_inside_support_equal_original_rgb": True,
        "intrinsics_matrix_equals_localization_intrinsics": True,
    })
    save_comparison(output_dir, bbox, masked, shared_runtime)
    save_json(output_dir / "runtime.json", {"total_wall_time_s": perf_counter() - start,
              "shared_setup": shared_runtime, "bbox": bbox[0]["runtime"], "masked": masked[0]["runtime"],
              "scope": "After imports through report creation, excluding final hashes; recorded geometry reused"})
    (output_dir / "SHA256SUMS").write_text("".join(
        f"{association.content_hash(path)}  {path.relative_to(output_dir).as_posix()}\n"
        for path in sorted(output_dir.rglob("*")) if path.is_file()), encoding="utf-8")
    print(f"Saved ablation: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
