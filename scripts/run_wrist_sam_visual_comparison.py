"""Compare saved SAM 3 and localization masks with the fixed DINO CLS baseline.

Run in the existing association environment from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python -m scripts.run_wrist_sam_visual_comparison

No API requests, rendering, registration, or production-path changes.
"""

import os
import platform
from datetime import datetime, timezone
from time import perf_counter

import cv2
import numpy as np
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import (
    ROOT, DATASET, read_json, read_rgb, save_csv, save_image, save_json, snapshot_hashes,
)
from scripts.run_wrist_patch_ablation import ABLATION_1, verify_artifacts
from visualize_localization_projection import mask_bbox, project_rgb_on_white


ABLATION_2 = ROOT / "outputs/ablation_2_wrist_association_2026-09-11/20260912T160053225612Z"
SAM = ROOT / "outputs/roboflow_sam_workspace/20260914T150601681311Z"
OUTPUT_ROOT = ROOT / "outputs/sam_visual_comparison_wrist_association_2026-09-11"


def prepare_masked_input(rgb, mask):
    """Use the same inclusive tight crop and white letterbox as Ablation 1."""
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("Expected a binary mask in the original RGB image coordinates.")
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("An empty mask is unavailable, not a blank DINO candidate.")
    white = project_rgb_on_white(rgb, mask)
    x1, y1, x2, y2 = bbox
    crop = white[y1:y2 + 1, x1:x2 + 1]
    return association.letterbox_rgb(crop), crop, bbox


def score_pairs(cad_features, observed_features, object_ids):
    """Missing SAM candidates remain null; every available pair uses all 14 views."""
    pairs = []
    for cad_id, cad in cad_features.items():
        cad = np.asarray(cad, dtype=np.float64)
        assert cad.shape == (14, 384)
        cad = cad / np.linalg.norm(cad, axis=1, keepdims=True)
        for oid in object_ids:
            feature = observed_features.get(oid)
            cosines = None
            if feature is not None:
                feature = np.asarray(feature, dtype=np.float64)
                cosines = np.clip(cad @ (feature / np.linalg.norm(feature)), -1, 1)
                assert np.isfinite(cosines).all()
            pairs.append({
                "cad_id": cad_id, "object_id": oid, "available": feature is not None,
                "raw_cosine": float(cosines.max()) if cosines is not None else None,
                "best_view_index_1based": int(cosines.argmax()) + 1 if cosines is not None else None,
                "view_cosines": cosines.tolist() if cosines is not None else [None] * 14,
            })
    return pairs


def evaluate(pairs, truth, candidates):
    """Independent CAD-to-candidate argmax; missing truth counts as an error."""
    targets = []
    for cad_id, expected in truth.items():
        rows = [p for p in pairs if p["cad_id"] == cad_id
                and p["object_id"] in candidates and p["available"]]
        ranking = association.rank_candidates(rows, "raw_cosine")
        correct = next((p for p in rows if p["object_id"] == expected), None)
        incorrect = [p for p in ranking if p["object_id"] != expected]
        best_incorrect = incorrect[0] if incorrect else None
        prediction = ranking[0]["object_id"] if ranking else None
        targets.append({
            "cad_id": cad_id, "expected_object_id": expected,
            "truth_candidate_available": correct is not None,
            "prediction": prediction, "correct": prediction == expected,
            "ranking": [p["object_id"] for p in ranking],
            "correct_raw_cosine": correct["raw_cosine"] if correct else None,
            "strongest_incorrect_object_id": best_incorrect["object_id"] if best_incorrect else None,
            "strongest_incorrect_raw_cosine": best_incorrect["raw_cosine"] if best_incorrect else None,
            "margin": correct["raw_cosine"] - best_incorrect["raw_cosine"]
            if correct and best_incorrect else None,
        })
    margins = [t["margin"] for t in targets if t["margin"] is not None]
    return {
        "accuracy": {"correct": sum(t["correct"] for t in targets), "total": len(targets),
                     "fraction": sum(t["correct"] for t in targets) / len(targets)},
        "mean_margin_over_available_targets": float(np.mean(margins)) if margins else None,
        "margin_target_count": len(margins), "targets": targets,
    }


def save_report(output, results, changes, object_ids, descriptions, images, runtime):
    lines = ["# SAM 3 versus localization masks: DINO visual association", "",
             "Same fixed scene, DINOv2-small CLS descriptors, 84 cached colored CAD templates "
             "(14 per CAD), raw maximum-view cosine, and white 224x224 aspect-preserving letterbox. "
             "Each mask is tightly cropped. SAM class labels and confidences never enter DINO scoring.", "",
             "The SAM run returned five masks and missed object_001 (pulley). "
             "Its six unavailable CAD/pulley-candidate scores are null, not fabricated or filled "
             "from localization. No new SAM call, mask cleanup, thresholds, tuning, or geometry changes.", "",
             "| Evaluation | Localization masks | SAM masks |",
             "|---|---:|---:|"]
    for scope, label in (("full_scene", "Six targets, all available candidates"),
                         ("common_five", "Same five candidates and five represented targets")):
        cells = [f"{results[mode][scope]['accuracy']['correct']}/{results[mode][scope]['accuracy']['total']}"
                 for mode in ("localization", "sam")]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    means = [results[m]["common_five"]["mean_margin_over_available_targets"] for m in ("localization", "sam")]
    lines += [f"| Mean correct-minus-best-incorrect margin, same five | {means[0]:+.6f} | {means[1]:+.6f} |", "",
              "The matched five-object comparison isolates RGB representation from the missing candidate. "
              "The six-target comparison includes the segmentation miss and therefore has different "
              "candidate availability. These are independent target queries, not a one-to-one assignment.", "",
              "| CAD target | Truth | Localization top-1 | SAM top-1 | Matched margin: localization → SAM |",
              "|---|---|---|---|---|"]
    for row in changes:
        a, b = row["localization_common_margin"], row["sam_common_margin"]
        margin = "unavailable: SAM missed pulley" if b is None else f"{a:+.6f} → {b:+.6f}"
        lines.append(f"| {descriptions[row['cad_id']]} | {row['expected_object_id']} | "
                     f"{row['localization_prediction']} | {row['sam_prediction']} | {margin} |")
    lines += ["", "All six CAD targets were scored against every available candidate: "
              "36 localization pairs / 504 view scores and 30 SAM pairs / 420 view scores. "
              "The CSV contains 36 slots for each variant, with six explicitly missing SAM slots.", "",
              "The observed DINO batches took "
              f"{runtime['localization_dino_time_s']:.4f} s (six localization inputs) and "
              f"{runtime['sam_dino_time_s']:.4f} s (five SAM inputs), CPU, one thread. "
              f"Shared model loading: {runtime['model_load_time_s']:.4f} s. "
              f"The earlier SAM cloud request took {runtime['previous_sam_cloud_request_time_s']:.4f} s; "
              "it was not rerun and is excluded from this comparison's runtime. "
              "CAD descriptors were cached. These single runs are not a throughput benchmark.", "",
              "[All raw scores and 14 per-view cosines](all_pair_scores.csv), "
              "[all pair deltas](score_changes.csv), [rankings and metrics](comparison.json), "
              "[protocol](protocol.json), [validation](validation.json). "
              "Exact DINO canvases, native crops, and masks are in each variant's subdirectory.", "",
              "![Localization input on left, SAM input on right for each object](input_comparison.png)", "",
              "This is a diagnostic on one fixed scene, not evidence of general accuracy. "
              "Production association is unchanged.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    sheet = np.full((3 * 288, 2 * 472, 3), 245, dtype=np.uint8)
    for i, oid in enumerate(object_ids):
        x, y = (i % 2) * 472, (i // 2) * 288
        for text, offset, scale in ((oid, (8, 22), .6), ("Localization", (8, 46), .48),
                                    ("SAM 3", (240, 46), .48)):
            cv2.putText(sheet, text, (x + offset[0], y + offset[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
        for mode, dx in (("localization", 8), ("sam", 240)):
            if oid in images[mode]:
                sheet[y + 54:y + 278, x + dx:x + dx + 224] = images[mode][oid]
            else:
                cv2.putText(sheet, "NO MASK", (x + dx + 30, y + 164),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (170, 35, 35), 1, cv2.LINE_AA)
    save_image(output / "input_comparison.png", sheet)


def main():
    start = perf_counter()
    os.chdir(ROOT)
    fixed_hashes = snapshot_hashes()
    a1_hashes, a2_hashes = verify_artifacts(ABLATION_1), verify_artifacts(ABLATION_2)
    sam_hashes = {str(p.relative_to(SAM)): association.content_hash(p) for p in SAM.rglob("*") if p.is_file()}
    baseline = read_json(ABLATION_1 / "masked/evaluation.json")
    protocol = read_json(ABLATION_2 / "protocol.json")
    manifest = read_json(DATASET / "manifest.json")
    truth = {t["cad_id"]: t["expected_object_id"] for t in baseline["per_target"]}
    object_ids = [r["object_id"] for r in baseline["inputs"]]
    sam_records = read_json(SAM / "inspection.json")["masks"]
    sam_ids = [oid for oid in object_ids if any(r["object_id"] == oid for r in sam_records)]
    assert len(sam_ids) == len(sam_records) == 5 and set(object_ids) - set(sam_ids) == {"object_001"}
    assert association.DINO_MODEL == protocol["encoder"] == "dinov2_vits14"
    assert association.DINO_REPO == protocol["encoder_source"] and association.IMAGE_SIZE == 224
    checkpoint_hash = association.content_hash(association.DINO_HUB_DIR / "checkpoints/dinov2_vits14_pretrain.pth")
    assert checkpoint_hash == protocol["checkpoint_sha256"]
    np.testing.assert_array_equal(association.VIEW_DIRECTIONS, manifest["template_camera_directions_cad_xyz"])
    rgb = read_rgb(DATASET / "scene/rgb.png")
    assert association.content_hash(DATASET / "scene/rgb.png") == read_json(SAM / "run.json")["image_sha256"]
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    images, features, pairs, results, timing = {}, {}, {}, {}, {}
    input_records = []
    for mode, ids, source in (("localization", object_ids, ABLATION_1 / "masked"), ("sam", sam_ids, SAM)):
        stage = perf_counter()
        images[mode] = {}
        for directory in ("inputs", "crops", "masks"):
            (output / mode / directory).mkdir(parents=True)
        for oid in ids:
            path = source / "masks" / f"{oid}.png"
            raw_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw_mask is None or not np.isin(raw_mask, [0, 255]).all():
                raise ValueError(f"Missing or nonbinary mask: {path}")
            canvas, crop, bbox = prepare_masked_input(rgb, raw_mask > 0)
            if mode == "localization":
                np.testing.assert_array_equal(canvas, read_rgb(source / "inputs" / f"{oid}.png"))
            images[mode][oid] = canvas
            for directory, image in (("inputs", canvas), ("crops", crop), ("masks", raw_mask)):
                save_image(output / mode / directory / f"{oid}.png", image)
            input_records.append({"variant": mode, "object_id": oid, "crop_bbox_xyxy_inclusive": bbox,
                                  "mask_pixels": int((raw_mask > 0).sum()),
                                  "input_sha256": association.content_hash(output / mode / "inputs" / f"{oid}.png")})
        timing[f"{mode}_input_preparation_and_saving_time_s"] = perf_counter() - stage
    stage = perf_counter()
    model = association.load_dino_encoder()
    assert str(next(model.parameters()).device) == "cpu" and torch.get_num_threads() == 1
    timing["model_load_time_s"] = perf_counter() - stage
    stage = perf_counter()
    cad_features = {cid: np.load(ABLATION_2 / "features" / f"{cid}.npz")["cls_features"] for cid in truth}
    timing["cached_cad_feature_read_time_s"] = perf_counter() - stage
    for mode in images:
        stage = perf_counter()
        encoded = association.extract_dino_features(list(images[mode].values()))
        timing[f"{mode}_dino_time_s"] = perf_counter() - stage
        features[mode] = dict(zip(images[mode], encoded, strict=True))
        np.savez_compressed(output / mode / "features.npz", object_ids=list(images[mode]), cls_features=encoded)
        stage = perf_counter()
        pairs[mode] = score_pairs(cad_features, features[mode], object_ids)
        results[mode] = {
            "full_scene": evaluate(pairs[mode], truth, object_ids),
            "common_five": evaluate(pairs[mode], {k: v for k, v in truth.items() if v in sam_ids}, sam_ids),
        }
        timing[f"{mode}_scoring_and_evaluation_time_s"] = perf_counter() - stage
        save_json(output / mode / "evaluation.json", results[mode])
        save_json(output / mode / "pair_scores.json", pairs[mode])
    old_rows = {(a["cad_id"], p["object_id"]): p for a in baseline["associations"] for p in a["candidate_ranking"]}
    max_delta = max(float(np.max(np.abs(np.array(p["view_cosines"]) - old_rows[p["cad_id"], p["object_id"]]["view_cosines"])))
                    for p in pairs["localization"])
    assert max_delta < 1e-6
    assert all(t["ranking"] == a["rankings"]["visual"] for t, a in zip(results["localization"]["full_scene"]["targets"], baseline["associations"], strict=True))
    changes, deltas, flat = [], [], []
    for a, b in zip(pairs["localization"], pairs["sam"], strict=True):
        deltas.append({"cad_id": a["cad_id"], "object_id": a["object_id"],
                       "localization_raw_cosine": a["raw_cosine"], "sam_raw_cosine": b["raw_cosine"],
                       "sam_minus_localization": b["raw_cosine"] - a["raw_cosine"] if b["available"] else None})
    for mode in pairs:
        for p in pairs[mode]:
            flat.append({"variant": mode, **{k: v for k, v in p.items() if k != "view_cosines"},
                         **{f"view_{i:02d}_cosine": v for i, v in enumerate(p["view_cosines"], 1)}})
    for a, b in zip(results["localization"]["full_scene"]["targets"], results["sam"]["full_scene"]["targets"], strict=True):
        common = {mode: next((t for t in results[mode]["common_five"]["targets"] if t["cad_id"] == a["cad_id"]), None)
                  for mode in results}
        changes.append({"cad_id": a["cad_id"], "expected_object_id": a["expected_object_id"],
                        "localization_prediction": a["prediction"], "sam_prediction": b["prediction"],
                        "prediction_changed": a["prediction"] != b["prediction"],
                        **{f"{mode}_common_margin": common[mode]["margin"] if common[mode] else None for mode in common}})
    timing["previous_sam_cloud_request_time_s"] = read_json(SAM / "run.json")["request_elapsed_seconds"]
    timing["comparison_wall_time_before_report_s"] = perf_counter() - start
    save_csv(output / "all_pair_scores.csv", flat)
    save_csv(output / "score_changes.csv", deltas)
    save_json(output / "comparison.json", {"evaluations": results, "prediction_changes": changes, "runtime": timing})
    save_report(output, results, changes, object_ids, manifest["targets"], images, timing)
    save_json(output / "protocol.json", {
        "only_changed_input": "SAM 3 mask versus raw projected localization mask; white background and same tight-crop/letterbox",
        "sam_source": str(SAM.relative_to(ROOT)), "ablation_1": str(ABLATION_1.relative_to(ROOT)),
        "cad_features_source": str(ABLATION_2.relative_to(ROOT)), "encoder": association.DINO_MODEL,
        "encoder_source": association.DINO_REPO, "checkpoint_sha256": checkpoint_hash,
        "score": "maximum raw normalized-CLS cosine over all 14 fixed CAD views",
        "ranking": "independent target queries; exact ties use object_id; no thresholds or assignment",
        "sam_labels_and_confidences_used_for_scoring": False,
        "missing_mask_policy": "null scores; no fallback or artificial image; six-target denominator retained",
        "common_five_policy": "same five candidates and five target CAD queries for both variants",
        "cad_ids": list(truth), "object_ids": object_ids, "sam_object_ids": sam_ids, "inputs": input_records,
        "sam_rerun": False, "geometry_used": False, "production_path_modified": False,
        "runtime": timing, "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__, "device": "cpu", "torch_threads": torch.get_num_threads()},
        "sam_source_sha256": sam_hashes, "source_sha256": association.content_hash(__file__),
    })
    assert snapshot_hashes() == fixed_hashes
    assert verify_artifacts(ABLATION_1) == a1_hashes and verify_artifacts(ABLATION_2) == a2_hashes
    assert all(association.content_hash(SAM / p) == h for p, h in sam_hashes.items())
    save_json(output / "validation.json", {
        "fixed_dataset_ablation_1_ablation_2_sam_unchanged": True,
        "localization_inputs_pixel_identical_to_ablation_1": True,
        "localization_rankings_equal_ablation_1": True, "max_localization_view_cosine_delta": max_delta,
        "localization_scored_pairs": 36, "sam_scored_pairs": 30, "sam_missing_pair_slots": 6,
        "views_per_available_pair": 14, "checkpoint_hash_matches_ablation_2": True,
    })
    (output / "SHA256SUMS").write_text("".join(
        f"{association.content_hash(p)}  {p.relative_to(output).as_posix()}\n"
        for p in sorted(output.rglob("*")) if p.is_file()), encoding="utf-8")
    for mode in results:
        print(mode, {scope: result["accuracy"] for scope, result in results[mode].items()}, flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
