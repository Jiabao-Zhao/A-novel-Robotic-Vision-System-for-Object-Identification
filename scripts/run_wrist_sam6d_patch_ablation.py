"""SAM-6D semantic + appearance scoring on the fixed, saved Large-encoder inputs.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.run_wrist_sam6d_patch_ablation

Scoring reproduction only: keep our white/bicubic inputs, masks, 14/42 views,
exhaustive fixed-CAD queries, recorded geometry and existing geometry fusion.
No segmentation, rendering, pose estimation, production edits or text report.
"""

import os
from datetime import datetime, timezone
from time import perf_counter

import cv2
import numpy as np
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, read_rgb, save_json, snapshot_hashes
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy, score_pair
from scripts.run_wrist_view_ablation import ABLATION_2
from scripts.run_wrist_large_encoder import BLENDER, SAM, encode, pair_scores, evaluate_cases
from scripts.run_wrist_top5_semantic_ablation import template_rotation
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.sam6d_patch_matching import (
    semantic_score, masked_patch_descriptors, appearance_score, combined_visual_score,
)


LARGE = ROOT / "outputs/large_encoder_wrist_association_2026-09-11/20260917T173743688156Z"
UPSTREAM = ROOT / "outputs/cache/sam6d_matching_source"
UPSTREAM_COMMIT = "1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2"
OUTPUT_ROOT = ROOT / "outputs/sam6d_patch_wrist_association_2026-09-11"
VARIANTS = {
    "global_only": "semantic_score",
    "previous_patch_only": "occupancy_weighted_patch_score",
    "global_plus_previous_patch": "semantic_plus_occupancy_score",
    "sam6d_patch_only": "appearance_score",
    "global_plus_sam6d_patch": "semantic_plus_appearance_score",
}


def prepare_masks(keys, manifest, output):
    """Reproduce each saved RGB crop's binary mask coordinates; change no RGB."""
    with np.load(ABLATION_2 / "features/observed.npz") as f:
        observed = dict(zip(f["object_ids"].tolist(), f["object_masks"]))
    cad_masks = {}
    for cid in read_json(DATASET / "manifest.json")["targets"]:
        with np.load(ABLATION_2 / "features" / f"{cid}.npz") as f:
            cad_masks[cid] = f["object_masks"]
    templates42 = {(r["cad_id"], r["view_index_1based"]): r for r in read_json(BLENDER / "templates.json")}
    rgb = read_rgb(DATASET / "scene/rgb.png")
    masks, provenance = {}, {}
    for key in sorted(keys):
        family, name, index = key.split("/")
        if family == "observed" and name == "localization":
            masks[key] = observed[index]
            source = ABLATION_2 / "features/observed.npz"
        elif family == "original14":
            masks[key] = cad_masks[name][int(index) - 1]
            source = ABLATION_2 / "features" / f"{name}.npz"
        else:
            source = (SAM / "sam/masks" / f"{index}.png" if family == "observed" else
                      BLENDER / "templates" / name / f"view_{int(index):02}_mask.png")
            full = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if full is None or not set(np.unique(full)) <= {0, 255}:
                raise ValueError(f"Missing/nonbinary saved object mask: {source}")
            full = full > 0
            y, x = np.where(full)
            bbox = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
            if family == "observed":
                expected = read_rgb(ROOT / manifest[key]["path"])
                np.testing.assert_array_equal(prepare_masked_input(rgb, full)[0], expected)
            else:
                assert bbox == templates42[name, int(index)]["crop_bbox_xyxy"]
            x1, y1, x2, y2 = bbox
            masks[key] = letterbox_mask(full[y1:y2 + 1, x1:x2 + 1])
        provenance[source.relative_to(ROOT).as_posix()] = association.content_hash(source)
    weights = {key: patch_occupancy(mask) for key, mask in masks.items()}
    np.savez_compressed(output / "masks.npz", keys=list(masks), object_masks=np.stack(list(masks.values())),
                        patch_occupancy=np.stack(list(weights.values())))
    return weights, provenance


def main():
    start = perf_counter()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    fixed_hashes = snapshot_hashes()
    production_hashes = {p: association.content_hash(ROOT / p) for p in
                         ("cad_object_association.py", "CADPointCloudRegistration.py", "main.py")}
    upstream_hashes = {p.relative_to(UPSTREAM).as_posix(): association.content_hash(p)
                       for p in sorted(UPSTREAM.rglob("*")) if p.is_file()}
    assert all(name in upstream_hashes for name in ("model/loss.py", "model/dinov2.py", "model/detector.py"))
    old_protocol = read_json(LARGE / "protocol.json")
    assert old_protocol["models"]["large"]["name"] == "dinov2_vitl14"
    checkpoint = ROOT / association.DINO_HUB_DIR / "checkpoints/dinov2_vitl14_pretrain.pth"
    assert association.content_hash(checkpoint) == old_protocol["models"]["large"]["checkpoint_sha256"]
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    manifest = read_json(LARGE / "inputs.json")
    with np.load(LARGE / "large/features.npz") as f:
        cls = f["cls_features"]
        patch_by_index = dict(zip(f["patch_indices"].tolist(), f["patch_tokens"]))
    assert cls.shape == (417, 1024)
    feature = lambda key: cls[manifest[key]["feature_index"]]
    targets = read_json(DATASET / "manifest.json")["targets"]
    audit = read_json(DATASET / "identity_audit.json")
    object_ids = sorted(r["object_id"] for r in audit)
    old_pairs = {(r["cad_id"], r["object_id"]): r for r in read_json(ABLATION_2 / "pair_scores.json")}
    old_evaluations = read_json(LARGE / "large/evaluations.json")
    needed, semantic_pairs = set(), {}
    for family, count in (("original14", 14), ("blenderproc42", 42)):
        for mode in ("localization", "sam3"):
            group = f"{family}/{mode}"
            semantic_pairs[group] = []
            for cid in targets:
                cad_cls = np.stack([feature(f"{family}/{cid}/{v}") for v in range(1, count + 1)])
                for oid in object_ids:
                    observed_key = f"observed/{mode}/{oid}"
                    if observed_key not in manifest:
                        assert mode == "sam3" and oid == "object_001"
                        continue
                    result = semantic_score(feature(observed_key), cad_cls)
                    template_key = f"{family}/{cid}/{result['best_view_index_1based']}"
                    needed.update((observed_key, template_key))
                    semantic_pairs[group].append({"cad_id": cid, "object_id": oid,
                        "observed_key": observed_key, "template_key": template_key, **result})
    assert {k: len(v) for k, v in semantic_pairs.items()} == {
        "original14/localization": 36, "original14/sam3": 30,
        "blenderproc42/localization": 36, "blenderproc42/sam3": 30}
    missing = sorted({manifest[key]["feature_index"] for key in needed} - patch_by_index.keys())
    key_by_index = {manifest[key]["feature_index"]: key for key in sorted(needed)}
    selected_manifest = {key: manifest[key] for key in sorted(needed)}
    for record in selected_manifest.values():
        assert association.content_hash(ROOT / record["path"]) == record["sha256"]
    save_json(output / "inputs.json", selected_manifest)
    timing = {}
    if missing:
        print(f"Need patch tokens for {len(missing)} saved inputs; reusing all CLS and existing patch features", flush=True)
        (output / "additional_features").mkdir()
        images = [read_rgb(ROOT / manifest[key_by_index[i]]["path"]) for i in missing]
        extracted_cls, new_patches, timing = encode("dinov2_vitl14", 1024, images,
            set(range(len(images))), output / "additional_features")
        np.testing.assert_allclose(extracted_cls, cls[missing], rtol=0, atol=1e-12)
        timing["max_reextracted_cls_error"] = float(np.max(np.abs(extracted_cls - cls[missing])))
        patch_by_index.update({original: new_patches[i] for i, original in enumerate(missing)})
        save_json(output / "additional_features/index.json", {"original_feature_indices": missing,
            "input_keys": [key_by_index[i] for i in missing]})
    weights, mask_hashes = prepare_masks(needed, manifest, output)
    stage = perf_counter()
    masked = {key: masked_patch_descriptors(patch_by_index[manifest[key]["feature_index"]], weights[key]) for key in needed}
    rotations42 = np.load(BLENDER / "cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    cases, pair_records = {}, {}
    for group, pairs in semantic_pairs.items():
        family, mode = group.split("/")
        pair_records[group] = []
        for pair in pairs:
            cid, oid = pair["cad_id"], pair["object_id"]
            qkey, rkey = pair["observed_key"], pair["template_key"]
            query = patch_by_index[manifest[qkey]["feature_index"]]
            reference = patch_by_index[manifest[rkey]["feature_index"]]
            appearance = appearance_score(masked[qkey], masked[rkey])
            # Hold top-five CLS and the selected template fixed when comparing patch rules.
            previous = score_pair(feature(qkey), query, weights[qkey], feature(rkey)[None],
                                  reference[None], weights[rkey][None])["patch_score"]
            view = pair["best_view_index_1based"]
            rotation = (rotations42[view - 1] if family == "blenderproc42" else
                        template_rotation(association.VIEW_DIRECTIONS[view - 1]))
            row = {**pair, **appearance, "occupancy_weighted_patch_score": previous,
                "semantic_plus_occupancy_score": combined_visual_score(pair["semantic_score"], previous),
                "semantic_plus_appearance_score": combined_visual_score(pair["semantic_score"], appearance["appearance_score"]),
                "observed_mask_retained_patch_count": int((weights[qkey] > .5).sum()),
                "cad_mask_retained_patch_count": int((weights[rkey] > .5).sum()),
                "best_template_id": rkey, "R_template_camera_from_cad": rotation.tolist()}
            pair_records[group].append(row)
            geometry = {k: old_pairs[cid, oid][k] for k in ("cad_id", "object_id", "geometry_raw", "geometry_score")}
            for variant, key in VARIANTS.items():
                cases.setdefault(f"{group}/{variant}", []).append(pair_scores(geometry, row[key], view))
        print(f"Scored {group}: all {len(pairs)} available pairs", flush=True)
    timing["scoring_all_pairs_s"] = perf_counter() - stage
    for group in ("original14/localization", "blenderproc42/localization"):
        for variant in VARIANTS:
            key = f"{group}/{variant}"
            cases[f"{key}/common5"] = [r for r in cases[key] if r["object_id"] != "object_001"]
    results = evaluate_cases(cases, targets, audit)
    baseline_errors = []
    for group in semantic_pairs:
        before, after = old_evaluations[f"{group}/top5"], results[f"{group}/global_only"]
        assert before["accuracy"] == after["accuracy"]
        assert [r["predictions"] for r in before["per_target"]] == [r["predictions"] for r in after["per_target"]]
        for a, b in zip(before["margins"], after["margins"], strict=True):
            baseline_errors.append(abs(a["visual"]["correct_minus_best_incorrect"] - b["visual"]["correct_minus_best_incorrect"]))
    assert max(baseline_errors) < 1e-6
    changes = []
    for name, result in results.items():
        suffix = "/common5" if name.endswith("/common5") else ""
        group = "/".join(name.split("/")[:2])
        baseline = results[f"{group}/global_only{suffix}"]
        for target, a, b in zip(result["per_target"], baseline["margins"], result["margins"], strict=True):
            i = next(i for i, t in enumerate(baseline["per_target"]) if t["cad_id"] == target["cad_id"])
            for mode in ("visual", "fused"):
                changes.append({"case": name, "cad_id": target["cad_id"], "mode": mode,
                    "global_prediction": baseline["per_target"][i]["predictions"][mode],
                    "prediction": target["predictions"][mode], "expected_object_id": target["expected_object_id"],
                    "correct_score": b[mode]["correct_score"],
                    "strongest_incorrect_object_id": b[mode]["strongest_incorrect_object_id"],
                    "strongest_incorrect_score": b[mode]["strongest_incorrect_score"],
                    "global_margin": a[mode]["correct_minus_best_incorrect"],
                    "margin": b[mode]["correct_minus_best_incorrect"],
                    "margin_delta": b[mode]["correct_minus_best_incorrect"] - a[mode]["correct_minus_best_incorrect"]})
    save_json(output / "pair_scores.json", pair_records)
    save_json(output / "evaluations.json", results)
    save_json(output / "comparison.json", {"summary": {name: {"accuracy": r["accuracy"],
        "mean_margins": r["mean_margins"]} for name, r in results.items()}, "per_target": changes})
    save_json(output / "protocol.json", {
        "upstream_commit": UPSTREAM_COMMIT, "upstream_source_sha256": upstream_hashes,
        "upstream_base_url": f"https://github.com/JiehongLin/SAM-6D/tree/{UPSTREAM_COMMIT}/SAM-6D/Instance_Segmentation_Model",
        "encoder": "Frozen DINOv2-L/14, final CLS + 16x16 patch tokens, same checkpoint and image inputs as Large replay",
        "checkpoint_sha256": old_protocol["models"]["large"]["checkpoint_sha256"],
        "feature_source": str(LARGE.relative_to(ROOT)), "additional_patch_inputs": len(missing),
        "semantic": "PairwiseSimilarity: normalize descriptors, cosine, clamp each view to [0,1], mean top five; best single clamped CLS view",
        "appearance": "AvgPool14x14 mask >0.5; zero invalid patch tokens then L2 normalize; full 256x256 dot matrix including zero background; mean query-wise maxima using count_nonzero(query.sum(-1))+1e-6; clamp final to [0,1]",
        "visual_total": "(semantic_score + appearance_score)/2; fixed equal weights, no fitting",
        "reverse_visibility": "Upstream compute_visible_ratio with strict cosine>0.5; diagnostic only, not an extra visual weight",
        "previous_patch_control": "Same top-five semantic score and best-CLS view, replacing only hard-gated equal-mean appearance with our prior occupancy-weighted foreground matcher",
        "geometry_fusion": "Secondary evaluation only: unchanged 0.5*((visual_total+1)/2)+0.5*saved full-CAD fitness. This is our fusion, not SAM-6D Eq.4",
        "paper_full_score": "(semantic+appearance+visible_ratio*projected_bbox_IoU)/(2+visible_ratio); its geometry is not registration fitness and is not implemented by this visual ablation",
        "scope": "Port of SAM-6D's semantic/patch scoring, not a full pipeline reproduction. Evaluate every CAD independently against all available candidates; no semantic identity preassignment, confidence filtering or NMS",
        "preprocessing_difference": "Keep saved white-background OpenCV bicubic crops. Upstream query path normalizes RGB before multiplying by masks, then uses zero tensor padding and nearest interpolation. The custom demo template path feeds masked [0,1] RGB without ImageNet normalization; benchmark template loading is separate. Neither preprocessing path is silently substituted here",
        "sam3_missing": "Pulley has no saved SAM3 mask; SAM3 and common5 use five targets and five candidates. All 30 available CAD/candidate scores are retained",
        "input_and_masks": mask_hashes, "production_sha256": production_hashes,
        "implementation_sha256": {p: association.content_hash(ROOT / p) for p in
            ("scripts/run_wrist_sam6d_patch_ablation.py", "scripts/sam6d_patch_matching.py", "tests/test_wrist_sam6d_patch.py")},
        "geometry_segmentation_rendering_rerun": False, "production_modified": False})
    validation = {"reextracted_cls_match_saved": timing.get("max_reextracted_cls_error", 0.) == 0.,
        "global_only_predictions_reproduce_large_baseline": True, "max_global_margin_delta": max(baseline_errors),
        "pair_count": sum(map(len, pair_records.values())), "common5_excludes_pulley_in_both_axes": True,
        "fixed_dataset_unchanged": snapshot_hashes() == fixed_hashes,
        "production_unchanged": all(association.content_hash(ROOT / p) == h for p, h in production_hashes.items()),
        "source_masks_unchanged": all(association.content_hash(ROOT / p) == h for p, h in mask_hashes.items()),
        "source_inputs_unchanged": all(association.content_hash(ROOT / r["path"]) == r["sha256"] for r in selected_manifest.values())}
    assert all(validation[k] for k in ("fixed_dataset_unchanged", "production_unchanged", "source_masks_unchanged", "source_inputs_unchanged"))
    save_json(output / "validation.json", validation)
    timing["wall_time_s"] = perf_counter() - start
    timing["geometry_rendering_segmentation_s"] = 0.
    save_json(output / "runtime.json", timing)
    for name, result in results.items():
        if name.endswith("common5"):
            continue
        visual, fused = result["accuracy"]["visual"], result["accuracy"]["fused"]
        print(f"{name}: visual {visual['correct']}/{visual['total']}, fused {fused['correct']}/{fused['total']}", flush=True)
    print(f"SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
