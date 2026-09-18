"""Native 3840 RGB versus the fixed 768 wrist capture; no new segmentation.

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.run_wrist_resolution_ablation

Keep camera/scene, masks, templates, DINOv2-L/14, top-five semantics and
SAM-6D appearance scoring fixed. Save numeric results and actual input crops.
"""

import copy
import os
from datetime import datetime, timezone
from time import perf_counter

import cv2
import numpy as np
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import (
    ROOT, DATASET, read_json, read_rgb, save_image, save_json, snapshot_hashes,
)
from scripts.run_wrist_large_encoder import encode, pair_scores, evaluate_cases, ORACLE, SAM
from scripts.run_wrist_patch_ablation import ABLATION_1
from scripts.run_wrist_view_ablation import ABLATION_2
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.run_wrist_sam6d_patch_ablation import LARGE, prepare_masks
from scripts.sam6d_patch_matching import semantic_score, masked_patch_descriptors, appearance_score
from scripts.wrist_oracle_pose import SOURCE
from simulation.wrist_cluster import make_environment, CAMERA
from scripts.run_wrist_association import configure_scene


PREVIOUS = ROOT / "outputs/sam6d_patch_wrist_association_2026-09-11/20260917T181518155829Z"
OUTPUT_ROOT = ROOT / "outputs/resolution_wrist_association_2026-09-11"
SIZE = 3840
SCALE = SIZE // 768


def scaled_roi(roi, scale):
    """Scale inclusive pixel-cell bounds, retaining the entire last pixel."""
    x1, y1, x2, y2 = roi
    return [x1 * scale, y1 * scale, (x2 + 1) * scale - 1, (y2 + 1) * scale - 1]


def observation_input(rgb, roi, mask=None):
    """Re-use physical image regions by integer replication of saved mask cells."""
    scale = rgb.shape[0] // 768
    assert rgb.shape == (768 * scale, 768 * scale, 3) and scale >= 1
    if mask is not None:
        assert mask.shape == (768, 768) and mask.dtype == bool
        expanded = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        canvas, crop, bbox = prepare_masked_input(rgb, expanded)
    else:
        bbox = scaled_roi(roi, scale)
        x1, y1, x2, y2 = bbox
        crop = rgb[y1:y2 + 1, x1:x2 + 1].copy()
        canvas = association.letterbox_rgb(crop)
    return canvas, crop, bbox


def render_wrist(output):
    """Restore the exact captured geometry, verify 768, then render at 3840."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map,
    )
    source = read_json(SOURCE / "manifest.json")
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    catalog = copy.deepcopy(source["catalog"])
    for cid, record in catalog.items():
        path = ROOT / library[cid]["file_path"]
        assert association.content_hash(path) == record["cad_sha256"]
        record["cad_path"] = str(path)
    expected = read_rgb(DATASET / "scene/rgb.png")
    K = np.load(DATASET / "scene/intrinsics.npy")
    pose = np.load(DATASET / "scene/world_T_camera.npy")
    stage = perf_counter()
    with make_environment(catalog, source["placements"], additional_scene=configure_scene) as env:
        env.reset(seed=source["seed"])
        sim = env.sim
        sim.set_state_from_flattened(np.load(SOURCE / "initial_state.npy"))
        sim.data.qpos[:] = np.load(ORACLE / "ground_truth/captured_qpos.npy")
        sim.forward()
        raw, buffer = sim.render(width=768, height=768, camera_name=CAMERA, depth=True)
        np.testing.assert_array_equal(raw[::-1], expected)
        np.testing.assert_array_equal(get_real_depth_map(sim, buffer)[::-1].astype(np.float32),
                                      np.load(DATASET / "scene/depth.npy"))
        np.testing.assert_array_equal(get_camera_intrinsic_matrix(sim, CAMERA, 768, 768), K)
        np.testing.assert_array_equal(get_camera_extrinsic_matrix(sim, CAMERA), pose)
        state = sim.get_state().flatten().copy()
        samples = int(sim.model.vis.quality.offsamples)
        print("Original wrist RGB, depth, intrinsics and pose replay exactly", flush=True)
        setup_s = perf_counter() - stage
        stage = perf_counter()
        rgb = sim.render(width=SIZE, height=SIZE, camera_name=CAMERA)[::-1].copy()
        render_s = perf_counter() - stage
        assert rgb.shape == (SIZE, SIZE, 3)
        high_K = get_camera_intrinsic_matrix(sim, CAMERA, SIZE, SIZE)
        np.testing.assert_allclose(high_K, np.diag([SCALE, SCALE, 1]) @ K, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(get_camera_extrinsic_matrix(sim, CAMERA), pose)
        np.testing.assert_array_equal(sim.get_state().flatten(), state)
        assert int(sim.model.vis.quality.offsamples) == samples
        # A larger framebuffer can change last-bit rasterization on a smaller
        # viewport. Record it separately; the pre-resize baseline must be exact.
        after = sim.render(width=768, height=768, camera_name=CAMERA)[::-1]
        difference = np.abs(after.astype(np.int16) - expected.astype(np.int16))
        roundtrip = {"changed_rgb_channels": int(np.count_nonzero(difference)),
                     "max_channel_difference": int(difference.max()),
                     "mean_channel_difference": float(difference.mean())}
    save_image(output / "rgb_3840.png", rgb)
    np.save(output / "intrinsics_3840.npy", high_K)
    np.save(output / "world_T_camera.npy", pose)
    print(f"Rendered native {SIZE}x{SIZE} RGB in {render_s:.3f}s", flush=True)
    return rgb, {"scene_setup_and_exact_replay_s": setup_s, "native_3840_render_s": render_s}, {
        "original_rgb_depth_camera_replay_exact": True, "state_and_camera_unchanged": True,
        "post_resize_768_rasterization_diagnostic": roundtrip, "intrinsics_scale_exact": True,
        "antialias_samples_unchanged": samples,
    }


def main():
    start = perf_counter()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    fixed_hashes = snapshot_hashes()
    protected = ("cad_object_association.py", "CADPointCloudRegistration.py", "main.py",
                 "simulation/wrist_cluster.py", "simulation/nist_task_board_1.py")
    source_hashes = {p: association.content_hash(ROOT / p) for p in protected}
    old_manifest = read_json(LARGE / "inputs.json")
    old_protocol = read_json(LARGE / "protocol.json")
    checkpoint = ROOT / association.DINO_HUB_DIR / "checkpoints/dinov2_vitl14_pretrain.pth"
    assert association.content_hash(checkpoint) == old_protocol["models"]["large"]["checkpoint_sha256"]
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    rgb, timing, validation = render_wrist(output)
    low_rgb = read_rgb(DATASET / "scene/rgb.png")
    audit = read_json(DATASET / "identity_audit.json")
    targets = read_json(DATASET / "manifest.json")["targets"]
    inputs, input_records, mask_hashes = {}, {}, {}
    for mode in ("bbox", "localization", "sam3"):
        (output / mode / "inputs").mkdir(parents=True)
        (output / mode / "crops").mkdir()
        for item in audit:
            oid = item["object_id"]
            key = f"observed/{mode}/{oid}"
            if key not in old_manifest:
                assert mode == "sam3" and oid == "object_001"
                continue
            roi = [item["roi"][k] for k in ("x1", "y1", "x2", "y2")]
            mask = None
            if mode != "bbox":
                path = (ABLATION_1 / "masked/masks" if mode == "localization" else SAM / "sam/masks") / f"{oid}.png"
                mask_hashes[path.relative_to(ROOT).as_posix()] = association.content_hash(path)
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            old_path = ROOT / old_manifest[key]["path"]
            assert association.content_hash(old_path) == old_manifest[key]["sha256"]
            old, low_crop, low_box = observation_input(low_rgb, roi, mask)
            np.testing.assert_array_equal(old, read_rgb(old_path))
            canvas, crop, bbox = observation_input(rgb, roi, mask)
            assert bbox == scaled_roi(low_box, SCALE)
            assert crop.shape[:2] == tuple(SCALE * x for x in low_crop.shape[:2])
            inputs[key] = canvas
            path = output / mode / "inputs" / f"{oid}.png"
            save_image(path, canvas)
            save_image(output / mode / "crops" / f"{oid}.png", crop)
            input_records[key] = {"path": path.relative_to(ROOT).as_posix(),
                "sha256": association.content_hash(path), "baseline": old_manifest[key],
                "bbox_768_inclusive": low_box, "bbox_3840_inclusive": bbox,
                "native_crop_768_hwc": list(low_crop.shape), "native_crop_3840_hwc": list(crop.shape),
                "mean_absolute_224_input_pixel_change": float(np.abs(canvas.astype(float) - old).mean())}
    del rgb
    with np.load(LARGE / "large/features.npz") as f:
        old_cls = f["cls_features"]
        cad_patches = dict(zip(f["patch_indices"].tolist(), f["patch_tokens"]))
    with np.load(PREVIOUS / "additional_features/features.npz") as f:
        original = read_json(PREVIOUS / "additional_features/index.json")["original_feature_indices"]
        cad_patches.update(dict(zip(original, f["patch_tokens"])))
    keys = list(inputs)
    controls = [f"observed/{mode}/object_006" for mode in ("bbox", "localization", "sam3")]
    images = list(inputs.values()) + [read_rgb(ROOT / old_manifest[k]["path"]) for k in controls]
    (output / "observed_features").mkdir()
    cls, patches, encoder_timing = encode("dinov2_vitl14", 1024, images, set(range(len(images))), output / "observed_features")
    np.testing.assert_allclose(cls[len(keys):], np.stack([old_cls[old_manifest[k]["feature_index"]] for k in controls]), rtol=0, atol=1e-12)
    timing["observed_encoder"] = encoder_timing
    save_json(output / "observed_features/index.json", {"native_3840_input_keys": keys, "768_control_input_keys": controls})
    high_cls, high_patch = dict(zip(keys, cls)), {k: patches[i] for i, k in enumerate(keys)}
    rows, needed = {}, set()
    stage = perf_counter()
    for family, count in (("original14", 14), ("blenderproc42", 42)):
        for mode in ("bbox", "localization", "sam3"):
            for size in (768, SIZE):
                group = f"{family}/{mode}/{size}"
                rows[group] = []
                for cid in targets:
                    template_keys = [f"{family}/{cid}/{v}" for v in range(1, count + 1)]
                    templates = np.stack([old_cls[old_manifest[k]["feature_index"]] for k in template_keys])
                    for key in keys:
                        if key.split("/")[1] != mode:
                            continue
                        observed = high_cls[key] if size == SIZE else old_cls[old_manifest[key]["feature_index"]]
                        score = semantic_score(observed, templates)
                        template = template_keys[score["best_view_index_1based"] - 1]
                        rows[group].append({"cad_id": cid, "object_id": key.split("/")[-1],
                            "observed_key": key, "best_template_id": template, **score})
                        if mode != "bbox":
                            needed.add(template)
    timing["global_scoring_s"] = perf_counter() - stage
    missing = sorted({old_manifest[k]["feature_index"] for k in needed} - cad_patches.keys())
    if missing:
        index_to_key = {old_manifest[k]["feature_index"]: k for k in sorted(needed)}
        (output / "additional_template_features").mkdir()
        extra_cls, extra_patch, extra_timing = encode("dinov2_vitl14", 1024,
            [read_rgb(ROOT / old_manifest[index_to_key[i]]["path"]) for i in missing],
            set(range(len(missing))), output / "additional_template_features")
        np.testing.assert_allclose(extra_cls, old_cls[missing], rtol=0, atol=1e-12)
        cad_patches.update({index: extra_patch[i] for i, index in enumerate(missing)})
        timing["additional_template_encoder"] = extra_timing
        save_json(output / "additional_template_features/index.json", {"original_feature_indices": missing})
    mask_keys = {k for k in keys if "/bbox/" not in k}
    weights, other_mask_hashes = prepare_masks(needed | mask_keys, old_manifest, output)
    mask_hashes.update(other_mask_hashes)
    # The 224x224 foreground gate is identical at both camera resolutions.
    reference = {k: masked_patch_descriptors(cad_patches[old_manifest[k]["feature_index"]], weights[k]) for k in needed}
    low_query = {k: masked_patch_descriptors(cad_patches[old_manifest[k]["feature_index"]], weights[k]) for k in mask_keys}
    high_query = {k: masked_patch_descriptors(high_patch[k], weights[k]) for k in mask_keys}
    old_pairs = {(r["cad_id"], r["object_id"]): r for r in read_json(ABLATION_2 / "pair_scores.json")}
    previous_rows = read_json(PREVIOUS / "pair_scores.json")
    cases = {}
    stage = perf_counter()
    for group, records in rows.items():
        family, mode, size = group.split("/")
        for row in records:
            query_key, template = row["observed_key"], row["best_template_id"]
            scores = {"global_only": row["semantic_score"]}
            if mode != "bbox":
                query = high_query[query_key] if int(size) == SIZE else low_query[query_key]
                local = appearance_score(query, reference[template])
                row.update({k: v for k, v in local.items() if k != "observed_patch_maxima"})
                row["combined_visual_score"] = .5 * (row["semantic_score"] + row["appearance_score"])
                scores["global_plus_patch"] = row["combined_visual_score"]
                if int(size) == 768:
                    old = next(r for r in previous_rows[f"{family}/{mode}"] if r["cad_id"] == row["cad_id"] and r["object_id"] == row["object_id"])
                    assert row["semantic_score"] == old["semantic_score"]
                    assert row["appearance_score"] == old["appearance_score"]
                    assert row["best_template_id"] == old["best_template_id"]
            geometry = {k: old_pairs[row["cad_id"], row["object_id"]][k] for k in
                        ("cad_id", "object_id", "geometry_raw", "geometry_score")}
            for variant, score in scores.items():
                cases.setdefault(f"{group}/{variant}", []).append(pair_scores(geometry, score, row["best_view_index_1based"]))
    timing["patch_scoring_and_fusion_s"] = perf_counter() - stage
    evaluations = evaluate_cases(cases, targets, audit)
    changes = []
    for name, result in evaluations.items():
        if f"/{SIZE}/" not in name:
            continue
        baseline = evaluations[name.replace(f"/{SIZE}/", "/768/")]
        for a, b, am, bm in zip(baseline["per_target"], result["per_target"], baseline["margins"], result["margins"], strict=True):
            changes.append({"case": name, "cad_id": b["cad_id"], "expected_object_id": b["expected_object_id"],
                "predictions_768": a["predictions"], "predictions_3840": b["predictions"],
                "margins_768": am, "margins_3840": bm})
    for k in old_manifest:
        if k.startswith(("original14/", "blenderproc42/")):
            assert association.content_hash(ROOT / old_manifest[k]["path"]) == old_manifest[k]["sha256"]
    validation.update({"fixed_dataset_unchanged": snapshot_hashes() == fixed_hashes,
        "production_and_scene_source_unchanged": all(association.content_hash(ROOT / p) == h for p, h in source_hashes.items()),
        "saved_masks_unchanged": all(association.content_hash(ROOT / p) == h for p, h in mask_hashes.items()),
        "all_336_template_image_hashes_match": True, "768_pin_feature_controls_match": True,
        "all_132_previous_masked_pair_scores_reproduced_exactly": True,
        "all_17_baseline_observation_inputs_reproduced_exactly": True,
        "scored_pair_count_including_both_resolutions": sum(map(len, rows.values()))})
    assert all(validation[k] for k in ("fixed_dataset_unchanged", "production_and_scene_source_unchanged", "saved_masks_unchanged"))
    save_json(output / "inputs.json", input_records)
    save_json(output / "pair_scores.json", rows)
    save_json(output / "evaluations.json", evaluations)
    save_json(output / "comparison.json", {"summary": {k: {"accuracy": v["accuracy"], "mean_margins": v["mean_margins"]} for k, v in evaluations.items()}, "per_target": changes})
    save_json(output / "protocol.json", {"camera": CAMERA, "resolutions": [768, SIZE],
        "change": "Native RGB camera render resolution only; exact same restored scene, poses, FoV, lights, materials and antialiasing",
        "masks": "Existing full-image localization/SAM3 masks replicated 5x in each axis; no new segmentation or 3D projection; same mask extent and same 224x224 patch gate",
        "scope": "RGB detail ablation with frozen segmentation boundaries; does not test benefits of rerunning localization/SAM3 at higher resolution",
        "encoder": "dinov2_vitl14, frozen 224x224 white bicubic letterbox, ImageNet normalization",
        "checkpoint_sha256": old_protocol["models"]["large"]["checkpoint_sha256"],
        "templates": "Same saved original14 and BlenderProc42 images/features; no template rendering",
        "scoring": "SAM-6D top-five mean CLS and optional equal mean with best-CLS-view foreground patch score",
        "geometry": "Unchanged recorded full-CAD fitness; secondary fusion is 0.5*((visual+1)/2)+0.5*geometry",
        "sam3_missing": "No saved pulley mask; SAM3 evaluates five targets and five candidates, retaining all 30 available pair scores",
        "source_features": str(LARGE.relative_to(ROOT)), "previous_patch_experiment": str(PREVIOUS.relative_to(ROOT)),
        "source_state": str((SOURCE / "initial_state.npy").relative_to(ROOT)),
        "source_hashes": source_hashes, "mask_hashes": mask_hashes,
        "implementation_sha256": association.content_hash(ROOT / "scripts/run_wrist_resolution_ablation.py"),
        "production_modified": False, "geometry_rerun": False, "segmentation_rerun": False})
    save_json(output / "validation.json", validation)
    timing["wall_time_s"] = perf_counter() - start
    save_json(output / "runtime.json", timing)
    for name, result in evaluations.items():
        v, f = result["accuracy"]["visual"], result["accuracy"]["fused"]
        print(f"{name}: visual {v['correct']}/{v['total']}, fused {f['correct']}/{f['total']}", flush=True)
    print(f"SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
