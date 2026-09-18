"""Replay the saved six-object visual protocols with DINOv2-S/14 and L/14.

Run in the existing WSL environment with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1:
    python -m scripts.run_wrist_large_encoder

Only the encoder changes. Images, masks, poses and geometry are read-only.
No rendering, segmentation, registration, training, tuning or text reports.
"""

import gc
import os
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, read_rgb, save_json, snapshot_hashes
from scripts.run_wrist_patch_ablation import ABLATION_1, score_pair
from scripts.run_wrist_view_ablation import ABLATION_2, score_all_views
from scripts.run_wrist_geometry_ablation import collapse_views, evaluate_pairs
from scripts.run_wrist_coverage_ablation import ABLATION_3, rescore
from scripts.run_wrist_coverage_view_ablation import score_views, select_views
from scripts.run_wrist_top5_semantic_ablation import score_view_similarities


BLENDER = ROOT / "outputs/blenderproc42_semantic_wrist_association_2026-09-11/20260917T154958686983Z"
SAM = ROOT / "outputs/sam_visual_comparison_wrist_association_2026-09-11/20260914T151728934102Z"
ORACLE = ROOT / "outputs/oracle_pose_wrist_association_2026-09-11/20260914T192456038746Z"
APPEARANCE = ROOT / "outputs/appearance_consistency_wrist_association_2026-09-11/20260917T171044087595Z"
OUTPUT_ROOT = ROOT / "outputs/large_encoder_wrist_association_2026-09-11"
MODELS = {"small": ("dinov2_vits14", 384), "large": ("dinov2_vitl14", 1024)}
BATCH_SIZE = 8
PARITY_ATOL = 2e-5  # FP32 CPU/CUDA numerical check, never an association threshold.


def input_catalog(targets, object_ids):
    paths = {}
    for family, folder, count in (("original14", DATASET, 14), ("blenderproc42", BLENDER, 42)):
        for cid in targets:
            for view in range(1, count + 1):
                paths[f"{family}/{cid}/{view}"] = folder / "templates" / cid / f"view_{view:02}.png"
    for mode, folder in (("bbox", ABLATION_1 / "bbox/inputs"),
                         ("localization", ABLATION_2 / "inputs"), ("sam3", SAM / "sam/inputs")):
        for oid in object_ids:
            path = folder / f"{oid}.png"
            if mode == "sam3" and oid == "object_001":
                assert not path.exists(), "Keep the same five available SAM3 masks."
                continue
            paths[f"observed/{mode}/{oid}"] = path
    for path in sorted((APPEARANCE / "inputs").glob("*.png")):
        paths[f"appearance/{path.stem}"] = path
    for cid in targets:
        for oid in object_ids:
            paths[f"oracle/{cid}/{oid}"] = ORACLE / "inputs" / f"{cid}__{oid}_cad.png"
    return paths


def encode(model_name, dimension, images, patch_indices, output, device="cpu"):
    """Match existing preprocessing and FP32 inference; normalize in float64."""
    # Existing replays retain CPU numerics; new experiments may explicitly use CUDA.
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    stage = perf_counter()
    torch.hub.set_dir(str(ROOT / association.DINO_HUB_DIR))
    model = torch.hub.load(association.DINO_REPO, model_name, pretrained=True,
                           trust_repo=True, skip_validation=True).eval().requires_grad_(False).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    runtime = {"device": device, "model_load_s": perf_counter() - stage}
    assert model.patch_size == 14 and model.embed_dim == dimension and not model.training
    cls, patches = [], {}
    stage = perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), BATCH_SIZE):
            tensors = []
            for image in images[start:start + BATCH_SIZE]:
                assert image.shape == (224, 224, 3)
                np.testing.assert_array_equal(association.letterbox_rgb(image), image)
                normalized = (image.astype(np.float32) / 255 - [0.485, .456, .406]) / [.229, .224, .225]
                tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)))
            features = model.forward_features(torch.stack(tensors).to(device))
            raw = features["x_norm_clstoken"].cpu().numpy().astype(np.float64)
            assert raw.shape == (len(tensors), dimension)
            raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
            cls.extend(raw)
            for offset in range(len(tensors)):
                index = start + offset
                if index in patch_indices:
                    p = features["x_norm_patchtokens"][offset].cpu().numpy().astype(np.float64)
                    assert p.shape == (256, dimension)
                    p /= np.linalg.norm(p, axis=-1, keepdims=True)
                    patches[index] = p.reshape(16, 16, dimension)
            if start % 80 == 0:
                print(f"{model_name}: encoded {min(start + BATCH_SIZE, len(images))}/{len(images)}", flush=True)
    runtime["feature_extraction_s"] = perf_counter() - stage
    runtime["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    runtime["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if device == "cuda" else None
    cls = np.stack(cls)
    assert np.isfinite(cls).all() and all(np.isfinite(p).all() for p in patches.values())
    np.testing.assert_allclose(np.linalg.norm(cls, axis=-1), 1, atol=1e-12)
    np.savez_compressed(output / "features.npz", cls_features=cls,
                        patch_indices=list(patches), patch_tokens=np.stack(list(patches.values())))
    del features, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return cls, patches, runtime


def pair_scores(pair, value, winning_view=None):
    normalized = association.normalize_visual_score(value)
    return {**pair, "visual_raw": float(value), "visual_score": normalized,
            "fused_score": association.fuse_scores(normalized, pair["geometry_score"]),
            "visual_winning_view_index_1based": winning_view}


def evaluate_cases(cases, targets, audit):
    """Use existing CAD-to-candidate rankings, margins and fusion evaluation."""
    truth = {r["simulator_instance"]: r["object_id"] for r in audit}
    results = {}
    for name, rows in cases.items():
        candidates = {r["object_id"] for r in rows}
        included = {cid: label for cid, label in targets.items() if truth[cid] in candidates}
        usable = [r for r in rows if r["cad_id"] in included]
        assert len(usable) == len(included) * len(candidates)
        result = evaluate_pairs(name, usable, included, audit)
        if name.startswith("appearance/"):
            # These earlier controls only evaluated visual similarity. Do not imply
            # that a matched-renderer geometric experiment has been performed.
            result = {"accuracy": {"visual": result["accuracy"]["visual"]},
                      "mean_margins": {"visual": result["mean_margins"]["visual"]},
                      "per_target": [{"cad_id": r["cad_id"], "expected_object_id": r["expected_object_id"],
                                      "predictions": {"visual": r["predictions"]["visual"]}}
                                     for r in result["per_target"]],
                      "margins": [{"cad_id": m["cad_id"], "expected_object_id": m["expected_object_id"],
                                   "visual": m["visual"]} for m in result["margins"]],
                      "rankings": {r["cad_id"]: r["rankings"]["visual"] for r in result["associations"]}}
        results[name] = result
    return results


def score_protocols(cls, patches, index, targets, object_ids, saved_pairs, saved_views, weights):
    feature = lambda key: cls[index[key]]
    common = [oid for oid in object_ids if f"observed/sam3/{oid}" in index]
    cases, per_view, patch_pairs = {}, {}, []
    rotations42 = np.load(BLENDER / "cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    # The current semantic experiments: bbox/masked/SAM3, 14/42 views, max/top-five.
    for family, modes, count in (("original14", ("bbox", "localization", "sam3"), 14),
                                 ("blenderproc42", ("localization", "sam3"), 42)):
        for mode in modes:
            name = f"{family}/{mode}"
            per_view[name] = []
            for old in saved_pairs:
                cid, oid = old["cad_id"], old["object_id"]
                observed_key = f"observed/{mode}/{oid}"
                if observed_key not in index:
                    continue
                cad = np.stack([feature(f"{family}/{cid}/{v}") for v in range(1, count + 1)])
                cosines = np.clip(cad @ feature(observed_key), -1, 1)
                scored = score_view_similarities(cid, cosines, rotations42 if count == 42 else None)
                per_view[name].append({"cad_id": cid, "object_id": oid, **scored})
                for aggregation, key in (("max", "max_view_score"), ("top5", "semantic_score")):
                    if mode == "bbox" and aggregation == "top5":
                        continue  # No prior bbox/top-five experiment.
                    row = pair_scores(old, scored[key], scored["best_view_index_1based"])
                    row.update(best_template_id=scored["best_template_id"],
                               best_template_rotation_camera_from_cad=scored["best_template_rotation_camera_from_cad"])
                    cases.setdefault(f"{name}/{aggregation}", []).append(row)
    for family in ("original14", "blenderproc42"):
        for aggregation in ("max", "top5"):
            key = f"{family}/localization/{aggregation}"
            cases[f"{key}/common5"] = [r for r in cases[key] if r["object_id"] in common]
    # Ablations 2 and 2b retain exactly the saved foreground occupancy masks.
    for old in saved_pairs:
        cid, oid = old["cad_id"], old["object_id"]
        obs_index = object_ids.index(oid)
        cad_cls = np.stack([feature(f"original14/{cid}/{v}") for v in range(1, 15)])
        cad_patch = np.stack([patches[index[f"original14/{cid}/{v}"]] for v in range(1, 15)])
        args = (feature(f"observed/localization/{oid}"), patches[index[f"observed/localization/{oid}"]],
                weights["observed"][obs_index], cad_cls, cad_patch, weights[cid])
        all_views = score_all_views(*args)
        fixed = score_pair(*args)
        np.testing.assert_allclose(fixed["patch_score"],
                                  all_views["patch_view_scores"][fixed["selected_view_index_1based"] - 1], atol=1e-12)
        patch_pairs.append({"cad_id": cid, "object_id": oid, **all_views})
        for label, values in (("ablation2", fixed), ("ablation2b", all_views)):
            for variant, score_key in (("patch_only", "patch_score"), ("cls_patch", "cls_patch_score")):
                view = fixed["selected_view_index_1based"] if label == "ablation2" else all_views["winning_view_index_1based"][variant]
                cases.setdefault(f"{label}/{variant}", []).append(pair_scores(old, values[score_key], view))
    # Ablations 3/3b/3c rescore only the 504 saved registrations/coverage values.
    by_pair = {(p["cad_id"], p["object_id"]): p for p in per_view["original14/localization"]}
    geometry_views = []
    for old in saved_views:
        value = by_pair[old["cad_id"], old["object_id"]]["per_view_cosines"][old["view_index_1based"] - 1]
        normalized = association.normalize_visual_score(value)
        geometry_views.append({**old, "cls_cosine": value, "normalized_cls": normalized,
                               "fused_score": association.fuse_scores(normalized, old["geometry_fitness"])})
    f1_views = score_views(geometry_views)
    for old in saved_pairs:
        pair_key = old["cad_id"], old["object_id"]
        matches = lambda rows: [r for r in rows if (r["cad_id"], r["object_id"]) == pair_key]
        current, original = matches(geometry_views), matches(saved_views)
        cases.setdefault("ablation3/visible_C_obs", []).append(collapse_views(current))
        cases.setdefault("ablation3c/reselected_C_f1", []).append(select_views(matches(f1_views)))
        # Preserve 3b's original frozen view exactly for the encoder-only test.
        # Also replay its two-stage selection protocol with each encoder's C_obs winner.
        for label, winner in (("saved_small_view", max(original, key=lambda r: r["fused_score"])),
                              ("encoder_C_obs_view", max(current, key=lambda r: r["fused_score"]))):
            record = next(r for r in current if r["view_index_1based"] == winner["view_index_1based"])
            assert record["C_f1"] is not None
            for geometry_key in ("C_obs", "C_f1"):
                cases.setdefault(f"ablation3b/{label}/{geometry_key}", []).extend(rescore([record], geometry_key))
    # Earlier exact-pose diagnostic, with its original recorded fixed-pose geometry.
    for old in read_json(ORACLE / "all_pairs.json"):
        cid, oid = old["cad_id"], old["object_id"]
        value = float(np.clip(feature(f"oracle/{cid}/{oid}") @ feature(f"observed/localization/{oid}"), -1, 1))
        for mode in ("full_cad", "visible_cad"):
            row = {"cad_id": cid, "object_id": oid, "geometry_score": old[mode]["registration_fitness"]}
            cases.setdefault(f"oracle_pose/{mode}", []).append(pair_scores(row, value))
    # Latest exact-pose renderer/mask consistency controls use their saved RGB directly.
    for renderer in ("raycast", "matched"):
        for mode in ("localization", "sam3", "oracle"):
            name = f"appearance/{renderer}_{mode}"
            cases[name] = []
            for old in saved_pairs:
                cid, oid = old["cad_id"], old["object_id"]
                observed = f"appearance/observed__{mode}__{oid}"
                if observed not in index:
                    continue
                cad = f"appearance/{renderer}__{cid}__{oid}"
                value = float(np.clip(feature(cad) @ feature(observed), -1, 1))
                cases[name].append(pair_scores(old, value))
            if mode != "sam3":
                cases[f"{name}/common5"] = [r for r in cases[name] if r["object_id"] in common]
    return cases, {"semantic": per_view, "patch": patch_pairs, "geometry": geometry_views}


def validate_small(cases, per_view, cls, patches, index, targets, object_ids):
    """Reproduce saved feature, per-view score and exact-pose controls before comparison."""
    errors = {}
    for cid in targets:
        with np.load(ABLATION_2 / "features" / f"{cid}.npz") as old:
            ids = [index[f"original14/{cid}/{v}"] for v in range(1, 15)]
            errors[f"{cid}_cls"] = float(np.max(np.abs(cls[ids] - old["cls_features"])))
            errors[f"{cid}_patch"] = float(np.max(np.abs(np.stack([patches[i] for i in ids]) - old["patch_tokens"])))
    with np.load(ABLATION_2 / "features/observed.npz") as old:
        assert old["object_ids"].tolist() == object_ids
        ids = [index[f"observed/localization/{oid}"] for oid in object_ids]
        errors["observed_cls"] = float(np.max(np.abs(cls[ids] - old["cls_features"])))
        errors["observed_patch"] = float(np.max(np.abs(np.stack([patches[i] for i in ids]) - old["patch_tokens"])))
    for name, folder in (("original14/localization", ABLATION_2), ("blenderproc42/localization", BLENDER)):
        baseline = {(p["cad_id"], p["object_id"]): p for p in read_json(folder / "pair_scores.json")}
        score_key = "cls_view_cosines" if folder == ABLATION_2 else "per_view_cosines"
        errors[name] = max(float(np.max(np.abs(np.asarray(p["per_view_cosines"]) -
                            baseline[p["cad_id"], p["object_id"]][score_key]))) for p in per_view["semantic"][name])
    previous = read_json(APPEARANCE / "results.json")["pair_scores"]
    for name, rows in previous.items():
        key = f"appearance/{name}"
        if key not in cases:
            continue
        now = {(p["cad_id"], p["object_id"]): p for p in cases[key]}
        errors[key] = max(abs(p["cosine"] - now[p["cad_id"], p["object_id"]]["visual_raw"]) for p in rows)
    assert max(errors.values()) < PARITY_ATOL, errors
    return errors


def comparison(small, large):
    changes = []
    for case in small:
        before, after = small[case], large[case]
        for a, b, am, bm in zip(before["per_target"], after["per_target"], before["margins"], after["margins"], strict=True):
            assert a["cad_id"] == b["cad_id"] == am["cad_id"] == bm["cad_id"]
            for mode in before["accuracy"]:
                changes.append({"case": case, "mode": mode, "cad_id": a["cad_id"],
                    "expected_object_id": a["expected_object_id"],
                    "small_prediction": a["predictions"][mode], "large_prediction": b["predictions"][mode],
                    "small_correct_score": am[mode]["correct_score"], "large_correct_score": bm[mode]["correct_score"],
                    "small_margin": am[mode]["correct_minus_best_incorrect"],
                    "large_margin": bm[mode]["correct_minus_best_incorrect"],
                    "margin_delta": bm[mode]["correct_minus_best_incorrect"] - am[mode]["correct_minus_best_incorrect"],
                    "large_strongest_incorrect_object_id": bm[mode]["strongest_incorrect_object_id"],
                    "large_strongest_incorrect_score": bm[mode]["strongest_incorrect_score"]})
    return {"summary": {name: {"small_accuracy": small[name]["accuracy"], "large_accuracy": large[name]["accuracy"],
                              "small_mean_margins": small[name]["mean_margins"], "large_mean_margins": large[name]["mean_margins"]}
                        for name in small}, "per_target": changes}


def main():
    started = perf_counter()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    before = snapshot_hashes()
    production = {name: association.content_hash(ROOT / name) for name in
                  ("cad_object_association.py", "CADPointCloudRegistration.py", "main.py")}
    targets = read_json(DATASET / "manifest.json")["targets"]
    audit = read_json(DATASET / "identity_audit.json")
    object_ids = sorted(r["object_id"] for r in audit)
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    paths = input_catalog(targets, object_ids)
    index, images, pixels_seen, manifest = {}, [], {}, {}
    for key, path in paths.items():
        image = read_rgb(path)
        assert image.shape == (224, 224, 3)
        pixels = image.tobytes()
        if pixels not in pixels_seen:
            pixels_seen[pixels] = len(images)
            images.append(image)
        index[key] = pixels_seen[pixels]
        manifest[key] = {"path": path.relative_to(ROOT).as_posix(), "sha256": association.content_hash(path),
                         "feature_index": index[key]}
    patch_indices = {i for key, i in index.items() if key.startswith(("original14/", "observed/localization/"))}
    save_json(output / "inputs.json", manifest)
    saved_pairs = read_json(ABLATION_2 / "pair_scores.json")
    # Retain only geometry fields, avoiding stale Small visual fields in Large pair rows.
    saved_pairs = [{k: p[k] for k in ("cad_id", "object_id", "geometry_raw", "geometry_score")} for p in saved_pairs]
    saved_views = read_json(ABLATION_3 / "all_view_scores.json")
    assert len(saved_pairs) == 36 and len(saved_views) == 504
    weights = {}
    for name in ["observed", *targets]:
        with np.load(ABLATION_2 / "features" / f"{name}.npz") as saved:
            weights[name] = saved["patch_occupancy"]
    geometry_files = [ABLATION_2 / "pair_scores.json", ABLATION_3 / "all_view_scores.json", ORACLE / "all_pairs.json"]
    mask_files = [ABLATION_2 / "features" / f"{name}.npz" for name in ["observed", *targets]]
    prior_hashes = {p.relative_to(ROOT).as_posix(): association.content_hash(p) for p in geometry_files + mask_files}
    results, runtime, validation = {}, {}, {}
    for label, (name, dimension) in MODELS.items():
        directory = output / label
        directory.mkdir()
        cls, patches, timing = encode(name, dimension, images, patch_indices, directory)
        stage = perf_counter()
        cases, per_view = score_protocols(cls, patches, index, targets, object_ids, saved_pairs, saved_views, weights)
        timing["score_all_protocols_s"] = perf_counter() - stage
        if label == "small":
            validation["small_max_absolute_errors_against_saved_cpu_results"] = validate_small(
                cases, per_view, cls, patches, index, targets, object_ids)
        stage = perf_counter()
        results[label] = evaluate_cases(cases, targets, audit)
        timing["evaluation_s"] = perf_counter() - stage
        for cid in targets:
            oid = next(r["object_id"] for r in audit if r["simulator_instance"] == cid)
            assert index[f"appearance/matched__{cid}__{oid}"] == index[f"appearance/observed__oracle__{oid}"]
            row = next(p for p in cases["appearance/matched_oracle"] if p["cad_id"] == cid and p["object_id"] == oid)
            assert abs(row["visual_raw"] - 1) < 1e-12
        save_json(directory / "pair_scores.json", cases)
        save_json(directory / "per_view_scores.json", per_view)
        save_json(directory / "evaluations.json", results[label])
        runtime[label] = timing
        del cls, patches, cases, per_view
        gc.collect()
        print(f"Completed {label}: {len(results[label])} fixed protocol/subset comparisons", flush=True)
    changes = comparison(results["small"], results["large"])
    save_json(output / "comparison.json", changes)
    validation.update(dataset_unchanged=snapshot_hashes() == before,
        production_unchanged=all(association.content_hash(ROOT / p) == h for p, h in production.items()),
        saved_geometry_and_occupancy_unchanged=all(association.content_hash(ROOT / p) == h for p, h in prior_hashes.items()),
        exact_input_files_unchanged=all(association.content_hash(ROOT / v["path"]) == v["sha256"] for v in manifest.values()),
        correct_matched_oracle_self_cosines_equal_1=True, source_image_count=len(paths),
        distinct_pixel_input_count=len(images), patch_feature_input_count=len(patch_indices))
    assert all(validation[k] for k in ("dataset_unchanged", "production_unchanged",
                                       "saved_geometry_and_occupancy_unchanged", "exact_input_files_unchanged"))
    save_json(output / "validation.json", validation)
    save_json(output / "protocol.json", {"encoder_source": association.DINO_REPO,
        "models": {label: {"name": name, "dimension": dim, "checkpoint_sha256": association.content_hash(
            ROOT / association.DINO_HUB_DIR / "checkpoints" / f"{name}_pretrain.pth")} for label, (name, dim) in MODELS.items()},
        "encoder": "Frozen final CLS and patch tokens, no registers; FP32, no AMP/TF32; unit normalization in float64",
        "input": "Exact saved RGB224 inputs; ImageNet normalization; unchanged masks, no new renders or crops",
        "matching": "Existing max/top-five CLS; occupancy-weighted nearest foreground patches; all candidates/views",
        "fusion": "Unchanged 0.5*((visual_raw+1)/2)+0.5*recorded_geometry_score",
        "view_selection": "Same score rules; Large can choose different views. 3b saved_small_view freezes the original Small C_obs winner; encoder_C_obs_view freezes each encoder's own C_obs winner",
        "sam3": "Pulley has no saved SAM3 mask. SAM3 and common5 comparisons use the same five targets AND candidates",
        "oracle": "True-pose/matched-appearance cases are diagnostics; identical-input cosine=1 is expected, not a real-world accuracy claim",
        "small_parity_atol": PARITY_ATOL, "batch_size": BATCH_SIZE,
        "torch_version": torch.__version__, "numpy_version": np.__version__, "cpu_threads": torch.get_num_threads(),
        "geometry_rendering_segmentation_localization_rerun": False, "production_modified": False,
        "geometry_and_mask_sources": prior_hashes, "production_sha256": production,
        "script_sha256": association.content_hash(__file__)})
    runtime["total_wall_s"] = perf_counter() - started
    runtime["scope"] = "Both encoders and all cached-image scoring; excludes checkpoint download, rendering and geometry. CUDA memory is PyTorch allocated/reserved, not whole-device usage."
    save_json(output / "runtime.json", runtime)
    for name, result in changes["summary"].items():
        a, b = result["small_accuracy"]["visual"], result["large_accuracy"]["visual"]
        print(f"{name}: {a['correct']}/{a['total']} -> {b['correct']}/{b['total']}", flush=True)
    print(f"SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
