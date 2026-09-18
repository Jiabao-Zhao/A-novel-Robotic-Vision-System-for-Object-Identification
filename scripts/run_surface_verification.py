"""Four-stage surface verification experiment. Frozen constants; no production edits.

WSL: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python -m scripts.run_surface_verification
"""

import os
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, save_json, read_rgb, save_image, snapshot_hashes
from scripts.run_wrist_large_encoder import encode, BLENDER, ORACLE, PARITY_ATOL
from scripts.run_wrist_sam6d_patch_ablation import LARGE, prepare_masks
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
from scripts.sam6d_patch_matching import semantic_score, masked_patch_descriptors, appearance_score
from scripts.surface_verification import CadSurface, verify_pose, downsample, depth_points, translation_fit, SIGMA_M, OCCLUSION_M


OUTPUT = ROOT / "outputs/surface_verification_20260917"
LOCAL_MASKS = ROOT / "outputs/ablation_1_wrist_association_2026-09-11/20260912T153218768705Z/masked/masks"
SAVED_SAM = ROOT / "outputs/roboflow_sam_workspace/20260914T150601681311Z"
METHODS = ("global", "patch", "visual", "geometry", "frozen_view_fused", "view_consistent_fused")
SEED = 20260918


def evaluate(rows, truth, methods=METHODS):
    """CAD-to-candidate association; include missing targets as failures."""
    records = []
    for cid, expected in truth.items():
        for method in methods:
            ranking = sorted((r for r in rows if r["cad_id"] == cid and r.get(method) is not None),
                             key=lambda r: (-r[method], r["object_id"]))
            correct = next((r for r in ranking if r["object_id"] in expected), None)
            incorrect = next((r for r in ranking if r["object_id"] not in expected), None)
            prediction = ranking[0]["object_id"] if ranking else None
            records.append({"cad_id": cid, "method": method, "expected": expected, "prediction": prediction,
                "correct": prediction in expected, "correct_score": correct[method] if correct else None,
                "incorrect_score": incorrect[method] if incorrect else None,
                "margin": correct[method] - incorrect[method] if correct and incorrect else None,
                "ranking": [{"object_id": r["object_id"], "score": r[method]} for r in ranking]})
    summary = {}
    for method in methods:
        selected = [r for r in records if r["method"] == method]
        margins = [r["margin"] for r in selected if r["margin"] is not None]
        summary[method] = {"correct": sum(r["correct"] for r in selected), "total": len(selected),
                           "mean_margin": float(np.mean(margins)) if margins else None,
                           "margin_count": len(margins),
                           "unassessable_correct_targets": sum(r["correct_score"] is None for r in selected)}
    return {"summary": summary, "targets": records}


def feature_cache(paths, folder):
    folder.mkdir(parents=True, exist_ok=True)
    keys = list(paths)
    hashes = [association.content_hash(p) for p in paths.values()]
    index = {"keys": keys, "sha256": hashes, "device": "cuda", "model": "dinov2_vitl14"}
    if (folder / "index.json").exists():
        assert read_json(folder / "index.json") == index
    else:
        _, _, runtime = encode("dinov2_vitl14", 1024, [read_rgb(p) for p in paths.values()], set(range(len(keys))), folder, device="cuda")
        save_json(folder / "runtime.json", runtime)
        save_json(folder / "index.json", index)
    with np.load(folder / "features.npz") as f:
        return {key: (c, p) for key, c, p in zip(keys, f["cls_features"], f["patch_tokens"], strict=True)}


def template_features(names):
    paths = {f"blenderproc42/{name}/{view}": BLENDER / "templates" / name / f"view_{view:02}.png"
             for name in names for view in range(1, 43)}
    features = feature_cache(paths, OUTPUT / "template_features")
    old = read_json(LARGE / "inputs.json")
    with np.load(LARGE / "large/features.npz") as f:
        errors = [float(np.max(np.abs(features[k][0] - f["cls_features"][old[k]["feature_index"]]))) for k in paths]
    save_json(OUTPUT / "gpu_parity.json", {"template_count": len(paths), "max_cls_abs_error_against_saved_cpu": max(errors),
              "tolerance": PARITY_ATOL, "strict_component_parity_passed": max(errors) < PARITY_ATOL,
              "outlying_templates": [k for k, error in zip(paths, errors) if error >= PARITY_ATOL],
              "gpu": torch.cuda.get_device_name(), "precision": "FP32; no autocast",
              "policy": "Record numerical mismatch, do not relax tolerance. All new branches use the same new CUDA features; old caches unchanged.",
              "fresh_cpu_gpu_diagnostic": "fresh_gpu_parity.json"})
    weights, _ = prepare_masks(paths, old, OUTPUT / "template_features")
    return {name: {"cls": np.stack([features[f"blenderproc42/{name}/{v}"][0] for v in range(1, 43)]),
            "patch": [masked_patch_descriptors(features[f"blenderproc42/{name}/{v}"][1], weights[f"blenderproc42/{name}/{v}"]).cuda()
                      for v in range(1, 43)]} for name in names}


def prepare_observations(rgb, masks, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if not masks:
        return {}
    paths, weights = {}, {}
    for oid, mask in masks.items():
        image, _, bounds = prepare_masked_input(rgb, mask)
        x1, y1, x2, y2 = bounds
        path = folder / f"{oid}.png"
        save_image(path, image)
        paths[oid] = path
        weights[oid] = patch_occupancy(letterbox_mask(mask[y1:y2 + 1, x1:x2 + 1]))
    features = feature_cache(paths, folder / "features")
    return {oid: {"cls": features[oid][0], "patch": masked_patch_descriptors(features[oid][1], weights[oid]).cuda()}
            for oid in paths}


def geometric_search(cads, visible, rotations, depth, masks, K, folder, table=None, offset=.5):
    """Exhaustive 42 fixed rotations, identical translation budget for every CAD."""
    cache = folder / "registrations.json"
    if cache.exists():
        return read_json(cache)
    start = perf_counter()
    rows = []
    observed = {oid: downsample(depth_points(depth, mask, K, offset)) for oid, mask in masks.items()}
    for cid, cad in cads.items():
        for oid, points in observed.items():
            if not len(points):
                continue
            for view, rotation in enumerate(rotations):
                t, fitness = translation_fit(visible[cid][view], points)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = rotation, t
                score = verify_pose(cad, T, depth, masks[oid], K, table, offset)
                rows.append({"cad_id": cid, "object_id": oid, "view_index_1based": view + 1,
                    "camera_T_cad": T.tolist(), "registration_fitness": fitness, **score})
        print(folder.name, cid, "geometry complete", flush=True)
    save_json(cache, rows)
    save_json(folder / "geometry_runtime.json", {"wall_s": perf_counter() - start, "registrations": len(rows),
              "rotations_per_pair": 42, "translation_iterations_per_rotation": 30, "device": "CPU"})
    return rows


def collapse_pair(pair, rows):
    """Only admissible poses may win geometry/fusion; visual remains independent."""
    for r in rows:
        r["view_consistent_fused"] = (.25 * r["global"] + .25 * r["patch"] + .5 * r["surface_score"]
                                      if r["status"] == "assessable" else None)
    usable = [r for r in rows if r["status"] == "assessable"]
    geo = max(usable, key=lambda r: r["surface_score"]) if usable else None
    fused = max(usable, key=lambda r: r["view_consistent_fused"]) if usable else None
    cls = next((r for r in usable if r["view_index_1based"] == pair["best_view_index_1based"]), None)
    return {**pair, "geometry": geo["surface_score"] if geo else None,
        "frozen_view_fused": cls["view_consistent_fused"] if cls else None,
        "view_consistent_fused": fused["view_consistent_fused"] if fused else None,
        "geometry_view": geo["view_index_1based"] if geo else None,
        "fused_view": fused["view_index_1based"] if fused else None, "admissible_view_count": len(usable)}


def rescore_saved(folder, truth):
    """Correct aggregation without rendering, registration or feature extraction."""
    start = perf_counter()
    views = read_json(folder / "all_view_scores.json")
    groups = {}
    for row in views:
        groups.setdefault((row["cad_id"], row["object_id"]), []).append(row)
    pairs = [collapse_pair(p, groups.get((p["cad_id"], p["object_id"]), []))
             for p in read_json(folder / "pair_scores.json")]
    result = {"scoring_version": 2, **evaluate(pairs, truth)}
    save_json(folder / "all_view_scores.json", views)
    save_json(folder / "pair_scores.json", pairs)
    save_json(folder / "evaluation.json", result)
    runtime = read_json(folder / "runtime.json")
    save_json(folder / "runtime.json", {**runtime, "saved_score_reaggregation_s": perf_counter() - start})
    return result


def score_scene(cads, visible, rotations, templates, depth, masks, K, rgb, truth, folder, table=None,
                frozen_registrations=None, offset=.5):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "evaluation.json").exists():
        saved = read_json(folder / "evaluation.json")
        if saved.get("scoring_version") == 2:
            return saved
        return rescore_saved(folder, truth)
    start = perf_counter()
    observed = prepare_observations(rgb, masks, folder / "inputs")
    if frozen_registrations is None:
        registrations = geometric_search(cads, visible, rotations, depth, masks, K, folder, table, offset)
    else:
        registrations = []
        for row in frozen_registrations:
            if row["object_id"] in masks:
                score = verify_pose(cads[row["cad_id"]], np.array(row["camera_T_cad"]), depth,
                    masks[row["object_id"]], K, table, offset)
                registrations.append({**row, **score})
        save_json(folder / "registrations.json", registrations)
    lookup = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r for r in registrations}
    pairs, views = [], []
    for cid, template in templates.items():
        for oid, query in observed.items():
            semantic = semantic_score(query["cls"], template["cls"])
            patches = [appearance_score(query["patch"], ref)["appearance_score"] for ref in template["patch"]]
            scores = []
            for view in range(1, 43):
                raw = lookup.get((cid, oid, view))
                if raw is None:
                    continue
                r = {**raw, "cls_cosine": semantic["cls_view_cosines"][view - 1], "global": semantic["semantic_score"],
                     "patch": patches[view - 1]}
                scores.append(r)
                views.append(r)
            selected = semantic["best_view_index_1based"]
            global_score, patch_score = semantic["semantic_score"], patches[selected - 1]
            pairs.append(collapse_pair({"cad_id": cid, "object_id": oid, **semantic,
                "global": global_score, "patch": patch_score, "visual": .5 * global_score + .5 * patch_score}, scores))
    save_json(folder / "all_view_scores.json", views)
    save_json(folder / "pair_scores.json", pairs)
    result = {"scoring_version": 2, **evaluate(pairs, truth)}
    save_json(folder / "evaluation.json", result)
    save_json(folder / "runtime.json", {"wall_this_invocation_s": perf_counter() - start,
              "frozen_registration_rescore": frozen_registrations is not None, "pair_count": len(pairs)})
    print(folder.name, result["summary"], flush=True)
    return result


def oracle_test(cads, masks, depth, K, truth):
    folder = OUTPUT / "stage1_verified_pose"
    folder.mkdir(exist_ok=True)
    poses = read_json(ORACLE / "ground_truth/poses.json")
    owner = {oids[0]: cid for cid, oids in truth.items()}
    rows = []
    start = perf_counter()
    for cid, cad in cads.items():
        for oid, mask in masks.items():
            T = np.array(poses[owner[oid]]["camera_T_cad"])
            # Same legacy wrong-CAD frame transplantation: diagnostic, not inference.
            rows.append({"cad_id": cid, "object_id": oid, "camera_T_cad": T.tolist(),
                         **verify_pose(cad, T, depth, mask, K)})
    for oid in masks:
        a = next(r for r in rows if r["cad_id"] == "red_block" and r["object_id"] == oid)
        b = next(r for r in rows if r["cad_id"] == "blue_block" and r["object_id"] == oid)
        assert a["surface_score"] == b["surface_score"], "Identical meshes must have identical geometry."
    save_json(folder / "pair_scores.json", rows)
    result = evaluate(rows, truth, ("box_iou", "mask_iou", "surface_score"))
    save_json(folder / "evaluation.json", result)
    save_json(folder / "runtime.json", {"wall_s": perf_counter() - start})
    print("oracle", result["summary"], flush=True)
    return result


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = perf_counter()
    protected = snapshot_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    library = read_json(DATASET / "cad_library.json")
    names = [r["cad_id"] for r in library]
    rotations = np.load(BLENDER / "cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    rng = np.random.default_rng(SEED)
    plan = {"seed": SEED, "object_order": names, "layouts": [{"scene_id": f"heldout_{i + 1:02}", "family": "joint_xy_yaw",
            "slots": rng.permutation(len(names)).tolist(), "yaw_delta_deg": rng.choice([-150., -90., -30., 30., 90., 150.], len(names)).tolist()}
            for i in range(4)]}
    protocol = {"seed": SEED, "layout_plan": plan, "sigma_m": SIGMA_M, "external_occlusion_tolerance_m": OCCLUSION_M,
        "parameter_rationale": "Both scales fixed to existing 5 mm geometric preprocessing, before scores; not sensor-noise estimates",
        "search": "42 existing rotations; visible-cloud centroid initialization; 30 translation-only ICP updates each; 5 mm sampling/7.5 mm correspondences",
        "limits": "Finite orientation search, not full 6D optimization. Reject CAD crossing image edge/camera or penetrating fitted table by >5 mm.",
        "visual": "DINOv2-L/14 FP32 GPU, frozen 42 BlenderProc templates; global top-five; patch at each same fixed rotation",
        "fusion": ".25 global + .25 same-view patch + .50 surface; max over views; original visual baseline remains CLS-selected patch",
        "stage3": "Rescore same transforms with alternate masks; distinguish saved text-prompted SAM3 from fresh identity-free SAM3",
        "real_pilot": "T-LESS: first BOP19 target frame from scenes 1,10,20; all 30 CAD models; provided visible masks, conditional association, not end-to-end detection/BOP AR",
        "ground_truth": "Pose used only in stage1; IDs only for evaluation and scene construction elsewhere",
        "production_modified": False}
    if (OUTPUT / "protocol.json").exists():
        assert read_json(OUTPUT / "protocol.json") == protocol
    save_json(OUTPUT / "protocol.json", protocol)
    cads = {r["cad_id"]: CadSurface(o3d.io.read_triangle_mesh(str(ROOT / r["file_path"]))) for r in library}
    masks = {p.stem: cv2.imread(str(p), 0) > 0 for p in sorted(LOCAL_MASKS.glob("*.png"))}
    depth, K = np.load(DATASET / "scene/depth.npy"), np.load(DATASET / "scene/intrinsics.npy")
    rgb = read_rgb(DATASET / "scene/rgb.png")
    truth = {r["simulator_instance"]: [r["object_id"]] for r in read_json(DATASET / "identity_audit.json")}
    results = {"stage1": oracle_test(cads, masks, depth, K, truth)}
    templates = template_features(names)
    visible = {cid: [cad.visible_points(R) for R in rotations] for cid, cad in cads.items()}
    table = np.array(read_json(DATASET / "scene/localization.json")["plane_model"])
    if table[3] < 0:
        table *= -1  # Camera side of tabletop is positive.
    results["stage2"] = score_scene(cads, visible, rotations, templates, depth, masks, K, rgb, truth,
                                    OUTPUT / "stage2_pose_search", table)
    frozen = read_json(OUTPUT / "stage2_pose_search/registrations.json")
    sam = {p.stem: cv2.imread(str(p), 0) > 0 for p in sorted((SAVED_SAM / "masks").glob("*.png"))}
    results["stage3_saved_sam"] = score_scene(cads, visible, rotations, templates, depth, sam, K, rgb, truth,
                                    OUTPUT / "stage3_saved_sam", table, frozen)
    if (OUTPUT / "sam3/index.json").exists():
        fresh = {p.stem.replace("_mask", ""): cv2.imread(str(p), 0) > 0 for p in (OUTPUT / "sam3").glob("*_mask.png")}
        results["stage3_fresh_sam"] = score_scene(cads, visible, rotations, templates, depth, fresh, K, rgb, truth,
                                    OUTPUT / "stage3_fresh_sam", table, frozen)
    # New predeclared layouts. Reuse capture code without editing its saved protocol.
    import scripts.run_wrist_layout_ablation as layouts
    layouts.OUTPUT = OUTPUT / "new_layouts"
    layouts.OUTPUT.mkdir(exist_ok=True)
    layouts.capture_layouts(plan)
    for layout in plan["layouts"]:
        source = layouts.OUTPUT / layout["scene_id"]
        masks_new = {p.stem: cv2.imread(str(p), 0) > 0 for p in (source / "masks").glob("*.png")}
        audit = read_json(source / "identity_audit.json")
        truth_new = {cid: [r["object_id"] for r in audit if r["simulator_instance"] == cid and r["status"] == "matched"] for cid in names}
        table = np.array(read_json(source / "point_cloud/point_cloud_localization.json")["plane_model"])
        if table[3] < 0:
            table *= -1
        results[layout["scene_id"]] = score_scene(cads, visible, rotations, templates, np.load(source / "depth.npy"),
            masks_new, np.load(source / "intrinsics.npy"), read_rgb(source / "rgb.png"), truth_new,
            OUTPUT / "stage4_layouts" / layout["scene_id"], table)
    assert snapshot_hashes() == protected
    save_json(OUTPUT / "simulation_summary.json", {key: r["summary"] for key, r in results.items()})
    save_json(OUTPUT / "simulation_runtime.json", {"wall_this_invocation_s": perf_counter() - start, "fixed_dataset_unchanged": True})


if __name__ == "__main__":
    main()
