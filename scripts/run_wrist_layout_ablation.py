"""Twenty fixed layout tests: localization, global CLS and selected-view patches.

Run with MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 and
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1: python -m scripts.run_wrist_layout_ablation
No SAM3, registration, geometric score, combined score or fusion is evaluated.
"""

import copy
import os
from collections import Counter
from time import perf_counter

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

import cad_object_association as association
from scripts.run_wrist_input_ablation import (
    ROOT, DATASET, read_json, read_rgb, save_image, save_json, snapshot_hashes, masked_crop,
)
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
from scripts.run_wrist_large_encoder import encode, ORACLE, BLENDER
from scripts.run_wrist_sam6d_patch_ablation import LARGE, prepare_masks
from scripts.sam6d_patch_matching import semantic_score, masked_patch_descriptors, appearance_score
from scripts.wrist_oracle_pose import SOURCE
from scripts.run_wrist_association import configure_scene, WORKSPACE_MIN, WORKSPACE_MAX
from scripts.run_wrist_cluster import identity_audit
from simulation.wrist_cluster import make_environment, CAMERA
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.nist_peg_task import localize_parts


OUTPUT = ROOT / "outputs/layout_wrist_association_2026-09-11/20260917T195444Z"
PREVIOUS_PATCH = ROOT / "outputs/sam6d_patch_wrist_association_2026-09-11/20260917T181518155829Z"
RESOLUTION = ROOT / "outputs/resolution_wrist_association_2026-09-11/20260917T190648808765Z"
IMAGE_SIZE = 3840
LAYOUT_SEED = 20260917
METHODS = {"global": "semantic_score", "patch": "appearance_score"}


def layout_plan(names):
    """Predeclare ten unique derangements and ten balanced yaw changes."""
    rng = np.random.default_rng(LAYOUT_SEED)
    count = len(names)
    identity = list(range(count))
    layouts = [{"scene_id": "baseline", "family": "control", "slots": identity, "yaw_delta_deg": [0.] * count}]
    permutations = set()
    while len(permutations) < 10:
        perm = tuple(rng.permutation(count).tolist())
        if perm in permutations or any(i == j for i, j in enumerate(perm)):
            continue
        permutations.add(perm)
        layouts.append({"scene_id": f"position_{len(permutations):02}", "family": "position",
                        "slots": list(perm), "yaw_delta_deg": [0.] * count})
    angles = np.array([-150., -120., -90., -60., -30., 30., 60., 90., 120., 150.])
    yaw = np.stack([rng.permutation(angles) for _ in names], axis=1)
    for i in range(10):
        layouts.append({"scene_id": f"yaw_{i + 1:02}", "family": "yaw", "slots": identity,
                        "yaw_delta_deg": yaw[i].tolist()})
    return {"seed": LAYOUT_SEED, "object_order": names, "layouts": layouts}


def rearranged_pose(base, slot, yaw_deg):
    """World XY comes from the slot; world Z and tilt remain the object's own."""
    pose = base.copy()
    pose[:2] = slot[:2]
    rotation = Rotation.from_quat(base[3:][[1, 2, 3, 0]])
    pose[3:] = (Rotation.from_euler("z", yaw_deg, degrees=True) * rotation).as_quat()[[3, 0, 1, 2]]
    return pose


def capture_layouts(plan):
    """Camera remains fixed; fresh native RGB-D and localization for every layout."""
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
    source = read_json(SOURCE / "manifest.json")
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    catalog = copy.deepcopy(source["catalog"])
    for cid, record in catalog.items():
        path = ROOT / library[cid]["file_path"]
        assert association.content_hash(path) == record["cad_sha256"]
        record["cad_path"] = str(path)
    names = plan["object_order"]
    expected_camera = np.load(DATASET / "scene/world_T_camera.npy")
    expected_K = np.diag([5., 5., 1.]) @ np.load(DATASET / "scene/intrinsics.npy")
    with make_environment(catalog, source["placements"], additional_scene=configure_scene) as env:
        env.reset(seed=source["seed"])
        sim = env.sim
        sim.set_state_from_flattened(np.load(SOURCE / "initial_state.npy"))
        sim.data.qpos[:] = np.load(ORACLE / "ground_truth/captured_qpos.npy")
        sim.forward()
        np.testing.assert_array_equal(sim.render(width=768, height=768, camera_name=CAMERA)[::-1], read_rgb(DATASET / "scene/rgb.png"))
        base_qpos = sim.data.qpos.copy()
        addresses = {}
        for name in names:
            bid = sim.model.body_name2id(f"nist_part_{name}")
            joint = int(sim.model.body_jntadr[bid])
            assert int(sim.model.jnt_type[joint]) == 0  # MuJoCo free joint.
            addresses[name] = int(sim.model.jnt_qposadr[joint])
        base = [base_qpos[addresses[name]:addresses[name] + 7].copy() for name in names]
        sensor = LiberoRGBDSensor(env, CAMERA)
        for layout in plan["layouts"]:
            folder = OUTPUT / layout["scene_id"]
            if (folder / "capture_complete.json").exists():
                assert read_json(folder / "capture_complete.json")["layout"] == layout
                print(f"Reusing completed capture {layout['scene_id']}", flush=True)
                continue
            start = perf_counter()
            folder.mkdir(exist_ok=True)
            sim.data.qpos[:] = base_qpos
            expected_qpos = base_qpos.copy()
            for i, name in enumerate(names):
                addr = addresses[name]
                expected_qpos[addr:addr + 7] = rearranged_pose(base[i], base[layout["slots"][i]], layout["yaw_delta_deg"][i])
            sim.data.qpos[:] = expected_qpos
            sim.forward()  # No dynamics: preserve the camera, support heights and specified poses.
            np.testing.assert_array_equal(sim.data.qpos, expected_qpos)
            np.testing.assert_array_equal(get_camera_extrinsic_matrix(sim, CAMERA), expected_camera)
            np.testing.assert_allclose(get_camera_intrinsic_matrix(sim, CAMERA, IMAGE_SIZE, IMAGE_SIZE), expected_K, rtol=0, atol=1e-10)
            raw, depth_buffer = sim.render(width=IMAGE_SIZE, height=IMAGE_SIZE, camera_name=CAMERA, depth=True)
            observation = sensor.capture({f"{CAMERA}_image": raw, f"{CAMERA}_depth": depth_buffer})
            np.testing.assert_array_equal(sim.data.qpos, expected_qpos)
            save_image(folder / "rgb.png", observation.rgb)
            np.save(folder / "depth.npy", observation.depth_m)
            np.save(folder / "intrinsics.npy", observation.intrinsics)
            np.save(folder / "world_T_camera.npy", observation.world_T_camera)
            np.save(folder / "qpos.npy", expected_qpos)
            render_s = perf_counter() - start
            stage = perf_counter()
            o3d.utility.random.seed(LAYOUT_SEED)
            localization, _ = localize_parts(observation, folder / "rgb.png", folder,
                workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX, cluster_in_table_plane=True)
            localization_s = perf_counter() - stage
            audit = identity_audit(env, localization, observation) if localization["objects"] else []
            save_json(folder / "identity_audit.json", audit)
            gt = {name: {"position_world_m": sim.data.body_xpos[sim.model.body_name2id(f"nist_part_{name}")].tolist(),
                         "rotation_world_from_cad": sim.data.body_xmat[sim.model.body_name2id(f"nist_part_{name}")].reshape(3, 3).tolist()}
                  for name in names}
            save_json(folder / "ground_truth.json", gt)
            for sub in ("inputs", "masks", "crops"):
                (folder / sub).mkdir(exist_ok=True)
            records, masks224 = [], []
            for item in localization["objects"]:
                oid = item["object_id"]
                cloud = o3d.io.read_point_cloud(item["pointcloud_path"])
                crop, mask, details = masked_crop(observation.rgb, cloud, localization["camera_intrinsics"])
                canvas = association.letterbox_rgb(crop)
                x1, y1, x2, y2 = details["crop_bbox_2d_xyxy"]
                mask224 = letterbox_mask(mask[y1:y2 + 1, x1:x2 + 1])
                masks224.append(mask224)
                save_image(folder / "inputs" / f"{oid}.png", canvas)
                save_image(folder / "masks" / f"{oid}.png", mask.astype(np.uint8) * 255)
                save_image(folder / "crops" / f"{oid}.png", crop)
                records.append({"object_id": oid, **details, "native_crop_hwc": list(crop.shape),
                    "input_sha256": association.content_hash(folder / "inputs" / f"{oid}.png")})
            np.savez_compressed(folder / "observed_masks.npz", object_ids=[r["object_id"] for r in records],
                masks=np.stack(masks224) if masks224 else np.empty((0, 224, 224), dtype=np.uint8))
            save_json(folder / "inputs.json", records)
            matched = {r["simulator_instance"] for r in audit if r["status"] == "matched"}
            validation = {"layout": layout, "candidate_count": len(records), "matched_target_count": len(matched),
                "missing_or_ambiguous_targets": [n for n in names if n not in matched],
                "unresolved_candidates": [r["object_id"] for r in audit if r["status"] != "matched"],
                "camera_pose_and_intrinsics_fixed": True, "rgbd_resolution": [IMAGE_SIZE, IMAGE_SIZE],
                "fresh_localization": True, "render_and_save_s": render_s, "localization_s": localization_s,
                "capture_and_input_preparation_s": perf_counter() - start}
            save_json(folder / "capture_complete.json", validation)
            print(f"{layout['scene_id']}: {len(records)} candidates, {len(matched)}/6 audited; {validation['capture_and_input_preparation_s']:.1f}s", flush=True)


def evaluate(rows, audit, names):
    """Independent CAD queries; missing/ambiguous ground truth remains a failure."""
    truth = {r["simulator_instance"]: r["object_id"] for r in audit if r["status"] == "matched"}
    identities = {r["object_id"]: r["simulator_instance"] for r in audit if r["status"] == "matched"}
    results = []
    for name in names:
        expected = truth.get(name)
        for method, key in METHODS.items():
            ranking = sorted([r for r in rows if r["cad_id"] == name], key=lambda r: (-r[key], r["object_id"]))
            right = next((r for r in ranking if r["object_id"] == expected), None)
            wrong = next((r for r in ranking if r["object_id"] != expected), None)
            prediction = ranking[0]["object_id"] if ranking else None
            results.append({"cad_id": name, "method": method, "expected_object_id": expected,
                "localized": expected is not None, "prediction": prediction,
                "predicted_identity": identities.get(prediction),
                "correct": expected is not None and prediction == expected,
                "correct_score": right[key] if right else None,
                "strongest_incorrect_score": wrong[key] if wrong else None,
                "strongest_incorrect_identity": identities.get(wrong["object_id"]) if wrong else None,
                "margin": right[key] - wrong[key] if right and wrong else None,
                "ranking": [{"object_id": r["object_id"], "identity": identities.get(r["object_id"]), "score": r[key]} for r in ranking]})
    return results


def score_layouts(plan):
    old = read_json(LARGE / "inputs.json")
    with np.load(LARGE / "large/features.npz") as f:
        cad_cls = f["cls_features"]
        patches = dict(zip(f["patch_indices"].tolist(), f["patch_tokens"]))
    for folder in (PREVIOUS_PATCH / "additional_features", RESOLUTION / "additional_template_features"):
        indices = read_json(folder / "index.json")["original_feature_indices"]
        with np.load(folder / "features.npz") as f:
            patches.update(dict(zip(indices, f["patch_tokens"])))
    inputs, weights = {}, {}
    for layout in plan["layouts"]:
        folder = OUTPUT / layout["scene_id"]
        with np.load(folder / "observed_masks.npz") as f:
            for oid, mask in zip(f["object_ids"].tolist(), f["masks"]):
                key = f"{layout['scene_id']}/{oid}"
                inputs[key] = folder / "inputs" / f"{oid}.png"
                weights[key] = patch_occupancy(mask)
    keys = list(inputs)
    feature_folder = OUTPUT / "observed_features"
    feature_folder.mkdir(exist_ok=True)
    if (feature_folder / "index.json").exists():
        assert read_json(feature_folder / "index.json")["keys"] == keys
        with np.load(feature_folder / "features.npz") as f:
            cls = f["cls_features"]
            observed_patches = dict(zip(f["patch_indices"].tolist(), f["patch_tokens"]))
        runtime = read_json(feature_folder / "runtime.json")
    else:
        cls, observed_patches, runtime = encode("dinov2_vitl14", 1024, [read_rgb(p) for p in inputs.values()], set(range(len(keys))), feature_folder)
        save_json(feature_folder / "runtime.json", runtime)
        save_json(feature_folder / "index.json", {"keys": keys, "input_sha256": [association.content_hash(p) for p in inputs.values()]})
    queries = {key: masked_patch_descriptors(observed_patches[i], weights[key]) for i, key in enumerate(keys)}
    needed, records = set(), {layout["scene_id"]: [] for layout in plan["layouts"]}
    for name in plan["object_order"]:
        templates = [f"blenderproc42/{name}/{v}" for v in range(1, 43)]
        features = np.stack([cad_cls[old[k]["feature_index"]] for k in templates])
        for i, key in enumerate(keys):
            scene, oid = key.split("/")
            result = semantic_score(cls[i], features)
            template = templates[result["best_view_index_1based"] - 1]
            needed.add(template)
            records[scene].append({"cad_id": name, "object_id": oid, "input_key": key, "best_template_id": template, **result})
    missing = sorted({old[k]["feature_index"] for k in needed} - patches.keys())
    template_folder = OUTPUT / "additional_template_features"
    template_folder.mkdir(exist_ok=True)
    if missing:
        if (template_folder / "index.json").exists():
            assert read_json(template_folder / "index.json")["original_feature_indices"] == missing
            with np.load(template_folder / "features.npz") as f:
                extra_cls = f["cls_features"]
                extra_patches = dict(zip(f["patch_indices"].tolist(), f["patch_tokens"]))
        else:
            index_to_key = {old[k]["feature_index"]: k for k in sorted(needed)}
            extra_cls, extra_patches, extra_runtime = encode("dinov2_vitl14", 1024,
                [read_rgb(ROOT / old[index_to_key[i]]["path"]) for i in missing], set(range(len(missing))), template_folder)
            save_json(template_folder / "runtime.json", extra_runtime)
            save_json(template_folder / "index.json", {"original_feature_indices": missing})
        np.testing.assert_allclose(extra_cls, cad_cls[missing], atol=1e-12, rtol=0)
        patches.update({index: extra_patches[i] for i, index in enumerate(missing)})
    cad_weights, mask_hashes = prepare_masks(needed, old, OUTPUT)
    references = {k: masked_patch_descriptors(patches[old[k]["feature_index"]], cad_weights[k]) for k in needed}
    stage = perf_counter()
    results = []
    for layout in plan["layouts"]:
        scene = layout["scene_id"]
        rows = records[scene]
        for row in rows:
            patch = appearance_score(queries[row["input_key"]], references[row["best_template_id"]])
            row.update({k: patch[k] for k in ("appearance_score", "observed_patch_count", "cad_patch_count")})
        save_json(OUTPUT / scene / "pair_scores.json", rows)
        scored = evaluate(rows, read_json(OUTPUT / scene / "identity_audit.json"), plan["object_order"])
        save_json(OUTPUT / scene / "evaluation.json", scored)
        results.extend({"scene_id": scene, "family": layout["family"], **r} for r in scored)
        print(f"{scene}: " + ", ".join(f"{m} {sum(r['correct'] for r in scored if r['method'] == m)}/6" for m in METHODS), flush=True)
    save_json(OUTPUT / "all_evaluations.json", results)
    summary = {}
    for family in ("control", "position", "yaw", "all_rearranged"):
        selected = [r for r in results if (r["family"] != "control" if family == "all_rearranged" else r["family"] == family)]
        summary[family] = {}
        for method in METHODS:
            rows = [r for r in selected if r["method"] == method]
            margins = [r["margin"] for r in rows if r["margin"] is not None]
            summary[family][method] = {"correct": sum(r["correct"] for r in rows), "total": len(rows),
                "localized": sum(r["localized"] for r in rows), "mean_margin": float(np.mean(margins)) if margins else None,
                "per_target": {name: {"correct": sum(r["correct"] for r in rows if r["cad_id"] == name),
                    "total": sum(r["cad_id"] == name for r in rows),
                    "failures_to": dict(Counter((r["predicted_identity"] or "unresolved") for r in rows if r["cad_id"] == name and not r["correct"]))}
                    for name in plan["object_order"]}}
    save_json(OUTPUT / "summary.json", summary)
    save_json(OUTPUT / "scoring_runtime.json", {"observation_encoder": runtime, "patch_scoring_and_evaluation_s": perf_counter() - stage})
    return {"pair_count": sum(len(r) for r in records.values()), "additional_template_count": len(missing), "template_mask_hashes": mask_hashes}


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    start = perf_counter()
    fixed = snapshot_hashes()
    names = list(read_json(DATASET / "manifest.json")["targets"])
    plan = layout_plan(names)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "layout_plan.json").exists():
        assert read_json(OUTPUT / "layout_plan.json") == plan
    save_json(OUTPUT / "layout_plan.json", plan)
    checkpoint = ROOT / association.DINO_HUB_DIR / "checkpoints/dinov2_vitl14_pretrain.pth"
    assert association.content_hash(checkpoint) == read_json(LARGE / "protocol.json")["models"]["large"]["checkpoint_sha256"]
    sources = ("cad_object_association.py", "CADPointCloudRegistration.py", "main.py", "point_cloud_localization.py",
        "simulation/wrist_cluster.py", "simulation/nist_peg_task.py", "simulation/nist_task_board_1.py", "simulation/perception_adapter.py")
    protected = {p: association.content_hash(ROOT / p) for p in sources}
    save_json(OUTPUT / "protocol.json", {"layout_plan": "layout_plan.json", "methods": METHODS,
        "encoder": "DINOv2-L/14; same frozen checkpoint, 224x224 white bicubic letterbox",
        "checkpoint_sha256": association.content_hash(checkpoint), "camera": CAMERA, "image_size": IMAGE_SIZE,
        "templates": "Same 42 saved BlenderProc RGB templates per CAD, cached CLS and normalized patch tokens",
        "patch_view": "Single highest-CLS template selected separately for every CAD-observation pair",
        "localization": "Fresh native 3840 RGB-D; unchanged localize_parts settings; cleaned-cloud projection; no dilation or hole filling",
        "control": "Original arrangement rerun with fresh 3840 localization; previous resolution experiment used scaled 768 masks",
        "pose_generation": "Restore captured robot/object state, change only object free-joint XY or world-Z yaw; preserve object Z and tilt; no dynamics steps",
        "ground_truth": "Simulator state is used only to construct layouts and audit identities; scoring gets RGB/masks/descriptors only",
        "selection": "Exhaustive six CAD queries against every localized candidate; misses/ambiguities count as failures; no candidate pruning",
        "geometry_and_fusion": "Excluded", "sam3": "Excluded", "protected_source_sha256": protected,
        "implementation_sha256": association.content_hash(ROOT / "scripts/run_wrist_layout_ablation.py")})
    capture_layouts(plan)
    details = score_layouts(plan)
    old = read_json(LARGE / "inputs.json")
    for key, item in old.items():
        if key.startswith("blenderproc42/"):
            assert association.content_hash(ROOT / item["path"]) == item["sha256"]
    assert snapshot_hashes() == fixed
    assert all(association.content_hash(ROOT / p) == h for p, h in protected.items())
    save_json(OUTPUT / "validation.json", {"fixed_dataset_unchanged": True, "production_unchanged": True,
        "all_252_template_images_unchanged": True, "layout_count_excluding_control": 20,
        "no_geometric_or_fused_score": True, "no_sam3": True, **details})
    save_json(OUTPUT / "runtime.json", {"wall_time_this_invocation_s": perf_counter() - start,
        "capture_and_preparation_s": sum(read_json(OUTPUT / r["scene_id"] / "capture_complete.json")["capture_and_input_preparation_s"] for r in plan["layouts"])})
    print(f"SAVED {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
