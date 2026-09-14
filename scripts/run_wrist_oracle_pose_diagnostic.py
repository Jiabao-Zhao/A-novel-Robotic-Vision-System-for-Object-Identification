"""All-six-object visual/geometric scoring with verified captured CAD poses.

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.run_wrist_oracle_pose_diagnostic

This is an oracle diagnostic, not a production association or pose estimator.
"""

import os
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d
import torch

import cad_object_association as association
from CADPointCloudRegistration import CADPointCloudRegistration
from scripts.run_wrist_input_ablation import (
    ROOT, DATASET, read_json, read_rgb, save_csv, save_image, save_json, snapshot_hashes,
)
from scripts.run_wrist_patch_ablation import ABLATION_1, verify_artifacts
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.wrist_visible_geometry import geometry_from_arrays
from scripts.wrist_oracle_pose import recover_capture, perspective_render, fixed_pose_metrics, distance_summary
from scripts.wrist_oracle_pose_report import save_report


ABLATION_3 = ROOT / "outputs/ablation_3_wrist_association_2026-09-11/20260912T163935340572Z"
OUTPUT_ROOT = ROOT / "outputs/oracle_pose_wrist_association_2026-09-11"


def unit_features(images):
    features = association.extract_dino_features(images).astype(np.float64)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def evaluate_pairs(pairs, truth, geometry_key):
    targets = []
    for cid, expected in truth.items():
        rows = [dict(p, fused_score=association.fuse_scores(
            association.normalize_visual_score(p["visual_raw"]), p[geometry_key]["registration_fitness"]),
                     geometry_score=p[geometry_key]["registration_fitness"])
                for p in pairs if p["cad_id"] == cid]
        target = {"cad_id": cid, "expected_object_id": expected}
        for mode, key in (("visual", "visual_raw"), ("geometry", "geometry_score"), ("fused", "fused_score")):
            ranked = association.rank_candidates(rows, key)
            correct = next(r for r in rows if r["object_id"] == expected)
            wrong = next(r for r in ranked if r["object_id"] != expected)
            target[mode] = {"prediction": ranked[0]["object_id"], "correct": ranked[0]["object_id"] == expected,
                            "ranking": [r["object_id"] for r in ranked],
                            "margin": correct[key] - wrong[key], "correct_score": correct[key],
                            "strongest_incorrect_score": wrong[key], "strongest_incorrect_object_id": wrong["object_id"]}
        targets.append(target)
    return {"per_target": targets,
            "accuracy": {mode: {"correct": sum(t[mode]["correct"] for t in targets), "total": len(targets)}
                         for mode in ("visual", "geometry", "fused")},
            "incorrect_pairs_with_fitness_1": sum(p["object_id"] != truth[p["cad_id"]]
                and p[geometry_key]["registration_fitness"] == 1. for p in pairs)}


def main():
    start = perf_counter()
    os.chdir(ROOT)
    # Keep EGL's GPU available while evaluating DINO on the baseline CPU device.
    association.DINO_DEVICE = "cpu"
    fixed_hashes = snapshot_hashes()
    a1_hashes, a3_hashes = verify_artifacts(ABLATION_1), verify_artifacts(ABLATION_3)
    production_files = [ROOT / p for p in ("cad_object_association.py", "CADPointCloudRegistration.py",
        "simulation/wrist_cluster.py", "simulation/nist_task_board_1.py", "simulation/libero_sensor.py")]
    source_hashes = {str(p.relative_to(ROOT)): association.content_hash(p) for p in production_files}
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for name in ("ground_truth", "diagonal", "inputs", "features"):
        (output / name).mkdir(parents=True)
    timing = {}
    stage = perf_counter()
    poses, replay = recover_capture(output / "ground_truth")
    timing["state_recovery_and_exact_replay_time_s"] = perf_counter() - stage
    print("Verified exact camera/RGB/depth replay and all six CAD frames", flush=True)
    registrar = CADPointCloudRegistration()
    assert registrar.config.voxel_size_m == .005
    assert (association.VISUAL_WEIGHT, association.GEOMETRY_WEIGHT, association.FUSION_METHOD) == (.5, .5, "weighted_mean")
    assert association.DINO_MODEL == "dinov2_vits14" and association.IMAGE_SIZE == 224
    protocol = read_json(ABLATION_1 / "protocol.json")
    assert association.DINO_REPO == protocol["encoder_source"]
    assert association.content_hash(association.DINO_HUB_DIR / "checkpoints/dinov2_vits14_pretrain.pth") == protocol["checkpoint_sha256"]
    baseline = read_json(ABLATION_1 / "masked/evaluation.json")
    truth = {r["cad_id"]: r["expected_object_id"] for r in baseline["per_target"]}
    by_object = {oid: cid for cid, oid in truth.items()}
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    object_ids = [r["object_id"] for r in baseline["inputs"]]
    localization = read_json(DATASET / "scene/localization.json")
    rgb = read_rgb(DATASET / "scene/rgb.png")
    depth = np.load(DATASET / "scene/depth.npy")
    K = np.load(DATASET / "scene/intrinsics.npy")
    local_masks, local_inputs, gt_inputs, raw_observed, observed = {}, {}, {}, {}, {}
    for item in localization["objects"]:
        oid = item["object_id"]
        local_masks[oid] = cv2.imread(str(ABLATION_1 / "masked/masks" / f"{oid}.png"), cv2.IMREAD_GRAYSCALE) > 0
        local_inputs[oid] = read_rgb(ABLATION_1 / "masked/inputs" / f"{oid}.png")
        np.testing.assert_array_equal(prepare_masked_input(rgb, local_masks[oid])[0], local_inputs[oid])
        raw_observed[oid] = o3d.io.read_point_cloud(str(ROOT / item["pointcloud_path"]))
        with np.load(ABLATION_3 / "observed_geometry" / f"{oid}.npz") as saved:
            observed[oid] = geometry_from_arrays(saved)[0]
        np.testing.assert_array_equal(np.asarray(observed[oid].points),
            np.asarray(raw_observed[oid].voxel_down_sample(.005).points))
        save_image(output / "inputs" / f"{oid}_localization.png", local_inputs[oid])
    full_geometry, mesh_by_id, cache_hashes = {}, {}, {}
    for cid in truth:
        path = ROOT / library[cid]["file_path"]
        key = association.cache_key(association.CACHE_VERSION, association.content_hash(path), asdict(registrar.config), o3d.__version__)
        cache = ROOT / association.CACHE_DIR / f"{key}_geometry.npz"
        cache_hashes[str(cache.relative_to(ROOT))] = association.content_hash(cache)
        with np.load(cache) as saved:
            full_geometry[cid] = geometry_from_arrays(saved)[0]
        mesh_by_id[cid] = o3d.io.read_triangle_mesh(str(path))
    pairs, diagonal, cad_inputs, paired_local_inputs, paired_gt_inputs = [], [], [], [], []
    stage = perf_counter()
    for cid in truth:
        for oid in object_ids:
            T = np.asarray(poses[by_object[oid]]["camera_T_cad"])
            render = perspective_render(mesh_by_id[cid], T, K, rgb.shape[:2], np.array(library[cid]["base_color_rgb"]))
            canvas = prepare_masked_input(render["rgb"], render["mask"])[0]
            cad_inputs.append(canvas)
            save_image(output / "inputs" / f"{cid}__{oid}_cad.png", canvas)
            points = render["surface_points_camera_m"]
            cad_points = (points - T[:3, 3]) @ T[:3, :3]
            visible = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad_points)).voxel_down_sample(.005)
            pair = {"cad_id": cid, "object_id": oid, "is_correct_pair": truth[cid] == oid,
                    "camera_T_cad": T.tolist(), "rendered_mask_pixels": int(render["mask"].sum()),
                    "full_cad": fixed_pose_metrics(full_geometry[cid], observed[oid], T, registrar),
                    "visible_cad": fixed_pose_metrics(visible, observed[oid], T, registrar)}
            pairs.append(pair)
            if truth[cid] != oid:
                continue
            # Given a verified CAD pose, its first-hit silhouette is an oracle
            # support mask. This intentionally uses known geometry, not an ID-color buffer.
            gt = render["mask"]
            gt_inputs[oid] = prepare_masked_input(rgb, gt)[0]
            save_image(output / "inputs" / f"{oid}_oracle_mask.png", gt_inputs[oid])
            overlap = gt
            surface_distances = render["scene"].compute_distance(
                o3d.core.Tensor(np.asarray(raw_observed[oid].points).astype(np.float32))).numpy()
            # The current localizer unprojects integer pixel indices. Quantify the
            # known half-pixel raster convention separately; do not alter its cloud.
            integer_render = perspective_render(mesh_by_id[cid], T, K, rgb.shape[:2],
                                                  np.array(library[cid]["base_color_rgb"]), pixel_offset=0.)
            integer_overlap = gt & integer_render["mask"]
            corrected = np.asarray(raw_observed[oid].points).copy()
            corrected[:, :2] += .5 * corrected[:, 2, None] / np.array([K[0, 0], K[1, 1]])
            corrected_distances = render["scene"].compute_distance(o3d.core.Tensor(corrected.astype(np.float32))).numpy()
            local_cad = prepare_masked_input(render["rgb"], local_masks[oid])[0]
            gt_cad = prepare_masked_input(render["rgb"], gt)[0]
            paired_local_inputs.append(local_cad)
            paired_gt_inputs.append(gt_cad)
            old = next(r for a in baseline["associations"] if a["cad_id"] == cid
                       for r in a["candidate_ranking"] if r["object_id"] == oid)
            shifted = T.copy(); shifted[0, 3] += .03
            record = {"cad_id": cid, "object_id": oid,
                "baseline_visual_raw": old["visual_raw"], "baseline_full_cad_registration": old["geometry_raw"],
                "oracle_full_cad": pair["full_cad"], "oracle_visible_cad": pair["visible_cad"],
                "point_to_exact_mesh": distance_summary(surface_distances),
                "half_pixel_corrected_point_to_mesh_diagnostic_only": distance_summary(corrected_distances),
                "localization_vs_oracle_silhouette_iou": float((local_masks[oid] & gt).sum() / (local_masks[oid] | gt).sum()),
                "depth_difference_pixel_centers": distance_summary(np.abs(depth[overlap] - render["depth_m"][overlap])),
                "depth_difference_integer_rays": distance_summary(np.abs(depth[integer_overlap] - integer_render["depth_m"][integer_overlap])),
                "self_geometry_control": fixed_pose_metrics(observed[oid], observed[oid], np.eye(4), registrar),
                "translated_30mm_camera_x_control": fixed_pose_metrics(full_geometry[cid], observed[oid], shifted, registrar),
            }
            assert record["self_geometry_control"]["registration_fitness"] == 1.
            assert record["self_geometry_control"]["observed_to_cad_rmse_m"] == 0.
            diagonal.append(record)
            case = output / "diagonal" / cid; case.mkdir()
            save_image(case / "cad_rgb.png", render["rgb"])
            save_image(case / "cad_mask.png", render["mask"].astype(np.uint8) * 255)
            save_image(case / "cad_same_localization_mask_input.png", local_cad)
            save_image(case / "cad_same_oracle_mask_input.png", gt_cad)
            np.savez_compressed(case / "visible_surface.npz", camera_T_cad=T,
                surface_points_camera_m=points, points_cad_m=np.asarray(visible.points),
                optical_depth_m=render["depth_m"])
            aligned = o3d.geometry.PointCloud(full_geometry[cid]).transform(T).paint_uniform_color([0, .5, 1])
            observed_color = o3d.geometry.PointCloud(raw_observed[oid]).paint_uniform_color([1, 0, 0])
            o3d.io.write_point_cloud(str(case / "oracle_alignment.ply"), aligned + observed_color)
        print(f"Oracle poses scored: {cid}, all six candidates", flush=True)
    timing["all_36_rendering_and_fixed_pose_geometry_time_s"] = perf_counter() - stage
    stage = perf_counter()
    model = association.load_dino_encoder()
    assert str(next(model.parameters()).device) == "cpu" and torch.get_num_threads() == 1
    timing["dino_model_load_time_s"] = perf_counter() - stage
    stage = perf_counter()
    local_features = unit_features([local_inputs[oid] for oid in object_ids])
    gt_features = unit_features([gt_inputs[oid] for oid in object_ids])
    template_features = unit_features(cad_inputs)
    same_local_features, same_gt_features = unit_features(paired_local_inputs), unit_features(paired_gt_inputs)
    timing["all_dino_feature_extraction_time_s"] = perf_counter() - stage
    for pair, feature in zip(pairs, template_features, strict=True):
        pair["visual_raw"] = float(np.clip(feature @ local_features[object_ids.index(pair["object_id"])], -1, 1))
    for i, record in enumerate(diagonal):
        index = object_ids.index(record["object_id"])
        pair = next(p for p in pairs if p["cad_id"] == record["cad_id"] and p["object_id"] == record["object_id"])
        record.update(
            oracle_pose_localization_cosine=pair["visual_raw"],
            oracle_pose_same_localization_mask_cosine=float(np.clip(same_local_features[i] @ local_features[index], -1, 1)),
            oracle_pose_same_cad_silhouette_cosine=float(np.clip(same_gt_features[i] @ gt_features[index], -1, 1)),
            self_image_cosine=float(local_features[index] @ local_features[index]),
        )
        assert abs(record["self_image_cosine"] - 1) < 1e-12
    evaluations = {mode: evaluate_pairs(pairs, truth, mode) for mode in ("full_cad", "visible_cad")}
    timing["elapsed_before_report_s"] = perf_counter() - start
    np.savez_compressed(output / "features/dino_cls.npz", object_ids=object_ids,
        local_observed=local_features, oracle_mask_observed=gt_features,
        cad_ids=[p["cad_id"] for p in pairs], candidate_ids=[p["object_id"] for p in pairs],
        perspective_cad=template_features, same_local_mask_cad=same_local_features, same_cad_silhouette=same_gt_features)
    save_json(output / "all_pairs.json", pairs)
    save_json(output / "correct_pairs.json", diagonal)
    save_json(output / "evaluations.json", evaluations)
    save_csv(output / "all_pair_scores.csv", [{"cad_id": p["cad_id"], "object_id": p["object_id"],
        "is_correct_pair": p["is_correct_pair"], "visual_raw": p["visual_raw"],
        **{f"{mode}_{key}": p[mode][key] for mode in ("full_cad", "visible_cad")
           for key in ("registration_fitness", "observed_to_cad_rmse_m", "C_obs", "C_cad", "C_f1", "correspondence_count")},
        **{f"{mode}_fused_score": association.fuse_scores(association.normalize_visual_score(p["visual_raw"]),
            p[mode]["registration_fitness"]) for mode in ("full_cad", "visible_cad")}} for p in pairs])
    save_json(output / "protocol.json", {
        "objective": "Fixed-pose mathematical and observation-consistency diagnostic for all six objects",
        "pose_source": replay, "baseline": str(ABLATION_1.relative_to(ROOT)), "geometry_cache_source": str(ABLATION_3.relative_to(ROOT)),
        "candidate_pose_hypothesis": "Each candidate's verified true body pose is applied to every native CAD, with no pose fitting or scaling",
        "ground_truth_use": "Oracle input for diagnostics only; wrong-CAD pose controls use the candidate's true frame; not an unbiased association benchmark",
        "camera": "Saved 768x768 pinhole intrinsics and world_T_camera; OpenCV camera axes; metric CAD coordinates",
        "ray_pixel_offset": .5, "visual": "DINOv2-small normalized CLS, same checkpoint/ImageNet normalization, white 224x224 bicubic letterbox",
        "cad_shading": "Same production .35 + .65 * abs(normal dot toward-camera) and saved CAD materials, evaluated per perspective ray",
        "mask_diagnostics": "Current localization input versus full visible CAD; additionally same localization mask and same exact-pose raycast CAD silhouette on both images",
        "oracle_mask_source": "First-hit CAD raycasting at the verified pose; self-occlusion included; the six separated objects have no mutual occlusion",
        "simulator_id_masks_used": False,
        "mask_diagnostic_limit": "Shared observation/ground-truth masks are consistency controls and are not used for main pair rankings",
        "geometry": "Saved 5mm observed and full-CAD clouds; perspective visible CAD sampled at image pixels and voxelized in CAD frame at 5mm",
        "geometry_evaluation": "Open3D evaluate_registration at fixed transform, threshold 7.5mm; original observed_to_cad_rmse plus bidirectional coverage",
        "dense_surface_diagnostic": "Continuous point-to-triangle distance separates sampling/voxel effects from CAD/frame errors",
        "half_pixel_diagnostic": "Add 0.5*z/fx and 0.5*z/fy only in an auxiliary copy; production clouds and scores remain unchanged",
        "negative_control": "Correct full CAD shifted +30mm along camera X; fixed illustrative perturbation, no search or tuning",
        "fusion": "0.5 * ((raw CLS cosine + 1)/2) + 0.5 * fixed-pose observed fitness",
        "ransac_icp_fpfh_or_localization_rerun": False, "production_modified": False,
        "cad_feature_checkpoint_sha256": protocol["checkpoint_sha256"], "runtime": timing,
        "source_sha256": source_hashes, "full_cad_cache_sha256": cache_hashes,
        "diagnostic_source_sha256": {name: association.content_hash(ROOT / name) for name in (
            "scripts/run_wrist_oracle_pose_diagnostic.py", "scripts/wrist_oracle_pose.py",
            "scripts/wrist_oracle_pose_report.py", "tests/test_wrist_oracle_pose.py")},
    })
    save_report(output, diagonal, pairs, evaluations, truth, library, timing)
    assert snapshot_hashes() == fixed_hashes
    assert verify_artifacts(ABLATION_1) == a1_hashes and verify_artifacts(ABLATION_3) == a3_hashes
    assert all(association.content_hash(ROOT / p) == h for p, h in source_hashes.items())
    assert all(association.content_hash(ROOT / p) == h for p, h in cache_hashes.items())
    save_json(output / "validation.json", {**replay, "all_self_image_cosines_equal_1_within_1e_12": True,
        "all_self_geometry_fitness_1_rmse_0": True, "fixed_dataset_prior_results_caches_and_production_unchanged": True,
        "pair_count": len(pairs), "correct_pairs": len(diagonal), "voxel_size_m": .005,
        "distance_threshold_m": .0075, "dino_device": "cpu", "torch_threads": torch.get_num_threads()})
    (output / "SHA256SUMS").write_text("".join(f"{association.content_hash(p)}  {p.relative_to(output).as_posix()}\n"
        for p in sorted(output.rglob("*")) if p.is_file()), encoding="utf-8")
    print({m: e["accuracy"] for m, e in evaluations.items()}, flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
