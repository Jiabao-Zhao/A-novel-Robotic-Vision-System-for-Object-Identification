"""Frozen 20-frame T-LESS pilot: depth localization -> CAD IDs -> BOP mask AP.

No GT annotations are read until every frame's predictions have been saved.
Reuse the 30-CAD/42-view CUDA feature cache from the previous oracle-mask pilot.
Run in the existing WSL GPU environment with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1.
No production changes, new SAM calls, simulation captures, or parameter fitting.
"""

import os
import subprocess
import sys
import zipfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/cache"
for dependency in (CACHE / "bop_eval_deps", CACHE / "bop_toolkit"):
    sys.path.insert(0, str(dependency))

import cv2
import numpy as np
import open3d as o3d
import torch
from bop_toolkit_lib import pycoco_utils

import cad_object_association as association
from point_cloud_localization import PointCloudConfig, PointCloudLocalization
from visualize_localization_projection import project_with_z_buffer
from scripts.run_wrist_input_ablation import read_json, save_json, read_rgb, save_image
from scripts.run_wrist_blenderproc_semantic import prepare_rgba
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
from scripts.sam6d_patch_matching import masked_patch_descriptors
from scripts.run_surface_verification import score_scene, METHODS
from scripts.surface_verification import CadSurface
from scripts.bop_segmentation_metrics import encode_mask, localization_metrics, coco_metrics


OUTPUT = ROOT / "outputs/bop_localization_20260917"
DATA = CACHE / "bop_tless/data/tless"
TEMPLATES = ROOT / "outputs/surface_verification_20260917/stage4_tless"
SEED = 20260917
PROTECTED = ("point_cloud_localization.py", "cad_object_association.py", "scripts/surface_verification.py",
             "scripts/sam6d_patch_matching.py", "scripts/run_surface_verification.py")


def plan_and_extract():
    targets = read_json(DATA / "test_targets_bop19.json")
    frames = sorted({(t["scene_id"], t["im_id"]) for t in targets})
    plan = [{"scene_id": s, "im_id": min(i for ss, i in frames if ss == s)}
            for s in sorted({s for s, _ in frames})]
    protocol = {"task": "BOP 2023 model-based 2D segmentation of unseen objects; 20-frame pilot",
        "selection": "First official target image of every scene, fixed before predictions", "plan": plan,
        "full_official_image_count": len(frames), "seed": SEED,
        "localization": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(PointCloudConfig()).items()},
        "mask": "Nearest-pixel projection of cleaned original-resolution clouds; no filling or dilation",
        "input": "Entire RGB-D frame; per-image K; raw_depth*depth_scale*0.001 meters; integer pixel rays",
        "appearance": "Frozen DINOv2-L/14 CUDA FP32; 42 cached BlenderProc views; white 224 letterbox",
        "scoring": "Unchanged global top5, CLS-selected patch and visual mean; existing surface geometry and fusion",
        "geometry": "42 fixed rotations, 30 translation-only ICP iterations, 5mm sampling; no table constraint, as prior T-LESS pilot",
        "prediction": "One highest-scoring CAD ID per localized mask for each method; all 30 CADs; no score threshold or NMS",
        "evaluation": "BOP COCO fork; GT visibility <0.10 ignored; all masks/IDs/poses withheld until prediction files exist",
        "localization_diagnostics": "One-to-one maximum-cardinality matching at IoU .50/.75; merge/split flags at .10 GT coverage, diagnostics only",
        "protected_sha256": {p: association.content_hash(ROOT / p) for p in PROTECTED},
        "runner_sha256": association.content_hash(Path(__file__)),
        "metrics_sha256": association.content_hash(ROOT / "scripts/bop_segmentation_metrics.py"),
        "versions": {p: version(p) for p in ("numpy", "scipy", "torch", "open3d", "pycocotools")},
        "template_index_sha256": association.content_hash(TEMPLATES / "template_features/index.json"),
        "toolkit_commit": subprocess.check_output(["git", "-C", str(CACHE / "bop_toolkit"), "rev-parse", "HEAD"], text=True).strip(),
        "coco_commit": subprocess.check_output(["git", "-C", str(CACHE / "bop_cocoapi"), "rev-parse", "HEAD"], text=True).strip()}
    path = OUTPUT / "protocol.json"
    if path.exists():
        assert read_json(path) == protocol, "Protocol changed; use a separate output directory."
    else:
        save_json(path, protocol)
    with zipfile.ZipFile(CACHE / "bop_tless/tless_test_primesense_bop19.zip") as archive:
        for item in plan:
            prefix = f"test_primesense/{item['scene_id']:06}/"
            frame = f"{item['im_id']:06}"
            members = [n for n in archive.namelist() if n.startswith(prefix) and
                       (n.endswith(".json") or f"/{frame}" in n)]
            archive.extractall(DATA, members=[n for n in members if not (DATA / n).exists()])
    return plan, protocol


def frame_input(item):
    source = DATA / "test_primesense" / f"{item['scene_id']:06}"
    name = f"{item['im_id']:06}"
    camera = read_json(source / "scene_camera.json")[str(item["im_id"])]
    rgb = read_rgb(source / "rgb" / f"{name}.png")
    raw = cv2.imread(str(source / "depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    K = np.array(camera["cam_K"]).reshape(3, 3)
    intrinsics = {"width": rgb.shape[1], "height": rgb.shape[0], "fx": K[0, 0], "fy": K[1, 1],
                  "cx": K[0, 2], "cy": K[1, 2]}
    return rgb, raw, camera["depth_scale"] * .001, K, intrinsics


def frame_folder(item):
    return OUTPUT / f"scene_{item['scene_id']:06}_{item['im_id']:06}"


def localize(rgb, raw, scale, intrinsics, folder):
    """Call existing numerical stages in memory; avoid unnecessary PLY exports."""
    destination = folder / "localization.json"
    if destination.exists():
        saved = read_json(destination)
        return {r["object_id"]: cv2.imread(str(folder / "masks" / f"{r['object_id']}.png"), 0) > 0
                for r in saved["regions"]}, saved
    start = perf_counter()
    o3d.utility.random.seed(SEED)
    localizer = PointCloudLocalization(PointCloudConfig())
    workspace = localizer.rgbd_to_pointcloud(rgb, raw, scale, intrinsics)
    downsampled = workspace.voxel_down_sample(localizer.config.voxel_size_m)
    objects, plane, _ = localizer.segment_table_plane(downsampled)
    clusters = localizer.cluster_objects(objects, intrinsics, plane)
    raw_objects = localizer.remove_table_from_raw_cloud(workspace, plane)
    masks, records = {}, []
    for i, cluster in enumerate(clusters, 1):
        clean = localizer.clean_raw_cluster(localizer.crop_raw_cluster(raw_objects, cluster), plane)
        projected, count = project_with_z_buffer(clean, intrinsics)
        mask = np.isfinite(projected)
        oid = f"object_{i:03}"
        if not mask.any():
            continue
        assert np.all(raw[mask] > 0), "Projected support must come from measured pixels."
        masks[oid] = mask
        records.append({"object_id": oid, "cluster_points": len(cluster.points), "clean_points": len(clean.points),
            "projected_points": count, "pixels": int(mask.sum()),
            "max_projected_depth_error_m": float(np.max(np.abs(projected[mask] - raw[mask] * scale)))})
    elapsed = perf_counter() - start
    (folder / "masks").mkdir(parents=True, exist_ok=True)
    for oid, mask in masks.items():
        save_image(folder / "masks" / f"{oid}.png", mask.astype(np.uint8) * 255)
    saved = {"regions": records, "plane_model": list(plane), "wall_s": elapsed,
             "initial_cluster_count": len(clusters), "workspace_point_count": len(workspace.points)}
    save_json(destination, saved)
    print(folder.name, "localized", len(masks), "regions", f"{elapsed:.2f}s", flush=True)
    return masks, saved


def cached_templates():
    """Read exact saved feature arrays and renderer alpha masks; no re-rendering."""
    folder = TEMPLATES / "templates42"
    index = read_json(TEMPLATES / "template_features/index.json")
    expected = [f"obj_{cid:06}/{v}" for cid in range(1, 31) for v in range(1, 43)]
    assert index["keys"] == expected and index["model"] == "dinov2_vitl14" and index["device"] == "cuda"
    with np.load(TEMPLATES / "template_features/features.npz") as features:
        cls, patch = features["cls_features"], features["patch_tokens"]
    assert len(cls) == len(patch) == 1260
    templates, cads = {}, {}
    for cid in range(1, 31):
        name = f"obj_{cid:06}"
        mesh = o3d.io.read_triangle_mesh(str(DATA / "models_cad" / f"{name}.ply"))
        mesh.scale(.001, center=(0, 0, 0))
        cads[name] = CadSurface(mesh)
        descriptors = []
        for v in range(1, 43):
            i = (cid - 1) * 42 + v - 1
            path = folder / f"{name}_view_{v:02}.png"
            assert association.content_hash(path) == index["sha256"][i]
            rgba = cv2.cvtColor(cv2.imread(str(folder / "renders" / name / f"view_{v:02}_rgba.png"), -1), cv2.COLOR_BGRA2RGBA)
            _, mask, bounds = prepare_rgba(rgba)
            x1, y1, x2, y2 = bounds
            weights = patch_occupancy(letterbox_mask(mask[y1:y2 + 1, x1:x2 + 1]))
            descriptors.append(masked_patch_descriptors(patch[i], weights).cuda())
        templates[name] = {"cls": cls[(cid - 1) * 42:cid * 42], "patch": descriptors}
    rotations = np.load(folder / "cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    visible = {cid: [cad.visible_points(R) for R in rotations] for cid, cad in cads.items()}
    return cads, visible, rotations, templates


def export_predictions(item, masks, pairs, seconds):
    """Ground truth is deliberately absent from this interface."""
    result = {}
    for method in METHODS:
        rows = []
        for oid, mask in masks.items():
            ranked = sorted((p for p in pairs if p["object_id"] == oid and p[method] is not None),
                            key=lambda p: (-p[method], p["cad_id"]))
            if ranked:
                best = ranked[0]
                rows.append({"scene_id": item["scene_id"], "image_id": item["im_id"],
                    "category_id": int(best["cad_id"].split("_")[1]), "score": float(best[method]),
                    "segmentation": encode_mask(mask), "time": float(seconds), "object_id": oid})
        result[method] = rows
    return result


def evaluate_saved(plan):
    """Annotation access starts here, after predictions for ALL frames exist."""
    all_gt = {"info": {}, "licenses": [], "categories": [{"id": i, "name": str(i)} for i in range(1, 31)],
              "images": [], "annotations": []}
    all_predictions = {m: [] for m in METHODS}
    localization, frames = [], []
    classification = {m: {"correct": 0, "matched_regions": 0} for m in METHODS}
    for ordinal, item in enumerate(plan, 1):
        folder = frame_folder(item)
        predicted = read_json(folder / "predictions.json")
        source = DATA / "test_primesense" / f"{item['scene_id']:06}"
        gt = read_json(source / "scene_gt.json")[str(item["im_id"])]
        info = read_json(source / "scene_gt_info.json")[str(item["im_id"])]
        rgb, _, _, _, _ = frame_input(item)
        annotations = []
        for i, (record, metadata) in enumerate(zip(gt, info, strict=True)):
            mask = cv2.imread(str(source / "mask_visib" / f"{item['im_id']:06}_{i:06}.png"), 0) > 0
            if not mask.any():
                continue
            ann = pycoco_utils.create_annotation_info(len(all_gt["annotations"]) + len(annotations) + 1,
                ordinal, record["obj_id"], mask, metadata["bbox_obj"], ignore=metadata["visib_fract"] < .1)
            ann["segmentation"] = encode_mask(mask)
            annotations.append(ann)
        all_gt["images"].append({"id": ordinal, "width": rgb.shape[1], "height": rgb.shape[0]})
        all_gt["annotations"].extend(annotations)
        masks = {r["object_id"]: cv2.imread(str(folder / "masks" / f"{r['object_id']}.png"), 0) > 0
                 for r in read_json(folder / "localization.json")["regions"]}
        metrics = localization_metrics(masks, annotations)
        localization.append({**item, **metrics})
        per_frame = {**item, "predicted_regions": len(masks), "eligible_instances": metrics["eligible_instances"],
                     "matched_iou50": metrics["iou_0.50"]["matched"], "predictions": {}}
        expected = {r["object_id"]: next(a["category_id"] for a in annotations if a["id"] == r["gt_id"])
                    for r in metrics["iou_0.50"]["matches"]}
        for method, predictions in predicted.items():
            all_predictions[method].extend([{**p, "image_id": ordinal} for p in predictions])
            per_frame["predictions"][method] = [{**{k: p[k] for k in ("object_id", "category_id", "score")},
                "matched_gt_category_id": expected.get(p["object_id"])} for p in predictions]
            guesses = {p["object_id"]: p["category_id"] for p in predictions}
            classification[method]["matched_regions"] += len(expected)
            classification[method]["correct"] += sum(guesses.get(oid) == cid for oid, cid in expected.items())
        frames.append(per_frame)
        # Minimal visual evidence: predicted regions and annotated CAD identities.
        overlay = rgb.copy()
        for i, (oid, mask) in enumerate(masks.items()):
            color = np.array(((73 * i + 40) % 200 + 40, (117 * i + 90) % 200 + 40, (157 * i + 20) % 200 + 40))
            overlay[mask] = (.55 * overlay[mask] + .45 * color).astype(np.uint8)
            y, x = np.nonzero(mask)
            cv2.putText(overlay, oid, (int(x.min()), max(14, int(y.min()))), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0), 1)
        truth_view = rgb.copy()
        for ann in annotations:
            if ann["ignore"]:
                continue
            from pycocotools import mask as mu
            mask = mu.decode(ann["segmentation"]).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(truth_view, contours, -1, (0, 255, 0), 1)
            y, x = np.nonzero(mask)
            cv2.putText(truth_view, str(ann["category_id"]), (int(x.min()), max(14, int(y.min()))), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0), 1)
        save_image(folder / "localization_vs_ground_truth.png", np.concatenate((overlay, truth_view), axis=1))
    save_json(OUTPUT / "coco_ground_truth.json", all_gt)
    save_json(OUTPUT / "localization_metrics.json", localization)
    save_json(OUTPUT / "per_frame.json", frames)
    scores = {}
    for method in METHODS:
        save_json(OUTPUT / f"{method}_coco_predictions.json", all_predictions[method])
        scores[method] = coco_metrics(all_gt, all_predictions[method])
    best_iou = [v for r in localization for v in r["best_iou_per_gt"]]
    count = sum(r["eligible_instances"] for r in localization)
    loc = {"eligible_instances": count, "predicted_regions": sum(r["predicted_regions"] for r in localization),
        "ignored_instances": sum(r["ignored_instances"] for r in localization),
        "mean_best_iou_all_gt": float(np.mean(best_iou)),
        "suspected_merged_regions": sum(len(r["suspected_merged_regions"]) for r in localization),
        "fragmented_or_duplicate_gt": sum(len(r["fragmented_or_duplicate_gt_ids"]) for r in localization)}
    for threshold in ("0.50", "0.75"):
        matched = sum(r[f"iou_{threshold}"]["matched"] for r in localization)
        loc[f"iou_{threshold}"] = {"matched": matched, "recall": matched / count, "missed": count - matched,
            "unmatched_predictions": sum(r[f"iou_{threshold}"]["unmatched_predictions"] for r in localization)}
    return {"frames": len(plan), "localization": loc, "segmentation": scores,
            "classification_on_iou50_matched_regions": classification}


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    plan, protocol = plan_and_extract()
    for item in plan:
        rgb, raw, scale, _, intrinsics = frame_input(item)
        localize(rgb, raw, scale, intrinsics, frame_folder(item))
    stage = perf_counter()
    cads, visible, rotations, templates = cached_templates()
    setup_s = perf_counter() - stage
    for item in plan:
        folder = frame_folder(item)
        if (folder / "predictions.json").exists():
            continue
        rgb, raw, scale, K, intrinsics = frame_input(item)
        masks, localized = localize(rgb, raw, scale, intrinsics, folder)
        stage = perf_counter()
        score_scene(cads, visible, rotations, templates, raw.astype(float) * scale, masks, K, rgb, {}, folder, offset=0.)
        seconds = perf_counter() - stage + localized["wall_s"]
        pairs = read_json(folder / "pair_scores.json")
        assert len(pairs) == 30 * len(masks)
        predictions = export_predictions(item, masks, pairs, seconds)
        save_json(folder / "predictions.json", predictions)
        save_json(folder / "inference_runtime.json", {"full_pipeline_s": seconds, "localization_s": localized["wall_s"],
            "includes": "Localization, observed CUDA model load/features, all-pair visual/geometry ablations and per-frame intermediate writes; excludes raw input loading and shared template setup"})
        print(folder.name, "predictions saved", flush=True)
    summary = evaluate_saved(plan)
    times = [read_json(frame_folder(i) / "inference_runtime.json")["full_pipeline_s"] for i in plan]
    summary["runtime"] = {"mean_full_pipeline_s_per_image": float(np.mean(times)), "total_inference_s": sum(times),
        "template_cache_setup_s_this_invocation": setup_s, "wall_s_this_invocation": perf_counter() - start,
        "gpu": torch.cuda.get_device_name(), "all_methods_computed_together": True, "zero_proposal_frames_included": True}
    assert protocol["protected_sha256"] == {p: association.content_hash(ROOT / p) for p in PROTECTED}
    summary["protected_sources_unchanged"] = True
    save_json(OUTPUT / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
