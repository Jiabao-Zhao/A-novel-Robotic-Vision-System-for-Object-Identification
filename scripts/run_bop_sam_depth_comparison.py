"""Frozen T-LESS pilot: SAM3, depth localization, and their proposal pool.

All variants share table checking, patch-top-five selection, and our surface
score. Exact old depth-mask descriptors/translations are reused; old geometry
scores are not. Ground truth is read only after all predictions are written.
"""

from collections import defaultdict
import os
from time import perf_counter

from scripts import run_bop_localization_benchmark as baseline
import cv2
import numpy as np
import torch
from bop_toolkit_lib import pycoco_utils
from scripts.bop_segmentation_metrics import encode_mask, localization_metrics, coco_metrics
from scripts.compare_workbench_framework import pose_status
from scripts.run_surface_verification import feature_cache
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
from scripts.run_wrist_branch_order import select_views, pair_result, METHODS
from scripts.sam6d_patch_matching import semantic_score, masked_patch_descriptors, appearance_score
from scripts.surface_verification import bbox, depth_points, downsample, translation_fit, equal_penalty_scores


ROOT, BASE = baseline.ROOT, baseline.OUTPUT
OUTPUT = ROOT / "outputs/bop_sam_depth_20260919"
VARIANTS = ("sam_leading", "depth_leading", "hybrid")
NMS_IOU = .5
read_json, save_json = baseline.read_json, baseline.save_json


def masks_for_frame(item, output=OUTPUT, proposal_dir="sam3"):
    name = baseline.frame_folder(item).name
    sam = read_json(output / proposal_dir / name / "proposals.json")
    depth = read_json(BASE / name / "localization.json")
    paths = {f"depth_{r['object_id'].split('_')[1]}": BASE / name / "masks" / f"{r['object_id']}.png"
             for r in depth["regions"]}
    paths.update({r["object_id"]: output / proposal_dir / name / "masks" / f"{r['object_id']}.png" for r in sam})
    masks = {oid: cv2.imread(str(path), 0) > 0 for oid, path in paths.items()}
    return masks, depth


def mask_nms(predictions, masks):
    """Same fixed, class-aware mask NMS in each variant; CAD score sets order."""
    retained, suppressed = [], []
    for row in sorted(predictions, key=lambda r: (-r["score"], r["object_id"])):
        current = masks[row["object_id"]]
        duplicate = None
        for old in retained:
            if old["category_id"] != row["category_id"]:
                continue
            other = masks[old["object_id"]]
            iou = np.count_nonzero(current & other) / np.count_nonzero(current | other)
            if iou > NMS_IOU:
                duplicate = old["object_id"]
                break
        if duplicate is None:
            retained.append(row)
        else:
            suppressed.append({"object_id": row["object_id"], "retained_object_id": duplicate})
    return retained, suppressed


def variant_ids(masks, variant):
    prefix = {"sam_leading": "sam_", "depth_leading": "depth_", "hybrid": ""}[variant]
    return [oid for oid in masks if oid.startswith(prefix)]


def make_predictions(item, masks, pairs, times):
    predictions, suppression = {}, {}
    for variant in VARIANTS:
        selected = variant_ids(masks, variant)
        rows = []
        for oid in selected:
            ranked = sorted((r for r in pairs if r["object_id"] == oid and r["score"] is not None),
                            key=lambda r: (-r["score"], r["cad_id"]))
            if ranked:
                winner = ranked[0]
                rows.append({"scene_id": item["scene_id"], "image_id": item["im_id"],
                    "category_id": int(winner["cad_id"].split("_")[1]), "object_id": oid,
                    "score": winner["score"], "segmentation": encode_mask(masks[oid]),
                    "time": sum(times[k] for k in selected)})
        predictions[variant], suppression[variant] = mask_nms(rows, masks)
    return predictions, suppression


def encode_sam(plan, output=OUTPUT, proposal_dir="sam3", cache_folder=None):
    paths, occupancy = {}, {}
    for item in plan:
        name = baseline.frame_folder(item).name
        masks, _ = masks_for_frame(item, output, proposal_dir)
        rgb, _, _, _, _ = baseline.frame_input(item)
        folder = output / "matching" / name / "inputs"
        folder.mkdir(parents=True, exist_ok=True)
        for oid in variant_ids(masks, "sam_leading"):
            key = f"{name}/{oid}"
            canvas, _, bounds = prepare_masked_input(rgb, masks[oid])
            paths[key] = folder / f"{oid}.png"
            baseline.save_image(paths[key], canvas)
            x1, y1, x2, y2 = bounds
            occupancy[key] = patch_occupancy(letterbox_mask(masks[oid][y1:y2 + 1, x1:x2 + 1]))
    if not paths:
        return {}
    features = feature_cache(paths, cache_folder or output / "sam_features")
    return {key: {"cls": cls, "patch": masked_patch_descriptors(patch, occupancy[key]).cuda()}
            for key, (cls, patch) in features.items()}


def score_frame(item, engine, observed):
    name = baseline.frame_folder(item).name
    folder = OUTPUT / "matching" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "predictions.json").exists():
        return
    cads, visible, rotations, templates = engine
    masks, localized = masks_for_frame(item)
    _, raw, scale, K, _ = baseline.frame_input(item)
    depth = raw.astype(float) * scale
    assert all(m.shape == depth.shape for m in masks.values())
    table = np.array(localized["plane_model"], float)
    table /= np.linalg.norm(table[:3])
    if table[3] < 0:
        table *= -1
    previous = defaultdict(list)
    for row in read_json(BASE / name / "all_view_scores.json"):
        previous[row["cad_id"], row["object_id"]].append(row)
    all_views, pairs, times = [], [], {}
    for oid, mask in masks.items():
        tick = perf_counter()
        points = downsample(depth_points(depth, mask, K, 0.))
        for cid, cad in cads.items():
            views = []
            if oid.startswith("sam_"):
                query = observed[f"{name}/{oid}"]
                semantic = semantic_score(query["cls"], templates[cid]["cls"])
            else:
                old = sorted(previous[cid, oid.replace("depth_", "object_")], key=lambda r: r["view_index_1based"])
                assert len(old) == 42
            for v, R in enumerate(rotations):
                if oid.startswith("sam_"):
                    T = np.eye(4)
                    T[:3, :3] = R
                    if len(points):
                        T[:3, 3], _ = translation_fit(visible[cid][v], points)
                    patch = appearance_score(query["patch"], templates[cid]["patch"][v])["appearance_score"]
                    global_score, cls = semantic["semantic_score"], semantic["cls_view_cosines"][v]
                else:
                    T = np.array(old[v]["camera_T_cad"])
                    global_score, patch, cls = old[v]["global"], old[v]["patch"], old[v]["cls_cosine"]
                views.append({"cad_id": cid, "object_id": oid, "view_index_1based": v + 1,
                    "camera_T_cad": T.tolist(), "cls": cls, "G": global_score, "P": patch, "H": None,
                    "pose_status": pose_status(cad, T, K, depth.shape, table) if len(points) else "no_valid_depth"})
            for row in select_views(views, "table_patch5"):
                rendered = cad.render(np.asarray(row["camera_T_cad"]), K, depth.shape, bbox(mask), 0.)
                assert rendered is not None and not rendered["clipped"]
                x1, y1, x2, y2 = rendered["roi"]
                metrics = equal_penalty_scores(depth[y1:y2, x1:x2], mask[y1:y2, x1:x2], rendered["depth"])
                row.update(metrics)
                row["H"] = metrics["geometry_score"]
            pairs.append(pair_result(views, METHODS["table_then_patch5"]))
            all_views.extend(views)
        times[oid] = perf_counter() - tick
    assert len(pairs) == 30 * len(masks) and len(all_views) == 42 * len(pairs)
    save_json(folder / "pair_scores.json", pairs)
    save_json(folder / "view_scores.json", all_views)
    predictions, suppression = make_predictions(item, masks, pairs, times)
    save_json(folder / "predictions.json", predictions)
    save_json(folder / "nms.json", suppression)
    save_json(folder / "runtime.json", {"new_matching_s_by_region": times,
        "cached_depth_localization_s": localized["wall_s"],
        "sam3_cloud_request_s": read_json(OUTPUT / "sam3" / name / "request.json")["elapsed_request_s"],
        "timing_scope": "Cache-assisted comparison; depth visual scores and translations reused, SAM computed fresh; not end-to-end speed comparison"})
    print(name, {v: len(predictions[v]) for v in VARIANTS}, flush=True)


def evaluate(plan, output=OUTPUT, proposal_dir="sam3",
             scope="20-frame T-LESS pilot, not full benchmark; fixed generic-object SAM3 cloud setup"):
    """Annotations enter only here, once every frame has saved predictions."""
    assert all((output / "matching" / baseline.frame_folder(i).name / "predictions.json").exists() for i in plan)
    gt = {"info": {}, "licenses": [], "categories": [{"id": i, "name": str(i)} for i in range(1, 31)],
          "images": [], "annotations": []}
    predictions = {v: [] for v in VARIANTS}
    per_frame = []
    for ordinal, item in enumerate(plan, 1):
        name = baseline.frame_folder(item).name
        source = baseline.DATA / "test_primesense" / f"{item['scene_id']:06}"
        truth = read_json(source / "scene_gt.json")[str(item["im_id"])]
        info = read_json(source / "scene_gt_info.json")[str(item["im_id"])]
        rgb, _, _, _, _ = baseline.frame_input(item)
        annotations = []
        for index, (obj, metadata) in enumerate(zip(truth, info, strict=True)):
            mask = cv2.imread(str(source / "mask_visib" / f"{item['im_id']:06}_{index:06}.png"), 0) > 0
            if not mask.any():
                continue
            ann = pycoco_utils.create_annotation_info(len(gt["annotations"]) + len(annotations) + 1,
                ordinal, obj["obj_id"], mask, metadata["bbox_obj"], ignore=metadata["visib_fract"] < .1)
            ann["segmentation"] = encode_mask(mask)
            annotations.append(ann)
        gt["images"].append({"id": ordinal, "width": rgb.shape[1], "height": rgb.shape[0]})
        gt["annotations"].extend(annotations)
        masks, _ = masks_for_frame(item, output, proposal_dir)
        saved = read_json(output / "matching" / name / "predictions.json")
        frame = {**item, "methods": {}}
        for variant in VARIANTS:
            selected = {oid: masks[oid] for oid in variant_ids(masks, variant)}
            metrics = localization_metrics(selected, annotations)
            predicted = saved[variant]
            detection_masks = {r["object_id"]: masks[r["object_id"]] for r in predicted}
            detected = localization_metrics(detection_masks, annotations)
            categories = {a["id"]: a["category_id"] for a in annotations}
            guesses = {r["object_id"]: r["category_id"] for r in predicted}
            matched = detected["iou_0.50"]["matches"]
            correct = sum(guesses[r["object_id"]] == categories[r["gt_id"]] for r in matched)
            frame["methods"][variant] = {"proposal_metrics": metrics,
                "detections": [{k: r[k] for k in ("object_id", "category_id", "score")} for r in predicted],
                "matched_detections_iou50": len(matched), "correct_identified_instances_iou50": correct}
            predictions[variant].extend([{**r, "image_id": ordinal} for r in predicted])
        per_frame.append(frame)
    summary = {}
    for variant in VARIANTS:
        entries = [f["methods"][variant] for f in per_frame]
        metrics = [r["proposal_metrics"] for r in entries]
        total = sum(m["eligible_instances"] for m in metrics)
        matched = sum(r["matched_detections_iou50"] for r in entries)
        summary[variant] = {"eligible_instances": total,
            "candidate_regions": sum(m["predicted_regions"] for m in metrics),
            "empty_candidate_frames": sum(m["predicted_regions"] == 0 for m in metrics),
            "proposal_recall_iou50": sum(m["iou_0.50"]["matched"] for m in metrics) / total,
            "proposal_recall_iou75": sum(m["iou_0.75"]["matched"] for m in metrics) / total,
            "correct_identified_instances_iou50": sum(r["correct_identified_instances_iou50"] for r in entries),
            "matched_detections_iou50": matched,
            "identity_accuracy_given_matched_detection": sum(r["correct_identified_instances_iou50"] for r in entries) / matched if matched else None,
            "segmentation": coco_metrics(gt, predictions[variant])}
        save_json(output / f"{variant}_coco_predictions.json", predictions[variant])
    save_json(output / "coco_ground_truth.json", gt)
    save_json(output / "per_frame.json", per_frame)
    save_json(output / "comparison.json", {"frames": len(plan), "methods": summary,
        "scope": scope,
        "ground_truth_used_in_inference": False, "tuning_performed": False})
    print({v: {"AP": s["segmentation"]["AP"], "AP50": s["segmentation"]["AP50"],
               "correct": s["correct_identified_instances_iou50"], "total": s["eligible_instances"]}
           for v, s in summary.items()}, flush=True)


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prior = read_json(BASE / "protocol.json")
    plan = prior["plan"]
    provenance = {}
    for item in plan:
        name = baseline.frame_folder(item).name
        source = baseline.DATA / "test_primesense" / f"{item['scene_id']:06}"
        paths = [BASE / name / "localization.json", BASE / name / "all_view_scores.json",
                 OUTPUT / "sam3" / name / "response.json", source / "scene_camera.json",
                 source / "rgb" / f"{item['im_id']:06}.png", source / "depth" / f"{item['im_id']:06}.png"]
        paths.extend((BASE / name / "masks").glob("*.png"))
        paths.extend((OUTPUT / "sam3" / name / "masks").glob("*.png"))
        provenance.update({str(p.relative_to(ROOT)): baseline.association.content_hash(p) for p in paths})
    protocol = {"plan": plan, "cad_count": 30, "views_per_cad": 42,
        "variants": {"sam_leading": "SAM3 RGB masks; all valid measured depth inside each mask",
            "depth_leading": "Exact saved original T-LESS depth-localization masks",
            "hybrid": "Pool both candidate sources; identical class-aware mask NMS after CAD scoring"},
        "sam": read_json(OUTPUT / "sam3_workflow.json"), "mask_nms_iou": NMS_IOU,
        "classification": "Best-scoring CAD per region; multiple regions may share one CAD; no forced bijection",
        "scoring": "Existing table_then_patch5; .25 global top5 mean + .25 same-view patch + .5 user surface geometry",
        "table": "Same saved depth-fitted plane for all variants; same 5mm penetration tolerance; no forced table contact",
        "geometry": "42 fixed rotations; 30 translation-only updates; 5mm point sampling; full-resolution mask/depth verification",
        "depth_cache": "Exact saved per-view CLS, patch, and CAD transforms; new table tests, geometry scores, view selection",
        "masked_rgb": "224px white aspect-preserving letterbox, DINOv2-L/14; SAM and depth remain separate alternatives",
        "timing": "Cache-assisted diagnostics only; not a fair end-to-end latency benchmark",
        "scope": "Same existing 20-frame pilot; no full benchmark claim, pose refinement, parameter search, or production edits",
        "provenance_sha256": provenance,
        "source_sha256": {p: baseline.association.content_hash(ROOT / p) for p in
            (*baseline.PROTECTED, "scripts/compare_workbench_framework.py", "scripts/run_wrist_branch_order.py",
             "scripts/run_bop_sam_depth_comparison.py")}}
    path = OUTPUT / "protocol.json"
    if path.exists():
        assert read_json(path) == protocol, "Frozen experiment protocol changed"
    else:
        save_json(path, protocol)
    start = perf_counter()
    if not all((OUTPUT / "matching" / baseline.frame_folder(i).name / "predictions.json").exists() for i in plan):
        print("Encoding saved SAM3 masks with DINOv2-L/14 on CUDA", flush=True)
        observed = encode_sam(plan)
        print("Loading the frozen 30-CAD, 42-view template cache", flush=True)
        engine = baseline.cached_templates()
        print("Scoring all 20 frames with shared CAD scoring", flush=True)
        for item in plan:
            score_frame(item, engine, observed)
    evaluate(plan)
    assert all(baseline.association.content_hash(ROOT / p) == h for p, h in provenance.items())
    assert all(baseline.association.content_hash(ROOT / p) == h for p, h in protocol["source_sha256"].items())
    save_json(OUTPUT / "validation.json", {"all_source_files_unchanged": True,
        "wall_this_invocation_s": perf_counter() - start, "gpu": torch.cuda.get_device_name(),
        "cad_candidate_pairs": sum(len(read_json(OUTPUT / "matching" / baseline.frame_folder(i).name / "pair_scores.json")) for i in plan)})


if __name__ == "__main__":
    main()
