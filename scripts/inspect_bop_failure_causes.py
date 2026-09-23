"""Saved-only audit of 40 missed identities and remaining hybrid mask AP losses."""

import contextlib
import copy
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts.bop_segmentation_metrics import coco_metrics
from scripts.inspect_bop_mask_cad_oracles import SOURCE, ROOT


HASHES = {}


def read(path):
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def distribution(values):
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist())) if values else None


def table_at_ground_truth(change, frame, annotations, correct_views, meshes):
    """Compare the same plane test at saved fits and the annotated CAD pose."""
    data = ROOT / "outputs/cache/bop_tless/data/tless"
    scene = data / "test_primesense" / f"{change['scene_id']:06}"
    poses = read(scene / "scene_gt.json")[str(change["im_id"])]
    info = read(scene / "scene_gt_info.json")[str(change["im_id"])]
    poses = [p for p, i in zip(poses, info) if i["px_count_visib"] > 0]
    image_id = annotations[change["gt_annotation_id"]]["image_id"]
    image_annotations = [a for a in annotations.values() if a["image_id"] == image_id]
    assert len(poses) == len(image_annotations)
    assert [p["obj_id"] for p in poses] == [a["category_id"] for a in image_annotations]
    pose = poses[next(i for i, a in enumerate(image_annotations) if a["id"] == change["gt_annotation_id"])]
    name = f"scene_{frame['scene_id']:06}_{frame['im_id']:06}"
    plane = np.array(read(ROOT / "outputs/bop_localization_20260917" / name / "localization.json")["plane_model"])
    plane /= np.linalg.norm(plane[:3])
    if plane[3] < 0:
        plane *= -1
    cid = change["new_cad_id"]
    if cid not in meshes:
        path = data / "models_cad" / f"obj_{cid:06}.ply"
        HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        meshes[cid] = np.asarray(o3d.io.read_triangle_mesh(str(path)).vertices) * .001
    vertices = meshes[cid]
    R, t = np.array(pose["cam_R_m2c"]).reshape(3, 3), np.array(pose["cam_t_m2c"]) * .001
    gt_clearance = float(((vertices @ R.T + t) @ plane[:3] + plane[3]).min())
    clearances = []
    for view in correct_views:
        T = np.array(view["camera_T_cad"])
        clearances.append(float(((vertices @ T[:3, :3].T + T[:3, 3]) @ plane[:3] + plane[3]).min()))
    assert max(clearances) < -.005
    return {"ground_truth_min_clearance_mm": gt_clearance * 1000,
            "ground_truth_passes_same_5mm_table_test": gt_clearance >= -.005,
            "best_saved_pose_min_clearance_mm": max(clearances) * 1000}


def detection_counts(truth, predictions, frames):
    """Use official greedy, confidence-ordered matches, including ignored GT."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(copy.deepcopy(truth))
        dt = gt.loadRes(copy.deepcopy(predictions))
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.evaluate()
    output = {}
    for threshold in (.5, .75, .9, .95):
        t = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, threshold))[0])
        counts, confidence = Counter(), defaultdict(list)
        for item in evaluator.evalImgs:
            if item is None or item["aRng"] != evaluator.params.areaRng[0]:
                continue
            for d, did in enumerate(item["dtIds"]):
                row = dt.anns[did]
                if item["dtIgnore"][t, d]:
                    kind = "ignored"
                elif item["dtMatches"][t, d]:
                    kind = "true_positive"
                else:
                    metrics = frames[row["image_id"] - 1]["methods"]["hybrid"]["proposal_metrics"]
                    overlaps = np.array(metrics["iou_matrix"][metrics["prediction_ids"].index(row["object_id"])])
                    categories = np.array(metrics["gt_category_ids"])
                    if np.any((overlaps >= threshold) & (categories == row["category_id"])):
                        kind = "duplicate_correct_label"
                    elif np.any(overlaps >= threshold):
                        kind = "wrong_label_on_matching_mask"
                    else:
                        kind = "no_eligible_mask_match"
                counts[kind] += 1
                confidence[kind].append(row["score"])
        assert sum(counts.values()) == len(predictions)
        output[str(threshold)] = {"counts": dict(counts),
            "confidence": {k: distribution(v) for k, v in confidence.items()}}
    return output


def main():
    start = perf_counter()
    truth = read(SOURCE / "coco_ground_truth.json")
    annotations = {a["id"]: a for a in truth["annotations"]}
    frames = read(SOURCE / "per_frame.json")
    original = read(SOURCE / "hybrid_coco_predictions.json")
    identity = read(SOURCE / "identity_correction_diagnostic.json")
    prior = read(SOURCE / "mask_cad_oracle_diagnostic.json")
    corrected = copy.deepcopy(original)
    kept = list(prior["methods"]["hybrid"]["retained_detection_matches"])
    for change in identity["changes"]:
        row = corrected[change["prediction_index"]]
        assert row["category_id"] == change["old_cad_id"]
        row["category_id"] = change["new_cad_id"]
        kept.append({"image_id": row["image_id"], "object_id": row["object_id"],
                     "gt_id": change["gt_annotation_id"], "iou": change["iou"]})
    by_key = {(r["image_id"], r["object_id"]): r for r in corrected}
    selected = [by_key[r["image_id"], r["object_id"]] for r in kept]
    assert len(kept) == len({r["gt_id"] for r in kept}) == 136
    perfect_masks = [{**row, "segmentation": annotations[match["gt_id"]]["segmentation"]}
                     for row, match in zip(selected, kept)]
    missing = [a for a in truth["annotations"] if not a["ignore"] and a["id"] not in {m["gt_id"] for m in kept}]
    complete = perfect_masks + [{"image_id": a["image_id"], "category_id": a["category_id"],
                                "segmentation": a["segmentation"], "score": 1.} for a in missing]
    stages = {"original": original, "correct_40_labels": corrected,
              "also_remove_extra_detections": selected, "also_make_136_masks_exact": perfect_masks,
              "also_recover_2_missed_instances": complete}
    stage_scores = {k: {"detections": len(rows), **coco_metrics(truth, rows)} for k, rows in stages.items()}
    for name, expected in (("original", identity["baseline"]),
                           ("correct_40_labels", identity["correct_40_identities"])):
        np.testing.assert_allclose(stage_scores[name]["AP"], expected["AP"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(stage_scores["also_recover_2_missed_instances"]["AP"], 1., atol=1e-12)

    cases, frame_pairs, meshes = [], {}, {}
    for change in identity["changes"]:
        row = original[change["prediction_index"]]
        frame = frames[row["image_id"] - 1]
        name = f"scene_{change['scene_id']:06}_{change['im_id']:06}"
        if name not in frame_pairs:
            frame_pairs[name] = {(p["object_id"], p["cad_id"]): p for p in
                read(SOURCE / "matching" / name / "pair_scores.json")}
        pairs = frame_pairs[name]
        oid, cid = change["object_id"], f"obj_{change['new_cad_id']:06}"
        correct = pairs[oid, cid]
        wrong = pairs[oid, f"obj_{change['old_cad_id']:06}"]
        assert wrong["score"] == row["score"]
        region = SOURCE / "matching" / name / "regions" / f"{oid}.json"
        if region.exists():
            views = read(region)["views"]
        else:
            views = [v for v in read(ROOT / "outputs/bop_sam_depth_20260919/matching" / name / "view_scores.json")
                     if v["object_id"] == oid]
        correct_views = [v for v in views if v["cad_id"] == cid]
        assert len(correct_views) == 42
        record = {**change, "correct_pair": correct, "wrong_pair": wrong,
                  "correct_view_statuses": dict(Counter(v["pose_status"] for v in correct_views))}
        if correct["score"] is None:
            record["failure"] = "correct_CAD_unavailable"
            record["table_pose_diagnostic"] = table_at_ground_truth(change, frame, annotations, correct_views, meshes)
        else:
            record["failure"] = "correct_CAD_outscored"
            record["wrong_minus_correct_score"] = wrong["score"] - correct["score"]
            record["component_deltas"] = {b: wrong["winning_components"][b] - correct["winning_components"][b]
                                           for b in ("G", "P", "H")}
            delta = record["component_deltas"]
            visual, geometry = .25 * (delta["G"] + delta["P"]), .5 * delta["H"]
            record["weighted_visual_delta"], record["weighted_geometry_delta"] = visual, geometry
            record["winner_driver"] = ("both_visual_and_geometry_favor_wrong" if visual > 0 and geometry > 0
                else "visual_overrules_correct_geometry" if visual > 0 else "geometry_overrules_correct_visual")
            ranked = sorted((p for (o, _), p in pairs.items() if o == oid and p["score"] is not None),
                            key=lambda p: (-p["score"], p["cad_id"]))
            record["correct_rank"] = next(i + 1 for i, p in enumerate(ranked) if p["cad_id"] == cid)
        record["winning_geometry_diagnostics"] = {}
        for label, pair in (("correct", correct), ("wrong", wrong)):
            winning = next((v for v in views if v["cad_id"] == pair["cad_id"] and
                            v["view_index_1based"] == pair["winning_view"]), None)
            if winning:
                record["winning_geometry_diagnostics"][label] = {k: winning.get(k) for k in (
                    "p_obs", "p_cad", "p_depth", "intersection_pixels", "union_pixels",
                    "observed_pixels", "assessable_rendered_pixels")}
        metrics = frame["methods"]["hybrid"]["proposal_metrics"]
        g = metrics["gt_annotation_ids"].index(change["gt_annotation_id"])
        supports = [o for o, overlaps in zip(metrics["prediction_ids"], metrics["iou_matrix"]) if overlaps[g] >= .5]
        available = [o for o in supports if pairs[o, cid]["score"] is not None]
        top_correct = [o for o in available if min((p for (p_oid, _), p in pairs.items()
            if p_oid == o and p["score"] is not None), key=lambda p: (-p["score"], p["cad_id"]))["cad_id"] == cid]
        record["all_supporting_proposals"] = {"count": len(supports), "correct_CAD_available": len(available),
            "correct_CAD_top1_before_NMS": top_correct}
        cases.append(record)

    outscored = [c for c in cases if c["failure"] == "correct_CAD_outscored"]
    summary = {"failures": dict(Counter(c["failure"] for c in cases)),
        "rejected_cases_GT_pose_passes_table": sum(c.get("table_pose_diagnostic", {}).get(
            "ground_truth_passes_same_5mm_table_test", False) for c in cases),
        "unavailable_view_statuses": dict(sum((Counter(c["correct_view_statuses"]) for c in cases
                                             if c["failure"] == "correct_CAD_unavailable"), Counter())),
        "outscored_driver": dict(Counter(c["winner_driver"] for c in outscored)),
        "outscored_margin": distribution([c["wrong_minus_correct_score"] for c in outscored]),
        "mask_iou_40": distribution([c["iou"] for c in cases]),
        "mask_iou_40_ge_075": sum(c["iou"] >= .75 for c in cases),
        "all_supports_correct_CAD_unavailable": sum(c["all_supporting_proposals"]["correct_CAD_available"] == 0 for c in cases),
        "any_correct_top1_before_NMS": sum(bool(c["all_supporting_proposals"]["correct_CAD_top1_before_NMS"]) for c in cases),
        "frequent_confusions": [{"true": a, "predicted": b, "count": n} for (a, b), n in
                                Counter((c["new_cad_id"], c["old_cad_id"]) for c in cases).most_common()],
        "selected_136_mask_iou": distribution([r["iou"] for r in kept]),
        "selected_136_mask_pass_counts": {str(t): sum(r["iou"] >= t for r in kept) for t in (.5, .75, .8, .85, .9, .95)}}
    result = {"scope": "GT-assisted saved-output diagnostics; no inference or production changes",
        "stage_definitions": "Sequential interventions: correct 40 labels, remove all detections except the same 96 previously matched and 40 corrected ones, replace those 136 masks with exact GT, add the two missing GT instances. Existing confidence values are unchanged. Effects depend on intervention order and are not independent causal percentages.",
        "stages": stage_scores, "summary": summary, "cases": cases, "selected_matches": kept,
        "original_detection_counts": detection_counts(truth, original, frames),
        "corrected_detection_counts": detection_counts(truth, corrected, frames),
        "input_sha256": HASHES, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_s": perf_counter() - start}
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h for path, h in HASHES.items())
    (SOURCE / "failure_cause_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stages": {k: {f: r[f] for f in ("detections", "AP", "AP50", "AP75")} for k, r in stage_scores.items()},
        "summary": summary, "original_counts": result["original_detection_counts"]["0.5"],
        "corrected_counts": result["corrected_detection_counts"]["0.5"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
