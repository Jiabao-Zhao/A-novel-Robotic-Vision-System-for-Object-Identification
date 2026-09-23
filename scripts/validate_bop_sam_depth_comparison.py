"""Check saved comparison coverage, cache parity, and official evaluator parity."""

import os
import subprocess
import sys

from scripts.run_bop_sam_depth_comparison import (
    baseline, BASE, OUTPUT, VARIANTS, read_json, save_json,
)
import numpy as np
from scripts.bop_segmentation_metrics import match_instances


def main(output=OUTPUT):
    protocol = read_json(output / "protocol.json")
    comparison = read_json(output / "comparison.json")
    predictions = {v: [] for v in VARIANTS}
    pair_count = view_count = 0
    no_sam_frames = 0
    for item in protocol["plan"]:
        name = baseline.frame_folder(item).name
        folder = output / "matching" / name
        pairs = read_json(folder / "pair_scores.json")
        views = read_json(folder / "view_scores.json")
        saved = read_json(folder / "predictions.json")
        objects = {r["object_id"] for r in pairs}
        assert len(pairs) == len(objects) * 30
        assert len(views) == len(pairs) * 42
        for oid in objects:
            assert {r["cad_id"] for r in pairs if r["object_id"] == oid} == {
                f"obj_{i:06}" for i in range(1, 31)}
        old = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r
               for r in read_json(BASE / name / "all_view_scores.json")}
        for row in views:
            if row["object_id"].startswith("depth_"):
                previous = old[row["cad_id"], row["object_id"].replace("depth_", "object_"), row["view_index_1based"]]
                assert row["camera_T_cad"] == previous["camera_T_cad"]
                assert (row["G"], row["P"], row["cls"]) == (previous["global"], previous["patch"], previous["cls_cosine"])
        for variant, rows in saved.items():
            for row in rows:
                candidates = [p for p in pairs if p["object_id"] == row["object_id"] and p["score"] is not None]
                winner = min(candidates, key=lambda p: (-p["score"], p["cad_id"]))
                assert row["score"] == winner["score"]
                assert row["category_id"] == int(winner["cad_id"].split("_")[1])
            # Cache-assisted timings cannot stand in for end-to-end latency.
            predictions[variant].extend([{**r, "time": -1.} for r in rows])
        if not any(oid.startswith("sam_") for oid in objects):
            assert saved["sam_leading"] == []
            assert saved["hybrid"] == saved["depth_leading"]
            no_sam_frames += 1
        pair_count += len(pairs)
        view_count += len(views)

    # Reuse the independently exported annotations for the identical frozen frames.
    assert read_json(output / "coco_ground_truth.json") == read_json(BASE / "coco_ground_truth.json")
    datasets = BASE / "official_evaluation/datasets"
    destination = output / "official_evaluation"
    results, scores = destination / "results", destination / "scores"
    results.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "BOP_PATH": str(datasets),
           "PYTHONPATH": os.pathsep.join(str(baseline.CACHE / p) for p in ("bop_eval_deps", "bop_toolkit"))}
    parity = {}
    for variant, rows in predictions.items():
        filename = f"{variant.replace('_', '-')}_tless-test-primesense.json"
        save_json(results / filename, rows)
        command = [sys.executable, str(baseline.CACHE / "bop_toolkit/scripts/eval_bop22_coco.py"),
                   "--result_filenames", filename, "--results_path", str(results), "--eval_path", str(scores),
                   "--targets_filename", "test_targets_pilot.json", "--ann_type", "segm"]
        with (results / f"{variant}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        official = read_json(scores / filename.removesuffix(".json") / "scores_bop22_coco_segm.json")
        for key in ("AP", "AP50", "AP75", "AR100"):
            np.testing.assert_allclose(official[key], comparison["methods"][variant]["segmentation"][key], atol=1e-12, rtol=0)
        parity[variant] = {k: official[k] for k in ("AP", "AP50", "AP75", "AR100")}
    validation = {"cad_candidate_pairs": pair_count, "view_hypotheses": view_count,
                  "frames_without_sam_preserved": no_sam_frames,
                  "depth_descriptors_and_transforms_exactly_reused": True,
                  "unmodified_official_evaluator_parity": parity}
    identified = {v: {"iou50": 0, "iou75": 0} for v in VARIANTS}
    for frame in read_json(output / "per_frame.json"):
        for variant, result in frame["methods"].items():
            metrics = result["proposal_metrics"]
            detections = result["detections"]
            ids, categories = metrics["prediction_ids"], metrics["gt_category_ids"]
            # A wrong-label duplicate must not steal a GT assignment from a
            # correctly labelled detection. AP separately penalizes false positives.
            iou = np.array([[metrics["iou_matrix"][ids.index(d["object_id"])][g]
                             if d["category_id"] == cid else 0.
                             for g, cid in enumerate(categories)] for d in detections]).reshape(len(detections), len(categories))
            for key, threshold in (("iou50", .5), ("iou75", .75)):
                identified[variant][key] += len(match_instances(iou, threshold))
    validation["identified_instances_class_aware_matching"] = identified
    save_json(output / "evaluator_validation.json", validation)
    print(validation, flush=True)


if __name__ == "__main__":
    main()
