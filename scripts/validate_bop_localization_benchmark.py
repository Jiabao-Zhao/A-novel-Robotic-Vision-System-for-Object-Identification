"""Export native BOP predictions and independently run its unmodified evaluator."""

import os
import subprocess
import sys

from scripts.run_bop_localization_benchmark import (
    ROOT, CACHE, DATA, OUTPUT, METHODS, PROTECTED, frame_folder,
    read_json, save_json, association,
)
import numpy as np


def main():
    protocol = read_json(OUTPUT / "protocol.json")
    summary = read_json(OUTPUT / "summary.json")
    ground_truth = read_json(OUTPUT / "coco_ground_truth.json")
    source_targets = read_json(DATA / "test_targets_bop19.json")
    selected = {(p["scene_id"], p["im_id"]) for p in protocol["plan"]}
    target_rows = [r for r in source_targets if (r["scene_id"], r["im_id"]) in selected]
    datasets = OUTPUT / "official_evaluation/datasets"
    result_folder = OUTPUT / "official_evaluation/results"
    evaluation_folder = OUTPUT / "official_evaluation/scores"
    (datasets / "tless").mkdir(parents=True, exist_ok=True)
    result_folder.mkdir(parents=True, exist_ok=True)
    save_json(datasets / "tless/test_targets_pilot.json", target_rows)
    predictions = {m: [] for m in METHODS}
    pair_count = view_count = 0
    max_depth_error = 0.
    projection_discrepancies = []
    for ordinal, item in enumerate(protocol["plan"], 1):
        folder = frame_folder(item)
        localization = read_json(folder / "localization.json")
        regions = localization["regions"]
        pairs = read_json(folder / "pair_scores.json")
        views = read_json(folder / "all_view_scores.json")
        assert len(pairs) == len(regions) * 30
        assert len(views) == len(pairs) * 42
        for region in regions:
            assert {p["cad_id"] for p in pairs if p["object_id"] == region["object_id"]} == {
                f"obj_{i:06}" for i in range(1, 31)}
        pair_count += len(pairs)
        view_count += len(views)
        max_depth_error = max(max_depth_error, max((r["max_projected_depth_error_m"] for r in regions), default=0.))
        projection_discrepancies.extend({**item, **r} for r in regions if r["max_projected_depth_error_m"] > 1e-6)
        saved = read_json(folder / "predictions.json")
        for method, rows in saved.items():
            assert len({r["object_id"] for r in rows}) == len(rows)
            for row in rows:
                candidates = sorted((p for p in pairs if p["object_id"] == row["object_id"] and p[method] is not None),
                                    key=lambda p: (-p[method], p["cad_id"]))
                assert row["category_id"] == int(candidates[0]["cad_id"].split("_")[1])
                assert row["score"] == candidates[0][method]
            predictions[method].extend(rows)
        image = next(r for r in ground_truth["images"] if r["id"] == ordinal)
        annotations = [{**r, "image_id": item["im_id"]} for r in ground_truth["annotations"] if r["image_id"] == ordinal]
        scene_gt = {**ground_truth, "images": [{**image, "id": item["im_id"]}], "annotations": annotations}
        scene_path = datasets / "tless/test_primesense" / f"{item['scene_id']:06}"
        scene_path.mkdir(parents=True, exist_ok=True)
        save_json(scene_path / "scene_gt_coco.json", scene_gt)
    assert protocol["protected_sha256"] == {p: association.content_hash(ROOT / p) for p in PROTECTED}
    env = {**os.environ, "BOP_PATH": str(datasets),
           "PYTHONPATH": os.pathsep.join(str(p) for p in (CACHE / "bop_eval_deps", CACHE / "bop_toolkit"))}
    parity = {}
    for method, rows in predictions.items():
        filename = f"{method.replace('_', '-')}_tless-test-primesense.json"
        save_json(result_folder / filename, rows)
        command = [sys.executable, str(CACHE / "bop_toolkit/scripts/eval_bop22_coco.py"),
                   "--result_filenames", filename, "--results_path", str(result_folder),
                   "--eval_path", str(evaluation_folder), "--targets_filename", "test_targets_pilot.json",
                   "--ann_type", "segm"]
        with (result_folder / f"{method}_evaluation.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = read_json(evaluation_folder / filename.removesuffix(".json") / "scores_bop22_coco_segm.json")
        for key in ("AP", "AP50", "AP75", "AR100"):
            np.testing.assert_allclose(actual[key], summary["segmentation"][method][key], atol=1e-12, rtol=0)
        parity[method] = {k: actual[k] for k in ("AP", "AP50", "AP75", "AR100")}
    result = {"frames": len(selected), "cad_candidate_pairs": pair_count, "view_hypotheses": view_count,
        "max_mask_projection_depth_error_m": max_depth_error, "protected_sources_unchanged": True,
        "regions_with_non_raw_projection": projection_discrepancies,
        "projection_limitation": "Existing crop_raw_cluster returns downsampled points when the cleaned raw crop is empty. Confirmed in scene 20, region 2: raw crop 0 points, cleaned fallback 160 points, 2.3mm maximum depth discrepancy. Kept unchanged; no support filling.",
        "unmodified_official_evaluator_parity": parity,
        "scope": "20-frame subset, not the full 1000-image T-LESS benchmark",
        "runtime_note": "Every method's exported time includes shared computation of all six variants, not isolated method timing"}
    result["dataset_archives_sha256"] = {p.name: association.content_hash(p) for p in (CACHE / "bop_tless").glob("*.zip")}
    with np.load(ROOT / "outputs/surface_verification_20260917/stage4_tless/template_features/features.npz") as features:
        np.testing.assert_array_equal(features["patch_indices"], np.arange(1260))
    result["template_patch_index_alignment_verified"] = True
    save_json(OUTPUT / "validation.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
