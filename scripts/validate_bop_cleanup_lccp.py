"""Check mask/score provenance, evaluate native BOP exports, and count recoveries."""

import os
import subprocess
import sys

from scripts.run_bop_cleanup_lccp import (
    ROOT, BASE, OUTPUT, VARIANTS, baseline, read_json, save_json, load_masks, mask_hash,
)
import numpy as np


def object_outcomes(folder):
    overlaps = read_json(folder / "localization_metrics.json")
    frames = read_json(folder / "per_frame.json")
    result = {}
    for overlap, frame in zip(overlaps, frames, strict=True):
        found = {r["gt_id"]: r["object_id"] for r in overlap["iou_0.50"]["matches"]}
        guesses = {m: {p["object_id"]: p for p in rows} for m, rows in frame["predictions"].items()}
        for gid, cad in zip(overlap["gt_annotation_ids"], overlap["gt_category_ids"], strict=True):
            oid = found.get(gid)
            result[gid] = {"scene_id": frame["scene_id"], "im_id": frame["im_id"], "cad_id": cad,
                "matched_object_id": oid,
                "predicted_cad": {m: guesses[m].get(oid, {}).get("category_id") for m in guesses},
                "correct": {m: guesses[m].get(oid, {}).get("category_id") == cad for m in guesses}}
    return result


def validate_variant(variant, plan):
    folder = OUTPUT / variant
    summary = read_json(folder / "summary.json")
    predictions = {m: [] for m in baseline.METHODS}
    counts = {"regions": 0, "pairs": 0, "views": 0, "new_masks": 0, "reused_masks": 0}
    for item in plan:
        frame = folder / baseline.frame_folder(item).name
        masks = load_masks(frame)
        pairs, views = read_json(frame / "pair_scores.json"), read_json(frame / "all_view_scores.json")
        assert len(pairs) == len(masks) * 30
        assert len(views) == len(pairs) * 42
        for oid in masks:
            assert {p["cad_id"] for p in pairs if p["object_id"] == oid} == {f"obj_{i:06}" for i in range(1, 31)}
            for cad in range(1, 31):
                assert {v["view_index_1based"] for v in views if v["object_id"] == oid and v["cad_id"] == f"obj_{cad:06}"} == set(range(1, 43))
        runtime = read_json(frame / "inference_runtime.json")
        counts["new_masks"] += runtime["new_masks"]
        counts["reused_masks"] += len(runtime["reused_masks"])
        for saved in runtime["reused_masks"]:
            source = ROOT / saved["source"]
            oid, old_id = saved["object_id"], saved["source_object_id"]
            assert mask_hash(masks[oid]) == mask_hash(load_masks(source)[old_id])
            expected = [{**r, "object_id": oid} for r in read_json(source / "pair_scores.json") if r["object_id"] == old_id]
            assert expected == [r for r in pairs if r["object_id"] == oid]
        for method, rows in read_json(frame / "predictions.json").items():
            for row in rows:
                ranking = sorted((p for p in pairs if p["object_id"] == row["object_id"] and p[method] is not None),
                                 key=lambda p: (-p[method], p["cad_id"]))
                assert row["category_id"] == int(ranking[0]["cad_id"].split("_")[1])
                assert row["score"] == ranking[0][method]
            predictions[method].extend(rows)
        counts["regions"] += len(masks)
        counts["pairs"] += len(pairs)
        counts["views"] += len(views)
    result_dir = folder / "official_evaluation/results"
    result_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "BOP_PATH": str(BASE / "official_evaluation/datasets"),
        "PYTHONPATH": os.pathsep.join(str(baseline.CACHE / p) for p in ("bop_eval_deps", "bop_toolkit"))}
    parity = {}
    for method, rows in predictions.items():
        filename = f"{method.replace('_', '-')}_tless-test-primesense.json"
        save_json(result_dir / filename, rows)
        with (result_dir / f"{method}.log").open("w") as log:
            subprocess.run([sys.executable, str(baseline.CACHE / "bop_toolkit/scripts/eval_bop22_coco.py"),
                "--result_filenames", filename, "--results_path", str(result_dir),
                "--eval_path", str(folder / "official_evaluation/scores"),
                "--targets_filename", "test_targets_pilot.json", "--ann_type", "segm"],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = read_json(folder / "official_evaluation/scores" / filename.removesuffix(".json") / "scores_bop22_coco_segm.json")
        for key in ("AP", "AP50", "AP75", "AR100"):
            np.testing.assert_allclose(actual[key], summary["segmentation"][method][key], atol=1e-12, rtol=0)
        parity[method] = {k: actual[k] for k in ("AP", "AP50", "AP75", "AR100")}
    return {**counts, "official_evaluator_parity": parity}


def main():
    protocol = read_json(OUTPUT / "protocol.json")
    assert protocol["protected_sha256"] == {p: baseline.association.content_hash(ROOT / p) for p in baseline.PROTECTED}
    assert protocol["experiment_sha256"] == {p: baseline.association.content_hash(ROOT / p) for p in protocol["experiment_sha256"]}
    old = object_outcomes(BASE)
    changes, validation = {}, {}
    for variant in VARIANTS:
        validation[variant] = validate_variant(variant, protocol["plan"])
        new = object_outcomes(OUTPUT / variant)
        assert new.keys() == old.keys()
        rows = [{"gt_id": gid, "before": old[gid], "after": new[gid]} for gid in old]
        changes[variant] = {"recovered_masks": sum(r["before"]["matched_object_id"] is None and r["after"]["matched_object_id"] is not None for r in rows),
            "lost_masks": sum(r["before"]["matched_object_id"] is not None and r["after"]["matched_object_id"] is None for r in rows),
            "classification_changes": {m: {
                "new_correct": sum(not r["before"]["correct"][m] and r["after"]["correct"][m] for r in rows),
                "lost_correct": sum(r["before"]["correct"][m] and not r["after"]["correct"][m] for r in rows)} for m in baseline.METHODS},
            "objects": rows}
    save_json(OUTPUT / "prediction_changes.json", changes)
    save_json(OUTPUT / "validation.json", {"variants": validation, "protected_sources_unchanged": True,
        "frozen_experiment_sources_unchanged": True, "test_frames": len(protocol["plan"])})
    print({v: {k: value for k, value in r.items() if k != "objects"} for v, r in changes.items()}, flush=True)


if __name__ == "__main__":
    main()
