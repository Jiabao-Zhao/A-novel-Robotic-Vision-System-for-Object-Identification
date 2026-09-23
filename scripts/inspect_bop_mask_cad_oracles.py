"""Ground-truth-assisted AP diagnostics on saved predictions; no inference."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from scripts.bop_segmentation_metrics import coco_metrics, encode_mask, match_instances


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/bop_automatic_sam_20260919"
DEPTH = ROOT / "outputs/bop_localization_20260917"
VARIANTS = ("sam_leading", "depth_leading", "hybrid")


def read_json(path):
    return json.loads(path.read_text())


def main():
    start = perf_counter()
    paths = [SOURCE / name for name in (
        "coco_ground_truth.json", "per_frame.json", "comparison.json",
        "evaluator_validation.json", *(f"{v}_coco_predictions.json" for v in VARIANTS))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    truth = read_json(paths[0])
    frames = read_json(paths[1])
    baseline = read_json(paths[2])["methods"]
    validated = read_json(paths[3])["identified_instances_class_aware_matching"]
    eligible = [a for a in truth["annotations"] if not a["ignore"]]
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"],
                "segmentation": a["segmentation"], "score": 1.} for a in eligible]
    ideal_metrics = coco_metrics(truth, perfect)
    np.testing.assert_allclose([ideal_metrics[k] for k in ("AP", "AP50", "AP75")], 1., atol=1e-12)
    result = {"scope": "Ground-truth-assisted diagnostic, not measured inference performance",
        "frames": len(frames), "eligible_instances": len(eligible),
        "definitions": {
            "keep_existing_correct": "Class-aware maximum-cardinality matching at IoU >= 0.5, summed IoU breaks ties; retain one saved detection per matched GT and its original confidence; remove all other detections",
            "oracle_existing_masks": "Use the saved class-agnostic one-to-one proposal matches at IoU >= 0.5; give each its GT CAD identity and IoU as oracle confidence; remove other proposals. Masks themselves are unchanged. This also bypasses pose/table rejection and is not an identity-only intervention or a proven optimal AP ceiling",
            "perfect_output": "Exactly one ground-truth mask and correct CAD per eligible target, confidence 1; no false detections"},
        "perfect_output": {"detections": len(perfect), **ideal_metrics}, "methods": {}}
    for variant in VARIANTS:
        predictions = read_json(SOURCE / f"{variant}_coco_predictions.json")
        reproduced = coco_metrics(truth, predictions)
        for key in ("AP", "AP50", "AP75", "AR100"):
            np.testing.assert_allclose(reproduced[key], baseline[variant]["segmentation"][key], atol=1e-12, rtol=0)
        retained, oracle, selections = [], [], []
        for ordinal, frame in enumerate(frames, 1):
            name = f"scene_{frame['scene_id']:06}_{frame['im_id']:06}"
            metrics = frame["methods"][variant]["proposal_metrics"]
            ids, gt_ids = metrics["prediction_ids"], metrics["gt_annotation_ids"]
            categories = metrics["gt_category_ids"]
            rows = [r for r in predictions if r["image_id"] == ordinal]
            iou = np.array([[metrics["iou_matrix"][ids.index(r["object_id"])][g]
                             if r["category_id"] == cid else 0.
                             for g, cid in enumerate(categories)] for r in rows]).reshape(len(rows), len(categories))
            for p, g, overlap in match_instances(iou, .5):
                retained.append(rows[p])
                selections.append({"image_id": ordinal, "object_id": rows[p]["object_id"],
                                   "gt_id": gt_ids[g], "iou": overlap})
            for match in metrics["iou_0.50"]["matches"]:
                oid = match["object_id"]
                path = (DEPTH / name / "masks" / f"{oid.replace('depth_', 'object_')}.png"
                        if oid.startswith("depth_") else SOURCE / "sam" / name / "masks" / f"{oid}.png")
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                mask = np.asarray(Image.open(path)) > 0
                oracle.append({"image_id": ordinal, "category_id": categories[gt_ids.index(match["gt_id"])],
                    "object_id": oid, "segmentation": encode_mask(mask), "score": match["iou"]})
        assert len(retained) == validated[variant]["iou50"]
        result["methods"][variant] = {
            "baseline": {"detections": len(predictions), **reproduced},
            "keep_existing_correct": {"detections": len(retained), **coco_metrics(truth, retained)},
            "oracle_existing_masks": {"detections": len(oracle), **coco_metrics(truth, oracle)},
            "retained_detection_matches": selections}
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result.update(input_sha256=hashes, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  inference_rerun=False, runtime_s=perf_counter() - start)
    (SOURCE / "mask_cad_oracle_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print({v: {name: round(data["AP"] * 100, 4) for name, data in rows.items()
               if name != "retained_detection_matches"} for v, rows in result["methods"].items()}, flush=True)
    print("Perfect output AP:", ideal_metrics["AP"] * 100, flush=True)


if __name__ == "__main__":
    main()
