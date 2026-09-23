"""Correct missed hybrid CAD identities using GT, keeping all other outputs fixed."""

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.bop_segmentation_metrics import coco_metrics, match_instances
from scripts.inspect_bop_mask_cad_oracles import SOURCE, read_json


def main():
    start = perf_counter()
    paths = [SOURCE / name for name in (
        "coco_ground_truth.json", "per_frame.json", "hybrid_coco_predictions.json",
        "evaluator_validation.json", "comparison.json")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    truth, frames, original, validated, comparison = map(read_json, paths)
    corrected = copy.deepcopy(original)
    changes = []
    before_count = after_count = 0
    for ordinal, frame in enumerate(frames, 1):
        metrics = frame["methods"]["hybrid"]["proposal_metrics"]
        indices = [i for i, row in enumerate(original) if row["image_id"] == ordinal]
        categories = np.array(metrics["gt_category_ids"])
        ids = metrics["prediction_ids"]
        iou = np.array([metrics["iou_matrix"][ids.index(original[i]["object_id"])]
                        for i in indices]).reshape(len(indices), len(categories))
        labels = np.array([original[i]["category_id"] for i in indices])
        existing = match_instances(iou * (labels[:, None] == categories), .5)
        before_count += len(existing)
        used_predictions = {p for p, _, _ in existing}
        used_truth = {g for _, g, _ in existing}
        free_predictions = [p for p in range(len(indices)) if p not in used_predictions]
        missed_truth = [g for g in range(len(categories)) if g not in used_truth]
        for p, g, overlap in match_instances(iou[np.ix_(free_predictions, missed_truth)], .5):
            p, g = free_predictions[p], missed_truth[g]
            index = indices[p]
            assert original[index]["category_id"] != int(categories[g])
            corrected[index]["category_id"] = int(categories[g])
            changes.append({"prediction_index": index, "scene_id": frame["scene_id"],
                "im_id": frame["im_id"], "object_id": original[index]["object_id"],
                "gt_annotation_id": metrics["gt_annotation_ids"][g], "iou": overlap,
                "old_cad_id": original[index]["category_id"], "new_cad_id": int(categories[g]),
                "unchanged_confidence": original[index]["score"]})
        labels = np.array([corrected[i]["category_id"] for i in indices])
        after_count += len(match_instances(iou * (labels[:, None] == categories), .5))

    assert before_count == validated["identified_instances_class_aware_matching"]["hybrid"]["iou50"] == 96
    assert len(changes) == 40 and after_count == 136
    for old, new in zip(original, corrected):
        assert {k: v for k, v in old.items() if k != "category_id"} == {
            k: v for k, v in new.items() if k != "category_id"}
    baseline = coco_metrics(truth, original)
    for key in ("AP", "AP50", "AP75", "AR100"):
        np.testing.assert_allclose(baseline[key], comparison["methods"]["hybrid"]["segmentation"][key],
                                   atol=1e-12, rtol=0)
    result = {"scope": "GT-assisted counterfactual, not inference performance",
        "definition": "Preserve the 96 class-aware matched detections at IoU >= 0.5. Match remaining saved final detections to missed GT instances by maximum cardinality, then summed IoU. Change only these 40 category IDs. Keep all 602 detections, masks, confidence values and ordering; do not rerun NMS or inference. This is one specified correction scenario, not an AP ceiling.",
        "detections": len(original), "identified_before": before_count, "identified_after": after_count,
        "baseline": baseline, "correct_40_identities": coco_metrics(truth, corrected),
        "changes": changes, "input_sha256": hashes,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inference_rerun": False, "runtime_s": perf_counter() - start}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in paths)
    (SOURCE / "identity_correction_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ("detections", "identified_before", "identified_after", "baseline",
                                  "correct_40_identities", "runtime_s")}, flush=True)


if __name__ == "__main__":
    main()
