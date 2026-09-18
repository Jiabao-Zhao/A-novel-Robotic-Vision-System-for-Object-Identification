"""Annotation-only diagnostics and the official BOP COCO evaluator.

Import with outputs/cache/bop_toolkit and outputs/cache/bop_eval_deps on
PYTHONPATH. The latter contains BOP's COCO fork, which preserves ignore flags.
"""

import copy
import contextlib
import io

import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_mask(mask):
    encoded = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


def overlap_matrices(predicted, truth):
    intersection = np.array([[np.count_nonzero(p & g) for g in truth] for p in predicted],
                            dtype=float).reshape(len(predicted), len(truth))
    p_area = np.array([p.sum() for p in predicted], dtype=float)[:, None]
    g_area = np.array([g.sum() for g in truth], dtype=float)[None, :]
    iou = intersection / np.maximum(p_area + g_area - intersection, 1)
    return iou, intersection / np.maximum(g_area, 1)


def match_instances(iou, threshold):
    """Maximum-cardinality matching first, summed IoU only breaks ties."""
    if not iou.size:
        return []
    reward = (iou >= threshold) * (min(iou.shape) + 1) + iou
    p, g = linear_sum_assignment(reward, maximize=True)
    return [(int(a), int(b), float(iou[a, b])) for a, b in zip(p, g)
            if iou[a, b] >= threshold]


def localization_metrics(masks, annotations):
    """Class-independent diagnostics; no proposal-confidence score is needed.

    Merge/split flags are diagnostics only: >=10% of a GT visible mask in
    multiple proposals (or multiple GT masks in one proposal). Overlapping
    duplicate proposals can also trigger a split flag; keep that explicit.
    """
    ids = list(masks)
    eligible = [a for a in annotations if not a["ignore"]]
    ignored = [a for a in annotations if a["ignore"]]
    predictions = list(masks.values())
    ground_truth = [mask_utils.decode(a["segmentation"]).astype(bool) for a in eligible]
    ignored_masks = [mask_utils.decode(a["segmentation"]).astype(bool) for a in ignored]
    iou, coverage = overlap_matrices(predictions, ground_truth)
    ignored_iou, _ = overlap_matrices(predictions, ignored_masks)
    result = {"predicted_regions": len(ids), "eligible_instances": len(eligible),
              "ignored_instances": len(ignored), "prediction_ids": ids,
              "gt_annotation_ids": [a["id"] for a in eligible],
              "gt_category_ids": [a["category_id"] for a in eligible], "iou_matrix": iou.tolist(),
              "gt_coverage_matrix": coverage.tolist(),
              "best_iou_per_gt": (iou.max(0) if len(ids) else np.zeros(len(eligible))).tolist(),
              "suspected_merged_regions": [ids[i] for i in np.flatnonzero((coverage >= .1).sum(1) >= 2)],
              "fragmented_or_duplicate_gt_ids": [eligible[i]["id"] for i in np.flatnonzero((coverage >= .1).sum(0) >= 2)]}
    for threshold in (.5, .75):
        matches = match_instances(iou, threshold)
        matched = {p for p, _, _ in matches}
        ignored_predictions = [p for p in range(len(ids)) if p not in matched and
                               ignored_iou.shape[1] and ignored_iou[p].max() >= threshold]
        result[f"iou_{threshold:.2f}"] = {
            "matched": len(matches), "missed": len(eligible) - len(matches),
            "unmatched_predictions": len(ids) - len(matches) - len(ignored_predictions),
            "ignored_predictions": len(ignored_predictions),
            "matches": [{"object_id": ids[p], "gt_id": eligible[g]["id"], "iou": score}
                        for p, g, score in matches]}
    return result


def coco_metrics(annotations, predictions):
    """Call the exact BOP-recommended COCO fork, including empty predictions."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(copy.deepcopy(annotations))
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO({**copy.deepcopy(annotations), "annotations": []})
        evaluation = COCOeval(gt, dt, "segm")
        evaluation.params.imgIds = sorted(gt.getImgIds())
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    names = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    values = {name: float(value) for name, value in zip(names, evaluation.stats)}
    per_object = {}
    for i, cid in enumerate(evaluation.params.catIds):
        precision = evaluation.eval["precision"][:, :, i, 0, -1]
        valid = precision[precision >= 0]
        per_object[str(cid)] = float(valid.mean()) if valid.size else None
    return {**values, "per_object_AP": per_object}
