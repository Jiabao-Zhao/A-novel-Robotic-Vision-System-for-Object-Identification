"""Check benchmark semantics, especially mistakes a high score must not hide."""

import unittest

import numpy as np

from scripts.bop_segmentation_metrics import (
    coco_metrics, encode_mask, localization_metrics, match_instances,
)


def rectangle(x1=2, y1=2, x2=12, y2=12):
    mask = np.zeros((32, 32), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def annotation(index, mask, category=1, ignore=False):
    y, x = np.nonzero(mask)
    return {"id": index, "image_id": 1, "category_id": category,
            "segmentation": encode_mask(mask), "area": int(mask.sum()), "iscrowd": 0,
            "ignore": ignore, "bbox": [int(x.min()), int(y.min()), int(np.ptp(x) + 1), int(np.ptp(y) + 1)]}


def dataset(annotations):
    return {"info": {}, "images": [{"id": 1, "width": 32, "height": 32}],
            "categories": [{"id": 1, "name": "1"}, {"id": 2, "name": "2"}],
            "annotations": annotations}


def prediction(mask, score=.9, category=1):
    return {"image_id": 1, "category_id": category, "score": score, "segmentation": encode_mask(mask)}


class BopSegmentationMetricsTests(unittest.TestCase):
    def test_exact_mask_and_identity_get_full_ap(self):
        mask = rectangle()
        scores = coco_metrics(dataset([annotation(1, mask)]), [prediction(mask)])
        self.assertAlmostEqual(scores["AP"], 1.)

    def test_exact_mask_wrong_identity_gets_no_credit(self):
        mask = rectangle()
        scores = coco_metrics(dataset([annotation(1, mask)]), [prediction(mask, category=2)])
        self.assertEqual(scores["AP"], 0.)

    def test_empty_predictions_are_zero_not_missing_metric(self):
        scores = coco_metrics(dataset([annotation(1, rectangle())]), [])
        self.assertEqual(scores["AP"], 0.)
        self.assertEqual(scores["AR100"], 0.)

    def test_bop_ignore_flag_does_not_turn_into_false_positive(self):
        good, ignored = rectangle(), rectangle(19, 19, 29, 29)
        scores = coco_metrics(dataset([annotation(1, good), annotation(2, ignored, ignore=True)]),
                              [prediction(ignored, .99), prediction(good, .8)])
        self.assertAlmostEqual(scores["AP"], 1.)

    def test_confident_wrong_region_lowers_ap(self):
        good, wrong = rectangle(), rectangle(19, 19, 29, 29)
        scores = coco_metrics(dataset([annotation(1, good)]),
                              [prediction(wrong, .99), prediction(good, .8)])
        self.assertAlmostEqual(scores["AP"], .5)

    def test_missing_image_cannot_disappear_from_recall(self):
        mask = rectangle()
        data = dataset([annotation(1, mask)])
        data["images"].append({"id": 2, "width": 32, "height": 32})
        data["annotations"].append({**annotation(2, mask), "image_id": 2})
        scores = coco_metrics(data, [prediction(mask)])
        self.assertAlmostEqual(scores["AR100"], .5)

    def test_maximum_cardinality_precedes_total_iou(self):
        matrix = np.array([[1., .51], [.51, .49]])
        pairs = match_instances(matrix, .5)
        self.assertEqual({(p, g) for p, g, _ in pairs}, {(0, 1), (1, 0)})

    def test_duplicate_proposals_do_not_recall_two_objects(self):
        a, b = rectangle(), rectangle(19, 19, 29, 29)
        result = localization_metrics({"a": a, "duplicate": a}, [annotation(1, a), annotation(2, b)])
        self.assertEqual(result["iou_0.50"]["matched"], 1)
        self.assertEqual(result["iou_0.50"]["missed"], 1)
        self.assertEqual(result["iou_0.50"]["unmatched_predictions"], 1)
        self.assertEqual(result["fragmented_or_duplicate_gt_ids"], [1])

    def test_merge_and_empty_localization_are_counted(self):
        a, b = rectangle(), rectangle(19, 19, 29, 29)
        annotations = [annotation(1, a), annotation(2, b)]
        merged = localization_metrics({"merged": a | b}, annotations)
        self.assertEqual(merged["suspected_merged_regions"], ["merged"])
        self.assertEqual(merged["iou_0.50"]["matched"], 1)
        empty = localization_metrics({}, annotations)
        self.assertEqual(empty["iou_0.50"]["missed"], 2)
        self.assertEqual(empty["best_iou_per_gt"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
