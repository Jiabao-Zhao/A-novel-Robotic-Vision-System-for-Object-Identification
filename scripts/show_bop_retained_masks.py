"""Compare three retained diagnostic masks with their exact BOP references."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_utils

from scripts.show_bop_mask_quality import ROOT, SOURCE, read_json


# All three belong to the unchanged set of 136 masks in the AP counterfactual.
EXAMPLES = ((16, "sam_022", 94), (6, "sam_002", 24), (17, "sam_036", 102))


def main():
    diagnostic = read_json(SOURCE / "failure_cause_diagnostic.json")
    selected = {(r["image_id"], r["object_id"], r["gt_id"]): r for r in diagnostic["selected_matches"]}
    frames = read_json(SOURCE / "per_frame.json")
    annotations = {r["id"]: r for r in read_json(SOURCE / "coco_ground_truth.json")["annotations"]}
    predictions = {(r["image_id"], r["object_id"]): r for r in read_json(SOURCE / "hybrid_coco_predictions.json")}
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    heading, text, small = [ImageFont.truetype(font_path, size) for size in (25, 20, 17)]
    panel = Image.new("RGB", (1440, 1085), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    draw.text((20, 12), "Retained masks versus the exact reference used by the evaluator", font=heading, fill="#172531")
    for x, color, label in ((20, "#29b370", "Agreement"), (300, "#ed5353", "Extra predicted pixels"),
                            (720, "#38a6ed", "Missing object pixels")):
        draw.rectangle((x, 52, x + 17, 69), fill=color)
        draw.text((x + 27, 49), label, font=text, fill="#172531")
    for x, label in zip((20, 375, 730, 1085), ("Original RGB crop", "Our saved mask", "BOP reference mask", "Pixel differences")):
        draw.text((x, 87), label, font=text, fill="#172531")
    results = []
    for index, key in enumerate(EXAMPLES):
        match = selected[key]
        image_id, oid, gt_id = key
        frame, annotation = frames[image_id - 1], annotations[gt_id]
        scene, image = frame["scene_id"], frame["im_id"]
        name = f"scene_{scene:06}_{image:06}"
        path = SOURCE / "sam" / name / "masks" / f"{oid}.png"
        predicted = np.asarray(Image.open(path)) > 0
        gt = mask_utils.decode(annotation["segmentation"]).astype(bool)
        np.testing.assert_array_equal(predicted, mask_utils.decode(predictions[image_id, oid]["segmentation"]).astype(bool))
        assert int(gt.sum()) == annotation["area"]
        rgb = np.asarray(Image.open(ROOT / "outputs/cache/bop_tless/data/tless/test_primesense" /
                                   f"{scene:06}/rgb/{image:06}.png").convert("RGB"))
        agree, extra, missing = predicted & gt, predicted & ~gt, gt & ~predicted
        iou = float(agree.sum() / (predicted | gt).sum())
        np.testing.assert_allclose(iou, match["iou"], atol=1e-12, rtol=0)
        difference = np.full_like(rgb, (28, 34, 42))
        difference[agree], difference[extra], difference[missing] = (41, 179, 112), (237, 83, 83), (56, 166, 237)
        y, x = np.nonzero(predicted | gt)
        padding = max(7, int(.08 * max(x.max() - x.min(), y.max() - y.min())))
        bounds = (max(0, x.min() - padding), max(0, y.min() - padding),
                  min(rgb.shape[1], x.max() + padding + 1), min(rgb.shape[0], y.max() + padding + 1))
        top = 121 + index * 310
        draw.text((20, top), f"{index + 1}. Scene {scene}, frame {image} | {oid} | reference CAD {annotation['category_id']} | IoU {iou:.3f}",
                  font=text, fill="#172531")
        pixels = (rgb, np.repeat(predicted[..., None], 3, axis=2).astype(np.uint8) * 255,
                  np.repeat(gt[..., None], 3, axis=2).astype(np.uint8) * 255, difference)
        for left, pixel in zip((20, 375, 730, 1085), pixels):
            tile = ImageOps.contain(Image.fromarray(pixel).crop(bounds), (335, 230), Image.Resampling.NEAREST)
            panel.paste(tile, (left + (335 - tile.width) // 2, top + 34 + (230 - tile.height) // 2))
        passes = [t for t in (.5, .55, .6, .65, .7, .75, .8, .85, .9, .95) if iou >= t]
        draw.text((20, top + 270), f"Extra pixels: {extra.sum():,}   |   Missing pixels: {missing.sum():,}   |   "
                  f"Passes {len(passes)}/10 IoU thresholds (0.50 through {max(passes):.2f})", font=small, fill="#405263")
        results.append({"scene_id": scene, "im_id": image, "object_id": oid, "gt_id": gt_id,
                        "reference_cad_id": annotation["category_id"], "iou": iou,
                        "predicted_pixels": int(predicted.sum()), "reference_pixels": int(gt.sum()),
                        "extra_pixels": int(extra.sum()), "missing_pixels": int(missing.sum()),
                        "reference_coverage": float(agree.sum() / gt.sum()),
                        "crop_xyxy_exclusive": [int(v) for v in bounds], "passed_iou_thresholds": passes})
    draw.text((20, 1058), "Actual saved original-SAM masks from the retained 136. Identical crops and scale within each row; masks are unchanged.",
              font=small, fill="#405263")
    panel.save(SOURCE / "retained_masks_vs_reference.png")
    metadata = {"scope": "Three illustrative members of the fixed 136-mask diagnostic; reference masks decoded directly from the evaluator annotations",
                "inference_rerun": False, "examples": results}
    (SOURCE / "retained_masks_vs_reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
