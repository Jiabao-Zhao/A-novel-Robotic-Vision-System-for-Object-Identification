"""Display four actual saved SAM masks spanning the requested overlap levels."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_utils

from scripts.inspect_bop_mask_cad_oracles import ROOT, SOURCE, read_json


BANDS = (("Fails 0.50", .2, .5, .40), ("Okay", .5, .7, .55),
         ("Good", .8, .92, .85), ("Very good", .95, 1.01, .96))


def main():
    frames = read_json(SOURCE / "per_frame.json")
    truth = {a["id"]: a for a in read_json(SOURCE / "coco_ground_truth.json")["annotations"]}
    candidates, same_object = [], []
    for image_id, frame in enumerate(frames, 1):
        metrics = frame["methods"]["hybrid"]["proposal_metrics"]
        overlaps = np.array(metrics["iou_matrix"])
        for g, gid in enumerate(metrics["gt_annotation_ids"]):
            options = [{"image_id": image_id, "scene_id": frame["scene_id"], "im_id": frame["im_id"],
                "object_id": oid, "gt_id": gid, "cad_id": metrics["gt_category_ids"][g], "iou": float(overlaps[p, g])}
                for p, oid in enumerate(metrics["prediction_ids"])
                if oid.startswith("sam_") and int(overlaps[p].argmax()) == g]
            candidates.extend(options)
            chosen = [min((r for r in options if lower <= r["iou"] < upper),
                          key=lambda r: abs(r["iou"] - target), default=None)
                      for _, lower, upper, target in BANDS]
            if all(chosen):
                same_object.append(chosen)
    if same_object:
        examples = min(same_object, key=lambda group: sum(abs(r["iou"] - band[3]) for r, band in zip(group, BANDS)))
    else:
        examples = [min((r for r in candidates if lower <= r["iou"] < upper),
                        key=lambda r: abs(r["iou"] - target)) for _, lower, upper, target in BANDS]
    loaded = []
    for row in examples:
        scene = f"scene_{row['scene_id']:06}_{row['im_id']:06}"
        path = SOURCE / "sam" / scene / "masks" / f"{row['object_id']}.png"
        predicted = np.asarray(Image.open(path)) > 0
        gt = mask_utils.decode(truth[row["gt_id"]]["segmentation"]).astype(bool)
        intersection, extra, missing = predicted & gt, predicted & ~gt, gt & ~predicted
        iou = float(intersection.sum() / (predicted | gt).sum())
        np.testing.assert_allclose(iou, row["iou"], atol=1e-12, rtol=0)
        rgb = np.asarray(Image.open(ROOT / "outputs/cache/bop_tless/data/tless/test_primesense" /
                         f"{row['scene_id']:06}/rgb/{row['im_id']:06}.png").convert("RGB"))
        row.update(mask_path=str(path.relative_to(ROOT)), intersection_pixels=int(intersection.sum()),
                   extra_pixels=int(extra.sum()), missed_pixels=int(missing.sum()),
                   pixel_precision=float(intersection.sum() / predicted.sum()),
                   pixel_recall=float(intersection.sum() / gt.sum()),
                   passes={str(t): iou >= t for t in (.5, .75, .9, .95)})
        loaded.append((rgb, predicted, gt, intersection, extra, missing))
    same = len({r["gt_id"] for r in examples}) == 1
    unions = [p | g for _, p, g, *_ in loaded]
    if same:
        unions = [np.logical_or.reduce(unions)] * 4
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_file, 18)
    small = ImageFont.truetype(font_file, 15)
    title_font = ImageFont.truetype(font_file, 23)
    panel = Image.new("RGB", (1360, 890), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    draw.text((22, 15), "Actual saved SAM masks: what mask IoU looks like", font=title_font, fill="#14202b")
    caption = (f"Same physical object: scene {examples[0]['scene_id']}, frame {examples[0]['im_id']}, CAD {examples[0]['cad_id']}"
               if same else "Examples from different objects; each comparison uses its own ground-truth mask")
    draw.text((22, 49), caption + " | CAD identity is ignored here", font=small, fill="#344454")
    for column, (row, data, support, band) in enumerate(zip(examples, loaded, unions, BANDS)):
        x = 16 + 336 * column
        draw.text((x + 10, 81), f"{band[0]}: IoU {row['iou']:.3f}", font=font, fill="#14202b")
        draw.text((x + 10, 108), row["object_id"], font=small, fill="#536171")
        rgb, predicted, gt, intersection, extra, missing = data
        yy, xx = np.nonzero(support)
        pad = max(7, int(.08 * max(xx.max() - xx.min(), yy.max() - yy.min())))
        bounds = (max(0, xx.min() - pad), max(0, yy.min() - pad),
                  min(rgb.shape[1], xx.max() + pad + 1), min(rgb.shape[0], yy.max() + pad + 1))
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        reference = rgb.copy()
        cv2.drawContours(reference, contours, -1, (255, 240, 0), 1)
        prediction_image = np.repeat(predicted[..., None].astype(np.uint8) * 255, 3, axis=2)
        difference = np.full_like(rgb, (28, 34, 42))
        difference[intersection] = (41, 179, 112)
        difference[extra] = (237, 83, 83)
        difference[missing] = (56, 166, 237)
        for y, pixels, label in ((157, reference, "RGB + true outline (yellow)"),
                                 (385, prediction_image, "Predicted mask (white)"),
                                 (613, difference, "Pixel agreement / disagreement")):
            draw.text((x + 6, y - 25), label, font=small, fill="#344454")
            tile = ImageOps.contain(Image.fromarray(pixels).crop(bounds), (320, 190), Image.Resampling.NEAREST)
            panel.paste(tile, (x + (320 - tile.width) // 2, y + (190 - tile.height) // 2))
        draw.text((x + 6, 808), f"Mask precision {row['pixel_precision']:.1%} | recall {row['pixel_recall']:.1%}",
                  font=small, fill="#344454")
    for x, color, text in ((24, "#29b370", "Correct pixels"), (330, "#ed5353", "Extra predicted pixels"),
                           (730, "#38a6ed", "Missed object pixels")):
        draw.rectangle((x, 854, x + 16, 870), fill=color)
        draw.text((x + 25, 852), text, font=font, fill="#14202b")
    panel.save(SOURCE / "mask_quality_examples.png")
    metadata = {"same_physical_object": same, "scope": "Actual saved SAM proposals selected for illustration; no synthetic mask degradation, inference or CAD-label correction",
                "examples": examples}
    (SOURCE / "mask_quality_examples.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
