"""Show an actual recognised object plus two retained false fragment detections."""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_utils

from scripts.inspect_bop_mask_cad_oracles import ROOT, SOURCE, read_json


def main():
    frames = read_json(SOURCE / "per_frame.json")
    annotations = {a["id"]: a for a in read_json(SOURCE / "coco_ground_truth.json")["annotations"]}
    outputs = read_json(SOURCE / "hybrid_coco_predictions.json")
    choices = []
    for ordinal, frame in enumerate(frames, 1):
        metrics = frame["methods"]["hybrid"]["proposal_metrics"]
        rows = [r for r in outputs if r["image_id"] == ordinal]
        for g, gid in enumerate(metrics["gt_annotation_ids"]):
            candidates = []
            for row in rows:
                p = metrics["prediction_ids"].index(row["object_id"])
                overlaps = np.array(metrics["iou_matrix"][p])
                if int(overlaps.argmax()) != g or overlaps[g] <= 0:
                    continue
                recall = metrics["gt_coverage_matrix"][p][g]
                precision = recall / (recall / overlaps[g] + recall - 1)
                candidates.append({"prediction": row, "iou": float(overlaps[g]),
                                   "pixel_precision": precision, "pixel_recall": recall})
            good = [r for r in candidates if r["iou"] >= .85 and
                    r["prediction"]["category_id"] == metrics["gt_category_ids"][g]]
            fragments = [r for r in candidates if .12 <= r["iou"] < .45 and r["pixel_precision"] >= .95]
            if good and len(fragments) >= 2:
                reference = max(good, key=lambda r: r["iou"])
                fragments.sort(key=lambda r: r["iou"], reverse=True)
                choices.append((reference["iou"], frame, gid, [reference, fragments[0], fragments[-1]]))
    assert choices, "No matching saved example; inspect other false-positive types."
    _, frame, gid, selected = max(choices, key=lambda r: r[0])
    gt = mask_utils.decode(annotations[gid]["segmentation"]).astype(bool)
    rgb_path = ROOT / "outputs/cache/bop_tless/data/tless/test_primesense" / f"{frame['scene_id']:06}/rgb/{frame['im_id']:06}.png"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    masks = [mask_utils.decode(r["prediction"]["segmentation"]).astype(bool) for r in selected]
    yy, xx = np.nonzero(gt | np.logical_or.reduce(masks))
    pad = 18
    bounds = (max(0, xx.min() - pad), max(0, yy.min() - pad),
              min(rgb.shape[1], xx.max() + pad + 1), min(rgb.shape[0], yy.max() + pad + 1))
    for row, mask in zip(selected, masks):
        actual = (mask & gt).sum() / (mask | gt).sum()
        np.testing.assert_allclose(actual, row["iou"], atol=1e-12)
        row["pixel_precision"] = float((mask & gt).sum() / mask.sum())
        row["pixel_recall"] = float((mask & gt).sum() / gt.sum())
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    large, normal, small = [ImageFont.truetype(font_path, size) for size in (23, 18, 16)]
    panel = Image.new("RGB", (1400, 600), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    draw.text((20, 16), "One real object, three retained output detections", font=large, fill="#172531")
    draw.text((20, 51), f"Scene {frame['scene_id']}, frame {frame['im_id']} | True CAD {annotations[gid]['category_id']} | All three masks survived final filtering",
              font=normal, fill="#405263")
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    reference = rgb.copy()
    cv2.drawContours(reference, contours, -1, (255, 235, 0), 1)
    panels = [(reference, "The actual object", "Yellow = ground-truth outline", None)]
    for index, (row, mask) in enumerate(zip(selected, masks)):
        overlay = (rgb.astype(float) * .40).astype(np.uint8)
        overlay[mask] = (.60 * rgb[mask] + .40 * np.array([0, 220, 255])).astype(np.uint8)
        cv2.drawContours(overlay, contours, -1, (255, 235, 0), 1)
        panels.append((overlay, "Correct detection" if index == 0 else f"Extra detection {index}",
                       f"{row['prediction']['object_id']} | predicted CAD {row['prediction']['category_id']}", mask))
    for i, (pixels, title, subtitle, mask) in enumerate(panels):
        x = 16 + i * 348
        draw.text((x, 97), title, font=normal, fill="#172531")
        draw.text((x, 126), subtitle, font=small, fill="#405263")
        tile = ImageOps.contain(Image.fromarray(pixels).crop(bounds), (330, 220), Image.Resampling.NEAREST)
        panel.paste(tile, (x + (330 - tile.width) // 2, 158 + (220 - tile.height) // 2))
        if mask is not None:
            binary = Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB").crop(bounds)
            tile = ImageOps.contain(binary, (330, 130), Image.Resampling.NEAREST)
            panel.paste(tile, (x + (330 - tile.width) // 2, 392 + (130 - tile.height) // 2))
            row = selected[i - 1]
            draw.text((x, 535), f"IoU {row['iou']:.3f} | score {row['prediction']['score']:.3f}", font=normal, fill="#172531")
        else:
            draw.text((x, 421), "Cyan = predicted region", font=normal, fill="#405263")
            draw.text((x, 450), "White below = binary mask", font=normal, fill="#405263")
    draw.text((20, 574), "The two fragment masks have IoU < 0.5 with every eligible target: both belong to the 416 failed-mask detections.",
              font=small, fill="#405263")
    panel.save(SOURCE / "extra_mask_example.png")
    result = {"scene_id": frame["scene_id"], "im_id": frame["im_id"], "gt_id": gid,
        "true_cad_id": annotations[gid]["category_id"], "scope": "Unmodified final predictions; no new SAM or CAD scoring",
        "examples": [{"object_id": r["prediction"]["object_id"], "predicted_cad_id": r["prediction"]["category_id"],
                      "score": r["prediction"]["score"], "iou": r["iou"],
                      "pixel_precision": r["pixel_precision"], "pixel_recall": r["pixel_recall"]} for r in selected]}
    (SOURCE / "extra_mask_example.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
