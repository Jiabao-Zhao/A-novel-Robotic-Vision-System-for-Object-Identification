"""Show the saved scene with the most final masks below target IoU 0.5."""

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_utils

from scripts.inspect_bop_mask_types import ROOT, SOURCE, DATA, read


def outline(pixels, mask, color, thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pixels, contours, -1, color, thickness)


def main():
    review = read(SOURCE / "mask_type_audit/visual_review.json")["detections"]
    scene, frame = Counter((r["scene_id"], r["im_id"]) for r in review).most_common(1)[0][0]
    frames = read(SOURCE / "per_frame.json")
    image_id = next(i for i, f in enumerate(frames, 1)
                    if (f["scene_id"], f["im_id"]) == (scene, frame))
    failures = [r for r in review if (r["scene_id"], r["im_id"]) == (scene, frame)]
    predictions = {r["object_id"]: r for r in read(SOURCE / "hybrid_coco_predictions.json")
                   if r["image_id"] == image_id}
    annotations = [a for a in read(SOURCE / "coco_ground_truth.json")["annotations"]
                   if a["image_id"] == image_id and not a["ignore"]]
    truth = [mask_utils.decode(a["segmentation"]).astype(bool) for a in annotations]
    rgb = np.asarray(Image.open(DATA / f"{scene:06}/rgb/{frame:06}.png").convert("RGB"))
    reference = rgb.copy()
    for mask in truth:
        outline(reference, mask, (255, 225, 0))

    font_file = Path("C:/Windows/Fonts/arial.ttf")
    if not font_file.exists():
        font_file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    title, text, small = [ImageFont.truetype(str(font_file), n) for n in (29, 23, 20)]
    panel = Image.new("RGB", (1480, 1090), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    draw.text((20, 15), f"Most unmatched final masks: T-LESS scene {scene}, frame {frame}",
              font=title, fill="#172531")
    draw.text((20, 56), f"{len(annotations)} target instances | {len(predictions)} final detections | "
              f"{len(failures)} masks below IoU 0.50 with every target", font=text, fill="#405263")
    for left, label, pixels in ((20, "Original RGB image", rgb),
                                (760, "Yellow outlines: seven reference target masks", reference)):
        draw.text((left, 101), label, font=text, fill="#172531")
        tile = ImageOps.contain(Image.fromarray(pixels), (700, 525))
        panel.paste(tile, (left, 139))

    # Choose the highest-scoring saved failure in each illustrative category.
    categories = (("target_part_or_incomplete_mask", "Part of a target"),
                  ("multiple_target_objects", "Several targets merged"),
                  ("non_catalog_object_or_its_part", "Non-catalog object"),
                  ("background_and_inner_wall_in_tape_opening", "Inside the tape opening"))
    draw.text((20, 690), "Four retained detections: cyan = predicted mask; yellow = reference targets",
              font=text, fill="#172531")
    selected = []
    for i, (category, label) in enumerate(categories):
        row = max((r for r in failures if r["visual_review_label"] == category),
                  key=lambda r: r["score"])
        prediction = predictions[row["object_id"]]
        mask = mask_utils.decode(prediction["segmentation"]).astype(bool)
        overlaps = [float((mask & g).sum() / (mask | g).sum()) for g in truth]
        np.testing.assert_allclose(max(overlaps), row["maximum_eligible_gt_iou"], atol=1e-12)
        support = mask | (truth[int(np.argmax(overlaps))] if max(overlaps) > .1 else False)
        y, x = np.nonzero(support)
        bounds = (max(0, x.min() - 22), max(0, y.min() - 22),
                  min(rgb.shape[1], x.max() + 23), min(rgb.shape[0], y.max() + 23))
        overlay = rgb.copy()
        overlay[mask] = (.62 * rgb[mask] + .38 * np.array([0, 220, 255])).astype(np.uint8)
        for g in truth:
            outline(overlay, g, (255, 225, 0), 1)
        outline(overlay, mask, (0, 235, 255), 1)
        left = 20 + 370 * i
        draw.text((left, 735), label, font=text, fill="#172531")
        draw.text((left, 767), f"{row['object_id']} | assigned CAD {prediction['category_id']}",
                  font=small, fill="#405263")
        tile = ImageOps.contain(Image.fromarray(overlay).crop(bounds), (340, 210), Image.Resampling.NEAREST)
        panel.paste(tile, (left + (340 - tile.width) // 2, 800 + (210 - tile.height) // 2))
        draw.text((left, 1020), f"Best target IoU {max(overlaps):.3f}", font=small, fill="#172531")
        selected.append({"object_id": row["object_id"], "category": category,
                         "best_target_iou": max(overlaps), "predicted_cad_id": prediction["category_id"]})
    draw.text((20, 1058), "Saved original-SAM + depth results. Real distractors may be segmented correctly but assigned an unsupported CAD identity.",
              font=small, fill="#405263")
    output = SOURCE / "worst_mask_scene.png"
    panel.save(output)
    print(json.dumps({"scene": scene, "frame": frame, "examples": selected,
                      "image": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
