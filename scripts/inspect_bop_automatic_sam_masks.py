"""Evaluate completed masks only; never feed annotations back into matching."""

from scripts.run_bop_automatic_sam_masks import OUTPUT, baseline, read_json, save_json
from scripts.bop_segmentation_metrics import localization_metrics
import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    plan = read_json(OUTPUT / "sam_protocol.json")["plan"]
    truth = read_json(baseline.OUTPUT / "coco_ground_truth.json")
    assert all((OUTPUT / "sam" / baseline.frame_folder(i).name / "proposals.json").exists() for i in plan)
    frames = []
    for ordinal, item in enumerate(plan, 1):
        name = baseline.frame_folder(item).name
        folder = OUTPUT / "sam" / name
        annotations = [a for a in truth["annotations"] if a["image_id"] == ordinal]
        result = {**item, "variants": {}}
        for label, manifest, subdir in (("raw", "raw_proposals.json", "raw_masks"),
                                        ("retained", "proposals.json", "masks")):
            records = read_json(folder / manifest)
            masks = {r["object_id"]: cv2.imread(str(folder / subdir / f"{r['object_id']}.png"), 0) > 0 for r in records}
            metrics = localization_metrics(masks, annotations)
            result["variants"][label] = {k: metrics[k] for k in (
                "predicted_regions", "eligible_instances", "iou_0.50", "iou_0.75")}
            if item == {"scene_id": 2, "im_id": 3} and label == "retained":
                rgb, *_ = baseline.frame_input(item)
                contour_image = rgb.copy()
                for mask in masks.values():
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 1)
                height, width = rgb.shape[:2]
                panel = Image.new("RGB", (2 * width, height + 45), "white")
                panel.paste(Image.fromarray(rgb), (0, 45))
                panel.paste(Image.fromarray(contour_image), (width, 45))
                draw = ImageDraw.Draw(panel)
                draw.text((12, 15), "T-LESS scene 2, frame 3: input", fill="black")
                draw.text((width + 12, 15), f"Automatic SAM: all {len(masks)} retained mask contours (overlaps allowed)", fill="black")
                panel.save(OUTPUT / "scene_000002_000003_proposals.png")
        frames.append(result)
    summary = {}
    for variant in ("raw", "retained"):
        rows = [f["variants"][variant] for f in frames]
        summary[variant] = {"candidate_regions": sum(r["predicted_regions"] for r in rows),
            "eligible_instances": sum(r["eligible_instances"] for r in rows),
            "matched_iou50": sum(r["iou_0.50"]["matched"] for r in rows),
            "matched_iou75": sum(r["iou_0.75"]["matched"] for r in rows),
            "empty_frames": sum(r["predicted_regions"] == 0 for r in rows)}
    save_json(OUTPUT / "proposal_evaluation.json", {"summary": summary, "frames": frames,
        "scope": "Class-agnostic mask recall; all masks generated before evaluation; no CAD predictions used",
        "visualization": "All retained contours, no ground-truth selection or filtering"})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
