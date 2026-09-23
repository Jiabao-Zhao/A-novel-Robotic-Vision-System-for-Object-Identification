"""Show saved background masks before and after CAD/table filtering."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/bop_automatic_sam_20260919"
DATA = ROOT / "outputs/cache/bop_tless/data/tless/test_primesense"


def read(path):
    return json.loads(path.read_text())


def load_example(scene, image, oid):
    name = f"scene_{scene:06}_{image:06}"
    folder = DATA / f"{scene:06}"
    rgb = np.asarray(Image.open(folder / "rgb" / f"{image:06}.png").convert("RGB"))
    mask = np.asarray(Image.open(SOURCE / "sam" / name / "masks" / f"{oid}.png")) > 0
    truth = np.logical_or.reduce([np.asarray(Image.open(p)) > 0
        for p in sorted((folder / "mask_visib").glob(f"{image:06}_*.png"))])
    camera = read(folder / "scene_camera.json")[str(image)]
    K = np.asarray(camera["cam_K"]).reshape(3, 3)
    depth = np.asarray(Image.open(folder / "depth" / f"{image:06}.png")) * camera["depth_scale"] * .001
    plane = np.asarray(read(ROOT / "outputs/bop_localization_20260917" / name / "localization.json")["plane_model"])
    plane /= np.linalg.norm(plane[:3])
    if plane[3] < 0:
        plane *= -1
    y, x = np.nonzero(mask & (depth > 0))
    z = depth[y, x]
    points = np.column_stack(((x - K[0, 2]) * z / K[0, 0],
                              (y - K[1, 2]) * z / K[1, 1], z))
    heights = points @ plane[:3] + plane[3]
    frame = next(f for f in read(SOURCE / "per_frame.json")
                 if f["scene_id"] == scene and f["im_id"] == image)
    metrics = frame["methods"]["hybrid"]["proposal_metrics"]
    maximum_iou = max(metrics["iou_matrix"][metrics["prediction_ids"].index(oid)])
    predictions = read(SOURCE / "matching" / name / "predictions.json")["hybrid"]
    detection = next((p for p in predictions if p["object_id"] == oid), None)
    region = read(SOURCE / "matching" / name / "regions" / f"{oid}.json")
    result = {"scene_id": scene, "im_id": image, "object_id": oid,
              "retained_final_detection": detection is not None,
              "predicted_cad_id": detection["category_id"] if detection else None,
              "score": detection["score"] if detection else None,
              "mask_pixels": int(mask.sum()), "maximum_eligible_gt_iou": maximum_iou,
              "fraction_outside_all_annotated_visible_objects": float((mask & ~truth).sum() / mask.sum()),
              "valid_depth_fraction": len(x) / int(mask.sum()),
              "valid_points_within_5mm_of_saved_plane": float(np.mean(np.abs(heights) <= .005)),
              "median_absolute_plane_distance_mm": float(np.median(np.abs(heights)) * 1000),
              "height_mm_q10_q50_q90": (np.quantile(heights, [.1, .5, .9]) * 1000).tolist(),
              "saved_pose_status_counts": dict(Counter(v["pose_status"] for v in region["views"]))}
    assert maximum_iou == 0 and result["fraction_outside_all_annotated_visible_objects"] == 1
    return rgb, mask, result


def render(rgb, mask, row, title, output_name):
    overlay = rgb.copy()
    overlay[mask] = (.45 * rgb[mask] + .55 * np.array([255, 30, 120])).astype(np.uint8)
    boundary = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) &
                        np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    overlay[boundary] = [255, 255, 0]
    panel = Image.new("RGB", (1464, 690), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if not font_path.exists():
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    heading, text = [ImageFont.truetype(str(font_path), size) for size in (25, 20)]
    draw.text((12, 10), title, font=heading, fill="#172531")
    draw.text((12, 49), f"Original RGB | Scene {row['scene_id']}, frame {row['im_id']}", font=text, fill="#405263")
    status = (f"CAD {row['predicted_cad_id']} | score {row['score']:.3f} | retained"
              if row["retained_final_detection"] else "already rejected by the CAD/table stage")
    draw.text((744, 49), f"Pink = {row['object_id']} | {status}", font=text, fill="#405263")
    panel.paste(Image.fromarray(rgb), (12, 82))
    panel.paste(Image.fromarray(overlay), (744, 82))
    draw.text((12, 633), f"Outside all annotated objects: {row['fraction_outside_all_annotated_visible_objects']:.1%}   |   "
              f"Valid depth within 5 mm of saved plane: {row['valid_points_within_5mm_of_saved_plane']:.1%}",
              font=text, fill="#172531")
    draw.text((12, 660), "Original SAM automatic proposals. Saved outputs only; no new segmentation, CAD fitting, or plane estimation.",
              font=text, fill="#405263")
    panel.save(SOURCE / output_name)


def main():
    examples = []
    for scene, image, oid, title, filename in (
        (1, 1, "sam_001", "SAM segmented the flat background marker strip", "table_background_proposal.png"),
        (18, 2, "sam_080", "Extra detection over a tape-roll opening and the background inside it", "background_mask_example.png"),
    ):
        rgb, mask, row = load_example(scene, image, oid)
        render(rgb, mask, row, title, filename)
        examples.append(row)
    assert not examples[0]["retained_final_detection"] and examples[1]["retained_final_detection"]
    assert examples[0]["saved_pose_status_counts"] == {"table_penetration": 1260}
    result = {"scope": "Illustrative saved masks; ground truth used only for diagnostics, not prediction.",
              "examples": examples}
    (SOURCE / "background_mask_example.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
