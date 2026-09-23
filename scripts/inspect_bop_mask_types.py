"""Describe the 416 saved mask failures; diagnostic rules do not alter predictions."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion, binary_fill_holes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/bop_automatic_sam_20260919"
OUTPUT = SOURCE / "mask_type_audit"
DATA = ROOT / "outputs/cache/bop_tless/data/tless/test_primesense"


def read(path):
    return json.loads(path.read_text())


def mask_file(name, oid):
    if oid.startswith("sam_"):
        return SOURCE / "sam" / name / "masks" / f"{oid}.png"
    return ROOT / "outputs/bop_localization_20260917" / name / "masks" / f"{oid.replace('depth_', 'object_')}.png"


def audit():
    rows, tiles = [], {}
    predictions = read(SOURCE / "hybrid_coco_predictions.json")
    for ordinal, frame in enumerate(read(SOURCE / "per_frame.json"), 1):
        scene, image = frame["scene_id"], frame["im_id"]
        name = f"scene_{scene:06}_{image:06}"
        folder = DATA / f"{scene:06}"
        rgb = np.asarray(Image.open(folder / "rgb" / f"{image:06}.png").convert("RGB"))
        paths = sorted((folder / "mask_visib").glob(f"{image:06}_*.png"))
        truth = np.stack([np.asarray(Image.open(p)) > 0 for p in paths])
        areas = truth.sum((1, 2))
        union = truth.any(0)
        holes = np.stack([binary_fill_holes(m) & ~m for m in truth])
        metrics = frame["methods"]["hybrid"]["proposal_metrics"]
        proposals = {oid: np.asarray(Image.open(mask_file(name, oid))) > 0
                     for oid in metrics["prediction_ids"]}
        all_holes = {oid: binary_fill_holes(m) & ~m for oid, m in proposals.items()}
        for prediction in predictions:
            if prediction["image_id"] != ordinal:
                continue
            oid = prediction["object_id"]
            maximum_iou = max(metrics["iou_matrix"][metrics["prediction_ids"].index(oid)])
            if maximum_iou >= .5:
                continue
            mask = proposals[oid]
            area = int(mask.sum())
            intersection = (truth & mask).sum((1, 2))
            precision = intersection / area
            recall = np.divide(intersection, areas, out=np.zeros(len(areas)), where=areas > 0)
            contributors = (precision >= .1) & (recall >= .1)
            if precision.max() >= .95:
                kind = "fragment_of_one_annotated_object"
            elif contributors.sum() >= 2:
                kind = "multiple_annotated_objects"
            elif (mask & union).sum() / area <= .1:
                kind = "outside_annotated_objects"
            else:
                kind = "mixed_or_inaccurate_boundary"
            parent = [(pid, float((pm & mask).sum() / area)) for pid, pm in proposals.items()
                      if pid != oid and pm.sum() >= 1.5 * area]
            parent = max(parent, key=lambda p: p[1]) if parent else (None, 0.)
            hole_parent = max(((pid, float((hm & mask).sum() / area)) for pid, hm in all_holes.items()),
                              key=lambda p: p[1])
            row = {"audit_id": len(rows), "scene_id": scene, "im_id": image, "object_id": oid,
                   "source": oid.split("_")[0], "predicted_cad_id": prediction["category_id"],
                   "score": prediction["score"], "maximum_eligible_gt_iou": maximum_iou,
                   "overlap_group": kind, "mask_pixels": area,
                   "fraction_in_best_annotated_object": float(precision.max()),
                   "fraction_in_any_annotated_object": float((mask & union).sum() / area),
                   "substantial_annotated_object_count": int(contributors.sum()),
                   "fraction_in_best_annotated_hole": float((holes & mask).sum((1, 2)).max() / area),
                   "best_larger_proposal": parent[0], "fraction_in_larger_proposal": parent[1],
                   "best_proposal_hole": hole_parent[0], "fraction_in_proposal_hole": hole_parent[1]}
            rows.append(row)
            y, x = np.nonzero(mask)
            padding = max(25, int(.3 * max(x.max() - x.min(), y.max() - y.min())))
            bounds = (max(0, x.min() - padding), max(0, y.min() - padding),
                      min(mask.shape[1], x.max() + padding + 1), min(mask.shape[0], y.max() + padding + 1))
            overlay = rgb.copy()
            overlay[mask] = (.6 * rgb[mask] + .4 * np.array([255, 40, 80])).astype(np.uint8)
            overlay[union & ~binary_erosion(union)] = [255, 225, 0]
            overlay[mask & ~binary_erosion(mask)] = [0, 255, 255]
            tiles[row["audit_id"]] = Image.fromarray(overlay).crop(bounds)
    assert len(rows) == 416 and Counter(r["source"] for r in rows) == {"sam": 408, "depth": 8}
    return rows, tiles


def review_sheets(rows, tiles):
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if not font_path.exists():
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 17)
    for group in sorted({r["overlap_group"] for r in rows}):
        selected = [r for r in rows if r["overlap_group"] == group]
        for start in range(0, len(selected), 25):
            sheet = Image.new("RGB", (1500, 1250), "#f4f6f8")
            draw = ImageDraw.Draw(sheet)
            for position, row in enumerate(selected[start:start + 25]):
                x, y = position % 5 * 300, position // 5 * 250
                label = f"#{row['audit_id']} S{row['scene_id']} {row['object_id']}"
                draw.text((x + 5, y + 3), label, font=font, fill="#162535")
                draw.text((x + 5, y + 24), f"GT {row['fraction_in_best_annotated_object']:.2f} / hole {row['fraction_in_proposal_hole']:.2f}",
                          font=font, fill="#405263")
                tile = ImageOps.contain(tiles[row["audit_id"]], (290, 195), Image.Resampling.NEAREST)
                sheet.paste(tile, (x + 5 + (290 - tile.width) // 2, y + 50 + (195 - tile.height) // 2))
            sheet.save(OUTPUT / f"{group}_{start // 25 + 1:02}.png")


def write_visual_review(rows):
    """Record the 2026-09-20 inspection of all mixed/outside and merge sheets.

    These are descriptive labels for this frozen output, not inference rules.
    High-purity fragments also have direct support from the reference masks.
    """
    printed = {
        214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
        228, 230, 231, 233, 234, 235, 244, 245, 247, 249, 251, 252, 253, 255,
        256, 262, 267, 269, 272, 274, 275, 276, 278, 279, 280, 281, 282, 283,
        285, 286, 290, 291, 374, 392, 393, 400,
        250, 277, 284, 288, 370, 372, 389, 397,
    }
    mixed = {232, 254, 289, 396, 398, 399, 410}
    ambiguous_holes = {383, 387}
    reviewed = []
    for row in rows:
        index = row["audit_id"]
        if index == 365:
            label = "background_and_inner_wall_in_tape_opening"
        elif index in ambiguous_holes:
            label = "ambiguous_hollow_distractor_surface"
        elif index in printed:
            label = "printed_background_or_book_surface"
        elif index in mixed:
            label = "target_and_background_mixture"
        elif row["overlap_group"] == "multiple_annotated_objects":
            label = "multiple_target_objects"
        elif row["overlap_group"] == "outside_annotated_objects":
            label = "non_catalog_object_or_its_part"
        else:
            label = "target_part_or_incomplete_mask"
        reviewed.append({**row, "visual_review_label": label})
    assert reviewed[365]["scene_id"] == 18 and reviewed[365]["object_id"] == "sam_080"
    summary = {
        "scope": "Descriptive visual audit of the saved 416 mask failures, not a new segmentation run or benchmark metric",
        "review_method": "134 high-purity target fragments established by >=95% reference-mask containment; all 169 mixed, 103 outside-target and 10 multi-target detections inspected in RGB overlays",
        "counts": dict(Counter(r["visual_review_label"] for r in reviewed)),
        "by_source": {source: dict(Counter(r["visual_review_label"] for r in reviewed if r["source"] == source))
                      for source in ("sam", "depth")},
        "separate_object_inside_opening": "No confirmed case in this review; not a proof that such scenes never occur",
        "caveats": ["Labels describe the dominant visible error; boundaries and object-part errors can coexist",
                    "Tape-opening mask includes the inner wall and the background seen through it",
                    "Two hollow distractor masks remain ambiguous: inner material versus background through the opening",
                    "Other masks include real objects absent from the 30-CAD catalog; missing a target annotation does not mean empty space"],
    }
    (OUTPUT / "visual_review.json").write_text(json.dumps({**summary, "detections": reviewed}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    rows, tiles = audit()
    summary = {"scope": "416 final hybrid detections with IoU < .5 to every eligible target; original SAM plus depth",
               "definitions": {"fragment": ">=95% mask pixels belong to one annotated object",
                               "merge": ">=2 objects each contribute >=10% of mask pixels and have >=10% of their visible area covered",
                               "outside": ">=90% of mask pixels outside all annotated visible objects",
                               "remaining": "Other mixtures or inaccurate boundaries; not automatically assigned a semantic cause"},
               "overlap_counts": dict(Counter(r["overlap_group"] for r in rows)),
               "source_counts": dict(Counter(r["source"] for r in rows)),
               "contained_95pct_in_1point5x_larger_proposal": sum(r["fraction_in_larger_proposal"] >= .95 for r in rows),
               "inside_annotated_hole_90pct": sum(r["fraction_in_best_annotated_hole"] >= .9 for r in rows),
               "inside_proposal_hole_90pct": sum(r["fraction_in_proposal_hole"] >= .9 for r in rows)}
    (OUTPUT / "overlap_audit.json").write_text(json.dumps({**summary, "detections": rows}, indent=2) + "\n")
    review_sheets(rows, tiles)
    print(json.dumps(summary, indent=2), flush=True)
    write_visual_review(rows)


if __name__ == "__main__":
    main()
