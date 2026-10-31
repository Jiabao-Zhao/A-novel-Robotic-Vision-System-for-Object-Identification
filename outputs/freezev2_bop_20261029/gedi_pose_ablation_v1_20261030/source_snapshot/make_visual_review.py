"""Render saved inference selections, without reading any ground-truth files.

Sheets are visual sanity evidence only. They do not establish pose correctness.
Original masks, crops, overlays, poses and scores are read but never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(repo, value):
    value = str(value).replace("\\", "/")
    if os.name != "nt" and value[1:2] == ":":
        value = "/mnt/" + value[0].lower() + value[2:]
    elif os.name == "nt" and value.startswith("/mnt/") and value[6:7] == "/":
        value = value[5].upper() + ":" + value[6:]
    p = Path(value)
    return p if p.is_absolute() else Path(repo) / p


def _font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _key(row):
    return (int(row["scene_id"]), int(row.get("im_id", row.get("image_id"))),
            int(row.get("obj_id", row.get("object_id"))))


def _candidate(image_folder, target, candidate_id, *, optional=False):
    matches = [c for c in target.get("candidates", []) if c["candidate_id"] == candidate_id]
    if len(matches) != 1:
        if optional and not matches:
            return None
        raise ValueError(f"Candidate selection is missing/ambiguous: {candidate_id}")
    entry = matches[0]
    directory = image_folder / entry["directory"].replace("\\", "/")
    if not directory.resolve().is_relative_to(image_folder.resolve()):
        raise ValueError("Candidate directory escapes its saved inference image")
    diagnostic_path = directory / "diagnostic.json"
    expected = entry.get("sha256", entry.get("diagnostic_sha256"))
    if expected and _sha(diagnostic_path) != expected:
        raise ValueError(f"Candidate diagnostic checksum changed: {candidate_id}")
    trace = _json(diagnostic_path)
    if trace.get("GT_pose_access") is True:
        raise ValueError("GT-assisted input is outside this visual review")
    files = {}
    for name in ("mask.png", "crop.png", "pose_overlay.png"):
        p = directory / name
        if p.exists():
            actual = _sha(p)
            old = trace.get("artifacts", {}).get(name)
            if old and actual != old:
                raise ValueError(f"Saved image artifact checksum changed: {p}")
            files[name] = {"path": str(p), "sha256": actual}
    return {"candidate_id": candidate_id, "record": int(candidate_id.rsplit(":", 1)[1]),
            "status": trace.get("status"), "reason": trace.get("reason"),
            "diagnostic": str(diagnostic_path), "diagnostic_sha256": _sha(diagnostic_path),
            "files": files}


def _selection(ledger_path, target_key):
    ledger_path = Path(ledger_path)
    ledger = _json(ledger_path)
    if ledger.get("GT_pose_access") is True:
        raise ValueError("GT-assisted inference ledger is outside this review")
    # Original suite rows keep scene/image in the image-ledger identity; the
    # new arm rows repeat them per target. Resolve both saved schemas.
    identity = ledger.get("identity", {})
    targets = [t for t in ledger["targets"] if _key({**identity, **t}) == target_key]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one saved target row: {target_key}")
    selected = [p for p in ledger.get("predictions", []) if _key(p) == target_key]
    if len(selected) > 1:
        raise ValueError("This fixed cohort expects one retained pose per target row")
    result = {"ledger": str(ledger_path), "ledger_sha256": _sha(ledger_path),
              "target_status": targets[0]["status"], "selected": None}
    if selected:
        p = selected[0]
        candidate = _candidate(ledger_path.parent, targets[0], p["candidate_id"])
        if candidate["status"] != "complete" or "pose_overlay.png" not in candidate["files"]:
            raise ValueError("A retained pose has no complete saved pose overlay")
        candidate["score"] = float(p["score"])
        result["selected"] = candidate
    return result, targets[0]


def _paste_fit(canvas, image, box, *, nearest=False):
    x, y, w, h = box
    image = image.convert("RGB")
    image.thumbnail((w, h), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (w - image.width) // 2, y + (h - image.height) // 2))


def _panel(canvas, box, label, candidate, kind):
    x, y, w, h = box
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(155, 163, 174), width=1)
    draw.text((x + 6, y + 4), label, font=_font(15), fill=(23, 39, 62))
    if candidate is None:
        draw.text((x + 10, y + 70), "No retained pose", font=_font(18), fill=(130, 58, 45))
        return
    if kind == "overlay":
        path = candidate["files"].get("pose_overlay.png")
        if path:
            with Image.open(path["path"]) as im:
                _paste_fit(canvas, im, (x + 3, y + 25, w - 6, h - 57))
    else:
        half = (w - 12) // 2
        for i, name in enumerate(("crop.png", "mask.png")):
            artifact = candidate["files"].get(name)
            subx = x + 4 + i * (half + 4)
            if artifact:
                with Image.open(artifact["path"]) as im:
                    if name == "mask.png":
                        bbox = im.convert("L").getbbox()
                        if bbox:
                            left, top, right, bottom = bbox
                            bbox = (max(0, left - 3), max(0, top - 3),
                                    min(im.width, right + 3), min(im.height, bottom + 3))
                            im = im.crop(bbox)
                    _paste_fit(canvas, im, (subx, y + 27, half, h - 70), nearest=name == "mask.png")
                draw.text((subx + 2, y + h - 40), "RGB crop" if i == 0 else "mask ROI", font=_font(12), fill=(30, 40, 50))
            else:
                draw.text((subx + 2, y + 65), "Not saved", font=_font(13), fill=(130, 58, 45))
    score = candidate.get("score")
    note = f"record {candidate['record']}"
    if score is not None:
        note += f" | score {score:.5g}"
    draw.text((x + 5, y + h - 20), note, font=_font(12), fill=(30, 40, 50))


def _save_sheet(rows, path, index):
    panel_w, panel_h, margin, row_h, top = 280, 245, 10, 299, 88
    canvas = Image.new("RGB", (5 * panel_w + 2 * margin, top + len(rows) * row_h + 26), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), f"Fixed inference-selected cohort: sheet {index}", font=_font(23), fill=(16, 36, 57))
    draw.text((margin, 39), "Orange = coarse CAD projection; green = ICP projection. No GT comparison or accuracy selection.", font=_font(15), fill=(35, 45, 58))
    draw.text((margin, 61), "Mask ROI display is cropped to foreground for readability; original image files remain unchanged. Scores are pose-ranking scores.", font=_font(13), fill=(55, 65, 78))
    for n, row in enumerate(rows):
        y = top + n * row_h
        scene, image, obj = row["target"]
        draw.text((margin, y), f"scene {scene:06d} | image {image:06d} | CAD {obj:02d} | {row['case_id']}", font=_font(17), fill=(16, 36, 57))
        y += 29
        cards = [("Original baseline", row["baseline"]["selected"], "overlay"),
                 ("Repair224: crop + mask", row["repair224"]["selected"], "inputs"),
                 ("Repair224: saved pose", row["repair224"]["selected"], "overlay"),
                 ("Repair476: crop + mask", row["repair476"]["selected"], "inputs"),
                 ("Repair476: saved pose", row["repair476"]["selected"], "overlay")]
        for col, (title, candidate, kind) in enumerate(cards):
            _panel(canvas, (margin + col * panel_w, y, panel_w - 4, panel_h), title, candidate, kind)
    assert canvas.height <= 1600
    canvas.save(path)
    return {"path": str(path), "sha256": _sha(path), "width": canvas.width, "height": canvas.height}


def _save_failure(candidates, path):
    canvas = Image.new("RGB", (1120, 435), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), "Repair224 failure: candidate 7748 | scene 000003 / image 000476 / CAD 08", font=_font(22), fill=(25, 35, 50))
    draw.text((12, 39), "Arm-specific saved status below; inputs are unchanged. No GT comparison.", font=_font(16), fill=(80, 45, 35))
    first = next(iter(candidates.values()))
    _panel(canvas, (12, 76, 365, 315), "Original crop + predicted mask", first, "inputs")
    y = 80
    for arm, candidate in candidates.items():
        retained = "selected" if candidate.get("selected") else "not selected"
        draw.text((395, y), f"{arm}: {candidate['status']} ({retained})", font=_font(19), fill=(110, 40, 35))
        fallback = ("A pose was computed and saved for this candidate; the image ledger retained another candidate."
                    if candidate["status"] == "complete" and not candidate.get("selected")
                    else "See saved diagnostic for status details.")
        text = str(candidate.get("reason") or fallback).replace("\n", " ")
        for line in textwrap.wrap(text, width=83)[:4]:
            y += 22
            draw.text((395, y), line, font=_font(14), fill=(25, 35, 50))
        y += 50
    canvas.save(path)
    return {"path": str(path), "sha256": _sha(path), "width": canvas.width, "height": canvas.height}


def render(repo_root, output_root, cohort_path=None):
    """Render only after both arms have finalized all fixed cohort target rows."""
    repo, out = Path(repo_root), _path(Path(repo_root), output_root)
    cohort_path = Path(cohort_path) if cohort_path else Path(__file__).with_name("cohort.json")
    cohort = _json(cohort_path)
    expected = {_key(t) for t in cohort["targets"]}
    summaries = {}
    for arm in ("repair224", "repair476"):
        path = out / arm / "inference_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Awaiting completed {arm} inference summary: {path}")
        summary = _json(path)
        if (summary.get("GT_pose_access") is True
                or len(summary["targets"]) != len(expected)
                or {_key(t) for t in summary["targets"]} != expected):
            raise ValueError(f"Incomplete or incompatible fixed target coverage: {arm}")
        summaries[arm] = {"path": str(path), "sha256": _sha(path), "predictions": summary["predictions"]}
    rows, candidate_7748 = [], {}
    baseline_run = _path(repo, cohort["baseline_run"])
    for case in sorted(cohort["cases"], key=lambda c: _key(c["target"])):
        key = _key(case["target"])
        row = {"case_id": case["case_id"], "target": list(key)}
        b = case["baseline"]
        baseline_path = _path(repo, b.get("ledger_repo_relative_path") or b["ledger_path"])
        if _sha(baseline_path) != b["ledger_sha256"]:
            raise ValueError("Original baseline ledger checksum changed")
        if not baseline_path.resolve().is_relative_to(baseline_run.resolve()):
            raise ValueError("Baseline ledger escapes its original run")
        row["baseline"], _ = _selection(baseline_path, key)
        for arm in ("repair224", "repair476"):
            ledger = out / arm / "images" / f"{key[0]:06d}_{key[1]:06d}" / "complete.json"
            row[arm], target = _selection(ledger, key)
            if key == (3, 476, 8):
                candidate = _candidate(ledger.parent, target, "cnos-fastsam:000003:000476:7748", optional=True)
                if candidate:
                    selected = row[arm]["selected"]
                    candidate["selected"] = bool(selected and selected["candidate_id"] == candidate["candidate_id"])
                    candidate_7748[arm] = candidate
        rows.append(row)
    if len(rows) != len(expected) or {_key(c["target"]) for c in cohort["cases"]} != expected:
        raise ValueError("Cohort case entries do not cover the exact fixed target rows")
    visual = out / "visuals"
    visual.mkdir(exist_ok=True)
    files = []
    for i in range(0, len(rows), 4):
        files.append(_save_sheet(rows[i:i + 4], visual / f"retained_pose_comparison_{i // 4 + 1}.png", i // 4 + 1))
    failed = {arm: c for arm, c in candidate_7748.items() if c["status"] != "complete"}
    if failed:
        files.append(_save_failure(candidate_7748, visual / "failed_candidate_7748.png"))
    result = {"scope": "Saved inference-selected poses and failure inputs; no GT or metric access.",
              "GT_pose_access": False, "GT_or_metric_used_for_selection": False,
              "cohort": str(cohort_path), "cohort_sha256": _sha(cohort_path),
              "source_summaries": summaries, "cases": rows, "failed_candidate_7748": failed,
              "candidate_7748_by_arm": candidate_7748,
              "retained_pose_counts": {a: sum(r[a]["selected"] is not None for r in rows)
                                       for a in ("baseline", "repair224", "repair476")},
              "files": files, "generator_sha256": _sha(__file__),
              "limitations": "Visual sanity checks alone do not establish correct 6D poses or benchmark performance."}
    manifest = visual / "contact_sheet_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output")
    parser.add_argument("--cohort", type=Path, default=Path(__file__).with_name("cohort.json"))
    args = parser.parse_args()
    output = args.output or _json(Path(__file__).with_name("config.json"))["output"]
    result = render(args.repo_root, output, args.cohort)
    print(json.dumps({"retained_pose_counts": result["retained_pose_counts"], "files": result["files"]}, indent=2))


if __name__ == "__main__":
    main()
