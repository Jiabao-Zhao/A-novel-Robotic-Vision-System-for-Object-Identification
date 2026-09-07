"""Frozen 500-scene OLD-ID versus NEW-letter visual labeling; no production changes.

Run check, pilot (two format-check pairs), full, or analyze from the repo root.
Both conditions return A/B/.../N and use only raw decision-token likelihood.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

from prompt import _vlm_prompt
from vlm_module import candidate_choice_map, candidate_prompt_records, create_roi_contact_sheet, load_localized_objects
from scripts.run_vlm_output_format_ablation import (
    MODEL, SCENE_ROOT, OUTPUT_ROOT as PREVIOUS_ROOT, SYSTEM_PROMPT,
    digest, extract_decision, load_samples, paced_completion, request_arguments,
)


OUTPUT_ROOT = SCENE_ROOT.parent / "visual_label_ablation"
CONDITIONS = ("old_object_ids", "new_direct_letters")
TILE_SIZE = 448
COLUMNS = 4
HEADER_HEIGHT = 34
WORKERS = 4
SCORE = "raw_sequence_likelihood"
ANALYSIS_PLAN = {
    "split": {"calibration": 300, "validation": 100, "test": 100},
    "operating_points": "Separate thresholds per condition maximizing validation coverage at >=90% and >=95% accuracy on both development splits; freeze before test evaluation.",
    "matched_coverage": "All exactly attainable common coverages; ties intact and unavailable scores deferred. Test curves descriptive, never select deployment thresholds on test.",
    "paired_accuracy": "OLD-to-NEW transitions, exact McNemar and paired scene bootstrap; fixed ten-target mix, not unseen-category inference.",
    "high_confidence_wrong": [0.99, 0.999, 0.99999, 1.0],
}


def save_json(path, payload):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prompts_and_map(target, detections):
    mapping = candidate_choice_map(detections)
    candidates = candidate_prompt_records(detections, mapping)
    old = _vlm_prompt(target, candidates)
    # Only identity representation changes; all geometric fields stay identical.
    # Persistent IDs and the letter-to-ID map are never sent in the NEW request.
    direct = [{key: value for key, value in c.items() if key != "object_id"} for c in candidates]
    return {CONDITIONS[0]: old, CONDITIONS[1]: _vlm_prompt(target, direct)}, {**mapping, "N": None}


def header_mask(shape, count):
    mask = np.zeros(shape[:2], dtype=bool)
    for index in range(1, count + 1):
        y, x = (index // COLUMNS) * TILE_SIZE, (index % COLUMNS) * TILE_SIZE
        mask[y:y + HEADER_HEIGHT, x:x + TILE_SIZE] = True
    return mask


def render_direct_sheet(rgb_path, localization_path, frozen_path, destination, mapping):
    """Reuse the existing renderer and prove every non-label pixel is unchanged."""
    original = cv2.imread(str(frozen_path))
    if original is None:
        raise FileNotFoundError(frozen_path)
    payload = json.loads(Path(localization_path).read_text())
    ids = [item["object_id"] for item in payload["objects"]]
    if ids != [value for value in mapping.values() if value is not None]:
        raise ValueError("Frozen crop order differs from deterministic candidate order.")
    labels = {value: key for key, value in mapping.items() if value is not None}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        create_roi_contact_sheet(rgb_path, localization_path, root / "old.png", tile_size_px=TILE_SIZE)
        if not np.array_equal(original, cv2.imread(str(root / "old.png"))):
            raise ValueError("Current renderer does not reproduce the frozen OLD image pixels.")
        renamed = {**payload, "objects": [{**item, "object_id": labels[item["object_id"]]} for item in payload["objects"]]}
        save_json(root / "letters.json", renamed)
        create_roi_contact_sheet(rgb_path, root / "letters.json", root / "new.png", tile_size_px=TILE_SIZE)
        new = cv2.imread(str(root / "new.png"))
        mask = header_mask(original.shape, len(ids))
        if new.shape != original.shape or not np.array_equal(new[~mask], original[~mask]):
            raise ValueError("Non-label image pixels changed.")
        image_bytes = (root / "new.png").read_bytes()
        if destination.exists() and destination.read_bytes() != image_bytes:
            raise ValueError("Saved direct-letter image changed.")
        if not destination.exists():
            destination.write_bytes(image_bytes)
    return {"unchanged_nonlabel_pixels": True, "shape": list(original.shape),
            "nonlabel_pixels_sha256": digest(original[~mask].tobytes()),
            "changed_label_pixels": int(np.any(new != original, axis=2).sum())}


def prepare():
    samples = load_samples()
    for directory in (OUTPUT_ROOT, OUTPUT_ROOT / "inputs", OUTPUT_ROOT / "responses"):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(samples, 1):
        image = SCENE_ROOT / sample["image_path"]
        localization = SCENE_ROOT / sample["localization_path"]
        rgb = image.parent / "rgb.png"
        gt = SCENE_ROOT / sample["evaluation_ground_truth_path"]
        prompts, mapping = prompts_and_map(sample["target_description"], load_localized_objects(localization))
        previous = json.loads((PREVIOUS_ROOT / "responses" / f"{sample['sample_id']}_compact.json").read_text())
        hashes = {"image_sha256": digest(image.read_bytes()), "localization_sha256": digest(localization.read_bytes())}
        if (any(previous[key] != value for key, value in hashes.items())
                or previous["prompt"] != prompts[CONDITIONS[0]] or previous["candidate_map"] != mapping
                or previous["ground_truth_object_id"] != sample["ground_truth_object_id"]
                or previous["partition"] != sample["partition"]):
            raise ValueError("Frozen previous inputs/prompt/mapping/ground truth changed.")
        if json.loads(gt.read_text())["ground_truth_object_id"] != sample["ground_truth_object_id"]:
            raise ValueError("Ground-truth file and manifest disagree.")
        new_image = OUTPUT_ROOT / "inputs" / f"{sample['sample_id']}.png"
        audit = render_direct_sheet(rgb, localization, image, new_image, mapping)
        row = {**sample, "old_image_path": str(image), "new_image_path": str(new_image),
               "localization_path": str(localization), "candidate_map": mapping, "prompts": prompts,
               "visual_audit": audit, **hashes, "rgb_sha256": digest(rgb.read_bytes()),
               "ground_truth_sha256": digest(gt.read_bytes()), "new_image_sha256": digest(new_image.read_bytes())}
        row["input_sha256"] = digest(json.dumps(row, sort_keys=True).encode())
        rows.append(row)
        if index % 50 == 0:
            print(f"Pixel/hash audit: {index}/500 frozen pairs", flush=True)
    settings = request_arguments(b"", "")
    protocol = {"model": MODEL, "settings": {k: v for k, v in settings.items() if k != "messages"},
                "system_prompt": SYSTEM_PROMPT, "image_detail": "high", "conditions": list(CONDITIONS),
                "score": "exp(sum(decision-bearing generated-label logprobs)); no length/candidate normalization",
                "pair_order": "OLD first when (task_index+initial_state_index) even, otherwise NEW first",
                "fresh_calls": 1000, "analysis_plan": ANALYSIS_PLAN,
                "inputs_sha256": digest(json.dumps(rows, sort_keys=True).encode()),
                "source_sha256": {name: digest(Path(name).read_bytes()) for name in (
                    "prompt.py", "vlm_module.py", "scripts/run_vlm_visual_label_ablation.py",
                    "scripts/run_vlm_output_format_ablation.py")}}
    payload = {"protocol": protocol, "samples": rows}
    path = OUTPUT_ROOT / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Frozen ablation protocol changed; refusing to mix runs.")
    if not path.exists():
        save_json(path, payload)
    print("Ready: 500 pixel-audited pairs; 300 calibration / 100 validation / 100 test.", flush=True)


def load_inputs():
    payload = json.loads((OUTPUT_ROOT / "inputs.json").read_text())
    for name, expected in payload["protocol"]["source_sha256"].items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError(f"Protocol source changed: {name}")
    if digest(json.dumps(payload["samples"], sort_keys=True).encode()) != payload["protocol"]["inputs_sha256"]:
        raise ValueError("Input manifest hash mismatch.")
    return payload


def parse_response(raw, mapping):
    output = raw["choices"][0]
    text = output["message"].get("content") or ""
    decision = extract_decision(text, (output.get("logprobs") or {}).get("content") or [], mapping)
    if output["finish_reason"] != "stop":
        decision.update(choice=None, predicted_object_id=None, raw_log_probability=None,
                        raw_sequence_likelihood=None, score_unavailable_reason="unfinished response")
    return {"generated_output_text": text, **decision}


def run_pair(sample):
    from openai import OpenAI

    rows = []
    order = CONDITIONS if (sample["task_index"] + sample["initial_state_index"]) % 2 == 0 else CONDITIONS[::-1]
    for order_index, condition in enumerate(order):
        prefix = "old" if condition == CONDITIONS[0] else "new"
        image = Path(sample[f"{prefix}_image_path"])
        image_bytes = image.read_bytes()
        expected = sample["image_sha256" if prefix == "old" else "new_image_sha256"]
        if digest(image_bytes) != expected or digest(Path(sample["localization_path"]).read_bytes()) != sample["localization_sha256"]:
            raise ValueError("Frozen image/localization changed before request.")
        args = request_arguments(image_bytes, sample["prompts"][condition])
        request_hash = digest(json.dumps(args, sort_keys=True).encode())
        path = OUTPUT_ROOT / "responses" / f"{sample['sample_id']}_{condition}.json"
        if path.exists():
            response = json.loads(path.read_text())
            if response["request_sha256"] != request_hash or response["input_sha256"] != sample["input_sha256"]:
                raise ValueError("Resume request/input mismatch.")
        else:
            with OpenAI(timeout=120, max_retries=3) as client:
                raw = paced_completion(client, args).model_dump(mode="json", exclude_none=True)
            response = {"request_sha256": request_hash, "input_sha256": sample["input_sha256"],
                        "condition": condition, "captured_at": datetime.now(timezone.utc).isoformat(),
                        "pair_order_index": order_index, "provider_response": raw}
            save_json(path, response)  # Checkpoint the real response before parsing it.
        raw = response["provider_response"]
        if raw["model"] != MODEL:
            raise ValueError("Returned model differs from the frozen snapshot.")
        decision = parse_response(raw, sample["candidate_map"])
        row = {key: sample[key] for key in ("sample_id", "partition", "task_index", "initial_state_index",
                                            "target_description", "ground_truth_object_id", "input_sha256")}
        row.update(condition=condition, model=raw["model"], provider="openai", **decision,
                   correct=decision["choice"] is not None and decision["predicted_object_id"] == sample["ground_truth_object_id"],
                   response_path=str(path), response_sha256=digest(path.read_bytes()),
                   image_sha256=expected, score_type="raw_label_likelihood")
        rows.append(row)
    return rows


def run(stage):
    if stage == "check":
        return prepare()
    if stage == "analyze":
        from scripts.analyze_vlm_visual_label_ablation import analyze
        return analyze()
    if stage not in ("pilot", "full"):
        raise ValueError("Use check, pilot, full, or analyze.")
    payload = load_inputs()
    samples = payload["samples"]
    if stage == "pilot":
        samples = [s for s in samples if s["task_index"] in (0, 5) and s["initial_state_index"] == 0]
    # Interleave targets/states; do not run all OLD and then all NEW.
    samples = sorted(samples, key=lambda s: (s["initial_state_index"], s["task_index"]))
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_pair, s) for s in samples]
        for count, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            results.extend(rows)
            print(f"{stage} {count}/{len(samples)} pairs | " + " | ".join(
                f"{r['condition']} {r['sample_id']} {r['choice']} correct={r['correct']} p={r[SCORE]}" for r in rows), flush=True)
    results.sort(key=lambda r: (r["sample_id"], r["condition"]))
    save_json(OUTPUT_ROOT / f"{stage}_records.json", {"protocol": payload["protocol"], "records": results})
    print(f"Saved {len(results)} compact records. Raw responses remain immutable on resume.", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) == 2 else "check")
