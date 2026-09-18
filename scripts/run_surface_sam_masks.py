"""SAM3-assisted masks from identity-free localized RGB crops (Windows SDK env)."""

import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/surface_verification_20260917/sam3"
DATASET = ROOT / "experiments/wrist_association_2026-09-11"
MASKS = ROOT / "outputs/ablation_1_wrist_association_2026-09-11/20260912T153218768705Z/masked/masks"


def decode_rle(rle):
    """COCO compressed RLE, Fortran order; validate decoded pixel count."""
    encoded = rle["counts"]
    counts, p = [], 0
    if isinstance(encoded, list):
        counts = encoded
    else:
        while p < len(encoded):
            value, shift = 0, 0
            while True:
                c = ord(encoded[p]) - 48
                p += 1
                value |= (c & 31) << shift
                shift += 5
                if not c & 32:
                    if c & 16:
                        value |= -1 << shift
                    break
            if len(counts) > 2:
                value += counts[-2]
            counts.append(value)
    assert min(counts) >= 0 and sum(counts) == np.prod(rle["size"])
    return np.repeat(np.arange(len(counts)) % 2, counts).reshape(rle["size"], order="F").astype(bool)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=os.environ["ROBOFLOW_API_KEY"])
    client.configure(InferenceConfiguration(api_key_transport="header", workflow_run_retries_enabled=False))
    rgb = cv2.imread(str(DATASET / "scene/rgb.png"))
    records = []
    for path in sorted(MASKS.glob("*.png")):
        oid = path.stem
        mask = cv2.imread(str(path), 0) > 0
        y, x = np.nonzero(mask)
        padding = round(.25 * max(np.ptp(x) + 1, np.ptp(y) + 1))
        x1, y1 = max(0, x.min() - padding), max(0, y.min() - padding)
        x2, y2 = min(rgb.shape[1], x.max() + padding + 1), min(rgb.shape[0], y.max() + padding + 1)
        crop_path = OUTPUT / f"{oid}_request.png"
        cv2.imwrite(str(crop_path), rgb[y1:y2, x1:x2])
        response_path = OUTPUT / f"{oid}_response.json"
        start = perf_counter()
        cached = response_path.exists()
        if cached:
            predictions = json.loads(response_path.read_text())
        else:
            response = client.run_workflow(workspace_name="gearassemblydetection", workflow_id="general-segmentation-api",
                images={"image": str(crop_path)}, parameters={"classes": ["object"]}, use_cache=True, disable_sinks=True)
            predictions = response[0]["predictions"]["predictions"]
            response_path.write_text(json.dumps(predictions, indent=2))
        options = [decode_rle(p["rle_mask"]) for p in predictions]
        reference = mask[y1:y2, x1:x2]
        ious = [float((m & reference).sum() / (m | reference).sum()) for m in options]
        selected = int(np.argmax(ious)) if ious and max(ious) > 0 else None
        if selected is not None:
            result = np.zeros(mask.shape, np.uint8)
            result[y1:y2, x1:x2] = options[selected] * 255
            cv2.imwrite(str(OUTPUT / f"{oid}_mask.png"), result)
        row = {"object_id": oid, "prompt": "object", "crop_xyxy_exclusive": list(map(int, [x1, y1, x2, y2])),
               "prediction_count": len(options), "selected_index": selected, "overlap_ious": ious,
               "selection": "maximum overlap with localized mask; no identity, class or confidence use",
               "response_reused": cached, "elapsed_s": perf_counter() - start}
        records.append(row)
        print(oid, len(options), "masks; selected", selected, flush=True)
    (OUTPUT / "index.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
