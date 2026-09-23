"""Run SAM3 segmentation on the user's uploaded figure, without CAD matching."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError
from PIL import Image, ImageDraw, ImageFont

from scripts.run_surface_sam_masks import decode_rle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("C:/Users/JIABAO~1/AppData/Local/Temp/codex-clipboard-6ab21a6a-c07c-4c03-9e11-4577141dae83.png")
OUTPUT = ROOT / "outputs/sam3_uploaded_scene_20260921"
CROP_XYXY = (24, 30, 452, 350)


def render_results():
    rgb = np.asarray(Image.open(OUTPUT / "input.png").convert("RGB"))
    automatic = json.loads((OUTPUT / "response.json").read_text())
    prompted = json.loads((OUTPUT / "items_response.json").read_text())
    predictions = {"automatic": automatic["predictions"],
                   "items": [p for group in prompted["prompt_results"] for p in group["predictions"]]}
    overlays, rows, individual = {}, {}, []
    for mode, detections in predictions.items():
        folder = OUTPUT / mode
        folder.mkdir(exist_ok=True)
        overlay = rgb.copy()
        rows[mode] = []
        for index, prediction in enumerate(detections, 1):
            if prediction["format"] == "polygon":
                mask = np.zeros(rgb.shape[:2], np.uint8)
                # Each polygon is a component of this one returned prediction.
                for polygon in prediction["masks"]:
                    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
                mask = mask.astype(bool)
            else:
                assert prediction["format"] == "rle"
                mask = decode_rle(prediction["masks"])
            assert mask.shape == rgb.shape[:2] and mask.any()
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"mask_{index:02}.png")
            color = cv2.cvtColor(np.uint8([[[index * 37 % 180, 200, 250]]]), cv2.COLOR_HSV2RGB)[0, 0]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            overlay[mask] = (.55 * overlay[mask] + .45 * color).astype(np.uint8)
            cv2.drawContours(overlay, contours, -1, tuple(map(int, color)), 1)
            yy, xx = np.nonzero(mask)
            point = (int(xx.mean()), int(yy.mean()))
            cv2.putText(overlay, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
            cv2.putText(overlay, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            rows[mode].append({"mask_id": index, "confidence": prediction["confidence"],
                               "pixels": int(mask.sum())})
            if mode == "items":
                tile = (rgb * .22).astype(np.uint8)
                tile[mask] = rgb[mask]
                cv2.drawContours(tile, contours, -1, tuple(map(int, color)), 1)
                individual.append(tile)
        Image.fromarray(overlay).save(folder / "overlay.png")
        overlays[mode] = overlay
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.exists():
        font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    title, text = [ImageFont.truetype(str(font), size) for size in (27, 21)]
    panel = Image.new("RGB", (1344, 490), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    draw.text((15, 12), "SAM3 on the uploaded image: automatic versus generic text prompt", font=title, fill="#172531")
    for i, (label, pixels) in enumerate((("RGB input: 428 x 320", rgb),
                                        (f"No prompt: {len(rows['automatic'])} returned mask", overlays["automatic"]),
                                        (f"Prompt 'items': {len(rows['items'])} masks at 0.50", overlays["items"]))):
        x = 15 + i * 448
        draw.text((x, 60), label, font=text, fill="#172531")
        panel.paste(Image.fromarray(pixels), (x, 98))
    draw.text((15, 441), "No CADs, depth, object names, boxes or points supplied. Generic text prompting is a separate test.",
              font=text, fill="#405263")
    panel.save(OUTPUT / "comparison.png")
    if individual:
        sheet = Image.new("RGB", (1320, 55 + 360 * ((len(individual) + 2) // 3)), "#f4f6f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 12), "All masks returned for the generic prompt 'items'", font=title, fill="#172531")
        for i, tile in enumerate(individual):
            x, y = (i % 3) * 440 + 6, 55 + (i // 3) * 360
            draw.text((x + 5, y), f"Mask {i + 1} | confidence {rows['items'][i]['confidence']:.3f}",
                      font=text, fill="#172531")
            sheet.paste(Image.fromarray(tile), (x, y + 31))
        sheet.save(OUTPUT / "individual_masks.png")
    result = {"input_size_wh": list(reversed(rgb.shape[:2])), "mask_records": rows,
              "automatic_request_s": json.loads((OUTPUT / "request.json").read_text())["elapsed_request_s"],
              "items_request_s": json.loads((OUTPUT / "items_request.json").read_text())["elapsed_request_s"],
              "automatic_server_time_s": automatic["time"], "items_server_time_s": prompted["time"],
              "accuracy": None, "accuracy_note": "No reference masks for this supplied screenshot"}
    (OUTPUT / "mask_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    OUTPUT.mkdir(exist_ok=True)
    image = Image.open(SOURCE).convert("RGB").crop(CROP_XYXY)
    image.save(OUTPUT / "input.png")
    if (OUTPUT / "response.json").exists() and (OUTPUT / "items_response.json").exists():
        assert hashlib.sha256((OUTPUT / "input.png").read_bytes()).hexdigest() == json.loads(
            (OUTPUT / "request.json").read_text())["input_sha256"]
        render_results()
        return
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com",
                                 api_key=os.environ["ROBOFLOW_API_KEY"])
    client.configure(InferenceConfiguration(api_key_transport="header",
                     workflow_run_retries_enabled=False))
    metadata = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "crop_xyxy_exclusive": CROP_XYXY, "input_size_wh": image.size,
                "input_sha256": hashlib.sha256((OUTPUT / "input.png").read_bytes()).hexdigest(),
                "sdk_version": importlib.metadata.version("inference-sdk"),
                "endpoint": "/sam3/visual_segment", "prompts": None,
                "multimask_output": True, "ground_truth_used": False,
                "cad_matching": False, "status": "requesting",
                "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (OUTPUT / "request.json").write_text(json.dumps(metadata, indent=2))
    started = perf_counter()
    try:
        response = client.sam3_visual_segment(str(OUTPUT / "input.png"), prompts=None,
                                             multimask_output=True, mask_input_format="json")
    except HTTPCallErrorError as error:
        metadata.update(status="failed", http_status=error.status_code,
                        elapsed_request_s=perf_counter() - started,
                        message=(error.api_message or "").replace(os.environ["ROBOFLOW_API_KEY"], "[redacted]"))
        (OUTPUT / "request.json").write_text(json.dumps(metadata, indent=2))
        raise SystemExit(f"SAM3 request failed (HTTP {error.status_code}). No masks returned.") from None
    metadata["status"] = "completed"
    metadata["elapsed_request_s"] = perf_counter() - started
    (OUTPUT / "response.json").write_text(json.dumps(response))
    (OUTPUT / "request.json").write_text(json.dumps(metadata, indent=2))
    started = perf_counter()
    items = client.sam3_concept_segment(str(OUTPUT / "input.png"),
        prompts=[{"type": "text", "text": "items"}], output_prob_thresh=.5, format="rle")
    (OUTPUT / "items_response.json").write_text(json.dumps(items))
    (OUTPUT / "items_request.json").write_text(json.dumps({
        "endpoint": "/sam3/concept_segment", "prompt": "items", "confidence_threshold": .5,
        "nms_iou_threshold": None, "format": "rle", "elapsed_request_s": perf_counter() - started,
        "input_sha256": metadata["input_sha256"], "cad_matching": False,
        "manual_boxes_or_points": False}, indent=2))
    render_results()


if __name__ == "__main__":
    main()
