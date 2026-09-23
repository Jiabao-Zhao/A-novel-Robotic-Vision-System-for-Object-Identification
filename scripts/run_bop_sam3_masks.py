"""SAM3 proposals for the frozen 20-frame T-LESS pilot (Windows SDK env).

Uses the same generic-object workflow as the workbench check. No CAD names,
ground-truth masks, instance counts, or point-cloud crops enter the API call.
"""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from scripts.run_surface_sam_masks import decode_rle


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/bop_localization_20260917"
OUTPUT = ROOT / "outputs/bop_sam_depth_20260919"
DATA = ROOT / "outputs/cache/bop_tless/data/tless/test_primesense"
WORKFLOW = {
    "version": "1.0", "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [{"type": "roboflow_core/sam3@v3", "name": "sam_3",
               "images": "$inputs.image", "model_id": "sam3/sam3_final",
               "class_names": ["object"], "threshold": .5}],
    "outputs": [{"type": "JsonField", "name": "predictions",
                 "selector": "$steps.sam_3.predictions"}],
}


def main():
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "sam3_workflow.json"
    if path.exists():
        assert json.loads(path.read_text()) == WORKFLOW
    else:
        path.write_text(json.dumps(WORKFLOW, indent=2))
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com",
                                 api_key=os.environ["ROBOFLOW_API_KEY"])
    client.configure(InferenceConfiguration(api_key_transport="header", workflow_run_retries_enabled=False))
    plan = json.loads((BASE / "protocol.json").read_text())["plan"]
    for item in plan:
        scene, frame = item["scene_id"], item["im_id"]
        source = DATA / f"{scene:06}/rgb/{frame:06}.png"
        folder = OUTPUT / "sam3" / f"scene_{scene:06}_{frame:06}"
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        response_path, metadata_path = folder / "response.json", folder / "request.json"
        if response_path.exists():
            assert json.loads(metadata_path.read_text())["image_sha256"] == digest
            response = json.loads(response_path.read_text())
        else:
            start = perf_counter()
            response = client.run_workflow(specification=WORKFLOW, images={"image": str(source)},
                use_cache=True, disable_sinks=True)
            metadata_path.write_text(json.dumps({"image_sha256": digest,
                "elapsed_request_s": perf_counter() - start, "scene_id": scene, "im_id": frame,
                "input": "Full RGB; generic object prompt only; no annotations or CAD input"}, indent=2))
            response_path.write_text(json.dumps(response))
        shape = cv2.imread(str(source)).shape[:2]
        predictions = response[0]["predictions"]["predictions"]
        (folder / "masks").mkdir(exist_ok=True)
        records = []
        for index, prediction in enumerate(predictions, 1):
            mask = decode_rle(prediction["rle_mask"])
            assert mask.shape == shape and mask.any()
            oid = f"sam_{index:03}"
            cv2.imwrite(str(folder / "masks" / f"{oid}.png"), mask.astype(np.uint8) * 255)
            records.append({"object_id": oid, "pixels": int(mask.sum()),
                            "sam_confidence_diagnostic": prediction.get("confidence")})
        (folder / "proposals.json").write_text(json.dumps(records, indent=2))
        print(f"SAM3 scene {scene:02}: {len(records)} masks", flush=True)


if __name__ == "__main__":
    main()
