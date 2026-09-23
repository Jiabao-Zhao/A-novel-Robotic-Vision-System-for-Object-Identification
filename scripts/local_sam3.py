"""Offline, text-free SAM3 point-grid proposals on the local CUDA GPU."""

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
from transformers.pipelines.mask_generation import MaskGenerationPipeline


MODEL_DIR = Path("D:/AI/models/sam3" if os.name == "nt" else "/mnt/d/AI/models/sam3")
SETTINGS = dict(points_per_side=32, points_per_batch=4, pred_iou_thresh=.88,
                box_nms_thresh=.7, crop_n_layers=0)
# The downloaded checkpoint contains both the concept detector and visual tracker.
TRACKER_KEYS = {r"tracker_model.(.+)": r"\1",
                "detector_model.vision_encoder.backbone.": "vision_encoder.backbone.",
                "tracker_neck.": "vision_encoder.neck."}


def model_provenance():
    files = [MODEL_DIR / name for name in ("config.json", "processor_config.json", "model.safetensors")]
    files.append(Path(__file__))
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"model": "facebook/sam3", "component": "Sam3TrackerModel",
            "model_directory": str(MODEL_DIR), "sha256": hashes,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "precision": "bfloat16", "device": "cuda", "local_files_only": True,
            "text_prompts": None, "proposal_policy": "uniform positive point grid",
            "score_definition": "SAM3 tracker predicted mask IoU, not identity confidence",
            "stability_score": "filtered internally; per-mask values unavailable"}


class _MaskPipeline(MaskGenerationPipeline):
    def postprocess(self, model_outputs, **kwargs):
        # Transformers 5.5.4 casts NMS boxes to FP32 but leaves scores in BF16.
        for output in model_outputs:
            output["iou_scores"] = output["iou_scores"].float()
        return super().postprocess(model_outputs, **kwargs)


def proposal_records(result, shape, stability):
    records = []
    for value, score in zip(result["masks"], result["scores"], strict=True):
        mask = np.asarray(value, dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError(f"SAM3 mask shape {mask.shape} differs from input {shape}")
        y, x = np.nonzero(mask)
        if not len(x):
            continue
        records.append({"segmentation": mask,
                        "bbox": [int(x.min()), int(y.min()),
                                 int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)],
                        "predicted_iou": float(score), "stability_score": None,
                        "stability_threshold": float(stability)})
    return records


class Sam3AutomaticMasks:
    def __init__(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Local SAM3 requires the WSL CUDA/BF16 Python environment.")
        model, info = Sam3TrackerModel.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa", key_mapping=TRACKER_KEYS, output_loading_info=True)
        failures = {key: list(info.get(key, [])) for key in ("missing_keys", "mismatched_keys", "error_msgs")}
        if any(failures.values()):
            raise RuntimeError(f"Incomplete SAM3 checkpoint load: {failures}")
        model = model.eval().requires_grad_(False).cuda()
        processor = Sam3TrackerProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.pipeline = _MaskPipeline(model=model, image_processor=processor.image_processor,
                                      device=0, dtype=torch.bfloat16)

    def generate(self, rgb, *, stability_score_thresh=.85, **settings):
        options = {**SETTINGS, **settings}
        grid, batch = options["points_per_side"], options["points_per_batch"]
        if grid <= 0 or batch <= 0 or grid * grid % batch:
            raise ValueError("Positive point grid must divide evenly into prompt batches.")
        if options["crop_n_layers"] != 0:
            raise ValueError("This validated SAM3 proposal policy uses one full-image grid.")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("SAM3 input must be an HxWx3 uint8 RGB image.")
        with torch.inference_mode():
            result = self.pipeline(
                Image.fromarray(rgb), points_per_crop=grid, points_per_batch=batch,
                pred_iou_thresh=options["pred_iou_thresh"],
                stability_score_thresh=stability_score_thresh,
                crops_nms_thresh=options["box_nms_thresh"], crops_n_layers=0)
        return proposal_records(result, rgb.shape[:2], stability_score_thresh)
