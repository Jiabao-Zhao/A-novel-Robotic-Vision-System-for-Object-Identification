"""Local SAM3 point-grid proposals, retaining SAM-6D resize/size filtering."""

from time import perf_counter

from scripts import run_bop_localization_benchmark as baseline
from scripts.local_sam3 import Sam3AutomaticMasks, model_provenance, SETTINGS as SAM3_SETTINGS
import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = baseline.ROOT
OUTPUT = ROOT / "outputs/bop_automatic_sam3_local_20260921"
SETTINGS = dict(**SAM3_SETTINGS, stability_score_thresh=.85)
IMAGE_WIDTH = 640
MIN_BOX_SIZE = .05
MIN_MASK_SIZE = 3e-4
read_json, save_json = baseline.read_json, baseline.save_json


def restore_proposal(prediction, native_shape, resized_shape):
    """SAM-6D bilinear resize, size filters, and >0 binary-mask export."""
    mask = torch.from_numpy(np.asarray(prediction["segmentation"], dtype=bool))
    soft = F.interpolate(mask[None, None].float(), size=native_shape,
                         mode="bilinear", align_corners=False)[0, 0].numpy()
    x, y, w, h = prediction["bbox"]
    box = np.array([x, y, x + w, y + h]) * (native_shape[1] / resized_shape[1])
    box[[0, 2]] = np.clip(box[[0, 2]], 0, native_shape[1])
    box[[1, 3]] = np.clip(box[[1, 3]], 0, native_shape[0])
    box = box.astype(int)  # Upstream Detections casts resized boxes to long.
    area = native_shape[0] * native_shape[1]
    retained = ((box[2] - box[0]) * (box[3] - box[1]) / area > MIN_BOX_SIZE ** 2
                and float(soft.sum()) / area > MIN_MASK_SIZE)
    return soft > 0, bool(retained), box.tolist()


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = read_json(baseline.OUTPUT / "protocol.json")["plan"]
    provenance = model_provenance()
    protocol = {"plan": plan, "model": "local SAM3 tracker", "device": "cuda",
        "text_prompts": None, "settings": SETTINGS, "input_width": IMAGE_WIDTH,
        "min_box_size": MIN_BOX_SIZE, "min_mask_size": MIN_MASK_SIZE,
        "batch_note": "4 prompts per batch instead of 64 for GPU memory; same 32x32 grid",
        "native_mask": "SAM-6D bilinear resize align_corners=False; binary mask >0",
        "sources": ["https://github.com/JiehongLin/SAM-6D/blob/main/SAM-6D/Instance_Segmentation_Model/model/sam.py",
                    "https://github.com/JiehongLin/SAM-6D/blob/main/SAM-6D/Instance_Segmentation_Model/configs/model/segmentor_model/sam.yaml"],
        "sam_source": provenance,
        "script_sha256": baseline.association.content_hash(__file__),
        "rgb_sha256": {baseline.frame_folder(i).name: baseline.association.content_hash(
            baseline.DATA / "test_primesense" / f"{i['scene_id']:06}/rgb/{i['im_id']:06}.png") for i in plan}}
    path = OUTPUT / "sam_protocol.json"
    if path.exists():
        assert read_json(path) == protocol, "Frozen SAM generation protocol changed"
    else:
        save_json(path, protocol)
    generator = None
    for item in plan:
        name = baseline.frame_folder(item).name
        folder = OUTPUT / "sam" / name
        if (folder / "proposals.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        for subdir in ("raw_masks", "masks"):
            (folder / subdir).mkdir(exist_ok=True)
        rgb, _, _, _, _ = baseline.frame_input(item)
        resized = cv2.resize(rgb.copy(), (IMAGE_WIDTH, int(IMAGE_WIDTH * rgb.shape[0] / rgb.shape[1])))
        baseline.save_image(folder / "sam_input.png", resized)
        if generator is None:
            start = perf_counter()
            generator = Sam3AutomaticMasks()
            save_json(OUTPUT / "sam_model.json", {"load_s": perf_counter() - start,
                "gpu": torch.cuda.get_device_name(), "provenance": provenance})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        with torch.inference_mode():
            generated = generator.generate(resized, **SETTINGS)
        torch.cuda.synchronize()
        generation_s = perf_counter() - start
        raw_records, proposals = [], []
        for index, prediction in enumerate(generated, 1):
            mask, retained, box = restore_proposal(prediction, rgb.shape[:2], resized.shape[:2])
            oid = f"sam_{index:03}"
            record = {"object_id": oid, "pixels": int(mask.sum()), "retained": retained,
                      "predicted_iou": float(prediction["predicted_iou"]),
                      "stability_score": prediction["stability_score"],
                      "stability_threshold": prediction["stability_threshold"], "bbox_xyxy": box}
            raw_records.append(record)
            baseline.save_image(folder / "raw_masks" / f"{oid}.png", mask.astype(np.uint8) * 255)
            if retained:
                proposals.append(record)
                baseline.save_image(folder / "masks" / f"{oid}.png", mask.astype(np.uint8) * 255)
        save_json(folder / "raw_proposals.json", raw_records)
        save_json(folder / "runtime.json", {"generation_s": generation_s,
            "generation_and_export_s": perf_counter() - start, "raw_count": len(generated),
            "retained_count": len(proposals), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
        save_json(folder / "proposals.json", proposals)
        print(name, "raw", len(generated), "retained", len(proposals), f"{generation_s:.1f}s", flush=True)
    assert baseline.association.content_hash(__file__) == protocol["script_sha256"]


if __name__ == "__main__":
    main()
