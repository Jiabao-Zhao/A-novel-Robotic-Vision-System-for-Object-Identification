"""Identity-free SAM3 check on the saved workbench image; no CAD matching."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "outputs/simulation/industrial_workbench/20260919T201225Z"
OUTPUT = SCENE / "sam3_segmentation"


def main():
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    from scripts.run_surface_sam_masks import decode_rle

    OUTPUT.mkdir(exist_ok=True)
    image_path = SCENE / "wrist_rgb.png"
    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    response_path = OUTPUT / "response.json"
    if response_path.exists():
        response = json.loads(response_path.read_text())
    else:
        client = InferenceHTTPClient(api_url="https://serverless.roboflow.com",
                                     api_key=os.environ["ROBOFLOW_API_KEY"])
        client.configure(InferenceConfiguration(api_key_transport="header",
                         workflow_run_retries_enabled=False))
        # Freeze the existing SAM3 workflow block; omit visualization-only steps.
        specification = {
            "version": "1.0",
            "inputs": [{"type": "InferenceImage", "name": "image"}],
            "steps": [{"type": "roboflow_core/sam3@v3", "name": "sam_3",
                       "images": "$inputs.image", "model_id": "sam3/sam3_final",
                       "class_names": ["object"], "threshold": 0.5}],
            "outputs": [{"type": "JsonField", "name": "predictions",
                         "selector": "$steps.sam_3.predictions"}],
        }
        (OUTPUT / "workflow.json").write_text(json.dumps(specification, indent=2))
        started = perf_counter()
        response = client.run_workflow(specification=specification,
            images={"image": str(image_path)}, use_cache=True, disable_sinks=True)
        elapsed = perf_counter() - started
        response_path.write_text(json.dumps(response))
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_shape": list(rgb.shape), "prompt": "object", "threshold": .5,
            "sdk_version": importlib.metadata.version("inference-sdk"),
            "model_id": "sam3/sam3_final", "request_elapsed_s": elapsed,
            "input": "Full wrist RGB only; no CADs, names, boxes, depth, or masks",
            "identity_prediction": False, "api_key_transport": "header"}
        (OUTPUT / "run.json").write_text(json.dumps(record, indent=2))
    predictions = response[0]["predictions"]["predictions"]
    overlay = cv2.resize(rgb, (1200, 1200))
    rows = []
    for i, prediction in enumerate(predictions, 1):
        mask = decode_rle(prediction["rle_mask"])
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Unexpected mask size: {mask.shape}")
        cv2.imwrite(str(OUTPUT / f"mask_{i:02d}.png"), mask.astype(np.uint8) * 255)
        small = cv2.resize(mask.astype(np.uint8), (1200, 1200), interpolation=cv2.INTER_NEAREST) > 0
        color = cv2.cvtColor(np.uint8([[[i * 37 % 180, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        overlay[small] = (overlay[small] * .55 + color * .45).astype(np.uint8)
        contours, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(map(int, color)), 2)
        y, x = np.nonzero(small)
        cv2.putText(overlay, str(i), (int(x.mean()), int(y.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 4)
        cv2.putText(overlay, str(i), (int(x.mean()), int(y.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        rows.append({"mask_id": i, "confidence": prediction.get("confidence"),
                     "area_pixels": int(mask.sum())})
    cv2.imwrite(str(OUTPUT / "mask_overlay.png"), overlay)
    (OUTPUT / "masks.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"masks_returned": len(rows), "output": str(OUTPUT)}))


def evaluate():
    """Compare saved predictions to simulator masks, used only for evaluation."""
    import mujoco
    from scipy.ndimage import binary_fill_holes
    from scripts.run_wrist_appearance_consistency import visible_mask
    from simulation.industrial_workbench import configure_scene
    from simulation.wrist_cluster import make_environment, CAMERA
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    source = json.loads((SCENE / "scene.json").read_text())
    rgb = cv2.imread(str(SCENE / "wrist_rgb.png"))
    size = rgb.shape[0]
    catalog, placements = source["catalog"], source["placements"]
    reference_dir = OUTPUT / "evaluation_masks"
    reference_dir.mkdir(exist_ok=True)
    with make_environment(catalog, placements,
            additional_scene=lambda model: configure_scene(model, catalog, placements)) as env:
        env.reset()
        sim = env.sim
        sim.set_state_from_flattened(np.load(SCENE / "initial_state.npy"))
        # As in wrist_oracle_pose.recover_capture, invert the final Euler
        # position update: the image used geometry preceding that update.
        assert int(sim.model.opt.integrator) == 0
        qpos = sim.data.qpos.copy()
        timestep = float(sim.model.opt.timestep)
        mujoco.mj_integratePos(sim.model._model, qpos, sim.data.qvel, -timestep)
        sim.data.qpos[:] = qpos
        sim.forward()
        replay = sim.render(width=size, height=size, camera_name=CAMERA)[::-1]
        np.testing.assert_array_equal(get_camera_extrinsic_matrix(sim, CAMERA),
                                      np.load(SCENE / "world_T_camera.npy"))
        np.testing.assert_array_equal(get_camera_intrinsic_matrix(sim, CAMERA, size, size),
                                      np.load(SCENE / "intrinsics.npy"))
        np.testing.assert_array_equal(replay, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        samples = sim.model.vis.quality.offsamples
        sim.model.vis.quality.offsamples = 0
        mask_context = mujoco.MjrContext(sim.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        sim.model.vis.quality.offsamples = samples
        assert mask_context.offSamples == 0
        try:
            for name in catalog:
                body = sim.model.body_name2id(f"nist_part_{name}")
                geoms = [g for g in range(sim.model.ngeom)
                         if sim.model.geom_bodyid[g] == body and sim.model.geom_group[g] == 1]
                assert len(geoms) == 1
                mask = visible_mask(sim, geoms[0], mask_context, image_size=size)
                cv2.imwrite(str(reference_dir / f"{name}.png"), mask.astype(np.uint8) * 255)
        finally:
            mask_context.free()
    predictions = [cv2.imread(str(p), 0) > 0 for p in sorted(OUTPUT.glob("mask_[0-9][0-9].png"))]
    areas = np.array([m.sum() for m in predictions])
    rows, details = [], {}
    for name in catalog:
        gt = cv2.imread(str(reference_dir / f"{name}.png"), 0) > 0
        intersections = np.array([np.count_nonzero(m & gt) for m in predictions])
        ious = intersections / (areas + gt.sum() - intersections)
        best = int(ious.argmax())
        mask = predictions[best]
        holes = binary_fill_holes(gt) & ~gt
        row = {"evaluation_object": name, "best_mask_id": best + 1,
               "iou": float(ious[best]), "pixel_recall": float(intersections[best] / gt.sum()),
               "pixel_precision": float(intersections[best] / areas[best]),
               "visible_hole_pixels": int(holes.sum()),
               "hole_pixels_incorrectly_filled": int(np.count_nonzero(holes & mask))}
        rows.append(row)
        y, x = np.nonzero(gt | mask)
        pad = max(12, round(.1 * max(np.ptp(x), np.ptp(y))))
        crop = np.s_[max(0, y.min()-pad):min(size, y.max()+pad+1),
                     max(0, x.min()-pad):min(size, x.max()+pad+1)]
        error = rgb[crop].copy()
        g, m = gt[crop], mask[crop]
        error[m & ~g] = (0, 0, 255)  # Red: background included by SAM3.
        error[g & ~m] = (255, 0, 0)  # Blue: object pixels missed by SAM3.
        details[name] = (rgb[crop], error)
    assigned = {r["best_mask_id"] for r in rows if r["iou"] >= .5}
    summary = {"scope": "Single-image class-agnostic mask check, not benchmark AP",
        "rgb_and_camera_replay_exact": True, "ground_truth_sent_to_sam3": False,
        "position_update_reversed_s": timestep,
        "mask_id_render_antialias_samples": 0, "objects": len(rows),
        "predictions": len(predictions), "mean_best_iou": float(np.mean([r["iou"] for r in rows])),
        "objects_with_iou_at_least_0_5": sum(r["iou"] >= .5 for r in rows),
        "objects_with_iou_at_least_0_75": sum(r["iou"] >= .75 for r in rows),
        "distinct_masks_at_iou_0_5": len(assigned),
        "unmatched_prediction_ids_at_iou_0_5": sorted(set(range(1, len(predictions)+1)) - assigned),
        "per_object": rows}
    (OUTPUT / "evaluation.json").write_text(json.dumps(summary, indent=2))
    worst = sorted(rows, key=lambda r: r["iou"])[:4]
    canvas = np.full((360, 1120, 3), 245, np.uint8)
    cv2.putText(canvas, "Red: extra mask pixels    Blue: missed object pixels", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 30, 30), 1)
    for i, row in enumerate(worst):
        name = row["evaluation_object"]
        for j, crop in enumerate(details[name]):
            scale = min(126/crop.shape[1], 245/crop.shape[0])
            resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            x, y = i*280 + j*135 + 6, 60
            canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
        cv2.putText(canvas, name, (i*280+6, 323), cv2.FONT_HERSHEY_SIMPLEX, .44, (20, 20, 20), 1)
        cv2.putText(canvas, f"Mask {row['best_mask_id']}  IoU {row['iou']:.3f}", (i*280+6, 345),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
    cv2.imwrite(str(OUTPUT / "boundary_details.png"), canvas)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
