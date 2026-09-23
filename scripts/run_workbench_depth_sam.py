"""Paper-equation RGB/depth SAM classification on the saved industrial workbench.

Run in the existing WSL GPU environment. No localization, pose refinement,
text prompts, oracle proposal selection, or production association changes.
"""

import gc
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

import cad_object_association as association
from scripts.local_sam3 import Sam3AutomaticMasks, model_provenance, SETTINGS as SAM_SETTINGS
from scripts.run_wrist_blenderproc_semantic import render_templates, prepare_rgba
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.surface_verification import CadSurface, bbox, box_iou, depth_points


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "outputs/simulation/industrial_workbench/20260919T201225Z"
OUTPUT = ROOT / "outputs/workbench_depth_sam3_local_20260921"
EO_THRESHOLD = .70
EVALUATION_IOU = .50
MODEL = "dinov2_vitl14"


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def inverse_depth_rgb(depth):
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("No valid depth.")
    inverse = 1. / (depth[valid].astype(np.float64) + 1e-6)
    if np.ptp(inverse) == 0:
        raise ValueError("Inverse-depth min/max normalization is undefined.")
    gray = np.zeros(depth.shape, np.uint8)
    gray[valid] = np.floor(255 * (inverse - inverse.min()) / np.ptp(inverse)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def paper_visual_scores(observed_cls, observed_pool, template_cls, template_pool):
    """Eqs. 10-15: maximum CLS view, then cosine of mean-pooled patch features."""
    cosines = np.clip(template_cls @ observed_cls, -1., 1.)
    view = int(cosines.argmax())
    return cosines, view, float(np.clip(template_pool[view] @ observed_pool, -1., 1.))


def paper_score(semantic, appearance, geometry, valid_ratio):
    return (semantic + appearance + valid_ratio * geometry) / (2 + valid_ratio)


def select_candidates(rows, method):
    """Eqs. 19-25; deterministic IDs resolve exact ties without using identity."""
    ordered = sorted(rows, key=lambda r: (-r["score"], r["candidate_id"]))
    if method in ("rgb_only", "depth_only"):
        return [r for r in ordered if r["branch"] == method.split("_")[0]], False
    depth = [r for r in ordered if r["branch"] == "depth"]
    if depth and depth[0]["score"] >= EO_THRESHOLD:
        return depth, True
    return ordered, False


def prepare_templates(catalog):
    folder = OUTPUT / "templates42"
    folder.mkdir(exist_ok=True)
    library = []
    for name, item in catalog.items():
        source = Path(item["cad_path"])
        if association.content_hash(source) != item["cad_sha256"]:
            raise ValueError(f"CAD changed: {name}")
        textured = "visual_mesh_path" in item
        path = Path(item["visual_mesh_path"]) if textured else folder / f"{name}.ply"
        if not path.exists():
            mesh = o3d.io.read_triangle_mesh(str(source))
            if not o3d.io.write_triangle_mesh(str(path), mesh):
                raise RuntimeError(f"CAD export failed: {name}")
        library.append({"cad_id": name, "file_path": str(path.relative_to(ROOT)),
                        "base_color_rgb": item["visual_rgba"][:3],
                        "preserve_materials": textured, "texture_vflip": textured})
    if not (folder / "render_job.json").exists():
        shutil.copyfile(ROOT / "outputs/cache/sam6d_render_source/cam_poses_level0.npy",
                        folder / "cam_poses_level0.npy")
        save_json(folder / "render_job.json", {"output": str(folder), "render_device": "GPU",
            "cad_library": [{**r, "absolute_mesh_path": str(ROOT / r["file_path"])} for r in library]})
    render_templates(folder, library)
    inputs = folder / "inputs"
    inputs.mkdir(exist_ok=True)
    paths = {}
    for name in catalog:
        for v in range(1, 43):
            path = inputs / f"{name}_{v:02}.png"
            if not path.exists():
                rgba = cv2.cvtColor(cv2.imread(str(folder / "renders" / name / f"view_{v:02}_rgba.png"),
                                              cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                image, _, _ = prepare_rgba(rgba)
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            paths[f"cad/{name}/{v}"] = path
    return paths


def generate_proposals(rgb, depth):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generation_protocol = {"model": model_provenance(), "settings": SAM_SETTINGS,
                           "rgb_stability": .90, "depth_stability": .95,
                           "input_shape": list(rgb.shape), "depth_dtype": str(depth.dtype),
                           "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                           "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                           "script_sha256": association.content_hash(__file__)}
    protocol_path = OUTPUT / "sam_protocol.json"
    if protocol_path.exists():
        assert read_json(protocol_path) == generation_protocol, "Frozen SAM3 generation protocol changed"
    else:
        save_json(protocol_path, generation_protocol)
    proposals = []
    sam = None
    for branch, image, stability, area_min in (
            ("rgb", rgb, .90, 200), ("depth", inverse_depth_rgb(depth), .95, 800)):
        folder = OUTPUT / f"sam_{branch}"
        folder.mkdir(exist_ok=True)
        if (folder / "proposals.json").exists():
            proposals.extend(read_json(folder / "proposals.json"))
            continue
        cv2.imwrite(str(folder / "sam_input.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if sam is None:
            start = perf_counter()
            sam = Sam3AutomaticMasks()
            save_json(OUTPUT / "sam_model.json", {"load_s": perf_counter() - start,
                      "source": generation_protocol["model"]})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        with torch.inference_mode():
            generated = sam.generate(image, stability_score_thresh=stability, **SAM_SETTINGS)
        torch.cuda.synchronize()
        generation_s = perf_counter() - start
        rows = []
        for prediction in generated:
            mask = prediction["segmentation"]
            assert mask.shape == depth.shape
            area = int(mask.sum())
            coverage = float((np.isfinite(depth[mask]) & (depth[mask] > 0)).mean()) if area else 0.
            if area < area_min or (branch == "depth" and coverage < .5):
                continue
            cid = f"{branch}_{len(rows) + 1:03}"
            path = folder / f"{cid}.png"
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            rows.append({"candidate_id": cid, "branch": branch, "mask_path": str(path),
                "area_pixels": area, "depth_valid_fraction": coverage,
                "predicted_iou": float(prediction["predicted_iou"]),
                "stability_score": prediction["stability_score"],
                "stability_threshold": prediction["stability_threshold"],
                "box_xyxy_exclusive": bbox(mask)})
        save_json(folder / "proposals.json", rows)
        save_json(folder / "runtime.json", {"generation_s": generation_s,
            "generation_and_filter_s": perf_counter() - start, "generated_count": len(generated),
            "retained_count": len(rows), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
        proposals.extend(rows)
        print(f"SAM {branch}: {len(generated)} proposals, {len(rows)} retained, {generation_s:.2f}s", flush=True)
        del generated
    del sam
    gc.collect()
    torch.cuda.empty_cache()
    return proposals


def observed_inputs(rgb, proposals):
    folder = OUTPUT / "observed_inputs"
    folder.mkdir(exist_ok=True)
    paths = {}
    for row in proposals:
        path = folder / f"{row['candidate_id']}.png"
        if not path.exists():
            mask = cv2.imread(row["mask_path"], 0) > 0
            image, _, _ = prepare_masked_input(rgb, mask)
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths[row["candidate_id"]] = path
    return paths


def encode_inputs(paths):
    """Frozen existing 224px DINO preprocessing; cache CLS and mean patch only."""
    index = {"keys": list(paths), "image_sha256": [association.content_hash(p) for p in paths.values()]}
    path = OUTPUT / "features.npz"
    if path.exists():
        assert read_json(OUTPUT / "feature_index.json") == index
        with np.load(path) as f:
            return {k: (c, p) for k, c, p in zip(index["keys"], f["cls"], f["pool"], strict=True)}
    torch.hub.set_dir(str(ROOT / association.DINO_HUB_DIR))
    start = perf_counter()
    model = torch.hub.load(association.DINO_REPO, MODEL, pretrained=True,
                          trust_repo=True, skip_validation=True).eval().requires_grad_(False).cuda()
    load_s = perf_counter() - start
    cls, pool = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    files = list(paths.values())
    with torch.inference_mode():
        for offset in range(0, len(files), 8):
            tensors = []
            for file in files[offset:offset + 8]:
                image = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
                assert image.shape == (224, 224, 3)
                normalized = (image.astype(np.float32) / 255 - [.485, .456, .406]) / [.229, .224, .225]
                tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)))
            features = model.forward_features(torch.stack(tensors).cuda())
            c = F.normalize(features["x_norm_clstoken"].double(), dim=-1)
            patches = F.normalize(features["x_norm_patchtokens"].double(), dim=-1)
            p = F.normalize(patches.mean(dim=1), dim=-1)
            cls.extend(c.cpu().numpy()); pool.extend(p.cpu().numpy())
            if offset % 80 == 0:
                print(f"DINO: {min(offset + 8, len(files))}/{len(files)} images", flush=True)
    torch.cuda.synchronize()
    save_json(OUTPUT / "dino_runtime.json", {"model_load_s": load_s,
        "encoding_s": perf_counter() - start, "images": len(files),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    np.savez_compressed(path, cls=np.stack(cls), pool=np.stack(pool))
    save_json(OUTPUT / "feature_index.json", index)
    del model, features, patches, c, p
    gc.collect(); torch.cuda.empty_cache()
    return {k: (c, p) for k, c, p in zip(paths, cls, pool, strict=True)}


def score_pairs(catalog, proposals, features, depth, K):
    path = OUTPUT / "pair_scores.json"
    if path.exists():
        return read_json(path)
    start = perf_counter()
    rotations = np.load(OUTPUT / "templates42/cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    cad = {name: CadSurface(o3d.io.read_triangle_mesh(item["cad_path"])) for name, item in catalog.items()}
    template = {name: tuple(np.stack([features[f"cad/{name}/{v}"][i] for v in range(1, 43)])
                            for i in range(2)) for name in catalog}
    rows = []
    for proposal in proposals:
        mask = cv2.imread(proposal["mask_path"], 0) > 0
        points = depth_points(depth, mask, K, pixel_offset=0.)
        center = points.mean(0) if len(points) else None
        for name in catalog:
            cosines, view, appearance = paper_visual_scores(*features[proposal["candidate_id"]], *template[name])
            T = np.eye(4)
            T[:3, :3] = rotations[view]
            geometry, ratio, cad_box = 0., 0., None
            if center is not None:
                # Same CAD origin used for the normalized visual templates, in metric units.
                T[:3, 3] = center - T[:3, :3] @ cad[name].center
                render = cad[name].render(T, K, depth.shape, pixel_offset=0.)
                if render is not None:
                    visible = np.isfinite(render["depth"]) & (render["depth"] > 0)
                    local_box = bbox(visible)
                    if local_box is not None:
                        x1, y1, x2, y2 = render["roi"]
                        cad_box = (local_box[0] + x1, local_box[1] + y1, local_box[2] + x1, local_box[3] + y1)
                        measured = depth[y1:y2, x1:x2]
                        ratio = float((np.isfinite(measured[visible]) & (measured[visible] > 0)).mean())
                        geometry = box_iou(cad_box, proposal["box_xyxy_exclusive"])
            semantic = float(cosines[view])
            rows.append({"cad_id": name, "candidate_id": proposal["candidate_id"], "branch": proposal["branch"],
                "all_view_cls_cosines": cosines.tolist(), "best_view_index_1based": view + 1,
                "semantic": semantic, "appearance": appearance, "geometry_box_iou": geometry,
                "valid_projected_depth_ratio": ratio, "projected_cad_box": cad_box,
                "camera_T_cad": T.tolist() if center is not None else None,
                "score": paper_score(semantic, appearance, geometry, ratio)})
        print(f"Scored {proposal['candidate_id']} against all {len(catalog)} CADs", flush=True)
    assert len(rows) == len(catalog) * len(proposals)
    save_json(path, rows)
    save_json(OUTPUT / "matching_runtime.json", {"all_pairs_s": perf_counter() - start,
              "pairs": len(rows), "pose_refinement": False,
              "note": "Both branches exhaustively recorded for audit; EO decisions replayed from scores."})
    return rows


def evaluate(catalog, proposals, rows, rgb):
    """Ground truth enters only here, after inference scores have been saved."""
    directory = SCENE / "sam3_segmentation/evaluation_masks"
    truth = {name: cv2.imread(str(directory / f"{name}.png"), 0) > 0 for name in catalog}
    assert all(mask.shape == rgb.shape[:2] for mask in truth.values())
    matches = {}
    for proposal in proposals:
        mask = cv2.imread(proposal["mask_path"], 0) > 0
        ious = {name: float(np.count_nonzero(mask & gt) / np.count_nonzero(mask | gt))
                for name, gt in truth.items()}
        identity = max(ious, key=ious.get)
        matches[proposal["candidate_id"]] = {"best_overlap_identity": identity,
            "identity_at_iou_0_5": identity if ious[identity] >= EVALUATION_IOU else None,
            "best_iou": ious[identity], "all_ious": ious}
    result = {"scope": "Single saved 16-object scene; independent find-object-for-CAD selection, not benchmark AP",
              "evaluation_iou": EVALUATION_IOU, "ground_truth_used_in_inference": False,
              "methods": {}, "proposal_evaluation": matches}
    for method in ("rgb_only", "depth_only", "dual_depth_first_eo"):
        targets = []
        for name in catalog:
            ranking, stopped = select_candidates([r for r in rows if r["cad_id"] == name], method)
            winner = ranking[0] if ranking else None
            is_correct = lambda r: matches[r["candidate_id"]]["identity_at_iou_0_5"] == name
            correct = next((r for r in ranking if is_correct(r)), None)
            wrong = next((r for r in ranking if not is_correct(r)), None)
            targets.append({"cad_id": name, "prediction": winner["candidate_id"] if winner else None,
                "selected_identity": matches[winner["candidate_id"]]["identity_at_iou_0_5"] if winner else None,
                "best_overlap_identity": matches[winner["candidate_id"]]["best_overlap_identity"] if winner else None,
                "selected_mask_iou_with_target": matches[winner["candidate_id"]]["all_ious"][name] if winner else None,
                "correct": bool(winner and is_correct(winner)), "depth_early_exit": stopped,
                "winner_score": winner["score"] if winner else None,
                "correct_score": correct["score"] if correct else None,
                "strongest_incorrect_score": wrong["score"] if wrong else None,
                "margin": correct["score"] - wrong["score"] if correct and wrong else None,
                "ranking": [{"candidate_id": r["candidate_id"], "score": r["score"]} for r in ranking]})
        result["methods"][method] = {"correct": sum(r["correct"] for r in targets), "total": len(catalog),
            "depth_early_exits": sum(r["depth_early_exit"] for r in targets),
            "targets_without_valid_correct_proposal": sum(r["correct_score"] is None for r in targets),
            "targets": targets}
    save_json(OUTPUT / "classification.json", result)
    overlay = cv2.resize(rgb, (1200, 1200))
    for index, row in enumerate(result["methods"]["dual_depth_first_eo"]["targets"], 1):
        if row["prediction"] is None:
            continue
        proposal = next(p for p in proposals if p["candidate_id"] == row["prediction"])
        mask = cv2.imread(proposal["mask_path"], 0)
        small = cv2.resize(mask, (1200, 1200), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (30, 210, 50) if row["correct"] else (240, 40, 30)
        cv2.drawContours(overlay, contours, -1, color, 2)
        x1, y1, _, _ = np.array(proposal["box_xyxy_exclusive"]) * 1200 / rgb.shape[0]
        cv2.putText(overlay, str(index), (int(x1), max(14, int(y1))), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    cv2.imwrite(str(OUTPUT / "selected_regions.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "targets"} for k, v in result["methods"].items()}), flush=True)
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("The existing CUDA environment is required.")
    scene = read_json(SCENE / "scene.json")
    catalog = scene["catalog"]
    assert len(catalog) == 16
    rgb = cv2.cvtColor(cv2.imread(str(SCENE / "wrist_rgb.png")), cv2.COLOR_BGR2RGB)
    depth, K = np.load(SCENE / "depth.npy"), np.load(SCENE / "intrinsics.npy")
    assert rgb.shape == (3840, 3840, 3) and depth.shape == rgb.shape[:2]
    protocol = {"scene": str(SCENE), "objects": list(catalog), "sam_model": "local SAM3 tracker", "sam_settings": SAM_SETTINGS,
        "sam_provenance": model_provenance(),
        "sam_input_size": list(rgb.shape), "rgb_stability": .9, "depth_stability": .95,
        "minimum_area_rgb_depth": [200, 800], "minimum_valid_depth_fraction": .5,
        "encoder": MODEL, "encoder_input": "224x224 white masked RGB, aspect-preserving letterbox",
        "encoder_source": association.DINO_REPO, "template_views": 42,
        "global_score": "max CLS cosine; no clipping to [0,1]",
        "patch_score": "cosine of normalized means of individually normalized patch tokens, all 256 patches",
        "geometry_score": "projected box IoU at best CLS rotation; metric CAD center placed at observed cloud centroid",
        "fusion": "(CLS + pooled_patch + valid_projected_depth_ratio * box_IoU)/(2 + valid_projected_depth_ratio)",
        "early_exit_threshold": EO_THRESHOLD, "ground_truth_only_for_evaluation": True,
        "pose_refinement": False, "precision": "SAM3 BF16; DINO FP32; normalized DINO outputs and pooling float64",
        "paper_reproduction_limits": ["Authors code unavailable; follow published scoring equations",
            "Existing 42-view BlenderProc setup and 224px DINOv2-L inputs retained",
            "SAM3 tracker replaces original SAM; 32x32 text-free point grid, 4 prompts per batch",
            "Native-resolution masks use SAM3 soft-mask resizing before binarization",
            "Pool all patch tokens because foreground pooling is not specified in the paper",
            "Ideal simulated depth; one inspection arrangement, no tuning or generalization claim"],
        "source_hashes": {p: association.content_hash(SCENE / p) for p in ("wrist_rgb.png", "depth.npy", "intrinsics.npy", "scene.json")}}
    if (OUTPUT / "protocol.json").exists():
        assert read_json(OUTPUT / "protocol.json") == protocol
    else:
        save_json(OUTPUT / "protocol.json", protocol)
    paths = prepare_templates(catalog)
    proposals = generate_proposals(rgb, depth)
    paths.update(observed_inputs(rgb, proposals))
    features = encode_inputs(paths)
    rows = score_pairs(catalog, proposals, features, depth, K)
    evaluate(catalog, proposals, rows, rgb)
    assert all(association.content_hash(SCENE / p) == h for p, h in protocol["source_hashes"].items())
    save_json(OUTPUT / "validation.json", {"source_inputs_unchanged": True,
        "cad_candidate_pairs": len(rows), "cad_view_cls_comparisons": 42 * len(rows),
        "candidate_counts": {b: sum(p["branch"] == b for p in proposals) for b in ("rgb", "depth")},
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "script_sha256": association.content_hash(__file__)})


if __name__ == "__main__":
    main()
