"""Predeclared three-frame T-LESS association pilot with provided visible regions.

This evaluates scoring conditional on GT masks, not segmentation or official BOP
pose AR. All 30 CAD identities compete for every region; poses are never provided
to the inference path. The masks themselves are an explicit oracle input.
"""

import os
import shutil
import zipfile
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d
import torch

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, save_json, read_json, read_rgb, save_image
from scripts.run_wrist_large_encoder import BLENDER
from scripts.run_wrist_blenderproc_semantic import render_templates, prepare_rgba
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
from scripts.sam6d_patch_matching import masked_patch_descriptors
from scripts.run_surface_verification import OUTPUT, feature_cache, score_scene, METHODS
from scripts.surface_verification import CadSurface, verify_pose


CACHE = ROOT / "outputs/cache/bop_tless"
PILOT = OUTPUT / "stage4_tless"
SCENES = (1, 10, 20)


def extract_pilot():
    destination = CACHE / "data"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(CACHE / "tless_base.zip") as archive:
        archive.extractall(destination)
    data = destination / "tless"
    with zipfile.ZipFile(CACHE / "tless_models.zip") as archive:
        archive.extractall(data, members=[n for n in archive.namelist() if n.startswith("models_cad/")])
    targets = read_json(data / "test_targets_bop19.json")
    plan = [{"scene_id": scene, "im_id": min(r["im_id"] for r in targets if r["scene_id"] == scene)} for scene in SCENES]
    with zipfile.ZipFile(CACHE / "tless_test_primesense_bop19.zip") as archive:
        selected = []
        for item in plan:
            prefix = f"test_primesense/{item['scene_id']:06}/"
            frame = f"{item['im_id']:06}"
            selected.extend(n for n in archive.namelist() if n.startswith(prefix) and
                            (n.endswith(".json") or ("/" + frame) in n))
        archive.extractall(data, members=selected)
    save_json(PILOT / "data_provenance.json", {"source": "https://huggingface.co/datasets/bop-benchmark/tless",
        "plan": plan, "selection": "first official target frame in scenes 1,10,20; no performance-based selection",
        "archive_sha256": {p.name: association.content_hash(p) for p in CACHE.glob("*.zip")}})
    return data, plan


def templates_and_geometry(data):
    folder = PILOT / "templates42"
    folder.mkdir(exist_ok=True)
    library, meshes = [], {}
    for index in range(1, 31):
        cid = f"obj_{index:06}"
        mesh = o3d.io.read_triangle_mesh(str(data / "models_cad" / f"{cid}.ply"))
        mesh.scale(.001, center=(0, 0, 0))  # Official BOP CAD millimeters -> meters.
        meshes[cid] = mesh
        path = folder / f"{cid}_m.ply"
        if not path.exists():
            o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
        library.append({"cad_id": cid, "file_path": str(path.relative_to(ROOT)), "base_color_rgb": [.95, .95, .95]})
    if not (folder / "render_job.json").exists():
        shutil.copyfile(BLENDER / "cam_poses_level0.npy", folder / "cam_poses_level0.npy")
        save_json(folder / "render_job.json", {"output": str(folder), "render_device": "GPU", "cad_library": [
            {**r, "absolute_mesh_path": str(ROOT / r["file_path"])} for r in library]})
    render_templates(folder, library)
    assert read_json(folder / "render_runtime.json")["device"] == "GPU"
    paths, weights = {}, {}
    for cid in meshes:
        for v in range(1, 43):
            source = folder / "renders" / cid / f"view_{v:02}_rgba.png"
            rgba = cv2.cvtColor(cv2.imread(str(source), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            image, mask, bounds = prepare_rgba(rgba)
            x1, y1, x2, y2 = bounds
            key = f"{cid}/{v}"
            path = folder / f"{cid}_view_{v:02}.png"
            save_image(path, image)
            paths[key] = path
            weights[key] = patch_occupancy(letterbox_mask(mask[y1:y2 + 1, x1:x2 + 1]))
    features = feature_cache(paths, PILOT / "template_features")
    templates = {cid: {"cls": np.stack([features[f"{cid}/{v}"][0] for v in range(1, 43)]),
        "patch": [masked_patch_descriptors(features[f"{cid}/{v}"][1], weights[f"{cid}/{v}"]).cuda() for v in range(1, 43)]}
        for cid in meshes}
    cads = {cid: CadSurface(mesh) for cid, mesh in meshes.items()}
    rotations = np.load(folder / "cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    visible = {cid: [cad.visible_points(R) for R in rotations] for cid, cad in cads.items()}
    return cads, visible, rotations, templates


def classify_regions(rows, identity):
    """All 30 CADs compete, including absent models; missing results are failures."""
    results = []
    for oid, correct_cid in identity.items():
        for method in METHODS:
            ranking = sorted((r for r in rows if r["object_id"] == oid and r[method] is not None), key=lambda r: (-r[method], r["cad_id"]))
            right = next((r for r in ranking if r["cad_id"] == correct_cid), None)
            wrong = next((r for r in ranking if r["cad_id"] != correct_cid), None)
            prediction = ranking[0]["cad_id"] if ranking else None
            results.append({"object_id": oid, "expected": correct_cid, "method": method, "prediction": prediction,
                "correct": prediction == correct_cid,
                "margin": right[method] - wrong[method] if right and wrong else None,
                "ranking": [{"cad_id": r["cad_id"], "score": r[method]} for r in ranking]})
    return results


def summarize(records):
    result = {}
    for method in METHODS:
        selected = [r for r in records if r["method"] == method]
        margins = [r["margin"] for r in selected if r["margin"] is not None]
        result[method] = {"correct": sum(r["correct"] for r in selected), "total": len(selected),
                          "margin_count": len(margins), "mean_margin": float(np.mean(margins)) if margins else None}
    return result


def verified_pose_diagnostic(cads, gt, depth, masks, K, pairs):
    """Post-evaluation diagnostic only. Does not feed the identity rankings."""
    rows = []
    for i, record in enumerate(gt):
        oid, cid = f"region_{i:03}", f"obj_{record['obj_id']:06}"
        if oid not in masks:
            continue
        T = np.eye(4)
        T[:3, :3] = np.array(record["cam_R_m2c"]).reshape(3, 3)
        T[:3, 3] = np.array(record["cam_t_m2c"]) * .001
        score = verify_pose(cads[cid], T, depth, masks[oid], K, pixel_offset=0.)
        estimate = next((r["geometry"] for r in pairs if r["cad_id"] == cid and r["object_id"] == oid), None)
        rows.append({"cad_id": cid, "object_id": oid, "geometry_with_estimated_pose": estimate,
                     "geometry_at_supplied_gt_pose": score, "used_for_prediction": False})
    return rows


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = perf_counter()
    PILOT.mkdir(parents=True, exist_ok=True)
    data, plan = extract_pilot()
    save_json(PILOT / "protocol.json", {"plan": plan, "regions": "All provided nonempty mask_visib regions; no visibility threshold",
        "task": "Conditional region-to-CAD classification against all 30 CADs; CAD-to-region rankings also saved",
        "pose": "No GT pose supplied; same 42 rotations and 30 translation updates as simulator",
        "units": "CAD mm and depth*depth_scale mm converted to meters; per-image cam_K; integer pixel rays",
        "appearance": "Same BlenderProc settings; neutral white material for untextured CADs; actual GPU device verified",
        "limits": "Three real frames, oracle segmentation, no fitted table-plane constraint; not official BOP metrics or end-to-end detection",
        "fusion": ".25 global + .25 patch(v) + .50 surface(v); fixed before evaluation"})
    cads, visible, rotations, templates = templates_and_geometry(data)
    all_results, all_queries, diagnostics = [], [], []
    for item in plan:
        source = data / "test_primesense" / f"{item['scene_id']:06}"
        frame = f"{item['im_id']:06}"
        camera = read_json(source / "scene_camera.json")[str(item["im_id"])]
        depth = cv2.imread(str(source / "depth" / f"{frame}.png"), cv2.IMREAD_UNCHANGED).astype(float) * camera["depth_scale"] * .001
        K = np.array(camera["cam_K"]).reshape(3, 3)
        rgb = read_rgb(source / "rgb" / f"{frame}.png")
        masks, identities = {}, {}
        for i, record in enumerate(read_json(source / "scene_gt.json")[str(item["im_id"])]):
            oid = f"region_{i:03}"
            mask = cv2.imread(str(source / "mask_visib" / f"{frame}_{i:06}.png"), 0) > 0
            identities[oid] = f"obj_{record['obj_id']:06}"
            if mask.any():
                masks[oid] = mask
        truth = {cid: [oid for oid, value in identities.items() if value == cid] for cid in sorted(set(identities.values()))}
        folder = PILOT / f"scene_{item['scene_id']:06}_{frame}"
        query_result = score_scene(cads, visible, rotations, templates, depth, masks, K, rgb, truth, folder, offset=0.)
        all_queries.extend(query_result["targets"])
        pairs = read_json(folder / "pair_scores.json")
        results = classify_regions(pairs, identities)
        save_json(folder / "region_classification.json", results)
        all_results.extend({"scene_id": item["scene_id"], **r} for r in results)
        diagnostic = verified_pose_diagnostic(cads, read_json(source / "scene_gt.json")[str(item["im_id"])], depth, masks, K, pairs)
        save_json(folder / "gt_pose_diagnostic.json", diagnostic)
        diagnostics.extend({"scene_id": item["scene_id"], **r} for r in diagnostic)
    save_json(PILOT / "region_classification.json", all_results)
    save_json(PILOT / "summary.json", summarize(all_results))
    save_json(PILOT / "cad_to_region_summary.json", summarize(all_queries))
    save_json(PILOT / "gt_pose_diagnostic.json", diagnostics)
    save_json(PILOT / "runtime.json", {"wall_this_invocation_s": perf_counter() - start})


if __name__ == "__main__":
    main()
