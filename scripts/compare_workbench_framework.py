"""Replay our fixed three-branch matcher beside the saved RGB/depth SAM method.

One 16-object capture, independent CAD-to-region queries, no final pose estimator.
Ground-truth masks enter only evaluation. No parameter or method search.
"""

from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
import torch

from cad_object_association import content_hash
from simulation.industrial_workbench import TABLE_CENTER_XY_M, TABLE_SIZE_M
from simulation.nist_peg_task import localize_parts
from scripts.run_workbench_depth_sam import ROOT, SCENE, OUTPUT as PAPER, read_json, save_json
from scripts.run_surface_verification import feature_cache
from scripts.run_wrist_input_ablation import masked_crop
from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
from scripts.run_wrist_sam_visual_comparison import prepare_masked_input
from scripts.sam6d_patch_matching import semantic_score, masked_patch_descriptors, appearance_score
from scripts.surface_verification import (
    CadSurface, bbox, downsample, depth_points, translation_fit, equal_penalty_scores, VOXEL_M,
)


OUTPUT = ROOT / "outputs/workbench_framework_sam3_local_20260921"
IOU = .5


def localize(rgb, depth, K):
    path = OUTPUT / "proposals.json"
    if path.exists():
        return read_json(path), read_json(OUTPUT / "point_cloud/point_cloud_localization.json")
    start = perf_counter()
    observation = SimpleNamespace(rgb=rgb, depth_m=depth, intrinsics=K,
                                 world_T_camera=np.load(SCENE / "world_T_camera.npy"))
    half = np.asarray(TABLE_SIZE_M[:2]) / 2
    # Table footprint is a workspace prior, not inferred from object poses.
    minimum = [*(np.asarray(TABLE_CENTER_XY_M) - half), -.04]
    maximum = [*(np.asarray(TABLE_CENTER_XY_M) + half), .23]
    o3d.utility.random.seed(20260919)
    localization, _ = localize_parts(observation, SCENE / "wrist_rgb.png", OUTPUT,
        workspace_min=minimum, workspace_max=maximum, cluster_in_table_plane=True)
    proposals = []
    (OUTPUT / "masks").mkdir(exist_ok=True)
    for item in localization["objects"]:
        cloud = o3d.io.read_point_cloud(str(ROOT / item["pointcloud_path"]))
        _, mask, details = masked_crop(rgb, cloud, localization["camera_intrinsics"])
        path = OUTPUT / "masks" / f"{item['object_id']}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        proposals.append({"candidate_id": item["object_id"], "mask_path": str(path),
                          "box_xyxy_exclusive": bbox(mask), **details})
    save_json(OUTPUT / "proposals.json", proposals)
    save_json(OUTPUT / "localization_runtime.json", {"wall_s": perf_counter() - start,
        "workspace_min_world_m": minimum, "workspace_max_world_m": maximum,
        "proposal_count": len(proposals), "settings": "Unchanged nist_peg_task.localize_parts"})
    print(f"Depth localization: {len(proposals)} regions", flush=True)
    return proposals, localization


def extract_features(catalog, proposals, rgb):
    paths, weights = {}, {}
    for cid in catalog:
        for v in range(1, 43):
            key = f"cad/{cid}/{v}"
            paths[key] = PAPER / "templates42/inputs" / f"{cid}_{v:02}.png"
            rgba = cv2.imread(str(PAPER / "templates42/renders" / cid / f"view_{v:02}_rgba.png"), -1)
            mask = rgba[..., 3] > 0
            x1, y1, x2, y2 = bbox(mask)
            weights[key] = patch_occupancy(letterbox_mask(mask[y1:y2, x1:x2]))
    (OUTPUT / "inputs").mkdir(exist_ok=True)
    for proposal in proposals:
        key = proposal["candidate_id"]
        mask = cv2.imread(proposal["mask_path"], 0) > 0
        image, _, _ = prepare_masked_input(rgb, mask)
        path = OUTPUT / "inputs" / f"{key}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths[key] = path
        x1, y1, x2, y2 = bbox(mask)
        weights[key] = patch_occupancy(letterbox_mask(mask[y1:y2, x1:x2]))
    features = feature_cache(paths, OUTPUT / "features")
    return {key: {"cls": cls, "patch": masked_patch_descriptors(patch, weights[key]).cuda()}
            for key, (cls, patch) in features.items()}


def pose_status(cad, T, K, shape, table):
    """Same camera/table admissibility as verify_pose, before expensive raycasts."""
    vertices = cad.vertices @ T[:3, :3].T + T[:3, 3]
    if np.min(vertices @ table[:3] + table[3]) < -VOXEL_M:
        return "table_penetration"
    if np.any(vertices[:, 2] <= 0):
        return "behind_or_outside_camera"
    pixels = vertices @ K.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    lower, upper = np.floor(pixels.min(0) - 1), np.ceil(pixels.max(0) + 1)
    if lower[0] < 0 or lower[1] < 0 or upper[0] >= shape[1] or upper[1] >= shape[0]:
        return "image_clipped"
    return "assessable"


def score(catalog, proposals, localization, features, depth, K):
    path = OUTPUT / "pair_scores.json"
    if path.exists():
        return read_json(path)
    rotations = np.load(PAPER / "templates42/cam_poses_level0.npy")[:, :3, :3].transpose(0, 2, 1)
    table = np.array(localization["plane_model"])
    if table[3] < 0:
        table *= -1
    masks = {p["candidate_id"]: cv2.imread(p["mask_path"], 0) > 0 for p in proposals}
    points = {oid: downsample(depth_points(depth, mask, K)) for oid, mask in masks.items()}
    timing = {"cad_setup_s": 0., "visual_s": 0., "alignment_and_table_s": 0., "surface_s": 0.}
    pairs, all_views = [], []
    for cid, item in catalog.items():
        tick = perf_counter()
        cad = CadSurface(o3d.io.read_triangle_mesh(item["cad_path"]))
        visible = [cad.visible_points(R) for R in rotations]
        templates = [features[f"cad/{cid}/{v}"] for v in range(1, 43)]
        template_cls = np.stack([t["cls"] for t in templates])
        timing["cad_setup_s"] += perf_counter() - tick
        for oid, mask in masks.items():
            torch.cuda.synchronize()
            tick = perf_counter()
            semantic = semantic_score(features[oid]["cls"], template_cls)
            patches = [appearance_score(features[oid]["patch"], t["patch"])["appearance_score"] for t in templates]
            torch.cuda.synchronize()
            timing["visual_s"] += perf_counter() - tick
            tick = perf_counter()
            views = []
            for v, R in enumerate(rotations):
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3], _ = translation_fit(visible[v], points[oid])
                views.append({"cad_id": cid, "candidate_id": oid, "view_index_1based": v + 1,
                    "camera_T_cad": T.tolist(), "pose_status": pose_status(cad, T, K, depth.shape, table),
                    "cls": semantic["cls_view_cosines"][v], "global": semantic["semantic_score"],
                    "patch": patches[v], "geometry": None, "score": None})
            timing["alignment_and_table_s"] += perf_counter() - tick
            selected = sorted((r for r in views if r["pose_status"] == "assessable"),
                              key=lambda r: (-r["patch"], r["view_index_1based"]))[:5]
            tick = perf_counter()
            for row in selected:
                render = cad.render(np.array(row["camera_T_cad"]), K, depth.shape, bbox(mask))
                assert render is not None and not render["clipped"]
                x1, y1, x2, y2 = render["roi"]
                metrics = equal_penalty_scores(depth[y1:y2, x1:x2], mask[y1:y2, x1:x2], render["depth"])
                row.update(metrics)
                row["geometry"] = metrics["geometry_score"]
                if row["geometry"] is not None:
                    row["score"] = .25 * row["global"] + .25 * row["patch"] + .5 * row["geometry"]
            timing["surface_s"] += perf_counter() - tick
            usable = sorted((r for r in selected if r["score"] is not None),
                            key=lambda r: (-r["score"], r["view_index_1based"]))
            winner = usable[0] if usable else None
            pairs.append({"cad_id": cid, "candidate_id": oid, "score": winner["score"] if winner else None,
                "winning_view": winner["view_index_1based"] if winner else None,
                "selected_views": [r["view_index_1based"] for r in selected],
                "winning_components": {k: winner[k] for k in ("global", "patch", "geometry")} if winner else None})
            all_views.extend(views)
        print(f"Our matcher: {cid} completed against all {len(proposals)} regions", flush=True)
    save_json(OUTPUT / "view_scores.json", all_views)
    save_json(path, pairs)
    save_json(OUTPUT / "matching_runtime.json", timing)
    return pairs


def evaluate(catalog, proposals, pairs):
    """Use the same mask-IoU >= 0.5 success criterion as the paper-method run."""
    truth = {cid: cv2.imread(str(SCENE / "sam3_segmentation/evaluation_masks" / f"{cid}.png"), 0) > 0
             for cid in catalog}
    matches = {}
    for proposal in proposals:
        mask = cv2.imread(proposal["mask_path"], 0) > 0
        ious = {cid: float(np.count_nonzero(mask & gt) / np.count_nonzero(mask | gt)) for cid, gt in truth.items()}
        identity = max(ious, key=ious.get)
        matches[proposal["candidate_id"]] = {"all_ious": ious,
            "identity_at_iou_0_5": identity if ious[identity] >= IOU else None}
    targets = []
    for cid in catalog:
        ranked = sorted((p for p in pairs if p["cad_id"] == cid and p["score"] is not None),
                        key=lambda p: (-p["score"], p["candidate_id"]))
        correct = [p for p in ranked if matches[p["candidate_id"]]["identity_at_iou_0_5"] == cid]
        wrong = [p for p in ranked if matches[p["candidate_id"]]["identity_at_iou_0_5"] != cid]
        winner = ranked[0] if ranked else None
        targets.append({"cad_id": cid, "prediction": winner["candidate_id"] if winner else None,
            "selected_identity": matches[winner["candidate_id"]]["identity_at_iou_0_5"] if winner else None,
            "correct": bool(winner and correct and winner == correct[0]),
            "proposal_available": any(m["all_ious"][cid] >= IOU for m in matches.values()),
            "winner_score": winner["score"] if winner else None,
            "correct_score": correct[0]["score"] if correct else None,
            "margin": correct[0]["score"] - wrong[0]["score"] if correct and wrong else None,
            "ranking": [{"candidate_id": p["candidate_id"], "score": p["score"]} for p in ranked]})
    return {"correct": sum(t["correct"] for t in targets), "total": len(targets),
            "objects_with_usable_masks": sum(t["proposal_available"] for t in targets),
            "targets": targets, "proposal_evaluation": matches}


def main():
    OUTPUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    paper_protocol = read_json(PAPER / "protocol.json")
    assert all(content_hash(SCENE / p) == h for p, h in paper_protocol["source_hashes"].items())
    protocol = {"scene": str(SCENE), "comparison": "Independent find-object-for-CAD; same 16 targets and IoU >= 0.5",
        "our_variant": "Existing table_then_patch5 from run_wrist_branch_order; selected before this run, not a sweep",
        "fusion": ".25 global top5 mean + .25 selected-view patch + .50 equal-penalty surface geometry",
        "localization": "Unchanged localize_parts settings; whole designed table footprint; world Z [-.04,.23] m",
        "geometry_placement": "Existing 42 rotations, 5mm sampling and 30 translation-only updates; no final pose refinement",
        "geometry_score": "Native-resolution observed/rendered support and clipped depth penalties; sigma 5mm",
        "templates_and_encoder": "Same 42 renders per CAD, white 224px DINOv2-L; SAM-6D-style foreground patch matching",
        "paper_source": str(PAPER), "paper_limitations": paper_protocol["paper_reproduction_limits"],
        "input_difference": "Our localizer uses a table workspace crop; paper SAM uses full RGB and inverse depth",
        "ground_truth": "Evaluation masks only, after all pair scores; no names given to either segmenter",
        "source_hashes": paper_protocol["source_hashes"], "production_modified": False}
    save_json(OUTPUT / "protocol.json", protocol)
    rgb = cv2.cvtColor(cv2.imread(str(SCENE / "wrist_rgb.png")), cv2.COLOR_BGR2RGB)
    depth, K = np.load(SCENE / "depth.npy"), np.load(SCENE / "intrinsics.npy")
    catalog = read_json(SCENE / "scene.json")["catalog"]
    proposals, localization = localize(rgb, depth, K)
    if not (OUTPUT / "pair_scores.json").exists():
        features = extract_features(catalog, proposals, rgb)
        score(catalog, proposals, localization, features, depth, K)
    ours = evaluate(catalog, proposals, read_json(OUTPUT / "pair_scores.json"))
    paper = read_json(PAPER / "classification.json")["methods"]["dual_depth_first_eo"]
    save_json(OUTPUT / "comparison.json", {"ours": ours, "paper_method_implementation": paper,
        "same_saved_scene": True, "pose_estimation_evaluated": False, "benchmark_ap": False})
    print({"ours": f"{ours['correct']}/{ours['total']}", "paper": f"{paper['correct']}/{paper['total']}"}, flush=True)


if __name__ == "__main__":
    main()
