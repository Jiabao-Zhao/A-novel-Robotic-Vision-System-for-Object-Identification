"""Two view-shortlist experiments on the saved six-object wrist scene.

Run with the existing CUDA/Open3D environment: python -m scripts.run_wrist_shortlist_geometry
Reuse DINO features and captured registrations; no encoder, ICP, SAM or scene run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/surface_verification_20260917"
POSE_SOURCE = SOURCE / "stage2_pose_search"
DATASET = ROOT / "experiments/wrist_association_2026-09-11"
MASKS = ROOT / "outputs/ablation_1_wrist_association_2026-09-11/20260912T153218768705Z/masked/masks"
OUTPUT = ROOT / "outputs/wrist_shortlist_geometry"
TOP_K = 5
PRIMARY_SCORE = "fused_score"
STRATEGIES = ("global_top5", "global_patch_intersection")


def top_views(scores, count=TOP_K):
    """One-based template IDs, with lower ID breaking exact similarity ties."""
    return sorted(range(1, len(scores) + 1), key=lambda v: (-scores[v - 1], v))[:count]


def intersect_views(global_views, patch_views):
    """Retain common IDs in global-ranking order."""
    return [v for v in global_views if v in patch_views]


def intersection_or_union(global_views, patch_views):
    shared = intersect_views(global_views, patch_views)
    return shared or list(dict.fromkeys(global_views + patch_views))


def winning_view(rows, score_name):
    usable = [r for r in rows if r.get(score_name) is not None]
    return min(usable, key=lambda r: (-r[score_name], r["view_index_1based"])) if usable else None


def rank_pairs(pairs, truth, query_field, choice_field, score_name):
    """Independent selections, without a one-to-one assignment constraint."""
    targets = []
    for query, expected in truth.items():
        group = [p for p in pairs if p[query_field] == query]
        ranked = sorted((p for p in group if p.get(score_name) is not None),
                        key=lambda p: (-p[score_name], p[choice_field]))
        correct = next((p for p in ranked if p[choice_field] == expected), None)
        incorrect = next((p for p in ranked if p[choice_field] != expected), None)
        top = ranked[0] if ranked else None
        targets.append({"query": query, "expected": expected,
            "prediction": top[choice_field] if top else None,
            "correct": bool(top and top[choice_field] == expected),
            "winning_score": top[score_name] if top else None,
            "correct_score": correct[score_name] if correct else None,
            "strongest_incorrect_score": incorrect[score_name] if incorrect else None,
            "margin": correct[score_name] - incorrect[score_name] if correct and incorrect else None,
            "tied_top_choices": [p[choice_field] for p in ranked if p[score_name] == top[score_name]],
            "ranking": [{"choice": p[choice_field], "score": p[score_name]} for p in ranked],
            "unavailable_choices": [p[choice_field] for p in group if p.get(score_name) is None]})
    return {"correct": sum(t["correct"] for t in targets), "total": len(targets), "targets": targets}


def main():
    import cv2
    import open3d as o3d
    import torch
    from cad_object_association import content_hash
    from scripts.run_wrist_input_ablation import read_json, save_json, snapshot_hashes
    from scripts.run_wrist_large_encoder import BLENDER
    from scripts.run_wrist_patch_ablation import letterbox_mask, patch_occupancy
    from scripts.sam6d_patch_matching import semantic_score, masked_patch_descriptors, appearance_score
    from scripts.surface_verification import CadSurface, bbox, equal_penalty_scores, SIGMA_M, OCCLUSION_M

    start = perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("The existing CUDA environment is required for patch comparisons.")
    fixed = snapshot_hashes()
    protected = {p: content_hash(ROOT / p) for p in ("cad_object_association.py", "CADPointCloudRegistration.py")}
    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    library = read_json(DATASET / "cad_library.json")
    names = [r["cad_id"] for r in library]
    cads = {r["cad_id"]: CadSurface(o3d.io.read_triangle_mesh(str(ROOT / r["file_path"]))) for r in library}
    depth = np.load(DATASET / "scene/depth.npy")
    K = np.load(DATASET / "scene/intrinsics.npy")
    masks = {p.stem: cv2.imread(str(p), 0) > 0 for p in sorted(MASKS.glob("*.png"))}
    assert len(names) == len(masks) == 6
    saved = read_json(POSE_SOURCE / "all_view_scores.json")
    pose_lookup = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r for r in saved}
    assert len(saved) == len(pose_lookup) == 6 * 6 * 42

    provenance = {}
    caches = {}
    for family, folder in (("templates", SOURCE / "template_features"),
                           ("observations", POSE_SOURCE / "inputs/features")):
        for file in (folder / "index.json", folder / "features.npz"):
            provenance[file.relative_to(ROOT).as_posix()] = content_hash(file)
        index = read_json(folder / "index.json")
        assert index["model"] == "dinov2_vitl14"
        for key, expected in zip(index["keys"], index["sha256"], strict=True):
            if family == "templates":
                _, cid, view = key.split("/")
                image = BLENDER / "templates" / cid / f"view_{int(view):02}.png"
            else:
                image = POSE_SOURCE / "inputs" / f"{key}.png"
            assert content_hash(image) == expected
        with np.load(folder / "features.npz") as features:
            caches[family] = {key: (cls, patch) for key, cls, patch in
                zip(index["keys"], features["cls_features"], features["patch_tokens"], strict=True)}
    with np.load(SOURCE / "template_features/masks.npz") as f:
        occupancy = dict(zip(f["keys"].tolist(), f["patch_occupancy"], strict=True))
    for file in [POSE_SOURCE / "all_view_scores.json", SOURCE / "template_features/masks.npz", *MASKS.glob("*.png")]:
        provenance[file.relative_to(ROOT).as_posix()] = content_hash(file)
    templates = {cid: {"cls": np.stack([caches["templates"][f"blenderproc42/{cid}/{v}"][0] for v in range(1, 43)]),
        "patch": [masked_patch_descriptors(caches["templates"][f"blenderproc42/{cid}/{v}"][1],
                    occupancy[f"blenderproc42/{cid}/{v}"]).cuda() for v in range(1, 43)]} for cid in names}
    queries = {}
    for oid, mask in masks.items():
        x1, y1, x2, y2 = bbox(mask)
        weights = patch_occupancy(letterbox_mask(mask[y1:y2, x1:x2]))
        cls, patch = caches["observations"][oid]
        queries[oid] = {"cls": cls, "patch": masked_patch_descriptors(patch, weights).cuda()}
    del caches
    # Initialize CUDA before timing the two strategies; this is not a pair evaluation.
    _ = torch.ones((1, 1), device="cuda") @ torch.ones((1, 1), device="cuda")
    torch.cuda.synchronize()
    setup_s = perf_counter() - start
    protocol = {"dataset": DATASET.relative_to(ROOT).as_posix(), "top_k": TOP_K,
        "primary_score": PRIMARY_SCORE, "views_per_cad": 42, "cad_candidate_pairs": 36,
        "strategies": {"global_top5": "Global top 5; patch and geometry only on those views; maximize final score.",
                       "global_patch_intersection": "Independent global and patch top 5; geometry on intersection, or union if empty."},
        "empty_intersection": "User-requested union fallback, up to ten views.",
        "equation": "1 - (P_obs + P_cad + mean(min(abs(D_obs-D_cad)/sigma_d,1)))/3",
        "support_penalties": "P_obs=observed-only/union; P_cad=CAD-only/union",
        "sigma_d_m": SIGMA_M, "depth_residual": "Optical-axis Z difference, not Euclidean camera-ray distance.",
        "visible_cad": "Mesh first-hit raycast at original resolution; existing external-occlusion and valid-depth rules.",
        "external_occlusion_tolerance_m": OCCLUSION_M,
        "no_pixel_overlap": "Undefined intersection mean: score null; no numerical replacement.",
        "pose_source": (POSE_SOURCE / "all_view_scores.json").relative_to(ROOT).as_posix(),
        "pose_policy": "Frozen original 42 rotations and translations. Preserve saved table/camera admissibility; do not refill rejected views.",
        "pose_history": "Saved translations used 5 mm clouds and 30 translation-only ICP updates. No new downsampling or registration.",
        "visual": "Cached DINOv2-L/14 FP32 features, unchanged localization masks, white 224x224 inputs, original patch arithmetic.",
        "fusion": "0.25 top-five global mean + 0.25 same-view patch + 0.50 new geometry; no weight tuning.",
        "tie_policy": "Lowest template ID for views; lexical candidate ID for exact classification ties, with ties reported.",
        "runtime_scope": "Actual cached-feature comparison and shortlisted geometry verification; excludes prior encoding and pose estimation.",
        "gpu": torch.cuda.get_device_name(), "geometry_device": "CPU", "setup_s": setup_s,
        "source_sha256": provenance, "production_modified": False}
    save_json(output / "protocol.json", protocol)

    results = {}
    for strategy in STRATEGIES:
        strategy_start = perf_counter()
        pairs, view_records = [], []
        timing = {"global_s": 0., "patch_s": 0., "geometry_s": 0.}
        counts = {"global_view_comparisons": 0, "patch_view_comparisons": 0,
                  "geometry_shortlisted": 0, "geometry_rendered": 0, "empty_intersections": 0,
                  "union_fallback_pairs": 0}
        for cid in names:
            for oid, query in queries.items():
                tick = perf_counter()
                semantic = semantic_score(query["cls"], templates[cid]["cls"])
                np.testing.assert_allclose(semantic["cls_view_cosines"],
                    [pose_lookup[cid, oid, v]["cls_cosine"] for v in range(1, 43)], atol=2e-6, rtol=0)
                global_views = top_views(semantic["clamped_cls_view_scores"])
                timing["global_s"] += perf_counter() - tick
                counts["global_view_comparisons"] += 42
                patch_views_to_compute = global_views if strategy == "global_top5" else list(range(1, 43))
                torch.cuda.synchronize()
                tick = perf_counter()
                patches = {v: appearance_score(query["patch"], templates[cid]["patch"][v - 1])["appearance_score"]
                           for v in patch_views_to_compute}
                torch.cuda.synchronize()
                timing["patch_s"] += perf_counter() - tick
                counts["patch_view_comparisons"] += len(patches)
                for v, patch in patches.items():
                    assert abs(patch - pose_lookup[cid, oid, v]["patch"]) < 2e-6, "Cached patch parity failed."
                patch_views = sorted(patches, key=lambda v: (-patches[v], v))[:TOP_K]
                shared = intersect_views(global_views, patch_views)
                fallback = strategy == "global_patch_intersection" and not shared
                selected = global_views if strategy == "global_top5" else intersection_or_union(global_views, patch_views)
                counts["empty_intersections"] += int(fallback)
                counts["union_fallback_pairs"] += int(fallback)
                counts["geometry_shortlisted"] += len(selected)
                rows = []
                tick = perf_counter()
                for v in selected:
                    old = pose_lookup[cid, oid, v]
                    row = {"cad_id": cid, "object_id": oid, "view_index_1based": v,
                        "camera_T_cad": old["camera_T_cad"], "global_score": semantic["semantic_score"],
                        "cls_cosine": semantic["cls_view_cosines"][v - 1], "patch_score": patches[v],
                        "saved_pose_status": old["status"], "status": old["status"],
                        "geometry_score": None, "fused_score": None}
                    if old["status"] == "assessable":
                        render = cads[cid].render(np.asarray(old["camera_T_cad"]), K, depth.shape, bbox(masks[oid]), .5)
                        assert render is not None and not render["clipped"]
                        x1, y1, x2, y2 = render["roi"]
                        row.update(equal_penalty_scores(depth[y1:y2, x1:x2], masks[oid][y1:y2, x1:x2], render["depth"]))
                        counts["geometry_rendered"] += 1
                        if row["geometry_score"] is not None:
                            row["fused_score"] = .25 * row["global_score"] + .25 * row["patch_score"] + .5 * row["geometry_score"]
                    rows.append(row)
                timing["geometry_s"] += perf_counter() - tick
                view_records.extend(rows)
                best = winning_view(rows, "geometry_score")
                fused = winning_view(rows, "fused_score")
                pairs.append({"cad_id": cid, "object_id": oid, "global_top5": global_views,
                    "patch_top5": patch_views, "patch_ranking_scope": "global shortlist" if strategy == "global_top5" else "all 42 views",
                    "intersection_views": shared, "union_fallback": fallback,
                    "geometry_views": selected, "all_cls_cosines": semantic["cls_view_cosines"],
                    "computed_patch_scores": patches, "global_score": semantic["semantic_score"],
                    "geometry_score": best["geometry_score"] if best else None,
                    "geometry_winning_view": best["view_index_1based"] if best else None,
                    "fused_score": fused["fused_score"] if fused else None,
                    "fused_winning_view": fused["view_index_1based"] if fused else None,
                    "status": "assessable" if best else "no_assessable_selected_view"})
        results[strategy] = {"pairs": pairs, "views": view_records, "counts": counts,
                             "runtime": {**timing, "wall_s": perf_counter() - strategy_start}}

    # Read identities only after scoring; labels do not select views or poses.
    audit = read_json(DATASET / "identity_audit.json")
    cad_truth = {r["simulator_instance"]: r["object_id"] for r in audit}
    observation_truth = {v: k for k, v in cad_truth.items()}
    for result in results.values():
        result["evaluation"] = {score: {
            "cad_to_observation": rank_pairs(result["pairs"], cad_truth, "cad_id", "object_id", score),
            "observation_to_cad": rank_pairs(result["pairs"], observation_truth, "object_id", "cad_id", score)
        } for score in ("geometry_score", "fused_score")}
    baseline = read_json(POSE_SOURCE / "pair_scores.json")
    baseline_evaluation = {score: {
        "cad_to_observation": rank_pairs(baseline, cad_truth, "cad_id", "object_id", score),
        "observation_to_cad": rank_pairs(baseline, observation_truth, "object_id", "cad_id", score)
    } for score in ("geometry", "view_consistent_fused")}
    assert snapshot_hashes() == fixed
    assert all(content_hash(ROOT / p) == expected for p, expected in protected.items())
    assert all(content_hash(ROOT / p) == expected for p, expected in provenance.items())
    save_json(output / "results.json", {"strategies": results, "old_all42_baseline": baseline_evaluation,
        "old_baseline_note": "Different geometry formula and view budget; not an isolated shortlist comparison.",
        "fixed_dataset_and_source_caches_unchanged": True, "wall_including_setup_s": perf_counter() - start})
    print(json.dumps({"output": str(output), "primary_score": PRIMARY_SCORE,
        "results": {name: {"counts": r["counts"], "runtime": r["runtime"],
            "accuracy": {direction: f"{evaluation['correct']}/{evaluation['total']}"
                         for direction, evaluation in r["evaluation"][PRIMARY_SCORE].items()}}
                    for name, r in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
