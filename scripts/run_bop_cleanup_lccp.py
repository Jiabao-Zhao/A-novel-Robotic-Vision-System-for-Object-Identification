"""Audit cleanup and compare retained components with PCL LCCP on the frozen pilot.

Run scripts/build_lccp.sh first, then this module in the existing WSL GPU env.
All segmentation settings are frozen before masks are evaluated. CAD scoring is
imported unchanged; exact duplicate masks reuse their saved exhaustive scores.
"""

import hashlib
import os
import subprocess
from time import perf_counter

from scripts import run_bop_localization_benchmark as baseline
import cv2
import numpy as np
import open3d as o3d
import torch
from pycocotools import mask as mask_utils

from scripts.bop_segmentation_metrics import localization_metrics


ROOT = baseline.ROOT
BASE = baseline.OUTPUT
OUTPUT = ROOT / "outputs/bop_cleanup_lccp_20260918"
VARIANTS = ("keep_components", "lccp")
HELPER = ROOT / "outputs/cache/lccp/segment"
LIBRARY = ROOT / "outputs/cache/lccp/prefix/usr/lib/x86_64-linux-gnu"
read_json, save_json = baseline.read_json, baseline.save_json


def mask_of(cloud, intrinsics):
    return np.isfinite(baseline.project_with_z_buffer(cloud, intrinsics)[0])


def mask_hash(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def filter_outliers(localizer, cloud):
    """Instrument the unchanged two outlier operations before component selection."""
    config = localizer.config
    filtered = o3d.geometry.PointCloud(cloud)
    if len(filtered.points) >= config.statistical_outlier_neighbors:
        filtered, _ = filtered.remove_statistical_outlier(
            nb_neighbors=config.statistical_outlier_neighbors, std_ratio=config.statistical_outlier_std_ratio)
    if len(filtered.points) >= config.radius_outlier_min_neighbors:
        filtered, _ = filtered.remove_radius_outlier(
            nb_points=config.radius_outlier_min_neighbors, radius=config.radius_outlier_radius_m)
    return filtered


def components(cloud, labels, background=-1):
    groups = [(int(label), cloud.select_by_index(np.flatnonzero(labels == label).tolist()))
              for label in np.unique(labels) if label != background]
    return sorted(groups, key=lambda item: (-len(item[1].points), item[0]))


def child_accepted(localizer, cloud, intrinsics, rank):
    # Preserve one dominant piece as the baseline does. Additional pieces must
    # pass the same size/ROI rules that already admit initial object clusters.
    if rank == 0:
        return True
    sampled = cloud.voxel_down_sample(localizer.config.voxel_size_m)
    return (len(sampled.points) >= localizer.config.min_cluster_points
            and localizer.is_valid_object_cluster(sampled, intrinsics))


def lccp_labels(cloud, folder, parent_id):
    binary_in, binary_out = folder / f"{parent_id}.xyz", folder / f"{parent_id}.labels"
    points = np.asarray(cloud.points, dtype=np.float32)
    if not len(points):
        return np.zeros(0, np.uint32)
    points.tofile(binary_in)
    env = {**os.environ, "LD_LIBRARY_PATH": str(LIBRARY), "OMP_NUM_THREADS": "1"}
    with (folder / f"{parent_id}.log").open("w") as log:
        subprocess.run([str(HELPER), str(binary_in), str(binary_out)], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    labels = np.fromfile(binary_out, dtype=np.uint32)
    assert len(labels) == len(points)
    return labels


def save_regions(folder, candidates, plane, seconds):
    (folder / "masks").mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    for parent_id, label, mask in candidates:
        digest = mask_hash(mask)
        if not mask.any() or digest in seen:
            continue
        seen.add(digest)
        oid = f"object_{len(records) + 1:03}"
        baseline.save_image(folder / "masks" / f"{oid}.png", mask.astype(np.uint8) * 255)
        records.append({"object_id": oid, "parent_id": parent_id, "component_label": label,
                        "pixels": int(mask.sum()), "mask_sha256": digest})
    save_json(folder / "localization.json", {"regions": records, "plane_model": list(plane),
                                             "wall_s": seconds})


def segment_frame(item):
    name = baseline.frame_folder(item).name
    audit_folder = OUTPUT / "audit" / name
    if (audit_folder / "stages.npz").exists() and all(
            (OUTPUT / v / name / "localization.json").exists() for v in VARIANTS):
        return
    audit_folder.mkdir(parents=True, exist_ok=True)
    rgb, raw, scale, _, intrinsics = baseline.frame_input(item)
    start = perf_counter()
    o3d.utility.random.seed(baseline.SEED)
    localizer = baseline.PointCloudLocalization(baseline.PointCloudConfig())
    workspace = localizer.rgbd_to_pointcloud(rgb, raw, scale, intrinsics)
    downsampled = workspace.voxel_down_sample(localizer.config.voxel_size_m)
    objects, plane, _ = localizer.segment_table_plane(downsampled)
    clusters = localizer.cluster_objects(objects, intrinsics, plane)
    raw_objects = localizer.remove_table_from_raw_cloud(workspace, plane)
    old = read_json(BASE / name / "localization.json")
    np.testing.assert_allclose(plane, old["plane_model"], atol=1e-12, rtol=0)
    assert len(clusters) == old["initial_cluster_count"]
    stages = {"measured": mask_of(workspace, intrinsics), "non_table": mask_of(raw_objects, intrinsics)}
    candidates = {v: [] for v in VARIANTS}
    parent_records = []
    lccp_seconds = 0.
    for i, cluster in enumerate(clusters, 1):
        parent = f"object_{i:03}"
        crop = localizer.crop_raw_cluster(raw_objects, cluster)
        filtered = filter_outliers(localizer, crop)
        labels = np.full(len(filtered.points), -1)
        if len(filtered.points) >= localizer.config.raw_cluster_dbscan_min_points:
            labels = np.asarray(localizer.clustering_cloud(filtered, plane).cluster_dbscan(
                eps=localizer.config.raw_cluster_dbscan_eps_m,
                min_points=localizer.config.raw_cluster_dbscan_min_points, print_progress=False))
        groups = components(filtered, labels)
        retained = groups[0][1] if groups else filtered
        if retained.is_empty():
            retained = crop
        reference = localizer.clean_raw_cluster(crop, plane)
        np.testing.assert_array_equal(np.asarray(retained.points), np.asarray(reference.points))
        old_mask = cv2.imread(str(BASE / name / "masks" / f"{parent}.png"), 0) > 0
        np.testing.assert_array_equal(mask_of(retained, intrinsics), old_mask)
        stages[f"{parent}_crop"] = mask_of(crop, intrinsics)
        stages[f"{parent}_filtered"] = mask_of(filtered, intrinsics)
        stages[f"{parent}_retained"] = old_mask
        rejected = np.zeros(raw.shape, bool)
        children = []
        if not groups:
            candidates["keep_components"].append((parent, -1, old_mask))
        for rank, (label, child) in enumerate(groups):
            mask = mask_of(child, intrinsics)
            accepted = child_accepted(localizer, child, intrinsics, rank)
            if rank:
                rejected |= mask
            children.append({"label": label, "points": len(child.points), "pixels": int(mask.sum()),
                             "baseline_kept": rank == 0, "additional_candidate_accepted": accepted})
            if accepted:
                candidates["keep_components"].append((parent, label, mask))
        stages[f"{parent}_discarded_components"] = rejected
        stages[f"{parent}_noise"] = mask_of(filtered.select_by_index(np.flatnonzero(labels < 0).tolist()), intrinsics)
        stage = perf_counter()
        lccp = lccp_labels(filtered, audit_folder, parent)
        pieces = components(filtered, lccp, background=0)
        lccp_children = []
        for rank, (label, child) in enumerate(pieces):
            accepted = child_accepted(localizer, child, intrinsics, rank)
            lccp_children.append({"label": label, "points": len(child.points), "accepted": accepted})
            if accepted:
                candidates["lccp"].append((parent, label, mask_of(child, intrinsics)))
        lccp_seconds += perf_counter() - stage
        parent_records.append({"parent_id": parent, "raw_crop_points": len(crop.points),
            "after_outliers_points": len(filtered.points), "baseline_retained_points": len(retained.points),
            "dbscan_children": children, "lccp_children": lccp_children,
            "lccp_unassigned_points": int(np.count_nonzero(lccp == 0))})
    elapsed = perf_counter() - start
    save_regions(OUTPUT / "keep_components" / name, candidates["keep_components"], plane, elapsed - lccp_seconds)
    save_regions(OUTPUT / "lccp" / name, candidates["lccp"], plane, elapsed)
    np.savez_compressed(audit_folder / "stages.npz", **stages)
    save_json(audit_folder / "trace.json", {"parents": parent_records, "baseline_exactly_reproduced": True,
        "instrumented_wall_s": elapsed, "lccp_helper_s": lccp_seconds,
        "runtime_note": "Instrumented audit includes duplicate baseline checks; not optimized production latency"})
    print(name, {v: len(read_json(OUTPUT / v / name / "localization.json")["regions"]) for v in VARIANTS}, flush=True)


def audit_saved(plan):
    """Annotations only evaluate already-frozen masks and discarded support."""
    gt = read_json(BASE / "coco_ground_truth.json")
    prior = read_json(BASE / "localization_metrics.json")
    rows = []
    for ordinal, (item, old) in enumerate(zip(plan, prior, strict=True), 1):
        name = baseline.frame_folder(item).name
        annotations = [a for a in gt["annotations"] if a["image_id"] == ordinal and not a["ignore"]]
        matched = {r["gt_id"] for r in old["iou_0.50"]["matches"]}
        with np.load(OUTPUT / "audit" / name / "stages.npz") as data:
            stage_groups = {suffix: {k: data[k] for k in data.files if k.endswith("_" + suffix)}
                            for suffix in ("crop", "filtered", "retained", "discarded_components", "noise")}
            unions = {k: np.logical_or.reduce(list(v.values())) if v else np.zeros_like(data["measured"])
                      for k, v in stage_groups.items()}
            unions.update({k: data[k] for k in ("measured", "non_table")})
            for ann in annotations:
                mask = mask_utils.decode(ann["segmentation"]).astype(bool)
                coverage = {k: float((mask & support).sum() / mask.sum()) for k, support in unions.items()}
                rows.append({**item, "gt_id": ann["id"], "cad_id": ann["category_id"],
                             "baseline_matched_iou50": ann["id"] in matched, "coverage": coverage})
    missing = [r for r in rows if not r["baseline_matched_iou50"]]
    summary = {"objects": len(rows), "baseline_missed_iou50": len(missing),
        "misses_with_at_least_half_gt_in_discarded_components": sum(r["coverage"]["discarded_components"] >= .5 for r in missing),
        "note": "Coverage is diagnostic overlap, not proof of a unique failure cause; losses can occur at multiple stages"}
    save_json(OUTPUT / "cleanup_audit.json", {"summary": summary, "objects": rows})
    print("CLEANUP AUDIT", summary, flush=True)


def load_masks(folder):
    records = read_json(folder / "localization.json")["regions"]
    return {r["object_id"]: cv2.imread(str(folder / "masks" / f"{r['object_id']}.png"), 0) > 0 for r in records}


def score_variant(item, variant, engine):
    name = baseline.frame_folder(item).name
    folder = OUTPUT / variant / name
    if (folder / "predictions.json").exists():
        return
    masks = load_masks(folder)
    sources = [BASE / name]
    if variant == "lccp":
        sources.append(OUTPUT / "keep_components" / name)
    cache = {}
    for source in sources:
        source_pairs = read_json(source / "pair_scores.json")
        source_views = read_json(source / "all_view_scores.json")
        for oid, mask in load_masks(source).items():
            cache[mask_hash(mask)] = (source, oid,
                [r for r in source_pairs if r["object_id"] == oid],
                [r for r in source_views if r["object_id"] == oid])
    novel, pairs, views, reused = {}, [], [], []
    for oid, mask in masks.items():
        saved = cache.get(mask_hash(mask))
        if saved is None:
            novel[oid] = mask
        else:
            source, old_id, pp, vv = saved
            pairs.extend({**r, "object_id": oid} for r in pp)
            views.extend({**r, "object_id": oid} for r in vv)
            reused.append({"object_id": oid, "source": str(source.relative_to(ROOT)), "source_object_id": old_id})
    rgb, raw, scale, K, _ = baseline.frame_input(item)
    start = perf_counter()
    if novel:
        baseline.score_scene(*engine, raw.astype(float) * scale, novel, K, rgb, {}, folder / "new_scores", offset=0.)
        pairs.extend(read_json(folder / "new_scores/pair_scores.json"))
        views.extend(read_json(folder / "new_scores/all_view_scores.json"))
    assert len(pairs) == len(masks) * 30
    assert len(views) == len(masks) * 30 * 42
    save_json(folder / "pair_scores.json", pairs)
    save_json(folder / "all_view_scores.json", views)
    seconds = perf_counter() - start
    save_json(folder / "predictions.json", baseline.export_predictions(item, masks, pairs, seconds))
    save_json(folder / "inference_runtime.json", {"new_scoring_s": seconds, "new_masks": len(novel),
        "reused_masks": reused, "note": "Cache-assisted experiment timing; excludes prior cached scoring, localization and shared setup"})
    print(name, variant, "scored", len(novel), "reused", len(reused), flush=True)


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = read_json(BASE / "protocol.json")
    protocol = {"plan": prior["plan"], "baseline": str(BASE.relative_to(ROOT)), "variants": list(VARIANTS),
        "localization": prior["localization"],
        "lccp": {"pcl_version": "1.12.1", "voxel_m": .003, "seed_m": .03, "color": 0, "spatial": 1, "normal": 4,
                 "concavity_deg": 10, "smoothness": .1, "sanity": False, "k_factor": 0, "min_segment_size": 0,
                 "single_camera_transform": False, "supervoxel_refinement": False},
        "lccp_rationale": "PCL 1.12.1 example settings; metric voxel uses existing 3mm localizer resolution; no tuning",
        "scope": "Replace final component selection inside each unchanged accepted coarse DBSCAN cluster, after identical outlier removal",
        "child_rule": "Keep largest child, as baseline does; additional children pass existing 100 downsampled-point and shape/ROI filters; exact duplicate masks removed",
        "unchanged": "All 30 CADs, 42 templates, DINOv2-L/14, global/patch/surface scores, pose search and fusion",
        "evaluation": "Same BOP evaluator and GT masks; annotations never passed to segmentation or scoring",
        "protected_sha256": prior["protected_sha256"],
        "experiment_sha256": {p: baseline.association.content_hash(ROOT / p) for p in
            ("scripts/run_bop_cleanup_lccp.py", "scripts/lccp_segment.cpp", "scripts/build_lccp.sh")},
        "helper_sha256": baseline.association.content_hash(HELPER)}
    if (OUTPUT / "protocol.json").exists():
        assert read_json(OUTPUT / "protocol.json") == protocol, "Frozen protocol changed"
    else:
        save_json(OUTPUT / "protocol.json", protocol)
    assert prior["protected_sha256"] == {p: baseline.association.content_hash(ROOT / p) for p in baseline.PROTECTED}
    start = perf_counter()
    for item in protocol["plan"]:
        segment_frame(item)
    audit_saved(protocol["plan"])
    engine = baseline.cached_templates()
    for item in protocol["plan"]:
        for variant in VARIANTS:
            score_variant(item, variant, engine)
    results = {"baseline": read_json(BASE / "summary.json")}
    try:
        for variant in VARIANTS:
            baseline.OUTPUT = OUTPUT / variant
            results[variant] = baseline.evaluate_saved(protocol["plan"])
            save_json(baseline.OUTPUT / "summary.json", results[variant])
    finally:
        baseline.OUTPUT = BASE
    assert prior["protected_sha256"] == {p: baseline.association.content_hash(ROOT / p) for p in baseline.PROTECTED}
    save_json(OUTPUT / "comparison.json", {"variants": results, "wall_s_this_invocation": perf_counter() - start,
        "protected_sources_unchanged": True, "tuning_performed": False})
    print({v: {"localization": r["localization"], "segmentation_AP": {m: s["AP"] for m, s in r["segmentation"].items()}}
           for v, r in results.items()}, flush=True)


if __name__ == "__main__":
    main()
