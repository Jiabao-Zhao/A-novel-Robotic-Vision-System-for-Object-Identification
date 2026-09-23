"""Exploratory depth-only parameter check on the existing 20-frame pilot.

Use the existing keep-components experiment; do not change production defaults.
Annotations evaluate frozen masks only. These localization counts are not AP.
"""

from dataclasses import asdict
from time import perf_counter

from scripts import run_bop_cleanup_lccp as cleanup
from scripts.bop_segmentation_metrics import localization_metrics
import cv2
import numpy as np
import open3d as o3d
from pycocotools import mask as mask_utils


baseline = cleanup.baseline
OUTPUT = baseline.ROOT / "outputs/localization_parameters_20260918"
# Set before running/evaluating any variant. No automatic parameter selection.
VARIANTS = {
    "current": {},
    "coarse_10mm": {"dbscan_eps_m": .010},
    "cleanup_4mm": {"raw_cluster_dbscan_eps_m": .004},
    "cleanup_3mm": {"raw_cluster_dbscan_eps_m": .003},
    "cleanup_4mm_min10": {"raw_cluster_dbscan_eps_m": .004,
                          "raw_cluster_dbscan_min_points": 10},
    "table_3mm": {"plane_distance_threshold_m": .003},
    "coarse_10mm_cleanup_4mm": {"dbscan_eps_m": .010,
                               "raw_cluster_dbscan_eps_m": .004},
}


def segment(rgb, raw, scale, intrinsics, changes):
    start = perf_counter()
    o3d.utility.random.seed(baseline.SEED)
    localizer = baseline.PointCloudLocalization(baseline.PointCloudConfig(**changes))
    workspace = localizer.rgbd_to_pointcloud(rgb, raw, scale, intrinsics)
    sampled = workspace.voxel_down_sample(localizer.config.voxel_size_m)
    objects, plane, _ = localizer.segment_table_plane(sampled)
    clusters = localizer.cluster_objects(objects, intrinsics, plane)
    raw_objects = localizer.remove_table_from_raw_cloud(workspace, plane)
    candidates = []
    for i, cluster in enumerate(clusters, 1):
        parent = f"object_{i:03}"
        crop = localizer.crop_raw_cluster(raw_objects, cluster)
        filtered = cleanup.filter_outliers(localizer, crop)
        labels = np.full(len(filtered.points), -1)
        if len(filtered.points) >= localizer.config.raw_cluster_dbscan_min_points:
            labels = np.asarray(localizer.clustering_cloud(filtered, plane).cluster_dbscan(
                eps=localizer.config.raw_cluster_dbscan_eps_m,
                min_points=localizer.config.raw_cluster_dbscan_min_points,
                print_progress=False))
        groups = cleanup.components(filtered, labels)
        if not groups:
            child = crop if filtered.is_empty() else filtered
            candidates.append((parent, -1, cleanup.mask_of(child, intrinsics)))
        for rank, (label, child) in enumerate(groups):
            if cleanup.child_accepted(localizer, child, intrinsics, rank):
                candidates.append((parent, label, cleanup.mask_of(child, intrinsics)))
    return candidates, plane, perf_counter() - start


def draw_panel(rgb, masks, title):
    """Diagnostic mask overlay; each color denotes a proposed region, not a CAD."""
    panel = rgb.copy()
    for i, mask in enumerate(masks.values()):
        color = np.array(((73*i+40) % 200+40, (117*i+90) % 200+40,
                          (157*i+20) % 200+40))
        panel[mask] = (.5 * panel[mask] + .5 * color).astype(np.uint8)
        y, x = np.nonzero(mask)
        if len(x):
            cv2.putText(panel, str(i+1), (int(np.median(x)), int(np.median(y))),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2)
    panel = cv2.copyMakeBorder(panel, 44, 0, 0, 0, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255))
    cv2.putText(panel, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
    return panel


def main():
    prior = cleanup.read_json(baseline.OUTPUT / "protocol.json")
    gt = cleanup.read_json(baseline.OUTPUT / "coco_ground_truth.json")
    protected = {p: baseline.association.content_hash(baseline.ROOT / p)
                 for p in baseline.PROTECTED}
    assert protected == prior["protected_sha256"]
    config = asdict(baseline.PointCloudConfig())
    config.pop("output_dir")
    config.pop("annotation_dir")
    protocol = {"purpose": "Exploratory localization-only parameter sensitivity; not AP",
                "plan": prior["plan"], "base": "keep_components", "config": config,
                "variants": VARIANTS, "seed": baseline.SEED,
                "protected_sha256": protected,
                "script_sha256": baseline.association.content_hash(__file__)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "protocol.json").exists():
        assert cleanup.read_json(OUTPUT / "protocol.json") == protocol
    else:
        cleanup.save_json(OUTPUT / "protocol.json", protocol)
    rows, panels = [], []
    start = perf_counter()
    for ordinal, item in enumerate(prior["plan"], 1):
        rgb, raw, scale, _, intrinsics = baseline.frame_input(item)
        annotations = [a for a in gt["annotations"] if a["image_id"] == ordinal]
        name = baseline.frame_folder(item).name
        if item["scene_id"] == 13:
            truth = {str(a["id"]): mask_utils.decode(a["segmentation"]).astype(bool)
                     for a in annotations if not a["ignore"]}
            panels.append(draw_panel(rgb, truth, "Ground truth: 8 objects (numbers are region indices)"))
        counts = {}
        for variant, changes in VARIANTS.items():
            folder = OUTPUT / variant / name
            if not (folder / "localization.json").exists():
                candidates, plane, seconds = segment(rgb, raw, scale, intrinsics, changes)
                cleanup.save_regions(folder, candidates, plane, seconds)
            masks = cleanup.load_masks(folder)
            if variant == "current":
                saved = cleanup.load_masks(cleanup.OUTPUT / "keep_components" / name)
                assert list(masks) == list(saved), f"Baseline count differs: {name}"
                for oid in masks:
                    np.testing.assert_array_equal(masks[oid], saved[oid])
            metrics = localization_metrics(masks, annotations)
            seconds = cleanup.read_json(folder / "localization.json")["wall_s"]
            rows.append({**item, "variant": variant, "wall_s": seconds, **metrics})
            counts[variant] = metrics["iou_0.50"]["matched"]
            if item["scene_id"] == 13:
                title = f"{variant}: {counts[variant]}/8 matched, {len(masks)} regions"
                panel = draw_panel(rgb, masks, title)
                panels.append(panel)
                baseline.save_image(folder / "overlay.png", panel)
        print(name, counts, flush=True)
    summary = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "objects": sum(r["eligible_instances"] for r in selected),
            "regions": sum(r["predicted_regions"] for r in selected),
            "matched_iou50": sum(r["iou_0.50"]["matched"] for r in selected),
            "matched_iou75": sum(r["iou_0.75"]["matched"] for r in selected),
            "unmatched_predictions_iou50": sum(r["iou_0.50"]["unmatched_predictions"] for r in selected),
            "merge_flags": sum(len(r["suspected_merged_regions"]) for r in selected),
            "split_or_duplicate_flags": sum(len(r["fragmented_or_duplicate_gt_ids"]) for r in selected),
            "segmentation_seconds": sum(r["wall_s"] for r in selected),
        }
    assert protected == {p: baseline.association.content_hash(baseline.ROOT / p)
                         for p in baseline.PROTECTED}
    cleanup.save_json(OUTPUT / "results.json", {
        "summary": summary, "frames": rows, "baseline_masks_exactly_reproduced": True,
        "production_sources_unchanged": True, "invocation_wall_s": perf_counter() - start})
    baseline.save_image(OUTPUT / "scene13_parameter_comparison.png", np.concatenate(
        [np.concatenate(panels[i:i+2], axis=1) for i in range(0, len(panels), 2)], axis=0))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
