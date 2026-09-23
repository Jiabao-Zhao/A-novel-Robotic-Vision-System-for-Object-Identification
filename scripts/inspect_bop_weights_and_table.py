"""Saved-score weight feasibility and table diagnostics for the same 136 masks."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
from scipy.optimize import linprog

from scripts.inspect_bop_mask_cad_oracles import ROOT, SOURCE, read_json
from scripts.inspect_bop_failure_causes import distribution


BASELINE = np.array([.25, .25, .5])
DATA = ROOT / "outputs/cache/bop_tless/data/tless"


def oracle_weights(features, labels, correct):
    """Maximize strict margin against every rival view, over each correct view."""
    rivals = features[labels != correct]
    best = None
    for index in np.flatnonzero(labels == correct):
        constraints = np.column_stack((rivals - features[index], np.ones(len(rivals))))
        result = linprog([0., 0., 0., -1.], A_ub=constraints, b_ub=np.zeros(len(rivals)),
                         A_eq=[[1., 1., 1., 0.]], b_eq=[1.],
                         bounds=[(0., 1.)] * 3 + [(None, None)], method="highs")
        assert result.success, result.message
        margin = float(result.x[3])
        if best is None or margin > best["margin"]:
            weights = result.x[:3]
            actual = float(features[index] @ weights - (rivals @ weights).max())
            np.testing.assert_allclose(actual, margin, atol=1e-8, rtol=0)
            best = {"weights_G_P_H": weights.tolist(), "margin": margin,
                    "strictly_recoverable": margin > 1e-8}
    if best is not None:
        weights = np.array(best["weights_G_P_H"])
        actual = float((features[labels == correct] @ weights).max() - (rivals @ weights).max())
        np.testing.assert_allclose(actual, best["margin"], atol=1e-8, rtol=0)
    return best


def main():
    start = perf_counter()
    audit = read_json(SOURCE / "failure_cause_diagnostic.json")
    frames = read_json(SOURCE / "per_frame.json")
    annotations = read_json(SOURCE / "coco_ground_truth.json")["annotations"]
    ann = {a["id"]: a for a in annotations}
    original = read_json(SOURCE / "hybrid_coco_predictions.json")
    prediction = {(r["image_id"], r["object_id"]): r for r in original}
    by_frame = defaultdict(list)
    for match in audit["selected_matches"]:
        by_frame[match["image_id"]].append(match)
    weights = np.array([[g / 100, p / 100, (100 - g - p) / 100]
                       for g in range(101) for p in range(101 - g)])
    results, grid_correct, meshes, hashes = [], [], {}, {}
    for image_id, matches in sorted(by_frame.items()):
        frame = frames[image_id - 1]
        name = f"scene_{frame['scene_id']:06}_{frame['im_id']:06}"
        plane = np.array(read_json(ROOT / "outputs/bop_localization_20260917" / name / "localization.json")["plane_model"])
        plane /= np.linalg.norm(plane[:3])
        if plane[3] < 0:
            plane *= -1
        scene = DATA / "test_primesense" / f"{frame['scene_id']:06}"
        poses = read_json(scene / "scene_gt.json")[str(frame["im_id"])]
        info = read_json(scene / "scene_gt_info.json")[str(frame["im_id"])]
        poses = [p for p, i in zip(poses, info) if i["px_count_visib"] > 0]
        frame_ann = [a for a in annotations if a["image_id"] == image_id]
        assert [a["category_id"] for a in frame_ann] == [p["obj_id"] for p in poses]
        gt_pose = {a["id"]: p for a, p in zip(frame_ann, poses)}
        depth_views = None
        for match in matches:
            oid, correct = match["object_id"], ann[match["gt_id"]]["category_id"]
            path = SOURCE / "matching" / name / "regions" / f"{oid}.json"
            if path.exists():
                raw = path.read_bytes()
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                views = json.loads(raw)["views"]
            else:
                if depth_views is None:
                    path = ROOT / "outputs/bop_sam_depth_20260919/matching" / name / "view_scores.json"
                    raw = path.read_bytes()
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                    depth_views = json.loads(raw)
                views = [r for r in depth_views if r["object_id"] == oid]
            groups = defaultdict(list)
            for view in views:
                groups[int(view["cad_id"].split("_")[1])].append(view)
            assert len(groups) == 30 and all(len(v) == 42 for v in groups.values())
            usable = sorted((v for v in views if v["H"] is not None),
                            key=lambda v: (v["cad_id"], v["view_index_1based"]))
            features = np.array([[v[b] for b in ("G", "P", "H")] for v in usable])
            labels = np.array([int(v["cad_id"].split("_")[1]) for v in usable])
            scores = features @ BASELINE
            winning = int(np.argmax(scores))
            saved = prediction[image_id, oid]
            assert int(labels[winning]) == saved["category_id"]
            np.testing.assert_allclose(scores[winning], saved["score"], atol=1e-12, rtol=0)
            grid_correct.append(labels[(features @ weights.T).argmax(axis=0)] == correct)
            oracle = oracle_weights(features, labels, correct)
            before = saved["category_id"] == correct
            if before:
                assert oracle is not None and oracle["strictly_recoverable"]
            record = {**match, "scene_id": frame["scene_id"], "im_id": frame["im_id"],
                      "true_cad_id": correct, "original_prediction": saved["category_id"],
                      "original_correct": before, "oracle": oracle}
            for label, w in (("global_only", [1, 0, 0]), ("patch_only", [0, 1, 0]),
                             ("geometry_only", [0, 0, 1]), ("equal_visual", [.5, .5, 0])):
                record[label] = int(labels[np.argmax(features @ np.array(w))])
            no_gate = {cid: max(.5 * (v["G"] + v["P"]) for v in rows) for cid, rows in groups.items()}
            with_gate = {cid: max(.5 * (v["G"] + v["P"]) for v in rows if v["pose_status"] == "assessable")
                         for cid, rows in groups.items() if any(v["pose_status"] == "assessable" for v in rows)}
            record["visual_before_table"] = min(no_gate, key=lambda c: (-no_gate[c], c))
            record["visual_after_table"] = min(with_gate, key=lambda c: (-with_gate[c], c))
            record["entire_wrong_CADs_rejected_by_table"] = sum(
                all(v["pose_status"] == "table_penetration" for v in rows)
                for cid, rows in groups.items() if cid != correct)
            true_views = groups[correct]
            record["true_view_status_counts"] = dict(Counter(v["pose_status"] for v in true_views))
            if correct not in meshes:
                meshes[correct] = np.asarray(o3d.io.read_triangle_mesh(str(DATA / "models_cad" / f"obj_{correct:06}.ply")).vertices) * .001
            vertices = meshes[correct]
            center = (vertices.min(0) + vertices.max(0)) / 2
            pose = gt_pose[match["gt_id"]]
            Rgt = np.array(pose["cam_R_m2c"]).reshape(3, 3)
            tgt = np.array(pose["cam_t_m2c"]) * .001
            cgt = Rgt @ center + tgt
            hgt = float(cgt @ plane[:3] + plane[3])
            extent_gt = float(((vertices - center) @ Rgt.T @ plane[:3]).min())
            fitted = []
            for v in true_views:
                if v["camera_T_cad"] is None:
                    continue
                T = np.array(v["camera_T_cad"])
                c = T[:3, :3] @ center + T[:3, 3]
                h = float(c @ plane[:3] + plane[3])
                extent = float(((vertices - center) @ T[:3, :3].T @ plane[:3]).min())
                fitted.append((h + extent, v, h, extent))
            if oracle is None:
                clearance, chosen, h, extent = max(fitted, key=lambda r: r[0])
            else:
                true_indices = np.flatnonzero(labels == correct)
                chosen = usable[true_indices[np.argmax(scores[true_indices])]]
                clearance, _, h, extent = next(r for r in fitted if r[1]["view_index_1based"] == chosen["view_index_1based"])
            record["pose_diagnostic"] = {"view": chosen["view_index_1based"],
                "selected_as": "least_penetrating" if oracle is None else "highest_baseline_fusion_for_true_CAD",
                "gt_clearance_mm": 1000 * (hgt + extent_gt), "fitted_clearance_mm": 1000 * clearance,
                "center_height_error_mm": 1000 * (h - hgt), "rotation_extent_error_mm": 1000 * (extent - extent_gt),
                "GT_center_with_fitted_rotation_passes": hgt + extent >= -.005,
                "GT_rotation_with_fitted_center_passes": h + extent_gt >= -.005,
                "p_depth": chosen.get("p_depth"), "p_obs": chosen.get("p_obs"), "p_cad": chosen.get("p_cad")}
            np.testing.assert_allclose(clearance - hgt - extent_gt, (h - hgt) + (extent - extent_gt), atol=1e-12)
            results.append(record)
        print(f"Audited scene {frame['scene_id']}: {len(matches)} masks", flush=True)
    success = np.array([r["original_correct"] for r in results])
    assert success.sum() == 96 and len(results) == 136
    grid_correct = np.array(grid_correct).T
    recovered = grid_correct[:, ~success].sum(1)
    broken = (~grid_correct[:, success]).sum(1)
    totals = grid_correct.sum(1)
    best = min(range(len(weights)), key=lambda i: (-totals[i], np.linalg.norm(weights[i] - BASELINE)))
    safe = min(np.flatnonzero(broken == 0), key=lambda i: (-recovered[i], np.linalg.norm(weights[i] - BASELINE)))
    def grid_row(i):
        return {"weights_G_P_H": weights[i].tolist(), "correct_of_136": int(totals[i]),
                "recovered_of_40": int(recovered[i]), "broken_of_96": int(broken[i])}
    summaries = {}
    for group, rows in (("success_96", [r for r in results if r["original_correct"]]),
                        ("failure_40", [r for r in results if not r["original_correct"]])):
        summaries[group] = {"true_CAD_fully_table_rejected": sum(r["true_view_status_counts"].get("table_penetration") == 42 for r in rows),
            "wrong_CADs_fully_table_rejected": sum(r["entire_wrong_CADs_rejected_by_table"] for r in rows),
            "wrong_CADs_removed_per_region": distribution([r["entire_wrong_CADs_rejected_by_table"] for r in rows]),
            "GT_pose_passes_table": sum(r["pose_diagnostic"]["gt_clearance_mm"] >= -5 for r in rows),
            "mask_iou": distribution([r["iou"] for r in rows]),
            "center_height_error_mm": distribution([r["pose_diagnostic"]["center_height_error_mm"] for r in rows]),
            "rotation_extent_error_mm": distribution([r["pose_diagnostic"]["rotation_extent_error_mm"] for r in rows]),
            "true_CAD_depth_penalty": distribution([r["pose_diagnostic"]["p_depth"] for r in rows if r["pose_diagnostic"]["p_depth"] is not None])}
    rejected = [r for r in results if r["oracle"] is None]
    summary = {"groups": summaries, "oracle_failures_recoverable": sum(not r["original_correct"] and r["oracle"] is not None and r["oracle"]["strictly_recoverable"] for r in results),
        "oracle_rejected_correct_CAD": len(rejected),
        "oracle_available_but_not_recoverable": sum(r["oracle"] is not None and not r["oracle"]["strictly_recoverable"] for r in results),
        "best_shared_grid": grid_row(best), "best_shared_grid_preserving_96": grid_row(safe),
        "fixed_components": {k: {"correct": sum(r[k] == r["true_cad_id"] for r in results),
            "recovered": sum(not r["original_correct"] and r[k] == r["true_cad_id"] for r in results),
            "broken": sum(r["original_correct"] and r[k] != r["true_cad_id"] for r in results)}
            for k in ("global_only", "patch_only", "geometry_only", "equal_visual")},
        "visual_table_comparison": {"before_correct": sum(r["visual_before_table"] == r["true_cad_id"] for r in results),
            "after_correct": sum(r["visual_after_table"] == r["true_cad_id"] for r in results),
            "helped": sum(r["visual_before_table"] != r["true_cad_id"] and r["visual_after_table"] == r["true_cad_id"] for r in results),
            "hurt": sum(r["visual_before_table"] == r["true_cad_id"] and r["visual_after_table"] != r["true_cad_id"] for r in results)},
        "rejected_pose_corrections": {k: sum(r["pose_diagnostic"][k] for r in rejected) for k in (
            "GT_center_with_fitted_rotation_passes", "GT_rotation_with_fitted_center_passes")}}
    result = {"scope": "Ground-truth-assisted diagnostics on one fixed mask per 136 matched targets, not AP or a deployable adaptive-weight model",
        "weight_protocol": "Nonnegative G/P/H weights summing to one. Exact LP per correct view against all rival CAD views, retaining the existing gate and cached geometry top-five. Winning fusion view may change. Shared weights exhaust a 1% simplex grid, not a continuous global optimum. No geometry computed for rejected/unvisited views.",
        "table_protocol": "Use all recorded table statuses. Unvisited lower-patch views are not counted as passing or failing. Visual-only gate comparison uses cached G plus max patch; it is not a full fusion-without-table experiment. Pose decomposition uses the CAD bounding-box center and ground-truth pose only for diagnosis.",
        "summary": summary, "cases": results, "input_view_sha256": hashes,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runtime_s": perf_counter() - start}
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (SOURCE / "weights_and_table_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
