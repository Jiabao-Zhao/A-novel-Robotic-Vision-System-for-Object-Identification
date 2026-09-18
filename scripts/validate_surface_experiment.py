"""Compact numerical audit/comparison of completed surface experiments."""

from collections import Counter

import cv2
import numpy as np

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, save_json, snapshot_hashes
from scripts.run_wrist_large_encoder import ORACLE
from scripts.run_wrist_sam6d_patch_ablation import LARGE
from scripts.run_surface_verification import OUTPUT, LOCAL_MASKS, SAVED_SAM, METHODS, evaluate


def audit_scores(folder):
    pairs, views = read_json(folder / "pair_scores.json"), read_json(folder / "all_view_scores.json")
    assert read_json(folder / "evaluation.json")["scoring_version"] == 2
    groups = {}
    for r in views:
        groups.setdefault((r["cad_id"], r["object_id"]), []).append(r)
        if r["status"] != "assessable":
            assert r["view_consistent_fused"] is None
        else:
            assert 0 <= r["surface_score"] <= r["mask_iou"] <= 1
            np.testing.assert_allclose(r["view_consistent_fused"], .25 * r["global"] + .25 * r["patch"] + .5 * r["surface_score"], atol=1e-12)
    for pair in pairs:
        key = pair["cad_id"], pair["object_id"]
        rows = groups.get(key, [])
        if rows:
            assert sorted(r["view_index_1based"] for r in rows) == list(range(1, 43))
        usable = [r for r in rows if r["status"] == "assessable"]
        assert len(usable) == pair["admissible_view_count"]
        for field, value, view in (("surface_score", "geometry", "geometry_view"),
                                    ("view_consistent_fused", "view_consistent_fused", "fused_view")):
            best = max(usable, key=lambda r: r[field]) if usable else None
            assert pair[value] == (best[field] if best else None)
            assert pair[view] == (best["view_index_1based"] if best else None)
        np.testing.assert_allclose(pair["global"], np.sort(np.clip(pair["cls_view_cosines"], 0, 1))[-5:].mean(), atol=1e-7)
    return {"pair_count": len(pairs), "view_count": len(views), "view_status_counts": dict(Counter(r["status"] for r in views)),
        "pairs_without_admissible_pose": sum(r["geometry"] is None for r in pairs),
        "selected_pose_changed_from_cls": sum(r["fused_view"] is not None and r["fused_view"] != r["best_view_index_1based"] for r in pairs)}


def distributions(rows, truth, field):
    result = {}
    for label, correct in (("correct", True), ("incorrect", False)):
        scores = [r[field] for r in rows if (r["object_id"] in truth.get(r["cad_id"], [])) == correct and r.get(field) is not None]
        result[label] = {"count": len(scores), "min": min(scores) if scores else None,
            "median": float(np.median(scores)) if scores else None, "max": max(scores) if scores else None,
            "mean": float(np.mean(scores)) if scores else None, "exactly_one": sum(v == 1 for v in scores)}
    return result


def main():
    truth = {r["simulator_instance"]: [r["object_id"]] for r in read_json(DATASET / "identity_audit.json")}
    identity = {oid: cid for cid, oids in truth.items() for oid in oids}
    base = read_json(OUTPUT / "stage2_pose_search/pair_scores.json")
    sam = read_json(OUTPUT / "stage3_saved_sam/pair_scores.json")
    common = {r["object_id"] for r in sam}
    included = {cid: oids for cid, oids in truth.items() if oids[0] in common}
    comparison = {"sam_same_five_candidates": {"localization": evaluate([r for r in base if r["object_id"] in common], included),
                     "saved_sam3": evaluate(sam, included)}, "mask_quality": []}
    for cid, oids in truth.items():
        oid = oids[0]
        reference = cv2.imread(str(ORACLE / "diagonal" / cid / "cad_mask.png"), 0) > 0
        row = {"cad_id": cid, "object_id": oid}
        for mode, folder in (("localization", LOCAL_MASKS), ("saved_sam3", SAVED_SAM / "masks")):
            path = folder / f"{oid}.png"
            mask = cv2.imread(str(path), 0) > 0 if path.exists() else np.zeros_like(reference)
            row[mode] = {"present": path.exists(), "pixels": int(mask.sum()),
                "iou_to_verified_cad_silhouette": float((mask & reference).sum() / (mask | reference).sum())}
        comparison["mask_quality"].append(row)
    original = ROOT / "outputs/sam6d_patch_wrist_association_2026-09-11/20260917T181518155829Z"
    old_rows = read_json(original / "pair_scores.json")["blenderproc42/localization"]
    old = {(r["cad_id"], r["object_id"]): r for r in old_rows}
    comparison["cpu_gpu_baseline"] = {
        "max_global_score_change": max(abs(r["global"] - old[r["cad_id"], r["object_id"]]["semantic_score"]) for r in base),
        "max_patch_score_change": max(abs(r["patch"] - old[r["cad_id"], r["object_id"]]["appearance_score"]) for r in base),
        "best_cls_view_changes": sum(r["best_view_index_1based"] != old[r["cad_id"], r["object_id"]]["best_view_index_1based"] for r in base)}
    cpu = [{**r, "global": r["semantic_score"], "patch": r["appearance_score"],
            "visual": r["semantic_plus_appearance_score"]} for r in old_rows]
    a = evaluate(base, truth, ("global", "patch", "visual"))["targets"]
    b = evaluate(cpu, truth, ("global", "patch", "visual"))["targets"]
    comparison["cpu_gpu_baseline"]["top1_prediction_changes"] = sum(x["prediction"] != y["prediction"] for x, y in zip(a, b, strict=True))
    old_inputs = read_json(LARGE / "inputs.json")
    assert all(association.content_hash(OUTPUT / "stage2_pose_search/inputs" / f"{oid}.png") ==
               old_inputs[f"observed/localization/{oid}"]["sha256"] for oid in identity)
    old_reg = {(r["cad_id"], r["object_id"], r["view_index_1based"]): r for r in read_json(OUTPUT / "stage2_pose_search/registrations.json")}
    for r in read_json(OUTPUT / "stage3_saved_sam/registrations.json"):
        assert r["camera_T_cad"] == old_reg[r["cad_id"], r["object_id"], r["view_index_1based"]]["camera_T_cad"]
    comparison["stages"] = {}
    for stage in ("stage2_pose_search", "stage3_saved_sam", "stage3_fresh_sam"):
        folder = OUTPUT / stage
        rows = read_json(folder / "pair_scores.json")
        result = read_json(folder / "evaluation.json")
        comparison["stages"][stage] = {"summary": result["summary"],
            "failures": [{"cad_id": r["cad_id"], "method": r["method"], "prediction_identity": identity.get(r["prediction"]),
                           "correct_score": r["correct_score"], "incorrect_score": r["incorrect_score"], "margin": r["margin"]}
                         for r in result["targets"] if not r["correct"]],
            "distributions": {m: distributions(rows, truth, m) for m in ("geometry", "view_consistent_fused")}}
    bank = read_json(OUTPUT / "stage2_pose_search/all_view_scores.json")
    geometry_rows = []
    for pair in base:
        usable = [r for r in bank if r["cad_id"] == pair["cad_id"] and r["object_id"] == pair["object_id"] and r["status"] == "assessable"]
        geometry_rows.append({"cad_id": pair["cad_id"], "object_id": pair["object_id"],
            **{m: max(r[m] for r in usable) if usable else None for m in ("box_iou", "mask_iou", "surface_score", "registration_fitness")}})
    comparison["same_pose_bank_geometry_comparison"] = evaluate(geometry_rows, truth,
            ("box_iou", "mask_iou", "surface_score", "registration_fitness"))
    layouts = []
    for path in sorted((OUTPUT / "stage4_layouts").glob("*/evaluation.json")):
        layouts.extend({"scene": path.parent.name, **r} for r in read_json(path)["targets"])
    comparison["new_layout_summary"] = {m: {"correct": sum(r["correct"] for r in layouts if r["method"] == m),
        "total": sum(r["method"] == m for r in layouts),
        "margin_count": sum(r["method"] == m and r["margin"] is not None for r in layouts),
        "mean_margin": float(np.mean([r["margin"] for r in layouts if r["method"] == m and r["margin"] is not None]))}
        for m in METHODS}
    comparison["stage1_verified_pose"] = read_json(OUTPUT / "stage1_verified_pose/evaluation.json")
    pilot = OUTPUT / "stage4_tless"
    if (pilot / "summary.json").exists():
        diagnostic = read_json(pilot / "gt_pose_diagnostic.json")
        comparison["tless_pilot"] = {"cad_to_region": read_json(pilot / "cad_to_region_summary.json"),
            "region_to_all_30_cads": read_json(pilot / "summary.json"),
            "limits": "3 predeclared real frames, 28 instances, 14 known-CAD queries; provided GT masks; not official BOP evaluation",
            "post_evaluation_gt_pose_diagnostic": {
                "mean_correct_cad_geometry_estimated_pose": float(np.mean([r["geometry_with_estimated_pose"] for r in diagnostic])),
                "mean_correct_cad_geometry_gt_pose": float(np.mean([r["geometry_at_supplied_gt_pose"]["surface_score"] for r in diagnostic])),
                "improved_with_gt_pose": sum(r["geometry_at_supplied_gt_pose"]["surface_score"] > r["geometry_with_estimated_pose"] for r in diagnostic)}}
    comparison["runtime"] = {"original_scene_geometry": read_json(OUTPUT / "stage2_pose_search/geometry_runtime.json"),
        "original_scene_dino": read_json(OUTPUT / "stage2_pose_search/inputs/features/runtime.json"),
        "existing_252_template_dino": read_json(OUTPUT / "template_features/runtime.json"),
        "new_layout_capture_and_localization_s": sum(read_json(p)["capture_and_input_preparation_s"] for p in (OUTPUT / "new_layouts").glob("*/capture_complete.json"))}
    save_json(OUTPUT / "comparison.json", comparison)
    folders = [OUTPUT / "stage2_pose_search", OUTPUT / "stage3_saved_sam", OUTPUT / "stage3_fresh_sam"] + sorted((OUTPUT / "stage4_layouts").glob("heldout_*"))
    folders += sorted((OUTPUT / "stage4_tless").glob("scene_*"))
    validation = {str(p.relative_to(OUTPUT)): audit_scores(p) for p in folders}
    snapshot_hashes()
    prior = read_json(ROOT / "outputs/layout_wrist_association_2026-09-11/20260917T195444Z/protocol.json")
    assert all(association.content_hash(ROOT / p) == h for p, h in prior["protected_source_sha256"].items())
    save_json(OUTPUT / "validation.json", {"stages": validation, "original_input_images_identical": True,
        "sam_transforms_identical": True, "fixed_dataset_and_production_unchanged": True,
        "source_sha256": {p.name: association.content_hash(p) for p in [ROOT / "scripts/surface_verification.py",
            ROOT / "scripts/run_surface_verification.py", ROOT / "scripts/run_tless_surface_pilot.py"]}})
    print("new layouts", comparison["new_layout_summary"], flush=True)
    print("CPU/GPU", comparison["cpu_gpu_baseline"], flush=True)


if __name__ == "__main__":
    main()
