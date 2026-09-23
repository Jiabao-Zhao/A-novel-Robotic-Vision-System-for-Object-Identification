"""Compare fixed branch subsets and selection orders using saved wrist scenes.

No new DINO, patch extraction, localization, alignment, thresholds or weights.
Render the new geometry score once for each admissible saved pose, then reuse it.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from scripts.run_wrist_shortlist_geometry import (
    ROOT, SOURCE, POSE_SOURCE, DATASET, MASKS, rank_pairs, intersection_or_union,
)


OUTPUT = ROOT / "outputs/wrist_branch_order"
WEIGHTS = {"G": .25, "P": .25, "H": .50}
METHODS = {
    **{f"{branches}_all": {"branches": branches, "selection": "all", "candidate_top_k": None}
       for branches in ("G", "P", "GP", "H", "GH", "PH", "GPH")},
    **{name: {"branches": "GPH", "selection": selection, "candidate_top_k": None}
       for name, selection in (
           ("global5_then_geometry", "global5"),
           ("patch5_then_geometry", "patch5"),
           ("intersection_then_geometry", "intersection"),
           ("table_then_global5", "table_global5"),
           ("table_then_patch5", "table_patch5"),
           ("table_then_intersection", "table_intersection"),
           ("geometry5_then_visual", "geometry5"),
           ("geometry10_then_global5", "geometry10_global5"))},
    **{f"geometry_candidates{k}_then_global5": {
        "branches": "GPH", "selection": "table_global5", "candidate_top_k": k} for k in (3, 5)},
}


def top(rows, field, count=5):
    return sorted((r for r in rows if r.get(field) is not None),
                  key=lambda r: (-r[field], r["view_index_1based"]))[:count]


def intersect_rows(first, second):
    lookup = {r["view_index_1based"]: r for r in first + second}
    ids = intersection_or_union([r["view_index_1based"] for r in first],
                                [r["view_index_1based"] for r in second])
    return [lookup[v] for v in ids]


def select_views(rows, selection):
    admissible = [r for r in rows if r["pose_status"] == "assessable"]
    selectors = {
        "all": lambda: rows,
        "global5": lambda: top(rows, "cls"),
        "patch5": lambda: top(rows, "P"),
        "intersection": lambda: intersect_rows(top(rows, "cls"), top(rows, "P")),
        "table_global5": lambda: top(admissible, "cls"),
        "table_patch5": lambda: top(admissible, "P"),
        "table_intersection": lambda: intersect_rows(top(admissible, "cls"), top(admissible, "P")),
        "geometry5": lambda: top(rows, "H"),
        "geometry10_global5": lambda: top(top(rows, "H", 10), "cls"),
    }
    return selectors[selection]()


def combine(row, branches):
    if any(row.get(branch) is None for branch in branches):
        return None
    return sum(WEIGHTS[b] * row[b] for b in branches) / sum(WEIGHTS[b] for b in branches)


def pair_result(rows, method):
    selected = select_views(rows, method["selection"])
    usable = [(r, combine(r, method["branches"])) for r in selected]
    usable = [(r, score) for r, score in usable if score is not None]
    winner = min(usable, key=lambda item: (-item[1], item[0]["view_index_1based"])) if usable else None
    geometry = top(rows, "H", 1)
    return {"cad_id": rows[0]["cad_id"], "object_id": rows[0]["object_id"],
            "selected_views": [r["view_index_1based"] for r in selected],
            "usable_selected_views": [r["view_index_1based"] for r, _ in usable],
            "score": winner[1] if winner else None,
            "winning_view": winner[0]["view_index_1based"] if winner and method["branches"] != "G" else None,
            "winning_components": {b: winner[0][b] for b in method["branches"]} if winner else None,
            "geometry_max": geometry[0]["H"] if geometry else None,
            "geometry_best_view": geometry[0]["view_index_1based"] if geometry else None}


def prune_candidates(pairs, query_field, choice_field, count):
    """Rank by best new geometry; preserve all exact ties at the cutoff."""
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[query_field]].append(pair)
    retained = {}
    for query, group in groups.items():
        ordered = sorted((p for p in group if p["geometry_max"] is not None),
                         key=lambda p: (-p["geometry_max"], p[choice_field]))
        cutoff = ordered[min(count, len(ordered)) - 1]["geometry_max"] if ordered else None
        retained[query] = [p[choice_field] for p in ordered if p["geometry_max"] >= cutoff]
    return [{**p, "score": p["score"] if p[choice_field] in retained[p[query_field]] else None}
            for p in pairs], retained


def evaluate_methods(groups, audit):
    cad_truth = {r["simulator_instance"]: r["object_id"] for r in audit if r["status"] == "matched"}
    observation_truth = {v: k for k, v in cad_truth.items()}
    assert len(cad_truth) == len(observation_truth) == 6, "These five saved scenes must each contain six matched objects."
    results = {}
    for name, method in METHODS.items():
        pairs = [pair_result(rows, method) for rows in groups.values()]
        directions = {}
        for direction, q, c, truth in (("cad_to_observation", "cad_id", "object_id", cad_truth),
                                       ("observation_to_cad", "object_id", "cad_id", observation_truth)):
            candidates, retained = (pairs, None)
            if method["candidate_top_k"]:
                candidates, retained = prune_candidates(pairs, q, c, method["candidate_top_k"])
            evaluation = rank_pairs(candidates, truth, q, c, "score")
            margins = [r["margin"] for r in evaluation["targets"] if r["margin"] is not None]
            evaluation.update({"mean_margin": float(np.mean(margins)) if margins else None,
                "margin_count": len(margins), "retained_candidates": retained,
                "correct_candidates_removed": [query for query, expected in truth.items()
                                               if retained is not None and expected not in retained[query]],
                "correct_pairs_unavailable": [t["query"] for t in evaluation["targets"] if t["correct_score"] is None]})
            directions[direction] = evaluation
        results[name] = {"pairs": pairs, **directions}
    return results


def select_original_winner(results):
    """Prefer CAD-to-observation accuracy, then object classification; no margin fitting."""
    best = max((r["cad_to_observation"]["correct"], r["observation_to_cad"]["correct"])
               for r in results.values())
    tied = [name for name, r in results.items()
            if (r["cad_to_observation"]["correct"], r["observation_to_cad"]["correct"]) == best]
    # Preserve the predeclared order instead of consulting rearranged-scene labels.
    return {"selected_method": tied[0], "tied_methods": tied,
            "cad_correct": best[0], "observation_correct": best[1]}


def score_scene(scene_id, source, captures, masks_dir, cads, output):
    import cv2
    from cad_object_association import content_hash
    from scripts.run_wrist_input_ablation import read_json, save_json
    from scripts.surface_verification import bbox, equal_penalty_scores

    start = perf_counter()
    audit_path = DATASET / "identity_audit.json" if scene_id == "original" else captures / "identity_audit.json"
    paths = [source / "all_view_scores.json", captures / "depth.npy", captures / "intrinsics.npy",
             audit_path, *sorted(masks_dir.glob("*.png"))]
    hashes = {str(p.relative_to(ROOT)): content_hash(p) for p in paths}
    old = read_json(source / "all_view_scores.json")
    depth = np.load(captures / "depth.npy")
    K = np.load(captures / "intrinsics.npy")
    masks = {p.stem: cv2.imread(str(p), 0) > 0 for p in sorted(masks_dir.glob("*.png"))}
    bounds = {oid: bbox(mask) for oid, mask in masks.items()}
    assert len(old) == 1512 and len(masks) == 6
    geometry_start = perf_counter()
    rows, render_count = [], 0
    for record in old:
        cid, oid = record["cad_id"], record["object_id"]
        row = {"cad_id": cid, "object_id": oid, "view_index_1based": record["view_index_1based"],
               "camera_T_cad": record["camera_T_cad"], "cls": float(np.clip(record["cls_cosine"], 0, 1)),
               "G": record["global"], "P": record["patch"], "H": None,
               "pose_status": record["status"], "status": record["status"]}
        if record["status"] == "assessable":
            rendered = cads[cid].render(np.array(record["camera_T_cad"]), K, depth.shape, bounds[oid], .5)
            assert rendered is not None and not rendered["clipped"]
            x1, y1, x2, y2 = rendered["roi"]
            metrics = equal_penalty_scores(depth[y1:y2, x1:x2], masks[oid][y1:y2, x1:x2], rendered["depth"])
            row.update(metrics)
            row["H"] = metrics["geometry_score"]
            render_count += 1
        rows.append(row)
    geometry_s = perf_counter() - geometry_start
    groups = defaultdict(list)
    for row in rows:
        groups[row["cad_id"], row["object_id"]].append(row)
    assert len(groups) == 36 and all(len(group) == 42 for group in groups.values())
    # The audit enters only evaluation, after every pair's geometry has been scored.
    evaluation_start = perf_counter()
    methods = evaluate_methods(groups, read_json(audit_path))
    evaluation_s = perf_counter() - evaluation_start
    assert all(content_hash(ROOT / p) == digest for p, digest in hashes.items())
    result = {"scene": scene_id, "image_shape": list(depth.shape), "source_sha256": hashes,
              "methods": methods, "all_view_scores": rows, "runtime": {
                  "new_geometry_render_and_score_s": geometry_s, "renders": render_count,
                  "cached_policy_evaluation_s": evaluation_s, "wall_s": perf_counter() - start}}
    save_json(output / f"{scene_id}.json", result)
    print(json.dumps({"scene": scene_id, "renders": render_count, "geometry_s": geometry_s,
                      "scores": {name: [r["cad_to_observation"]["correct"], r["observation_to_cad"]["correct"]]
                                 for name, r in methods.items()}}), flush=True)
    return result


def main():
    import open3d as o3d
    from cad_object_association import content_hash
    from scripts.run_wrist_input_ablation import read_json, save_json, snapshot_hashes
    from scripts.surface_verification import CadSurface, SIGMA_M, OCCLUSION_M

    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    fixed = snapshot_hashes()
    protected = {p: content_hash(ROOT / p) for p in ("cad_object_association.py", "CADPointCloudRegistration.py")}
    protocol = {"methods": METHODS, "base_weights": WEIGHTS,
        "subset_weights": "Remove absent branches and renormalize remaining fixed weights; no weight search.",
        "G": "Original mean of five highest clamped CLS similarities among all 42 templates; unchanged by view filtering.",
        "P": "Saved SAM-6D-style patch score for each view, using localization masks and DINOv2-L/14.",
        "H": "User formula: 1-(observed-only/union + CAD-only/union + mean(min(abs(Zobs-Zcad)/sigma,1)))/3.",
        "sigma_d_m": SIGMA_M, "external_occlusion_tolerance_m": OCCLUSION_M,
        "geometry_support": "Existing assessable first-hit visible CAD mask; full-resolution measured depth and localized mask.",
        "poses": "All saved rotations/translations frozen. Existing table/camera rejection is unchanged. No support-contact repositioning.",
        "selection": "Top five views; one predeclared top-ten geometry then top-five CLS variant. Empty intersections use union.",
        "candidate_pruning": "Top 3 or 5 by max-view H for each query; exact cutoff ties retained. Directions evaluated independently.",
        "identity_selection": "Independent CAD-to-region and region-to-CAD selection; no forced bijection.",
        "winner_rule": "Maximize original-scene CAD query accuracy, then object classification accuracy; ties keep predeclared order.",
        "validation": "Freeze original winner before scoring four saved rearrangements. Report all frozen methods without retuning.",
        "resolution_note": "Original capture is 768x768; rearranged captures are 3840x3840. This also changes sampling, not only layout.",
        "runtime": "Shared new geometry computation and cached-score policy evaluation, not separate live end-to-end strategy timings.",
        "production_modified": False,
        "source_code_sha256": {p: content_hash(ROOT / p) for p in (
            "scripts/run_wrist_branch_order.py", "scripts/run_wrist_shortlist_geometry.py", "scripts/surface_verification.py")}}
    save_json(output / "protocol.json", protocol)
    cads = {r["cad_id"]: CadSurface(o3d.io.read_triangle_mesh(str(ROOT / r["file_path"])))
            for r in read_json(DATASET / "cad_library.json")}
    original = score_scene("original", POSE_SOURCE, DATASET / "scene", MASKS, cads, output)
    winner = select_original_winner(original["methods"])
    save_json(output / "original_selection.json", winner)
    scenes = {"original": original}
    for number in range(1, 5):
        name = f"heldout_{number:02}"
        captures = SOURCE / "new_layouts" / name
        scenes[name] = score_scene(name, SOURCE / "stage4_layouts" / name, captures, captures / "masks", cads, output)
    comparison = {}
    for name in METHODS:
        entry = {"original": {direction: original["methods"][name][direction]["correct"]
                               for direction in ("cad_to_observation", "observation_to_cad")}}
        for direction in ("cad_to_observation", "observation_to_cad"):
            scores = [scene["methods"][name][direction] for key, scene in scenes.items() if key != "original"]
            entry[direction] = {"rearranged_correct": sum(s["correct"] for s in scores),
                "rearranged_total": sum(s["total"] for s in scores),
                "per_scene_correct": [s["correct"] for s in scores],
                "correct_candidates_removed": sum(len(s["correct_candidates_removed"]) for s in scores)}
        comparison[name] = entry
    assert snapshot_hashes() == fixed
    assert all(content_hash(ROOT / p) == digest for p, digest in protected.items())
    save_json(output / "comparison.json", {"original_selection": winner, "methods": comparison,
              "source_data_and_production_unchanged": True})
    print(json.dumps({"output": str(output), "original_selection": winner,
                      "selected_validation": comparison[winner["selected_method"]]}, indent=2))


if __name__ == "__main__":
    main()
