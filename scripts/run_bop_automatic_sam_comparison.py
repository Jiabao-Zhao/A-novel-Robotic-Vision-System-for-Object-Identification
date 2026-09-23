"""Text-free SAM versus depth on the frozen pilot, with unchanged CAD scoring."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter

from scripts import run_bop_sam_depth_comparison as shared
from scripts.run_bop_automatic_sam_masks import OUTPUT
from scripts.run_wrist_branch_order import select_views, pair_result, METHODS
from scripts.validate_bop_sam_depth_comparison import main as validate
import numpy as np
import torch


baseline = shared.baseline
ROOT, BASE, PREVIOUS = baseline.ROOT, baseline.OUTPUT, shared.OUTPUT
read_json, save_json = shared.read_json, shared.save_json
SCOPE = "20-frame T-LESS pilot; local SAM3 automatic point-grid proposals with SAM-6D size filtering; our unchanged CAD matcher"
CAD_WORKERS = 8
GEOMETRY = None


def align_patch_top_five(rows, align):
    """Same table-then-patch5 result, stopping after five admissible views.

    Alignment validity is independent of patch score. Once five valid views
    have been found in descending patch order, no later view can be selected.
    """
    count = 0
    for row in sorted(rows, key=lambda r: (-r["P"], r["view_index_1based"])):
        row.update(align(row))
        if row["pose_status"] == "assessable":
            count += 1
            if count == 5:
                break
    return select_views(rows, "table_patch5")


def verify_saved_parity(plan):
    """Verify lazy selection and shared export on every prior SAM3 pair/frame."""
    count = 0
    for item in plan:
        name = baseline.frame_folder(item).name
        folder = PREVIOUS / "matching" / name
        groups = defaultdict(list)
        for row in read_json(folder / "view_scores.json"):
            if row["object_id"].startswith("sam_"):
                groups[row["cad_id"], row["object_id"]].append(row)
        for old in groups.values():
            lookup = {r["view_index_1based"]: r for r in old}
            rows = [{**r, "pose_status": "not_visited_below_patch5", "H": None} for r in old]
            align_patch_top_five(rows, lambda r: lookup[r["view_index_1based"]])
            actual, expected = (pair_result(r, METHODS["table_then_patch5"]) for r in (rows, old))
            assert actual == expected
            count += 1
        masks, _ = shared.masks_for_frame(item)
        pairs = read_json(folder / "pair_scores.json")
        times = read_json(folder / "runtime.json")["new_matching_s_by_region"]
        predictions, nms = shared.make_predictions(item, masks, pairs, times)
        assert predictions == read_json(folder / "predictions.json")
        assert nms == read_json(folder / "nms.json")
    save_json(OUTPUT / "selection_parity.json", {"prior_sam3_pairs_verified": count,
        "identical_selected_views_and_scores": True, "prior_frame_exports_exactly_reproduced": len(plan)})


def score_geometry(rows, cad, visible, rotations, points, depth, mask, K, table):
    """One CAD's independent CPU work; CUDA appearance stays on the main thread."""
    def align(row):
        if not len(points):
            return {"pose_status": "no_valid_depth"}
        v = row["view_index_1based"] - 1
        T = np.eye(4)
        T[:3, :3] = rotations[v]
        T[:3, 3], _ = shared.translation_fit(visible[v], points)
        return {"camera_T_cad": T.tolist(),
                "pose_status": shared.pose_status(cad, T, K, depth.shape, table)}

    for row in align_patch_top_five(rows, align):
        rendered = cad.render(np.array(row["camera_T_cad"]), K, depth.shape, shared.bbox(mask), 0.)
        assert rendered is not None and not rendered["clipped"]
        x1, y1, x2, y2 = rendered["roi"]
        metrics = shared.equal_penalty_scores(depth[y1:y2, x1:x2], mask[y1:y2, x1:x2], rendered["depth"])
        row.update(metrics)
        row["H"] = metrics["geometry_score"]
    return pair_result(rows, METHODS["table_then_patch5"]), rows


def initialize_geometry(visible, rotations):
    """Own CPU raycasters in each spawned process; never initialize CUDA here."""
    global GEOMETRY
    torch.set_num_threads(1)
    cads = {}
    for cid in visible:
        mesh = baseline.o3d.io.read_triangle_mesh(str(baseline.DATA / "models_cad" / f"{cid}.ply"))
        mesh.scale(.001, center=(0, 0, 0))
        cads[cid] = baseline.CadSurface(mesh)
    GEOMETRY = cads, visible, rotations


def geometry_job(rows, cid, points, depth, mask, K, table):
    cads, visible, rotations = GEOMETRY
    return score_geometry(rows, cads[cid], visible[cid], rotations, points, depth, mask, K, table)


def verify_parallel_parity(plan, engine, pool):
    """Recompute one available and one rejected cached region without GT access."""
    cads, visible, rotations, _ = engine
    selected = {}
    for item in plan:
        folder = OUTPUT / "matching" / baseline.frame_folder(item).name / "regions"
        for path in sorted(folder.glob("sam_*.json")):
            saved = read_json(path)
            kind = "available" if any(p["score"] is not None for p in saved["pairs"]) else "rejected"
            selected.setdefault(kind, (item, path, saved))
            if len(selected) == 2:
                break
        if len(selected) == 2:
            break
    checked = []
    for kind, (item, path, saved) in selected.items():
        masks, localized = shared.masks_for_frame(item, OUTPUT, "sam")
        mask = masks[path.stem]
        _, raw, scale, K, _ = baseline.frame_input(item)
        depth = raw.astype(float) * scale
        points = shared.downsample(shared.depth_points(depth, mask, K, 0.))
        table = np.array(localized["plane_model"], float)
        table /= np.linalg.norm(table[:3])
        if table[3] < 0:
            table *= -1
        jobs = []
        for cid, cad in cads.items():
            old = [r for r in saved["views"] if r["cad_id"] == cid]
            rows = [{**{k: r[k] for k in ("cad_id", "object_id", "view_index_1based", "cls", "G", "P")},
                     "H": None, "camera_T_cad": None, "pose_status": "not_visited_below_patch5"} for r in old]
            serial_pair, serial_rows = score_geometry(
                [dict(r) for r in rows], cads[cid], visible[cid], rotations, points, depth, mask, K, table)
            assert serial_rows == old, f"Serial geometry changed {path.stem}/{cid}"
            jobs.append((cid, old, serial_pair, pool.submit(geometry_job, rows, cid, points, depth, mask, K, table)))
        for cid, old, serial_pair, job in jobs:
            pair, rows = job.result()
            assert rows == old, f"Parallel geometry changed {path.stem}/{cid}"
            assert pair == serial_pair
            assert pair == next(p for p in saved["pairs"] if p["cad_id"] == cid)
        checked.append({"region": str(path.relative_to(OUTPUT)), "kind": kind, "cad_pairs": len(jobs)})
    save_json(OUTPUT / "parallel_parity.json", {"exact_numeric_parity": True if checked else None,
                                               "status": "verified" if checked else "no_regions_to_check", "regions": checked,
                                               "workers": CAD_WORKERS})


def score_frame(item, engine, observed, pool):
    name = baseline.frame_folder(item).name
    folder = OUTPUT / "matching" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "predictions.json").exists():
        return
    cads, visible, rotations, templates = engine
    masks, localized = shared.masks_for_frame(item, OUTPUT, "sam")
    _, raw, scale, K, _ = baseline.frame_input(item)
    depth = raw.astype(float) * scale
    table = np.array(localized["plane_model"], float)
    table /= np.linalg.norm(table[:3])
    if table[3] < 0:
        table *= -1
    prior = PREVIOUS / "matching" / name
    pairs = [r for r in read_json(prior / "pair_scores.json") if r["object_id"].startswith("depth_")]
    views = [r for r in read_json(prior / "view_scores.json") if r["object_id"].startswith("depth_")]
    times = {oid: 0. for oid in masks if oid.startswith("depth_")}
    for oid in shared.variant_ids(masks, "sam_leading"):
        start = perf_counter()
        cache = folder / "regions" / f"{oid}.json"
        if cache.exists():
            saved = read_json(cache)
            pairs.extend(saved["pairs"])
            views.extend(saved["views"])
            times[oid] = saved["matching_s"]
            continue
        mask = masks[oid]
        points = shared.downsample(shared.depth_points(depth, mask, K, 0.))
        query = observed[f"{name}/{oid}"]
        region_pairs, region_views, jobs = [], [], []
        for cid, cad in cads.items():
            semantic = shared.semantic_score(query["cls"], templates[cid]["cls"])
            rows = [{"cad_id": cid, "object_id": oid, "view_index_1based": v + 1,
                "cls": semantic["cls_view_cosines"][v], "G": semantic["semantic_score"],
                "P": shared.appearance_score(query["patch"], templates[cid]["patch"][v])["appearance_score"],
                "H": None, "camera_T_cad": None, "pose_status": "not_visited_below_patch5"}
                for v in range(42)]

            jobs.append(pool.submit(geometry_job, rows, cid, points, depth, mask, K, table))
        for job in jobs:
            pair, rows = job.result()
            region_pairs.append(pair)
            region_views.extend(rows)
        times[oid] = perf_counter() - start
        cache.parent.mkdir(exist_ok=True)
        save_json(cache, {"pairs": region_pairs, "views": region_views, "matching_s": times[oid]})
        pairs.extend(region_pairs)
        views.extend(region_views)
    assert len(pairs) == len(masks) * 30 and len(views) == len(pairs) * 42
    save_json(folder / "pair_scores.json", pairs)
    save_json(folder / "view_scores.json", views)
    predictions, nms = shared.make_predictions(item, masks, pairs, times)
    save_json(folder / "predictions.json", predictions)
    save_json(folder / "nms.json", nms)
    save_json(folder / "runtime.json", {"new_matching_s_by_region": times,
        "sam_generation": read_json(OUTPUT / "sam" / name / "runtime.json"),
        "scope": "Cached depth scores; fresh SAM scoring; shared template setup excluded; not end-to-end latency"})
    print(name, {k: len(v) for k, v in predictions.items()}, f"matching {sum(times.values()):.1f}s", flush=True)


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sam = read_json(OUTPUT / "sam_protocol.json")
    prior = read_json(PREVIOUS / "protocol.json")
    plan = sam["plan"]
    assert plan == prior["plan"]
    hashes = {p: h for p, h in prior["provenance_sha256"].items() if "bop_sam_depth" not in p}
    for item in plan:
        folder = OUTPUT / "sam" / baseline.frame_folder(item).name
        assert (folder / "proposals.json").exists(), "Finish automatic mask generation first"
        for path in [folder / "proposals.json", *sorted((folder / "masks").glob("*.png"))]:
            hashes[str(path.relative_to(ROOT))] = baseline.association.content_hash(path)
    source = {p: baseline.association.content_hash(ROOT / p) for p in (
        *prior["source_sha256"], "scripts/run_bop_automatic_sam_masks.py",
        "scripts/local_sam3.py", "scripts/run_bop_automatic_sam_comparison.py", "scripts/validate_bop_sam_depth_comparison.py")}
    protocol = {"plan": plan, "sam": sam, "cad_count": 30, "views_per_cad": 42,
        "scoring": prior["scoring"], "mask_nms_iou": prior["mask_nms_iou"],
        "depth_scores": "Exact completed depth-leading pairs reused",
        "alignment": "Same 30 translation-only iterations and 5mm sampling; visit views in descending patch order until five pass the same table/camera checks",
        "inference_inputs": "RGB, measured depth, intrinsics, estimated plane, all 30 CADs; no GT or text prompts",
        "scope": SCOPE, "source_sha256": source, "input_sha256": hashes,
        "execution": {"cad_processes": CAD_WORKERS, "start_method": "spawn"}}
    path = OUTPUT / "protocol.json"
    if path.exists():
        assert read_json(path) == protocol, "Frozen comparison protocol changed"
    else:
        save_json(path, protocol)
    start = perf_counter()
    verify_saved_parity(plan)
    if not all((OUTPUT / "matching" / baseline.frame_folder(i).name / "predictions.json").exists() for i in plan):
        print("Encoding automatic SAM regions on CUDA", flush=True)
        observed = shared.encode_sam(plan, OUTPUT, "sam")
        print("Loading the same 30-CAD, 42-view template cache", flush=True)
        engine = baseline.cached_templates()
        with ProcessPoolExecutor(max_workers=CAD_WORKERS, mp_context=get_context("spawn"),
                                 initializer=initialize_geometry, initargs=(engine[1], engine[2])) as pool:
            for item in plan:
                score_frame(item, engine, observed, pool)
            verify_parallel_parity(plan, engine, pool)
    shared.evaluate(plan, OUTPUT, "sam", SCOPE)
    validate(OUTPUT)
    result = read_json(OUTPUT / "comparison.json")
    previous = read_json(PREVIOUS / "comparison.json")
    assert result["methods"]["depth_leading"] == previous["methods"]["depth_leading"]
    assert all(baseline.association.content_hash(ROOT / p) == h for p, h in hashes.items())
    assert all(baseline.association.content_hash(ROOT / p) == h for p, h in source.items())
    save_json(OUTPUT / "validation.json", {"inputs_and_sources_unchanged": True,
        "depth_result_exactly_reproduced": True, "wall_s": perf_counter() - start,
        "gpu": torch.cuda.get_device_name(), "production_modified": False})


if __name__ == "__main__":
    main()
