"""Exact-pose, matched-renderer semantic controls for the fixed six-object scene.

Run in the existing simulation environment with MUJOCO_GL=egl,
PYOPENGL_PLATFORM=egl, OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1.
No registration, model training, production changes, or generated text reports.
"""

import copy
import os
from datetime import datetime, timezone
from time import perf_counter

import cv2
import mujoco
import numpy as np
import open3d as o3d

import cad_object_association as association
from scripts.run_wrist_association import configure_scene
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, read_rgb, save_json, save_image, snapshot_hashes
from scripts.run_wrist_sam_visual_comparison import ABLATION_1, SAM, prepare_masked_input
from scripts.wrist_oracle_pose import SOURCE, perspective_render
from simulation.wrist_cluster import CAMERA, make_environment


ORACLE = ROOT / "outputs/oracle_pose_wrist_association_2026-09-11/20260914T192456038746Z"
SAM_COMPARISON = ROOT / "outputs/sam_visual_comparison_wrist_association_2026-09-11/20260914T151728934102Z"
OUTPUT_ROOT = ROOT / "outputs/appearance_consistency_wrist_association_2026-09-11"
GEOM_FIELDS = ("geom_type", "geom_dataid", "geom_matid", "geom_rgba", "geom_size",
               "geom_pos", "geom_quat", "geom_rbound", "geom_sameframe", "geom_aabb")


def visible_mask(sim, geom_id, mask_context):
    """Read ID colors without multisampling, which can blend IDs at edges."""
    context = sim._render_context_offscreen
    site_groups = context.vopt.sitegroup.copy()
    context.vopt.sitegroup[:] = 0  # Visualization sites are not physical occluders.
    mujoco.mjv_updateScene(sim.model._model, sim.data._data, context.vopt, context.pert,
                          context.cam, mujoco.mjtCatBit.mjCAT_ALL, context.scn)
    context.vopt.sitegroup[:] = site_groups
    flags = context.scn.flags.copy()
    context.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
    context.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
    viewport = mujoco.MjrRect(0, 0, 768, 768)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, mask_context)
    mujoco.mjr_render(viewport, context.scn, mask_context)
    encoded = np.empty((768, 768, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(encoded, None, viewport, mask_context)
    context.scn.flags[:] = flags
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context.con)
    encoded = encoded[::-1].astype(np.uint32)
    codes = encoded[..., 0] + 256 * encoded[..., 1] + 65536 * encoded[..., 2]
    matches = [g.segid + 1 for g in context.scn.geoms[:context.scn.ngeom]
               if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == geom_id and g.segid >= 0]
    assert len(matches) == 1
    mask = codes == matches[0]
    assert mask.any(), "CAD hypothesis is not visible at the supplied pose"
    return mask


def evaluate(rows, truth, candidates):
    per_target = []
    for cid, expected in truth.items():
        if expected not in candidates:
            continue
        ranked = sorted((r for r in rows if r["cad_id"] == cid and r["object_id"] in candidates),
                        key=lambda r: (-r["cosine"], r["object_id"]))
        correct = next(r for r in ranked if r["object_id"] == expected)
        wrong = next(r for r in ranked if r["object_id"] != expected)
        per_target.append({"cad_id": cid, "expected_object_id": expected,
                           "prediction": ranked[0]["object_id"],
                           "correct": ranked[0]["object_id"] == expected,
                           "correct_cosine": correct["cosine"],
                           "strongest_incorrect_object_id": wrong["object_id"],
                           "strongest_incorrect_cosine": wrong["cosine"],
                           "margin": correct["cosine"] - wrong["cosine"],
                           "ranking": [{"object_id": r["object_id"], "cosine": r["cosine"]} for r in ranked]})
    return {"correct": sum(r["correct"] for r in per_target), "total": len(per_target), "per_target": per_target}


def main():
    started = perf_counter()
    os.chdir(ROOT)
    association.DINO_DEVICE = "cpu"
    before = snapshot_hashes()
    production = [ROOT / name for name in ("cad_object_association.py", "simulation/wrist_cluster.py",
                  "simulation/nist_task_board_1.py", "CADPointCloudRegistration.py")]
    production_hashes = {str(p.relative_to(ROOT)): association.content_hash(p) for p in production}
    output = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (output / "inputs").mkdir(parents=True)
    (output / "masks").mkdir()
    source = read_json(SOURCE / "manifest.json")
    poses = read_json(ORACLE / "ground_truth/poses.json")
    previous = read_json(ORACLE / "all_pairs.json")
    truth = {r["cad_id"]: r["object_id"] for r in read_json(ORACLE / "correct_pairs.json")}
    by_object = {oid: cid for cid, oid in truth.items()}
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    catalog = copy.deepcopy(source["catalog"])
    meshes = {}
    for cid in catalog:
        path = ROOT / library[cid]["file_path"]
        assert association.content_hash(path) == catalog[cid]["cad_sha256"]
        catalog[cid]["cad_path"] = str(path)
        meshes[cid] = o3d.io.read_triangle_mesh(str(path))
    old_protocol = read_json(ORACLE / "protocol.json")
    assert association.content_hash(association.DINO_HUB_DIR / "checkpoints/dinov2_vits14_pretrain.pth") == old_protocol["cad_feature_checkpoint_sha256"]
    rgb = read_rgb(DATASET / "scene/rgb.png")
    K = np.load(DATASET / "scene/intrinsics.npy")
    images, masks = {}, {}
    validation = {"correct_pair_render_pixel_max_error": {}, "max_geometry_transform_error_m": 0.,
                  "correct_pair_mask_vs_raycast_iou": {}}
    for oid in by_object:
        for variant, directory in (("localization", ABLATION_1 / "masked/masks"), ("sam3", SAM / "masks")):
            path = directory / f"{oid}.png"
            if not path.exists():
                assert variant == "sam3" and oid == "object_001"
                continue
            masks[variant, oid] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            canvas = prepare_masked_input(rgb, masks[variant, oid])[0]
            reference = (ABLATION_1 / "masked/inputs" if variant == "localization" else SAM_COMPARISON / "sam/inputs") / f"{oid}.png"
            np.testing.assert_array_equal(canvas, read_rgb(reference))
            images[f"observed__{variant}__{oid}"] = canvas
    stage = perf_counter()
    with make_environment(catalog, source["placements"], additional_scene=configure_scene) as env:
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map

        env.reset(seed=source["seed"])
        sim = env.sim
        sim.set_state_from_flattened(np.load(SOURCE / "initial_state.npy"))
        sim.data.qpos[:] = np.load(ORACLE / "ground_truth/captured_qpos.npy")
        sim.forward()
        replay, depth_buffer = sim.render(width=768, height=768, camera_name=CAMERA, depth=True)
        np.testing.assert_array_equal(replay[::-1], rgb)
        np.testing.assert_array_equal(get_real_depth_map(sim, depth_buffer)[::-1].astype(np.float32), np.load(DATASET / "scene/depth.npy"))
        np.testing.assert_array_equal(get_camera_intrinsic_matrix(sim, CAMERA, 768, 768), K)
        np.testing.assert_array_equal(get_camera_extrinsic_matrix(sim, CAMERA), np.load(DATASET / "scene/world_T_camera.npy"))
        validation["original_rgb_depth_camera_replay_exact"] = True
        original_samples = sim.model.vis.quality.offsamples
        sim.model.vis.quality.offsamples = 0
        mask_context = mujoco.MjrContext(sim.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        sim.model.vis.quality.offsamples = original_samples
        assert mask_context.offSamples == 0
        validation["id_mask_antialias_samples"] = 0
        validation["visualization_sites_excluded_from_id_mask_only"] = True
        validation["pin_overlay_site"] = {
            "id": 5, "name": sim.model.site_id2name(5),
            "rgba": sim.model.site_rgba[5].tolist(), "size": sim.model.site_size[5].tolist()}
        original = {cid: {field: np.array(getattr(sim.model, field)[p["visual_geom_id"]], copy=True)
                         for field in GEOM_FIELDS} for cid, p in poses.items()}
        compiled_body = {}
        for cid, p in poses.items():
            gid, T = p["visual_geom_id"], np.asarray(p["world_T_cad"])
            bid = sim.model.body_name2id(f"nist_part_{cid}")
            np.testing.assert_allclose(sim.data.body_xmat[bid].reshape(3, 3), T[:3, :3], atol=1e-14)
            np.testing.assert_allclose(sim.data.body_xpos[bid], T[:3, 3], atol=1e-14)
            mid = int(sim.model.geom_dataid[gid])
            first, count = int(sim.model.mesh_vertadr[mid]), int(sim.model.mesh_vertnum[mid])
            vertices = sim.model.mesh_vert[first:first + count].astype(float)
            world = vertices @ sim.data.geom_xmat[gid].reshape(3, 3).T + sim.data.geom_xpos[gid]
            compiled_body[cid] = (vertices, (world - T[:3, 3]) @ T[:3, :3])
            oid = truth[cid]
            masks["oracle", oid] = visible_mask(sim, gid, mask_context)
            save_image(output / "masks" / f"observed__{oid}.png", masks["oracle", oid].astype(np.uint8) * 255)
            images[f"observed__oracle__{oid}"] = prepare_masked_input(rgb, masks["oracle", oid])[0]
        for cid in truth:
            for oid, candidate_cid in by_object.items():
                gid = poses[candidate_cid]["visual_geom_id"]
                # Only replace the candidate's visual mesh and material. Retain
                # its captured body pose, camera, lights, scene, and all dynamics.
                for field in GEOM_FIELDS:
                    getattr(sim.model, field)[gid] = original[cid][field]
                if cid != candidate_cid:
                    # Compiler shortcuts can refer to the source body's inertia
                    # frame. The substituted mesh must use its explicit pose.
                    sim.model.geom_sameframe[gid] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
                sim.forward()  # No simulation steps or pose/registration fitting.
                vertices, body = compiled_body[cid]
                T = np.asarray(poses[candidate_cid]["world_T_cad"])
                actual = vertices @ sim.data.geom_xmat[gid].reshape(3, 3).T + sim.data.geom_xpos[gid]
                error = float(np.max(np.abs(actual - (body @ T[:3, :3].T + T[:3, 3]))))
                assert error < 1e-12, (cid, oid, error)
                validation["max_geometry_transform_error_m"] = max(validation["max_geometry_transform_error_m"], error)
                rendered = sim.render(width=768, height=768, camera_name=CAMERA)[::-1].copy()
                mask = visible_mask(sim, gid, mask_context)
                if truth[cid] == oid:
                    np.testing.assert_array_equal(rendered, rgb)
                    np.testing.assert_array_equal(mask, masks["oracle", oid])
                    validation["correct_pair_render_pixel_max_error"][cid] = 0
                raycast = perspective_render(meshes[cid], np.asarray(poses[candidate_cid]["camera_T_cad"]),
                                             K, rgb.shape[:2], np.asarray(library[cid]["base_color_rgb"]))
                y, x = np.where(raycast["mask"])
                my, mx = np.where(mask)
                assert mx.min() >= x.min() - 1 and mx.max() <= x.max() + 1
                assert my.min() >= y.min() - 1 and my.max() <= y.max() + 1
                if truth[cid] == oid:
                    validation["correct_pair_mask_vs_raycast_iou"][cid] = float(
                        (mask & raycast["mask"]).sum() / (mask | raycast["mask"]).sum())
                # Identical per-pair CAD mask for both renderers isolates appearance.
                for renderer, pixels in (("raycast", raycast["rgb"]), ("matched", rendered)):
                    images[f"{renderer}__{cid}__{oid}"] = prepare_masked_input(pixels, mask)[0]
                save_image(output / "masks" / f"{cid}__{oid}.png", mask.astype(np.uint8) * 255)
                for field in GEOM_FIELDS:
                    getattr(sim.model, field)[gid] = original[candidate_cid][field]
                sim.forward()
            print(f"Rendered {cid}: all six exact-pose hypotheses", flush=True)
        np.testing.assert_array_equal(sim.render(width=768, height=768, camera_name=CAMERA)[::-1], rgb)
        mask_context.free()
    rendering_s = perf_counter() - stage
    names = list(images)
    for name, canvas in images.items():
        save_image(output / "inputs" / f"{name}.png", canvas)
    stage = perf_counter()
    raw = association.extract_dino_features(list(images.values())).astype(np.float64)
    features = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    extraction_s = perf_counter() - stage
    feature_by_name = dict(zip(names, features))
    np.savez_compressed(output / "features.npz", names=np.array(names), cls_features=raw)
    scores, evaluations = {}, {}
    common = [oid for oid in by_object if ("sam3", oid) in masks]
    max_independent_error = 0.
    for renderer in ("raycast", "matched"):
        for variant in ("localization", "sam3", "oracle"):
            key = f"{renderer}_{variant}"
            rows = []
            for cid in truth:
                for oid in by_object:
                    if (variant, oid) not in masks:
                        continue
                    a = feature_by_name[f"{renderer}__{cid}__{oid}"]
                    b = feature_by_name[f"observed__{variant}__{oid}"]
                    cosine = float(np.clip(a @ b, -1, 1))
                    independent = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    max_independent_error = max(max_independent_error, abs(cosine - independent))
                    rows.append({"cad_id": cid, "object_id": oid, "cosine": cosine})
            scores[key] = rows
            candidates = common if variant == "sam3" else list(by_object)
            evaluations[key] = {"available_set": evaluate(rows, truth, candidates),
                                "common_five": evaluate(rows, truth, common)}
            summary = evaluations[key]["available_set"]
            print(f"{key}: {summary['correct']}/{summary['total']}", flush=True)
    for cid, oid in truth.items():
        np.testing.assert_array_equal(images[f"matched__{cid}__{oid}"], images[f"observed__oracle__{oid}"])
        row = next(r for r in scores["matched_oracle"] if r["cad_id"] == cid and r["object_id"] == oid)
        assert abs(row["cosine"] - 1) < 1e-12
    validation.update(all_six_identical_pose_appearance_mask_inputs_exact=True,
                      all_six_identical_input_cosines_equal_1=True,
                      max_independent_cosine_error=max_independent_error,
                      source_dataset_unchanged=snapshot_hashes() == before,
                      production_unchanged=all(association.content_hash(ROOT / p) == h for p, h in production_hashes.items()))
    assert validation["source_dataset_unchanged"] and validation["production_unchanged"]
    scores["previous_exact_pose_localization"] = [{"cad_id": p["cad_id"], "object_id": p["object_id"], "cosine": p["visual_raw"]} for p in previous]
    evaluations["previous_exact_pose_localization"] = {"available_set": evaluate(scores["previous_exact_pose_localization"], truth, list(by_object)),
                                                       "common_five": evaluate(scores["previous_exact_pose_localization"], truth, common)}
    save_json(output / "results.json", {"evaluations": evaluations, "pair_scores": scores})
    save_json(output / "validation.json", validation)
    save_json(output / "protocol.json", {
        "source_capture": str(SOURCE.relative_to(ROOT)), "pose_source": str(ORACLE.relative_to(ROOT)),
        "renderer": f"MuJoCo {mujoco.__version__}; original camera, lights, materials, scene and rasterization",
        "hypotheses": "Each CAD's native visual mesh/material substitutes the candidate's visual geom at the candidate's exact body pose; no scaling, optimization or dynamics stepping",
        "masks": "Same MuJoCo object-ID CAD mask for both renderers. Observations independently use unchanged localization/SAM3 masks or captured-scene object-ID masks",
        "oracle_warning": "True poses and object-ID masks are diagnostic inputs. Wrong CADs inherit the candidate frame and can intersect the table. This is not an unbiased association benchmark",
        "consistency_control": "Independent correct-CAD scene renders reproduce captured pixels exactly. Cosine=1 with identical masks is expected; it does not prove real-world reliability",
        "sam3_missing": "object_001 pulley; no fallback. Compare common-five candidate AND target sets separately",
        "sam_source": str(SAM.relative_to(ROOT)), "model": association.DINO_MODEL,
        "checkpoint_sha256": old_protocol["cad_feature_checkpoint_sha256"],
        "preprocessing": "Unchanged white 224x224 tight crop, aspect-preserving bicubic letterbox and ImageNet normalization",
        "score": "Single exact-pose normalized CLS cosine; no top-five aggregation because each pair has one pose hypothesis",
        "geometry": "Not evaluated or altered", "source_sha256": production_hashes,
        "script_sha256": association.content_hash(__file__),
        "runtime_s": {"simulation_and_rendering": rendering_s, "dino_load_and_extraction": extraction_s, "total": perf_counter() - started}})
    print(f"SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
