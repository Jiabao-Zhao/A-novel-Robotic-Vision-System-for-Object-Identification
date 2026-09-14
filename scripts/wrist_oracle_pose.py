"""Exact captured poses and independent perspective CAD rendering for diagnostics."""

import copy

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import cad_object_association as association
from scripts.run_wrist_input_ablation import ROOT, DATASET, read_json, read_rgb, save_json
from scripts.wrist_visible_geometry import coverage_diagnostics


SOURCE = ROOT / "outputs/simulation/wrist_association/20260911T194945829142Z"


def recover_capture(output):
    """Invert the final Euler position update; require bit-identical RGB-D replay."""
    import mujoco
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map,
    )
    from simulation.wrist_cluster import make_environment, CAMERA
    from scripts.run_wrist_association import configure_scene

    manifest = read_json(SOURCE / "manifest.json")
    library = {r["cad_id"]: r for r in read_json(DATASET / "cad_library.json")}
    catalog = copy.deepcopy(manifest["catalog"])
    for cid, record in catalog.items():
        path = ROOT / library[cid]["file_path"]
        assert association.content_hash(path) == record["cad_sha256"]
        record["cad_path"] = str(path)
    expected_rgb = read_rgb(DATASET / "scene/rgb.png")
    expected_depth = np.load(DATASET / "scene/depth.npy")
    expected_camera = np.load(DATASET / "scene/world_T_camera.npy")
    poses, mesh_errors = {}, {}
    with make_environment(catalog, manifest["placements"], additional_scene=configure_scene) as env:
        env.reset(seed=manifest["seed"])
        sim = env.sim
        saved = np.load(SOURCE / "initial_state.npy")
        sim.set_state_from_flattened(saved)
        sim.forward()
        np.testing.assert_array_equal(sim.get_state().flatten(), saved)
        post_camera_error = float(np.max(np.abs(get_camera_extrinsic_matrix(sim, CAMERA) - expected_camera)))
        assert int(sim.model.opt.integrator) == 0  # The original scene uses Euler.
        timestep = float(sim.model.opt.timestep)
        qpos = sim.data.qpos.copy()
        # MuJoCo's final integration updates qpos; the render used the preceding
        # derived geometry. Inverting that update recovers the captured geometry.
        mujoco.mj_integratePos(sim.model._model, qpos, sim.data.qvel, -timestep)
        sim.data.qpos[:] = qpos
        sim.forward()
        K = get_camera_intrinsic_matrix(sim, CAMERA, 768, 768)
        world_T_camera = get_camera_extrinsic_matrix(sim, CAMERA)
        raw, buffer = sim.render(width=768, height=768, camera_name=CAMERA, depth=True)
        rgb = raw[::-1].copy()
        depth = get_real_depth_map(sim, buffer)[::-1].astype(np.float32)
        np.testing.assert_array_equal(K, np.load(DATASET / "scene/intrinsics.npy"))
        np.testing.assert_array_equal(world_T_camera, expected_camera)
        np.testing.assert_array_equal(rgb, expected_rgb)
        np.testing.assert_array_equal(depth, expected_depth)
        for cid in catalog:
            bid = sim.model.body_name2id(f"nist_part_{cid}")
            body_geoms = np.flatnonzero(sim.model.geom_bodyid == bid)
            gids = body_geoms[sim.model.geom_group[body_geoms] == 1]
            assert len(gids) == 1
            gid = int(gids[0])
            world_T_cad = np.eye(4)
            world_T_cad[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
            world_T_cad[:3, 3] = sim.data.body_xpos[bid]
            # The original XML places each visual mesh directly in its body.
            # Verify MuJoCo's compiled/recentered vertices give the same surface.
            mid = int(sim.model.geom_dataid[gid])
            first, count = int(sim.model.mesh_vertadr[mid]), int(sim.model.mesh_vertnum[mid])
            compiled = sim.model.mesh_vert[first:first + count].astype(float)
            compiled_world = compiled @ sim.data.geom_xmat[gid].reshape(3, 3).T + sim.data.geom_xpos[gid]
            mesh = o3d.io.read_triangle_mesh(catalog[cid]["cad_path"])
            original_world = np.asarray(mesh.vertices) @ world_T_cad[:3, :3].T + world_T_cad[:3, 3]
            distance = max(cKDTree(original_world).query(compiled_world)[0].max(),
                           cKDTree(compiled_world).query(original_world)[0].max())
            assert distance < 1e-6, f"CAD/body frame mismatch: {cid}: {distance} m"
            mesh_errors[cid] = float(distance)
            poses[cid] = {
                "world_T_cad": world_T_cad.tolist(),
                "camera_T_cad": (np.linalg.inv(world_T_camera) @ world_T_cad).tolist(),
                "visual_geom_id": gid,
            }
        np.save(output / "captured_qpos.npy", qpos)
    validation = {
        "source_state": str((SOURCE / "initial_state.npy").relative_to(ROOT)),
        "source_state_sha256": association.content_hash(SOURCE / "initial_state.npy"),
        "position_update_reversed_seconds": timestep,
        "post_integration_camera_matrix_max_error": post_camera_error,
        "restored_rgb_pixel_identical": True, "restored_depth_bit_identical": True,
        "restored_intrinsics_and_camera_exact": True,
        "compiled_vs_original_mesh_max_vertex_distance_m": mesh_errors,
        "pose_source": "Captured simulator body transforms after verified inverse final Euler position update; no pose fitting",
    }
    save_json(output / "replay_validation.json", validation)
    save_json(output / "poses.json", poses)
    return poses, validation


def perspective_render(mesh, camera_T_cad, K, shape, color, pixel_offset=.5):
    """First surface per OpenGL pixel center; same shading law as CAD templates.

    Stored K has cx=width/2, cy=height/2. MuJoCo raster samples are u+.5,v+.5.
    Z is optical-axis depth, not Euclidean range. No mesh normalization/rescaling.
    """
    transformed = o3d.geometry.TriangleMesh(mesh).transform(camera_T_cad)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(transformed))
    v, u = np.indices(shape)
    directions = np.stack(((u + pixel_offset - K[0, 2]) / K[0, 0],
                           (v + pixel_offset - K[1, 2]) / K[1, 1], np.ones(shape)), axis=-1)
    rays = np.concatenate((np.zeros_like(directions), directions), axis=-1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    depth = hit["t_hit"].numpy()
    mask = np.isfinite(depth)
    points = directions[mask] * depth[mask, None]
    # Production illumination: .35 ambient + .65 abs(normal dot camera direction).
    toward_camera = -directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    illumination = .35 + .65 * np.abs(np.sum(hit["primitive_normals"].numpy() * toward_camera, axis=-1))
    rgb = np.full((*shape, 3), 255, dtype=np.uint8)
    rgb[mask] = np.clip(255 * illumination[mask, None] * color, 0, 255).astype(np.uint8)
    return {"rgb": rgb, "mask": mask, "depth_m": depth, "surface_points_camera_m": points, "scene": scene}


def fixed_pose_metrics(cad, observed, camera_T_cad, registrar):
    """Evaluate the unchanged fitness/RMSE at a supplied pose; do not optimize it."""
    result = o3d.pipelines.registration.evaluate_registration(
        observed, cad, 1.5 * registrar.config.voxel_size_m, np.linalg.inv(camera_T_cad))
    raw = {
        "registration_fitness": float(result.fitness), "inlier_rmse_m": float(result.inlier_rmse),
        "correspondence_count": len(result.correspondence_set),
        "max_correspondence_distance_m": 1.5 * registrar.config.voxel_size_m,
        "T_observed_from_cad": camera_T_cad.tolist(),
        "observed_to_cad_rmse_m": registrar.observed_to_cad_rmse(cad, observed, camera_T_cad),
    }
    return {**raw, **coverage_diagnostics(cad, observed, raw)}


def distance_summary(distances_m):
    values = np.asarray(distances_m, dtype=float)
    return {"count": len(values), "mean_mm": float(values.mean() * 1000),
            "rmse_mm": float(np.sqrt(np.mean(values ** 2)) * 1000),
            "median_mm": float(np.median(values) * 1000),
            "p95_mm": float(np.quantile(values, .95) * 1000), "max_mm": float(values.max() * 1000)}
