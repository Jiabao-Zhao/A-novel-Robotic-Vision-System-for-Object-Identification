"""Build and inspect the workbench; do not run localization or association."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from simulation.industrial_workbench import (
    ROOT, TABLE_SIZE_M, TABLE_TOP_Z_M, FLOOR_Z_M, prepare_catalog,
    preview_placements, configure_scene,
)
from simulation.wrist_cluster import CAMERA, make_environment


OUTPUT = ROOT / "outputs/simulation/industrial_workbench"
IMAGE_SIZE = 3840


def main():
    import open3d as o3d
    from scripts.run_wrist_cluster import prepare_observation
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map,
    )

    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True)
    catalog, placements = prepare_catalog(), preview_placements()
    with make_environment(catalog, placements,
                          additional_scene=lambda model: configure_scene(model, catalog, placements)) as env:
        env.task_name = "industrial_workbench_preview"
        prepare_observation(env)
        sim = env.sim
        if not np.all(np.isfinite(sim.data.qpos)):
            raise RuntimeError("Invalid simulator state after settling.")
        rgb, depth = sim.render(width=IMAGE_SIZE, height=IMAGE_SIZE, camera_name=CAMERA, depth=True)
        rgb = rgb[::-1].copy()
        depth = get_real_depth_map(sim, depth)[::-1].astype(np.float32)
        K = get_camera_intrinsic_matrix(sim, CAMERA, IMAGE_SIZE, IMAGE_SIZE)
        world_T_camera = get_camera_extrinsic_matrix(sim, CAMERA)
        preview = sim.render(width=1400, height=1000, camera_name="workbench_preview")[::-1].copy()
        for name, image in (("wrist_rgb.png", rgb), ("workbench_preview.png", preview)):
            if not cv2.imwrite(str(output / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not save {name}")
        cv2.imwrite(str(output / "wrist_preview.png"),
                    cv2.cvtColor(cv2.resize(rgb, (1000, 1000)), cv2.COLOR_RGB2BGR))
        for name, array in (("depth", depth), ("intrinsics", K), ("world_T_camera", world_T_camera),
                            ("initial_state", sim.get_state().flatten())):
            np.save(output / f"{name}.npy", array)
        poses = {}
        for name, record in catalog.items():
            index = sim.model.body_name2id(f"nist_part_{name}")
            position = sim.data.body_xpos[index].copy()
            rotation = sim.data.body_xmat[index].reshape(3, 3)
            mesh = o3d.io.read_triangle_mesh(record["cad_path"])
            vertices = np.asarray(mesh.vertices) @ rotation.T + position
            camera_points = (vertices - world_T_camera[:3, 3]) @ world_T_camera[:3, :3]
            pixels = camera_points @ K.T
            pixels = pixels[:, :2] / pixels[:, 2:]
            if vertices[:, 2].min() < -.001 or vertices[:, 2].min() > .003:
                raise RuntimeError(f"Part is not resting on the workbench: {name}")
            if (camera_points[:, 2].min() <= 0 or pixels.min() < 0 or pixels.max() >= IMAGE_SIZE):
                raise RuntimeError(f"Part is outside the wrist image: {name}")
            poses[name] = {"position_world_m": position.tolist(),
                           "rotation_world": rotation.tolist(),
                           "mesh_bottom_world_z_m": float(vertices[:, 2].min()),
                           "image_bbox_xyxy": np.r_[pixels.min(0), pixels.max(0)].tolist()}
            assert hashlib.sha256(Path(record["cad_path"]).read_bytes()).hexdigest() == record["cad_sha256"]
        summary = {
            "purpose": "Environment preview only; no recognition benchmark run",
            "table_size_m": TABLE_SIZE_M, "table_top_world_z_m": TABLE_TOP_Z_M,
            "floor_world_z_m": FLOOR_Z_M, "table_height_m": TABLE_TOP_Z_M - FLOOR_Z_M,
            "table_contact": {"solref": [.006, 1.], "solimp": [.99, .99, .001], "priority": 1},
            "camera": CAMERA, "image_shape": list(rgb.shape), "depth_unit": "m",
            "catalog": catalog, "placements": placements, "evaluation_only_object_poses": poses,
            "association_run": False, "localization_run": False,
            "limitations": ["Inspection grid is not a benchmark split",
                "LM-O drill and glue use baked source vertex colors; YCB-V drill retains its source UV texture",
                "NIST materials are illustrative; BOP geometry uses published scale with mm-to-m conversion",
                "Glue is upright; bottom-view descriptions and tip-down placement are excluded",
                "Existing simplified collision geometry; no assembly mechanics validated",
                "Ideal simulator depth does not reproduce metal-induced sensor failures"],
        }
        (output / "scene.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(output), "parts": len(catalog), "association_run": False}))


if __name__ == "__main__":
    main()
