import blenderproc as bproc

# BlenderProc worker, invoked by run_wrist_blenderproc_semantic inside Blender.

import json
import sys
from pathlib import Path
from time import perf_counter

import bpy
import numpy as np
from PIL import Image


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    output = Path(job["output"])
    poses = np.load(output / "cam_poses_level0.npy")
    assert poses.shape == (42, 4, 4)
    bproc.init()
    bproc.renderer.set_render_devices(desired_gpu_device_type="CUDA")
    started = perf_counter()
    records = []
    for item in job["cad_library"]:
        bproc.clean_up()
        if job.get("render_device") == "GPU":
            # clean_up resets the scene's Cycles device. Explicit new jobs restore it.
            bproc.renderer.set_render_devices(desired_gpu_device_type="CUDA")
            bpy.context.scene.cycles.device = "GPU"
        obj, = bproc.loader.load_obj(item["absolute_mesh_path"])
        vertices = np.array([vertex.co[:] for vertex in obj.blender_obj.data.vertices])
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        radius = np.linalg.norm(vertices - center, axis=1).max()
        # Exact vertices replace SAM-6D's random surface sample for deterministic fitting.
        scale = 1 / (2 * radius)
        obj.set_scale([scale] * 3)
        obj.set_location(-center * scale)
        material = bproc.material.create(item["cad_id"])
        material.set_principled_shader_value("Base Color", [*item["base_color_rgb"], 1.])
        material.set_principled_shader_value("Roughness", .5)
        material.set_principled_shader_value("Metallic", 0.)
        obj.set_material(0, material)
        bproc.camera.set_intrinsics_from_blender_params(
            lens=.691111, image_width=512, image_height=512, lens_unit="FOV")
        bproc.renderer.set_max_amount_of_samples(50)
        bproc.renderer.set_output_format(enable_transparency=True)
        bpy.context.scene.cycles.seed = 0
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_energy(1000)
        directory = output / "renders" / item["cad_id"]
        directory.mkdir(parents=True)
        for index, saved_pose in enumerate(poses):
            pose = saved_pose.copy()
            pose[:3, 1:3] *= -1  # OpenCV camera axes -> Blender camera axes.
            pose[:3, 3] *= .002  # Official SAM-6D normalization: radius 2 scene units.
            bproc.camera.add_camera_pose(pose, frame=index)
            light.set_location(2.5 * pose[:3, 3], frame=index)
        stage = perf_counter()
        rendered = bproc.renderer.render()
        elapsed = perf_counter() - stage
        assert len(rendered["colors"]) == 42
        for index, rgba in enumerate(rendered["colors"]):
            assert rgba.shape == (512, 512, 4) and rgba.dtype == np.uint8
            Image.fromarray(rgba).save(directory / f"view_{index + 1:02}_rgba.png")
        record = {"cad_id": item["cad_id"], "render_time_s": elapsed,
                  "center_cad_m": center.tolist(), "scale_render_units_per_m": float(scale),
                  "radius_cad_m": float(radius), "vertex_count": len(vertices),
                  "triangle_count": len(obj.blender_obj.data.polygons),
                  "K_render_camera": bproc.camera.get_intrinsics_as_K_matrix().tolist()}
        records.append(record)
        (output / "render_progress.json").write_text(json.dumps(records, indent=2))
        print(f"COMPLETED {item['cad_id']}: 42 views in {elapsed:.2f}s", flush=True)
    scene = bpy.context.scene
    result = {"objects": records, "wall_time_s": perf_counter() - started,
              "blender_version": bpy.app.version_string,
              "device": scene.cycles.device,
              "compute_device_type": bpy.context.preferences.addons['cycles'].preferences.compute_device_type,
              "samples": scene.cycles.samples, "denoiser": scene.cycles.denoiser,
              "view_transform": scene.view_settings.view_transform,
              "look": scene.view_settings.look, "exposure": scene.view_settings.exposure,
              "gamma": scene.view_settings.gamma}
    (output / "render_runtime.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
