"""Render six pickup closeups from frozen recorded states, without stepping."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025/runs/20261025T192913Z"
OUTPUT = RUN / "gallery/closeups"
PARTS = ("Gear_Large", "Gear_Medium", "M16_Hex_Nut")
ARM = "cad_grconvnet_score"
WIDTH, HEIGHT = 1200, 900


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def input_files(scene_path):
    """Find the unchanged scene and the native assets it references."""
    tree = ET.parse(scene_path)
    compiler = tree.getroot().find("compiler")
    files = {scene_path.resolve()}
    for element in tree.getroot().findall("asset/*"):
        name = element.get("file")
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            directory = "meshdir" if element.tag == "mesh" else "texturedir"
            base = scene_path.parent
            if compiler is not None and compiler.get(directory):
                base = Path(compiler.get(directory))
                if not base.is_absolute():
                    base = scene_path.parent / base
            path = base / path
        files.add(path.resolve())
    return files


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def caption(rgb, part, phase, row):
    image = Image.new("RGB", (WIDTH, HEIGHT + 106), "#f3f5f7")
    image.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(image)
    state = "after fingers closed" if phase == "close" else "after lifting and holding"
    draw.text((20, HEIGHT + 9), f"{part.replace('_', ' ')}: saved state {state}", fill="#152338", font=font(25))
    draw.text((20, HEIGHT + 45), f"Recorded time {row['time_s']:.2f} s; both fingers touching in the recorded contact log.", fill="#29394b", font=font(20))
    draw.text((20, HEIGHT + 74), "Camera chosen after the trial to show the hand and part. This image was not a VLM input.", fill="#29394b", font=font(20))
    return image


def render_part(part):
    source = RUN / ARM / part
    scene_path = source / "scene.xml"
    trajectory_path = source / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    site = model.site("gripper0_grip_site").id
    target = model.body("nist_part_" + part).id
    options = mujoco.MjvOption()
    options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    records = []
    folder = OUTPUT / part
    folder.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for phase, filename in (("close", "close.png"), ("hold_lift", "hold.png")):
            index = max(i for i, row in enumerate(trajectory) if row["phase"] == phase)
            row = trajectory[index]
            assert sorted(row["grasp_contacts"]) == ["left", "right"]
            data = mujoco.MjData(model)
            saved_qpos = np.asarray(row["qpos"], dtype=float)
            saved_qvel = np.asarray(row["qvel"], dtype=float)
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            data.time = row["time_s"]
            mujoco.mj_forward(model, data)
            grip = np.eye(4)
            grip[:3, :3] = data.site_xmat[site].reshape(3, 3)
            grip[:3, 3] = data.site_xpos[site]
            cad = np.eye(4)
            cad[:3, :3] = data.xmat[target].reshape(3, 3)
            cad[:3, 3] = data.xpos[target]
            grip_error = float(np.max(abs(grip - np.asarray(row["grip_site_pose"])) ))
            cad_error = float(np.max(abs(cad - np.asarray(row["target_pose_audit"])) ))
            assert grip_error < 1e-10 and cad_error < 1e-10
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = (cad[:3, 3] + grip[:3, 3]) / 2 + [0, 0, .025]
            # View across the jaws from their open side, so both fingers show.
            side = grip[:3, 1].copy()
            side[2] = 0
            side /= np.linalg.norm(side)
            camera.distance = .24
            camera.azimuth = float(np.rad2deg(np.arctan2(side[1], side[0])))
            camera.elevation = -30.
            renderer.update_scene(data, camera=camera, scene_option=options)
            rgb = renderer.render().copy()
            path = folder / filename
            caption(rgb, part, phase, row).save(path)
            qpos_error = float(np.max(abs(data.qpos - saved_qpos)))
            qvel_error = float(np.max(abs(data.qvel - saved_qvel)))
            assert qpos_error == 0 and qvel_error == 0 and data.time == row["time_s"]
            records.append({
                "part": part, "arm": ARM, "phase": phase,
                "image": str(path.relative_to(OUTPUT)), "image_sha256": digest(path),
                "source_trajectory": str(trajectory_path.relative_to(RUN)),
                "source_trajectory_sha256": digest(trajectory_path),
                "source_scene_sha256": digest(scene_path),
                "state_index": index, "state_time_s": row["time_s"],
                "recorded_finger_contacts": row["grasp_contacts"],
                "maximum_qpos_difference": qpos_error,
                "maximum_qvel_difference": qvel_error,
                "maximum_grip_pose_matrix_difference": grip_error,
                "maximum_cad_pose_matrix_difference": cad_error,
                "camera_lookat_world_m": camera.lookat.tolist(),
                "camera_distance_m": camera.distance,
                "camera_azimuth_deg": camera.azimuth,
                "camera_elevation_deg": camera.elevation,
                "render_camera_positions_world_m": [entry.pos.tolist() for entry in renderer.scene.camera],
                "physics_steps": 0, "model_calls": 0, "VLM_calls": 0,
                "purpose": "Post-run saved-state pickup visualization, not ranking input",
            })
    return records


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tracked = set()
    for part in PARTS:
        source = RUN / ARM / part
        tracked.update(input_files(source / "scene.xml"))
        tracked.update(source / name for name in ("trajectory.json", "contacts.json", "result.json", "controller_audit.json"))
    before = {str(path): digest(path) for path in sorted(tracked)}
    records = [record for part in PARTS for record in render_part(part)]
    after = {str(path): digest(path) for path in sorted(tracked)}
    assert before == after
    evidence = {
        "render_helper_sha256": digest(Path(__file__)),
        "run": str(RUN), "source_arm": ARM, "images": records,
        "image_count": len(records), "physics_steps": 0,
        "model_calls": 0, "VLM_calls": 0, "mj_forward_calls": len(records),
        "source_files_and_assets_unchanged": before == after,
        "source_file_sha256": before,
        "native_scene_appearance_preserved": True,
        "objects_hidden_or_deleted": False,
        "final_assembly_views_or_goals": False,
        "note": "Camera and captions were chosen after the timed trial; all rendered poses are recorded states.",
    }
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"images": len(records), "output": str(OUTPUT), "unchanged": before == after}))


if __name__ == "__main__":
    main()
