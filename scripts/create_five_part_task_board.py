"""Create a separate five-part scene from the retained task-board environment.

Run with the existing WSL MuJoCo environment. Existing scenes and CAD assets are
read-only inputs. Positions use the simulator world frame, in metres.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml"
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
OUTPUT_ROOT = ROOT / "outputs/simulation/five_part_task_board"
PARTS = {
    "RGOCG8-50_8mm": "8 mm round pin",
    "RGOCG12-50_12mm": "12 mm round pin",
    "RGOCG16-50_16mm": "16 mm round pin",
    "Waterproof_Male": "Waterproof connector",
    "Gear_Large": "Large gear",
}
PIN_YAWS_DEG = {"RGOCG8-50_8mm": 0., "RGOCG12-50_12mm": 55., "RGOCG16-50_16mm": 115.}
KEY = "five_parts_ready"
INITIAL_CLEARANCE_M = .001
SETTLING_SECONDS = 5.
PHYSICS_TIMESTEP_S = .0005


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values(array):
    return " ".join(format(float(x), ".17g") for x in np.asarray(array).ravel())


def joint_widths(kind):
    return (7, 6) if kind == mujoco.mjtJoint.mjJNT_FREE else ((4, 3) if kind == mujoco.mjtJoint.mjJNT_BALL else (1, 1))


def load_environment(path):
    """Load the generated MJCF and restore its ready-to-use saved state."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(KEY).id)
    mujoco.mj_forward(model, data)
    return model, data


def prepare():
    source_hash = digest(SOURCE)
    original = mujoco.MjModel.from_xml_path(str(SOURCE))
    original_data = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, original_data, original.key("objects_centered").id)
    mujoco.mj_forward(original, original_data)
    root = ET.parse(SOURCE).getroot()
    # Resolve the smallest pin's contact integration more finely than the
    # source's 2 ms step. Persist the same setting in the exported environment.
    root.find("option").set("timestep", values([PHYSICS_TIMESTEP_S]))
    world = root.find("worldbody")
    removed = []
    for body in list(world.findall("body")):
        name = body.get("name", "")
        if name.startswith("nist_part_") and name.removeprefix("nist_part_") not in PARTS:
            removed.append(name)
            world.remove(body)
    if len(removed) != 9:
        raise ValueError(f"Expected nine other loose bodies, found {len(removed)}")
    root.remove(root.find("keyframe"))
    # Remove unused NIST mesh declarations, retaining all shared robot assets.
    referenced = {geom.get("mesh") for geom in root.findall(".//geom")}
    for mesh in list(root.findall("asset/mesh")):
        if mesh.get("name", "").startswith("nist_") and mesh.get("name") not in referenced:
            root.find("asset").remove(mesh)
    native_points, placements = {}, {}
    for name in PARTS:
        path = MESHES / f"{name}_m.stl"
        points = np.asarray(o3d.io.read_triangle_mesh(str(path)).vertices).copy()
        if not len(points) or not np.isfinite(points).all():
            raise ValueError(f"Invalid native CAD: {name}")
        native_points[name] = points
        body = world.find(f"body[@name='nist_part_{name}']")
        xy = np.fromstring(body.get("pos"), sep=" ")[:2]
        old_id = original.body("nist_part_" + name).id
        rotation = original_data.xmat[old_id].reshape(3, 3).copy()
        if name in PIN_YAWS_DEG:
            rotation = (Rotation.from_euler("z", PIN_YAWS_DEG[name], degrees=True)
                        * Rotation.from_euler("y", 90., degrees=True)).as_matrix()
        rotated = points @ rotation.T
        position = np.r_[xy, INITIAL_CLEARANCE_M - rotated[:, 2].min()]
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        body.set("pos", values(position))
        body.set("quat", values(quaternion))
        placements[name] = {"initial_position_world_m": position.tolist(),
                            "requested_pin_axis_yaw_deg": PIN_YAWS_DEG.get(name),
                            "native_cad_sha256": digest(path)}
    # A supplementary close view leaves all existing camera definitions intact.
    eye, target = np.array([.05, -.55, .65]), np.array([-.14, .01, .015])
    backward = (eye - target) / np.linalg.norm(eye - target)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    quaternion = Rotation.from_matrix(np.column_stack((right, up, backward))).as_quat()[[3, 0, 1, 2]]
    ET.SubElement(world, "camera", name="five_part_closeup", pos=values(eye), quat=values(quaternion), fovy="40")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    robot_qpos, robot_dofs = [], []
    for jid in range(model.njnt):
        name = model.joint(jid).name
        if name.startswith("nist_part_"):
            continue
        old = original.joint(name).id
        nq, nv = joint_widths(model.jnt_type[jid])
        qa, va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        old_qa = int(original.jnt_qposadr[old])
        data.qpos[qa:qa+nq] = original_data.qpos[old_qa:old_qa+nq]
        robot_qpos.extend(range(qa, qa+nq))
        robot_dofs.extend(range(va, va+nv))
    if model.nq != 44 or len(robot_qpos) != 9:
        raise ValueError("Unexpected retained robot/free-joint state dimensions")
    mujoco.mj_forward(model, data)
    return root, model, data, native_points, placements, np.array(robot_qpos), np.array(robot_dofs), source_hash, removed


def settle_and_check(model, data, points, robot_qpos, robot_dofs, parts=PARTS,
                     settling_seconds=SETTLING_SECONDS):
    robot_pose = data.qpos[robot_qpos].copy()
    step_count = int(np.ceil(settling_seconds / model.opt.timestep))
    probe = None
    for step in range(step_count):
        # Hold only the robot's saved joint state. All five loose parts retain
        # their free joints, gravity and normal contact response throughout.
        data.qpos[robot_qpos] = robot_pose
        data.qvel[robot_dofs] = 0.
        mujoco.mj_step(model, data)
        if step == step_count - 1 - int(.5 / model.opt.timestep):
            probe = data.qpos.copy()
    data.qpos[robot_qpos] = robot_pose
    data.qvel[robot_dofs] = 0.
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Non-finite settled scene")
    free_names = {model.body(int(model.jnt_bodyid[j])).name.removeprefix("nist_part_")
                  for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    if free_names != set(parts):
        raise RuntimeError(f"Unexpected movable body inventory: {free_names}")
    table = model.geom("workbench_top").id
    table_top = float(data.geom_xpos[table, 2] + model.geom_size[table, 2])
    records = {}
    for name in parts:
        bid = model.body("nist_part_" + name).id
        jid = model.joint("nist_part_" + name + "_joint").id
        qa, va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        rotation = data.xmat[bid].reshape(3, 3)
        vertices = points[name] @ rotation.T + data.xpos[bid]
        minimum, maximum = vertices.min(0), vertices.max(0)
        contact_geoms = []
        for contact in data.contact:
            gids = [int(contact.geom1), int(contact.geom2)]
            if any(int(model.geom_bodyid[g]) == bid for g in gids):
                contact_geoms.append([model.geom(g).name for g in gids])
        supported = any("workbench_top" in pair for pair in contact_geoms)
        speed = float(np.linalg.norm(data.qvel[va:va+3]))
        spin = float(np.linalg.norm(data.qvel[va+3:va+6]))
        drift = float(np.linalg.norm(data.qpos[qa:qa+3] - probe[qa:qa+3]))
        collisions = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid
                      and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if not collisions or any(not model.geom_contype[g] or not model.geom_conaffinity[g] for g in collisions):
            raise RuntimeError(f"Disabled collision geometry: {name}")
        if not supported or not (-.0003 < minimum[2]-table_top < .0005):
            raise RuntimeError(f"Part is not supported on the table: {name}: {minimum[2]}, {contact_geoms}")
        if speed > .0001 or spin > .01 or drift > .0001:
            raise RuntimeError(f"Part has not settled: {name}: speed={speed}, spin={spin}, drift={drift}")
        bounds = np.array([data.geom_xpos[table, :2]-model.geom_size[table, :2],
                           data.geom_xpos[table, :2]+model.geom_size[table, :2]])
        if np.any(minimum[:2] <= bounds[0]) or np.any(maximum[:2] >= bounds[1]):
            raise RuntimeError(f"Part extends beyond table: {name}")
        records[name] = {"label": parts[name], "position_world_m": data.xpos[bid].tolist(),
                         "quaternion_wxyz": data.qpos[qa+3:qa+7].tolist(),
                         "minimum_cad_height_above_table_m": float(minimum[2]-table_top),
                         "linear_speed_m_s": speed, "angular_speed_rad_s": spin,
                         "drift_last_half_second_m": drift, "table_contact": supported,
                         "active_collision_geoms": [model.geom(g).name for g in collisions],
                         "contacts": contact_geoms,
                         "pin_axis_world": rotation[:, 2].tolist() if name in PIN_YAWS_DEG else None}
    return records, step_count


def capture(model, data, folder, camera, width, height):
    # Multisampling blends segmentation ID colours at edges into other IDs.
    # Disable it for these audit renders before the framebuffer is allocated.
    model.vis.quality.offsamples = 0
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        Image.fromarray(rgb).save(folder / f"{camera}.png")
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        depth = renderer.render().copy()
    visible = {}
    for name in PARTS:
        gid = model.geom("nist_part_" + name + "_visual").id
        ys, xs = np.nonzero((segmentation[:, :, 0] == gid) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))
        visible[name] = {"pixels": len(xs), "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None}
    if camera == "five_part_closeup":
        if any(item["pixels"] < 100 for item in visible.values()):
            raise RuntimeError(f"Not all five parts are visible: {visible}")
        review = Image.fromarray(rgb)
        draw = ImageDraw.Draw(review)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        for name, item in visible.items():
            x0, y0, x1, y1 = item["bbox_xyxy"]
            draw.rectangle((x0-4, y0-4, x1+4, y1+4), outline=(0, 110, 160), width=2)
            draw.text((x0, y1+8), PARTS[name], font=font, fill=(15, 30, 40), stroke_width=2, stroke_fill=(245, 245, 245))
        review.save(folder / "parts_labeled.png")
    elif any(item["pixels"] < 16 for item in visible.values()):
        raise RuntimeError(f"Some parts are not visible in {camera}: {visible}")
    if camera == "robot0_eye_in_hand":
        cid = model.camera(camera).id
        focal = height / (2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
        intrinsics = np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
        world_T_camera = np.eye(4)
        world_T_camera[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        world_T_camera[:3, 3] = data.cam_xpos[cid]
        np.save(folder / "depth_m.npy", depth)
        np.save(folder / "intrinsics.npy", intrinsics)
        np.save(folder / "world_T_camera.npy", world_T_camera)
        np.savez_compressed(folder / "audit_segmentation.npz", segmentation=segmentation)
    return visible


def main():
    root, model, data, points, placements, robot_qpos, robot_dofs, source_hash, removed = prepare()
    records, steps = settle_and_check(model, data, points, robot_qpos, robot_dofs)
    folder = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        body = root.find(f"worldbody/body[@name='nist_part_{name}']")
        body.set("pos", values(record["position_world_m"]))
        body.set("quat", values(record["quaternion_wxyz"]))
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name=KEY, qpos=values(data.qpos), qvel=values(data.qvel))
    scene = folder / "scene.xml"
    ET.ElementTree(root).write(scene, encoding="unicode")
    reloaded, ready = load_environment(scene)
    np.testing.assert_allclose(ready.qpos, data.qpos, atol=1e-14)
    np.testing.assert_allclose(ready.xpos, data.xpos, atol=1e-12)
    views = {}
    for camera, width, height in (("agentview", 1600, 1100), ("five_part_closeup", 1280, 1000), ("robot0_eye_in_hand", 960, 960)):
        views[camera] = capture(reloaded, ready, folder, camera, width, height)
    if digest(SOURCE) != source_hash:
        raise RuntimeError("Original input scene changed")
    summary = {"source_scene": str(SOURCE), "source_scene_sha256": source_hash,
               "generated_scene_sha256": digest(scene), "keyframe": KEY,
               "loose_part_count": len(records), "removed_other_loose_bodies": removed,
               "placements": placements, "settled_parts": records, "views": views,
               "settling_seconds": float(data.time), "settling_steps": steps,
               "physics_timestep_s": float(reloaded.opt.timestep),
               "physics_note": "Time step refined from the source's 2 ms to 0.5 ms to reduce small-pin contact integration jitter.",
               "robot_joint_state_held_during_settling": True,
               "all_five_part_contacts_enabled": True, "native_scale_preserved": True,
               "mujoco_version": mujoco.__version__,
               "render_multisampling_samples": 0,
               "limitations": ["Environment preparation only; no picking, assembly or perception test.",
                               "Robot joints held at their saved pose while loose parts settled.",
                               "Existing part collision geometry retained, including the gear's two collision geoms.",
                               "Simulator positions and segmentation are audit-only.",
                               "Scene references existing local CAD/robot assets; not a portable asset bundle."]}
    (folder / "scene_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (folder / "README.md").write_text(
        "# Five-part task-board scene\n\n"
        "Three 50 mm round pins (8, 12 and 16 mm diameter) lie on the table at different angles; "
        "the waterproof male connector and large gear retain their original XY locations. "
        "The robot, table, task board and fixed hardware are retained. The other nine loose parts are removed.\n\n"
        "Load `scene.xml` in the existing WSL MuJoCo environment and reset keyframe `five_parts_ready`. "
        "The helper `scripts.create_five_part_task_board.load_environment(path)` does both. "
        "All five parts have enabled contacts and free joints. The original robot controls/assets remain; "
        "a running robot still requires its controller. No robot execution was tested.\n\n"
        "`parts_labeled.png` is the close view; `agentview.png` shows the whole setup. "
        "Wrist RGB, depth in metres, intrinsics and camera-to-world transform are also saved. "
        "Camera axes are X right, Y down, Z forward. Simulator labels/poses are audit-only.\n"
    )
    print(json.dumps({"output": str(folder), "loose_parts": list(PARTS), "scene": str(scene), "all_checks_passed": True}), flush=True)


if __name__ == "__main__":
    main()
