"""Compact seven-part wrist-camera scene for the manuscript framework figure.

This creates a new simulation scene and RGB-D export. Existing assets, captures,
camera mounting geometry and prompts are never modified.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.create_five_part_task_board import (
    ROOT, SOURCE, MESHES, PHYSICS_TIMESTEP_S, digest, values, joint_widths, settle_and_check,
)
import mujoco
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation


PARTS = {
    "Gear_Large": "Large gear", "Waterproof_Male": "Waterproof connector",
    "DSUB_Male": "D-sub connector", "USB_Male": "USB connector",
    "RGOCG8-50_8mm": "8 mm round pin", "RGOCG12-50_12mm": "12 mm round pin",
    "RGOCG16-50_16mm": "16 mm round pin",
}
# XY positions are in the existing world frame. Final poses are settled below.
LAYOUT = {
    "Gear_Large": (-.165, .01775, -20., "flat"),
    "Waterproof_Male": (-.165, .12825, 20., "flat"),
    "DSUB_Male": (-.0715, .17075, -20., "flat"),
    "USB_Male": (.00075, .12825, 15., "broad_side"),
    "RGOCG8-50_8mm": (-.08425, .0305, 10., "pin_side"),
    "RGOCG12-50_12mm": (.00075, .022, 55., "pin_side"),
    "RGOCG16-50_16mm": (-.06725, .09425, 115., "pin_side"),
}
# Bounded inverse kinematics of the unchanged mounted wrist camera. This is a
# static simulation configuration, not a recorded motion or real-robot command.
ARM_QPOS_RAD = np.array([-.3592652813, .1475761906, -.1509016648,
                        -2.1090268700, .5687801849, 1.8744432110, -1.5772099146])
FULL_SIZE_WH = (2560, 1920)
KEY = "framework_figure_ready"


def build_scene():
    source_hash = digest(SOURCE)
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    old_data = mujoco.MjData(old)
    mujoco.mj_resetDataKeyframe(old, old_data, old.key("objects_centered").id)
    root = ET.parse(SOURCE).getroot()
    root.find("option").set("timestep", values([PHYSICS_TIMESTEP_S]))
    root.remove(root.find("keyframe"))
    world = root.find("worldbody")
    removed = []
    for body in list(world.findall("body")):
        name = body.get("name", "")
        if name.startswith("nist_part_") and name.removeprefix("nist_part_") not in PARTS:
            removed.append(name)
            world.remove(body)
    if len(removed) != 7:
        raise ValueError("Expected seven other loose parts to be removed")
    points = {}
    for name, (x, y, yaw, rest) in LAYOUT.items():
        native = MESHES / f"{name}_m.stl"
        vertices = np.asarray(o3d.io.read_triangle_mesh(str(native)).vertices).copy()
        rotation = Rotation.from_euler("z", yaw, degrees=True)
        if rest == "pin_side":
            rotation *= Rotation.from_euler("y", 90., degrees=True)
        elif rest == "broad_side":
            rotation *= Rotation.from_euler("x", 90., degrees=True)
        rotated = vertices @ rotation.as_matrix().T
        position = [x, y, .001 - rotated[:, 2].min()]
        body = world.find(f"body[@name='nist_part_{name}']")
        body.set("pos", values(position))
        body.set("quat", values(rotation.as_quat()[[3, 0, 1, 2]]))
        points[name] = vertices
    referenced = {geom.get("mesh") for geom in root.findall(".//geom")}
    for mesh in list(root.findall("asset/mesh")):
        if mesh.get("name", "").startswith("nist_") and mesh.get("name") not in referenced:
            root.find("asset").remove(mesh)
    # Presentation lighting and matte table only; native part colours retained.
    material = root.find("asset/material[@name='workbench_matte']")
    material.set("rgba", ".48 .53 .55 1")
    material.set("specular", ".04")
    for name, strength, shadow in (("light1", .55, "true"), ("light2", .35, "false"),
                                   ("workbench_task_light", .30, "false")):
        light = root.find(f"worldbody/light[@name='{name}']")
        light.set("diffuse", values([strength]*3))
        light.set("castshadow", shadow)
    quality = root.find("visual/quality")
    if quality is None:
        quality = ET.SubElement(root.find("visual"), "quality")
    quality.set("shadowsize", "4096")
    quality.set("offsamples", "4")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    robot_qa, robot_va = [], []
    for jid in range(model.njnt):
        name = model.joint(jid).name
        if name.startswith("nist_part_"):
            continue
        old_jid = old.joint(name).id
        nq, nv = joint_widths(model.jnt_type[jid])
        qa, va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        old_qa = int(old.jnt_qposadr[old_jid])
        data.qpos[qa:qa+nq] = old_data.qpos[old_qa:old_qa+nq]
        robot_qa.extend(range(qa, qa+nq)); robot_va.extend(range(va, va+nv))
    arm_ids = [model.joint(f"robot0_joint{i}").id for i in range(1, 8)]
    lower, upper = model.jnt_range[arm_ids].T
    if np.any(ARM_QPOS_RAD <= lower) or np.any(ARM_QPOS_RAD >= upper):
        raise ValueError("Camera pose violates robot joint limits")
    data.qpos[model.jnt_qposadr[arm_ids]] = ARM_QPOS_RAD
    if model.nq != 58:
        raise ValueError("Unexpected seven-part state dimension")
    records, steps = settle_and_check(model, data, points, np.array(robot_qa), np.array(robot_va),
                                     parts=PARTS, settling_seconds=5.)
    robot_contacts = []
    for contact in data.contact:
        names = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
        if any(name.startswith(("robot0_", "gripper0_")) for name in names):
            robot_contacts.append(names)
    if robot_contacts:
        raise RuntimeError(f"Robot contacts at the camera pose: {robot_contacts}")
    for name, record in records.items():
        body = world.find(f"body[@name='nist_part_{name}']")
        body.set("pos", values(record["position_world_m"]))
        body.set("quat", values(record["quaternion_wxyz"]))
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name=KEY,
                  qpos=values(data.qpos), qvel=values(data.qvel))
    return root, model, data, records, steps, source_hash, removed


def export(model, data, folder):
    width, height = FULL_SIZE_WH
    camera = "robot0_eye_in_hand"
    option = mujoco.MjvOption(); option.geomgroup[:] = [0, 1, 1, 0, 0, 0]; option.sitegroup[:] = 0
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
    model.vis.quality.offsamples = 0
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.enable_depth_rendering(); renderer.update_scene(data, camera=camera, scene_option=option)
        depth = renderer.render().copy(); renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering(); renderer.update_scene(data, camera=camera, scene_option=option)
        segmentation = renderer.render().copy()
    boxes, visibility = [], {}
    for name in PARTS:
        gid = model.geom("nist_part_"+name+"_visual").id
        y, x = np.nonzero((segmentation[:, :, 0] == gid) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))
        if len(x) < 150:
            raise RuntimeError(f"Part is too small or outside view: {name}: {len(x)}")
        box = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
        boxes.append(box); visibility[name] = {"pixels": len(x), "bbox_xyxy": box}
    boxes = np.array(boxes)
    low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    center, span = (low+high)/2, (high-low)*1.22
    crop_width = max(span[0], span[1]*4/3)
    crop_height = crop_width*3/4
    x0, y0 = np.floor(center-[crop_width/2, crop_height/2]).astype(int)
    x1, y1 = np.ceil(center+[crop_width/2, crop_height/2]).astype(int)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise RuntimeError("Figure crop exceeds sensor frame")
    robot_gids = [gid for gid in range(model.ngeom)
                  if model.body(int(model.geom_bodyid[gid])).name.startswith(("robot0_", "gripper0_"))]
    robot_mask = np.isin(segmentation[:, :, 0], robot_gids) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
    if robot_mask[y0:y1, x0:x1].any():
        raise RuntimeError("Figure crop still contains the robot/gripper")
    Image.fromarray(rgb).save(folder/"wrist_rgb_full.png")
    Image.fromarray(rgb[y0:y1, x0:x1]).save(folder/"figure_rgb.png")
    np.save(folder/"depth_full_m.npy", depth)
    crop_depth = depth[y0:y1, x0:x1].copy()
    np.save(folder/"depth_m.npy", crop_depth)
    Image.fromarray(np.rint(crop_depth*1000).astype(np.uint16)).save(folder/"depth_mm.png")
    np.savez_compressed(folder/"audit_segmentation.npz", segmentation=segmentation)
    cid = model.camera(camera).id
    f = height/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
    K = np.array([[f, 0., width/2], [0., f, height/2], [0., 0., 1.]])
    K_crop = K.copy(); K_crop[0, 2] -= x0; K_crop[1, 2] -= y0
    T = np.eye(4); T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.]); T[:3, 3] = data.cam_xpos[cid]
    np.save(folder/"intrinsics_full.npy", K); np.save(folder/"intrinsics.npy", K_crop); np.save(folder/"world_T_camera.npy", T)
    with mujoco.Renderer(model, width=1600, height=1100) as renderer:
        renderer.update_scene(data, camera="agentview", scene_option=option)
        Image.fromarray(renderer.render()).save(folder/"agentview.png")
    return {"camera": camera, "native_fovy_deg": float(model.cam_fovy[cid]),
            "camera_position_world_m": T[:3, 3].tolist(),
            "camera_tilt_from_vertical_deg": float(np.degrees(np.arccos(-T[2, 2]))),
            "full_resolution_wh": [width, height], "figure_crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "figure_resolution_wh": [int(x1-x0), int(y1-y0)], "figure_crop_robot_pixels": 0,
            "intrinsics_full": K.tolist(), "intrinsics_figure_crop": K_crop.tolist(),
            "world_T_camera": T.tolist(), "audit_only_visibility": visibility,
            "raw_depth_range_m": [float(depth.min()), float(depth.max())],
            "figure_depth_range_m": [float(crop_depth.min()), float(crop_depth.max())]}


def main():
    root, model, data, parts, steps, source_hash, removed = build_scene()
    folder = ROOT/"outputs/simulation/framework_figure_scene"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=False)
    scene = folder/"scene.xml"; ET.ElementTree(root).write(scene, encoding="unicode")
    reloaded = mujoco.MjModel.from_xml_path(str(scene)); ready = mujoco.MjData(reloaded)
    mujoco.mj_resetDataKeyframe(reloaded, ready, reloaded.key(KEY).id); mujoco.mj_forward(reloaded, ready)
    np.testing.assert_allclose(data.xpos, ready.xpos, atol=1e-12)
    capture = export(reloaded, ready, folder)
    if digest(SOURCE) != source_hash:
        raise RuntimeError("Original scene changed")
    summary = {"source_scene": str(SOURCE), "source_sha256": source_hash,
               "scene_sha256": digest(scene), "source_unchanged": True, "keyframe": KEY,
               "loose_parts": PARTS, "removed_other_loose_parts": removed,
               "settled_parts": parts, "settling_steps": steps, "physics_timestep_s": PHYSICS_TIMESTEP_S,
               "robot_arm_qpos_rad": ARM_QPOS_RAD.tolist(), "robot_pose_contacts": 0,
               "robot_camera_mount_unchanged": True, "native_part_scales_and_colours_retained": True,
               "presentation_changes": ["Compact seven-part layout", "Oblique static robot wrist configuration",
                                        "Matte neutral table, lower diffuse lighting and key-light shadows",
                                        "Documented figure crop with correspondingly adjusted intrinsics"],
               "capture": capture,
               "limitations": ["Figure scene, not a new perception accuracy experiment.",
                               "Robot held at the selected simulated pose during part settling; no motion trajectory tested.",
                               "Full native wrist frame retained; figure is a crop, not a changed camera calibration.",
                               "Simulator poses and segmentation are audit-only."]}
    (folder/"scene_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (folder/"README.md").write_text("# Framework figure scene\n\nLoad scene.xml in the existing WSL MuJoCo environment and reset keyframe framework_figure_ready.\n\nfigure_rgb.png is the clean, unlabelled figure crop of the unchanged mounted wrist camera. wrist_rgb_full.png retains the complete native sensor frame. depth_m.npy and depth_mm.png align with the crop; depth_full_m.npy aligns with the full image. Both intrinsics are saved, with the crop principal point adjusted. world_T_camera.npy uses X right, Y down, Z forward.\n\nThe scene contains three round pins, a large gear, waterproof, USB and D-sub connectors. All are physically settled at native CAD scale. Robot pose is a static simulation setup; no real robot or motion trajectory was executed. Table/lighting are presentation changes documented in scene_summary.json.\n")
    print(json.dumps({"output": str(folder), "figure_resolution_wh": capture["figure_resolution_wh"],
                      "camera_tilt_deg": capture["camera_tilt_from_vertical_deg"], "parts": list(PARTS)}), flush=True)


if __name__ == "__main__":
    main()
