"""Render saved grasp selections without moving objects or running physics."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/runtime_antipodal_assembly_20261023"
ORIGINAL = BASE / "20261023T213101Z"
FOLLOWUP = BASE / "8mm_runtime_followup/20261024T012239Z"
SOURCE = BASE / "input_scene/scene.xml"
VALIDATION = BASE / "grasp_pose_visualization_validation/20261024T021959Z/validation.json"
SOURCE_AUDIT = BASE / "grasp_pose_gallery_20261023/source_audit.json"
GROUPS = {
    "gears_and_nuts": [
        ("Gear_Large", "Large gear"), ("Gear_Medium", "Medium gear"),
        ("Gear_Small", "Small gear"), ("M12_Hex_Nut", "M12 nut"),
        ("M16_Hex_Nut", "M16 nut")],
    "pins": [
        ("RGOCG16-50_16mm", "16 mm round pin"), ("KET16_Square_16mm", "16 mm rectangular pin"),
        ("RGOCG12-50_12mm", "12 mm round pin"), ("KET12_Square_12mm", "12 mm rectangular pin"),
        ("RGOCG8-50_8mm", "8 mm round pin — successful follow-up"),
        ("KET8_Square_8mm", "8 mm rectangular pin")],
    "connectors": [
        ("Waterproof_Male", "Waterproof connector"), ("USB_Male", "USB connector"),
        ("DSUB_Male", "D-sub connector")],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size, bold=False):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def setup_model():
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("objects_centered").id)
    return model, data


def show_only_part_and_gripper(model, part):
    """Color only this derived picture, leaving all input files unchanged."""
    target_prefix = "nist_part_" + part
    hand_geom = None
    for index in range(model.ngeom):
        name = model.geom(index).name or ""
        model.geom_group[index] = 5
        if name == target_prefix + "_visual":
            model.geom_group[index] = 1
            model.geom_rgba[index] = [1., .57, .06, 1.]
            model.geom_matid[index] = -1
        elif name.startswith("gripper0_") and "collision" in name:
            model.geom_group[index] = 2
            model.geom_matid[index] = -1
            if "hand" in name:
                model.geom_rgba[index] = [.18, .42, .73, .20]
                hand_geom = index
            elif "pad" in name:
                model.geom_rgba[index] = [.10, .36, .70, .58]
            else:
                model.geom_rgba[index] = [.18, .46, .78, .43]
        elif name == "workbench_top":
            model.geom_group[index] = 1
            model.geom_matid[index] = -1
            model.geom_rgba[index] = [.92, .93, .94, 1.]
    # A plain background keeps the very small contacts visible.
    model.vis.rgba.haze[:] = [.98, .98, .98, 1.]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 800)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 640)
    return hand_geom


def add_contact_markers(scene, contacts, pose, detail):
    for point in contacts:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([.001, .001, .001]), np.asarray(point), np.eye(3).ravel(),
            np.array([.02, .88, .28, 1.], np.float32))
        geom.emission = .6
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
    midpoint = np.mean(contacts, axis=0)
    start = midpoint - pose[:3, 2] * (.026 if detail else .045)
    end = midpoint - pose[:3, 2] * .006
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.ones(3),
        np.zeros(3), np.eye(3).ravel(), np.array([.06, .36, .95, .95], np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .001,
                         np.asarray(start), np.asarray(end))
    geom.emission = .3
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def camera_for(pose, contacts, extent, detail=False):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    midpoint = np.mean(contacts, axis=0)
    camera.lookat[:] = midpoint if detail else midpoint - .045 * pose[:3, 2]
    direction = (pose[:3, 1] * (-1. if detail else 1.)
                 + pose[:3, 0] * .22 + np.array([0., 0., .48 if detail else .60]))
    camera.azimuth = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
    camera.elevation = float(-np.rad2deg(np.arctan2(direction[2], np.linalg.norm(direction[:2]))))
    camera.distance = max(.10, 1.8 * float(np.max(extent))) if detail else .30
    return camera


def mark_contacts_on_image(image, scene, contacts):
    """Keep both planned surface points readable, including the far-side one."""
    left, right = scene.camera
    position = (left.pos + right.pos) / 2
    forward = (left.forward + right.forward) / 2
    forward /= np.linalg.norm(forward)
    up = (left.up + right.up) / 2
    up /= np.linalg.norm(up)
    horizontal = np.cross(forward, up)
    near = (left.frustum_near + right.frustum_near) / 2
    top = (left.frustum_top + right.frustum_top) / 2
    bottom = (left.frustum_bottom + right.frustum_bottom) / 2
    width, height = image.size
    draw = ImageDraw.Draw(image)
    pixels = []
    for point in contacts:
        delta = point - position
        depth = float(np.dot(delta, forward))
        x = width/2 + height * near * np.dot(delta, horizontal) / (depth * (top-bottom))
        y = height * (top - near*np.dot(delta, up)/depth) / (top-bottom)
        assert depth > 0 and 0 <= x < width and 0 <= y < height
        radius = 7
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="white", width=6)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="#00c85a", width=3)
        pixels.append([float(x), float(y)])
    return pixels


def compose_card(label, overview, detail, source_kind, opening, span):
    card = Image.new("RGB", (1660, 850), "#fbfcfe")
    draw = ImageDraw.Draw(card)
    draw.text((22, 14), label, font=font(30, True), fill="#17253b")
    subtitle = ("8 mm follow-up: saved successful automatic grasp" if "follow-up" in source_kind
                else "Original trial: saved generated grasp")
    draw.text((22, 56), subtitle + " — fingers shown open before closing", font=font(19), fill="#4b5c70")
    draw.text((22, 92), "Gripper position", font=font(21, True), fill="#263b53")
    draw.text((842, 92), "Finger contact close-up", font=font(21, True), fill="#263b53")
    card.paste(overview, (20, 126))
    card.paste(detail, (840, 126))
    draw.text((22, 786), "Orange: part   •   Blue: gripper + approach arrow   •   Green rings: planned touches on both sides",
              font=font(20), fill="#263b53")
    draw.text((22, 818), f"Saved open-jaw setting: {opening*1000:.1f} mm   |   Distance between touch points: {span*1000:.1f} mm",
              font=font(17), fill="#526276")
    return card


def render_part(part, label, output, validated):
    folder = output / part
    folder.mkdir()
    result_path = FOLLOWUP / "result.json" if part == "RGOCG8-50_8mm" else ORIGINAL / part / "result.json"
    result = json.loads(result_path.read_text())
    saved = result["selected_grasp"]
    pose = np.asarray(saved["world_T_grip"])
    contacts = np.asarray(saved["contacts_world_m"])
    check = validated[part]
    assert check["passed"] and digest(result_path) == check["source_result_sha256"]
    model, data = setup_model()
    joint_ids = [model.joint(f"robot0_joint{i}").id for i in range(1, 8)]
    qadr = model.jnt_qposadr[joint_ids]
    initial_qpos = data.qpos.copy()
    data.qpos[qadr] = check["robot_qpos"]
    for index, sign in ((1, 1), (2, -1)):
        joint = model.joint(f"gripper0_finger_joint{index}").id
        data.qpos[model.jnt_qposadr[joint]] = sign * saved["opening_m"] / 2
    mujoco.mj_forward(model, data)
    site = model.site("gripper0_grip_site").id
    actual = np.eye(4)
    actual[:3, :3] = data.site_xmat[site].reshape(3, 3)
    actual[:3, 3] = data.site_xpos[site]
    position_error = float(np.linalg.norm(actual[:3, 3] - pose[:3, 3]))
    angle_error = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(
        actual[:3, :3] @ pose[:3, :3].T).as_rotvec())))
    assert position_error <= .0001 and angle_error <= .1
    target_joint = model.joint("nist_part_" + part + "_joint").id
    target_adr = model.jnt_qposadr[target_joint]
    np.testing.assert_array_equal(data.qpos[target_adr:target_adr+7], initial_qpos[target_adr:target_adr+7])
    visual = model.geom("nist_part_" + part + "_visual").id
    mesh = model.geom_dataid[visual]
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[first:first+count] @ data.geom_xmat[visual].reshape(3, 3).T + data.geom_xpos[visual]
    extent = np.ptp(vertices, axis=0)
    hand = show_only_part_and_gripper(model, part)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    images = {}
    contact_pixels = {}
    with mujoco.Renderer(model, height=640, width=800) as renderer:
        for name, detailed in (("overview", False), ("detail", True)):
            if hand is not None:
                model.geom_rgba[hand, 3] = .07 if detailed else .20
            renderer.update_scene(data, camera=camera_for(pose, contacts, extent, detailed), scene_option=option)
            add_contact_markers(renderer.scene, contacts, pose, detailed)
            image = Image.fromarray(renderer.render().copy())
            contact_pixels[name] = mark_contacts_on_image(image, renderer.scene, contacts)
            image.save(folder / f"{name}.png")
            images[name] = image
    source_kind = "8 mm automatic follow-up" if part == "RGOCG8-50_8mm" else "original 14-part trial"
    card = compose_card(label, images["overview"], images["detail"], source_kind,
                        saved["opening_m"], saved["width_m"])
    card.save(folder / "card.png")
    np.savez_compressed(folder / "rendered_state.npz", qpos=data.qpos,
                        world_T_grip_saved=pose, world_T_grip_rendered=actual,
                        contacts_world_m=contacts)
    record = {"part": part, "label": label, "source_kind": source_kind,
              "source_result": str(result_path), "source_result_sha256": digest(result_path),
              "grasp_opening_m": saved["opening_m"], "touch_span_m": saved["width_m"],
              "world_T_grip": pose.tolist(), "contacts_world_m": contacts.tolist(),
              "gripper_position_error_m": position_error, "gripper_rotation_error_deg": angle_error,
              "target_pose_unchanged": True, "physics_steps": 0,
              "contact_markers_pixels": contact_pixels,
              "images": {key: f"{part}/{key}.png" for key in ("overview", "detail", "card")}}
    (folder / "render_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def contact_sheet(output, records, group, members):
    cards = [Image.open(output / record["images"]["card"]).resize((830, 425), Image.Resampling.LANCZOS)
             for record in records if record["part"] in {name for name, _ in members}]
    rows = (len(cards) + 1) // 2
    sheet = Image.new("RGB", (1660, 95 + rows*425), "#f0f4f9")
    draw = ImageDraw.Draw(sheet)
    title = {"gears_and_nuts": "Generated grasp poses — gears and nuts",
             "pins": "Generated grasp poses — round and rectangular pins",
             "connectors": "Generated grasp poses — connectors"}[group]
    draw.text((22, 15), title, font=font(30, True), fill="#17253b")
    draw.text((22, 59), "Saved poses, shown before closing. These pictures do not show assembly results.",
              font=font(21), fill="#465a70")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2)*830, 95 + (index//2)*425))
    path = output / f"{group}.png"
    sheet.save(path)
    return path.name


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = BASE / "grasp_pose_gallery" / stamp
    output.mkdir(parents=True, exist_ok=False)
    validation = json.loads(VALIDATION.read_text())
    assert validation["passed"]
    validated = {row["part"]: row for row in validation["parts"]}
    requested = os.environ.get("RENDER_PARTS")
    selected = set(requested.split(",")) if requested else None
    records = []
    print(f"OUTPUT={output}", flush=True)
    for members in GROUPS.values():
        for part, label in members:
            if selected is not None and part not in selected:
                continue
            records.append(render_part(part, label, output, validated))
            print(f"RENDERED {label}", flush=True)
    sheets = {name: contact_sheet(output, records, name, members)
              for name, members in GROUPS.items() if any(row["part"] in {part for part, _ in members} for row in records)}
    manifest = {"scope": "Static illustrations of saved generated grasp poses, not assembly results",
                "created_utc": stamp, "parts": records, "contact_sheets": sheets,
                "legend": "Orange part; transparent blue actual gripper shapes; green rings mark both planned touch points, including the far side; blue approach arrow",
                "gripper_state": "saved open-jaw setting just before closure",
                "source_scene": str(SOURCE), "source_scene_sha256": digest(SOURCE),
                "validation_source": str(VALIDATION), "validation_sha256": digest(VALIDATION),
                "source_audit": str(SOURCE_AUDIT), "source_audit_sha256": digest(SOURCE_AUDIT),
                "renderer_source_sha256": digest(__file__), "physics_steps": 0,
                "colors_changed_for_illustration_only": True,
                "source_files_changed": False,
                "mixed_source_note": "13 original trial grasps plus the separately marked successful automatic 8 mm follow-up"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
