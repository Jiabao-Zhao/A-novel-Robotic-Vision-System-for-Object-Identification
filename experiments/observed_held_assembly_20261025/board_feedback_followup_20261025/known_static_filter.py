"""Remove depth explained by known robot/table/board geometry, without a part model.

Inputs are sanitized static XML, robot encoders/FK, raw RGB-D and calibration.
There is no target name, estimated CAD pose, mask, body lookup, or state replay.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np


MATCH_TOLERANCE_M = .001
ROBOT_PREFIXES = ("robot0_", "gripper0_")
VISUAL_GROUPS = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_model(template_path, feedback):
    template_path = Path(template_path).resolve()
    tree = ET.parse(template_path)
    root = tree.getroot()
    if root.find("keyframe") is not None:
        raise ValueError("Known-static model must not contain saved keyframes")
    for body in root.findall(".//body"):
        if body.get("name", "").startswith(("nist_part_", "estimated_cad_")):
            raise ValueError("Known-static model must not contain a product or estimated-CAD body")
    for joint in root.findall(".//joint"):
        if joint.get("name") and not joint.get("name").startswith(ROBOT_PREFIXES):
            raise ValueError("Only robot joints may move in the known-static model")
    model = mujoco.MjModel.from_xml_path(str(template_path))
    if model.nq != 9 or model.nv != 9:
        raise ValueError("Known-static model must contain exactly seven robot and two finger joints")
    data = mujoco.MjData(model)
    arm = [f"robot0_joint{index}" for index in range(1, 8)]
    fingers = [f"gripper0_finger_joint{index}" for index in (1, 2)]
    if feedback["joint_names"] != arm or feedback["finger_joint_names"] != fingers:
        raise ValueError("Robot feedback joint names/order disagree")
    for name, position, velocity in zip(arm, feedback["joint_position_rad"], feedback["joint_velocity_rad_s"]):
        joint = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint]] = position
        data.qvel[model.jnt_dofadr[joint]] = velocity
    for name, position in zip(fingers, feedback["finger_position_m"]):
        joint = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint]] = position
    mujoco.mj_forward(model, data)
    site = model.site("gripper0_grip_site").id
    pose = np.eye(4)
    pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site]
    if not np.allclose(pose, np.asarray(feedback["world_T_grip"]), atol=1e-7):
        raise ValueError("Known robot FK disagrees with feedback")
    return model, data


def known_model_asset_files(template_path):
    path = Path(template_path).resolve()
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    paths = {path}
    for element in root.findall("asset/*"):
        name = element.get("file")
        if not name:
            continue
        asset_path = Path(name)
        if not asset_path.is_absolute():
            base = path.parent
            directory = "meshdir" if element.tag == "mesh" else "texturedir"
            if compiler is not None and compiler.get(directory):
                base = Path(compiler.get(directory))
                if not base.is_absolute():
                    base = path.parent / base
            asset_path = base / asset_path
        paths.add(asset_path.resolve())
    return paths


def render_known_depth(model, data, calibration):
    """Metric optical Z from known geometry and camera/robot calibration only."""
    width, height = int(calibration["width"]), int(calibration["height"])
    intrinsics = np.asarray(calibration["intrinsics"], float)
    transform = np.asarray(calibration["world_T_camera"], float)
    if not np.allclose(intrinsics, [[intrinsics[1, 1], 0, (width - 1) / 2],
                                    [0, intrinsics[1, 1], (height - 1) / 2], [0, 0, 1]], atol=1e-7):
        raise ValueError("Known depth renderer expects the calibrated square-pixel centered pinhole camera")
    camera = model.camera("frontview").id
    if int(model.cam_bodyid[camera]) != 0:
        raise ValueError("Known-depth camera must attach to the fixed world")
    model.cam_pos[camera] = transform[:3, 3]
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, (transform[:3, :3] @ np.diag([1., -1., -1.])).ravel())
    model.cam_quat[camera] = quaternion
    model.cam_fovy[camera] = np.rad2deg(2 * np.arctan(height / (2 * intrinsics[1, 1])))
    model.cam_ipd[camera] = 0
    model.vis.quality.offsamples = 0
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    qpos, time = data.qpos.copy(), float(data.time)
    mujoco.mj_forward(model, data)
    options = mujoco.MjvOption()
    options.geomgroup[:] = VISUAL_GROUPS
    options.sitegroup[:] = 0
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera, scene_option=options)
        depth = renderer.render().copy()
    # Exclude the empty rendering far plane, whose floating-point optical-Z
    # conversion is inaccurate near the clip limit. Workspace surfaces are near.
    depth[depth > .98 * model.stat.extent * model.vis.map.zfar] = np.inf
    assert np.array_equal(data.qpos, qpos) and data.time == time
    return depth


def filter_depth(raw_depth_m, known_depth_m):
    raw, known = np.asarray(raw_depth_m), np.asarray(known_depth_m)
    if raw.shape != known.shape or raw.ndim != 2:
        raise ValueError("Raw and known-scene depths must have identical H by W shape")
    valid = np.isfinite(raw) & (raw > 0)
    predicted = np.isfinite(known) & (known > 0)
    with np.errstate(invalid="ignore"):
        removed = valid & predicted & (np.abs(raw - known) <= MATCH_TOLERANCE_M)
    filtered = raw.astype(float, copy=True)
    filtered[removed] = np.nan
    return filtered, removed


def filter_observations(template_path, observation_folder, output_folder):
    """Write an ordinary observation folder for the unchanged held estimator."""
    template_path = Path(template_path).resolve()
    observation_folder, output_folder = Path(observation_folder).resolve(), Path(output_folder).resolve()
    if output_folder == observation_folder or observation_folder in output_folder.parents:
        raise ValueError("Write filtered observations to a separate folder")
    output_folder.mkdir(parents=True, exist_ok=False)
    feedback_path = observation_folder / "robot_feedback.json"
    feedback = json.loads(feedback_path.read_text())
    model, data = checked_model(template_path, feedback)
    inputs = known_model_asset_files(template_path) | {feedback_path}
    calibration_paths = sorted(observation_folder.glob("*_calibration.json"))
    if not calibration_paths:
        raise ValueError("No calibrated camera files found")
    for calibration_path in calibration_paths:
        name = calibration_path.name.removesuffix("_calibration.json")
        inputs.update((calibration_path, observation_folder / f"{name}_rgb.png",
                       observation_folder / f"{name}_depth_m.npy", observation_folder / f"{name}_robot_depth_m.npy"))
    before = {str(path): digest(path) for path in sorted(inputs)}
    cameras = []
    for calibration_path in calibration_paths:
        name = calibration_path.name.removesuffix("_calibration.json")
        calibration = json.loads(calibration_path.read_text())
        raw_path = observation_folder / f"{name}_depth_m.npy"
        raw = np.load(raw_path, allow_pickle=False)
        known = render_known_depth(model, data, calibration)
        filtered, removed = filter_depth(raw, known)
        diagnostic = output_folder / "filter_diagnostics" / name
        diagnostic.mkdir(parents=True, exist_ok=False)
        np.save(diagnostic / "known_static_depth.npy", known)
        np.save(diagnostic / "removed_mask.npy", removed)
        np.save(diagnostic / "filtered_depth.npy", filtered)
        np.save(output_folder / f"{name}_depth_m.npy", filtered)
        for suffix in ("_rgb.png", "_robot_depth_m.npy", "_calibration.json"):
            shutil.copyfile(observation_folder / (name + suffix), output_folder / (name + suffix))
        cameras.append({"camera": name, "raw_valid_pixels": int(np.sum(np.isfinite(raw) & (raw > 0))),
                        "known_scene_valid_pixels": int(np.sum(np.isfinite(known) & (known > 0))),
                        "removed_matching_pixels": int(removed.sum()),
                        "retained_valid_pixels": int(np.sum(np.isfinite(filtered) & (filtered > 0))),
                        "diagnostics": str(diagnostic)})
    shutil.copyfile(feedback_path, output_folder / "robot_feedback.json")
    after = {str(path): digest(path) for path in sorted(inputs)}
    if before != after:
        raise AssertionError("Known-scene filter inputs changed")
    manifest = {"method": "Remove raw optical depth agreeing with calibrated known robot/table/board depth",
        "match_tolerance_m": MATCH_TOLERANCE_M, "invalid_depth_value": "NaN",
        "template_source": "Public sanitized robot/static-board XML; no product bodies or saved state",
        "input_files_sha256": before, "inputs_unchanged": before == after,
        "robot_qpos_count": model.nq, "cameras": cameras,
        "estimated_CAD_or_target_state_input": False, "segmentation_used": False,
        "physics_steps": 0, "pose_estimator_gates_changed": False,
        "limitation": "Actual part surfaces agreeing with known geometry within 1 mm can also be removed",
        "helper_sha256": digest(Path(__file__))}
    manifest_path = output_folder / "known_static_filter.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"observation_folder": str(output_folder), "filter_manifest": str(manifest_path),
            "removed_pixels_per_camera": {row["camera"]: row["removed_matching_pixels"] for row in cameras},
            "physics_steps": 0}
