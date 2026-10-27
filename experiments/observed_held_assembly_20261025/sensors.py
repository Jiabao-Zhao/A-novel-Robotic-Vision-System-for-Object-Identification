"""RGB-D and robot-feedback boundary for observation-only held-part estimation.

Camera locations are workspace constants. This module never requests object
poses, instance IDs, segmentation, target names, or target masks.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image


WIDTH, HEIGHT = 1024, 1024
CAMERAS = (
    ("overhead", "birdview", (.20, 0., 1.30), (.20, 0., 0.), 45.),
    ("inspection_front", "frontview", (.30, -.55, .42), (-.15, -.025, .16), 45.),
    ("inspection_back", "sideview", (.30, .55, .42), (-.15, -.025, .16), 45.),
    ("board_inspection", "frontview", (0., -.90, .65), (-.52, -.275, .08), 50.),
)
VISUAL_GROUPS = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
ROBOT_PREFIXES = ("robot0_", "gripper0_")


def world_T_camera(position, lookat):
    """Known OpenCV camera axes: right, down, forward in world metres."""
    position, lookat = np.asarray(position, float), np.asarray(lookat, float)
    forward = lookat - position
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0., 0., 1.])
    if abs(forward @ reference_up) > .99:
        reference_up = np.array([0., 1., 0.])
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((right, down, forward))
    result[:3, 3] = position
    return result


def calibration(position, lookat, fovy_deg, name):
    focal = HEIGHT / (2 * np.tan(np.deg2rad(fovy_deg / 2)))
    return {
        "camera_name": name, "width": WIDTH, "height": HEIGHT,
        "intrinsics": [[focal, 0., (WIDTH - 1) / 2],
                       [0., focal, (HEIGHT - 1) / 2], [0., 0., 1.]],
        "world_T_camera": world_T_camera(position, lookat).tolist(),
        "camera_axes": "OpenCV: +X right, +Y down, +Z forward",
        "depth_units": "metres; optical camera Z", "world_units": "metres",
        "table_z_world_m": 0.,
        "calibration_source": "Explicit fixed simulator camera calibration",
    }


def unproject_pixels(row_col, optical_depth_m, camera_calibration):
    pixels = np.atleast_2d(np.asarray(row_col, float))
    depth = np.broadcast_to(np.asarray(optical_depth_m, float), (len(pixels),))
    intrinsics = np.asarray(camera_calibration["intrinsics"])
    transform = np.asarray(camera_calibration["world_T_camera"])
    points = np.column_stack(((pixels[:, 1] - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                              (pixels[:, 0] - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth))
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_world(points_world_m, camera_calibration):
    transform = np.asarray(camera_calibration["world_T_camera"])
    intrinsics = np.asarray(camera_calibration["intrinsics"])
    points = (np.atleast_2d(np.asarray(points_world_m)) - transform[:3, 3]) @ transform[:3, :3]
    pixels = np.column_stack((intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2],
                              intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2]))
    return pixels, points[:, 2]


def robot_feedback(model, data):
    arm_names = [f"robot0_joint{i}" for i in range(1, 8)]
    finger_names = [f"gripper0_finger_joint{i}" for i in (1, 2)]
    arm = np.asarray([model.joint(name).id for name in arm_names])
    fingers = np.asarray([model.joint(name).id for name in finger_names])
    site = model.site("gripper0_grip_site").id
    transform = np.eye(4)
    transform[:3, :3] = data.site_xmat[site].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[site]
    return {
        "joint_names": arm_names,
        "joint_position_rad": data.qpos[model.jnt_qposadr[arm]].tolist(),
        "joint_velocity_rad_s": data.qvel[model.jnt_dofadr[arm]].tolist(),
        "finger_joint_names": finger_names,
        "finger_position_m": data.qpos[model.jnt_qposadr[fingers]].tolist(),
        "world_T_grip": transform.tolist(),
    }


def _robot_geoms(model):
    """Known robot geometry only; do not identify or inspect any product."""
    return np.array([model.body(int(body)).name.startswith(ROBOT_PREFIXES)
                     for body in model.geom_bodyid], dtype=bool)


def capture_observation(model, data, output_dir=None):
    """Return raw cameras and robot feedback, with no object-state information.

    robot_depth_m predicts visible known robot geometry at the measured joints;
    its empty pixels are infinity. It is separate from the unmodified scene RGB-D.
    All temporary camera and appearance changes are restored before returning.
    """
    qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), float(data.time)
    rgba = model.geom_rgba.copy()
    camera_arrays = {name: getattr(model, name).copy()
                     for name in ("cam_pos", "cam_quat", "cam_fovy", "cam_ipd")}
    offsamples = int(model.vis.quality.offsamples)
    offwidth, offheight = int(model.vis.global_.offwidth), int(model.vis.global_.offheight)
    cameras = {}
    options = mujoco.MjvOption()
    options.geomgroup[:] = VISUAL_GROUPS
    options.sitegroup[:] = 0
    known_robot = _robot_geoms(model)
    try:
        model.vis.quality.offsamples = 0
        model.vis.global_.offwidth = max(WIDTH, offwidth)
        model.vis.global_.offheight = max(HEIGHT, offheight)
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            for name, scene_name, position, lookat, fovy in CAMERAS:
                camera = model.camera(scene_name).id
                if int(model.cam_bodyid[camera]) != 0:
                    raise ValueError("Sensor camera must attach to the fixed world")
                cal = calibration(position, lookat, fovy, name)
                optical_rotation = np.asarray(cal["world_T_camera"])[:3, :3]
                render_rotation = optical_rotation @ np.diag([1., -1., -1.])
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, render_rotation.ravel())
                model.cam_pos[camera] = position
                model.cam_quat[camera] = quaternion
                model.cam_fovy[camera] = fovy
                model.cam_ipd[camera] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=options)
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=camera, scene_option=options)
                depth = renderer.render().copy()
                model.geom_rgba[~known_robot, 3] = 0
                renderer.update_scene(data, camera=camera, scene_option=options)
                robot_depth = renderer.render().copy()
                # Empty far-plane values have noticeable float error. The known
                # robot's bounding spheres give a conservative finite limit.
                robot_far = np.max(np.linalg.norm(data.geom_xpos[known_robot] - np.asarray(position), axis=1)
                                   + model.geom_rbound[known_robot])
                robot_depth[robot_depth > robot_far + .001] = np.inf
                model.geom_rgba[:] = rgba
                renderer.disable_depth_rendering()
                cameras[name] = {"rgb": rgb, "depth_m": depth,
                                 "robot_depth_m": robot_depth, "calibration": cal}
    finally:
        model.geom_rgba[:] = rgba
        for name, values in camera_arrays.items():
            getattr(model, name)[:] = values
        model.vis.quality.offsamples = offsamples
        model.vis.global_.offwidth, model.vis.global_.offheight = offwidth, offheight
        mujoco.mj_forward(model, data)
    assert np.array_equal(qpos, data.qpos) and np.array_equal(qvel, data.qvel) and data.time == time
    observation = {"cameras": cameras, "robot_feedback": robot_feedback(model, data)}
    if output_dir is not None:
        save_observation(observation, output_dir)
    return observation


def save_observation(observation, output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "robot_feedback.json").write_text(json.dumps(observation["robot_feedback"], indent=2) + "\n")
    for name, entry in observation["cameras"].items():
        Image.fromarray(entry["rgb"]).save(folder / f"{name}_rgb.png")
        np.save(folder / f"{name}_depth_m.npy", entry["depth_m"])
        np.save(folder / f"{name}_robot_depth_m.npy", entry["robot_depth_m"])
        (folder / f"{name}_calibration.json").write_text(json.dumps(entry["calibration"], indent=2) + "\n")
