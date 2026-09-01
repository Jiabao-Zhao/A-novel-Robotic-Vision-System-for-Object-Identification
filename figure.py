import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from robot_controller import RTDECommander, RTDEStateFeedback, rg2_gripper_plan


ROOT = Path(__file__).resolve().parent
REGISTRATION_SUMMARY_PATH = ROOT / "output" / "registered_point_cloud" / "cad_registration_summary.json"
ROBOT_BASE_POSE_PATH = ROOT / "output" / "robot_pose" / "object_pose_base.json"

APPROACH_CLEARANCE_MM = 80.0
PLACE_Z_MARGIN_MM = 2.0
GEAR_GRASP_WIDTH_MM = 60.0
FIGURE_MOVE_SPEED = 0.12
FIGURE_MOVE_ACCELERATION = 0.20


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_output_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _selected_record(records, role, fallback_words):
    for record in records:
        if record.get("instruction_role") == role:
            return record

    words = tuple(word.lower() for word in fallback_words)
    for record in records:
        text = f"{record.get('object_type', '')} {record.get('object_class', '')}".lower()
        if all(word in text for word in words):
            return record

    raise ValueError(f"Could not find object record for role={role!r}, words={fallback_words!r}.")


def _cloud_points_base_mm(cloud_path, T_base_from_camera_mm):
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {cloud_path}")

    points_camera_m = np.asarray(cloud.points, dtype=float)
    rotation = T_base_from_camera_mm[:3, :3]
    translation = T_base_from_camera_mm[:3, 3]
    return points_camera_m @ rotation.T * 1000.0 + translation


def _pose_from_registration(record, T_base_from_camera_mm):
    registration = record["registration"]
    T_camera_from_cad_m = np.asarray(registration["T_observed_from_cad"], dtype=float)
    T_camera_from_cad_mm = T_camera_from_cad_m.copy()
    T_camera_from_cad_mm[:3, 3] *= 1000.0
    T_base_from_cad_mm = T_base_from_camera_mm @ T_camera_from_cad_mm

    cloud_path = _resolve_output_path(registration["aligned_cad_cloud_path"])
    points_base_mm = _cloud_points_base_mm(cloud_path, T_base_from_camera_mm)
    center_base_mm = points_base_mm.mean(axis=0)
    bottom_z_mm = float(points_base_mm[:, 2].min())
    top_z_mm = float(points_base_mm[:, 2].max())

    object_rpy_base_deg = Rotation.from_matrix(T_base_from_cad_mm[:3, :3]).as_euler(
        "xyz",
        degrees=True,
    )
    pick_rpy_base_deg = np.array([180.0, 0.0, object_rpy_base_deg[2]], dtype=float)

    return {
        "object_id": record["object_id"],
        "object_type": record["object_type"],
        "instruction_role": record.get("instruction_role"),
        "center_base_mm": center_base_mm,
        "bottom_z_mm": bottom_z_mm,
        "top_z_mm": top_z_mm,
        "height_mm": top_z_mm - bottom_z_mm,
        "object_rpy_base_deg": object_rpy_base_deg,
        "pick_rpy_base_deg": pick_rpy_base_deg,
    }


def _load_scene_poses():
    records = _read_json(REGISTRATION_SUMMARY_PATH)
    robot_pose = _read_json(ROBOT_BASE_POSE_PATH)
    T_base_from_camera_mm = np.asarray(robot_pose["T_base_from_camera_mm"], dtype=float)

    white_record = _selected_record(records, "moved_object", ("white", "gear"))
    red_record = _selected_record(records, "reference_object", ("red", "block"))

    return (
        _pose_from_registration(white_record, T_base_from_camera_mm),
        _pose_from_registration(red_record, T_base_from_camera_mm),
    )


def _pose_xyz_rpy(xyz_mm, rpy_deg):
    return [
        float(xyz_mm[0]),
        float(xyz_mm[1]),
        float(xyz_mm[2]),
        float(rpy_deg[0]),
        float(rpy_deg[1]),
        float(rpy_deg[2]),
    ]


def _print_pose(label, pose):
    x, y, z, roll, pitch, yaw = pose
    print(
        f"{label}: "
        f"x={x:.1f} mm, y={y:.1f} mm, z={z:.1f} mm, "
        f"roll={roll:.1f} deg, pitch={pitch:.1f} deg, yaw={yaw:.1f} deg"
    )


def _checkpoint(message):
    input(f"\n{message}\nPress Enter to continue...")


def sequence():
    white, red = _load_scene_poses()
    gripper_plan = rg2_gripper_plan(GEAR_GRASP_WIDTH_MM)

    white_pick_z = white["bottom_z_mm"] + gripper_plan["tcp_reference_z_offset_mm"]
    white_pick_pose = _pose_xyz_rpy(
        [white["center_base_mm"][0], white["center_base_mm"][1], white_pick_z],
        white["pick_rpy_base_deg"],
    )
    white_approach_pose = white_pick_pose.copy()
    white_approach_pose[2] += APPROACH_CLEARANCE_MM

    place_z = red["top_z_mm"] + gripper_plan["tcp_reference_z_offset_mm"] + PLACE_Z_MARGIN_MM
    place_pose = _pose_xyz_rpy(
        [red["center_base_mm"][0], red["center_base_mm"][1], place_z],
        red["pick_rpy_base_deg"],
    )
    red_approach_pose = place_pose.copy()
    red_approach_pose[2] += APPROACH_CLEARANCE_MM

    print("Figure sequence loaded from current CAD registration output.")
    print(f"Moved object: {white['object_type']} ({white['object_id']})")
    print(f"Reference object: {red['object_type']} ({red['object_id']})")
    _print_pose("White approach", white_approach_pose)
    _print_pose("White pick", white_pick_pose)
    _print_pose("Red approach", red_approach_pose)
    _print_pose("Red place", place_pose)
    print(
        "\nThe robot will move after each keyboard checkpoint. "
        "Clear the workspace and keep the emergency stop reachable."
    )
    if input("Type RUN to start the figure sequence: ").strip() != "RUN":
        print("Sequence cancelled.")
        return

    state = None
    robot = None
    try:
        state = RTDEStateFeedback()
        robot = RTDECommander(state)
        robot.vel = FIGURE_MOVE_SPEED
        robot.acc = FIGURE_MOVE_ACCELERATION

        robot.gripper_set(gripper_plan["open_width_mm"], 10.0)
        robot.move_to_cartesian(*white_approach_pose)
        _checkpoint("Checkpoint 1: approach point above the white gear.")

        robot.move_to_cartesian(*white_pick_pose)
        robot.gripper_set(gripper_plan["close_width_mm"], 10.0)
        robot.move_to_cartesian(*white_approach_pose)
        robot.go_home(open_gripper=False)
        _checkpoint("Checkpoint 2: home position while holding the white gear.")

        robot.move_to_cartesian(*red_approach_pose)
        _checkpoint("Checkpoint 3: approach point above the red block.")

        robot.move_to_cartesian(*place_pose)
        robot.gripper_set(gripper_plan["open_width_mm"], 10.0)
        _checkpoint("Checkpoint 4: white gear placed on top of the red block.")

        robot.move_to_cartesian(*red_approach_pose)
        robot.go_home(open_gripper=True)
        _checkpoint("Checkpoint 5: final home position.")

    finally:
        if robot is not None:
            robot.disconnect()
        if state is not None:
            state.stop()


if __name__ == "__main__":
    sequence()
