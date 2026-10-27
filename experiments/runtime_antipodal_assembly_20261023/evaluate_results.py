"""Independently evaluate saved runtime-grasp trials; never change their results.

Usage: python -m experiments.runtime_antipodal_assembly_20261023.evaluate_results RUN_PATH
This evaluator uses NumPy/SciPy, the saved physics trajectories, and native CAD.
It does not import a controller, start a simulator, or command robot hardware.
"""

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .extract_goal_audit import native_mesh, top_boundaries


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
GEOMETRY_TOLERANCE_M = .000002
ROBOT_PENETRATION_LIMIT_M = .001
NOMINAL_POSITION_TOLERANCE_M = .005
NOMINAL_ROTATION_TOLERANCE_DEG = 5.
SETTLE_LINEAR_SPEED_M_S = .01
SETTLE_ANGULAR_SPEED_RAD_S = .05
STAGES = ("runtime_grasp_generated", "sampled_pickup_plan_passed", "sampled_plan_passed", "bilateral_physical_pickup",
          "physical_lift_passed", "released_nominal_pose_reached", "released_and_settled", "released_nominal_placement",
          "coarse_insertion_passed", "strict_aperture_compatible", "strict_recorded_insertion_passed")


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rigid_pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Invalid saved 4x4 pose")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous pose row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Invalid saved rotation")
    return pose


def ordered_polygon(points):
    """Board apertures are star-shaped about their measured bounding centers."""
    points = np.asarray(points, dtype=float)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    return points[order]


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_containment(points, polygon, tolerance=GEOMETRY_TOLERANCE_M):
    """Test a convex peg section against the actual (possibly nonconvex) aperture.

    A winding test checks the section's vertices. Proper edge intersections
    additionally reject sections spanning an inward notch between those vertices.
    Returned distances are numerical geometry diagnostics, not 1 mm fit gates.
    """
    points, polygon = np.asarray(points, float), np.asarray(polygon, float)
    if len(points) < 3 or len(polygon) < 3:
        return {"compatible": False, "minimum_clearance_m": None,
                "maximum_outside_distance_m": None, "reason": "degenerate_section_or_aperture"}
    edges = np.roll(polygon, -1, axis=0) - polygon
    rel = points[:, None, :] - polygon[None, :, :]
    lengths2 = np.sum(edges * edges, axis=1)
    fractions = np.clip(np.sum(rel * edges[None, :, :], axis=2) / lengths2, 0, 1)
    distances = np.linalg.norm(rel - fractions[:, :, None] * edges[None, :, :], axis=2)
    distance = distances.min(axis=1)
    x, y = points[:, 0, None], points[:, 1, None]
    x1, y1 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x2, y2 = np.roll(polygon, -1, axis=0).T[:, None, :]
    denominator = y2 - y1
    crossing_x = x1 + (y - y1) * (x2 - x1) / np.where(abs(denominator) < 1e-30, 1e-30, denominator)
    crossings = ((y1 > y) != (y2 > y)) & (x < crossing_x)
    inside = (np.count_nonzero(crossings, axis=1) % 2 == 1) | (distance <= tolerance)
    signed = np.where(inside, distance, -distance)
    # The sections are convex because the six native peg models are convex.
    section = points[ConvexHull(points).vertices]
    section_edges = np.roll(section, -1, axis=0) - section
    a = section[:, None, :]
    b = a + section_edges[:, None, :]
    c = polygon[None, :, :]
    d = c + edges[None, :, :]
    o1, o2 = cross2(b - a, c - a), cross2(b - a, d - a)
    o3, o4 = cross2(d - c, a - c), cross2(d - c, b - c)
    # A proper crossing has endpoints on opposite sides by more than the
    # numerical tolerance. Merely touching an edge is handled by distance.
    eps1 = tolerance * np.linalg.norm(b - a, axis=2)
    eps2 = tolerance * np.linalg.norm(d - c, axis=2)
    proper = (((o1 > eps1) & (o2 < -eps1)) | ((o2 > eps1) & (o1 < -eps1))) & (
        ((o3 > eps2) & (o4 < -eps2)) | ((o4 > eps2) & (o3 < -eps2)))
    compatible = bool(np.all(inside) and not np.any(proper))
    return {"compatible": compatible, "minimum_clearance_m": float(signed.min()),
            "maximum_outside_distance_m": float(max(0., -signed.min())),
            "proper_boundary_crossing": bool(np.any(proper)),
            "section_boundary_vertices": len(section)}


def slice_mesh(vertices, edges, pose, plane_z):
    """Intersect native triangle edges with a horizontal plate plane exactly."""
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    first, second = world[edges[:, 0]], world[edges[:, 1]]
    dz = second[:, 2] - first[:, 2]
    valid = abs(dz) > 1e-12
    fraction = (plane_z - first[valid, 2]) / dz[valid]
    crossed = (fraction >= 0.) & (fraction <= 1.)
    points = first[valid][crossed] + fraction[crossed, None] * (second[valid][crossed] - first[valid][crossed])
    endpoints = world[np.isclose(world[:, 2], plane_z, atol=1e-10)]
    if len(endpoints):
        points = np.concatenate([points, endpoints])
    if not len(points):
        return None
    xy = np.unique(np.round(points[:, :2], 12), axis=0)
    return xy if len(xy) >= 3 else None


def xml_body_pose(root, requested):
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") == "degree"

    def visit(body, parent):
        pose = np.eye(4)
        pose[:3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        if "quat" in body.attrib:
            quat = np.fromstring(body.get("quat"), sep=" ")
            pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        elif "euler" in body.attrib:
            pose[:3, :3] = Rotation.from_euler("xyz", np.fromstring(body.get("euler"), sep=" "), degrees=degrees).as_matrix()
        world = parent @ pose
        if body.get("name") == requested:
            return world
        for child in body.findall("body"):
            found = visit(child, world)
            if found is not None:
                return found
        return None

    for body in root.findall("worldbody/body"):
        found = visit(body, np.eye(4))
        if found is not None:
            return found
    raise ValueError(f"Missing body: {requested}")


def target_velocity_slice(root, name, states):
    """Recover free-joint velocity indices and verify against saved target XYZ."""
    qadr, dadr = 0, 0
    for body in root.find("worldbody").iter("body"):
        for joint in body:
            if joint.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
            nq, nv = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
            if body.get("name") == "nist_part_" + name:
                if kind != "free":
                    raise ValueError("Target joint is not free")
                for state in (states[0], states[-1]):
                    if not np.allclose(np.asarray(state["qpos"])[qadr:qadr + 3],
                                       np.asarray(state["target_pose_audit"])[:3, 3], atol=1e-7):
                        raise ValueError("XML joint ordering does not reproduce saved target position")
                return slice(dadr, dadr + 3), slice(dadr + 3, dadr + 6)
            qadr += nq
            dadr += nv
    raise ValueError("Could not locate target free joint")


def task_orientation_error(pose, goal, name):
    if name.startswith("RGOCG"):
        return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(pose[:3, 2], goal[:3, 2])), 0, 1))))
    symmetries = [np.eye(3)]
    if name.startswith("KET"):
        # Opposite ends and 180-degree cross-section rotations of these
        # unchamfered rectangular prisms have identical geometry.
        symmetries += [Rotation.from_euler(axis, 180, degrees=True).as_matrix() for axis in "xyz"]
    errors = [np.linalg.norm(Rotation.from_matrix(pose[:3, :3] @ (goal[:3, :3] @ symmetry).T).as_rotvec())
              for symmetry in symmetries]
    return float(np.rad2deg(min(errors)))


def peg_aperture(name, goal_record, audit, xml_root):
    plate_path = MESHES / "GMC_Laser_Plate_Virtual_m.stl"
    plate, plate_triangles = native_mesh(plate_path)
    top_circles, top_other = top_boundaries(plate, plate_triangles)
    flipped = plate.copy()
    flipped[:, 2] *= -1
    bottom_circles, bottom_other = top_boundaries(flipped, plate_triangles)
    if name.startswith("RGOCG"):
        opening_id = goal_record["opening_id"]
        top = next(item for item in top_circles if item["audit_hole_id"] == opening_id)
    else:
        target_extent = np.sort(np.asarray(audit["parts"][name]["extent_m"][:2]))
        matches = [item for item in top_other if np.all((np.sort(item["extent_xy_m"]) - target_extent > 0)
                   & (np.sort(item["extent_xy_m"]) - target_extent < .0005))]
        if len(matches) != 1:
            raise ValueError("No unique dimension-compatible rectangular aperture")
        top = matches[0]
        opening_id = "measured_rectangular_opening"
    center = np.asarray(top["center_board_xy_m"])
    bottom = min([*bottom_circles, *bottom_other], key=lambda item: np.linalg.norm(np.asarray(item["center_board_xy_m"]) - center))
    if np.linalg.norm(np.asarray(bottom["center_board_xy_m"]) - center) > .001:
        raise ValueError("Could not match top and bottom aperture boundaries")
    world_T_board = xml_body_pose(xml_root, "nist_board")
    if not np.allclose(world_T_board[:3, 2], [0, 0, 1], atol=1e-6):
        raise ValueError("Evaluator currently requires a horizontal board")
    planes = []
    for label, boundary, z in (("top", top, float(plate[:, 2].max())),
                               ("bottom", bottom, float(plate[:, 2].min()))):
        polygon = ordered_polygon(boundary["top_boundary_xy_m"])
        world = np.c_[polygon, np.full(len(polygon), z)] @ world_T_board[:3, :3].T + world_T_board[:3, 3]
        planes.append({"label": label, "z_m": float(world[0, 2]), "polygon_xy_m": world[:, :2]})
    vertices, triangles = native_mesh(MESHES / f"{name}_m.stl")
    edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                             triangles[:, [2, 0]]]), axis=1), axis=0)
    return vertices, edges, planes, {"opening_id": opening_id,
        "plate_mesh_sha256": file_hash(plate_path), "board_world_T_mesh": world_T_board.tolist(),
        "top_extent_xy_m": top["extent_xy_m"], "bottom_extent_xy_m": bottom["extent_xy_m"],
        "top_minimum_inscribed_diameter_m": top.get("minimum_inscribed_diameter_m"),
        "bottom_minimum_inscribed_diameter_m": bottom.get("minimum_inscribed_diameter_m")}


def sections_at_pose(vertices, edges, planes, pose):
    sections = []
    for plane in planes:
        points = slice_mesh(vertices, edges, pose, plane["z_m"])
        if points is None:
            sections.append({"plane": plane["label"], "intersects_plane": False, "compatible": False})
        else:
            sections.append({"plane": plane["label"], "intersects_plane": True,
                             **polygon_containment(points, plane["polygon_xy_m"])})
    return sections


def evaluate_part(part_dir, audit):
    result = read_json(part_dir / "result.json", {})
    name = result.get("part", part_dir.name)
    states = read_json(part_dir / "trajectory.json", [])
    contacts = read_json(part_dir / "contacts.json", [])
    controller = read_json(part_dir / "controller_audit.json", result.get("controller_audit", {}))
    row = {"part": name, "raw_status": result.get("status", "missing_result"),
           "error": result.get("error"), "recorded_frames": len(states),
           "runtime_grasp_generated": bool(result.get("qualified_unique_pairs", 0) > 0 and result.get("generated_gripper_poses", 0) > 0),
           "qualified_contact_pairs": result.get("qualified_unique_pairs"),
           "generated_gripper_poses": result.get("generated_gripper_poses"),
           "table_clear_candidates": result.get("table_clear_candidates"),
           "sampling_seconds": result.get("sampling_seconds"),
           "planning_seconds": result.get("planning_seconds"), "total_seconds": result.get("total_seconds"),
           "sampled_plan_passed": bool(result.get("plan") and len(result.get("path_checks", [])) == 7 and
                                       all(item.get("status") == "passed" for item in result["path_checks"])),
           "strict_mating_status": "unavailable_collision_model_and_mating_specification",
           "strict_mating_passed": None,
           "source_hashes": {path.name: file_hash(path) for path in [part_dir / "result.json", part_dir / "trajectory.json",
                              part_dir / "controller_audit.json", part_dir / "contacts.json", part_dir / "scene.xml"] if path.is_file()}}
    row.update({stage: None for stage in STAGES if stage not in row})
    pickup_checks = result.get("pickup_path_checks", [])
    row["sampled_pickup_plan_passed"] = bool(row["sampled_plan_passed"] or (
        result.get("plan") and len(pickup_checks) == 3 and all(item.get("status") == "passed" for item in pickup_checks)))
    row["assembly_planning_status"] = result.get("assembly_planning_status")
    pickup_only = bool(pickup_checks and not row["sampled_plan_passed"])
    row["pickup_only_fallback"] = pickup_only
    physics_executed = bool(len(states) > 1 and any(state.get("phase") != "initial" for state in states))
    row["physics_executed"] = physics_executed
    if not states:
        row["evaluation_status"] = "unavailable_missing_trajectory"
        return row
    poses = np.asarray([rigid_pose(state["target_pose_audit"]) for state in states])
    final, initial = poses[-1], poses[0]
    phases = np.asarray([state["phase"] for state in states])
    assembly_attempted = bool(any(phase in phases for phase in ("turn", "transfer", "insert_or_nominal_place")))
    insertion_attempted = bool("insert_or_nominal_place" in phases)
    row["assembly_execution_attempted"] = assembly_attempted
    row["insertion_or_placement_attempted"] = insertion_attempted
    bilateral = np.asarray([len(set(state.get("grasp_contacts", []))) == 2 for state in states])
    rise = poses[:, 2, 3] - initial[2, 3]
    has_no_attachments = controller.get("attachments_present") is False
    no_target_feedback = controller.get("target_feedback_in_controller") is False
    contact_driven = has_no_attachments and no_target_feedback
    close_ids, lift_ids = np.flatnonzero(phases == "close"), np.flatnonzero(phases == "lift")
    close_ok = bool(len(close_ids) and bilateral[close_ids[-1]])
    row["bilateral_close_confirmed"] = close_ok
    row["attachment_free_contact_execution"] = contact_driven
    row["maximum_recorded_lift_m"] = float(rise.max())
    row["bilateral_physical_pickup"] = bool(contact_driven and close_ok and np.any((phases == "lift") & bilateral & (rise >= .02))) if physics_executed else None
    row["physical_lift_passed"] = bool(row["bilateral_physical_pickup"] and len(lift_ids) and
        rise[lift_ids[-1]] >= .12 and bilateral[lift_ids[-1]]) if physics_executed else None
    row["end_lift_height_m"] = float(rise[lift_ids[-1]]) if len(lift_ids) else None
    row["final_world_T_cad"] = final.tolist()
    row["last_recorded_phase"] = str(phases[-1])
    settled_ids = np.flatnonzero(phases == "settle")
    tail = settled_ids[-min(25, len(settled_ids)):]
    xml_root = ET.parse(part_dir / "scene.xml").getroot()
    try:
        linear_slice, angular_slice = target_velocity_slice(xml_root, name, states)
        linear = np.asarray([np.asarray(state["qvel"])[linear_slice] for state in states])
        angular = np.asarray([np.asarray(state["qvel"])[angular_slice] for state in states])
        linear_speed, angular_speed = np.linalg.norm(linear, axis=1), np.linalg.norm(angular, axis=1)
        row["velocity_source"] = "saved_free_joint_qvel_verified_against_target_translation"
    except (KeyError, ValueError) as error:
        times = np.asarray([state["time_s"] for state in states])
        delta_t = np.diff(times)
        linear_speed = np.r_[0., np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / np.maximum(delta_t, 1e-12)]
        rotations = poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)
        angular_speed = np.r_[0., np.linalg.norm(Rotation.from_matrix(rotations).as_rotvec(), axis=1) / np.maximum(delta_t, 1e-12)]
        row["velocity_source"] = "recorded_frame_pose_difference"
        row["velocity_index_warning"] = str(error)
    row["final_linear_speed_m_s"] = float(linear_speed[-1])
    row["final_angular_speed_rad_s"] = float(angular_speed[-1])
    row["settle_max_linear_speed_m_s"] = float(linear_speed[tail].max()) if len(tail) else None
    row["settle_max_angular_speed_rad_s"] = float(angular_speed[tail].max()) if len(tail) else None
    completed = bool(len(settled_ids) >= 10 and phases[-1] == "settle" and "release" in phases and "retreat" in phases)
    stable = bool(completed and linear_speed[tail].max() <= SETTLE_LINEAR_SPEED_M_S and angular_speed[tail].max() <= SETTLE_ANGULAR_SPEED_RAD_S)
    last_contacts = states[-1].get("grasp_contacts", [])
    selected_width = result.get("selected_grasp", {}).get("width_m")
    actual_opening = abs(float(states[-1]["gripper_qpos"][0]) - float(states[-1]["gripper_qpos"][1]))
    fingers_open = bool(selected_width is not None and actual_opening >= selected_width + .003)
    released = bool(completed and has_no_attachments and not last_contacts and fingers_open)
    row["release_sequence_completed"] = completed
    row["fingers_open_after_release"] = fingers_open
    row["settled"] = stable
    row["released_and_settled"] = bool(released and stable) if physics_executed and not pickup_only else None
    penetration = dict(controller.get("max_penetration_m", {}))
    for frame in contacts:
        for item in frame.get("contacts", []):
            kind = item.get("kind")
            penetration[kind] = max(penetration.get(kind, 0.), float(item.get("penetration_m", 0.)))
    row["max_penetration_m"] = penetration
    collision_ok = all(penetration.get(key, float("inf")) < ROBOT_PENETRATION_LIMIT_M
                       for key in ("robot_self", "robot_table", "robot_board"))
    row["robot_contact_audit_passed_at_1mm"] = collision_ok if physics_executed else None
    goal_record = result.get("goal", audit["nominal_simulation_goals"].get(name, {}))
    goal = rigid_pose(result.get("selected_world_T_cad_goal", goal_record.get("world_T_cad_goal")))
    row["goal_position_error_m"] = float(np.linalg.norm(final[:3, 3] - goal[:3, 3]))
    row["goal_xy_error_m"] = float(np.linalg.norm(final[:2, 3] - goal[:2, 3]))
    row["task_orientation_error_deg"] = task_orientation_error(final, goal, name)
    row["released_nominal_pose_reached"] = bool(row["physical_lift_passed"] and released and collision_ok
        and row["goal_position_error_m"] <= NOMINAL_POSITION_TOLERANCE_M
        and row["task_orientation_error_deg"] <= NOMINAL_ROTATION_TOLERANCE_DEG) if physics_executed and not pickup_only else None
    row["released_nominal_placement"] = bool(row["physical_lift_passed"] and released and stable and collision_ok
        and row["goal_position_error_m"] <= NOMINAL_POSITION_TOLERANCE_M
        and row["task_orientation_error_deg"] <= NOMINAL_ROTATION_TOLERANCE_DEG) if physics_executed and not pickup_only else None
    if name.startswith(("RGOCG", "KET")):
        vertices, edges, planes, aperture = peg_aperture(name, goal_record, audit, xml_root)
        row["aperture"] = aperture
        world_vertices = vertices @ final[:3, :3].T + final[:3, 3]
        top_z = next(plane["z_m"] for plane in planes if plane["label"] == "top")
        row["actual_lowest_cad_point_depth_below_plate_top_m"] = float(top_z - world_vertices[:, 2].min())
        row["axis_tilt_deg"] = float(np.rad2deg(np.arccos(np.clip(abs(final[2, 2]), 0, 1))))
        row["coarse_insertion_passed"] = bool(row["physical_lift_passed"] and released and stable and collision_ok
            and row["actual_lowest_cad_point_depth_below_plate_top_m"] > .009
            and world_vertices[:, 2].max() > top_z and row["goal_xy_error_m"] < .001
            and row["axis_tilt_deg"] < 3 and penetration.get("target_board", float("inf")) < .001) if physics_executed and not pickup_only else None
        final_sections = sections_at_pose(vertices, edges, planes, final)
        row["final_aperture_sections"] = final_sections
        nominal_sections = sections_at_pose(vertices, edges, planes, goal)
        row["nominal_goal_aperture_sections"] = nominal_sections
        row["nominal_goal_aperture_compatible"] = bool(all(section["intersects_plane"] and section["compatible"] for section in nominal_sections))
        row["strict_aperture_compatible"] = bool(all(section["intersects_plane"] and section["compatible"] for section in final_sections)) if insertion_attempted else None
        sampled = []
        for index, state in enumerate(states):
            if state["phase"] in ("insert_or_nominal_place", "hold_goal", "release", "retreat", "settle"):
                for section in sections_at_pose(vertices, edges, planes, poses[index]):
                    if section["intersects_plane"]:
                        sampled.append({"time_s": state["time_s"], "phase": state["phase"], **section})
        violations = [section for section in sampled if not section["compatible"]]
        row["recorded_aperture_sections_checked"] = len(sampled)
        row["recorded_aperture_violations"] = len(violations)
        row["maximum_recorded_aperture_outside_distance_m"] = max((section["maximum_outside_distance_m"] or 0. for section in sampled), default=None)
        row["first_recorded_aperture_violation"] = violations[0] if violations else None
        row["strict_recorded_insertion_passed"] = bool(row["coarse_insertion_passed"] and row["strict_aperture_compatible"]
            and sampled and not violations) if physics_executed and not pickup_only else None
        row["strict_mating_status"] = "geometric_peg_aperture_check_available_not_official_assembly_validation"
    row["evaluation_status"] = "evaluated_saved_evidence"
    row["first_unsatisfied_stage"] = next((stage for stage in STAGES[:7] if row.get(stage) is False), None)
    return row


def plot_summary(run_dir, summary, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return False
    labels = ["Grasps", "Pickup plan", "Assembly plan", "Physical pickup", "Lift >=12 cm", "Released nominal pose",
              "Released + settled", "Settled nominal goal", "Coarse insertion", "Final aperture", "Recorded strict insertion"]
    values = np.asarray([[1 if row.get(stage) is True else 0 if row.get(stage) is False else -1
                          for stage in STAGES] for row in rows])
    if not len(rows):
        return False
    fig, (bars, matrix) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 2.7]},
                                      layout="constrained")
    counts = summary["counts"]
    success = [counts[stage]["passed"] for stage in STAGES]
    available = [counts[stage]["available"] for stage in STAGES]
    x = np.arange(len(STAGES))
    bars.bar(x, available, color="#e0e5e8", width=.65)
    bars.bar(x, success, color="#22836c", width=.65)
    for index, (passed, tested) in enumerate(zip(success, available)):
        bars.text(index, max(tested, 0) + .25, f"{passed}/{tested}", ha="center", fontsize=10)
    bars.set_ylim(0, max(len(rows) + 2, 4))
    bars.set_xticks(x, labels, rotation=22, ha="right")
    bars.set_ylabel("Parts (passed / available)")
    bars.spines[["top", "right"]].set_visible(False)
    bars.set_title("Runtime CAD grasps and nominal assembly: independently recomputed outcomes", loc="left", fontsize=15)
    matrix.imshow(values, cmap=ListedColormap(["#e0e5e8", "#c96358", "#22836c"]), vmin=-1, vmax=1, aspect="auto")
    matrix.set_xticks(x, labels, rotation=25, ha="right")
    matrix.set_yticks(np.arange(len(rows)), [row["part"] for row in rows])
    for y in range(len(rows)):
        for x_index in range(len(STAGES)):
            matrix.text(x_index, y, "PASS" if values[y, x_index] == 1 else "FAIL" if values[y, x_index] == 0 else "—",
                        ha="center", va="center", color="white" if values[y, x_index] >= 0 else "#5b6872", fontsize=8)
    matrix.set_title("Green = passed · red = failed · gray = unavailable or not attempted", loc="left", fontsize=11)
    fig.savefig(run_dir / "summary.png", dpi=160, facecolor="white")
    plt.close(fig)
    return True


def write_report(run_dir, summary, rows):
    fields = ["part", "raw_status", *STAGES, "recorded_frames", "maximum_recorded_lift_m", "goal_position_error_m",
              "goal_xy_error_m", "task_orientation_error_deg", "final_linear_speed_m_s", "final_angular_speed_rad_s",
              "actual_lowest_cad_point_depth_below_plate_top_m", "maximum_recorded_aperture_outside_distance_m",
              "recorded_aperture_violations", "strict_mating_status", "planning_seconds", "total_seconds", "error"]
    with (run_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "evaluator_path": str(Path(__file__).resolve()),
                  "evaluator_sha256": file_hash(Path(__file__)),
                  "goal_audit_sha256": summary["goal_audit_sha256"],
                  "raw_run_manifest_sha256": file_hash(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else None,
                  "note": "Evaluation source was added after run start; original run manifest and raw results are preserved.",
                  "input_hashes_by_part": {row["part"]: row.get("source_hashes", {}) for row in rows}}
    (run_dir / "evaluation_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plotted = plot_summary(run_dir, summary, rows)
    lines = ["# Runtime CAD Grasp / Nominal Assembly Evaluation", "", 
             "This is an isolated simulator trial with assumed correct identities, perfect initial CAD alignment, and user-authorized nominal goals. Other loose parts are excluded. It does not measure perception or real assembly reliability.", "",
             f"Completed result records: **{len(rows)} / {summary['expected_parts']}**. Run complete: **{summary['run_complete']}**.", "",
             "| Recomputed stage | Passed | Available |", "| --- | ---: | ---: |"]
    for stage, count in summary["counts"].items():
        lines.append(f"| {stage.replace('_', ' ')} | {count['passed']} | {count['available']} |")
    lines += ["", "Grasp generation, sampled kinematic plans, physical pickup, nominal placement, coarse insertion and strict aperture compatibility are separate outcomes. `sampled_plan_passed` requires the full seven-segment assembly plan; `sampled_pickup_plan_passed` also includes a separately validated pickup-only fallback. A successful pickup-only fallback has unavailable release/assembly outcomes, not a failed grasp. An upstream `insertion_passed` flag is not accepted as precision insertion evidence.", "",
              "Strict peg geometry uses native CAD cross sections at both plate surfaces, actual top/bottom aperture polygons, and a 2 µm numerical tolerance. Recorded insertion frames are also checked. This test detects rectangular yaw errors and lateral errors larger than native clearance; it does not certify unrecorded substeps or real contact tolerances.", "",
              "`released_nominal_pose_reached` reports release and <=5 mm / <=5° final pose error without asserting settling or mating. `released_nominal_placement` additionally requires settling and physical lift. Round-pin yaw and rectangular-prism half-turn symmetries are handled geometrically. Settling requires the final recorded window to remain below 0.01 m/s and 0.05 rad/s.", "",
              "Robot self/table/board overlap is reported separately against the existing 1 mm numerical-contact limit. Peg aperture compatibility uses the native opening instead of that coarse overlap limit.", "",
              "Gear/nut bores and connector cavities/threading/latches are not modeled adequately for strict mating claims. Their strict assembly outcome remains unavailable. Connector nominal targets have zero validated engagement.", "",
              "See [per-part JSON](evaluation.json), [CSV](evaluation.csv), and [snapshot gallery](gallery.html). Raw result, controller and trajectory files are unchanged.", ""]
    if plotted:
        lines += ["![Recomputed stage outcomes](summary.png)", ""]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for row in rows:
        name = row["part"]
        pictures = []
        for stem in ("initial", "closed", "lifted", "above_goal", "at_goal", "released", "last_executed_frame"):
            image = run_dir / name / (stem + ".png")
            if image.is_file():
                relative = image.relative_to(run_dir).as_posix()
                pictures.append(f'<figure><img loading="lazy" src="{html.escape(relative)}"><figcaption>{stem}</figcaption></figure>')
        values = " | ".join(f"{stage.replace('_', ' ')}: {row.get(stage)}" for stage in STAGES)
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(str(row.get("raw_status", row.get("evaluation_status"))))}</p><p>{html.escape(values)}</p><div class="images">{"".join(pictures)}</div></section>')
    document = '<!doctype html><meta charset="utf-8"><title>Runtime CAD grasp trial</title><style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#17212b}section{border-top:1px solid #ccd3d9;padding:14px 0}.images{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;width:460px}img{width:100%}figcaption{font-size:13px;color:#52616c}</style><h1>Runtime CAD grasp / nominal assembly</h1><p>Independent evaluation; nominal simulated goals. Strict mating is unavailable for unsupported collision models.</p>' + "".join(cards)
    (run_dir / "gallery.html").write_text(document, encoding="utf-8")


def evaluate_run(run_path):
    run_dir = Path(run_path).resolve()
    manifest = read_json(run_dir / "manifest.json", {})
    audit_path = run_dir / "goal_audit.json"
    if not audit_path.is_file():
        audit_path = HERE / "goal_audit.json"
    audit = read_json(audit_path)
    if audit is None:
        raise FileNotFoundError("Missing goal_audit.json")
    expected = list(manifest.get("parts", audit["existing_objects_centered_parts"]))
    rows = []
    for name in expected:
        part_dir = run_dir / name
        if not (part_dir / "result.json").is_file():
            continue
        try:
            rows.append(evaluate_part(part_dir, audit))
        except Exception as error:
            rows.append({"part": name, "evaluation_status": "evaluation_error", "error": str(error),
                         **{stage: None for stage in STAGES}})
    counts = {stage: {"passed": sum(row.get(stage) is True for row in rows),
                      "failed": sum(row.get(stage) is False for row in rows),
                      "available": sum(row.get(stage) is not None for row in rows),
                      "unavailable": sum(row.get(stage) is None for row in rows)} for stage in STAGES}
    summary = {"schema_version": 1, "run_directory": str(run_dir), "expected_parts": len(expected),
        "evaluated_parts": len(rows), "run_complete": len(rows) == len(expected),
        "evaluation_errors": sum(row.get("evaluation_status") == "evaluation_error" for row in rows),
        "counts": counts, "rows": rows,
        "goal_audit_sha256": file_hash(audit_path),
        "goal_audit_matches_manifest": file_hash(audit_path) == manifest.get("goal_audit_sha256"),
        "evaluator_sha256": file_hash(Path(__file__)),
        "thresholds": {"geometry_numerical_tolerance_m": GEOMETRY_TOLERANCE_M,
            "robot_penetration_limit_m": ROBOT_PENETRATION_LIMIT_M,
            "nominal_goal_position_tolerance_m": NOMINAL_POSITION_TOLERANCE_M,
            "nominal_goal_rotation_tolerance_deg": NOMINAL_ROTATION_TOLERANCE_DEG,
            "settle_linear_speed_m_s": SETTLE_LINEAR_SPEED_M_S,
            "settle_angular_speed_rad_s": SETTLE_ANGULAR_SPEED_RAD_S},
        "limitations": ["Finite grasp sampling and ten kinematic waypoints per planned segment do not certify a continuous path.",
            "Strict aperture checks cover recorded states, not every physics substep.",
            "Gear, nut and connector strict mating remains unavailable under current collision models.",
            "All goals are nominal simulation assumptions; round pin 12 uses a larger clearance opening.",
            "Independent per-part initializations are not one uninterrupted assembly sequence."]}
    write_report(run_dir, summary, rows)
    return summary


def self_test():
    aperture = np.array([[-.0061, -.0041], [.0061, -.0041], [.0061, .0041], [-.0061, .0041]])
    section = np.array([[-.006, -.004], [.006, -.004], [.006, .004], [-.006, .004]])
    assert polygon_containment(section, aperture)["compatible"]
    assert not polygon_containment(section[:, ::-1], aperture)["compatible"], "90-degree rectangular yaw must fail"
    assert not polygon_containment(section + [.0002, 0], aperture)["compatible"], "Native 0.1 mm margin cannot accept 0.2 mm offset"
    circle = np.c_[np.cos(np.linspace(0, 2 * np.pi, 72, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 72, endpoint=False))]
    assert polygon_containment(circle * .004, circle * .0041)["compatible"]
    assert not polygon_containment(circle * .004 + [.0003, 0], circle * .0041)["compatible"]
    prism = np.array([[x, y, z] for z in (-.025, .025) for x, y in section])
    edges = np.array([[i, i + 4] for i in range(4)])
    pose = np.eye(4)
    sliced = slice_mesh(prism, edges, pose, 0.)
    assert sliced.shape == (4, 2) and polygon_containment(sliced, aperture)["compatible"]
    tilted = np.eye(4)
    tilted[:3, :3] = Rotation.from_euler("y", 3, degrees=True).as_matrix()
    assert not polygon_containment(slice_mesh(prism, edges, tilted, .02), aperture)["compatible"]
    print("Passed geometry regression checks: aligned fit, wrong yaw, excess offset, circle clearance, exact slicing and tilted prism.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2:
        output = evaluate_run(Path(sys.argv[1]))
        print(json.dumps({"run": output["run_directory"], "complete": output["run_complete"],
                          "evaluation_errors": output["evaluation_errors"], "counts": output["counts"]}, indent=2))
    else:
        raise SystemExit("Usage: python -m experiments.runtime_antipodal_assembly_20261023.evaluate_results RUN_PATH")
