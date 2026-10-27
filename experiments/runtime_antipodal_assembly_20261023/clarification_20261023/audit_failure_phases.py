"""Read-only audit of failure phases in the frozen 14-part simulator trial."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

from experiments.runtime_antipodal_assembly_20261023.evaluate_results import (
    MESHES, peg_aperture, sections_at_pose,
)
from experiments.runtime_antipodal_assembly_20261023.extract_goal_audit import native_mesh


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
RUN = ROOT / "outputs/runtime_antipodal_assembly_20261023/20261023T213101Z"
PHASE_ORDER = ("approach", "descend", "lift", "turn", "transfer", "insert", "retreat")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planning_audit(rejections):
    groups = defaultdict(list)
    for rejected in rejections:
        last = rejected["checks"][-1]
        groups[(last["phase"], last["status"])].append((rejected, last))
    records = []
    for (phase, status), items in groups.items():
        record = {"failed_segment": phase, "reason": status, "attempts": len(items)}
        errors = []
        pair_counts = Counter()
        body_counts = Counter()
        depths = []
        for _, check in items:
            match = re.search(r"position_error_m=([0-9.e+-]+), rotation_error_rad=([0-9.e+-]+)", check.get("reason", ""))
            if match:
                errors.append((float(match[1]), float(match[2])))
            for contact in check.get("contacts", []):
                depths.append(float(contact.get("penetration_m", 0.)))
            for pair in {tuple(sorted(contact["bodies"])) for contact in check.get("contacts", []) if "bodies" in contact}:
                pair_counts[pair] += 1
            for body in {contact["body"] for contact in check.get("contacts", []) if "body" in contact}:
                body_counts[body] += 1
        if errors:
            values = np.asarray(errors)
            record["failed_IK_position_error_range_m"] = [float(values[:, 0].min()), float(values[:, 0].max())]
            record["failed_IK_rotation_error_range_deg"] = np.rad2deg([values[:, 1].min(), values[:, 1].max()]).tolist()
        if pair_counts:
            record["collision_pairs"] = [{"bodies": list(pair), "rejected_attempts": count} for pair, count in pair_counts.items()]
        if body_counts:
            record["target_obstructing_robot_bodies"] = dict(body_counts)
        if depths:
            record["reported_overlap_range_m"] = [min(depths), max(depths)]
        record["example"] = items[0][0]
        records.append(record)
    best = max(rejections, key=lambda item: (PHASE_ORDER.index(item["checks"][-1]["phase"]),
                                             item["checks"][-1]["checked_waypoints"]), default=None)
    return {"rejected_candidate_goal_pairs": len(rejections),
            "distinct_grasp_candidates_tried": len({item["candidate_index"] for item in rejections}),
            "failure_distribution": records, "furthest_partial_plan": best}


def audit_part(name, audit):
    directory = RUN / name
    result = json.loads((directory / "result.json").read_text())
    trajectory = json.loads((directory / "trajectory.json").read_text())
    contacts = json.loads((directory / "contacts.json").read_text())
    goal = np.asarray(result.get("selected_world_T_cad_goal", result["goal"]["world_T_cad_goal"]))
    pose0 = np.asarray(trajectory[0]["target_pose_audit"])
    phase_indices = defaultdict(list)
    for index, state in enumerate(trajectory):
        phase_indices[state["phase"]].append(index)
    phases = []
    for phase, ids in phase_indices.items():
        state = trajectory[ids[-1]]
        pose = np.asarray(state["target_pose_audit"])
        phases.append({"phase": phase, "recorded_frames": len(ids),
            "start_time_s": trajectory[ids[0]]["time_s"], "end_time_s": state["time_s"],
            "end_target_world_xyz_m": pose[:3, 3].tolist(),
            "end_target_goal_position_error_m": float(np.linalg.norm(pose[:3, 3] - goal[:3, 3])),
            "end_target_axis_tilt_deg": float(np.rad2deg(np.arccos(np.clip(abs(pose[2, 2]), 0, 1)))),
            "end_target_rotation_error_deg": float(np.rad2deg(np.arccos(np.clip((np.trace(pose[:3, :3] @ goal[:3, :3].T) - 1) / 2, -1, 1)))),
            "end_bilateral_finger_contact": len(set(state["grasp_contacts"])) == 2})
    row = {"part": name, "raw_status": result["status"], "raw_error": result.get("error"),
        "last_recorded_phase": trajectory[-1]["phase"], "phases": phases,
        "sampling_counts": {key: result.get(key) for key in ("qualified_unique_pairs", "generated_gripper_poses", "table_clear_candidates", "table_clear_at_pick_and_goal")},
        "assembly_planning": planning_audit(result.get("planning_rejections", [])),
        "pickup_fallback_planning": planning_audit(result.get("pickup_planning_rejections", [])),
        "maximum_proposed_joint_change_rad_per_frame": result["controller_audit"]["max_joint_frame_step_rad"],
        "maximum_recorded_lift_m": float(max(np.asarray(state["target_pose_audit"])[2, 3] - pose0[2, 3] for state in trajectory)),
        "max_substep_penetration_m": result["controller_audit"]["max_penetration_m"],
        "source_files": {path.name: {"path": str(path), "sha256": sha(path)} for path in [directory / "result.json", directory / "trajectory.json", directory / "contacts.json", directory / "controller_audit.json", directory / "scene.xml"]}}
    if name == "M16_Hex_Nut":
        for threshold in (5, 15, 45, 80):
            event = next((state for state in trajectory if np.rad2deg(np.arccos(np.clip(abs(np.asarray(state["target_pose_audit"])[2, 2]), 0, 1))) > threshold), None)
            row[f"first_axis_tilt_exceeds_{threshold}_deg"] = {"time_s": event["time_s"], "phase": event["phase"]} if event else None
    for kind in ("robot_board", "target_board"):
        events = [{"time_s": frame["time_s"], "phase": frame["phase"], **contact}
                  for frame in contacts for contact in frame["contacts"] if contact["kind"] == kind]
        row[f"first_recorded_{kind}_contact"] = events[0] if events else None
        row[f"maximum_recorded_{kind}_normal_force_contact"] = max(events, key=lambda event: event["normal_force_n"], default=None)
        row[f"last_recorded_{kind}_contacts"] = [item for item in contacts[-1]["contacts"] if item["kind"] == kind]
    if name.startswith(("RGOCG", "KET")):
        vertices, _ = native_mesh(MESHES / f"{name}_m.stl")
        xml = ET.parse(directory / "scene.xml").getroot()
        v, edges, planes, aperture = peg_aperture(name, result["goal"], audit, xml)
        top_z = next(plane["z_m"] for plane in planes if plane["label"] == "top")
        downward = [state for state in trajectory if state["phase"] == "insert_or_nominal_place"]
        geometry = []
        for state in downward:
            pose = np.asarray(state["target_pose_audit"])
            points = vertices @ pose[:3, :3].T + pose[:3, 3]
            sections = sections_at_pose(v, edges, planes, pose)
            geometry.append({"time_s": state["time_s"],
                "lowest_cad_point_above_plate_top_m": float(points[:, 2].min() - top_z),
                "plate_plane_sections": sections})
        row["peg_downward_command_recorded"] = bool(downward)
        row["peg_downward_command_frames"] = len(downward)
        row["peg_sections_intersecting_plate_during_downward_command"] = sum(section["intersects_plane"] for sample in geometry for section in sample["plate_plane_sections"])
        row["closest_peg_to_plate_during_downward_command"] = min(geometry, key=lambda sample: sample["lowest_cad_point_above_plate_top_m"], default=None)
        row["opening"] = aperture
    return row


def main():
    audit_path = EXPERIMENT / "goal_audit.json"
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    rows = [audit_part(name, audit) for name in manifest["parts"]]
    value = {"source_run": str(RUN), "read_only_frozen_inputs": True,
        "meaning_of_two_peg_attempts": "Two peg runs entered the downward-command phase. Neither peg crossed the plate surface; this is not two actual hole-entry trials.",
        "scene": {"source_scene": manifest["source_scene"], "source_scene_sha256": manifest["source_scene_sha256"],
            "objects": manifest["parts"], "object_count": len(manifest["parts"]),
            "independent_initializations": True,
            "planning_collision_scope": ["robot self", "workbench", "target obstruction during open approach"],
            "dynamic_contacts_also_include": ["task board", "fixed shafts/screws/female connector fixtures", "target and fingers"],
            "other_loose_parts": "hidden, noncolliding and reset/frozen in each part trial"},
        "goal_audit_sha256": sha(audit_path), "source_manifest_sha256": sha(RUN / "manifest.json"),
        "helper_sha256": sha(Path(__file__)), "parts": rows}
    output = HERE / "failure_phase_audit.json"
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "parts": len(rows),
        "peg_downward_commands": sum(row.get("peg_downward_command_recorded", False) for row in rows),
        "actual_plate_plane_intersections": sum(row.get("peg_sections_intersecting_plate_during_downward_command", 0) for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
