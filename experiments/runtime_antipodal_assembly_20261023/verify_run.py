"""Independent saved-evidence checks for the known-pose simulator experiment.

Run only after the requested experiment has completed. This script reads the
saved models/states directly; it does not import the experiment's planner or
controller, step physics, alter source evidence, or contact robot hardware.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = ROOT / "outputs/runtime_antipodal_assembly_20261023/20261023T213101Z"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(rows, name, passed, **details):
    rows.append({"check": name, "passed": bool(passed), **details})


def close(first, second, atol=1e-9):
    return np.shape(first) == np.shape(second) and bool(np.allclose(
        first, second, atol=atol, rtol=0., equal_nan=False))


def pose_of(data, body=None, site=None):
    pose = np.eye(4)
    pose[:3, :3] = (data.xmat[body] if body is not None
                      else data.site_xmat[site]).reshape(3, 3)
    pose[:3, 3] = data.xpos[body] if body is not None else data.site_xpos[site]
    return pose


def logged_finger_sides(contact_frame):
    """Recount both force-bearing fingers from raw saved native contact rows."""
    sides = set()
    for row in contact_frame["contacts"]:
        if row["kind"] != "finger_target" or row["normal_force_n"] <= .02:
            continue
        for body in row["bodies"]:
            if "leftfinger" in body or "finger_joint1" in body:
                sides.add("left")
            if "rightfinger" in body or "finger_joint2" in body:
                sides.add("right")
    return sorted(sides)


def verify_input_scene(manifest, rows):
    source = Path(manifest["source_scene"])
    audit = read_json(source.parent / "scene_input_audit.json")
    check(rows, "prepared_input_scene_hash", digest(source) == manifest["source_scene_sha256"]
          == audit["prepared_scene_sha256"])
    check(rows, "external_source_preserved", digest(audit["external_source"])
          == audit["external_source_sha256"])
    records = {item["part"]: item for item in audit["parts"]}
    check(rows, "fourteen_distinct_matching_input_parts",
          len(records) == audit["part_count"] == 14
          and set(records) == set(manifest["parts"])
          and audit["all_14_visual_cad_matches"])
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "objects_centered")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    detail = []
    for part, record in records.items():
        native = Path(record["native_cad_path"])
        body = model.body("nist_part_" + part).id
        pose = pose_of(data, body=body)
        mesh = o3d.io.read_triangle_mesh(str(native))
        vertices = np.asarray(mesh.vertices)
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        native_match = digest(native) == record["native_cad_sha256"]
        pose_match = close(pose, record["world_T_cad_initial"])
        minimum_match = abs(float(world[:, 2].min())
                            - record["native_mesh_minimum_world_z_m"]) < 1e-9
        check(rows, "input_cad_" + part, native_match and pose_match and minimum_match
              and record["visual_cad_exact_match"] and record["active_collision_count"] > 0,
              cad_hash_match=native_match, pose_match=pose_match,
              minimum_height_match=minimum_match)
        detail.append({"part": part, "minimum_world_z_m": float(world[:, 2].min())})
    return audit, records, detail


def verify_transforms(result, rows):
    if "selected_grasp" not in result:
        return {"selected_grasp_available": False}
    initial = np.asarray(result["world_T_cad_input"])
    selected = np.asarray(result["selected_grasp"]["world_T_grip"])
    plan = {key: np.asarray(value) for key, value in result["plan"].items()}
    object_T_grip = plan["object_T_grip"]
    goal = np.asarray(result["goal"]["world_T_cad_goal"])
    yaw = float(result.get("selected_goal_equivalent_yaw_deg", 0.))
    equivalent_goal = goal.copy()
    equivalent_goal[:3, :3] = goal[:3, :3] @ Rotation.from_euler(
        "z", yaw, degrees=True).as_matrix()
    check(rows, "selected_symmetric_goal_yaw",
          close(result.get("selected_world_T_cad_goal", equivalent_goal), equivalent_goal)
          and (yaw == 0. or result["part"].startswith("RGOCG")), yaw_deg=yaw)
    check(rows, "pick_object_to_gripper_composition",
          close(initial @ object_T_grip, selected) and close(plan["grasp"], selected),
          maximum_matrix_error=float(np.max(np.abs(initial @ object_T_grip - selected))))
    check(rows, "assembly_object_to_gripper_composition",
          close(equivalent_goal @ object_T_grip, plan["assembly"]),
          maximum_matrix_error=float(np.max(np.abs(equivalent_goal @ object_T_grip
                                                   - plan["assembly"]))))
    check(rows, "grasp_inverse_transform", close(object_T_grip @ plan["grip_T_object"], np.eye(4)))
    rotations = [pose[:3, :3] for pose in plan.values()]
    orthogonality_error = max(float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
                              for rotation in rotations)
    determinant_error = max(abs(float(np.linalg.det(rotation)) - 1.) for rotation in rotations)
    # Contact endpoints originate in float32 tensors. Use the same 1e-6 rigid-
    # pose tolerance as the controller, and retain the measured error explicitly.
    rigid_tolerance = 1e-6
    check(rows, "planned_rigid_transforms_finite_and_proper",
          all(np.all(np.isfinite(pose)) and close(pose[3], [0., 0., 0., 1.])
              for pose in plan.values())
          and orthogonality_error <= rigid_tolerance and determinant_error <= rigid_tolerance,
          maximum_orthogonality_error=orthogonality_error,
          maximum_determinant_error=determinant_error, tolerance=rigid_tolerance)
    return {"selected_grasp_available": True, "selected_goal_equivalent_yaw_deg": yaw}


def verify_trajectory(part_dir, result, rows):
    model = mujoco.MjModel.from_xml_path(str(part_dir / "scene.xml"))
    audit = read_json(part_dir / "controller_audit.json")
    states = read_json(part_dir / "trajectory.json")
    contacts = read_json(part_dir / "contacts.json")
    check(rows, "no_equality_or_attachment_constraints", model.neq == 0
          and not audit["attachments_present"], model_equality_count=int(model.neq))
    check(rows, "controller_audit_matches_result", audit == result["controller_audit"])
    check(rows, "recorded_frame_count", len(states) == audit["trajectory_frames"] == len(contacts)
          and len(states) > 0)
    times = np.asarray([state["time_s"] for state in states])
    qpos = np.asarray([state["qpos"] for state in states])
    qvel = np.asarray([state["qvel"] for state in states])
    phases = np.asarray([state["phase"] for state in states])
    step = np.diff(times)
    expected_dt = float(audit["recording_frame_dt_s"])
    check(rows, "strict_monotonic_fixed_recording_times", np.all(step > 0.)
          and np.allclose(step, expected_dt, atol=1e-9, rtol=0.),
          expected_dt_s=expected_dt,
          maximum_dt_error_s=float(np.max(np.abs(step - expected_dt))) if len(step) else 0.)
    check(rows, "finite_state_dimensions", qpos.shape == (len(states), model.nq)
          and qvel.shape == (len(states), model.nv) and np.all(np.isfinite(qpos))
          and np.all(np.isfinite(qvel)) and np.all(np.isfinite(times)))
    with np.load(part_dir / "trajectory.npz", allow_pickle=False) as saved:
        check(rows, "json_npz_trajectory_match", close(saved["qpos"], qpos)
              and close(saved["qvel"], qvel) and close(saved["time_s"], times)
              and np.array_equal(saved["phase"], phases))
    check(rows, "native_contacts_time_phase_match",
          close([item["time_s"] for item in contacts], times)
          and np.array_equal([item["phase"] for item in contacts], phases))
    joints = [model.joint(f"robot0_joint{i}").id for i in range(1, 8)]
    qa = model.jnt_qposadr[joints]
    actual_q = qpos[:, qa]
    check(rows, "recorded_robot_qpos_consistent", close(
        actual_q, [state["robot_qpos"] for state in states]))
    lo, hi = model.jnt_range[joints].T
    limit_violation = float(max(0., np.max(lo - actual_q), np.max(actual_q - hi)))
    actual_step = float(np.max(np.abs(np.diff(actual_q, axis=0)))) if len(states) > 1 else 0.
    check(rows, "actual_recorded_arm_joint_limits", limit_violation <= 1e-5,
          maximum_limit_violation_rad=limit_violation, tolerance_rad=1e-5)
    # The controller also records an attempted-command step before aborting.
    # That value need not describe any executed state and is not used here.
    check(rows, "actual_recorded_joint_continuity", actual_step <= .15,
          maximum_actual_frame_step_rad=actual_step,
          maximum_attempted_command_step_rad=audit["max_joint_frame_step_rad"],
          attempted_command_step_used_as_failure_criterion=False)
    sites = np.asarray([state["grip_site_pose"] for state in states])
    targets = np.asarray([state["target_pose_audit"] for state in states])
    check(rows, "finite_recorded_pose_matrices", np.all(np.isfinite(sites)) and np.all(np.isfinite(targets)))
    sample = {0, len(states) - 1, *np.linspace(0, len(states) - 1, min(25, len(states)), dtype=int)}
    for phase in set(phases):
        indices = np.flatnonzero(phases == phase)
        sample.update((int(indices[0]), int(indices[-1])))
    data = mujoco.MjData(model)
    body = model.body("nist_part_" + result["part"]).id
    site = model.site("gripper0_grip_site").id
    max_site_error, max_target_error = 0., 0.
    for index in sorted(sample):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        max_site_error = max(max_site_error, float(np.max(np.abs(pose_of(data, site=site) - sites[index]))))
        max_target_error = max(max_target_error, float(np.max(np.abs(pose_of(data, body=body) - targets[index]))))
    check(rows, "sampled_forward_kinematics_replay", max_site_error < 1e-9 and max_target_error < 1e-9,
          replayed_frames=len(sample), maximum_site_matrix_error=max_site_error,
          maximum_target_matrix_error=max_target_error)
    force_sides = [logged_finger_sides(item) for item in contacts]
    bilateral = np.asarray([len(sides) == 2 for sides in force_sides])
    check(rows, "independent_force_contact_side_recount", all(
        sides == state["grasp_contacts"] for sides, state in zip(force_sides, states)))
    check(rows, "independent_bilateral_frame_recount", int(bilateral.sum())
          == audit["bilateral_contact_frames"], bilateral_frames=int(bilateral.sum()))
    initial_z = float(result["world_T_cad_input"][2][3])
    heights = targets[:, 2, 3] - initial_z
    lift_indices = np.flatnonzero(phases == "lift")
    last_lift = int(lift_indices[-1]) if len(lift_indices) else None
    bilateral_lift = bool(bilateral[last_lift]) if last_lift is not None else None
    lift_m = float(heights[last_lift]) if last_lift is not None else None
    max_bilateral_lift = (float(heights[lift_indices[bilateral[lift_indices]]].max())
                          if len(lift_indices) and np.any(bilateral[lift_indices]) else None)
    if "bilateral_lift" in result:
        check(rows, "reported_bilateral_lift_recomputed", result["bilateral_lift"] == bilateral_lift)
    if "lift_m" in result:
        check(rows, "reported_lift_height_recomputed", lift_m is not None
              and abs(result["lift_m"] - lift_m) < 1e-9)
    if "max_observed_lift_m" in result:
        check(rows, "maximum_observed_lift_recomputed",
              abs(result["max_observed_lift_m"] - float(heights.max())) < 1e-9)
    if "final_world_T_cad_audit" in result:
        check(rows, "final_pose_matches_last_state", close(result["final_world_T_cad_audit"], targets[-1]))
    return {"frames": len(states), "phase_frame_counts": dict(Counter(phases.tolist())),
            "actual_maximum_joint_step_rad": actual_step,
            "maximum_joint_limit_violation_rad": limit_violation,
            "bilateral_force_frames": int(bilateral.sum()),
            "lift_frames_recorded": len(lift_indices),
            "last_lift_height_m": lift_m, "last_lift_bilateral_force_contact": bilateral_lift,
            "maximum_bilateral_lift_height_m": max_bilateral_lift,
            "reached_12cm_with_bilateral_force_contact": bool(
                max_bilateral_lift is not None and max_bilateral_lift >= .12),
            "contact_force_recount_source": "Saved raw native contact rows, threshold >0.02 N per finger. Forces are not recomputed without saved actuator commands.",
            "last_phase": str(phases[-1]),
            "completed_settle_recording": bool(phases[-1] == "settle")}


def verify(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    results = read_json(run_dir / "results.json")
    checks = []
    parts = manifest["parts"]
    check(checks, "all_requested_parts_completed", len(parts) == 14
          and [result["part"] for result in results] == parts
          and len(set(parts)) == len(parts) and (run_dir / "summary.csv").is_file())
    for relative, expected in manifest["source_hashes"].items():
        path = HERE / relative.replace("\\", "/")
        check(checks, "captured_source_hash_" + relative, path.is_file() and digest(path) == expected)
    check(checks, "goal_audit_hash", digest(HERE / "goal_audit.json") == manifest["goal_audit_sha256"])
    goal_records = read_json(HERE / "goal_audit.json")["nominal_simulation_goals"]
    _, input_parts, initial_heights = verify_input_scene(manifest, checks)
    part_reports = []
    for result in results:
        part = result["part"]
        part_dir = run_dir / part
        rows = []
        try:
            check(rows, "aggregate_and_per_part_result_match", result == read_json(part_dir / "result.json"))
            check(rows, "nominal_goal_record_match", result["goal"] == goal_records[part])
            check(rows, "input_pose_matches_frozen_input_scene", close(
                result["world_T_cad_input"], input_parts[part]["world_T_cad_initial"]))
            check(rows, "mesh_hash_matches_audited_cad", result["mesh_sha256"]
                  == input_parts[part]["native_cad_sha256"])
            check(rows, "part_source_scene_hash", result["controller_audit"]["source_scene_sha256"]
                  == manifest["source_scene_sha256"])
            transform_report = verify_transforms(result, rows)
            trajectory_report = verify_trajectory(part_dir, result, rows)
            part_reports.append({"part": part, "recorded_status": result["status"],
                                 "passed": all(row["passed"] for row in rows),
                                 **transform_report, **trajectory_report, "checks": rows})
        except Exception as error:
            check(rows, "verification_completed", False, error=f"{type(error).__name__}: {error}")
            part_reports.append({"part": part, "recorded_status": result.get("status"),
                                 "passed": False, "checks": rows})
        print(f"VERIFY {part}: {'PASS' if part_reports[-1]['passed'] else 'FAIL'}", flush=True)
    added_sources = sorted(str(path.relative_to(HERE)) for path in HERE.rglob("*.py")
                           if str(path.relative_to(HERE)) not in manifest["source_hashes"])
    report = {"verification_passed": all(row["passed"] for row in checks)
              and all(part["passed"] for part in part_reports),
              "verified_utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(run_dir),
              "verifier_sha256": digest(__file__), "manifest_sha256": digest(run_dir / "manifest.json"),
              "checks": checks, "parts": part_reports,
              "initial_native_mesh_heights_m": initial_heights,
              "source_files_added_after_manifest_not_part_of_frozen_run": added_sources,
              "interpretation": [
                  "Checks audit saved simulator evidence under explicitly assumed correct identity and initial CAD alignment.",
                  "Verification does not validate perception, real-world calibration, grasp optimality, unseen objects, or physical robot assembly.",
                  "Recorded failure/abort outcomes can pass evidence verification; no outcome is relabeled as assembly success.",
                  "Only captured source hashes are checked. Adding this verifier after the run started does not change frozen experiment code.",
                  "Forward kinematics is recomputed from saved qpos at sampled frames. Raw native contact forces are recounted, not independently dynamically replayed.",
                  "Actual recorded joint continuity is checked separately from attempted commands that were rejected before execution."]}
    output = run_dir / "verification.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"VERIFICATION={output} PASS={report['verification_passed']}", flush=True)
    return report


if __name__ == "__main__":
    report = verify(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN)
    raise SystemExit(0 if report["verification_passed"] else 1)
