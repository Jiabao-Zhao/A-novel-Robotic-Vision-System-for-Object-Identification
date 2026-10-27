"""Known-CAD/known-pose simulator trial; never imports physical robot control."""
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_planning import (sample_contacts, candidate_frames, table_filter,
                             linear_pose_path, validate_path)
from .robot_trial import RobotTrial
from .prepare_scene import prepare_scene


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = ROOT / "outputs/datasets/nist_task_board/legacy_workspace_inputs"
SOURCE = LOCAL / "nist-taskboard-workspace/objects-centered/scene.xml"
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
PARTS = ("Gear_Large", "Gear_Medium", "Gear_Small", "Waterproof_Male", "M12_Hex_Nut",
         "M16_Hex_Nut", "RGOCG16-50_16mm", "KET16_Square_16mm", "RGOCG12-50_12mm",
         "RGOCG8-50_8mm", "KET12_Square_12mm", "KET8_Square_8mm", "USB_Male", "DSUB_Male")
SAMPLES = 2048
MAX_CONTACT_PAIRS = 320
MAX_PLANNING_CANDIDATES = 48


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def offset(pose, vector):
    result = pose.copy()
    result[:3, 3] += vector
    return result


def build_plan(trial, candidate, world_T_object, world_T_goal):
    grasp = candidate.world_T_grip
    object_T_grip = np.linalg.inv(world_T_object) @ grasp
    grip_T_object = np.linalg.inv(object_T_grip)
    assembly = world_T_goal @ object_T_grip
    pregrasp = offset(grasp, -.10 * grasp[:3, 2])
    lift = offset(grasp, [0, 0, .18])
    turn = lift.copy()
    turn[:3, :3] = assembly[:3, :3]
    above_goal = offset(assembly, [0, 0, .16])
    retreat = offset(assembly, -.10 * assembly[:3, 2] + [0, 0, .10])
    return {"object_T_grip": object_T_grip, "grip_T_object": grip_T_object,
            "grasp": grasp, "pregrasp": pregrasp, "lift": lift, "turn": turn,
            "above_goal": above_goal, "assembly": assembly, "retreat": retreat}


def check_plan(trial, candidate, plan, pickup_only=False):
    segments = [
        ("approach", trial.pose(), plan["pregrasp"], candidate.opening_m, False, True),
        ("descend", plan["pregrasp"], plan["grasp"], candidate.opening_m, False, True),
        ("lift", plan["grasp"], plan["lift"], candidate.width_m, True, False),
        ("turn", plan["lift"], plan["turn"], candidate.width_m, True, False),
        ("transfer", plan["turn"], plan["above_goal"], candidate.width_m, True, False),
        ("insert", plan["above_goal"], plan["assembly"], candidate.width_m, True, False),
        ("retreat", plan["assembly"], plan["retreat"], candidate.opening_m, False, False),
    ]
    if pickup_only:
        segments = segments[:3]
    q = trial.d.qpos[trial.qa].copy()
    reports = []
    for name, first, last, opening, held, open_target in segments:
        qpath, report = validate_path(trial, linear_pose_path(first, last, 10), opening,
            seed=q, target_relative=plan["grip_T_object"] if held else None,
            check_open_target=open_target)
        reports.append({"phase": name, **report})
        if qpath is None:
            return False, reports
        q = qpath[-1]
    return True, reports


def execute(trial, candidate, plan, world_T_initial, world_T_goal, goal_kind,
            pickup_only=False):
    """Actual contact simulation; commanded poses never use updated target truth."""
    snapshots = []
    def snapshot(name):
        path = trial.output_dir / f"{name}.png"
        trial.render(path)
        snapshots.append(str(path))

    snapshot("initial")
    trial.gripper(candidate.opening_m)
    trial.move(plan["pregrasp"], 100, "approach")
    trial.move(plan["grasp"], 65, "descend")
    trial.gripper(0.)
    trial.hold(35, "close")
    snapshot("closed")
    bilateral_close = len(trial.states[-1]["grasp_contacts"]) == 2
    if not bilateral_close:
        return {"status": "grasp_failed", "reason": "no bilateral contact after closing",
                "snapshots": snapshots}
    trial.move(plan["lift"], 100, "lift")
    lift_pose = trial.target_pose()  # evaluation only, never a control correction
    lift_m = float(lift_pose[2, 3] - world_T_initial[2, 3])
    snapshot("lifted")
    bilateral_lift = len(trial.states[-1]["grasp_contacts"]) == 2
    if lift_m < .12 or not bilateral_lift:
        return {"status": "lift_failed", "lift_m": lift_m,
                "bilateral_lift": bilateral_lift, "snapshots": snapshots}
    if pickup_only:
        return {"status": "pickup_passed_goal_path_unavailable", "lift_m": lift_m,
                "bilateral_lift": bilateral_lift, "snapshots": snapshots,
                "insertion_check_passed": None}
    trial.move(plan["turn"], 100, "turn")
    trial.move(plan["above_goal"], 160, "transfer")
    snapshot("above_goal")
    trial.move(plan["assembly"], 120, "insert_or_nominal_place")
    trial.hold(20, "hold_goal")
    snapshot("at_goal")
    trial.gripper(candidate.opening_m)
    trial.hold(30, "release")
    trial.move(plan["retreat"], 80, "retreat")
    trial.hold(50, "settle")
    snapshot("released")
    final = trial.target_pose()
    translation_error = float(np.linalg.norm(final[:3, 3] - world_T_goal[:3, 3]))
    angle = float(np.linalg.norm(Rotation.from_matrix(
        final[:3, :3] @ world_T_goal[:3, :3].T).as_rotvec()))
    axis_tilt = float(np.rad2deg(np.arccos(np.clip(abs(final[2, 2]), 0, 1))))
    xy_error = float(np.linalg.norm(final[:2, 3] - world_T_goal[:2, 3]))
    speed = float(np.linalg.norm(trial.d.qvel[trial.target_dadr:trial.target_dadr + 3]))
    collision_ok = max(trial.audit[k] for k in ("robot_self", "robot_table", "robot_board")) < .001
    if "pin_guided_insertion" in goal_kind:
        board_top = .0289916
        tip_depth = board_top - (final[2, 3] - .025 * abs(final[2, 2]))
        insertion_ok = bool(tip_depth > .009 and final[2, 3] + .025 > board_top
            and xy_error < .001 and axis_tilt < 3 and speed < .01
            and collision_ok and trial.audit["target_board"] < .001)
        status = "insertion_passed" if insertion_ok else "insertion_failed"
    else:
        tip_depth = None
        insertion_ok = None
        status = "nominal_placement_only" if translation_error < .005 and collision_ok else "placement_failed"
    return {"status": status, "lift_m": lift_m, "bilateral_lift": bilateral_lift,
            "final_world_T_cad_audit": final.tolist(), "goal_error_m": translation_error,
            "goal_rotation_error_deg": float(np.rad2deg(angle)), "axis_tilt_deg": axis_tilt,
            "goal_xy_error_m": xy_error, "tip_depth_m": tip_depth, "final_speed_m_s": speed,
            "insertion_check_passed": insertion_ok, "snapshots": snapshots}


def run_part(name, out, goal_record, physics):
    started = time.perf_counter()
    part_dir = out / name
    trial = RobotTrial(SOURCE, part_dir, name)
    result = {"part": name, "input_pose_source": "simulator pose supplied as perfect alignment assumption",
              "classification_assumed_correct": True, "goal": goal_record,
              "other_loose_parts_considered": False, "planning_collision_scope": ["robot_self", "table"],
              "open_gripper_target_geometry_checked": True, "physics_requested": physics}
    try:
        world_T_object = trial.target_pose()
        world_T_goal = np.asarray(goal_record["world_T_cad_goal"])
        mesh_path = MESHES / f"{name}_m.stl"
        result["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        result["world_T_cad_input"] = world_T_object.tolist()
        mesh, _, _, pairs, sampling = sample_contacts(mesh_path, world_T_object, SAMPLES)
        result.update(sampling)
        center = world_T_object[:3, 3]
        candidates = candidate_frames(pairs, trial.pose()[:3, :3], center, MAX_CONTACT_PAIRS)
        result["generated_gripper_poses"] = len(candidates)
        candidates = table_filter(candidates, trial.m, trial.d)
        result["table_clear_candidates"] = len(candidates)
        # Check the same grasp at the known assembly pose before spending IK time.
        goal_candidates = [replace(candidate, world_T_grip=(world_T_goal
            @ np.linalg.inv(world_T_object) @ candidate.world_T_grip)) for candidate in candidates]
        cleared_ids = {id(candidate) for candidate in table_filter(goal_candidates, trial.m, trial.d)}
        assembly_clear = [candidate for candidate, goal_candidate in zip(candidates, goal_candidates)
                          if id(goal_candidate) in cleared_ids]
        result["table_clear_at_pick_and_goal"] = len(assembly_clear)
        rejected = []
        selected = None
        # A round cylinder's yaw is physically unobservable and does not alter
        # insertion. Search its explicit goal-equivalence class, not a tuned
        # wrist angle for a particular recorded case.
        yaw_options = range(0, 360, 30) if name.startswith("RGOCG") else (0,)
        for index, candidate in enumerate(assembly_clear[:MAX_PLANNING_CANDIDATES]):
            for yaw in yaw_options:
                equivalent_goal = world_T_goal.copy()
                equivalent_goal[:3, :3] = (world_T_goal[:3, :3]
                    @ Rotation.from_euler("z", yaw, degrees=True).as_matrix())
                plan = build_plan(trial, candidate, world_T_object, equivalent_goal)
                passed, reports = check_plan(trial, candidate, plan)
                if passed:
                    selected = candidate
                    result["selected_candidate_index"] = index
                    result["selected_goal_equivalent_yaw_deg"] = yaw
                    result["selected_world_T_cad_goal"] = equivalent_goal.tolist()
                    result["selected_grasp"] = candidate.record()
                    result["plan"] = {key: value.tolist() for key, value in plan.items()}
                    result["path_checks"] = reports
                    break
                rejected.append({"candidate_index": index, "goal_equivalent_yaw_deg": yaw,
                                 "checks": reports})
            if selected is not None:
                break
        result["planning_rejections"] = rejected
        pickup_only = False
        if selected is None:
            result["assembly_planning_status"] = "no_feasible_sampled_goal_path"
            pickup_only = True
            pickup_rejections = []
            for index, candidate in enumerate(candidates[:MAX_PLANNING_CANDIDATES]):
                plan = build_plan(trial, candidate, world_T_object, world_T_goal)
                passed, reports = check_plan(trial, candidate, plan, pickup_only=True)
                if passed:
                    selected = candidate
                    equivalent_goal = world_T_goal
                    result["selected_candidate_index"] = index
                    result["selected_grasp"] = candidate.record()
                    result["plan"] = {key: value.tolist() for key, value in plan.items()}
                    result["pickup_path_checks"] = reports
                    break
                pickup_rejections.append({"candidate_index": index, "checks": reports})
            result["pickup_planning_rejections"] = pickup_rejections
        else:
            result["assembly_planning_status"] = "sampled_kinematic_path_found"
        result["planning_seconds"] = time.perf_counter() - started
        if selected is None:
            result["status"] = "no_feasible_sampled_grasp_and_goal_path"
        elif physics:
            result.update(execute(trial, selected, plan, world_T_object, equivalent_goal,
                                  goal_record["goal_kind"], pickup_only=pickup_only))
        else:
            result["status"] = "kinematic_plan_only"
    except Exception as error:
        result["status"] = "motion_aborted" if len(trial.states) > 1 else "experiment_error"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        if len(trial.states) > 1:
            result["last_executed_phase"] = trial.states[-1]["phase"]
            result["max_observed_lift_m"] = float(max(state["target_pose_audit"][2][3]
                for state in trial.states) - result["world_T_cad_input"][2][3])
            result["final_world_T_cad_audit"] = trial.target_pose().tolist()
            result["post_release_check_completed"] = trial.states[-1]["phase"] == "settle"
            if not result["post_release_check_completed"]:
                result["insertion_check_passed"] = False
            trial.render(part_dir / "last_executed_frame.png")
        result["controller_audit"] = trial.save()
        trial.close()
        result["total_seconds"] = time.perf_counter() - started
        write_json(part_dir / "result.json", result)
    return result


def main():
    global SOURCE
    SOURCE = prepare_scene()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs/runtime_antipodal_assembly_20261023" / stamp
    out.mkdir(parents=True, exist_ok=False)
    requested = os.environ.get("RUN_PARTS")
    parts = tuple(requested.split(",")) if requested else PARTS
    physics = os.environ.get("EXECUTE_DYNAMICS", "1") == "1"
    goals = json.loads((HERE / "goal_audit.json").read_text())["nominal_simulation_goals"]
    manifest = {"source_scene": str(SOURCE), "source_scene_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "created_utc": stamp, "parts": parts, "physics": physics,
                "sampler_surface_rays": SAMPLES, "max_contact_pairs_expanded": MAX_CONTACT_PAIRS,
                "max_planning_candidates": MAX_PLANNING_CANDIDATES,
                "sampled_path_waypoints_per_segment": 10,
                "assumptions": ["perfect identity and initial CAD alignment", "nominal goals supplied",
                    "one dynamic target, other loose parts ignored", "finite geometric sampling",
                    "robot self/table planned; board contacts audited in dynamics"],
                "source_hashes": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(HERE.rglob("*.py"))},
                "goal_audit_sha256": hashlib.sha256((HERE / "goal_audit.json").read_bytes()).hexdigest()}
    write_json(out / "manifest.json", manifest)
    print(f"OUTPUT={out}", flush=True)
    results = []
    for name in parts:
        print(f"START {name}", flush=True)
        result = run_part(name, out, goals[name], physics)
        results.append(result)
        write_json(out / "results.json", results)
        print(json.dumps({key: result.get(key) for key in ("part", "status", "qualified_unique_pairs",
            "generated_gripper_poses", "table_clear_candidates", "planning_seconds", "total_seconds", "error")}), flush=True)
    with (out / "summary.csv").open("w", newline="") as stream:
        fields = ["part", "status", "qualified_unique_pairs", "generated_gripper_poses",
                  "table_clear_candidates", "table_clear_at_pick_and_goal", "planning_seconds", "total_seconds",
                  "lift_m", "goal_error_m", "goal_xy_error_m", "axis_tilt_deg", "tip_depth_m"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
