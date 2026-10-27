"""Observe CAD before/after pickup, then attempt the three nominal board goals.

Online code sees raw camera files and robot-only feedback from PlantClient.
The physical simulation's object state is inaccessible through that interface.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from .plant import PlantClient, ROOT, write
from .planning import PlanningModel
from .pose_estimation import protocol as pose_protocol
from .preflight import estimate
from .sensors import CAMERAS
from experiments.grconvnet_vlm_pickup_20261024.grconvnet_adapter import GrConvNetAdapter
from experiments.grconvnet_vlm_pickup_20261024.cad_refinement_20261025.refinement import refine_candidates, protocol as grasp_protocol
from experiments.runtime_antipodal_assembly_20261023.grasp_planning import linear_pose_path


HERE = Path(__file__).resolve().parent
BASE = ROOT/"outputs/observed_held_assembly_20261025"
MESHES = ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"
GOALS = ROOT/"experiments/runtime_antipodal_assembly_20261023/goal_audit.json"
PARTS = ("Gear_Large", "Gear_Medium", "M16_Hex_Nut")
LOWERING_HEIGHTS_M = (.06, .04, .03, .02, .018, .016, .014, .012, .010,
                      .008, .006, .004, .002, 0.)
MIN_PICKUP_RISE_M = .12
OBSERVED_STALL_M = .001
MAX_PART_IN_GRIP_CHANGE_M = .003
MAX_PART_IN_GRIP_ROTATION_CHANGE_DEG = 5.
FINAL_POSITION_TOLERANCE_M = .001
FINAL_AXIS_TOLERANCE_DEG = 2.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def change(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    return {"translation_m": float(np.linalg.norm(current[:3,3]-reference[:3,3])),
            "rotation_deg": float(np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(reference[:3,:3].T @ current[:3,:3]).as_rotvec())))}


def goal_error(pose, goal):
    pose, goal = np.asarray(pose), np.asarray(goal)
    return {"position_m": float(np.linalg.norm(pose[:3,3]-goal[:3,3])),
            "xy_m": float(np.linalg.norm(pose[:2,3]-goal[:2,3])),
            "height_m": float(pose[2,3]-goal[2,3]),
            "signed_CAD_Z_axis_deg": float(np.rad2deg(np.arccos(np.clip(
                pose[:3,2] @ goal[:3,2], -1., 1.)))),
            "raw_rotation_deg": change(goal,pose)["rotation_deg"]}


def gripper_for_part(part_pose, grip_T_cad):
    return np.asarray(part_pose) @ np.linalg.inv(np.asarray(grip_T_cad))


def check_segments(planner, segments, grip_T_cad=None):
    reports, seed = [], planner.d.qpos[planner.qa].copy()
    for name, first, last, opening, held, open_target in segments:
        qpath, report = planner.check_path(linear_pose_path(first,last,16), opening, seed=seed,
            target_relative=grip_T_cad if held else None, check_open_target=open_target)
        reports.append({"phase":name, **report})
        if qpath is None:
            return False, reports
        seed = qpath[-1]
    return True, reports


def run_part(part, run, plant, model, goal_record):
    folder = run/"controller"/part
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"part":part, "status":"not_started", "physical_pickup_attempts":0,
              "board_lowering_moves":0, "release_commanded":False,
              "object_state_read_online":False, "target_segmentation_used":False,
              "goal":goal_record, "events":[], "observed_estimates":[],
              "assembly_claim_scope":"Nominal CAD pose alignment only; gear meshing and nut threading unsupported"}
    feedback, planner, latest = None, None, None

    def save_result():
        result["wall_seconds"] = time.perf_counter()-started
        write(folder/"result.json",result)

    def command(action, **kwargs):
        nonlocal feedback
        response = plant.call(action, **kwargs)
        if "robot_feedback" in response:
            feedback = response["robot_feedback"]
            if planner is not None:
                planner.set_feedback(feedback)
        result["events"].append({"action":action, **kwargs,
                                 "robot_feedback":feedback})
        save_result()
        return response

    def observe(label, previous=None, pickup_feedback_path=None, static_after_release=False):
        nonlocal latest, feedback
        response = command("observe",observation_folder=part+"/"+label)
        observation_folder = Path(response["observation_folder"])
        destination = folder/"estimates"/label
        feedback_path = observation_folder/"robot_feedback.json"
        if previous is None:
            fitted = estimate("table",MESHES/(part+"_m.stl"),observation_folder,destination)
        else:
            previous_pose = np.asarray(previous["estimate"]["world_T_cad"])
            reference_feedback_path = (Path(pickup_feedback_path) if pickup_feedback_path is not None
                                       else Path(previous["robot_feedback_path"]))
            reference_feedback = json.loads(reference_feedback_path.read_text())
            if static_after_release:
                prior = previous_pose
                prior_source = "previous_observed_world_pose_after_release"
            else:
                prior = (np.asarray(feedback["world_T_grip"])
                         @ np.linalg.inv(np.asarray(reference_feedback["world_T_grip"])) @ previous_pose)
                prior_source = ("observed_initial_pose_transported_by_measured_robot_FK" if pickup_feedback_path is not None
                                else "previous_observed_pose_transported_by_measured_robot_FK")
            previous_file = Path(previous["estimate_path"])
            lineage = {"source_estimate_path":str(previous_file),
                       "source_estimate_sha256":digest(previous_file),
                       "reference_robot_feedback_path":str(reference_feedback_path),
                       "reference_robot_feedback_sha256":digest(reference_feedback_path),
                       "current_robot_feedback_path":str(feedback_path),
                       "current_robot_feedback_sha256":digest(feedback_path),
                       "prior_source":prior_source,
                       "part_stationary_during_approach_assumption":pickup_feedback_path is not None,
                       "formula":"prior = current_grip @ inverse(reference_grip) @ previous_observed_CAD" if not static_after_release
                                 else "prior = previous_observed_CAD; attachment no longer assumed"}
            fitted = estimate("held",MESHES/(part+"_m.stl"),observation_folder,destination,
                              prior=prior.tolist(),lineage=lineage,prior_source=prior_source)
        item = {"label":label,"observation_folder":str(observation_folder),
                "robot_feedback_path":str(feedback_path),"estimate_path":str(destination/"estimate.json"),
                "estimate":fitted}
        latest = item
        result["observed_estimates"].append({key:value for key,value in item.items() if key != "estimate"})
        result["latest_estimate_status"] = fitted["status"]
        if fitted["status"] == "available":
            cad_pose = np.asarray(fitted["world_T_cad"])
            if planner is not None:
                planner.set_estimated_pose(cad_pose)
            item["grip_T_cad"] = np.linalg.inv(np.asarray(feedback["world_T_grip"])) @ cad_pose
            result["latest_observed_world_T_cad"] = cad_pose.tolist()
            result["latest_observed_grip_T_cad"] = item["grip_T_cad"].tolist()
        save_result()
        print(json.dumps({"part":part,"stage":label,"fit_status":fitted["status"],
                          "fit_seconds":fitted.get("diagnostics",{}).get("seconds")}),flush=True)
        return item

    def available(item):
        return item["estimate"]["status"] == "available"

    def move_checked(target, opening, phase, frames, held_relation=None, check_open_target=False):
        if held_relation is not None:
            opening = min(.08, abs(float(feedback["finger_position_m"][0])
                                   -float(feedback["finger_position_m"][1])))
        first = planner.pose().copy()
        passed, reports = check_segments(planner,[(phase,first,target,opening,
                                                  held_relation is not None,check_open_target)],held_relation)
        result.setdefault("movement_checks",[]).append({"phase":phase,"passed":passed,"reports":reports,
            "world_T_grip_requested":np.asarray(target).tolist(),
            "observed_grip_T_cad_used":None if held_relation is None else held_relation.tolist()})
        save_result()
        if not passed:
            result["status"] = "robot_path_rejected"
            result["stopping_phase"] = phase
            return False
        command("move",world_T_grip=np.asarray(target).tolist(),frames=frames,phase=phase)
        return True

    try:
        reset = command("reset",part=part)
        initial = observe("initial")
        if not available(initial):
            result["status"] = "initial_CAD_pose_unavailable"
            return result
        initial_pose = np.asarray(initial["estimate"]["world_T_cad"])
        result["initial_observed_world_T_cad"] = initial_pose.tolist()
        planner = PlanningModel(reset["planning_model"],feedback,MESHES/(part+"_m.stl"),initial_pose)
        result["planning_model_provenance"] = planner.provenance
        camera_folder = Path(initial["observation_folder"])
        observation = {"rgb":np.asarray(Image.open(camera_folder/"overhead_rgb.png")),
            "depth_m":np.load(camera_folder/"overhead_depth_m.npy",allow_pickle=False),
            "target_mask":np.load(initial["estimate"]["observed_mask_path"],allow_pickle=False),
            "calibration":json.loads((camera_folder/"overhead_calibration.json").read_text())}
        prediction = model.predict(observation["rgb"],observation["depth_m"],observation["target_mask"],4)
        pool, records, rejected, diagnostics = refine_candidates(prediction,observation,planner,
                                                        MESHES/(part+"_m.stl"),initial_pose)
        result.update(raw_network_proposals=prediction["proposals"],grasp_candidates=records,
                      grasp_mapping_rejections=rejected,cad_refinement=diagnostics,
                      grasp_mask_source="Observed depth connected component, not renderer segmentation")
        selected = None
        for index,candidate in enumerate(pool):
            grasp = candidate.world_T_grip.copy()
            pregrasp,lift = grasp.copy(),grasp.copy()
            pregrasp[:3,3] -= .10*grasp[:3,2]
            lift[:3,3] += [0.,0.,.18]
            relative = np.linalg.inv(grasp) @ initial_pose
            passed,checks = check_segments(planner,[
                ("pickup_approach",planner.pose(),pregrasp,candidate.opening_m,False,True),
                ("pickup_descend",pregrasp,grasp,candidate.opening_m,False,True),
                ("pickup_lift",grasp,lift,candidate.width_m,True,False)],relative)
            result.setdefault("pickup_candidate_checks",[]).append({"pool_index":index,"passed":passed,"checks":checks})
            if passed:
                selected = candidate
                result["selected_grasp"] = candidate.record()
                break
        if selected is None:
            result["status"] = "no_observation_based_pickup_path"
            return result
        command("gripper",opening_m=selected.opening_m)
        command("move",world_T_grip=pregrasp.tolist(),frames=100,phase="pickup_approach")
        command("move",world_T_grip=grasp.tolist(),frames=65,phase="pickup_descend")
        pickup_feedback = folder/"measured_pickup_robot_feedback.json"
        write(pickup_feedback,feedback)
        result["physical_pickup_attempts"] = 1
        command("gripper",opening_m=0.)
        command("hold",frames=35,phase="close")
        command("move",world_T_grip=lift.tolist(),frames=100,phase="pickup_lift")
        command("hold",frames=25,phase="hold_lift")
        held = observe("after_pickup",initial,pickup_feedback_path=pickup_feedback)
        if not available(held):
            result["status"] = "held_CAD_pose_unavailable"
            return result
        observed_rise = float(np.asarray(held["estimate"]["world_T_cad"])[2,3]-initial_pose[2,3])
        result["observed_pickup_rise_m"] = observed_rise
        if observed_rise < MIN_PICKUP_RISE_M:
            result["status"] = "observed_pickup_rise_too_small"
            return result
        result["observed_pickup_passed"] = True
        goal = np.asarray(goal_record["world_T_cad_goal"])
        turn = planner.pose().copy()
        turn[:3,:3] = gripper_for_part(goal,held["grip_T_cad"])[:3,:3]
        if not move_checked(turn,selected.width_m,"turn_for_nominal_goal",100,held["grip_T_cad"]):
            return result
        turned = observe("after_turn",held)
        if not available(turned):
            result["status"] = "pose_unavailable_after_turn"
            return result
        above = goal.copy()
        above[2,3] += .16
        above_grip = gripper_for_part(above,turned["grip_T_cad"])
        if not move_checked(above_grip,selected.width_m,"transfer_above_board",160,turned["grip_T_cad"]):
            return result
        current = observe("above_board",turned)
        if not available(current):
            result["status"] = "pose_unavailable_above_board"
            return result
        for index,height in enumerate(LOWERING_HEIGHTS_M):
            previous = current
            previous_pose = np.asarray(previous["estimate"]["world_T_cad"])
            desired_part = goal.copy()
            desired_part[2,3] += height
            target = gripper_for_part(desired_part,previous["grip_T_cad"])
            if not move_checked(target,selected.width_m,f"board_lower_{index:02d}",55,
                                previous["grip_T_cad"]):
                return result
            result["board_lowering_moves"] += 1
            command("hold",frames=10,phase="hold_for_observation")
            current = observe(f"board_step_{index:02d}",previous)
            if not available(current):
                result["status"] = "pose_unavailable_during_board_lowering"
                return result
            current_pose = np.asarray(current["estimate"]["world_T_cad"])
            measured = goal_error(current_pose,goal)
            relation_change = change(previous["grip_T_cad"],current["grip_T_cad"])
            expected_drop = float(previous_pose[2,3]-desired_part[2,3])
            observed_drop = float(previous_pose[2,3]-current_pose[2,3])
            check = {"step":index,"commanded_CAD_height_above_goal_m":height,
                     "observed_goal_error":measured,"observed_grip_relation_change":relation_change,
                     "expected_drop_m":expected_drop,"observed_drop_m":observed_drop}
            result.setdefault("board_observation_checks",[]).append(check)
            if (relation_change["translation_m"] > MAX_PART_IN_GRIP_CHANGE_M
                    or relation_change["rotation_deg"] > MAX_PART_IN_GRIP_ROTATION_CHANGE_DEG):
                result["status"] = "observed_part_shift_during_lowering"
                return result
            if (expected_drop > .0015
                    and expected_drop-observed_drop > max(OBSERVED_STALL_M, .5*expected_drop)):
                result["status"] = "observed_lowering_stalled"
                return result
        last_error = goal_error(current["estimate"]["world_T_cad"],goal)
        result["observed_goal_error_before_release"] = last_error
        if last_error["position_m"] > FINAL_POSITION_TOLERANCE_M or last_error["signed_CAD_Z_axis_deg"] > FINAL_AXIS_TOLERANCE_DEG:
            result["status"] = "observed_nominal_pose_outside_tolerance"
            return result
        command("gripper",opening_m=selected.opening_m)
        result["release_commanded"] = True
        command("hold",frames=25,phase="release")
        released = observe("after_release",current,static_after_release=True)
        if not available(released):
            result["status"] = "released_part_pose_unavailable"
            return result
        retreat = planner.pose().copy()
        retreat[2,3] += .08
        if not move_checked(retreat,selected.opening_m,"retreat",80):
            return result
        command("hold",frames=25,phase="settle")
        final = observe("after_retreat",released,static_after_release=True)
        if not available(final):
            result["status"] = "final_part_pose_unavailable"
            return result
        error = goal_error(final["estimate"]["world_T_cad"],goal)
        result["observed_final_goal_error"] = error
        result["status"] = ("observed_nominal_pose_reached_no_engagement_claim" if
            error["position_m"] <= FINAL_POSITION_TOLERANCE_M and error["signed_CAD_Z_axis_deg"] <= FINAL_AXIS_TOLERANCE_DEG
            else "observed_part_moved_after_release")
    except Exception as error:
        result.update(status="controller_or_estimation_error",error=str(error),traceback=traceback.format_exc())
    finally:
        save_result()
        command("finish")
    return result


def main():
    torch.set_num_threads(4)
    output = BASE/"runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True,exist_ok=False)
    print("OUTPUT="+str(output),flush=True)
    goals = json.loads(GOALS.read_text())["nominal_simulation_goals"]
    model = GrConvNetAdapter(device="cpu")
    manifest = {"parts":PARTS,"created_utc":datetime.now(timezone.utc).isoformat(),
        "source_files_sha256":{str(path.relative_to(ROOT)):digest(path) for path in
            (HERE/name for name in ("run_trial.py","plant.py","planning.py","sensors.py",
                                   "pose_estimation.py","estimate_from_files.py","preflight.py"))},
        "pose_protocol":pose_protocol(),"grasp_protocol":grasp_protocol(),
        "goals_file_sha256":digest(GOALS),"nominal_goals":{part:goals[part] for part in PARTS},
        "camera_constants":CAMERAS,"GR_model_provenance":model.provenance,
        "GR_model_load_seconds_once":model.model_setup_seconds,
        "controller_protocol":{"lowering_heights_above_goal_m":LOWERING_HEIGHTS_M,
            "minimum_observed_pickup_rise_m":MIN_PICKUP_RISE_M,
            "observed_stall_margin_m":OBSERVED_STALL_M,
            "maximum_incremental_observed_part_in_grip_change_m":MAX_PART_IN_GRIP_CHANGE_M,
            "maximum_incremental_observed_part_in_grip_rotation_change_deg":MAX_PART_IN_GRIP_ROTATION_CHANGE_DEG,
            "nominal_final_position_tolerance_m":FINAL_POSITION_TOLERANCE_M,
            "nominal_final_signed_CAD_Z_axis_tolerance_deg":FINAL_AXIS_TOLERANCE_DEG,
            "yaw_tooth_thread_phase_verified":False},
        "boundary":{"online_inputs":"RGB-D, camera calibration, robot encoders/FK, native CAD and known nominal static board goals",
            "simulator_object_state_used_for_estimation_or_control":False,
            "renderer_target_segmentation_or_truth_crops_used":False,
            "robot_static_planner_plus_estimated_CAD_only":True,
            "plant_private_truth_read_only_after_online_estimates_and_commands_frozen":True},
        "scope":"Stepwise visual updates between motion segments in rendered RGB-D simulation; no physical hardware or continuously running real-time tracking; no gear-meshing/nut-threading model"}
    write(output/"manifest.json",manifest)
    server_log = (output/"plant_server.log").open("w")
    process = subprocess.Popen([sys.executable,"-m","experiments.observed_held_assembly_20261025.plant",
                                "--serve",str(output)],stdout=server_log,stderr=subprocess.STDOUT)
    plant = PlantClient(output)
    results = []
    try:
        for part in PARTS:
            print("START "+part,flush=True)
            result = run_part(part,output,plant,model,goals[part])
            results.append(result)
            write(output/"results.json",results)
            print(json.dumps({key:result.get(key) for key in ("part","status","physical_pickup_attempts",
                "observed_pickup_rise_m","board_lowering_moves","release_commanded","wall_seconds","error")}),flush=True)
    finally:
        try:
            plant.call("shutdown")
        finally:
            process.wait(timeout=30)
            server_log.close()
    print("COMPLETE="+str(output),flush=True)


if __name__ == "__main__":
    main()
