"""Independent evidence/leakage checks for observation-only held CAD updates."""
import ast
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from experiments.grconvnet_vlm_pickup_20261024.cad_refinement_20261025.audit import (
    digest, grip_T_cad, inverse_pose, multiply, pose_change, pose_valid,
    quaternion_xyzw, read,
)
from experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.audit import contact_sides


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/"outputs/observed_held_assembly_20261025"
OLD_CAD_BASE = ROOT/"outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025"
OLD_CAD_RUN = OLD_CAD_BASE/"runs/20261025T192913Z"
PARTS = ("Gear_Large","Gear_Medium","M16_Hex_Nut")
CALIBRATION_FIELDS = {"camera_name","width","height","intrinsics","world_T_camera",
    "camera_axes","depth_units","world_units","table_z_world_m","calibration_source"}
OBSERVATION_FIELDS = {"rgb_path","depth_m_path","robot_depth_m_path","calibration_path"}
FEEDBACK_FIELDS = {"joint_names","joint_position_rad","joint_velocity_rad_s",
    "finger_joint_names","finger_position_m","world_T_grip"}


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def snapshot_protected():
    """Protect both earlier pickup studies before the observation study runs."""
    destination = BASE/"protected_before_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    values = read(OLD_CAD_BASE/"protected_before_audit.json")
    paths = [path for path in OLD_CAD_RUN.rglob("*") if path.is_file()]
    paths += [path for path in (ROOT/"experiments/grconvnet_vlm_pickup_20261024/cad_refinement_20261025").rglob("*")
              if path.is_file() and "__pycache__" not in path.parts]
    paths += [OLD_CAD_BASE/name for name in ("static_preflight.json","vlm_shared_warmup.json")]
    paths += [ROOT/"experiments/runtime_antipodal_assembly_20261023/goal_audit.json"]
    paths += [ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"/(part+"_m.stl") for part in PARTS]
    for path in paths:
        if path.is_file():
            values[path.relative_to(ROOT).as_posix()] = digest(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    save(destination,values)
    return {"snapshot":str(destination),"protected_files":len(values)}


def protected_changes():
    values = read(BASE/"protected_before_audit.json")
    changed = [name for name,sha in values.items() if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    return {"passed":not changed,"file_count":len(values),"changed":changed}


def local_path(recorded):
    """Resolve this repository's recorded Windows or WSL paths portably."""
    text = str(recorded).replace("\\","/")
    marker = "/"+ROOT.name+"/"
    if marker in text:
        path = ROOT/text.split(marker,1)[1]
    elif text.startswith(("experiments/","outputs/","scripts/","docs/")):
        path = ROOT/text
    else:
        path = Path(recorded)
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Recorded input is outside the declared workspace: "+text)
    return path


def close_matrix(a,b,tolerance=1e-10):
    return len(a) == len(b) and all(len(x) == len(y) for x,y in zip(a,b)) and max(
        abs(x-y) for row1,row2 in zip(a,b) for x,y in zip(row1,row2)) <= tolerance


def source_boundary_checks():
    """Supplement the manual source review with focused AST assertions."""
    banned_attributes = {"target_pose","target_bid","target_qadr","target_dadr",
        "target_pose_audit","enable_segmentation_rendering","mj_ray"}
    rows = []
    for name in ("pose_estimation.py","estimate_from_files.py"):
        path = HERE/name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        hits = sorted({node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute) and node.attr in banned_attributes})
        dynamic = sorted({node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call)
                          and isinstance(node.func,ast.Name) and node.func.id in {"eval","exec","__import__"}})
        checks = {"no_simulator_import":not any(value and (value.startswith("mujoco") or "robot_trial" in value or "plant" in value) for value in imports),
            "no_target_state_or_engine_mask_calls":not hits,"no_dynamic_code_execution":not dynamic}
        rows.append({"source":str(path.relative_to(ROOT)),"sha256":digest(path),"checks":checks,"passed":all(checks.values())})
    sensor = ast.parse((HERE/"sensors.py").read_text())
    sensor_hits = sorted({node.attr for node in ast.walk(sensor) if isinstance(node,ast.Attribute) and node.attr in banned_attributes})
    rows.append({"source":str((HERE/"sensors.py").relative_to(ROOT)),"sha256":digest(HERE/"sensors.py"),
                 "checks":{"no_target_pose_or_segmentation_calls":not sensor_hits},"passed":not sensor_hits})
    return {"passed":all(row["passed"] for row in rows),"sources":rows,
        "scope":"Focused static assertions supplement the manual source and exact runtime-input review; they are not an operating-system sandbox proof."}


def feedback_checks(feedback):
    return {"only_robot_feedback_fields":set(feedback) == FEEDBACK_FIELDS,
        "seven_named_arm_encoders":feedback["joint_names"] == [f"robot0_joint{i}" for i in range(1,8)] and len(feedback["joint_position_rad"]) == len(feedback["joint_velocity_rad_s"]) == 7,
        "two_named_finger_encoders":feedback["finger_joint_names"] == [f"gripper0_finger_joint{i}" for i in (1,2)] and len(feedback["finger_position_m"]) == 2,
        "robot_encoder_values_finite":all(math.isfinite(value) for key in ("joint_position_rad","joint_velocity_rad_s","finger_position_m") for value in feedback[key]),
        "grip_site_FK_rigid":pose_valid(feedback["world_T_grip"])}


def audit_rpc(run):
    run = Path(run).resolve()
    responses = []
    allowed = {"reset":{"ok","robot_feedback","planning_model"},
        "observe":{"ok","observation_folder","robot_feedback"},
        "move":{"ok","robot_feedback"},"hold":{"ok","robot_feedback"},
        "gripper":{"ok","robot_feedback"},"feedback":{"ok","robot_feedback"},
        "finish":{"ok"},"shutdown":{"ok"}}
    for request_path in sorted((run/"plant_queue").glob("*.request.json")):
        request = read(request_path)
        response_path = request_path.with_name(request_path.name.replace(".request.json",".response.json"))
        response = read(response_path)
        action = request["action"]
        checks = {"supported_sensor_or_robot_action":action in allowed,
            "response_fields_allowlisted":set(response) == (allowed[action] if response["ok"] else {"ok","error"})}
        if "robot_feedback" in response:
            checks.update(feedback_checks(response["robot_feedback"]))
        if "observation_folder" in response:
            checks["only_observation_inbox_path_returned"] = local_path(response["observation_folder"]).is_relative_to(run/"observations")
        if "planning_model" in response:
            checks["only_known_static_planning_path_returned"] = local_path(response["planning_model"]).is_relative_to(run/"known_static_model")
        responses.append({"action":action,"request":str(request_path.relative_to(run)),
            "response":str(response_path.relative_to(run)),"checks":checks,"passed":all(checks.values())})
    return {"passed":bool(responses) and all(row["passed"] for row in responses),
        "request_count":len(responses),"responses":responses,
        "scope":"Every returned online payload is checked; private target poses, force labels, full scene qpos and execution success flags are excluded."}


def audit_robot_template(path):
    path = Path(path)
    root = ET.parse(path).getroot()
    joints = [joint for joint in root.findall(".//joint") if joint.get("name")]
    checks = {"no_saved_keyframe":root.find("keyframe") is None,
        "no_original_product_body_or_metadata":"nist_part_" not in path.read_text(),
        "only_robot_movable_joints":all(joint.get("name").startswith(("robot0_","gripper0_")) for joint in joints),
        "no_product_freejoint":not root.findall(".//freejoint") and not any(joint.get("type") == "free" for joint in joints),
        "all_referenced_meshes_declared":{geom.get("mesh") for geom in root.findall(".//geom") if geom.get("mesh")} <= {mesh.get("name") for mesh in root.findall("asset/mesh")}}
    return {"path":str(path),"sha256":digest(path),"checks":checks,"passed":all(checks.values())}


def packet_raw_sensor_folder(packet_path):
    """Resolve preprocessing lineage without treating filtered depth as raw."""
    packet_path = Path(packet_path)
    preprocessing = packet_path.parent/"preprocessing.json"
    if preprocessing.is_file():
        return local_path(read(preprocessing)["raw_observation_folder"])
    return local_path(read(packet_path)["observations"][0]["depth_m_path"]).parent


def audit_prior_lineage(packet_path,run):
    packet_path,run = Path(packet_path).resolve(),Path(run).resolve()
    packet = read(packet_path)
    lineage = read(packet_path.parent/"prior_lineage.json")
    paths = {key:local_path(lineage[key+"_path"]) for key in
             ("source_estimate","reference_robot_feedback","current_robot_feedback")}
    source = read(paths["source_estimate"])
    reference,current = read(paths["reference_robot_feedback"]),read(paths["current_robot_feedback"])
    kind = packet["prior_source"]
    kinds = {"observed_initial_pose_transported_by_measured_robot_FK",
             "previous_observed_pose_transported_by_measured_robot_FK",
             "previous_observed_world_pose_after_release"}
    checks = {"declared_permitted_observation_prior":kind in kinds and kind == lineage["prior_source"],
        "source_is_frozen_available_observed_estimate":source["status"] == "available" and paths["source_estimate"].is_relative_to(run/"controller"),
        "lineage_files_unchanged":all(digest(path) == lineage[key+"_sha256"] for key,path in paths.items()),
        "no_private_truth_lineage":all(not path.is_relative_to(run/"plant_private") for path in paths.values()),
        "source_frozen_before_new_packet":paths["source_estimate"].stat().st_mtime_ns <= packet_path.stat().st_mtime_ns,
        "current_feedback_matches_observation":paths["current_robot_feedback"] == packet_raw_sensor_folder(packet_path)/"robot_feedback.json"}
    checks.update({"reference_"+key:value for key,value in feedback_checks(reference).items()})
    checks.update({"current_"+key:value for key,value in feedback_checks(current).items()})
    if kind == "previous_observed_world_pose_after_release":
        prior = source["world_T_cad"]
        checks["no_attachment_assumption_after_release"] = lineage["part_stationary_during_approach_assumption"] is False
    else:
        prior = multiply(multiply(current["world_T_grip"],inverse_pose(reference["world_T_grip"])),source["world_T_cad"])
        if kind == "observed_initial_pose_transported_by_measured_robot_FK":
            checks["initial_reference_is_observed_table_pose"] = source["provenance"]["mode"] == "table" and lineage["part_stationary_during_approach_assumption"] is True
            checks["reference_is_saved_pickup_encoder_measurement"] = paths["reference_robot_feedback"].name == "measured_pickup_robot_feedback.json"
        else:
            source_packet = read(paths["source_estimate"].parent/"packet.json")
            expected = packet_raw_sensor_folder(paths["source_estimate"].parent/"packet.json")/"robot_feedback.json"
            checks["previous_reference_feedback_is_from_same_observation"] = paths["reference_robot_feedback"] == expected and lineage["part_stationary_during_approach_assumption"] is False
    checks["prior_matrix_reconstructs_without_environment_pose"] = close_matrix(prior,packet["prior_world_T_cad"])
    return {"source":kind,"checks":checks,"passed":all(checks.values())}


def audit_estimate(packet_path,run,permitted_sensor_directory=None):
    packet_path,run = Path(packet_path).resolve(),Path(run).resolve()
    packet = read(packet_path)
    estimate_path = packet_path.parent/"estimate.json"
    estimate = read(estimate_path)
    mode = packet["mode"]
    provenance = estimate["provenance"]
    expected = {packet_path}
    cad = local_path(packet["cad_path"])
    expected.add(cad)
    checks = {"packet_fields_allowlisted":set(packet) <= {"mode","cad_path","observations","prior_world_T_cad","prior_source"},
        "mode_declared":mode in {"table","held"},
        "native_supplied_CAD":cad in {ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"/(part+"_m.stl") for part in PARTS},
        "table_has_no_pose_prior":mode != "table" or not {"prior_world_T_cad","prior_source"}.intersection(packet),
        "no_training_or_model_weights":provenance["model_training_or_weights_used"] is False,
        "provenance_declares_no_scene_truth_reads":provenance["simulator_scene_trial_trajectory_result_files_read"] is False,
        "provenance_mode_matches":provenance["mode"] == mode}
    requested = next((part for part in PARTS if part in packet_path.parts),None)
    checks["CAD_identity_matches_requested_part"] = requested is not None and cad.name == requested+"_m.stl"
    calibrations = []
    for index,entry in enumerate(packet["observations"]):
        checks[f"camera_{index}_only_sensor_file_fields"] = set(entry) == OBSERVATION_FIELDS
        paths = {key:local_path(value) for key,value in entry.items()}
        expected.update(paths.values())
        sensor_directory = run/"observations" if permitted_sensor_directory is None else Path(permitted_sensor_directory).resolve()
        checks[f"camera_{index}_files_in_observation_inbox"] = all(path.is_relative_to(sensor_directory) for path in paths.values())
        cal = read(paths["calibration_path"])
        checks[f"camera_{index}_ordinary_calibration_only"] = set(cal) <= CALIBRATION_FIELDS and {"intrinsics","world_T_camera","width","height"} <= set(cal)
        checks[f"camera_{index}_rigid_transform"] = pose_valid(cal["world_T_camera"])
        K = cal["intrinsics"]
        checks[f"camera_{index}_pinhole_parameters"] = len(K) == 3 and all(len(row) == 3 for row in K) and all(math.isfinite(x) for row in K for x in row) and K[0][0] > 0 and K[1][1] > 0 and K[2] == [0.,0.,1.]
        checks[f"camera_{index}_metric_optical_depth"] = cal["depth_units"] == "metres; optical camera Z"
        calibrations.append(cal)
    actual = {local_path(name):sha for name,sha in provenance["input_files_sha256"].items()}
    checks["exact_input_file_allowlist_reproduces"] = set(actual) == expected
    checks["all_input_hashes_unchanged"] = all(path.is_file() and digest(path) == sha for path,sha in actual.items())
    source_hashes = {local_path(name):sha for name,sha in provenance["source_files_sha256"].items()}
    checks["fitting_sources_unchanged"] = all(digest(path) == sha for path,sha in source_hashes.items())
    diagnostics = estimate["diagnostics"]
    checks["absolute_symmetric_orientation_not_claimed"] = diagnostics["absolute_symmetric_orientation_resolved"] is False
    checks["available_pose_is_rigid"] = estimate["status"] != "available" or pose_valid(estimate["world_T_cad"])
    checks["unavailable_pose_not_fabricated"] = estimate["status"] == "available" or estimate["world_T_cad"] is None
    if mode == "table":
        bank = diagnostics["candidate_components"]
        checks["no_initial_object_pose_seed"] = diagnostics["prior_object_pose_used"] is False
        checks["component_mask_has_observation_provenance"] = diagnostics["target_mask_source"] == "Connected component of observed depth above calibrated plane"
        checks["table_candidates_rank_by_frozen_score"] = bank == sorted(bank,key=lambda row:(-row["score"],row["component_id"]))
        available = bool(bank) and bank[0]["score"] >= .55 and not (len(bank) > 1 and bank[1]["score"] >= .55 and bank[0]["score"]-bank[1]["score"] < .02)
        checks["table_availability_gate_reproduces"] = available == (estimate["status"] == "available")
        checks["all_tested_table_poses_preserved"] = all(row["hypothesis_count"] == len(row["pose_hypotheses"]) for row in bank)
        if available:
            best = bank[0]
            checks["selected_component_and_pose_reproduce"] = diagnostics["selected_component_id"] == best["component_id"] and estimate["world_T_cad"] == best["world_T_cad"]
            composed = multiply(calibrations[0]["world_T_camera"],best["best_fit"]["T_camera_from_cad"])
            checks["initial_camera_to_world_composition_reproduces"] = close_matrix(composed,estimate["world_T_cad"])
    else:
        lineage = audit_prior_lineage(packet_path,run)
        checks.update({"lineage_"+key:value for key,value in lineage["checks"].items()})
        checks["held_prior_rigid"] = pose_valid(packet["prior_world_T_cad"])
        checks["held_prior_retained_exactly"] = packet["prior_world_T_cad"] == diagnostics["predicted_world_T_cad"]
        checks["current_pose_and_engine_masks_excluded"] = diagnostics["current_object_pose_or_simulator_segmentation_used"] is False
        bank = diagnostics.get("local_pose_candidates",[])
        if bank:
            checks["five_fixed_local_yaw_starts"] = sorted(row["initial_local_yaw_offset_deg"] for row in bank) == [-10.,-5.,0.,5.,10.]
            checks["held_candidates_rank_by_frozen_loss"] = bank == sorted(bank,key=lambda row:(row["clipped_surface_loss_m2"],abs(row["initial_local_yaw_offset_deg"])))
            for index,row in enumerate(bank):
                history = row["iterations"]
                checks[f"local_start_{index}_bounded_ICP"] = all(item["gate_m"] in (.012,.006,.003) and 0 <= item["iteration"] < 15 and item["inlier_points"] >= 30 and item["step_fraction"] in (1.,.5,.25,.125) for item in history)
                checks[f"local_start_{index}_loss_never_increases"] = all(b["loss_m2"] <= a["loss_m2"]+1e-14 for a,b in zip(history,history[1:]))
                change = pose_change(packet["prior_world_T_cad"],row["world_T_cad"])
                checks[f"local_start_{index}_pose_stays_within_prior_bounds"] = change["cad_origin_displacement_m"] <= .020+1e-9 and change["rotation_deg"] <= 20.+1e-6
            forward = diagnostics["forward_depth_check"]
            total = sum(row["assessable_CAD_pixels"] for row in forward["cameras"])
            agreed = sum(row["depth_agreeing_CAD_pixels"] for row in forward["cameras"])
            fraction = agreed/total if total else 0.
            checks["forward_depth_counts_and_gate_reproduce"] = total == forward["assessable_CAD_pixels"] and abs(fraction-forward["fraction_within_2mm"]) < 1e-12 and forward["passed"] == (total >= 30 and fraction >= .5)
            available = bank[0]["inlier_fraction_within_1mm"] >= .5 and forward["passed"]
            checks["held_availability_gate_reproduces"] = available == (estimate["status"] == "available")
            checks["selected_held_pose_reproduces"] = not available or estimate["world_T_cad"] == bank[0]["world_T_cad"]
            values = diagnostics["local_observability_singular_values"]
            checks["local_observability_diagnostic_reproduces"] = diagnostics["local_six_parameter_support"] == (len(values) == 6 and values[-1] > values[0]*1e-4)
    return {"packet":str(packet_path.relative_to(run)),"estimate":str(estimate_path.relative_to(run)),
        "mode":mode,"status":estimate["status"],"input_file_count":len(actual),
        "checks":checks,"passed":all(checks.values())}


def goal_error(pose,goal):
    position = math.dist([row[3] for row in pose[:3]],[row[3] for row in goal[:3]])
    xy = math.dist([pose[i][3] for i in range(2)],[goal[i][3] for i in range(2)])
    cosine = sum(pose[i][2]*goal[i][2] for i in range(3))
    return {"position_m":position,"xy_m":xy,"height_m":pose[2][3]-goal[2][3],
        "signed_CAD_Z_axis_deg":math.degrees(math.acos(max(-1.,min(1.,cosine)))),
        "raw_rotation_deg":pose_change(goal,pose)["rotation_deg"]}


def audit_controller(run,part,manifest):
    folder = run/"controller"/part
    result = read(folder/"result.json")
    goal = manifest["nominal_goals"][part]["world_T_cad_goal"]
    events = result["events"]
    estimates = {row["label"]:read(local_path(row["estimate_path"])) for row in result["observed_estimates"]}
    feedback = {row["label"]:read(local_path(row["robot_feedback_path"])) for row in result["observed_estimates"]}
    checks = {"correct_part":result["part"] == part,"no_online_object_state_or_engine_mask_declared":result["object_state_read_online"] is False and result["target_segmentation_used"] is False,
        "approved_nominal_goal_exact":result["goal"] == manifest["nominal_goals"][part],
        "at_most_one_pickup_attempt":result["physical_pickup_attempts"] in (0,1),
        "finish_command_recorded":events[-1]["action"] == "finish",
        "terminal_status_recorded":result["status"] not in {"not_started","not_started_yet"}}
    moves = [event for event in events if event["action"] == "move"]
    checks["lowering_move_count_reproduces"] = result["board_lowering_moves"] == sum(event["phase"].startswith("board_lower_") for event in moves)
    checks["release_flag_has_release_hold"] = result["release_commanded"] == any(event["action"] == "hold" and event["phase"] == "release" for event in events)
    if "initial_observed_world_T_cad" in result:
        checks["initial_pose_is_observed_worker_output"] = result["initial_observed_world_T_cad"] == estimates["initial"]["world_T_cad"]
        checks["grasp_mask_observation_derived"] = result["grasp_mask_source"] == "Observed depth connected component, not renderer segmentation"
    if "observed_pickup_rise_m" in result:
        rise = estimates["after_pickup"]["world_T_cad"][2][3]-estimates["initial"]["world_T_cad"][2][3]
        checks["observed_pickup_rise_reproduces"] = abs(rise-result["observed_pickup_rise_m"]) < 1e-10
        checks["pickup_decision_uses_observed_rise_only"] = result.get("observed_pickup_passed",False) == (rise >= .12)
    if result.get("physical_pickup_attempts"):
        pickup_feedback = read(folder/"measured_pickup_robot_feedback.json")
        descend = next(event for event in reversed(events) if event["action"] == "move" and event["phase"] == "pickup_descend")
        checks["pickup_prior_FK_is_measured_after_descent"] = pickup_feedback == descend["robot_feedback"]
    observation_sequence = [row["label"] for row in result["observed_estimates"]]
    observed_events = [Path(event["observation_folder"]).name for event in events if event["action"] == "observe"]
    checks["every_completed_observation_preserved_in_order"] = observation_sequence == observed_events[:len(observation_sequence)]
    latest_label,used = None,[]
    movement = {row["phase"]:row for row in result.get("movement_checks",[])}
    for event in events:
        if event["action"] == "observe":
            label = Path(event["observation_folder"]).name
            if label in estimates and estimates[label]["status"] == "available":
                latest_label = label
        if event["action"] != "move" or event["phase"] not in movement:
            continue
        record = movement[event["phase"]]
        label = latest_label
        if record["observed_grip_T_cad_used"] is not None:
            relative = grip_T_cad(feedback[label]["world_T_grip"],estimates[label]["world_T_cad"])
            checks["move_"+event["phase"]+"_uses_latest_observed_grip_relation"] = close_matrix(relative,record["observed_grip_T_cad_used"])
            staged_route = result.get("staged_route_search")
            if staged_route and (event["phase"] == "turn_for_nominal_goal" or event["phase"].startswith("route_")):
                chosen = staged_route["candidates"][staged_route["selected_route_index"]]
                planned = next(row for row in chosen["prefix"] if row["phase"] == event["phase"])
                correct = close_matrix(planned["world_T_grip"],event["world_T_grip"])
            elif event["phase"] == "turn_for_nominal_goal":
                expected = multiply(goal,inverse_pose(relative))
                correct = all(abs(expected[i][j]-event["world_T_grip"][i][j]) < 1e-10 for i in range(3) for j in range(3))
                correct &= all(abs(event["world_T_grip"][i][3]-feedback[label]["world_T_grip"][i][3]) < 1e-10 for i in range(3))
            else:
                desired = [row.copy() for row in goal]
                if event["phase"] == "transfer_above_board":
                    desired[2][3] += .16
                elif event["phase"].startswith("board_lower_"):
                    index = int(event["phase"].split("_")[-1])
                    desired[2][3] += manifest["controller_protocol"]["lowering_heights_above_goal_m"][index]
                expected = multiply(desired,inverse_pose(relative))
                correct = close_matrix(expected,event["world_T_grip"])
            checks["move_"+event["phase"]+"_goal_transform_reproduces"] = correct
            used.append({"phase":event["phase"],"estimate_label":label,"grip_T_cad":relative})
        checks["move_"+event["phase"]+"_passed_before_execution"] = record["passed"] is True and close_matrix(record["world_T_grip_requested"],event["world_T_grip"])
    board_checks = []
    previous_label = "above_board"
    for row in result.get("board_observation_checks",[]):
        label = f"board_step_{row['step']:02d}"
        prior,current = estimates[previous_label]["world_T_cad"],estimates[label]["world_T_cad"]
        previous_relative = grip_T_cad(feedback[previous_label]["world_T_grip"],prior)
        current_relative = grip_T_cad(feedback[label]["world_T_grip"],current)
        shifted = pose_change(previous_relative,current_relative)
        error = goal_error(current,goal)
        expected_drop = prior[2][3]-goal[2][3]-row["commanded_CAD_height_above_goal_m"]
        actual_drop = prior[2][3]-current[2][3]
        shift_stop = shifted["cad_origin_displacement_m"] > .003 or shifted["rotation_deg"] > 5.
        stall_stop = expected_drop > .0015 and expected_drop-actual_drop > max(.001,.5*expected_drop)
        valid = all(abs(error[key]-row["observed_goal_error"][key]) < 1e-5 for key in error)
        valid &= abs(expected_drop-row["expected_drop_m"]) < 1e-10 and abs(actual_drop-row["observed_drop_m"]) < 1e-10
        valid &= abs(shifted["cad_origin_displacement_m"]-row["observed_grip_relation_change"]["translation_m"]) < 1e-10 and abs(shifted["rotation_deg"]-row["observed_grip_relation_change"]["rotation_deg"]) < 1e-5
        checks[f"board_step_{row['step']:02d}_observation_math_reproduces"] = valid
        if shift_stop or stall_stop:
            checks["observed_stop_reason_reproduces"] = result["status"] == ("observed_part_shift_during_lowering" if shift_stop else "observed_lowering_stalled")
        board_checks.append({"step":row["step"],"shift_stop_replayed":shift_stop,"stall_stop_replayed":stall_stop})
        previous_label = label
    return {"part":part,"status":result["status"],"actual_successful_move_phases":[event["phase"] for event in moves],
        "physical_pickup_attempts":result["physical_pickup_attempts"],"observed_lowering_checks":board_checks,
        "observed_relations_used_for_moves":used,"checks":checks,"passed":all(checks.values())}


def freeze_online_before_truth(run):
    """Require server shutdown and freeze online evidence before private reads."""
    run = Path(run).resolve()
    results = read(run/"results.json")
    if {row["part"] for row in results} != set(PARTS) or any(row["status"] in {"not_started","not_started_yet"} for row in results):
        raise RuntimeError("All three online decisions must be terminal before offline truth evaluation")
    log = read(run/"client_rpc_log.json")
    if log[-1]["action"] != "shutdown" or read(run/log[-1]["response"]) != {"ok":True}:
        raise RuntimeError("Plant shutdown must complete before offline truth evaluation")
    paths = [run/name for name in ("manifest.json","results.json","client_rpc_log.json")]
    for name in ("controller","observations","plant_queue","known_static_model"):
        paths.extend(path for path in (run/name).rglob("*") if path.is_file())
    values = {path.relative_to(run).as_posix():digest(path) for path in sorted(set(paths))}
    destination = run/"online_freeze_before_offline_eval.json"
    if destination.exists():
        existing = {name.replace("\\","/"):sha for name,sha in read(destination).items()}
        if existing != values:
            raise RuntimeError("Frozen online files changed before offline evaluation")
    else:
        save(destination,values)
    return values


def evaluate_offline(run):
    """Read simulator target answers only after the online run is frozen."""
    run = Path(run).resolve()
    frozen = freeze_online_before_truth(run)
    manifest = read(run/"manifest.json")
    rows = []
    for part in PARTS:
        result = read(run/"controller"/part/"result.json")
        states = read(run/"plant_private"/part/"trajectory.json")
        contacts = read(run/"plant_private"/part/"contacts.json")
        controller = read(run/"plant_private"/part/"controller_audit.json")
        goal = manifest["nominal_goals"][part]["world_T_cad_goal"]
        index,observations = 0,[]
        by_label = {row["label"]:row for row in result["observed_estimates"]}
        for event in result["events"]:
            if event["action"] in {"move","hold"}:
                index += event["frames"]
            if event["action"] != "observe":
                continue
            label = Path(event["observation_folder"]).name
            if label not in by_label:
                continue
            record = by_label[label]
            estimate = read(local_path(record["estimate_path"]))
            feedback = read(local_path(record["robot_feedback_path"]))
            state = states[index]
            matrix = state["target_pose_audit"]
            checks = {"observation_frame_robot_FK_matches":close_matrix(feedback["world_T_grip"],state["grip_site_pose"]),
                "observation_frame_robot_encoders_match":feedback["joint_position_rad"] == state["robot_qpos"] and feedback["finger_position_m"] == state["gripper_qpos"]}
            error = None if estimate["status"] != "available" else goal_error(estimate["world_T_cad"],matrix)
            prior_error = None
            packet = read(local_path(record["estimate_path"]).parent/"packet.json")
            if packet["mode"] == "held":
                prior_error = goal_error(packet["prior_world_T_cad"],matrix)
            observations.append({"label":label,"status":estimate["status"],"state_index":index,"simulation_time_s":state["time_s"],
                "estimated_world_T_cad":estimate["world_T_cad"],"actual_world_T_cad_OFFLINE_ONLY":matrix,
                "estimated_pose_error":error,"transport_prior_error":prior_error,
                "actual_goal_error_OFFLINE_ONLY":goal_error(matrix,goal),
                "checks":checks,"passed":all(checks.values())})
        hold_index = next((i for i in range(len(states)-1,-1,-1) if states[i]["phase"] == "hold_lift"),None)
        pickup = None
        if hold_index is not None:
            rise = states[hold_index]["target_pose_audit"][2][3]-states[0]["target_pose_audit"][2][3]
            sides = contact_sides(contacts[hold_index])
            pickup = {"rise_m":rise,"two_finger_force_contacts":sides,"height_and_two_finger_hold_passed":rise >= .12 and len(sides) == 2,
                "one_second_hold_frames":sum(state["phase"] == "hold_lift" for state in states)}
        actual_phases = list(dict.fromkeys(state["phase"] for state in states))
        final_error = goal_error(states[-1]["target_pose_audit"],goal)
        rows.append({"part":part,"online_status_unchanged":result["status"],"observations":observations,
            "pickup_OFFLINE_ONLY":pickup,"actual_recorded_phases":actual_phases,
            "final_actual_goal_error_OFFLINE_ONLY":final_error,
            "actual_nominal_position_and_signed_axis_within_tolerance_OFFLINE_ONLY":final_error["position_m"] <= .001 and final_error["signed_CAD_Z_axis_deg"] <= 2.,
            "physical_engagement_verified":False,"controller_audit":controller,
            "truth_input_hashes":{name:digest(run/"plant_private"/part/name) for name in ("trajectory.json","contacts.json","controller_audit.json","scene.xml")},
            "passed":all(row["passed"] for row in observations)})
    unchanged = all(digest(run/name) == sha for name,sha in frozen.items())
    value = {"scope":"Final offline evaluation only. Target truth and contact labels never enter estimation, planning, stopping decisions or observed status.",
        "online_files_frozen_before_truth_reads":True,"online_files_unchanged_after_evaluation":unchanged,
        "raw_orientation_error_warning":"Raw rotation is full CAD-frame error. Locally similar visible surfaces do not make different keyway, tooth or thread phases equivalent; those phases are not verified by an available local fit.",
        "parts":rows,"passed":unchanged and all(row["passed"] for row in rows)}
    save(run/"offline_evaluation.json",value)
    return value


def audit_run(run):
    run = Path(run).resolve()
    freeze_online_before_truth(run)
    manifest = read(run/"manifest.json")
    goal_path = ROOT/"experiments/runtime_antipodal_assembly_20261023/goal_audit.json"
    goals = read(goal_path)["nominal_simulation_goals"]
    runtime = {"run_trial.py","plant.py","planning.py","sensors.py","pose_estimation.py","estimate_from_files.py","preflight.py"}
    checks = {"all_three_requested_parts":manifest["parts"] == list(PARTS),
        "exact_seven_frozen_runtime_sources":{Path(name.replace("\\","/")).name for name in manifest["source_files_sha256"]} == runtime,
        "runtime_sources_unchanged":all(digest(local_path(name)) == sha for name,sha in manifest["source_files_sha256"].items()),
        "approved_goal_file_unchanged":digest(goal_path) == manifest["goals_file_sha256"],
        "exact_previously_approved_nominal_goals":manifest["nominal_goals"] == {part:goals[part] for part in PARTS},
        "goals_not_claimed_verified_mating_poses":all(goals[part]["official_final_assembly_pose"] is False and goals[part]["actual_insertion_validated"] is False for part in PARTS),
        "four_fixed_camera_views":len(manifest["camera_constants"]) == 4,
        "no_online_simulator_object_state_declared":manifest["boundary"]["simulator_object_state_used_for_estimation_or_control"] is False,
        "no_engine_target_segmentation_or_truth_crop_declared":manifest["boundary"]["renderer_target_segmentation_or_truth_crops_used"] is False,
        "estimated_surrogate_only_planning_declared":manifest["boundary"]["robot_static_planner_plus_estimated_CAD_only"] is True,
        "yaw_phase_not_claimed_verified":manifest["controller_protocol"]["yaw_tooth_thread_phase_verified"] is False}
    model = manifest["GR_model_provenance"]
    checks["frozen_pretrained_GR_checkpoint"] = digest(ROOT/model["checkpoint"]["relative_path"]) == model["checkpoint"]["sha256"]
    sources = source_boundary_checks()
    rpc = audit_rpc(run)
    template = audit_robot_template(run/"known_static_model"/"robot_board.xml")
    packets = []
    for packet in sorted((run/"controller").rglob("packet.json")):
        if not (packet.parent/"estimate.json").is_file():
            packets.append({"packet":str(packet.relative_to(run)),"status":"missing_estimator_output","passed":False})
            continue
        row = audit_estimate(packet,run)
        protocol = read(packet.parent/"estimate.json")["diagnostics"]["protocol"]
        row["checks"]["pose_threshold_protocol_matches_frozen_run"] = protocol == manifest["pose_protocol"]
        row["passed"] = all(row["checks"].values())
        packets.append(row)
    controllers = [audit_controller(run,part,manifest) for part in PARTS]
    offline = evaluate_offline(run)
    preserved = protected_changes()
    value = {"passed":all(checks.values()) and sources["passed"] and rpc["passed"] and template["passed"] and all(row["passed"] for row in packets+controllers) and offline["passed"] and preserved["passed"],
        "scope":"Independent final source/input/decision/transform replay. No model inference or physical simulation rerun. Private truth read only after online files frozen and plant shutdown confirmed.",
        "manifest_checks":checks,"source_boundary_checks":sources,"online_RPC_boundary":rpc,
        "target_free_planning_template":template,"observation_estimates":packets,
        "controller_decisions":controllers,"offline_evaluation_passed":offline["passed"],
        "protected_files":preserved,
        "summary":{"parts":len(PARTS),"completed_estimates":len(packets),"available_estimates":sum(row["status"] == "available" for row in packets),
            "pickup_attempts":sum(row["physical_pickup_attempts"] for row in controllers),
            "offline_height_and_two_finger_pickups":sum(bool(row["pickup_OFFLINE_ONLY"] and row["pickup_OFFLINE_ONLY"]["height_and_two_finger_hold_passed"]) for row in offline["parts"]),
            "physical_engagement_verified":0}}
    save(run/"independent_audit.json",value)
    return value


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        print(json.dumps(snapshot_protected(),indent=2))
    else:
        value = audit_run(Path(sys.argv[1]))
        print(json.dumps({key:value[key] for key in ("passed","summary","protected_files")},indent=2))
        raise SystemExit(0 if value["passed"] else 1)
