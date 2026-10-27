"""Independent replay for new static-depth and exact-route board trials."""
import ast
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import audit as original
from .audit import ROOT, HERE, PARTS, digest, local_path, read, save, multiply, inverse_pose, close_matrix


FIRST_RUN = ROOT/"outputs/observed_held_assembly_20261025/runs/20261025T201252Z"
FOLLOWUP = HERE/"board_feedback_followup_20261025"
ORIGINAL_RUNTIME = {"run_trial.py","plant.py","planning.py","sensors.py","pose_estimation.py","estimate_from_files.py","preflight.py"}


def input_path(name):
    text = str(name).replace("\\","/")
    return local_path(name) if "/"+ROOT.name+"/" in text else Path(name).resolve()


def static_asset_files(template):
    root = ET.parse(template).getroot()
    compiler = root.find("compiler")
    paths = {Path(template).resolve()}
    for element in root.findall("asset/*"):
        if not element.get("file"):
            continue
        path = Path(element.get("file"))
        if not path.is_absolute():
            base = Path(template).parent
            directory = "meshdir" if element.tag == "mesh" else "texturedir"
            if compiler is not None and compiler.get(directory):
                base = Path(compiler.get(directory))
                if not base.is_absolute():
                    base = Path(template).parent/base
            path = base/path
        paths.add(path.resolve())
    return paths


def audit_preprocessing(packet_path,run):
    packet_path,run = Path(packet_path).resolve(),Path(run).resolve()
    packet = read(packet_path)
    record_path = packet_path.parent/"preprocessing.json"
    record = read(record_path)
    raw = local_path(record["raw_observation_folder"])
    checks = {"raw_input_is_public_sensor_inbox":raw.is_relative_to(run/"observations"),
        "declared_no_pose_core_or_gate_changes":record.get("pose_core_or_acceptance_gates_changed",record.get("pose_core_or_original_acceptance_gates_changed")) is False,
        "declared_no_estimated_or_true_part_used_for_filter":record["part_state_or_estimated_CAD_used_for_filter"] is False,
        "preprocessing_mode_matches":record["mode"] == packet["mode"]}
    rows = []
    if packet["mode"] == "table":
        checks["initial_table_images_are_raw"] = all(local_path(path).parent == raw for observation in packet["observations"] for path in observation.values())
        return {"checks":checks,"passed":all(checks.values()),"sensor_directory":raw,"cameras":rows}
    filtered = local_path(record["observation_folder"])
    manifest_path = local_path(record["filter_manifest"])
    manifest = read(manifest_path)
    checks.update({"filtered_directory_is_separate_estimate_artifact":filtered == packet_path.parent/"filtered_observations" and not filtered.is_relative_to(raw),
        "filter_manifest_at_exact_filtered_path":manifest_path == filtered/"known_static_filter.json",
        "fixed_1mm_static_subtraction":manifest["match_tolerance_m"] == .001,
        "robot_only_nine_qpos_template":manifest["robot_qpos_count"] == 9,
        "zero_physics_steps":manifest["physics_steps"] == 0 and record["physics_steps"] == 0,
        "no_part_pose_segmentation_or_gate_changes":all(manifest[key] is False for key in ("estimated_CAD_or_target_state_input","segmentation_used","pose_estimator_gates_changed")),
        "filter_declared_inputs_unchanged":manifest["inputs_unchanged"] is True,
        "filter_source_hash_matches":manifest["helper_sha256"] == digest(FOLLOWUP/"known_static_filter.py"),
        "robot_feedback_copy_exact":digest(raw/"robot_feedback.json") == digest(filtered/"robot_feedback.json")})
    expected = static_asset_files(run/"known_static_model/robot_board.xml") | {raw/"robot_feedback.json"}
    for entry in packet.get("filtered_observations",packet["observations"]):
        name = read(local_path(entry["calibration_path"]))["camera_name"]
        base_name = local_path(entry["calibration_path"]).name.removesuffix("_calibration.json")
        camera = next(row for row in manifest["cameras"] if row["camera"] == base_name)
        diagnostic = local_path(camera["diagnostics"])
        raw_path = raw/(base_name+"_depth_m.npy")
        output_path = local_path(entry["depth_m_path"])
        actual_raw = np.load(raw_path,allow_pickle=False)
        known = np.load(diagnostic/"known_static_depth.npy",allow_pickle=False)
        removed = np.load(diagnostic/"removed_mask.npy",allow_pickle=False)
        actual_filtered = np.load(output_path,allow_pickle=False)
        valid = np.isfinite(actual_raw) & (actual_raw > 0)
        predicted = np.isfinite(known) & (known > 0)
        with np.errstate(invalid="ignore"):
            expected_removed = valid & predicted & (np.abs(actual_raw-known) <= .001)
        expected_filtered = actual_raw.astype(float,copy=True)
        expected_filtered[expected_removed] = np.nan
        copied = ("_rgb.png","_robot_depth_m.npy","_calibration.json")
        copied_checks = {suffix:digest(raw/(base_name+suffix)) == digest(filtered/(base_name+suffix)) for suffix in copied}
        expected.update(raw/(base_name+suffix) for suffix in (*copied,"_depth_m.npy"))
        camera_checks = {"output_paths_in_filtered_directory":all(local_path(path).parent == filtered for key,path in entry.items() if key != "known_static_depth_m_path"),
            "stored_removed_mask_exact":np.array_equal(removed,expected_removed),
            "stored_filtered_depth_exact":np.array_equal(actual_filtered,expected_filtered,equal_nan=True),
            "diagnostic_filtered_copy_exact":np.array_equal(actual_filtered,np.load(diagnostic/"filtered_depth.npy",allow_pickle=False),equal_nan=True),
            "unchanged_RGB_robot_depth_calibration":all(copied_checks.values()),
            "reported_pixel_counts_reproduce":camera["raw_valid_pixels"] == int(valid.sum()) and camera["known_scene_valid_pixels"] == int(predicted.sum()) and camera["removed_matching_pixels"] == int(removed.sum()) and camera["retained_valid_pixels"] == int(np.sum(np.isfinite(actual_filtered)&(actual_filtered>0))),
            "per_camera_removed_summary_matches":record["removed_pixels_per_camera"][base_name] == int(removed.sum())}
        rows.append({"camera":name,"checks":camera_checks,"passed":all(camera_checks.values())})
    actual = {input_path(name):sha for name,sha in manifest["input_files_sha256"].items()}
    checks["exact_known_geometry_sensor_FK_input_set"] = set(actual) == expected
    checks["all_declared_input_hashes_unchanged"] = all(path.is_file() and digest(path) == sha for path,sha in actual.items())
    return {"checks":checks,"passed":all(checks.values()) and all(row["passed"] for row in rows),
        "sensor_directory":filtered,"cameras":rows}


def audit_yaw_estimate(packet_path,run,manifest):
    packet,estimate = read(packet_path),read(packet_path.parent/"estimate.json")
    diagnostics,provenance = estimate["diagnostics"],estimate["provenance"]
    bank_path = local_path(estimate["ranked_candidates_path"])
    bank = read(bank_path)
    part = next(part for part in PARTS if part in packet_path.parts)
    label = packet_path.parent.name
    expected_mode = "full_circle" if label == "after_pickup" else "local_phase"
    step = {"Gear_Large":6.,"Gear_Medium":9.}[part]
    offsets = [index*step for index in range(round(360/step))] if expected_mode == "full_circle" else [0.,-step,step,-2*step,2*step]
    cad = ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"/(part+"_m.stl")
    checks = {"exact_packet_field_allowlist":set(packet) == {"mode","cad_path","observations","filtered_observations","prior_world_T_cad","prior_source","yaw_search_mode","yaw_step_deg"},
        "native_requested_CAD":local_path(packet["cad_path"]) == cad,
        "correct_declared_search":packet["mode"] == "held" and packet["yaw_search_mode"] == expected_mode and packet["yaw_step_deg"] == step,
        "motion_prior_unchanged":packet["prior_world_T_cad"] == diagnostics["predicted_world_T_cad"],
        "protocol_matches_frozen_yaw_protocol":diagnostics["protocol"] == manifest["yaw_protocol"],
        "no_engine_truth_or_selected_diagnostic_pose":provenance["simulator_scene_trial_trajectory_result_files_read"] is False and provenance["diagnostic_chosen_pose_or_actual_orientation_used"] is False,
        "no_training_or_model_weights":provenance["training_or_model_weights_used"] is False,
        "raw_ranking_filtered_fitting_declared":provenance["ranking_uses_raw_depth"] is True and provenance["local_fitting_uses_filtered_depth"] is True,
        "bank_saved_in_same_estimate_directory":bank_path == packet_path.parent/"ranked_candidates.json",
        "ranked_bank_hash_unchanged":digest(bank_path) == provenance["ranked_candidates_sha256"],
        "available_pose_rigid":estimate["status"] != "available" or original.pose_valid(estimate["world_T_cad"]),
        "unavailable_pose_not_fabricated":estimate["status"] == "available" or estimate["world_T_cad"] is None,
        "orientation_confidence_not_overclaimed":diagnostics["absolute_symmetric_orientation_resolved"] is False and diagnostics["ranking_confidence_calibrated"] is False}
    checks.update({"lineage_"+key:value for key,value in original.audit_prior_lineage(packet_path,run)["checks"].items()})
    inputs = {packet_path,cad}
    raw_calibrations = {}
    for kind in ("observations","filtered_observations"):
        for index,entry in enumerate(packet[kind]):
            expected_keys = original.OBSERVATION_FIELDS | ({"known_static_depth_m_path"} if kind == "filtered_observations" else set())
            checks[f"{kind}_{index}_exact_sensor_fields"] = set(entry) == expected_keys
            paths = {key:local_path(name) for key,name in entry.items()}
            inputs.update(paths.values())
            calibration = read(paths["calibration_path"])
            checks[f"{kind}_{index}_calibration_allowlist"] = set(calibration) <= original.CALIBRATION_FIELDS and original.pose_valid(calibration["world_T_camera"])
            if kind == "observations":
                checks[f"raw_{index}_public_sensor_inbox"] = all(path.is_relative_to(run/"observations") for path in paths.values())
                raw_calibrations[calibration["camera_name"]] = calibration
            else:
                checks[f"filtered_{index}_calibration_matches_raw"] = raw_calibrations[calibration["camera_name"]] == calibration
    actual = {local_path(name):sha for name,sha in provenance["input_files_sha256"].items()}
    checks["exact_worker_input_file_set"] = set(actual) == inputs
    checks["all_worker_input_hashes_unchanged"] = all(digest(path) == sha for path,sha in actual.items())
    checks["all_fitting_sources_unchanged"] = all(digest(local_path(name)) == sha for name,sha in provenance["source_files_sha256"].items())
    if bank:
        checks["all_declared_yaw_starts_retained"] = len(bank) == len(offsets) and sorted(row["initial_local_Z_offset_deg"] for row in bank) == sorted(offsets)
        checks["bank_ranked_by_geometry_loss"] = bank == sorted(bank,key=lambda row:(row["combined_loss_m2"],row["candidate_id"]))
        checks["declared_rank_order_matches_bank"] = diagnostics["ranked_candidate_ids"] == [row["candidate_id"] for row in bank]
        for row in bank:
            identifier = row["candidate_id"]
            angle = math.radians(row["initial_local_Z_offset_deg"])
            rotation = [[math.cos(angle),-math.sin(angle),0.,0.],[math.sin(angle),math.cos(angle),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
            seed = multiply(packet["prior_world_T_cad"],rotation)
            change = original.pose_change(seed,row["world_T_cad"])
            checks[f"candidate_{identifier}_bounded_around_own_declared_seed"] = change["cad_origin_displacement_m"] <= .020+1e-9 and change["rotation_deg"] <= 20.+1e-6
            history = row["local_refinement_history"]
            checks[f"candidate_{identifier}_fixed_ICP_schedule"] = all(item["gate_m"] in (.012,.006,.003) and 0 <= item["iteration"] < 15 and item["inlier_points"] >= 30 and item["step_fraction"] in (1.,.5,.25,.125) for item in history)
            checks[f"candidate_{identifier}_ICP_loss_nonincreasing"] = all(b["loss_m2"] <= a["loss_m2"]+1e-14 for a,b in zip(history,history[1:]))
            forward = row["forward_depth_check"]
            count = sum(camera["assessable_CAD_pixels"] for camera in forward["cameras"])
            agreed = sum(camera["depth_agreeing_CAD_pixels"] for camera in forward["cameras"])
            fraction = agreed/count if count else 0.
            loss = sum(camera["clipped_squared_residual_sum_m2"] for camera in forward["cameras"])/count if count else .002**2
            checks[f"candidate_{identifier}_forward_depth_arithmetic"] = count == forward["assessable_CAD_pixels"] and abs(fraction-forward["fraction_within_2mm"]) < 1e-12 and forward["passed"] == (count >= 30 and fraction >= .5) and abs(loss-row["forward_depth_loss_m2"]) < 1e-18
            checks[f"candidate_{identifier}_equal_weight_loss"] = abs(row["combined_loss_m2"]-.5*(row["point_surface_loss_m2"]+row["forward_depth_loss_m2"])) < 1e-18
        distinct = []
        for row in bank:
            if all(original.pose_change(previous["world_T_cad"],row["world_T_cad"])["rotation_deg"] > 2. for previous in distinct):
                distinct.append(row)
        gap = distinct[1]["combined_loss_m2"]-bank[0]["combined_loss_m2"] if len(distinct)>1 else None
        tolerance = 64*sys.float_info.epsilon*max(abs(bank[0]["combined_loss_m2"]),abs(distinct[1]["combined_loss_m2"]),1e-12) if len(distinct)>1 else None
        tied = gap is not None and gap <= tolerance
        checks["distinct_orientation_bank_and_tie_reproduce"] = diagnostics["distinct_yaw_ranked_candidate_ids"] == [row["candidate_id"] for row in distinct] and diagnostics["distinct_solution_combined_loss_gap_m2"] == gap and diagnostics["numerical_exact_tie_tolerance_m2"] == tolerance and diagnostics["numerical_exact_tie_between_distinct_rotations"] == tied
        available = not tied and bank[0]["observed_point_inlier_fraction_within_1mm"] >= .5 and bank[0]["forward_depth_check"]["passed"]
        checks["original_numeric_quality_gates_and_exact_tie_reproduce"] = available == (estimate["status"] == "available")
        checks["selected_pose_is_lowest_loss_candidate"] = diagnostics["selected_candidate_id"] == bank[0]["candidate_id"] and (not available or estimate["world_T_cad"] == bank[0]["world_T_cad"])
        singular = diagnostics["local_observability_singular_values"]
        checks["local_support_diagnostic_reproduces"] = diagnostics["local_six_parameter_support"] == (len(singular) == 6 and singular[-1] > singular[0]*1e-4)
    else:
        checks["empty_bank_is_unavailable_insufficient_depth"] = estimate["status"] == "unavailable_insufficient_nonrobot_depth"
    return {"packet":str(packet_path.relative_to(run)),"mode":"held_yaw_bank","status":estimate["status"],"checks":checks,"passed":all(checks.values())}


def audit_staged_route(run,part,manifest):
    result = read(run/"controller"/part/"result.json")
    route = result.get("staged_route_search")
    checks = {}
    if route is None:
        return {"part":part,"route_search_attempted":False,"checks":checks,"passed":True}
    goal = manifest["nominal_goals"][part]["world_T_cad_goal"]
    record = next(row for row in result["observed_estimates"] if row["label"] == "after_pickup")
    estimate,feedback = read(local_path(record["estimate_path"])),read(local_path(record["robot_feedback_path"]))
    relative = original.grip_T_cad(feedback["world_T_grip"],estimate["world_T_cad"])
    checks.update({"no_symmetry_substituted_goal":route["exact_original_goal_world_T_cad"] == goal,
        "relation_uses_observed_pose_and_FK":close_matrix(route["observed_grip_T_cad_used"],relative),
        "measured_finger_separation_used":abs(route["opening_m"]-min(.08,abs(feedback["finger_position_m"][0]-feedback["finger_position_m"][1]))) < 1e-12,
        "route_protocol_frozen":route["protocol"] == manifest["controller_protocol"]["staged_exact_goal_route"],
        "no_physics_in_route_search":route["protocol"]["physics_steps_in_route_search"] == 0})
    selected = route["selected_route_index"]
    passed = [row["route_index"] for row in route["candidates"] if row["passed"]]
    checks["first_passing_route_selected"] = (selected is None and not passed) or passed == [selected] and selected == len(route["candidates"])-1
    heights = list(dict.fromkeys([feedback["world_T_grip"][2][3]]+[max(feedback["world_T_grip"][2][3],height) for height in (.30,.35,.40)]))
    yaw_sequences = [[],[90.],[-90.],[90.,180.],[-90.,-180.]]
    final_grip = multiply(goal,inverse_pose(relative))
    above = [row.copy() for row in goal];above[2][3] += .16
    for candidate in route["candidates"]:
        index = candidate["route_index"]
        checks[f"route_{index}_fixed_search_order"] = candidate["tool_height_m"] == heights[index//5] and candidate["world_Z_via_yaw_sequence_deg"] == yaw_sequences[index%5]
        checks[f"route_{index}_transfer_original_goal_plus_160mm"] = close_matrix(candidate["planned_transfer_world_T_grip"],multiply(above,inverse_pose(relative)))
        last = candidate["prefix"][-1]["world_T_grip"]
        checks[f"route_{index}_turn_exact_original_orientation"] = all(abs(last[i][j]-final_grip[i][j]) < 1e-10 for i in range(3) for j in range(3))
        checks[f"route_{index}_turn_preserves_XY_uses_declared_height"] = all(abs(last[i][3]-feedback["world_T_grip"][i][3]) < 1e-10 for i in range(2)) and abs(last[2][3]-candidate["tool_height_m"]) < 1e-10
        checks[f"route_{index}_report_pass_reproduces"] = candidate["passed"] == (len(candidate["reports"]) == len(candidate["prefix"])+1 and all(row["status"] == "passed" for row in candidate["reports"]))
        for stage in candidate["prefix"]:
            if not stage["phase"].startswith("route_turn_via_"):
                continue
            angle = candidate["world_Z_via_yaw_sequence_deg"][int(stage["phase"].split("_")[-1])]
            cosine,sine = math.cos(math.radians(angle)),math.sin(math.radians(angle))
            rotate = [[cosine,-sine,0.,0.],[sine,cosine,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
            expected_rotation = multiply(rotate,feedback["world_T_grip"])
            checks[f"route_{index}_{stage['phase']}_world_Z_rotation_reproduces"] = all(abs(stage["world_T_grip"][i][j]-expected_rotation[i][j]) < 1e-10 for i in range(3) for j in range(3))
    return {"part":part,"route_search_attempted":True,"selected_route_index":selected,"checks":checks,"passed":all(checks.values())}


def audit_release_gate(run,part,manifest):
    result = read(run/"controller"/part/"result.json")
    checks = {}
    goal = manifest["nominal_goals"][part]["world_T_cad_goal"]
    before = result.get("observed_goal_error_before_release")
    if before is not None:
        record = next(row for row in result["observed_estimates"] if row["label"] == "board_step_13")
        measured = original.goal_error(read(local_path(record["estimate_path"]))["world_T_cad"],goal)
        checks["pre_release_observed_error_reproduces"] = all(abs(before[key]-measured[key]) < 1e-5 for key in measured)
        passed = before["position_m"] <= .001 and before["signed_CAD_Z_axis_deg"] <= 2. and before["raw_rotation_deg"] <= 2.
        checks["release_only_after_all_three_observed_limits_passed"] = result["release_commanded"] == passed
    else:
        checks["no_release_without_completed_observed_pose_gate"] = result["release_commanded"] is False
    final = result.get("observed_final_goal_error")
    if final is not None:
        passed = final["position_m"] <= .001 and final["signed_CAD_Z_axis_deg"] <= 2. and final["raw_rotation_deg"] <= 2.
        checks["final_observed_status_replays_full_rotation_gate"] = (result["status"] == "observed_nominal_pose_reached_no_engagement_claim") == passed
    return {"part":part,"checks":checks,"passed":all(checks.values())}


def yaw_source_boundary_checks():
    rows = []
    banned = {"target_pose","target_pose_audit","target_bid","target_qadr","enable_segmentation_rendering","mj_step","mj_forward"}
    for name in ("yaw_search.py","estimate_from_files.py"):
        path = FOLLOWUP/"yaw_feedback"/name
        tree = ast.parse(path.read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        checks = {"no_simulator_or_plant_import":not any(value and (value.startswith("mujoco") or "plant" in value or "robot_trial" in value) for value in imports),
            "no_target_state_segmentation_or_simulation_calls":not any(isinstance(node,ast.Attribute) and node.attr in banned for node in ast.walk(tree)),
            "no_prior_diagnostic_or_offline_result_paths":not any(isinstance(node,ast.Constant) and isinstance(node.value,str) and any(token in node.value for token in ("plant_private","global_yaw_diagnostic/","offline_evaluation.json")) for node in ast.walk(tree))}
        rows.append({"source":str(path.relative_to(ROOT)),"sha256":digest(path),"checks":checks,"passed":all(checks.values())})
    return {"sources":rows,"passed":all(row["passed"] for row in rows)}


def audit_run(run):
    run = Path(run).resolve()
    original.freeze_online_before_truth(run)
    manifest,first = read(run/"manifest.json"),read(FIRST_RUN/"manifest.json")
    original.freeze_online_before_truth(FIRST_RUN)
    checks = {"all_three_requested_parts":manifest["parts"] == list(PARTS),
        "all_frozen_runtime_sources_unchanged":all(digest(local_path(name)) == sha for name,sha in manifest["source_files_sha256"].items()),
        "original_seven_sources_preserved":all(manifest["source_files_sha256"].get(name) == sha and digest(local_path(name)) == sha for name,sha in first["source_files_sha256"].items()),
        "original_pose_protocol_unchanged":manifest["pose_protocol"] == first["pose_protocol"],
        "original_grasp_protocol_unchanged":manifest["grasp_protocol"] == first["grasp_protocol"],
        "original_controller_thresholds_unchanged":all(manifest["controller_protocol"].get(key) == value for key,value in first["controller_protocol"].items()),
        "original_nominal_goals_exact":manifest["nominal_goals"] == first["nominal_goals"] and manifest["goals_file_sha256"] == first["goals_file_sha256"],
        "same_four_calibrated_cameras":manifest["camera_constants"] == first["camera_constants"],
        "full_observed_CAD_rotation_gate_added_at_2deg":manifest["controller_protocol"]["nominal_final_full_CAD_rotation_tolerance_deg"] == 2.,
        "phase_verification_not_overclaimed":manifest["controller_protocol"]["yaw_tooth_thread_phase_verified"] is False,
        "same_pretrained_GR_checkpoint":manifest["GR_model_provenance"]["checkpoint"] == first["GR_model_provenance"]["checkpoint"]}
    boundary = manifest["boundary"]
    checks.update({"no_online_engine_target_state_declared":boundary["simulator_object_state_used_for_estimation_or_control"] is False,
        "no_engine_segmentation_or_truth_crop_declared":boundary["renderer_target_segmentation_or_truth_crops_used"] is False,
        "planner_uses_only_robot_static_and_estimated_CAD":boundary["robot_static_planner_plus_estimated_CAD_only"] is True})
    if "yaw_protocol" in manifest:
        checks["no_previous_selected_diagnostic_pose_or_error_input"] = boundary["prior_diagnostic_selected_poses_or_offline_errors_used"] is False
        checks["declared_parse_retry_never_resends"] = boundary["response_parse_retry_resends_physical_command"] is False
    checkpoint = manifest["GR_model_provenance"]["checkpoint"]
    checks["pretrained_checkpoint_hash_unchanged"] = digest(ROOT/checkpoint["relative_path"]) == checkpoint["sha256"]
    source_boundary = original.source_boundary_checks()
    yaw_boundary = yaw_source_boundary_checks() if "yaw_protocol" in manifest else {"passed":True,"sources":[]}
    rpc = original.audit_rpc(run)
    log = read(run/"client_rpc_log.json")
    checks["no_physical_command_resent"] = all(row.get("physical_command_resent") is False for row in log)
    checks["one_unique_request_per_RPC"] = len({row["request"] for row in log}) == len(log)
    checks["all_request_files_accounted_in_log"] = {str(path.relative_to(run)).replace("\\","/") for path in (run/"plant_queue").glob("*.request.json")} == {row["request"].replace("\\","/") for row in log}
    template = original.audit_robot_template(run/"known_static_model/robot_board.xml")
    packets,preprocessing = [],[]
    for packet_path in sorted((run/"controller").rglob("packet.json")):
        filtered = audit_preprocessing(packet_path,run)
        if "yaw_search_mode" in read(packet_path):
            filtered.pop("sensor_directory")
            estimate = audit_yaw_estimate(packet_path,run,manifest)
        else:
            estimate = original.audit_estimate(packet_path,run,permitted_sensor_directory=filtered.pop("sensor_directory"))
            estimate["checks"]["pose_protocol_matches_frozen_run"] = read(packet_path.parent/"estimate.json")["diagnostics"]["protocol"] == manifest["pose_protocol"]
        estimate["passed"] = all(estimate["checks"].values())
        packets.append(estimate)
        preprocessing.append({"packet":str(packet_path.relative_to(run)),**filtered})
    controllers = [original.audit_controller(run,part,manifest) for part in PARTS]
    routes = [audit_staged_route(run,part,manifest) for part in PARTS]
    release_gates = [audit_release_gate(run,part,manifest) for part in PARTS]
    offline = original.evaluate_offline(run)
    preserved = original.protected_changes()
    value = {"scope":"Independent final filtered-input, prior, exact-goal route, observed decision and transform replay. No inference or simulation. Private truth read only after all online outputs frozen and plant shutdown.",
        "checks":checks,"source_boundary":source_boundary,"yaw_source_boundary":yaw_boundary,"online_RPC_boundary":rpc,
        "target_free_planning_template":template,"preprocessing":preprocessing,"observation_estimates":packets,
        "controller_decisions":controllers,"staged_routes":routes,"release_pose_gates":release_gates,"offline_evaluation_passed":offline["passed"],
        "protected_files":preserved,"summary":{"parts":len(PARTS),"completed_estimates":len(packets),
            "available_estimates":sum(row["status"] == "available" for row in packets),
            "pickup_attempts":sum(row["physical_pickup_attempts"] for row in controllers),
            "offline_height_and_two_finger_pickups":sum(bool(row["pickup_OFFLINE_ONLY"] and row["pickup_OFFLINE_ONLY"]["height_and_two_finger_hold_passed"]) for row in offline["parts"]),
            "physical_engagement_verified":0}}
    value["passed"] = all(checks.values()) and all(row["passed"] for row in (source_boundary,yaw_boundary,rpc,template,preserved)) and all(row["passed"] for row in packets+preprocessing+controllers+routes+release_gates) and offline["passed"]
    save(run/"independent_audit.json",value)
    return value


if __name__ == "__main__":
    value = audit_run(Path(sys.argv[1]))
    print(json.dumps({key:value[key] for key in ("passed","summary","checks")},indent=2))
    raise SystemExit(0 if value["passed"] else 1)
