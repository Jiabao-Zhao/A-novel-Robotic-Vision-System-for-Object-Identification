"""Replay saved CAD-refined pickup records without model calls or simulation."""
import json
import math
from pathlib import Path
import statistics
import struct
import sys

from experiments.grconvnet_vlm_pickup_20261024.audit import (
    DEPTHS_M, SOURCE, audit_candidates as old_candidate_checks,
    digest, pose_valid, read, replay_pickup,
)


HERE = Path(__file__).resolve().parent
OLD_HERE = HERE.parent
ROOT = HERE.parents[2]
BASE = ROOT / "outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025"
OLD_RUN = ROOT / "outputs/grconvnet_vlm_pickup_20261024/runs/20261024T135155Z"
PARTS = ("Gear_Large", "Gear_Medium", "M16_Hex_Nut")
ARMS = ("geometric", "grconvnet_score", "grconvnet_vlm",
        "cad_grconvnet_score", "cad_grconvnet_vlm")
MESHES = ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"
_TRIANGLES, _RAY_REPLAY = {}, {}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def snapshot_protected():
    """Freeze the complete old 14-part run and all earlier protected hashes."""
    destination = BASE/"protected_before_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    values = read(OLD_RUN.parents[1]/"protected_before_audit.json")
    paths = [path for path in OLD_RUN.rglob("*") if path.is_file()]
    paths += [path for path in OLD_HERE.rglob("*") if path.is_file()
              and HERE not in path.parents and "__pycache__" not in path.parts]
    paths += [ROOT/"docs/cad_view_prompt.txt", ROOT/"cad-view-review/vlm_prompt.txt",
              ROOT/"cad-view-review/vlm_prompt.lock.json"]
    for path in paths:
        if path.is_file():
            values[path.relative_to(ROOT).as_posix()] = digest(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save(destination, values)
    return {"snapshot":str(destination), "protected_files":len(values)}


def protected_changes():
    values = read(BASE/"protected_before_audit.json")
    changed = [name for name, sha in values.items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    return {"passed":not changed, "file_count":len(values), "changed":changed}


def multiply(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def inverse_pose(pose):
    # Candidate axes include float32 CAD arithmetic. Invert their stored 3x3
    # values exactly rather than assuming numerically perfect orthogonality.
    a,b,c = pose[0][:3]
    d,e,f = pose[1][:3]
    g,h,i = pose[2][:3]
    determinant = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    rotation = [[(e*i-f*h)/determinant,(c*h-b*i)/determinant,(b*f-c*e)/determinant],
                [(f*g-d*i)/determinant,(a*i-c*g)/determinant,(c*d-a*f)/determinant],
                [(d*h-e*g)/determinant,(b*g-a*h)/determinant,(a*e-b*d)/determinant]]
    return [row+[-sum(row[k]*pose[k][3] for k in range(3))] for row in rotation]+[[0.,0.,0.,1.]]


def grip_T_cad(world_T_grip, world_T_cad):
    """Map CAD-local coordinates into gripper grip-site coordinates (meters)."""
    return multiply(inverse_pose(world_T_grip), world_T_cad)


def pose_change(reference, current):
    """CAD-origin displacement in the grip frame and relative rotation angle."""
    distance = math.dist([row[3] for row in reference[:3]], [row[3] for row in current[:3]])
    relative = multiply(inverse_pose(reference), current)
    quaternion = quaternion_xyzw(relative)
    angle = math.degrees(2*math.atan2(math.sqrt(sum(item*item for item in quaternion[:3])),abs(quaternion[3])))
    return {"cad_origin_displacement_m":distance, "rotation_deg":angle}


def quaternion_xyzw(pose):
    """Normalized quaternion for a stored near-rigid rotation; q and -q agree."""
    r = pose
    trace = sum(r[i][i] for i in range(3))
    if trace > 0.:
        scale = 2*math.sqrt(trace+1.)
        value = [(r[2][1]-r[1][2])/scale,(r[0][2]-r[2][0])/scale,
                 (r[1][0]-r[0][1])/scale,scale/4]
    else:
        i = max(range(3),key=lambda item:r[item][item])
        j,k = (i+1)%3,(i+2)%3
        scale = 2*math.sqrt(max(0.,1.+r[i][i]-r[j][j]-r[k][k]))
        value = [0.,0.,0.,(r[k][j]-r[j][k])/scale]
        value[i] = scale/4
        value[j] = (r[j][i]+r[i][j])/scale
        value[k] = (r[k][i]+r[i][k])/scale
    norm = math.sqrt(sum(item*item for item in value))
    return [item/norm for item in value]


def relative_trace(folder, result):
    """Recompute part-in-hand relations directly from actual recorded poses."""
    states = read(Path(folder)/"trajectory.json")
    moments = {}
    if result.get("selected_grasp") is not None:
        moments["planned"] = {"grip_T_cad":grip_T_cad(result["selected_grasp"]["world_T_grip"],
                                                     result["world_T_cad_input"]),
                              "source":"Selected commanded grip-site pose and supplied initial CAD pose"}
    for phase in ("descend", "close", "lift", "hold_lift"):
        state = next((item for item in reversed(states) if item["phase"] == phase), None)
        if state is not None:
            moments[phase] = {"time_s":state["time_s"],
                "grip_T_cad":grip_T_cad(state["grip_site_pose"], state["target_pose_audit"]),
                "world_T_grip":state["grip_site_pose"], "world_T_cad":state["target_pose_audit"],
                "grasp_contacts":state["grasp_contacts"]}
    changes = {}
    for before, after, label in (("planned", "close", "planned_to_closed"),
                                  ("descend", "close", "during_closing"),
                                  ("close", "lift", "closed_to_end_lift"),
                                  ("close", "hold_lift", "closed_to_end_hold")):
        if before in moments and after in moments:
            changes[label] = pose_change(moments[before]["grip_T_cad"], moments[after]["grip_T_cad"])
    lifted = [item for item in states if item["phase"] in ("lift", "hold_lift")]
    if "close" in moments and lifted:
        sampled = [pose_change(moments["close"]["grip_T_cad"],
                   grip_T_cad(item["grip_site_pose"], item["target_pose_audit"])) for item in lifted]
        changes["maximum_after_closure"] = {"cad_origin_displacement_m":max(item["cad_origin_displacement_m"] for item in sampled),
                                            "rotation_deg":max(item["rotation_deg"] for item in sampled)}
    checks = {"all_recorded_poses_rigid":all(pose_valid(item["grip_site_pose"]) and pose_valid(item["target_pose_audit"]) for item in states),
              "all_relative_poses_rigid":all(pose_valid(item["grip_T_cad"]) for item in moments.values())}
    for label, item in moments.items():
        item["CAD_origin_in_grip_site_mm"] = [item["grip_T_cad"][i][3]*1000 for i in range(3)]
        item["CAD_orientation_in_grip_site_quaternion_xyzw"] = quaternion_xyzw(item["grip_T_cad"])
        if label == "planned":
            grip, cad = result["selected_grasp"]["world_T_grip"], result["world_T_cad_input"]
        else:
            grip, cad = item["world_T_grip"], item["world_T_cad"]
        reconstructed = multiply(grip, item["grip_T_cad"])
        checks["world_reconstruction_"+label] = max(abs(a-b) for row1,row2 in zip(reconstructed,cad)
                                                   for a,b in zip(row1,row2)) < 1e-10
        recovered_grip = multiply(cad,inverse_pose(item["grip_T_cad"]))
        checks["world_grip_recovery_"+label] = max(abs(a-b) for row1,row2 in zip(recovered_grip,grip)
                                                   for a,b in zip(row1,row2)) < 1e-10
    return {"formula":"grip_T_cad = inverse(world_T_grip) @ world_T_cad",
            "assembly_pose_formula_if_relative_pose_is_maintained":"world_T_grip_goal = world_T_cad_goal @ inverse(grip_T_cad)",
            "length_units":"Matrices use metres; explicitly labelled position summaries use millimetres",
            "gripper_frame":"gripper0_grip_site; same site as Candidate.world_T_grip",
            "CAD_frame":"target CAD body frame used by target_pose_audit; native metric scale",
            "meaning":"CAD-local point transformed into grip-site coordinates. Relative pose need not stay constant during closing or if the part slips.",
            "physical_attempts":result["physical_attempts"], "moments":moments,
            "changes":changes, "checks":checks, "passed":all(checks.values())}


def maximum_error(a,b):
    return max(abs(x-y) for row1,row2 in zip(a,b) for x,y in zip(row1,row2))


def audit_relative_record(result, independent):
    saved = result["part_in_gripper_audit"]
    moments = independent["moments"]
    checks = {"relative_trace_reconstructs":independent["passed"],
              "relative_audit_availability_matches_plan":saved["available"] == ("planned" in moments)}
    if not saved["available"]:
        return checks
    planned = moments["planned"]["grip_T_cad"]
    checks["planned_relative_transform_reproduces"] = maximum_error(planned,saved["planned_grip_T_cad"]) < 1e-10
    checks["pickup_plan_uses_same_relative_transform"] = maximum_error(planned,result["plan"]["grip_T_object"]) < 1e-10
    for phase,record in saved["phases"].items():
        actual = moments[phase]["grip_T_cad"]
        change = pose_change(planned,actual)
        checks[phase+"_part_in_gripper_reproduces"] = maximum_error(actual,record["grip_T_cad_audit"]) < 1e-10
        checks[phase+"_planned_displacement_reproduces"] = abs(change["cad_origin_displacement_m"]-record["translation_change_from_planned_m"]) < 1e-10
        checks[phase+"_planned_angle_reproduces"] = abs(change["rotation_deg"]-record["rotation_change_from_planned_deg"]) < 2e-5
        if "close" in moments:
            from_close = pose_change(moments["close"]["grip_T_cad"],actual)
            checks[phase+"_closure_displacement_reproduces"] = abs(from_close["cad_origin_displacement_m"]-record["translation_change_from_closure_m"]) < 1e-10
            checks[phase+"_closure_angle_reproduces"] = abs(from_close["rotation_deg"]-record["rotation_change_from_closure_deg"]) < 2e-5
    return checks


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def subtract(a,b):
    return [x-y for x,y in zip(a,b)]


def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def point_in_frame(pose,point):
    return [sum(pose[i][k]*point[k] for k in range(3))+pose[i][3] for i in range(3)]


def replay_outer_CAD_hits(result,candidate):
    """Independent binary-STL Moller-Trumbore line intersection, no Open3D."""
    key = (result["part"],json.dumps(candidate["contacts_world_m"]))
    if key in _RAY_REPLAY:
        return _RAY_REPLAY[key]
    path = MESHES/(result["part"]+"_m.stl")
    if result["part"] not in _TRIANGLES:
        raw = path.read_bytes()
        count = struct.unpack_from("<I",raw,80)[0]
        if len(raw) != 84+50*count:
            raise ValueError("This local replay expects native binary STL triangles")
        _TRIANGLES[result["part"]] = [tuple(struct.unpack_from("<9f",raw,84+50*i+12)[j:j+3])
                                             for i in range(count) for j in (0,3,6)]
    points = candidate["contacts_world_m"]
    origin_world = [(a+b)/2 for a,b in zip(*points)]
    axis_world = [candidate["world_T_grip"][i][0] for i in range(3)]
    cad_T_world = inverse_pose(result["world_T_cad_input"])
    origin = point_in_frame(cad_T_world,origin_world)
    axis = [dot(cad_T_world[i][:3],axis_world) for i in range(3)]
    hits = []
    vertices = _TRIANGLES[result["part"]]
    for i in range(0,len(vertices),3):
        a,b,c = vertices[i:i+3]
        edge1,edge2 = subtract(b,a),subtract(c,a)
        p = cross(axis,edge2)
        determinant = dot(edge1,p)
        if abs(determinant) < 1e-14:
            continue
        offset = subtract(origin,a)
        u = dot(offset,p)/determinant
        if u < -1e-7 or u > 1.+1e-7:
            continue
        q = cross(offset,edge1)
        v = dot(axis,q)/determinant
        if v < -1e-7 or u+v > 1.+1e-7:
            continue
        hits.append(dot(edge2,q)/determinant)
    if not hits:
        value = {"passed":False,"reason":"Independent line missed native CAD"}
    else:
        expected = [[origin[j]+t*axis[j] for j in range(3)] for t in (min(hits),max(hits))]
        actual = sorted([point_in_frame(cad_T_world,point) for point in points],key=lambda point:dot(subtract(point,origin),axis))
        errors = [math.dist(a,b) for a,b in zip(expected,actual)]
        value = {"passed":max(errors) < 2e-6,"native_STL_line_intersections":len(hits),
                 "endpoint_error_m":errors,"numerical_replay_tolerance_m":2e-6,
                 "scope":"Contact endpoints match the two outer native CAD intersections on their closing line; no robot or physics rerun"}
    _RAY_REPLAY[key] = value
    return value


def refinement_checks(result):
    details = result["cad_refinement"]
    protocol = details["protocol"]
    checks = {"refinement_no_input_robot_state_change":details["input_robot_state_unchanged"] is True,
        "refinement_no_physics_or_model_or_VLM_calls":all(details[key] == 0 for key in ("physics_steps","model_calls","VLM_calls")),
        "refinement_mesh_hash_matches":details["mesh_sha256"] == result["mesh_sha256"],
        "one_retained_pose_per_network_seed":protocol["maximum_returned_poses_per_network_seed"] == 1,
        "refinement_no_global_sampler_or_annotations":protocol["no_global_sampler"] is True and protocol["no_part_specific_grasp_annotations"] is True,
        "no_IK_or_execution_based_refinement":protocol["no_inverse_kinematics_in_refinement"] is True and protocol["no_VLM_or_execution_outcome_in_variant_selection"] is True,
        "one_diagnostic_per_original_seed":[item["network_candidate_id"] for item in details["per_network_seed"]] == [item["network_candidate_id"] for item in result["raw_network_proposals"]]}
    for summary in details["per_network_seed"]:
        if "CAD_lines_tested" not in summary:
            continue
        scope = "seed_"+str(summary["network_candidate_id"])+"_"
        checks[scope+"fifteen_fixed_CAD_lines_tested"] = summary["CAD_lines_tested"] == 15
        checks[scope+"all_CAD_lines_accounted_for"] = summary["qualified_CAD_lines"]+sum(summary["rejected_lines"].values()) == 15
        checks[scope+"variant_budget_bounded"] = summary["pose_opening_variants_formed"] <= summary["qualified_CAD_lines"]*6*3
        checks[scope+"table_check_counts_reproduce"] = summary["table_checks"] == summary["table_rejections"]+summary["static_target_checks"]
        checks[scope+"target_check_counts_reproduce"] = summary["static_target_checks"] == summary["target_rejections"]+int("selected_correction" in summary)
        checks[scope+"no_extra_variants_after_selected"] = summary["table_checks"] <= summary["pose_opening_variants_formed"]
        variants = summary["geometry_checks"]
        checks[scope+"all_checked_variants_preserved"] = len(variants) == summary["table_checks"]
        checks[scope+"geometry_checks_follow_fixed_rank"] = [item["geometry_rank"] for item in variants] == sorted(item["geometry_rank"] for item in variants)
        feasible = []
        for item in variants:
            table_ok = item["minimum_gripper_table_gap_m"] >= .0001
            target = item.get("open_hand_target_check")
            checks.setdefault(scope+"table_pass_flags_reproduce",True)
            checks[scope+"table_pass_flags_reproduce"] &= item["table_passed"] == table_ok and (target is not None) == table_ok
            if target is not None:
                maximum = max((contact["overlap_m"] for contact in target["contacts"]),default=0.)
                checks.setdefault(scope+"target_overlap_flags_reproduce",True)
                checks[scope+"target_overlap_flags_reproduce"] &= target["maximum_gripper_target_overlap_m"] == maximum and target["passed"] == (maximum <= .00005) and all(abs(contact["overlap_m"]-max(0.,-contact["signed_distance_m"])) < 1e-12 for contact in target["contacts"])
            feasible.append(table_ok and target["passed"])
        checks[scope+"first_feasible_checked_variant_selected"] = (not any(feasible) if "selected_correction" not in summary else bool(feasible) and feasible[-1] and not any(feasible[:-1]))
    ray_replays = []
    for index,candidate in enumerate(result["candidates"]):
        correction = candidate["CAD_correction"]
        before,after = candidate["before_depth_geometry"],candidate["depth_geometry"]
        pair = candidate["contacts_world_m"]
        midpoint = [(a+b)/2 for a,b in zip(*pair)]
        axis = correction["seed_jaw_axis_world_unit"]
        axis_norm = math.sqrt(dot(axis,axis))
        normals = correction["surface_normals_world_unit"]
        cosines = [abs(dot(normal,axis)) for normal in normals]
        shift = math.dist(midpoint[:2],before["center_world_m"][:2])
        chosen = correction["open_hand_target_check"]
        scope = "seed_"+str(candidate["network_candidate_id"])+"_"
        values = {
            "original_network_seed_retained":candidate["proposal"] == next(item for item in result["raw_network_proposals"] if item["network_candidate_id"] == candidate["network_candidate_id"]),
            "CAD_contact_input_declared":candidate["CAD_alignment_assumed_exact"] is True and candidate["predicted_endpoints_are_actual_surface_contacts"] is True and after["object_pose_used"] is True,
            "original_depth_input_excludes_CAD_pose":before["object_pose_used"] is False,
            "jaw_line_unchanged":correction["jaw_direction_changed"] is False and abs(axis_norm-1.) < 1e-8 and abs(abs(dot(axis,[candidate["world_T_grip"][i][0] for i in range(3)]))-1.) < 1e-8,
            "downward_approach_unchanged":max(abs(candidate["world_T_grip"][i][2]-value) for i,value in enumerate((0.,0.,-1.))) < 1e-10,
            "midpoint_XY_shift_reproduces":abs(shift-correction["xy_shift_m"]) < 1e-8,
            "midpoint_within_10mm_seed_bound":shift <= .010+1e-9,
            "bounded_height_slice":correction["chosen_height_fraction"] in (.5,.25,.75),
            "bounded_across_offset_and_zero_along_offset":correction["local_across_offset_m"] in (0.,-.002,.002,-.004,.004) and correction["local_along_offset_m"] == 0.,
            "fixed_CAD_line_index_reproduces":correction["CAD_line_index"] == (.5,.25,.75).index(correction["chosen_height_fraction"])*5+(0.,-.002,.002,-.004,.004).index(correction["local_across_offset_m"]),
            "height_slice_reproduces":abs(midpoint[2]-(details["aligned_CAD_world_z_range_m"][0]+correction["chosen_height_fraction"]*(details["aligned_CAD_world_z_range_m"][1]-details["aligned_CAD_world_z_range_m"][0]))) < 2e-7,
            "span_change_reproduces":abs(candidate["width_m"]-before["predicted_width_m"]-correction["contact_span_change_m"]) < 1e-8,
            "height_change_reproduces":abs(midpoint[2]-before["center_world_m"][2]-correction["contact_center_height_change_m"]) < 1e-8,
            "normal_cosines_reproduce":max(abs(a-b) for a,b in zip(cosines,correction["contact_normal_axis_cosines"])) < 1e-8,
            "normals_within_30deg_of_closing_axis":min(cosines) >= math.cos(math.radians(30.))-1e-7,
            "opposed_contact_normals":dot(normals[0],normals[1]) <= -.5+1e-7,
            "opening_is_fixed_permitted_padding":any(abs(candidate["opening_m"]-min(.08,candidate["width_m"]+padding)) < 1e-10 for padding in (.012,.016,.020)),
            "network_score_preserved":candidate["score"] == -candidate["proposal"]["quality"],
            "selected_correction_matches_seed_diagnostic":correction == next(item for item in details["per_network_seed"] if item["network_candidate_id"] == candidate["network_candidate_id"])["selected_correction"],
            "static_open_hand_has_no_steps_or_robot_angle_changes":chosen["physics_steps"] == 0 and chosen["robot_joint_angles_changed"] is False,
            "static_open_hand_clearance_reproduces":chosen["passed"] is True and chosen["maximum_gripper_target_overlap_m"] == max((item["overlap_m"] for item in chosen["contacts"]),default=0.) and chosen["maximum_gripper_target_overlap_m"] <= .00005,
        }
        replay = replay_outer_CAD_hits(result,candidate)
        last = next(item for item in details["per_network_seed"] if item["network_candidate_id"] == candidate["network_candidate_id"])["geometry_checks"][-1]
        values["selected_pose_is_last_fixed_variant"] = last["geometry_rank"] == correction["chosen_geometry_rank"] and last["finger_depth_m"] == candidate["pad_depth_m"] and last["opening_m"] == candidate["opening_m"] and abs(last["minimum_gripper_table_gap_m"]-candidate["table_clearance_m"]) < 1e-12
        values["outer_native_CAD_hits_reproduce"] = replay["passed"]
        ray_replays.append({"network_candidate_id":candidate["network_candidate_id"],**replay})
        checks.update({scope+key:value for key,value in values.items()})
    return checks,ray_replays


def candidate_checks(folder,result):
    # The original parser covers shared pose, ordering, path and exact VLM
    # request checks. CAD-refined generation explicitly has different inputs.
    normalized = dict(result)
    if result["arm"].startswith("cad_"):
        normalized["arm"] = result["arm"].removeprefix("cad_")
    checks = old_candidate_checks(folder,normalized)
    if result["arm"].startswith("cad_"):
        for key in ("no_CAD_pose_for_learned_generation","depth_geometry_not_CAD_pose",
                    "estimated_points_not_claimed_surface_contacts"):
            checks.pop(key)
        checks["CAD_pose_explicitly_declared_for_refinement"] = result["object_pose_used_for_grasp_generation"] is True and result["cad_refinement_used"] is True
        checks["CAD_mesh_hash_reproduces"] = result["mesh_sha256"] == digest(ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"/(result["part"]+"_m.stl"))
    return checks


def audit_run(run):
    run = Path(run).resolve()
    manifest = read(run/"manifest.json")
    manifest_checks = {
        "exact_three_parts":manifest["parts"] == list(PARTS),
        "exact_five_conditions":manifest["arms"] == list(ARMS),
        "same_six_finger_depths":set(manifest["finger_depths_m"]) == DEPTHS_M,
        "source_scene_unchanged":manifest["source_scene_sha256"] == digest(SOURCE),
        "frozen_parent_runner":manifest["frozen_parent_source_sha256"] == digest(OLD_HERE/"run_trial.py"),
        "frozen_pickup_prompt":manifest["prompt_sha256"] == digest(OLD_HERE/"grasp_rank_prompt.txt"),
        "no_pose_estimator_claimed":manifest["input_conditions"]["CAD_pose_estimator_run"] is False,
        "no_assembly_views_or_goals":manifest["input_conditions"]["final_assembly_views_or_goals"] is False,
        "plain_learned_generator_excludes_CAD_pose":manifest["input_conditions"]["plain_GR_generation_uses_CAD_pose"] is False,
        "perfect_CAD_alignment_assumption_explicit":manifest["input_conditions"]["perfect_CAD_alignment_given_to_geometric_and_CAD_refinement"] is True,
        "one_attempt_limit":manifest["trial"]["physical_attempt_limit_per_part_arm"] == 1,
        "same_pickup_criteria":manifest["trial"]["lift_command_m"] == .18 and manifest["trial"]["min_lift_m"] == .12 and manifest["trial"]["hold_after_lift_s"] == 1. and manifest["trial"]["robot_overlap_limit_m"] == .001 and manifest["trial"]["bilateral_contact_required"] is True,
        "shared_VLM_preload_outside_part_timers":manifest["vlm_shared_warmup"]["raw_response"]["done"] is True and manifest["vlm_shared_warmup"]["raw_response"]["done_reason"] == "load" and manifest["vlm_shared_warmup"]["wall_seconds"] > 0.,
        "fixed_small_CAD_refinement_grid":manifest["cad_refinement_protocol"]["maximum_CAD_lines_per_seed"] == 15 and manifest["cad_refinement_protocol"]["maximum_pose_opening_variants_per_seed"] == 270 and manifest["cad_refinement_protocol"]["local_offset_along_jaw_m"] == 0.,
    }
    for name,sha in manifest["source_hashes"].items():
        manifest_checks["runtime_source_unchanged_"+name] = digest(ROOT/name) == sha
    model = manifest["model_provenance"]
    manifest_checks["checkpoint_unchanged"] = digest(ROOT/model["checkpoint"]["relative_path"]) == model["checkpoint"]["sha256"]
    for name,source in model["source_files"].items():
        manifest_checks["model_source_unchanged_"+name] = digest(OLD_HERE/"vendor/grconvnet_source"/name) == source["sha256"]
    records, rows = {}, []
    for part in PARTS:
        for arm in ARMS:
            folder = run/arm/part
            result = read(folder/"result.json")
            records[arm,part] = result
            replay = replay_pickup(folder,result)
            checks = replay["checks"]
            checks["part_and_condition_match"] = result["part"] == part and result["arm"] == arm
            checks["no_training"] = result["training_performed"] is False
            checks["fresh_generation_not_reused"] = result["shared_pool_cost_reused"] is False
            checks["refinement_flag_matches_condition"] = result["cad_refinement_used"] == arm.startswith("cad_")
            capture = read(folder/"capture/capture_audit.json")
            calibration = read(folder/"capture/calibration.json")
            checks["capture_has_no_simulator_steps_or_pose_changes"] = capture["physics_steps"] == 0 and capture["qpos_unchanged"] is True
            checks["visible_target_mask_available"] = capture["target_pixel_count"] > 0
            checks["fixed_camera_transform_rigid"] = pose_valid(calibration["world_T_camera"])
            checks["camera_not_chosen_from_target_pose"] = calibration["object_pose_used_for_camera_or_grasp_generation"] is False
            checks["metric_optical_depth"] = calibration["depth_units"] == "metres; optical camera Z, not ray range"
            checks.update(candidate_checks(folder,result))
            ray_replays = []
            if arm.startswith("cad_"):
                refinement,ray_replays = refinement_checks(result)
                checks.update(refinement)
                checks["refinement_protocol_matches_frozen_manifest"] = result["cad_refinement"]["protocol"] == manifest["cad_refinement_protocol"]
            checks["pickup_only_plan_fields"] = not result.get("plan") or set(result["plan"]) == {"grasp","pregrasp","lift","grip_T_object","object_T_grip"}
            relative = relative_trace(folder,result)
            checks.update(audit_relative_record(result,relative))
            timing = {**result["stage_seconds"],**{key:result[key] for key in ("setup_seconds","decision_seconds","planning_seconds_including_setup","total_seconds")}}
            checks["finite_nonnegative_timing"] = all(isinstance(item,(int,float)) and math.isfinite(item) and item >= 0. for item in timing.values())
            checks["decision_includes_all_recorded_stages"] = result["decision_seconds"]+1e-9 >= sum(result["stage_seconds"].values())
            checks["setup_and_decision_within_planning_time"] = result["planning_seconds_including_setup"]+1e-9 >= result["setup_seconds"]+result["decision_seconds"]
            checks["total_includes_planning_time"] = result["total_seconds"] >= result["planning_seconds_including_setup"]
            replay.update(part=part,arm=arm,passed=all(checks.values()),reported_status=result["status"],
                          candidate_count=len(result["candidates"]),relative_changes=relative["changes"],timing=timing,
                          independent_CAD_line_replays=ray_replays)
            rows.append(replay)
    matched = []
    for part in PARTS:
        part_rows = [row for row in rows if row["part"] == part]
        learned = [records[arm,part] for arm in ARMS if arm != "geometric"]
        checks = {"same_initial_robot_and_target_state":all(row["initial_qpos"] == part_rows[0]["initial_qpos"] for row in part_rows),
            "same_fresh_RGBD_mask_and_calibration":all(len({digest(run/arm/part/"capture"/name) for arm in ARMS}) == 1 for name in ("rgb.png","depth_m.npy","target_mask.npy","calibration.json")),
            "same_original_network_proposals_across_four_conditions":all(item["raw_network_proposals"] == learned[0]["raw_network_proposals"] for item in learned),
            "same_plain_pool_before_VLM":records["grconvnet_score",part]["candidates"] == records["grconvnet_vlm",part]["candidates"],
            "same_CAD_refined_pool_before_VLM":records["cad_grconvnet_score",part]["candidates"] == records["cad_grconvnet_vlm",part]["candidates"],
            "CAD_score_and_VLM_actual_trajectories_identical":digest(run/"cad_grconvnet_score"/part/"trajectory.json") == digest(run/"cad_grconvnet_vlm"/part/"trajectory.json"),
            "CAD_score_and_VLM_recorded_contacts_identical":digest(run/"cad_grconvnet_score"/part/"contacts.json") == digest(run/"cad_grconvnet_vlm"/part/"contacts.json")}
        matched.append({"part":part,"checks":checks,"passed":all(checks.values())})
    protected = protected_changes()
    summary = {arm:{"part_count":len(PARTS),"pickup_passed":sum(row["pickup_success_replayed"] for row in rows if row["arm"] == arm),
                   "physical_attempts":sum(records[arm,part]["physical_attempts"] for part in PARTS),
                   "total_decision_seconds":sum(records[arm,part]["decision_seconds"] for part in PARTS),
                   "median_decision_seconds":statistics.median(records[arm,part]["decision_seconds"] for part in PARTS),
                   "total_recorded_seconds":sum(records[arm,part]["total_seconds"] for part in PARTS)} for arm in ARMS}
    load_exceptions = []
    for part in PARTS:
        for arm in ("grconvnet_vlm","cad_grconvnet_vlm"):
            path = run/arm/part/"vlm/ranking.json"
            if path.is_file():
                rank = read(path)
                seconds = rank["raw_response"].get("load_duration",0)/1e9
                if seconds > 1.:
                    load_exceptions.append({"part":part,"arm":arm,"API_reported_model_load_seconds":seconds,
                        "included_in_recorded_call_and_decision":True,"bridge_wall_seconds":rank["bridge_wall_seconds"]})
    value = {"passed":all(manifest_checks.values()) and all(row["passed"] for row in rows) and all(row["passed"] for row in matched) and protected["passed"],
        "method":"Independent standard-library replay of saved poses, forces, contacts, transforms, candidate maps and frozen hashes; no simulation or model calls.",
        "timing_condition_exception":{"all_warm_VLM_intent_met":not load_exceptions,"recorded_calls_with_load_over_one_second":load_exceptions,"meaning":"The evidence is internally consistent, but the preload did not prevent this later reload. All measured costs remain unchanged."},
        "manifest_checks":manifest_checks,"condition_count":len(rows),"conditions":rows,
        "matched_conditions":matched,"protected_files":protected,"summary":summary}
    save(run/"independent_audit.json",value)
    return value


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        print(json.dumps(snapshot_protected(), indent=2))
    else:
        value = audit_run(Path(sys.argv[1]))
        print(json.dumps({key:value[key] for key in ("passed","condition_count","protected_files","summary")},indent=2))
        raise SystemExit(0 if value["passed"] else 1)
