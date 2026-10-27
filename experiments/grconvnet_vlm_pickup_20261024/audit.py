"""Independently replay saved grasp-comparison records; never run a model.

This module uses the standard library and the earlier read-only XML/record
parsers. It does not import a robot controller or create a simulator.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.audit import (
    PARTS, SOURCE, contact_sides, digest, pose_valid, protected_files,
    read, scene_checks,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/grconvnet_vlm_pickup_20261024"
PREVIOUS = ROOT / "outputs/runtime_antipodal_assembly_20261023/smaller_search/20261024T030119Z"
DEPTHS_M = {-.008, -.004, 0., .004, .006, .008}
ARMS = ("geometric", "grconvnet_score", "grconvnet_vlm")


def snapshot_protected():
    """Freeze historical evidence and protected source hashes before new runs."""
    path = BASE / "protected_before_audit.json"
    if path.exists():
        raise FileExistsError(path)
    values = protected_files()
    previous_paths = [PREVIOUS / name for name in (
        "manifest.json", "results.json", "comparison.json", "comparison.csv",
        "independent_audit.json", "independent_surface_audit.json", "preservation.json")]
    for arm in ("small", "large_control"):
        for part in PARTS:
            previous_paths += [PREVIOUS / arm / part / name for name in (
                "result.json", "controller_audit.json", "trajectory.json", "contacts.json",
                "generated_candidates.npz", "scene.xml")]
    previous_paths += list((ROOT / "experiments/runtime_antipodal_assembly_20261023").rglob("*.py"))
    for previous in previous_paths:
        if previous.is_file():
            values[previous.relative_to(ROOT).as_posix()] = digest(previous)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protected_snapshot": str(path), "files": len(values)}))
    return path


def protected_changes():
    values = read(BASE / "protected_before_audit.json")
    changed = [name for name, sha in values.items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    return {"file_count": len(values), "changed": changed, "passed": not changed}


def replay_pickup(folder, result):
    """Rebuild the pickup outcome from recorded heights, forces and contacts."""
    folder = Path(folder)
    states = read(folder / "trajectory.json")
    contacts = read(folder / "contacts.json")
    controller = read(folder / "controller_audit.json")
    checks, offsets = scene_checks(folder / "scene.xml", result["part"])
    checks["nonempty_trajectory"] = bool(states)
    checks["contact_frame_count"] = len(contacts) == len(states) == controller["trajectory_frames"]
    checks["time_monotonic"] = all(b["time_s"] >= a["time_s"] for a, b in zip(states, states[1:]))
    checks["force_contacts_match_trajectory"] = all(
        state["grasp_contacts"] == contact_sides(frame) for state, frame in zip(states, contacts))
    checks["no_attachments_in_record"] = controller["attachments_present"] is False
    checks["no_target_feedback_in_controller"] = controller["target_feedback_in_controller"] is False
    checks["other_parts_visible"] = controller["other_loose_parts_hidden_and_noncolliding"] == []
    checks["friction_unchanged"] = controller["friction_coefficients_changed"] is False
    checks["source_scene_hash"] = controller["source_scene_sha256"] == digest(SOURCE)
    checks["fresh_initial_state_time_zero"] = states[0]["time_s"] == 0.
    checks["at_most_one_physical_attempt"] = result["physical_attempts"] in (0, 1)
    checks["no_assembly_attempt"] = result["assembly_attempted"] is False
    checks["no_assembly_phases"] = not any(state["phase"] in
        {"turn", "transfer", "insert", "retreat", "settle"} for state in states)
    # Existing controller resets inactive parts every 2 ms, then integrates one
    # step under gravity before each saved frame. The offset must not accumulate.
    offset_z = -9.81 * controller["physics_timestep_s"]**2
    fixed_checks = []
    for name, (first, size) in offsets.items():
        if name == "nist_part_" + result["part"]:
            continue
        expected = states[0]["qpos"][first:first+size].copy()
        expected[2] += offset_z
        fixed_checks.append(all(max(abs(a-b) for a, b in zip(
            state["qpos"][first:first+size], expected)) < 1e-10 for state in states[1:]))
    checks["thirteen_other_parts_reset_with_one_step_gravity_shift"] = len(fixed_checks) == 13 and all(fixed_checks)
    lift = states[-1]["target_pose_audit"][2][3] - states[0]["target_pose_audit"][2][3]
    bilateral = len(contact_sides(contacts[-1])) == 2
    held = states[-1]["phase"] == "hold_lift"
    hold_states = [state for state in states if state["phase"] == "hold_lift"]
    one_second_hold = len(hold_states) == 25
    collision_maxima = {key: controller["max_penetration_m"][key]
                        for key in ("robot_self", "robot_table", "robot_board")}
    collision_ok = max(collision_maxima.values()) < .001
    passed = held and one_second_hold and lift >= .12 and bilateral and collision_ok
    checks["pickup_outcome_reproduces"] = passed == (result["status"] == "pickup_passed")
    if "lift_m" in result:
        checks["reported_lift_reproduces"] = abs(lift-result["lift_m"]) < 1e-12
    if result["physical_attempts"] == 0:
        checks["no_motion_when_no_attempt"] = len(states) == 1 and states[-1]["phase"] == "initial"
    initial = states[0]["target_pose_audit"]
    checks["initial_target_pose_rigid"] = pose_valid(initial)
    return {"checks": checks, "passed": all(checks.values()),
            "pickup_success_replayed": passed, "lift_m_replayed": lift,
            "final_bilateral_contact": bilateral, "one_second_hold": one_second_hold,
            "last_phase": states[-1]["phase"], "trajectory_frames": len(states),
            "robot_overlap_max_m": collision_maxima, "inactive_part_one_step_shift_m": offset_z,
            "initial_qpos": states[0]["qpos"], "initial_target_pose": initial}


def audit_candidates(folder, result):
    candidates = result["candidates"]
    learned = result["arm"] != "geometric"
    checks = {
        "candidate_count_matches": result["shortlisted_count"] == len(candidates),
        "bounded_candidate_count": len(candidates) <= (4 if learned else 12),
        "candidate_poses_rigid": all(pose_valid(c["world_T_grip"]) for c in candidates),
        "candidate_depths_in_common_grid": all(round(c["pad_depth_m"], 6) in DEPTHS_M for c in candidates),
        "candidate_widths_within_gripper": all(.002 < c["width_m"] < .075 and
            0. < c["opening_m"] <= .08 for c in candidates),
        "candidate_table_clearance": all(c["table_clearance_m"] >= .0001 for c in candidates),
        "candidate_list_preserved": result["shortlisted_grasps"] == candidates,
    }
    if learned:
        raw = result["raw_network_proposals"]
        checks["at_most_four_network_peaks"] = len(raw) <= 4
        checks["no_CAD_pose_for_learned_generation"] = result["object_pose_used_for_grasp_generation"] is False
        checks["estimated_points_not_claimed_surface_contacts"] = all(
            c["predicted_endpoints_are_actual_surface_contacts"] is False for c in candidates)
        checks["depth_geometry_not_CAD_pose"] = all(c["depth_geometry"]["object_pose_used"] is False
            for c in candidates)
        checks["unique_network_candidate_ids"] = len({c["network_candidate_id"] for c in candidates}) == len(candidates)
        accounted = [c["network_candidate_id"] for c in candidates] + [
            c["network_candidate_id"] for c in result["proposal_mapping_rejections"]]
        checks["every_network_peak_retained_or_explained"] = sorted(accounted) == sorted(
            p["network_candidate_id"] for p in raw)
    for c in candidates:
        pose, points = c["world_T_grip"], c["contacts_world_m"]
        middle = [(a+b)/2 for a, b in zip(*points)]
        predicted = [pose[i][3]+pose[i][2]*c["pad_depth_m"] for i in range(3)]
        checks.setdefault("centres_and_depths_reproduce", True)
        checks["centres_and_depths_reproduce"] &= max(abs(a-b) for a, b in zip(middle, predicted)) < 1e-7
        checks.setdefault("spans_reproduce", True)
        checks["spans_reproduce"] &= abs(math.dist(*points)-c["width_m"]) < 1e-7
    ordering = result["candidate_check_order"]
    if result.get("vlm_status") not in ("invalid_output", "vlm_output_unavailable") and result["status"] != "vlm_output_unavailable":
        checks["candidate_check_order_is_permutation"] = sorted(ordering) == list(range(len(candidates)))
    rejected = result["planning_rejections"]
    selected_index = result.get("selected_pool_index")
    checks["path_check_count_reproduces"] = result["path_candidates_checked"] == len(rejected)+int(selected_index is not None)
    checks["rejected_paths_follow_order"] = [x["pool_index"] for x in rejected] == ordering[:len(rejected)]
    checks["rejected_paths_have_failure"] = all(any(c["status"] != "passed" for c in item["checks"]) for item in rejected)
    if selected_index is not None:
        selected = result["selected_grasp"]
        checks["first_passing_candidate_selected"] = selected_index == ordering[len(rejected)]
        checks["selected_candidate_from_shown_pool"] = all(selected[key] == value for key, value in candidates[selected_index].items()
            if key in selected)
        checks["selected_path_three_pickup_steps_passed"] = [c["phase"] for c in result["path_checks"]] == ["approach", "descend", "lift"] and all(
            c["status"] == "passed" for c in result["path_checks"])
    if result["arm"] == "grconvnet_vlm":
        if not candidates:
            checks["no_VLM_call_without_candidates"] = result["vlm_status"] == "skipped_no_candidates" and not (Path(folder)/"vlm/request.json").exists()
        else:
            mapping = read(Path(folder)/"display_id_map.json")
            request = read(Path(folder)/"vlm/request.json")
            ranking = read(Path(folder)/"vlm/ranking.json")
            ids = list(range(1, len(candidates)+1))
            checks["display_ids_cover_pool"] = sorted(map(int, mapping)) == ids and sorted(x["pool_index"] for x in mapping.values()) == list(range(len(candidates)))
            checks["display_map_exact_poses"] = all(item["world_T_grip"] == candidates[item["pool_index"]]["world_T_grip"] and
                item["pad_depth_m"] == candidates[item["pool_index"]]["pad_depth_m"] for item in mapping.values())
            checks["VLM_receives_same_candidate_ids"] = sorted(request["allowed_ids"]) == ids
            checks["VLM_input_image_hash"] = request["image_sha256"] == digest(Path(folder)/"rank_input.png") == ranking["image_sha256"]
            checks["VLM_prompt_frozen"] = request["prompt_file_sha256"] == digest(Path(__file__).parent/"grasp_rank_prompt.txt")
            checks["VLM_prompt_text_exact"] = request["prompt"] == (Path(__file__).parent/"grasp_rank_prompt.txt").read_text().strip()+"\nSupplied IDs: "+json.dumps(request["allowed_ids"])
            checks["no_goal_or_outcome_fields_in_VLM_request"] = set(request) == {
                "image_file", "image_sha256", "allowed_ids", "prompt", "prompt_file_sha256"}
            checks["VLM_status_reproduces"] = ranking["status"] == result["vlm_status"]
            if ranking["status"] == "ok":
                parsed = json.loads(ranking["raw_text"])
                ranked_ids = parsed["ranked_ids"]
                checks["VLM_raw_text_retained"] = ranking["raw_text"] == ranking["raw_response"]["message"]["content"]
                checks["VLM_completion_finished"] = ranking["raw_response"].get("done") is True and ranking["raw_response"].get("done_reason") == "stop" and not ranking["raw_response"]["message"].get("thinking")
                checks["VLM_ranking_is_complete_permutation"] = all(type(x) is int for x in ranked_ids) and sorted(ranked_ids) == ids
                observations = parsed["observations"]
                checks["VLM_observations_cover_supplied_ids"] = len(observations) == len(ids) and sorted(x["id"] for x in observations) == ids and all(
                    type(x["id"]) is int and isinstance(x["observation"],str) and bool(x["observation"].strip()) for x in observations)
                checks["VLM_ranking_retained_verbatim"] = ranked_ids == result["vlm_ranked_display_ids"] == ranking["ranked_ids"]
                checks["VLM_order_controls_path_checks"] = ordering == [mapping[str(x)]["pool_index"] for x in ranked_ids]
            else:
                checks["invalid_VLM_answer_not_silently_replaced"] = ordering == [] and result["physical_attempts"] == 0 and result["status"] == "vlm_output_unavailable"
    return checks


def audit_run(run):
    run = Path(run).resolve()
    manifest = read(run/"manifest.json")
    manifest_checks = {
        "all_fourteen_parts": manifest["parts"] == list(PARTS),
        "three_methods": manifest["arms"] == list(ARMS),
        "common_six_depths": set(manifest["finger_depths_m"]) == DEPTHS_M,
        "pickup_lift_18cm": manifest["trial"]["lift_command_m"] == .18,
        "hold_one_second": manifest["trial"]["hold_after_lift_s"] == 1.,
        "pass_lift_12cm": manifest["trial"]["min_lift_m"] == .12,
        "one_physical_attempt": manifest["trial"]["physical_attempt_limit_per_part_arm"] == 1,
        "robot_overlap_limit_1mm": manifest["trial"]["robot_overlap_limit_m"] == .001,
        "no_final_assembly_views_or_goals": manifest["input_conditions"]["final_assembly_views_or_goals"] is False,
        "learned_method_no_CAD_pose": manifest["input_conditions"]["learned_generation_uses_CAD_pose"] is False,
        "source_scene_unchanged": manifest["source_scene_sha256"] == digest(SOURCE),
        "grasp_prompt_unchanged": manifest["prompt_sha256"] == digest(Path(__file__).parent/"grasp_rank_prompt.txt"),
    }
    for name in ("run_trial.py", "capture.py", "grconvnet_adapter.py", "vlm_client.py"):
        manifest_checks["runtime_source_unchanged_"+name] = manifest["source_hashes"][name] == digest(Path(__file__).parent/name)
    model = manifest["model_provenance"]
    model_path = ROOT/model["checkpoint"]["relative_path"]
    manifest_checks["pretrained_checkpoint_hash"] = digest(model_path) == model["checkpoint"]["sha256"]
    for name, source in model["source_files"].items():
        manifest_checks["pinned_model_source_"+name] = digest(Path(__file__).parent/"vendor/grconvnet_source"/name) == source["sha256"]
    records, rows = {}, []
    for arm in ARMS:
        for part in PARTS:
            folder = run/arm/part
            result = read(folder/"result.json")
            records[arm, part] = result
            row = replay_pickup(folder, result)
            checks = row["checks"]
            checks["part_and_arm_match"] = result["part"] == part and result["arm"] == arm
            checks["no_training"] = result["training_performed"] is False
            capture = read(folder/"capture/capture_audit.json")
            calibration = read(folder/"capture/calibration.json")
            checks["capture_no_physics_or_pose_change"] = capture["physics_steps"] == 0 and capture["qpos_unchanged"] is True
            checks["visible_target_mask_available"] = capture["target_pixel_count"] > 0
            checks["capture_no_object_pose"] = calibration["object_pose_used_for_camera_or_grasp_generation"] is False
            checks["camera_transform_rigid"] = pose_valid(calibration["world_T_camera"])
            checks["metric_optical_depth"] = calibration["depth_units"] == "metres; optical camera Z, not ray range"
            checks.update(audit_candidates(folder, result))
            timing = {**result["stage_seconds"], "setup_seconds": result["setup_seconds"],
                      "decision_seconds": result["decision_seconds"], "planning_seconds_including_setup": result["planning_seconds_including_setup"],
                      "total_seconds": result["total_seconds"]}
            checks["timings_finite_nonnegative"] = all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in timing.values())
            checks["decision_covers_recorded_stages"] = result["decision_seconds"]+1e-9 >= sum(result["stage_seconds"].values())
            checks["setup_and_decision_within_planning"] = result["planning_seconds_including_setup"]+1e-9 >= result["setup_seconds"]+result["decision_seconds"]
            checks["planning_within_total_time"] = result["total_seconds"] >= result["planning_seconds_including_setup"]
            checks["fresh_generation_not_reused"] = result["shared_pool_cost_reused"] is False
            row.update(part=part,arm=arm,passed=all(checks.values()),timing=timing,
                       candidate_count=len(result["candidates"]),reported_status=result["status"])
            rows.append(row)
    matched = []
    for part in PARTS:
        part_rows = [row for row in rows if row["part"] == part]
        matched.append({"part":part,
            "same_initial_target_and_robot": all(row["initial_qpos"] == part_rows[0]["initial_qpos"] for row in part_rows),
            "same_RGBD_and_target_mask": all(len({digest(run/arm/part/"capture"/name) for arm in ARMS}) == 1
                for name in ("rgb.png", "depth_m.npy", "target_mask.npy", "calibration.json")),
            "same_learned_candidate_pool": records["grconvnet_score",part]["candidates"] == records["grconvnet_vlm",part]["candidates"],
            "same_learned_network_peaks": records["grconvnet_score",part]["raw_network_proposals"] == records["grconvnet_vlm",part]["raw_network_proposals"]})
    protected = protected_changes()
    summary = {arm:{"pickup_passed":sum(row["pickup_success_replayed"] for row in rows if row["arm"] == arm),
                    "part_count":len(PARTS),
                    "total_decision_seconds":sum(row["timing"]["decision_seconds"] for row in rows if row["arm"] == arm),
                    "median_decision_seconds":statistics.median(row["timing"]["decision_seconds"] for row in rows if row["arm"] == arm)}
               for arm in ARMS}
    value = {"passed":all(row["passed"] for row in rows) and all(manifest_checks.values()) and protected["passed"] and
                        all(all(value for key,value in match.items() if key != "part") for match in matched),
             "part_records":rows,"manifest_checks":manifest_checks,"matched_conditions":matched,
             "protected_files":protected,"replayed_summary":summary,
             "scope":"Saved-record, force, JSON, camera and source-hash replay. No new inference, physics or hardware commands."}
    (run/"independent_audit.json").write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"passed":value["passed"],"records":len(rows),"protected_files":protected,
        "failed_manifest_checks":[name for name,passed in manifest_checks.items() if not passed],
        "failed_part_checks":[{"arm":r["arm"],"part":r["part"],"checks":[k for k,v in r["checks"].items() if not v]} for r in rows if not r["passed"]],
        "matched_conditions":matched,"summary":summary},indent=2))
    return value


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "snapshot":
        snapshot_protected()
    else:
        audit_run(Path(sys.argv[1]))
