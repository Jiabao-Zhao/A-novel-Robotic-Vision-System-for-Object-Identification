"""Read-only independent checks of saved small/large pickup comparisons.

Uses only the standard library: no grasp planner, simulator or controller runs.
Usage: python -m experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.audit RUN_DIRECTORY
"""
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/runtime_antipodal_assembly_20261023"
LEGACY = BASE / "20261023T213101Z"
SOURCE = BASE / "input_scene/scene.xml"
PARTS = ("Gear_Large", "Gear_Medium", "Gear_Small", "Waterproof_Male", "M12_Hex_Nut",
         "M16_Hex_Nut", "RGOCG16-50_16mm", "KET16_Square_16mm", "RGOCG12-50_12mm",
         "RGOCG8-50_8mm", "KET12_Square_12mm", "KET8_Square_8mm", "USB_Male", "DSUB_Male")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz_array(archive, name):
    """Read this trial's basic numeric .npy records without importing NumPy."""
    raw = archive.read(name + ".npy")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("Not a NumPy record")
    major = raw[6]
    size_bytes = 2 if major == 1 else 4
    header_size = struct.unpack("<H" if size_bytes == 2 else "<I", raw[8:8+size_bytes])[0]
    start = 8 + size_bytes
    info = ast.literal_eval(raw[start:start+header_size].decode("latin1"))
    if info["fortran_order"]:
        raise ValueError("Fortran-order array unsupported in this independent audit")
    kind = {"<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i"}[info["descr"]]
    values = array.array(kind)
    values.frombytes(raw[start+header_size:])
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == math.prod(info["shape"])
    return values, info["shape"]


def pose_valid(pose):
    if len(pose) != 4 or any(len(row) != 4 for row in pose):
        return False
    if not all(math.isfinite(value) for row in pose for value in row):
        return False
    if max(abs(a - b) for a, b in zip(pose[3], (0, 0, 0, 1))) > 1e-8:
        return False
    for i in range(3):
        for j in range(3):
            value = sum(pose[k][i] * pose[k][j] for k in range(3))
            if abs(value - (1 if i == j else 0)) > 1e-5:
                return False
    r = pose
    determinant = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                   - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                   + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    return abs(determinant - 1) <= 1e-5


def contact_sides(frame):
    """Rebuild bilateral finger contact from individual recorded forces."""
    sides = set()
    for contact in frame["contacts"]:
        if contact["kind"] != "finger_target" or contact["normal_force_n"] <= .02:
            continue
        for body in contact["bodies"]:
            if "leftfinger" in body or "finger_joint1" in body:
                sides.add("left")
            if "rightfinger" in body or "finger_joint2" in body:
                sides.add("right")
    return sorted(sides)


def phase_last(states, phase):
    return next((state for state in reversed(states) if state["phase"] == phase), None)


def legacy_report():
    rows = []
    for part in PARTS:
        folder = LEGACY / part
        result = read(folder / "result.json")
        states = read(folder / "trajectory.json")
        lift = phase_last(states, "lift")
        initial_z = states[0]["target_pose_audit"][2][3]
        lift_m = None if lift is None else lift["target_pose_audit"][2][3] - initial_z
        passed = lift is not None and lift_m >= .12 and len(lift["grasp_contacts"]) == 2
        rows.append({"part": part, "pickup_end_of_lift_passed": passed,
                     "lift_m": lift_m, "planning_seconds": result["planning_seconds"],
                     "sampling_seconds": result["sampling_seconds"],
                     "generated_gripper_poses": result["generated_gripper_poses"],
                     "last_phase": states[-1]["phase"],
                     "post_lift_hold_phase_present": phase_last(states, "hold_lift") is not None})
    return {"parts": rows, "passed_end_of_lift": sum(row["pickup_end_of_lift_passed"] for row in rows),
            "median_planning_seconds": statistics.median(row["planning_seconds"] for row in rows),
            "median_sampling_seconds": statistics.median(row["sampling_seconds"] for row in rows),
            "historical_caveat": "Old timing includes assembly-route searches. Old pickup checked the end of lift without the new one-second hold. Old four-depth search omitted +6/+8 mm; other loose parts were hidden. Matched new large/small pickup-only runs are the direct budget comparison."}


def protected_files():
    experiment = ROOT / "experiments/runtime_antipodal_assembly_20261023"
    candidates = [*LEGACY.rglob("*"),
                  *(BASE / "8mm_runtime_followup/20261024T012239Z").rglob("*"),
                  SOURCE, experiment / "grasp_planning.py", experiment / "robot_trial.py",
                  experiment / "run_experiment.py", experiment / "vendor/antipodal_sampler.py",
                  ROOT / "docs/cad_view_prompt.txt"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in candidates if path.is_file()}


def snapshot_protected():
    output = BASE / "smaller_search/protected_before_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(protected_files(), indent=2) + "\n", encoding="utf-8")
    return output


def scene_checks(path, part):
    """Inspect saved XML independently of MuJoCo's model loader."""
    root = ET.parse(path).getroot()
    original = ET.parse(SOURCE).getroot()
    originals = {geom.get("name"): geom for geom in original.findall("worldbody//geom")}
    target_name = "nist_part_" + part
    visible, noncolliding, physical_target = [], [], []
    for body in root.findall("worldbody//body"):
        name = body.get("name", "")
        if not name.startswith("nist_part_"):
            continue
        geoms = body.findall(".//geom")
        if name != target_name:
            noncolliding.append(all(geom.get("contype", "1") == "0" and
                                    geom.get("conaffinity", "1") == "0" for geom in geoms))
            visible.append(all(geom.get("rgba") == originals[geom.get("name")].get("rgba")
                               for geom in geoms))
        else:
            physical_target.append(any(int(geom.get("contype", "1")) or
                                       int(geom.get("conaffinity", "1")) for geom in geoms))
    # MuJoCo stores joint qpos in XML body traversal order. Only free joints on
    # other parts are needed here, so derive their offsets from explicit types.
    offsets = {}
    offset = 0
    for body in root.findall("worldbody//body"):
        for joint in body:
            if joint.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
            size = {"free": 7, "ball": 4}.get(kind, 1)
            if body.get("name", "").startswith("nist_part_"):
                offsets[body.get("name")] = (offset, size)
            offset += size
    return {"no_attachment_constraints": not root.findall("equality/*"),
            "thirteen_other_parts_noncolliding": len(noncolliding) == 13 and all(noncolliding),
            "thirteen_other_parts_visible": len(visible) == 13 and all(visible),
            "target_has_contact_geometry": len(physical_target) == 1 and all(physical_target)}, offsets


def audit_part(run, arm, part):
    folder = run / arm / part
    result = read(folder / "result.json")
    states = read(folder / "trajectory.json")
    contacts = read(folder / "contacts.json")
    audit = read(folder / "controller_audit.json")
    checks, offsets = scene_checks(folder / "scene.xml", part)
    checks["result_part"] = result["part"] == part
    checks["initial_pose_valid"] = pose_valid(result["world_T_cad_input"])
    checks["initial_pose_matches_trajectory"] = result["world_T_cad_input"] == states[0]["target_pose_audit"]
    checks["no_hidden_parts_in_controller_record"] = audit["other_loose_parts_hidden_and_noncolliding"] == []
    checks["no_attachments_in_controller_record"] = audit["attachments_present"] is False
    checks["no_target_feedback"] = audit["target_feedback_in_controller"] is False
    checks["same_contact_parameters_as_original"] = audit["common_contact_settings"] == read(LEGACY / part / "controller_audit.json")["common_contact_settings"]
    checks["no_friction_change"] = audit["friction_coefficients_changed"] is False
    checks["scene_source_matches_manifest"] = result["source_scene_sha256"] == digest(SOURCE) == audit["source_scene_sha256"]
    checks["mesh_hash_matches_native_cad"] = result["mesh_sha256"] == digest(ROOT / "outputs/simulation/nist_task_board_1/assets/meshes" / (part + "_m.stl"))
    checks["time_monotonic"] = all(b["time_s"] >= a["time_s"] for a, b in zip(states, states[1:]))
    checks["trajectory_contact_frame_count"] = len(states) == len(contacts) == audit["trajectory_frames"]
    checks["trajectory_contacts_replay"] = all(state["grasp_contacts"] == contact_sides(frame)
                                                for state, frame in zip(states, contacts))
    frozen = []
    gravity_shift_m = -9.81 * audit["physics_timestep_s"] ** 2
    max_other_part_recorded_shift_m = 0.
    for name, (first, size) in offsets.items():
        if name == "nist_part_" + part:
            continue
        original = states[0]["qpos"][first:first+size]
        expected = original.copy()
        expected[2] += gravity_shift_m
        # Existing controller resets inactive qpos/velocity each substep, then
        # lets the final integration run before recording. The analytical
        # semi-implicit gravity offset is g*dt^2; it must not accumulate.
        frozen.append(all(max(abs(a-b) for a, b in zip(state["qpos"][first:first+size], expected)) < 1e-10
                          for state in states[1:]))
        max_other_part_recorded_shift_m = max(max_other_part_recorded_shift_m,
            *(math.dist(state["qpos"][first:first+3], original[:3]) for state in states))
    checks["thirteen_other_parts_reset_with_exact_single_step_gravity_offset"] = len(frozen) == 13 and all(frozen)
    initial_z = states[0]["target_pose_audit"][2][3]
    final_lift = states[-1]["target_pose_audit"][2][3] - initial_z
    final_bilateral = len(contact_sides(contacts[-1])) == 2
    final_hold = states[-1]["phase"] == "hold_lift"
    collision_ok = max(audit["max_penetration_m"][key]
                       for key in ("robot_self", "robot_table", "robot_board")) < .001
    passed = final_hold and final_lift >= .12 and final_bilateral and collision_ok
    checks["pickup_success_replays"] = passed == (result["status"] == "pickup_passed")
    if "lift_m" in result:
        checks["lift_value_replays"] = abs(result["lift_m"] - final_lift) < 1e-12
    shortlist = result["shortlisted_grasps"]
    checks["shortlist_count_matches_record"] = len(shortlist) == result["shortlisted_count"]
    checks["shortlist_fits_arm_budget"] = len(shortlist) <= (12 if arm == "small" else 48)
    checks["shortlist_unique_ids"] = len({item["candidate_id"] for item in shortlist}) == len(shortlist)
    checks["shortlist_poses_rigid"] = all(pose_valid(item["world_T_grip"]) for item in shortlist)
    checks["shortlist_table_clear"] = all(item["table_clearance_m"] >= .0001 for item in shortlist)
    checks["shortlist_finger_depths_allowed"] = all(round(item["pad_depth_m"], 6) in
                                                     (.004, 0., -.004, -.008, .006, .008) for item in shortlist)
    checks["shortlist_jaw_widths_allowed"] = all(.002 < item["width_m"] < .075 and
                                                 0 < item["opening_m"] <= .08 for item in shortlist)
    for item in shortlist:
        pose, points = item["world_T_grip"], item["contacts_world_m"]
        midpoint = [(a + b) / 2 for a, b in zip(*points)]
        expected = [pose[i][3] + pose[i][2] * item["pad_depth_m"] for i in range(3)]
        checks.setdefault("contacts_reconstruct_grasp_center", True)
        checks["contacts_reconstruct_grasp_center"] &= max(abs(a-b) for a, b in zip(midpoint, expected)) < 1e-7
        width = math.dist(*points)
        checks.setdefault("contact_span_matches_width", True)
        checks["contact_span_matches_width"] &= abs(width - item["width_m"]) < 1e-7
    selected = result.get("selected_shortlist_index")
    if selected is not None:
        chosen = shortlist[selected]
        checks["selected_pose_from_shortlist"] = chosen["world_T_grip"] == result["selected_grasp"]["world_T_grip"]
        checks["selected_path_passed"] = chosen["path_check_status"] == "passed"
        checks["selected_is_first_passing_path"] = all(item["path_check_status"] == "failed" for item in shortlist[:selected])
        checks["later_paths_not_checked"] = all(item["path_check_status"] == "not_checked" for item in shortlist[selected+1:])
    checks["path_check_count_replays"] = sum(item["path_check_status"] != "not_checked" for item in shortlist) == result["path_candidates_checked"]
    checks["at_most_one_physical_attempt"] = result["physical_attempts"] in (0, 1)
    checks["no_assembly_attempt"] = result["assembly_attempted"] is False
    timing = {key: result[key] for key in ("sampling_seconds", "frame_generation_seconds", "table_check_seconds",
                                         "shortlist_seconds", "candidate_generation_and_filter_seconds",
                                         "robot_path_check_seconds", "search_seconds", "setup_seconds",
                                         "planning_seconds_including_setup", "total_seconds")}
    checks["timings_finite_nonnegative"] = all(math.isfinite(value) and value >= 0 for value in timing.values())
    generation_sum = sum(timing[key] for key in ("sampling_seconds", "frame_generation_seconds", "table_check_seconds", "shortlist_seconds"))
    generation_gap = timing["candidate_generation_and_filter_seconds"] - generation_sum
    search_gap = timing["search_seconds"] - timing["candidate_generation_and_filter_seconds"] - timing["robot_path_check_seconds"]
    setup_gap = timing["planning_seconds_including_setup"] - timing["setup_seconds"] - timing["search_seconds"]
    checks["generation_time_covers_stages"] = generation_gap >= -1e-6
    checks["search_time_covers_generation_and_paths"] = search_gap >= -1e-6
    checks["planning_time_covers_setup_and_search"] = setup_gap >= -1e-6
    checks["total_time_covers_planning"] = timing["total_seconds"] >= timing["planning_seconds_including_setup"]
    with zipfile.ZipFile(folder / "generated_candidates.npz") as archive:
        pose_array, pose_shape = npz_array(archive, "world_T_grip")
        pair_array, pair_shape = npz_array(archive, "contacts_world_m")
        depths, _ = npz_array(archive, "pad_depth_m")
        scores, _ = npz_array(archive, "score")
        openings, _ = npz_array(archive, "opening_m")
        clearances, _ = npz_array(archive, "table_clearance_m")
        ids, _ = npz_array(archive, "shortlisted_candidate_ids")
        checks["saved_generated_pose_count"] = pose_shape == (result["generated_gripper_poses"], 4, 4)
        checks["saved_contact_pair_shape"] = pair_shape == (result["generated_gripper_poses"], 2, 3)
        checks["shortlist_ids_match_generated_archive"] = list(ids) == [item["candidate_id"] for item in shortlist]
        archive_matches = []
        for item in shortlist:
            index = item["candidate_id"]
            archive_matches.append(list(pose_array[index*16:(index+1)*16]) == [value for row in item["world_T_grip"] for value in row]
                                   and list(pair_array[index*6:(index+1)*6]) == [value for row in item["contacts_world_m"] for value in row]
                                   and depths[index] == item["pad_depth_m"] and scores[index] == item["score"]
                                   and openings[index] == item["opening_m"] and clearances[index] == item["table_clearance_m"])
        checks["shortlisted_data_exactly_from_generated_archive"] = all(archive_matches)
        checks["all_six_finger_depths_present_in_generated_pool"] = set(round(value, 6) for value in depths) == {.004, 0., -.004, -.008, .006, .008}
    return {"part": part, "arm": arm, "checks": checks,
            "passed": all(checks.values()), "pickup_success_replayed": passed,
            "final_lift_m": final_lift, "final_bilateral_contact": final_bilateral,
            "last_phase": states[-1]["phase"], "shortlist_count": len(shortlist),
            "contact_frame_count": len(contacts), "timing": timing,
            "inactive_part_gravity_offset_m": gravity_shift_m,
            "maximum_inactive_part_recorded_translation_from_start_m": max_other_part_recorded_shift_m,
            "unassigned_timing_overhead_seconds": {"generation": generation_gap, "search": search_gap, "setup": setup_gap},
            "generated_pose_count": result["generated_gripper_poses"],
            "path_candidates_checked": result["path_candidates_checked"]}


def audit_run(run):
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    rows = [audit_part(run, arm, part) for arm in ("small", "large_control") for part in PARTS]
    previous = BASE / "smaller_search/protected_before_audit.json"
    protected = read(previous)
    changed = [path for path, value in protected.items() if not (ROOT/path).is_file() or digest(ROOT/path) != value]
    matched = []
    for part in PARTS:
        a = read(run / "small" / part / "result.json")
        b = read(run / "large_control" / part / "result.json")
        matched.append({"part": part, "same_initial_pose": a["world_T_cad_input"] == b["world_T_cad_input"],
                        "same_initial_robot_and_scene": read(run / "small" / part / "trajectory.json")[0]["qpos"] ==
                                                         read(run / "large_control" / part / "trajectory.json")[0]["qpos"]})
    manifest_checks = {
        "all_fourteen_parts": list(PARTS) == manifest["parts"],
        "same_six_finger_depths": sorted(manifest["finger_depths_m"]) == sorted([.004, 0., -.004, -.008, .006, .008]),
        "small_budgets": manifest["small"] == {"surface_rays": 512, "contact_pairs": 24, "shortlist": 12},
        "large_budgets": manifest["large_control"] == {"surface_rays": 2048, "contact_pairs": 320, "shortlist": 48},
        "same_lift_command": manifest["pickup_lift_command_m"] == .18,
        "single_attempt_rule": manifest["maximum_physical_attempts_per_part_per_arm"] == 1,
        "source_scene_hash": manifest["source_scene_sha256"] == digest(SOURCE),
        "runtime_search_source_unchanged_since_run_start": manifest["source_hashes"]["search.py"] == digest(Path(__file__).parent / "search.py"),
        "runtime_runner_source_unchanged_since_run_start": manifest["source_hashes"]["run_comparison.py"] == digest(Path(__file__).parent / "run_comparison.py"),
    }
    summaries = {}
    for arm in ("small", "large_control"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "pickup_passed": sum(row["pickup_success_replayed"] for row in arm_rows),
            "median_search_seconds": statistics.median(row["timing"]["search_seconds"] for row in arm_rows),
            "total_search_seconds": sum(row["timing"]["search_seconds"] for row in arm_rows),
            "median_setup_inclusive_planning_seconds": statistics.median(row["timing"]["planning_seconds_including_setup"] for row in arm_rows),
            "total_generated_poses": sum(row["generated_pose_count"] for row in arm_rows),
            "total_shortlist_poses": sum(row["shortlist_count"] for row in arm_rows),
        }
    value = {"passed": all(row["passed"] for row in rows) and not changed and
                       all(manifest_checks.values()) and
                       all(row["same_initial_pose"] and row["same_initial_robot_and_scene"] for row in matched),
             "parts": rows, "matched_start_conditions": matched,
             "manifest_checks": manifest_checks, "replayed_summaries": summaries,
             "protected_file_count": len(protected), "protected_changed": changed,
             "legacy": legacy_report(),
             "scope": "Saved-record and XML audit. No new physics, grasp generation or hardware commands."}
    (run / "independent_audit.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": value["passed"], "part_records": len(rows),
                      "protected_files": len(protected), "protected_changed": changed,
                      "failed_checks": [{"part": row["part"], "arm": row["arm"],
                                         "checks": [key for key, passed in row["checks"].items() if not passed]}
                                        for row in rows if not row["passed"]]}, indent=2))
    return value


def audit_contact_surfaces(run):
    """Check every saved shortlisted touch against native CAD, without physics.

    This optional geometry pass uses the existing NumPy/Open3D environment.
    It imports neither the sampler nor the controller.
    """
    import numpy as np
    import open3d as o3d

    run = Path(run).resolve()
    rows = []
    for part in PARTS:
        path = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes" / (part + "_m.stl")
        mesh = o3d.io.read_triangle_mesh(str(path))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        for arm in ("small", "large_control"):
            result = read(run / arm / part / "result.json")
            initial = np.asarray(result["world_T_cad_input"])
            shortlist = result["shortlisted_grasps"]
            points = np.asarray([item["contacts_world_m"] for item in shortlist])
            local = (points.reshape(-1, 3)-initial[:3, 3]) @ initial[:3, :3]
            distances = scene.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy()
            midpoint_error, line_error = [], []
            for item in shortlist:
                pose = np.asarray(item["world_T_grip"])
                touches = np.asarray(item["contacts_world_m"])
                grip_local = (touches - pose[:3, 3]) @ pose[:3, :3]
                midpoint_error.append(float(np.linalg.norm(grip_local.mean(axis=0) - [0, 0, item["pad_depth_m"]])))
                line_error.append(float(max(abs(grip_local[:, 1]).max(),
                                             abs(grip_local[:, 2] - item["pad_depth_m"]).max(),
                                             abs(abs(grip_local[:, 0]) - item["width_m"]/2).max())))
            row = {"part": part, "arm": arm, "shortlisted_poses": len(shortlist),
                   "touch_points": len(local), "max_contact_to_native_cad_surface_m": float(distances.max()),
                   "max_grasp_center_error_m": max(midpoint_error),
                   "max_gripper_contact_line_error_m": max(line_error)}
            # Distances are numerical diagnostics, not a new fitted acceptance
            # gate. Pickup success is independently evaluated from physics.
            row["gripper_contact_line_checks_passed"] = max(midpoint_error) < 1e-7 and max(line_error) < 1e-7
            rows.append(row)
    value = {"gripper_contact_line_checks_passed": all(row["gripper_contact_line_checks_passed"] for row in rows), "parts": rows,
             "poses_checked": sum(row["shortlisted_poses"] for row in rows),
             "contact_points_checked": sum(row["touch_points"] for row in rows),
             "max_contact_to_native_cad_surface_m": max(row["max_contact_to_native_cad_surface_m"] for row in rows),
             "surface_distance_status": "Numerical distances reported without an acceptance threshold; these do not change pickup outcomes.",
             "scope": "All shortlisted poses in both arms; native CAD surface distance and contact-to-gripper geometry only. No physics or reachability rerun."}
    (run / "independent_surface_audit.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value[key] for key in ("gripper_contact_line_checks_passed", "poses_checked", "contact_points_checked", "max_contact_to_native_cad_surface_m")}, indent=2))
    return value


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "surfaces":
        audit_contact_surfaces(sys.argv[1])
    else:
        audit_run(sys.argv[1])
