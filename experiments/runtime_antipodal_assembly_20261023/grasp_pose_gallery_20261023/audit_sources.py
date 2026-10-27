"""Audit planned grasp records used for the 14-part pose gallery.

No trajectory frame, closed-gripper snapshot, simulator, or renderer is used.
Thirteen poses are from the original trial; the 8 mm pin pose is from its
separately labeled automatic runtime follow-up with additional depth samples.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
OUTPUT_ROOT = ROOT / "outputs/runtime_antipodal_assembly_20261023"
ORIGINAL = OUTPUT_ROOT / "20261023T213101Z"
FOLLOWUP = OUTPUT_ROOT / "8mm_runtime_followup/20261024T012239Z"
OUTPUT = OUTPUT_ROOT / "grasp_pose_gallery_20261023/source_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def source_hash_audit(manifest):
    records = []
    for relative, expected in manifest["source_hashes"].items():
        path = EXPERIMENT / relative
        actual = sha(path) if path.is_file() else None
        records.append({"path": str(path), "expected_sha256": expected,
                        "current_sha256": actual, "matches": actual == expected})
    return {"all_recorded_sources_match": all(record["matches"] for record in records),
            "records": records}


def main():
    manifest = read(ORIGINAL / "manifest.json")
    followup_manifest = read(FOLLOWUP / "manifest.json")
    frozen_audit = read(ORIGINAL / "evaluation_manifest.json")
    rows = []
    for name in manifest["parts"]:
        is_followup = name == "RGOCG8-50_8mm"
        original_result = ORIGINAL / name / "result.json"
        path = FOLLOWUP / "result.json" if is_followup else original_result
        result = read(path)
        grasp = result["selected_grasp"]
        pose = np.asarray(grasp["world_T_grip"], dtype=float)
        object_pose = np.asarray(result["world_T_cad_input"], dtype=float)
        points = np.asarray(grasp["contacts_world_m"], dtype=float)
        assert pose.shape == (4, 4) and object_pose.shape == (4, 4) and points.shape == (2, 3)
        assert np.all(np.isfinite(pose)) and np.all(np.isfinite(points))
        assert np.allclose(pose[3], [0, 0, 0, 1])
        assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-6)
        assert abs(np.linalg.det(pose[:3, :3]) - 1) < 2e-6
        midpoint = points.mean(axis=0)
        point_separation = np.linalg.norm(points[1] - points[0])
        local_midpoint = pose[:3, :3].T @ (midpoint - pose[:3, 3])
        assert abs(point_separation - grasp["width_m"]) < 2e-7
        assert np.allclose(local_midpoint, [0, 0, grasp["pad_depth_m"]], atol=2e-7)
        assert np.allclose(pose, result["plan"]["grasp"], atol=1e-12)
        object_T_grip = np.linalg.inv(object_pose) @ pose
        if "object_T_grip" in result["plan"]:
            assert np.allclose(object_T_grip, result["plan"]["object_T_grip"], atol=2e-7)
        original_record = read(original_result)
        original_hash_expected = frozen_audit["input_hashes_by_part"][name]["result.json"]
        assert sha(original_result) == original_hash_expected
        full_goal = bool(result.get("assembly_planning_status") == "sampled_kinematic_path_found")
        source_tag = "separate_8mm_automatic_runtime_followup" if is_followup else "original_14part_trial"
        rows.append({"part": name, "source_label": source_tag,
            "display_label_suffix": "8 mm automatic follow-up: extra generic depth samples" if is_followup else None,
            "result_path": str(path), "result_sha256": sha(path),
            "pose_source_json_pointer": "/selected_grasp/world_T_grip",
            "contact_source_json_pointer": "/selected_grasp/contacts_world_m",
            "object_pose_source_json_pointer": "/world_T_cad_input",
            "planned_world_T_grip": pose.tolist(),
            "world_T_cad_input": object_pose.tolist(),
            "cad_T_grip": object_T_grip.tolist(),
            "contacts_world_m": points.tolist(),
            "contacts_cad_m": ((points - object_pose[:3, 3]) @ object_pose[:3, :3]).tolist(),
            "planned_contact_width_m": grasp["width_m"], "pregrasp_opening_m": grasp["opening_m"],
            "roll_deg": grasp["roll_deg"], "selected_contact_depth_m": grasp["pad_depth_m"],
            "contact_depth_source": "/selected_grasp/pad_depth_m",
            "depth_generation": "additional generic +6/+8 mm full-finger sampling; selected +8 mm" if is_followup
                                else "original generic +4/0/-4/-8 mm pad-depth sampling",
            "table_clearance_m": grasp["table_clearance_m"],
            "selected_candidate_index": result.get("selected_table_clear_candidate_index", result.get("selected_candidate_index")),
            "planning_scope": "full_nominal_goal_path" if full_goal else "pickup_only",
            "raw_execution_status": result["status"],
            "original_result_path": str(original_result),
            "original_result_sha256": sha(original_result),
            "original_result_unchanged_since_evaluation": True,
            "original_run_had_selected_grasp": bool(original_record.get("selected_grasp")),
            "original_run_status": original_record["status"],
            "planned_pose_matches_saved_plan_grasp": True,
            "width_matches_contact_separation": True,
            "midpoint_depth_matches_saved_pose": True,
            "uses_executed_closed_gripper_pose": False})
    assert len(rows) == 14 and len({row["part"] for row in rows}) == 14
    assert sum(row["source_label"] == "original_14part_trial" for row in rows) == 13
    assert sha(ORIGINAL / "RGOCG8-50_8mm/result.json") == followup_manifest["frozen_result_sha256"]
    value = {"purpose": "Independent provenance audit for the planned grasp pose gallery.",
        "part_count": 14, "all_displayed_parts_have_saved_selected_grasp": True,
        "original_trial_selected_grasps": 13, "separately_labeled_8mm_followup_grasps": 1,
        "original_trial_is_preserved": True,
        "not_a_new_14_part_success_benchmark": True,
        "trajectory_pose_substitution": False,
        "rendering_not_performed_by_this_audit": True,
        "units": "metres; rotations are world-frame homogeneous matrices",
        "original_run_manifest": {"path": str(ORIGINAL / "manifest.json"), "sha256": sha(ORIGINAL / "manifest.json"),
                                  **source_hash_audit(manifest)},
        "followup_run_manifest": {"path": str(FOLLOWUP / "manifest.json"), "sha256": sha(FOLLOWUP / "manifest.json"),
                                  **source_hash_audit(followup_manifest)},
        "gallery_audit_source": {"path": str(Path(__file__)), "sha256": sha(Path(__file__)),
            "note": "Gallery helpers were added after the frozen run and are not claimed to appear in its creation-time manifest."},
        "parts": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "selected_poses": len(rows), "original": 13,
                      "followup": 1, "original_source_hashes_match": value["original_run_manifest"]["all_recorded_sources_match"],
                      "followup_source_hashes_match": value["followup_run_manifest"]["all_recorded_sources_match"]}, indent=2))


if __name__ == "__main__":
    main()
