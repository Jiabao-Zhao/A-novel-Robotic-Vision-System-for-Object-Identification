"""Independent offline check after a geometry-only yaw ranking is sealed."""
import json
import math
from pathlib import Path
import sys

from .audit import ROOT, digest, freeze_online_before_truth, goal_error, local_path, pose_change, read, save


def audit(run):
    run = Path(run).resolve()
    seal_path = run/"frozen_ranking_hashes.json"
    seal = read(seal_path)
    seal_checks = {name:digest(local_path(name)) == sha for name,sha in seal.items()}
    if not all(seal_checks.values()):
        raise RuntimeError("A sealed ranking/protocol file changed before independent truth evaluation")
    protocol = read(run/"protocol.json")
    source_checks = {
        "diagnostic_source_matches_sealed_protocol":digest(run.parent/"run_diagnostic.py") == protocol["source_sha256"],
        "preserved_source_snapshot_matches":digest(run/"source_snapshot/run_diagnostic.py") == protocol["source_sha256"],
        "declared_workspace_dependencies_unchanged":all(digest(local_path(name)) == sha for name,sha in protocol["dependencies_sha256"].items())}
    source_run = local_path(protocol["source_run"])
    freeze_online_before_truth(source_run)
    offline = {row["part"]:row for row in read(source_run/"offline_evaluation.json")["parts"]}
    rows = []
    for part in ("Gear_Large","Gear_Medium"):
        ranked_path = run/part/"ranked_result.json"
        value = read(ranked_path)
        truth_record = next(row for row in offline[part]["observations"] if row["label"] == "after_pickup")
        truth = truth_record["actual_world_T_cad_OFFLINE_ONLY"]
        step = protocol["CAD_Z_seed_steps_deg"][part]
        candidates = value["all_candidates"]
        ranked = sorted(candidates,key=lambda row:row["combined_loss_m2"])
        checks = {"ranking_sealed_before_independent_truth_read":value["ranking_frozen_before_truth_comparison"] is True,
            "diagnostic_declares_no_truth_read":value["truth_or_actual_orientation_files_read"] is False,
            "diagnostic_never_used_by_running_controller":value["sent_to_or_used_by_running_controller"] is False,
            "all_declared_yaw_starts_saved":len(candidates) == round(360/step) and sorted(row["initial_local_Z_offset_deg"] for row in candidates) == [index*step for index in range(round(360/step))],
            "rank_order_reproduces":value["ranked_candidate_ids"] == [row["candidate_id"] for row in ranked],
            "selected_lowest_geometry_loss":value["selected_candidate_id"] == ranked[0]["candidate_id"] and value["selected_world_T_cad"] == ranked[0]["world_T_cad"],
            "all_combined_losses_equal_declared_mean":all(abs(row["combined_loss_m2"]-.5*(row["point_surface_loss_m2"]+row["forward_depth_loss_m2"])) < 1e-18 for row in candidates)}
        inputs = value["input_files_sha256"]
        repo_inputs,external_inputs = {},{}
        for name,sha in inputs.items():
            if "/"+ROOT.name+"/" in str(name).replace("\\","/"):
                repo_inputs[name] = digest(local_path(name)) == sha
            else:
                external_inputs[name] = sha
        checks["all_workspace_input_hashes_unchanged"] = all(repo_inputs.values())
        checks["no_truth_or_prior_diagnostic_files_in_declared_inputs"] = not any(any(
            token in name for token in ("plant_private","offline_evaluation","offline_initial_phase","offline_contact","global_yaw_diagnostic")) for name in inputs)
        distinct = []
        for candidate in ranked:
            if all(pose_change(previous["world_T_cad"],candidate["world_T_cad"])["rotation_deg"] >= step/2 for previous in distinct):
                distinct.append(candidate)
        recorded_distinct = value["top_distinct_yaw_candidates"]
        checks["distinct_pose_ranking_reproduces"] = [row["candidate_id"] for row in distinct[:len(recorded_distinct)]] == [row["candidate_id"] for row in recorded_distinct]
        gap = distinct[1]["combined_loss_m2"]-distinct[0]["combined_loss_m2"]
        checks["distinct_score_gap_reproduces"] = abs(gap-value["distinct_solution_combined_loss_gap_m2"]) < 1e-18
        def metrics(candidate):
            return {"candidate_id":candidate["candidate_id"],"combined_clipped_rms_m":math.sqrt(candidate["combined_loss_m2"]),
                "pose_error_OFFLINE_ONLY":goal_error(candidate["world_T_cad"],truth)}
        rows.append({"part":part,"candidate_count":len(candidates),
            "original_local_estimate_error_OFFLINE_ONLY":goal_error(value["estimated_pose_prior_from_public_file"],truth),
            "selected":metrics(distinct[0]),"next_distinct":metrics(distinct[1]),
            "second_to_best_squared_loss_ratio":distinct[1]["combined_loss_m2"]/distinct[0]["combined_loss_m2"],
            "distinct_squared_loss_gap_m2":gap,"diagnostic_wall_seconds":value["seconds"],
            "workspace_input_hashes_checked":repo_inputs,
            "external_known_robot_assets_declared_not_rehashed_by_this_windows_audit":external_inputs,
            "checks":checks,"passed":all(checks.values())})
    output = {"scope":"Independent offline comparison performed only after saved geometric ranking/protocol hashes were sealed. No search, fitting, motion or online pose is changed.",
        "seal_checks":seal_checks,"source_checks":source_checks,"source_truth_evaluation_sha256":digest(source_run/"offline_evaluation.json"),
        "interpretation":"The selected phases agree with withheld truth in these two noiseless, calibrated, four-view simulated pickup frames. The score gaps are geometric separation, not calibrated probabilities or evidence of performance with real sensor noise. Nothing proves unique observability for another view or layout.",
        "parts":rows,"passed":all(seal_checks.values()) and all(source_checks.values()) and all(row["passed"] for row in rows)}
    save(run/"independent_offline_evaluation.json",output)
    return output


if __name__ == "__main__":
    value = audit(Path(sys.argv[1]))
    print(json.dumps({"passed":value["passed"],"parts":[{
        "part":row["part"],"selected_error":row["selected"]["pose_error_OFFLINE_ONLY"],
        "checks":row["checks"]} for row in value["parts"]]},indent=2))
    raise SystemExit(0 if value["passed"] else 1)
