"""Three-part CAD-assisted pickup retry; frozen network, no assembly goals/views."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from .. import run_trial as old
from .. import vlm_client
from .refinement import refine_candidates, protocol

np, torch, Image = old.np, old.torch, old.Image
HERE = Path(__file__).resolve().parent
BASE = old.BASE / "cad_refinement_20261025"
vlm_client.QUEUE = BASE / "bridge_queue"
PARTS = ("Gear_Large", "Gear_Medium", "M16_Hex_Nut")
ARMS = ("geometric", "grconvnet_score", "grconvnet_vlm",
        "cad_grconvnet_score", "cad_grconvnet_vlm")


def relative_pose_audit(trial, result):
    """Measure closure/lift change after execution; never feeds movement control."""
    planned = result.get("plan", {}).get("grip_T_object")
    if planned is None:
        return {"available": False, "reason": "No pickup plan was selected"}
    planned = np.asarray(planned)
    phases = {}
    for phase in ("descend", "close", "lift", "hold_lift"):
        rows = [row for row in trial.states if row["phase"] == phase]
        if not rows:
            continue
        row = rows[-1]
        relative = np.linalg.inv(np.asarray(row["grip_site_pose"])) @ np.asarray(row["target_pose_audit"])
        delta = np.linalg.inv(planned) @ relative
        phases[phase] = {"time_s": row["time_s"], "grip_T_cad_audit": relative.tolist(),
                         "translation_change_from_planned_m": float(np.linalg.norm(relative[:3, 3]-planned[:3, 3])),
                         "rotation_change_from_planned_deg": float(np.rad2deg(np.linalg.norm(
                             old.Rotation.from_matrix(delta[:3, :3]).as_rotvec())))}
    if "close" in phases:
        closed = np.asarray(phases["close"]["grip_T_cad_audit"])
        for phase, row in phases.items():
            actual = np.asarray(row["grip_T_cad_audit"])
            row["translation_change_from_closure_m"] = float(np.linalg.norm(actual[:3, 3]-closed[:3, 3]))
            row["rotation_change_from_closure_deg"] = float(np.rad2deg(np.linalg.norm(
                old.Rotation.from_matrix(closed[:3, :3].T @ actual[:3, :3]).as_rotvec())))
    return {"available": True, "frame": "gripper0_grip_site", "units": "metres and degrees",
            "planned_grip_T_cad": planned.tolist(), "phases": phases,
            "source": "Simulator target pose used only for post-execution audit; no tracking or feedback claimed"}


def run_part(part, arm, index, output, model):
    folder = output / arm / part
    started = time.perf_counter()
    trial = old.RobotTrial(old.SOURCE, folder, part)
    result = {"part": part, "arm": arm, "physical_attempts": 0, "assembly_attempted": False,
              "stage_seconds": {}, "candidates": [], "shortlisted_grasps": [], "training_performed": False,
              "source_scene_sha256": old.digest(old.SOURCE), "shared_pool_cost_reused": False,
              "cad_refinement_used": arm.startswith("cad_"),
              "input_pose_source": "Perfect initial CAD alignment supplied from simulator; pose estimator is not run"}
    stages = result["stage_seconds"]
    decision_started = None
    try:
        result["other_parts_visible_but_collision_off"] = old.restore_other_part_visibility(trial)
        initial = trial.target_pose()
        result["world_T_cad_input"] = initial.tolist()
        result["setup_seconds"] = time.perf_counter()-started
        decision_started = time.perf_counter()
        stage = time.perf_counter()
        observation = old.capture_scene(trial, folder / "capture")
        stages["capture"] = time.perf_counter()-stage
        if arm == "geometric":
            stage = time.perf_counter()
            _, _, _, pairs, stats = old.sample_contacts(old.MESHES/(part+"_m.stl"), initial, 512)
            stages["contact_sampling"] = time.perf_counter()-stage
            result.update(stats)
            stage = time.perf_counter()
            generated = old.frames_with_full_finger_depths(pairs, trial.pose()[:3, :3], initial[:3, 3], 24)
            clear = old.table_filter(generated, trial.m, trial.d)
            pool = old.small_shortlist(clear, 12)
            records = [{"candidate_id": j, **candidate.record()} for j, candidate in enumerate(pool)]
            result.update(generated_gripper_poses=len(generated), table_clear_candidates=len(clear),
                          object_pose_used_for_grasp_generation=True)
            stages["frame_generation"] = time.perf_counter()-stage
        else:
            prediction = model.predict(observation["rgb"], observation["depth_m"], observation["target_mask"], 4)
            stages.update(preprocessing=prediction["timings"]["preprocess_seconds"],
                          inference=prediction["timings"]["network_seconds"],
                          postprocessing=prediction["timings"]["postprocess_seconds"])
            stage = time.perf_counter()
            if arm.startswith("cad_"):
                pool, records, rejected, diagnostics = refine_candidates(
                    prediction, observation, trial, old.MESHES/(part+"_m.stl"), initial)
                result["cad_refinement"] = diagnostics
                result["mesh_sha256"] = old.digest(old.MESHES/(part+"_m.stl"))
            else:
                pool, records, rejected = old.learned_candidates(prediction, observation, trial)
            stages["frame_generation"] = time.perf_counter()-stage
            result.update(raw_network_proposals=prediction["proposals"], proposal_mapping_rejections=rejected,
                          crop=prediction["crop"], model_sha256=prediction["model_sha256"],
                          object_pose_used_for_grasp_generation=arm.startswith("cad_"),
                          maximum_depth_combinations=len(prediction["proposals"])*len(old.DEPTHS),
                          table_clear_candidates=len(pool))
            stage = time.perf_counter()
            np.savez_compressed(folder/"network_evidence.npz", **prediction["maps"],
                                input_tensor=prediction["input_tensor"],
                                preprocessed_depth_scaled=prediction["preprocessed_depth_scaled"])
            Image.fromarray(prediction["preprocessed_rgb"]).save(folder/"network_rgb.png")
            stages["evidence_writes"] = time.perf_counter()-stage
        result["candidates"] = records
        result["shortlisted_grasps"] = records
        result["shortlisted_count"] = len(pool)
        ordering = list(range(len(pool)))
        if arm.endswith("_vlm"):
            if pool:
                stage = time.perf_counter()
                # Same original part index preserves the frozen display shuffle.
                sheet, mapping = old.rank_sheet(observation, records, old.PARTS.index(part), folder)
                stages["rendering"] = time.perf_counter()-stage
                stage = time.perf_counter()
                ranking = old.rank(sheet, list(mapping), folder/"vlm")
                stages["vlm"] = time.perf_counter()-stage
                result["vlm_status"] = ranking["status"]
                if ranking["status"] == "ok":
                    result["vlm_ranked_display_ids"] = ranking["ranked_ids"]
                    ordering = [mapping[x] for x in ranking["ranked_ids"]]
                else:
                    result["status"] = "vlm_output_unavailable"
                    ordering = []
            else:
                result["vlm_status"] = "skipped_no_candidates"
        result["candidate_check_order"] = ordering
        selected, rejected_paths = None, []
        stage = time.perf_counter()
        for pool_index in ordering:
            candidate = pool[pool_index]
            grasp = candidate.world_T_grip.copy()
            pregrasp, lift = grasp.copy(), grasp.copy()
            pregrasp[:3, 3] -= .10*grasp[:3, 2]
            lift[:3, 3] += [0., 0., .18]
            plan = {"grasp": grasp, "pregrasp": pregrasp, "lift": lift,
                    "grip_T_object": np.linalg.inv(grasp) @ initial,
                    "object_T_grip": np.linalg.inv(initial) @ grasp}
            passed, checks = check_pickup_plan(trial, candidate, plan)
            if passed:
                selected = candidate
                result.update(selected_pool_index=pool_index, selected_shortlist_index=pool_index,
                              selected_grasp=candidate.record(), path_checks=checks,
                              plan={key: value.tolist() for key, value in plan.items()})
                break
            rejected_paths.append({"pool_index": pool_index, "checks": checks})
        stages["path_checks"] = time.perf_counter()-stage
        result["planning_rejections"] = rejected_paths
        result["path_candidates_checked"] = len(rejected_paths)+int(selected is not None)
        result["decision_seconds"] = time.perf_counter()-decision_started
        result["planning_seconds_including_setup"] = time.perf_counter()-started
        if selected is None:
            result.setdefault("status", "no_pickup_path_found_in_candidates")
        else:
            result["physical_attempts"] = 1
            stage = time.perf_counter()
            result.update(old.execute_once(trial, selected, plan, initial))
            result["execution_and_snapshots_seconds"] = time.perf_counter()-stage
    except Exception as error:
        result.update(status="motion_aborted" if len(trial.states)>1 else "experiment_error",
                      error=str(error), traceback=traceback.format_exc())
    finally:
        result.setdefault("setup_seconds", time.perf_counter()-started)
        if decision_started is not None:
            result.setdefault("decision_seconds", time.perf_counter()-decision_started)
            result.setdefault("planning_seconds_including_setup", time.perf_counter()-started)
        result["last_executed_phase"] = trial.states[-1]["phase"]
        result["maximum_observed_lift_m"] = float(max(s["target_pose_audit"][2][3] for s in trial.states)
            -result.get("world_T_cad_input", trial.target_pose())[2][3])
        result["part_in_gripper_audit"] = relative_pose_audit(trial, result)
        result["controller_audit"] = trial.save()
        trial.close()
        result["total_seconds"] = time.perf_counter()-started
        old.save(folder/"result.json", result)
    return result


def check_pickup_plan(trial, candidate, plan):
    """Identical first three checks from the frozen planner, without assembly keys."""
    from experiments.runtime_antipodal_assembly_20261023.grasp_planning import linear_pose_path, validate_path
    segments = (("approach", trial.pose(), plan["pregrasp"], candidate.opening_m, False, True),
                ("descend", plan["pregrasp"], plan["grasp"], candidate.opening_m, False, True),
                ("lift", plan["grasp"], plan["lift"], candidate.width_m, True, False))
    seed, reports = trial.d.qpos[trial.qa].copy(), []
    for name, first, last, opening, held, open_target in segments:
        qpath, report = validate_path(trial, linear_pose_path(first, last, 10), opening, seed=seed,
                                     target_relative=plan["grip_T_object"] if held else None,
                                     check_open_target=open_target)
        reports.append({"phase": name, **report})
        if qpath is None:
            return False, reports
        seed = qpath[-1]
    return True, reports


def main():
    torch.set_num_threads(4)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = BASE / "runs" / stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    model = old.GrConvNetAdapter(device="cpu")
    inventory = old.bridge({"action": "inventory"}, output/"vlm_inventory")
    sources = {str(p.relative_to(old.ROOT)): old.digest(p) for p in
               (HERE/"run_trial.py", HERE/"refinement.py", HERE/"worker.py")}
    manifest = {"parts": PARTS, "arms": ARMS, "created_utc": stamp,
                "source_scene_sha256": old.digest(old.SOURCE), "source_hashes": sources,
                "frozen_parent_source_sha256": old.digest(Path(old.__file__)),
                "model_setup_seconds_one_time": model.model_setup_seconds, "model_provenance": model.provenance,
                "vlm_shared_warmup": json.loads((BASE/"vlm_shared_warmup.json").read_text()),
                "vlm_inventory": inventory, "vlm_model": old.MODEL, "vlm_digest": old.EXPECTED_DIGEST,
                "vlm_options": old.OPTIONS, "prompt_sha256": old.digest(old.HERE/"grasp_rank_prompt.txt"),
                "cad_refinement_protocol": protocol(), "finger_depths_m": old.DEPTHS,
                "input_conditions": {"same_old_14part_starting_scene": True,
                    "perfect_target_mask_from_renderer": True, "CAD_pose_estimator_run": False,
                    "perfect_CAD_alignment_given_to_geometric_and_CAD_refinement": True,
                    "plain_GR_generation_uses_CAD_pose": False, "final_assembly_views_or_goals": False},
                "trial": {"lift_command_m": .18, "hold_after_lift_s": 1., "min_lift_m": .12,
                    "bilateral_contact_required": True, "robot_overlap_limit_m": .001,
                    "physical_attempt_limit_per_part_arm": 1},
                "timing": "Fresh capture/inference per learned arm. Five arms serial with rotating order. Decision includes capture/generation/VLM/pathchecks; total includes setup/executed simulation/evidence. No outcome-based parameter changes. Initial GR model load and common VLM preload are separate; all timed VLM calls use the same warmed model."}
    old.save(output/"manifest.json", manifest)
    results = []
    for index, part in enumerate(PARTS):
        order = ARMS[index:]+ARMS[:index]
        for arm in order:
            print(f"START {arm} {part}", flush=True)
            result = run_part(part, arm, index, output, model)
            results.append(result)
            old.save(output/"results.json", results)
            print(json.dumps({key: result.get(key) for key in ("arm", "part", "status", "physical_attempts",
                 "shortlisted_count", "vlm_status", "path_candidates_checked", "decision_seconds", "total_seconds", "lift_m", "error")}), flush=True)
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
