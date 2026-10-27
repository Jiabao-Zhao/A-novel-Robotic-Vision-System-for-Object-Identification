"""Fresh small/large pickup comparison; no assembly or physical robot access."""
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

from ..grasp_planning import sample_contacts, table_filter
from ..robot_trial import RobotTrial
from ..run_experiment import PARTS, build_plan, check_plan
from ..clarification_20261023.run_8mm_runtime_followup import (
    restore_other_part_visibility, execute_once,
)
from .search import SMALL, LARGE, FINGER_DEPTHS_M, frames_with_full_finger_depths, small_shortlist


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "outputs/runtime_antipodal_assembly_20261023"
SOURCE = BASE / "input_scene/scene.xml"
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_part(part, arm, output):
    config = SMALL if arm == "small" else LARGE
    folder = output / arm / part
    started = time.perf_counter()
    trial = RobotTrial(SOURCE, folder, part)
    result = {"part": part, "arm": arm, "config": config, "physical_attempts": 0,
              "assembly_attempted": False, "shortlisted_grasps": [],
              "source_scene_sha256": digest(SOURCE), "mesh_sha256": digest(MESHES / (part + "_m.stl"))}
    try:
        restored = restore_other_part_visibility(trial)
        initial = trial.target_pose()
        result.update(world_T_cad_input=initial.tolist(),
                      other_parts_visible_but_collision_off=restored,
                      setup_seconds=time.perf_counter()-started)
        search_started = time.perf_counter()
        _, _, _, pairs, sampling = sample_contacts(MESHES / (part + "_m.stl"), initial, config["surface_rays"])
        result.update(sampling)
        stage = time.perf_counter()
        candidates = frames_with_full_finger_depths(pairs, trial.pose()[:3, :3], initial[:3, 3], config["contact_pairs"])
        result["frame_generation_seconds"] = time.perf_counter()-stage
        stage = time.perf_counter()
        clear = table_filter(candidates, trial.m, trial.d)
        result["table_check_seconds"] = time.perf_counter()-stage
        stage = time.perf_counter()
        shortlist = (small_shortlist(clear, config["shortlist"]) if arm == "small"
                     else clear[:config["shortlist"]])
        result["shortlist_seconds"] = time.perf_counter()-stage
        ids = {id(candidate): index for index, candidate in enumerate(candidates)}
        result.update(generated_gripper_poses=len(candidates), table_clear_candidates=len(clear),
                      shortlisted_count=len(shortlist),
                      candidate_generation_and_filter_seconds=time.perf_counter()-search_started)
        result["shortlisted_grasps"] = [dict(c.record(), candidate_id=ids[id(c)],
                                           path_check_status="not_checked", path_checks=[])
                                        for c in shortlist]
        selected = None
        stage = time.perf_counter()
        for index, candidate in enumerate(shortlist):
            # Identical 18 cm pickup path in both arms; final assembly targets
            # do not influence this experiment's grasp ranking or motion search.
            plan = build_plan(trial, candidate, initial, initial)
            passed, checks = check_plan(trial, candidate, plan, pickup_only=True)
            record = result["shortlisted_grasps"][index]
            record.update(path_check_status="passed" if passed else "failed", path_checks=checks)
            if passed:
                selected = candidate
                result.update(selected_shortlist_index=index, selected_grasp=candidate.record(),
                              selected_candidate_id=ids[id(candidate)], path_checks=checks,
                              plan={key: value.tolist() for key, value in plan.items()})
                break
        result["robot_path_check_seconds"] = time.perf_counter()-stage
        result["search_seconds"] = time.perf_counter()-search_started
        result["planning_seconds_including_setup"] = time.perf_counter()-started
        result["path_candidates_checked"] = sum(r["path_check_status"] != "not_checked" for r in result["shortlisted_grasps"])
        # Save bulky evidence after stopping the search timer, before physics.
        np.savez_compressed(folder / "generated_candidates.npz",
                            contacts_sampled_world_m=pairs,
                            world_T_grip=np.asarray([c.world_T_grip for c in candidates]),
                            contacts_world_m=np.asarray([c.contacts_world_m for c in candidates]),
                            pad_depth_m=np.asarray([c.pad_depth_m for c in candidates]),
                            score=np.asarray([c.score for c in candidates]),
                            opening_m=np.asarray([c.opening_m for c in candidates]),
                            table_clearance_m=np.asarray([c.table_clearance_m for c in candidates]),
                            shortlisted_candidate_ids=np.asarray([ids[id(c)] for c in shortlist], dtype=int))
        if selected is None:
            result["status"] = "no_pickup_path_found_in_shortlist"
        else:
            result["physical_attempts"] = 1
            stage = time.perf_counter()
            result.update(execute_once(trial, selected, plan, initial))
            result["execution_and_snapshots_seconds"] = time.perf_counter()-stage
    except Exception as error:
        result.update(status="motion_aborted" if len(trial.states)>1 else "experiment_error",
                      error=str(error), traceback=traceback.format_exc())
    finally:
        result["last_executed_phase"] = trial.states[-1]["phase"]
        result["maximum_observed_lift_m"] = float(max(s["target_pose_audit"][2][3] for s in trial.states)
            - result.get("world_T_cad_input", trial.target_pose())[2][3])
        result["controller_audit"] = trial.save()
        trial.close()
        result["total_seconds"] = time.perf_counter()-started
        write_json(folder / "result.json", result)
    return result


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = BASE / "smaller_search" / stamp
    output.mkdir(parents=True, exist_ok=False)
    protected = [SOURCE, BASE / "20261023T213101Z/results.json",
                 BASE / "8mm_runtime_followup/20261024T012239Z/result.json"]
    protected += list((ROOT / "experiments/runtime_antipodal_assembly_20261023").glob("*.py"))
    manifest = {"created_utc": stamp, "parts": PARTS, "small": SMALL, "large_control": LARGE,
                "finger_depths_m": FINGER_DEPTHS_M, "source_scene": str(SOURCE),
                "source_scene_sha256": digest(SOURCE), "pickup_lift_command_m": .18,
                "maximum_physical_attempts_per_part_per_arm": 1,
                "pass_rule": "At least 0.12 m lift after 1 s hold, both fingers touching, robot/table/self/board overlap below 0.001 m",
                "timing": "Serial same-process runs; arm order alternates per part. Imports excluded; setup and stage times recorded. One observation per part/arm.",
                "assumptions": ["Perfect CAD identity and initial world-frame alignment", "Old objects_centered 14-part scene",
                    "Other 13 loose parts visible, frozen and noncolliding", "Same unchanged contact-driven controller; no attachments",
                    "Planner checks target approach, robot self and table; board/fixed hardware contacts audited in physics",
                    "Pickup only; assembly success not tested", "Fixed budgets; no expansion after observing failure"],
                "protected_sha256_before": {str(p): digest(p) for p in protected},
                "source_hashes": {str(p.relative_to(HERE)): digest(p) for p in HERE.glob("*.py")}}
    write_json(output / "manifest.json", manifest)
    print(f"OUTPUT={output}", flush=True)
    results = []
    for index, part in enumerate(PARTS):
        order = ("small", "large_control") if index%2 == 0 else ("large_control", "small")
        for arm in order:
            print(f"START {arm} {part}", flush=True)
            result = run_part(part, arm, output)
            results.append(result)
            write_json(output / "results.json", results)
            print(json.dumps({key: result.get(key) for key in ("arm", "part", "status", "generated_gripper_poses", "shortlisted_count", "path_candidates_checked", "search_seconds", "planning_seconds_including_setup", "lift_m", "error")}), flush=True)
    after = {str(p): digest(p) for p in protected}
    write_json(output / "preservation.json", {"unchanged": after == manifest["protected_sha256_before"], "sha256_after": after})
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
