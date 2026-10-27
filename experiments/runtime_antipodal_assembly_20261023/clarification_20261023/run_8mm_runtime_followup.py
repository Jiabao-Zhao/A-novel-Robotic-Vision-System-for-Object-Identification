"""One automatic 8 mm pickup attempt using freshly generated CAD contacts.

Extend only the generic finger-depth sampling; keep the frozen sampler, score,
table geometry, robot checks and contact-driven controller unchanged. No manual
contact pair, stored part-specific grasp, attachment or target feedback is used.
"""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiments.runtime_antipodal_assembly_20261023.grasp_planning import (
    sample_contacts, candidate_frames, table_filter, linear_pose_path, validate_path,
)
from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/runtime_antipodal_assembly_20261023"
SOURCE = ROOT / "outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml"
FROZEN_RUN = ROOT / "outputs/runtime_antipodal_assembly_20261023/20261023T213101Z"
PART = "RGOCG8-50_8mm"
MESH = ROOT / f"outputs/simulation/nist_task_board_1/assets/meshes/{PART}_m.stl"
SURFACE_RAYS = 2048
CONTACT_PAIR_BUDGET = 320
PLANNING_BUDGET = 48
EXTRA_GENERIC_FINGER_DEPTHS_M = (.006, .008)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def restore_other_part_visibility(trial):
    """Restore original appearance in the new XML/model; retain collision-off."""
    reference = mujoco.MjModel.from_xml_path(str(SOURCE))
    original_root = ET.parse(SOURCE).getroot()
    original_geoms = {element.get("name"): element
                      for element in original_root.findall("worldbody//geom")}
    tree = ET.parse(trial.scene_path)
    restored = []
    for body in tree.getroot().findall("worldbody//body"):
        name = body.get("name", "")
        if not name.startswith("nist_part_") or name == trial.target_name:
            continue
        restored.append(name)
        for geom in body.findall(".//geom"):
            old = original_geoms[geom.get("name")]
            if old.get("rgba") is None:
                geom.attrib.pop("rgba", None)
            else:
                geom.set("rgba", old.get("rgba"))
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom_id = trial.m.geom(geom.get("name")).id
            ref_id = reference.geom(geom.get("name")).id
            trial.m.geom_rgba[geom_id] = reference.geom_rgba[ref_id]
            assert trial.m.geom_contype[geom_id] == 0
            assert trial.m.geom_conaffinity[geom_id] == 0
    tree.write(trial.scene_path, encoding="unicode")
    trial.hidden_other_parts = []
    return restored


def build_pick_plan(candidate, initial):
    grasp = candidate.world_T_grip.copy()
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= .10 * grasp[:3, 2]
    lift = grasp.copy()
    lift[:3, 3] += [0., 0., .16]
    return {"grasp": grasp, "pregrasp": pregrasp, "lift": lift,
            "grip_T_object": np.linalg.inv(grasp) @ initial,
            "object_T_grip": np.linalg.inv(initial) @ grasp}


def check_pick_plan(trial, candidate, plan):
    segments = (
        ("approach", trial.pose(), plan["pregrasp"], candidate.opening_m, False),
        ("descend", plan["pregrasp"], plan["grasp"], candidate.opening_m, False),
        ("lift", plan["grasp"], plan["lift"], candidate.width_m, True),
    )
    seed = trial.d.qpos[trial.qa].copy()
    reports = []
    for name, first, last, opening, held in segments:
        qpath, report = validate_path(
            trial, linear_pose_path(first, last, 10), opening, seed=seed,
            target_relative=plan["grip_T_object"] if held else None,
            check_open_target=not held)
        reports.append({"phase": name, **report})
        if qpath is None:
            return False, reports
        seed = qpath[-1]
    return True, reports


def execute_once(trial, candidate, plan, initial):
    snapshots = []
    def snapshot(name):
        path = trial.output_dir / (name + ".png")
        trial.render(path)
        snapshots.append(str(path))
    snapshot("initial_all_parts_visible")
    trial.gripper(candidate.opening_m)
    trial.move(plan["pregrasp"], 100, "approach")
    trial.move(plan["grasp"], 65, "descend")
    trial.gripper(0.)
    trial.hold(35, "close")
    snapshot("closed")
    if len(trial.states[-1]["grasp_contacts"]) != 2:
        return {"status": "grasp_failed", "reason": "No bilateral contact after closing.",
                "snapshots": snapshots}
    trial.move(plan["lift"], 100, "lift")
    trial.hold(25, "hold_lift")
    snapshot("lifted_all_parts_visible")
    # Simulator truth is evaluated after motion; it is never a command correction.
    final = trial.target_pose()
    lift_m = float(final[2, 3] - initial[2, 3])
    bilateral = len(trial.states[-1]["grasp_contacts"]) == 2
    collision_ok = max(trial.audit[key] for key in ("robot_self", "robot_table", "robot_board")) < .001
    passed = lift_m >= .12 and bilateral and collision_ok
    return {"status": "pickup_passed" if passed else "pickup_failed", "lift_m": lift_m,
            "bilateral_after_lift_and_hold": bilateral,
            "robot_collision_penetration_below_1mm": collision_ok,
            "final_world_T_cad_audit": final.tolist(), "snapshots": snapshots}


def main():
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs/runtime_antipodal_assembly_20261023/8mm_runtime_followup" / stamp
    out.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={out}", flush=True)
    manifest = {"part": PART, "scope": "One runtime-generated 8 mm pickup; no assembly attempt.",
                "source_scene": str(SOURCE), "source_scene_sha256": digest(SOURCE),
                "mesh_sha256": digest(MESH), "frozen_comparison_run": str(FROZEN_RUN),
                "frozen_result_sha256": digest(FROZEN_RUN / PART / "result.json"),
                "surface_rays": SURFACE_RAYS, "contact_pair_budget": CONTACT_PAIR_BUDGET,
                "planning_candidate_budget": PLANNING_BUDGET,
                "generic_additional_finger_depths_m": EXTRA_GENERIC_FINGER_DEPTHS_M,
                "finger_mesh_maximum_tool_z_m_from_static_audit": .009249,
                "depth_rationale": "Additional generic full-finger contact depths beyond the original +4 mm pad-only sample. Finger collision mesh extends to about +9.249 mm in tool Z. No CAD-specific contact location is supplied.",
                "training_or_pretrained_grasp_model": False, "manual_contact_pair": False,
                "classification_and_initial_alignment_assumed_correct": True,
                "maximum_physical_attempts": 1,
                "source_hashes": {str(path.relative_to(EXPERIMENT)): digest(path)
                                  for path in (Path(__file__), EXPERIMENT / "grasp_planning.py",
                                               EXPERIMENT / "robot_trial.py",
                                               EXPERIMENT / "vendor/antipodal_sampler.py")}}
    write_json(out / "manifest.json", manifest)
    trial = RobotTrial(SOURCE, out, PART)
    result = {"part": PART, "physical_attempts": 0, "assembly_attempted": False,
              "training_required": False, "manual_contact_pair": False,
              "classification_and_initial_alignment_assumed_correct": True}
    try:
        restored = restore_other_part_visibility(trial)
        result["other_parts_visible_but_collision_off"] = restored
        result["visible_scene_xml_sha256"] = digest(trial.scene_path)
        initial = trial.target_pose()
        result["world_T_cad_input"] = initial.tolist()
        _, _, _, pairs, sampling = sample_contacts(MESH, initial, SURFACE_RAYS)
        result.update(sampling)
        np.savez_compressed(out / "fresh_runtime_contact_pairs.npz", contacts_world_m=pairs)
        original = candidate_frames(pairs, trial.pose()[:3, :3], initial[:3, 3], CONTACT_PAIR_BUDGET)
        added = []
        for candidate in original:
            if not np.isclose(candidate.pad_depth_m, .004):
                continue
            for depth in EXTRA_GENERIC_FINGER_DEPTHS_M:
                pose = candidate.world_T_grip.copy()
                pose[:3, 3] -= pose[:3, 2] * (depth - candidate.pad_depth_m)
                added.append(replace(candidate, world_T_grip=pose, pad_depth_m=depth))
        candidates = sorted(original + added, key=lambda item: item.score)
        clear = table_filter(candidates, trial.m, trial.d)
        result.update(original_generated_gripper_poses=len(original),
                      added_generic_depth_gripper_poses=len(added),
                      total_generated_gripper_poses=len(candidates),
                      table_clear_original_depth_candidates=sum(candidate.pad_depth_m not in EXTRA_GENERIC_FINGER_DEPTHS_M for candidate in clear),
                      table_clear_extended_depth_candidates=sum(candidate.pad_depth_m in EXTRA_GENERIC_FINGER_DEPTHS_M for candidate in clear),
                      table_clear_candidates=len(clear))
        np.savez_compressed(out / "generated_gripper_candidates.npz",
                            world_T_grip=np.asarray([candidate.world_T_grip for candidate in candidates]),
                            pad_depth_m=np.asarray([candidate.pad_depth_m for candidate in candidates]),
                            score=np.asarray([candidate.score for candidate in candidates]),
                            table_clearance_m=np.asarray([candidate.table_clearance_m for candidate in candidates]),
                            opening_m=np.asarray([candidate.opening_m for candidate in candidates]),
                            contacts_world_m=np.asarray([candidate.contacts_world_m for candidate in candidates]))
        print(json.dumps({key: result[key] for key in ("qualified_unique_pairs", "total_generated_gripper_poses", "table_clear_candidates")}), flush=True)
        selected, rejections = None, []
        for index, candidate in enumerate(clear[:PLANNING_BUDGET]):
            plan = build_pick_plan(candidate, initial)
            passed, checks = check_pick_plan(trial, candidate, plan)
            if passed:
                selected = candidate
                result.update(selected_table_clear_candidate_index=index,
                              selected_grasp=candidate.record(), path_checks=checks,
                              plan={key: value.tolist() for key, value in plan.items()})
                break
            rejections.append({"candidate_index": index, "checks": checks})
        result["planning_rejections"] = rejections
        result["planning_seconds"] = time.perf_counter() - started
        if selected is None:
            result["status"] = "no_feasible_sampled_pickup"
        else:
            result["physical_attempts"] = 1
            print(json.dumps({"selected_depth_m": selected.pad_depth_m,
                              "selected_contacts_world_m": selected.contacts_world_m.tolist()}), flush=True)
            result.update(execute_once(trial, selected, plan, initial))
    except Exception as error:
        result.update(status="motion_aborted" if len(trial.states) > 1 else "experiment_error",
                      error=str(error), traceback=traceback.format_exc())
    finally:
        result["last_executed_phase"] = trial.states[-1]["phase"]
        result["maximum_observed_lift_m"] = float(max(state["target_pose_audit"][2][3]
            for state in trial.states) - result.get("world_T_cad_input", trial.target_pose())[2][3])
        result["controller_audit"] = trial.save()
        trial.close()
        result["total_seconds"] = time.perf_counter() - started
        result["interpretation"] = "Single automatic pickup follow-up with additional generic finger-depth samples; this is not a new 14-part benchmark or an assembly-success result."
        write_json(out / "result.json", result)
    print(json.dumps({key: result.get(key) for key in ("status", "physical_attempts", "lift_m", "maximum_observed_lift_m", "bilateral_after_lift_and_hold", "error", "total_seconds")}), flush=True)
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
