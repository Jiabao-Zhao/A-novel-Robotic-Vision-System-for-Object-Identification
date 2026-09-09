"""Wrist-only scene: one independent pickup attempt per object, no placement."""

import hashlib
import json
import os
from pathlib import Path
import traceback

import cv2
import numpy as np

from simulation.wrist_cluster import (CAMERA, DESCRIPTIONS, ROOT, SEED, WORKSPACE_MIN,
                                      WORKSPACE_MAX, BOARD_EXCLUSION_XY, OBSERVATION_CAMERA_POSITION_M,
                                      make_environment, prepare_catalog, random_placements)
from simulation.libero_control import (OPEN_GRIPPER, CLOSE_GRIPPER, hold_gripper,
                                      move_eef_to_pose, top_down_grasp_pose, select_grasp_orientation)
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.libero_io import save_libero_observation


THRESHOLD = .99
MODEL = "gpt-4.1-mini-2025-04-14"
CAMERA_DESCRIPTION = "You are given a near-top-down wrist-camera view of the workspace."
LIFT_M = .020
HOLD_STEPS = 20  # One second at 20 Hz.


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def prepare_observation(environment, seed=SEED):
    """Move to a fixed observation view using robot/camera calibration only."""
    from robosuite.utils.transform_utils import quat2mat
    raw = environment.reset(seed=seed)
    environment.set_control_mode("relative")
    raw = hold_gripper(environment, raw, OPEN_GRIPPER, "settle", 30)
    observation = LiberoRGBDSensor(environment, CAMERA).capture(raw)
    pose = np.eye(4)
    pose[:3, :3] = quat2mat(raw["robot0_eef_quat"])
    pose[:3, 3] = raw["robot0_eef_pos"] + np.array(OBSERVATION_CAMERA_POSITION_M) - observation.world_T_camera[:3, 3]
    raw = move_eef_to_pose(environment, raw, pose, OPEN_GRIPPER, "wrist_observation_pose")
    raw = hold_gripper(environment, raw, OPEN_GRIPPER, "settle_observation_pose", 20)
    return raw


def identity_audit(environment, localization, observation):
    from simulation.libero_assumed_human import match_candidates
    objects = [{"instance": name, "semantic_identity": DESCRIPTIONS[name],
                "position_world_m": environment.sim.data.body_xpos[
                    environment.sim.model.body_name2id(f"nist_part_{name}")].tolist()}
               for name in DESCRIPTIONS]
    return match_candidates(localization, observation.world_T_camera, objects)


class PickupEvaluator:
    """Simulator identities and contacts are evaluation-only, never control inputs."""

    def __init__(self, environment, target):
        self.environment, self.target = environment, target
        sim = environment.sim
        self.ids = {name: sim.model.body_name2id(f"nist_part_{name}") for name in DESCRIPTIONS}
        self.start = {name: sim.data.body_xpos[index].copy() for name, index in self.ids.items()}
        self.part_geoms = {}
        for name, body_id in self.ids.items():
            self.part_geoms[name] = set()
            for geom_id in range(sim.model.ngeom):
                parent = int(sim.model.geom_bodyid[geom_id])
                while parent and parent != body_id:
                    parent = int(sim.model.body_parentid[parent])
                if parent == body_id:
                    self.part_geoms[name].add(geom_id)
        gripper = environment.robots[0].gripper
        self.fingers = [set(sim.model.geom_name2id(name) for name in gripper.important_geoms[key])
                        for key in ("left_fingerpad", "right_fingerpad")]
        self.hold_steps = 0
        self.max_hold_steps = 0
        self.max_lift_m = 0.
        self.wrong_lifted = set()

    def score(self):
        sim = self.environment.sim
        touched = set()
        for contact in sim.data.contact[:sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.part_geoms[self.target]:
                touched |= pair - self.part_geoms[self.target]
        lift = float(sim.data.body_xpos[self.ids[self.target], 2] - self.start[self.target][2])
        grasped = all(touched & finger for finger in self.fingers)
        for name, index in self.ids.items():
            if name != self.target and sim.data.body_xpos[index, 2] - self.start[name][2] >= LIFT_M:
                self.wrong_lifted.add(name)
        held = bool(grasped and lift >= LIFT_M and not self.wrong_lifted)
        self.hold_steps = self.hold_steps + 1 if held else 0
        self.max_hold_steps = max(self.max_hold_steps, self.hold_steps)
        self.max_lift_m = max(self.max_lift_m, lift)
        return {"target_grasped": bool(grasped), "lift_mm": lift*1000,
                "max_lift_mm": self.max_lift_m*1000, "consecutive_hold_steps": self.hold_steps,
                "max_consecutive_hold_steps": self.max_hold_steps,
                "achieved_one_second_hold_at_any_time": self.max_hold_steps >= HOLD_STEPS,
                "wrong_objects_lifted": sorted(self.wrong_lifted), "success": self.hold_steps >= HOLD_STEPS}


def register_target(name, object_id, localization, observation, catalog, output_dir):
    from CADPointCloudRegistration import CADPointCloudRegistration, CADRegistrationConfig
    record = catalog[name]
    item = next(o for o in localization["objects"] if o["object_id"] == object_id)
    registrar = CADPointCloudRegistration(CADRegistrationConfig(
        voxel_size_m=.001, cad_sample_points=6000, xy_step_m=.002,
        min_xy_step_m=.0001, constrained_iterations=20))
    aligned = registrar.run(record["cad_path"], item["pointcloud_path"],
                            plane_model=localization["plane_model"], cad_metadata=record,
                            output_dir=output_dir)
    if aligned["warnings"] or aligned["constrained_rmse_m"] > record["max_registration_rmse_m"]:
        raise RuntimeError(f"CAD alignment rejected: RMSE={aligned['constrained_rmse_m']}; {aligned['warnings']}")
    transform = observation.world_T_camera @ np.asarray(aligned["T_observed_from_cad"])
    center = np.asarray(aligned["cad_center_m"])
    return {"object_id": object_id, "world_T_cad": transform.tolist(), "cad_center_cad_m": center.tolist(),
            "cad_extent_m": aligned["cad_extent_m"], "cad_up_axis": record["cad_up_axis"],
            "registered_center_world_m": (transform @ np.r_[center, 1.])[:3].tolist(),
            "constrained_rmse_m": aligned["constrained_rmse_m"], "cad_source": record["source"]}


def run_target(environment, frozen_state, name, catalog, root, localization, observation, visual, paths,
               *, frozen_gripper_action, seed=SEED):
    from robosuite.utils.transform_utils import quat2mat
    from simulation.libero_joint_association import associate_instruction_from_localization
    from simulation.libero_planning import build_simulation_context, generate_simulation_plan, execute_simulation_plan
    from vlm_module import candidate_bbox_map, load_localized_objects, candidate_choice_map
    case = root / name
    case.mkdir()
    stage = "initial_pose_estimation"
    environment.reset(seed=seed)
    raw = environment.env.set_init_state(frozen_state)
    environment.last_observation = raw
    environment.set_control_mode("relative")
    # The gripper's accumulated command is not part of the MuJoCo state vector.
    environment.robots[0].gripper.current_action = frozen_gripper_action.copy()
    restored = np.ascontiguousarray(environment.sim.get_state().flatten())
    np.testing.assert_allclose(restored, frozen_state, atol=0, rtol=0)
    instruction = f"Pick up the {DESCRIPTIONS[name]} and hold it above the workspace."
    result = {"target": name, "instruction": instruction, "success": False,
              "camera": CAMERA, "threshold": THRESHOLD, "seed": seed,
              "initial_gripper_action": environment.robots[0].gripper.current_action.tolist(),
              "frozen_state_sha256": hashlib.sha256(restored.tobytes()).hexdigest()}
    audit = identity_audit(environment, localization, observation)
    save_json(case / "identity_audit.json", audit)
    expected = [row["object_id"] for row in audit if row["simulator_instance"] == name]
    result["target_localized"] = len(expected) == 1
    evaluator = PickupEvaluator(environment, name)
    video = cv2.VideoWriter(str(case / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (768, 768))
    if not video.isOpened():
        raise RuntimeError("Cannot open wrist video writer.")
    actions = []
    try:
        def human(request):
            save_json(case / "assumed_human.json", {"actual_human_response": False, "audit": audit,
                      "requested_description": request["target_description"]})
            if len(expected) != 1:
                raise RuntimeError("Target has no uniquely localized candidate for assumed human correction.")
            return expected[0]
        stage = "classification"
        associate_instruction_from_localization(instruction, [DESCRIPTIONS[name]], image_path=visual,
            localization_path=paths["localization"], output_path=case / "vlm_result.json", threshold=THRESHOLD,
            human_resolver=human, assumed_human=True,
            camera_description=CAMERA_DESCRIPTION,
            clarification_bboxes=candidate_bbox_map(load_localized_objects(paths["localization"])))
        association = json.loads((case / "vlm_result.json").read_text())["associations"][0]
        result.update(association=association,
                      original_identity_correct=association["vlm_object_id"] in expected if expected else None,
                      final_identity_correct=association["final_object_id"] in expected if expected else None)
        if association["final_object_id"] is None:
            raise RuntimeError("Association returned no object to pick.")
        stage = "cad_to_observation_alignment"
        registration = register_target(name, association["final_object_id"], localization, observation,
                                       catalog, case / "registration")
        # This diagnostic is saved only; it does not reject or correct estimated poses.
        result["cad_center_error_mm_evaluation_only"] = float(1000*np.linalg.norm(
            np.asarray(registration["registered_center_world_m"]) - evaluator.start[name]))
        stage = "robot_execution_setup"
        grasp = top_down_grasp_pose(registration["world_T_cad"], registration["cad_center_cad_m"],
            registration["cad_extent_m"], registration["cad_up_axis"], quat2mat(raw["robot0_eef_quat"]),
            free_yaw=catalog[name].get("grasp_yaw_free", False))
        if name == "l_bracket":
            # Grasp its vertical leg at a fixed CAD-local location, not a simulator pose.
            grasp[:3, 3] = (np.asarray(registration["world_T_cad"]) @ [-.026, 0, .004, 1])[:3]
        grasp, result["robot_grasp_orientation_check"] = select_grasp_orientation(environment, grasp)
        context = build_simulation_context(localization, [association], registration,
                                           observation.world_T_camera, grasp, raw)
        context["completion_condition"] = "held"
        save_json(case / "execution_inputs.json", {"registration": registration, "context": context})
        stage = "llm_planning"
        planning = generate_simulation_plan(instruction, context, case / "llm_plan.json")
        result["planning_status"] = planning["status"]
        if planning["status"] != "ready":
            raise RuntimeError(f"Planner returned {planning['status']}")
        result["plan_consistent_with_resolved_identity"] = planning["plan"]["actions"] == [
            {"action": "pick", "object_id": association["final_object_id"]}]
        def record(plan_index, planned_action, phase, phase_step, raw, action, reward, done, info):
            metrics = evaluator.score()
            actions.append({"phase": phase, "action": np.asarray(action).tolist(), "metrics": metrics})
            if len(actions) % 2 == 0:
                rgb = np.ascontiguousarray(raw[f"{CAMERA}_image"][::-1])
                video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if len(actions) >= 600:
                raise RuntimeError("Pickup action budget exhausted.")
        stage = "execution"
        raw = execute_simulation_plan(environment, raw, planning["plan"], context, record)
        raw = hold_gripper(environment, raw, CLOSE_GRIPPER, "hold_above_workspace", HOLD_STEPS,
                           lambda *args: record(0, planning["plan"]["actions"][-1], *args))
        result["success"] = evaluator.hold_steps >= HOLD_STEPS
        if not result["success"]:
            result["error"] = ("Final hold failed: require bilateral contact with the requested object "
                "at least 20 mm above its start for 20 consecutive control steps ending at trial end, "
                "with no wrong object lifted during the trial.")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        (case / "error.txt").write_text(traceback.format_exc())
    finally:
        video.release()
        provider_path = case / "vlm_provider_response.json"
        if provider_path.exists():
            from simulation.libero_joint_association import joint_inferences
            inferred, parsed = joint_inferences(json.loads(provider_path.read_text()), instruction,
                candidate_choice_map(load_localized_objects(paths["localization"])), [DESCRIPTIONS[name]])
            result.update(raw_score=inferred[0]["association_score"],
                          raw_log_score=inferred[0]["diagnostics"]["raw_log_probability"],
                          vlm_output=parsed["generated_output_text"],
                          original_identity_correct=inferred[0]["vlm_object_id"] in expected if expected else None)
        if not actions:
            (case / "wrist.mp4").unlink()
        save_json(case / "actions.json", actions)
        result["stopped_at"] = None if result["success"] else stage
        result["failure_attribution"] = (None if result["success"] else
            "initial_pose_estimation" if not result["target_localized"] else
            "classification" if result.get("final_identity_correct") is False else
            "execution_or_grasp_unresolved" if stage == "execution" else stage)
        result["pickup_metrics"] = {"max_lift_mm": evaluator.max_lift_m*1000,
            "max_consecutive_hold_steps": evaluator.max_hold_steps,
            "achieved_one_second_hold_at_any_time": evaluator.max_hold_steps >= HOLD_STEPS,
            "consecutive_hold_steps": evaluator.hold_steps, "wrong_objects_lifted": sorted(evaluator.wrong_lifted)}
        save_json(case / "evaluation.json", result)
    return result


def main(execute=True, trial_number=2):
    from datetime import datetime, timezone
    from simulation.nist_peg_task import localize_parts
    from vlm_module import annotate_candidate_boxes
    os.environ["OPENAI_VLM_MODEL"] = MODEL
    os.environ["OPENAI_LLM_MODEL"] = MODEL
    if trial_number not in range(1, 11):
        raise ValueError("This study uses trials 1 through 10 for each object.")
    seed = SEED + trial_number - 1
    catalog = prepare_catalog()
    object_count = len(catalog)
    placements = random_placements(catalog, seed=seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ROOT / (f"try_{trial_number:02d}_{stamp}" if execute else f"setup_try_{trial_number:02d}")
    root.mkdir(parents=True, exist_ok=not execute)
    with make_environment(catalog, placements) as environment:
        raw = prepare_observation(environment, seed=seed)
        frozen = np.ascontiguousarray(environment.sim.get_state().flatten())
        frozen_gripper_action = environment.robots[0].gripper.current_action.copy()
        np.save(root / "initial_state.npy", frozen)
        observation = LiberoRGBDSensor(environment, CAMERA).capture(raw)
        capture = save_libero_observation(observation, environment, root / "capture")
        manifest = {"objects": list(DESCRIPTIONS), "object_count": object_count, "seed": seed,
            "trial_number": trial_number, "planned_trials_per_object": 10,
            "initial_state_variation": "Seeded XY positions and yaw; supported resting orientations retained.",
            "camera_inputs": [CAMERA], "depth_input": "simulated wrist depth",
            "camera_description": CAMERA_DESCRIPTION,
            "external_camera_enabled": False, "placements": placements, "catalog": catalog,
            "workspace_min_m": WORKSPACE_MIN, "workspace_max_m": WORKSPACE_MAX,
            "excluded_fixed_board_xy_m": BOARD_EXCLUSION_XY.tolist(),
            "clustering_space": "estimated_table_plane; retain original 3D points",
            "protocol": f"One target per attempt, all {object_count} present, same frozen scene restored before each attempt.",
            "destinations": "NIST board remains in background; no destination is used for pickup-only trials.",
            "success": "At trial end: correct object lifted >=20mm with bilateral contact for >=20 consecutive steps at 20Hz; no wrong object lifted during the trial.",
            "threshold": THRESHOLD, "model": MODEL,
            "initial_state_sha256": hashlib.sha256(frozen.tobytes()).hexdigest(),
            "initial_gripper_action": frozen_gripper_action.tolist(),
            "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), Path(__file__).parents[1]/"simulation/wrist_cluster.py",
                 Path(__file__).parents[1]/"simulation/libero_joint_association.py",
                 Path(__file__).parents[1]/"simulation/libero_control.py",
                 Path(__file__).parents[1]/"simulation/libero_planning.py",
                 Path(__file__).parents[1]/"point_cloud_localization.py",
                 Path(__file__).parents[1]/"simulation/nist_peg_task.py",
                 Path(__file__).parents[1]/"simulation/perception_adapter.py")}}
        save_json(root / "manifest.json", manifest)
        localization, paths = localize_parts(observation, capture["rgb"], root,
            workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX,
            cluster_in_table_plane=True, exclude_xy_bounds=BOARD_EXCLUSION_XY)
        visual = annotate_candidate_boxes(capture["rgb"], paths["localization"], root / "wrist_boxed.png")
        save_json(root / "identity_audit.json", identity_audit(environment, localization, observation))
        print(f"WRIST SCENE: {root}; {len(localization['objects'])} localized candidates / {object_count} objects", flush=True)
        if not execute:
            return root
        results = []
        for name in DESCRIPTIONS:
            result = run_target(environment, frozen, name, catalog, root, localization, observation, visual, paths,
                                frozen_gripper_action=frozen_gripper_action, seed=seed)
            results.append(result)
            save_json(root / "results.json", results)
            print(f"{len(results)}/{object_count} {name}: success={result['success']}, {result.get('failure_attribution')}", flush=True)
        print(f"COMPLETE: {sum(r['success'] for r in results)}/{object_count} pickups; {root}", flush=True)
    return root


if __name__ == "__main__":
    main()
