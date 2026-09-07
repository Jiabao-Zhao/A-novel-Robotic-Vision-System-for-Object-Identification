"""Run one RGB-D/VLM/CAD/LLM LIBERO-Object episode or a reference-plan replay."""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from simulation.libero_cad import register_libero_cad_to_observation
from simulation.libero_assumed_human import assumed_human_selection
from simulation.libero_joint_association import associate_instruction_from_localization
from simulation.libero_control import (
    OPEN_GRIPPER,
    hold_gripper,
    top_down_grasp_pose,
)
from simulation.libero_planning import (
    build_simulation_context, generate_simulation_plan, execute_simulation_plan,
    evaluate_simulation_plan, reference_pick_place_plan, validate_simulation_plan,
)
from simulation.libero_env import LiberoIntegrationError, LiberoTaskEnvironment
from simulation.libero_experiment import (
    PROPOSED_METHOD_FOLDER,
    episode_result_dir,
    libero_object_task,
)
from simulation.libero_io import save_libero_observation
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.perception_adapter import run_libero_localization
from vlm_module import (
    candidate_choice_map,
    contact_sheet_candidate_bbox_map,
    create_roi_contact_sheet,
    opencv_human_resolver,
)


SUITE_NAME = "libero_object"
DEFAULT_TASK_INDEX = 7
CAMERA_NAME = "agentview"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
VLM_CONTACT_SHEET_TILE_SIZE_PX = 448
RUN_VARIANT = "768x768_joint_instruction_assumed_human_v2"
ASSUME_CORRECT_HUMAN = True  # User-requested simulation condition, not measured human input.
# Development operating point selected on the 500-trial LIBERO
# calibration/validation partitions for the one-target prompt. It is provisional
# for this joint prompt and is not a calibrated probability of correctness.
LIBERO_RAW_ASSOCIATION_THRESHOLD = 0.9999832372181827
CONTROL_FREQUENCY_HZ = 20
EPISODE_HORIZON_STEPS = 280
CONTROL_MODE = "relative"
RANDOM_SEED = 1000
INITIAL_STATE_INDEX = 0
INITIAL_PHYSICS_SETTLE_STEPS = 10
VIDEO_FRAME_STRIDE = 2
METHOD_NAME = "rgbd_vlm_cad_llm_planner"


class _EpisodeSucceeded(RuntimeError):
    pass


class _EpisodeHorizonReached(RuntimeError):
    pass


class _EnvironmentTerminated(RuntimeError):
    pass


def main(task_index=DEFAULT_TASK_INDEX, *, initial_state_index=INITIAL_STATE_INDEX,
         output_root=None, reference_source=None):
    task_index, target_slug, instruction = libero_object_task(task_index)
    output_root = Path(output_root or episode_result_dir(
        PROPOSED_METHOD_FOLDER, task_index, initial_state_index, run_variant=RUN_VARIANT,
    )).resolve()
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise SystemExit(
            f"Refusing to overwrite the existing proposed-method result: {output_root}. "
            "Move the directory aside before deliberately rerunning this task."
        )
    progress = {"stage": "simulation_setup"}
    try:
        return _run_task(task_index, output_root, initial_state_index, reference_source, progress)
    except Exception as error:
        _write_pipeline_failure(
            output_root,
            task_index,
            target_slug,
            instruction,
            error,
            initial_state_index=initial_state_index,
            failure_stage=progress["stage"],
            simulator_identity_used=progress.get("simulator_identity_used", False),
        )
        raise


def _run_task(task_index, output_root, initial_state_index, reference_source, progress):
    task_index, target_slug, expected_instruction = libero_object_task(task_index)
    target_name = target_slug.replace("_", " ")
    frozen = None
    if reference_source is not None:
        frozen = json.loads((Path(reference_source) / "execution_inputs.json").read_text())
        if frozen["execution_config"] != _execution_config(task_index, initial_state_index):
            raise ValueError("Reference replay requires the same task, state, controller, and settings.")
        if frozen["instruction"] != expected_instruction:
            raise ValueError("Reference instruction does not match the task catalog.")
        if frozen["inputs_sha256"] != _json_sha256(frozen["perception"]):
            raise ValueError("Frozen perception inputs have changed.")
    perception_root = output_root / "perception"
    action_log = []
    video_frames = []
    step_count = 0
    first_success_step = None
    termination_reason = "controller_completed"
    execution_error = None

    with LiberoTaskEnvironment(
        suite_name=SUITE_NAME,
        task_index=task_index,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ) as environment:
        raw_observation = environment.reset(
            seed=RANDOM_SEED,
            init_state_index=initial_state_index,
        )
        initial_state_sha256 = environment.last_init_state_sha256
        initial_state_count = environment.initial_state_count
        if frozen is not None and frozen["initial_state_sha256"] != initial_state_sha256:
            raise ValueError("Reference replay initial-state hash differs from the original.")
        raw_observation = hold_gripper(
            environment,
            raw_observation,
            OPEN_GRIPPER,
            "initial_physics_settle",
            INITIAL_PHYSICS_SETTLE_STEPS,
        )
        environment.set_control_mode(CONTROL_MODE)
        # Integrity evidence only; no simulator object state is exposed to the planner.
        settled_state_sha256 = hashlib.sha256(
            np.ascontiguousarray(environment.sim.get_state().flatten()).tobytes()
        ).hexdigest()
        if frozen is not None and frozen["settled_state_sha256"] != settled_state_sha256:
            raise ValueError("Reference replay simulator state differs after settling.")
        sensor = LiberoRGBDSensor(environment, CAMERA_NAME)
        observation = sensor.capture(raw_observation)
        if observation.instruction != expected_instruction:
            raise RuntimeError(
                "LIBERO instruction does not match the experiment catalog: "
                f"expected {expected_instruction!r}, received {observation.instruction!r}."
            )
        if frozen is None:
            capture_paths = save_libero_observation(
                observation,
                environment,
                perception_root / "capture",
            )
            progress["stage"] = "initial_pose_estimation"
            localization, localization_paths, workspace_mask = run_libero_localization(
                observation,
                rgb_path=capture_paths["rgb"],
                output_root=perception_root,
            )
            vlm_visual_prompt_path = create_roi_contact_sheet(
                capture_paths["rgb"],
                localization_paths["localization"],
                output_root / "vlm_roi_contact_sheet.png",
                tile_size_px=VLM_CONTACT_SHEET_TILE_SIZE_PX,
                candidate_labels={object_id: label for label, object_id in
                                  candidate_choice_map(localization["objects"]).items()},
            )
            progress["stage"] = "classification"

            def correct_human(request):
                progress["simulator_identity_used"] = True
                return assumed_human_selection(
                    request, environment, localization, observation.world_T_camera,
                    output_root / "assumed_human",
                )

            vlm_result_path = associate_instruction_from_localization(
                expected_instruction, [target_name, "basket"],
                image_path=vlm_visual_prompt_path,
                localization_path=localization_paths["localization"],
                output_path=output_root / "vlm_result.json",
                threshold=LIBERO_RAW_ASSOCIATION_THRESHOLD,
                human_resolver=correct_human if ASSUME_CORRECT_HUMAN else opencv_human_resolver,
                assumed_human=ASSUME_CORRECT_HUMAN,
                clarification_bboxes=contact_sheet_candidate_bbox_map(
                    localization_paths["localization"],
                    tile_size_px=VLM_CONTACT_SHEET_TILE_SIZE_PX,
                ),
            )
            task_associations = task_associations_from_vlm_result(
                vlm_result_path,
                target_name,
            )
            objects_by_id = {
                item["object_id"]: item for item in localization["objects"]
            }
            target_object = objects_by_id[task_associations["target_object_id"]]
            basket = objects_by_id[task_associations["basket_object_id"]]
            target_depth_centroid_world_m = camera_point_to_world(
                target_object["centroid_3d_m"], observation.world_T_camera
            )
            basket_world_m = camera_point_to_world(
                basket["centroid_3d_m"], observation.world_T_camera
            )
            progress["stage"] = "cad_alignment"
            target_cad_registration = register_libero_cad_to_observation(
                object_type=target_name,
                object_id=task_associations["target_object_id"],
                localization=localization,
                world_T_camera=observation.world_T_camera,
                output_dir=output_root / "cad_registration" / target_slug,
            )
            target_world_m = np.asarray(
                target_cad_registration["registered_center_world_m"],
                dtype=float,
            )
            progress["stage"] = "pose_to_grasp_binding"
            from robosuite.utils.transform_utils import quat2mat

            world_T_grasp = top_down_grasp_pose(
                target_cad_registration["world_T_cad"],
                target_cad_registration["cad_center_cad_m"],
                target_cad_registration["cad_extent_m"],
                target_cad_registration["cad_up_axis"],
                quat2mat(np.asarray(raw_observation["robot0_eef_quat"], dtype=float)),
            )
            if not np.allclose(world_T_grasp[:3, 2], [0.0, 0.0, -1.0], atol=2e-3):
                raise RuntimeError("CAD-derived grasp pose is not top-down in the world frame.")

            output_root.mkdir(parents=True, exist_ok=True)
            task_associations_path = output_root / "task_associations.json"
            task_associations_path.write_text(
                json.dumps(task_associations, indent=2),
                encoding="utf-8",
            )
            planning_context = build_simulation_context(
                localization, task_associations["associations"], target_cad_registration,
                observation.world_T_camera, world_T_grasp, observation.robot_state,
            )
            perception = {
                "context": planning_context,
                "simulator_identity_used_for_assumed_human": task_associations.get(
                    "simulator_identity_used_for_assumed_human", False),
                "task_associations": task_associations,
                "target_depth_centroid_world_m": target_depth_centroid_world_m.tolist(),
                "target_cad_registration": target_cad_registration,
                "workspace_rgbd_pixels": int(workspace_mask.sum()),
                "localized_object_count": localization["object_count"],
                "artifacts": {
                    "localization": str(localization_paths["localization"]),
                    "annotation": str(localization_paths["annotated_rgb"]),
                    "task_associations": str(task_associations_path),
                    "vlm_visual_prompt": str(vlm_visual_prompt_path),
                    "vlm_result": str(vlm_result_path),
                    "vlm_request": str(output_root / "vlm_request.json"),
                    "vlm_provider_response": str(output_root / "vlm_provider_response.json"),
                    "target_cad_registration": target_cad_registration["result_path"],
                    "target_aligned_cad_cloud": target_cad_registration["aligned_cad_cloud_path"],
                    "target_augmented_cloud": target_cad_registration["augmented_cloud_path"],
                },
            }
        else:
            perception = frozen["perception"]
            planning_context = perception["context"]
            task_associations = perception["task_associations"]
            target_cad_registration = perception["target_cad_registration"]
            target_depth_centroid_world_m = np.array(perception["target_depth_centroid_world_m"])
            by_id = {item["object_id"]: item for item in planning_context["objects"]}
            world_T_grasp = np.array(by_id[task_associations["target_object_id"]]["world_T_grasp"])
            basket_world_m = np.array(by_id[task_associations["basket_object_id"]]["centroid_world_m"])
            target_world_m = np.array(target_cad_registration["registered_center_world_m"])

        output_root.mkdir(parents=True, exist_ok=True)
        inputs = {
            "schema_version": 1, "instruction": expected_instruction,
            "execution_config": _execution_config(task_index, initial_state_index),
            "initial_state_sha256": initial_state_sha256, "perception": perception,
            "settled_state_sha256": settled_state_sha256,
            "inputs_sha256": _json_sha256(perception),
        }
        (output_root / "execution_inputs.json").write_text(
            json.dumps(inputs, indent=2), encoding="utf-8",
        )
        progress["stage"] = "llm_planning"
        planner_path = output_root / "llm_plan.json"
        if frozen is None:
            planning = generate_simulation_plan(expected_instruction, planning_context, planner_path)
        else:
            reference = reference_pick_place_plan(
                task_associations["target_object_id"], task_associations["basket_object_id"],
            )
            validate_simulation_plan(reference, planning_context)
            planning = {"source": "reference_diagnostic", "status": "ready", "plan": reference,
                        "context": planning_context, "instruction": expected_instruction,
                        "reference_source": str(Path(reference_source).resolve())}
            planner_path.write_text(json.dumps(planning, indent=2), encoding="utf-8")
        video_frames.append(_agentview_rgb(raw_observation))

        def record_step(plan_index, planned_action, phase, phase_step, raw, action, reward, done, info):
            nonlocal first_success_step, step_count
            step_count += 1
            is_success = environment.check_success()
            action_log.append(
                {
                    "step": step_count,
                    "plan_action_index": plan_index,
                    "planned_action": planned_action,
                    "phase": phase,
                    "phase_step": int(phase_step),
                    "action": [float(value) for value in action],
                    "eef_position_m": [
                        float(value) for value in raw["robot0_eef_pos"]
                    ],
                    "eef_quaternion_xyzw": [
                        float(value) for value in raw["robot0_eef_quat"]
                    ],
                    "reward": float(reward),
                    "done": bool(done),
                    "is_success": is_success,
                }
            )
            if step_count % VIDEO_FRAME_STRIDE == 0:
                video_frames.append(_agentview_rgb(raw))
            if is_success:
                first_success_step = step_count
                raise _EpisodeSucceeded
            if done:
                raise _EnvironmentTerminated
            if step_count >= EPISODE_HORIZON_STEPS:
                raise _EpisodeHorizonReached

        final_raw_observation = raw_observation
        try:
            if planning["status"] == "ready":
                progress["stage"] = "execution"
                final_raw_observation = execute_simulation_plan(
                    environment, raw_observation, planning["plan"], planning_context, record_step,
                )
            else:
                termination_reason = "planning_" + planning["status"]
        except _EpisodeSucceeded:
            termination_reason = "success"
            final_raw_observation = environment.last_observation
        except _EpisodeHorizonReached:
            termination_reason = "episode_horizon"
            final_raw_observation = environment.last_observation
        except _EnvironmentTerminated:
            termination_reason = "environment_terminated_without_success"
            final_raw_observation = environment.last_observation
        except Exception as error:
            termination_reason = "execution_error"
            execution_error = str(error)
            final_raw_observation = environment.last_observation

        success = planning["status"] == "ready" and environment.check_success()
        if success and first_success_step is None:
            first_success_step = step_count
        if not success and termination_reason == "controller_completed":
            termination_reason = "controller_completed_without_success"
        video_frames.append(_agentview_rgb(final_raw_observation))
        final_agentview = sensor.capture(final_raw_observation)
        final_wrist = LiberoRGBDSensor(
            environment, "robot0_eye_in_hand"
        ).capture(final_raw_observation)
        final_agentview_paths = save_libero_observation(
            final_agentview,
            environment,
            output_root / "final_agentview",
        )
        final_wrist_paths = save_libero_observation(
            final_wrist,
            environment,
            output_root / "final_wrist",
        )

    progress["stage"] = "result_saving"
    plan_evaluation = evaluate_simulation_plan(
        planning, task_associations["target_object_id"], task_associations["basket_object_id"],
    )
    video_path = output_root / "episode.mp4"
    save_video(video_frames, video_path, fps=10.0)
    episode_path = output_root / "episode.json"
    episode_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": METHOD_NAME if frozen is None else "reference_plan_diagnostic",
                "run_status": "completed" if planning["status"] == "ready" else "planning_stopped",
                "planning_status": planning["status"],
                "planning_error": planning.get("error"),
                "plan_evaluation": plan_evaluation,
                "reference_source": None if reference_source is None else str(Path(reference_source).resolve()),
                "suite": SUITE_NAME,
                "task_index": task_index,
                "target_object": target_name,
                "instruction": observation.instruction,
                "instruction_matches_catalog": observation.instruction
                == expected_instruction,
                "success": success,
                "success_predicate": "LIBERO task environment check_success()",
                "first_success_step": first_success_step,
                "termination_reason": termination_reason,
                "execution_error": execution_error,
                "method_inputs": [
                    "rendered agentview RGB",
                    "rendered metric depth",
                    "camera intrinsics",
                    "world_T_camera",
                    "robot proprioception",
                    "language instruction",
                    "known target CAD prior retrieved from the semantic target description",
                    "assumed correct human object selection on deferral, if required",
                ],
                "simulator_object_identity_or_pose_used_by_method": perception.get(
                    "simulator_identity_used_for_assumed_human", False),
                "assumed_correct_human_on_deferral": ASSUME_CORRECT_HUMAN,
                "execution_poses_from_simulator_ground_truth": False,
                "seed": RANDOM_SEED,
                "initial_state_index": initial_state_index,
                "initial_state_sha256": initial_state_sha256,
                "settled_state_sha256": settled_state_sha256,
                "available_initial_state_count": initial_state_count,
                "episode_horizon_steps": EPISODE_HORIZON_STEPS,
                "control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "control_mode": environment.control_mode,
                "observation_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "initial_physics_settle_steps": INITIAL_PHYSICS_SETTLE_STEPS,
                "initial_physics_settle_duration_s": (
                    INITIAL_PHYSICS_SETTLE_STEPS / CONTROL_FREQUENCY_HZ
                ),
                "exact_rendering_cad_prior_used": target_cad_registration[
                    "exact_rendering_cad_prior_used"
                ],
                "workspace_rgbd_pixels": perception["workspace_rgbd_pixels"],
                "localized_object_count": perception["localized_object_count"],
                "semantic_associations": task_associations,
                "target_depth_centroid_world_m": target_depth_centroid_world_m.tolist(),
                "target_registered_center_world_m": target_world_m.tolist(),
                "pick_position_source": (
                    "registered CAD center shifted toward the object top to maintain "
                    "Panda hand clearance"
                ),
                "target_grasp_point_world_m": world_T_grasp[:3, 3].tolist(),
                "center_to_grasp_height_offset_m": float(
                    world_T_grasp[2, 3] - target_world_m[2]
                ),
                "world_T_grasp": world_T_grasp.tolist(),
                "world_T_grasp_convention": (
                    "Desired robosuite grip-site pose in the MuJoCo world frame; "
                    "CAD Y-up/Z-up is handled explicitly, tool Z points down, and "
                    "the closest 180-degree-equivalent wrist yaw is used."
                ),
                "simulation_controller": (
                    "robosuite OSC_POSE using robot0_eef_pos and robot0_eef_quat"
                ),
                "place_position_source": "depth-localized basket centroid",
                "basket_centroid_world_m": basket_world_m.tolist(),
                "target_cad_registration": target_cad_registration,
                "action_steps": step_count,
                "actions": action_log,
                "artifacts": {
                    **perception["artifacts"],
                    "planning": str(planner_path),
                    "execution_inputs": str(output_root / "execution_inputs.json"),
                    "video": str(video_path),
                    "final_agentview_rgb": str(final_agentview_paths["rgb"]),
                    "final_wrist_rgb": str(final_wrist_paths["rgb"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Instruction: {expected_instruction}")
    print(f"Plan: {planning['status']}; termination: {termination_reason}")
    print(f"LIBERO task success: {success}; action steps: {step_count}")
    print(f"Episode report: {episode_path}")
    return episode_path


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _execution_config(task_index, initial_state_index):
    simulation_dir = Path(__file__).resolve().parents[1] / "simulation"
    return {
        "suite": SUITE_NAME, "task_index": task_index, "initial_state_index": initial_state_index,
        "seed": RANDOM_SEED, "control_mode": CONTROL_MODE,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ, "episode_horizon_steps": EPISODE_HORIZON_STEPS,
        "observation_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "initial_physics_settle_steps": INITIAL_PHYSICS_SETTLE_STEPS,
        "assumed_correct_human_on_deferral": ASSUME_CORRECT_HUMAN,
        "association_source_sha256": {name: hashlib.sha256((simulation_dir / name).read_bytes()).hexdigest()
                                      for name in ("libero_joint_association.py", "libero_assumed_human.py")},
        "cad_catalog_sha256": hashlib.sha256((simulation_dir.parent / "CAD/libero_object_library.json").read_bytes()).hexdigest(),
        "joint_protocol_sha256": hashlib.sha256((simulation_dir.parent / "scripts/run_vlm_multi_object_pilot.py").read_bytes()).hexdigest(),
        "simulation_versions": {name: importlib.metadata.version(name)
                                for name in ("mujoco", "robosuite", "hf_libero", "numpy")},
        "controller_sha256": hashlib.sha256((simulation_dir / "libero_control.py").read_bytes()).hexdigest(),
        "plan_executor_sha256": hashlib.sha256((simulation_dir / "libero_planning.py").read_bytes()).hexdigest(),
    }


def replay_with_reference_plan(source_dir, output_root):
    """Replay a failed LLM episode with frozen perception and a reference plan.

    No VLM, CAD registration, or LLM request is made. The source is never changed;
    these extra executions are diagnostic and excluded from the main trial count.
    """
    source_dir = Path(source_dir).resolve()
    source = json.loads((source_dir / "episode.json").read_text())
    if source.get("method") != METHOD_NAME or source.get("success") is not False:
        raise ValueError("Reference replay requires an unsuccessful LLM-runner episode.")
    if "planning_status" not in source:
        raise ValueError("Episode did not reach planning; no plan-only replay is possible.")
    original_planning = json.loads((source_dir / "llm_plan.json").read_text())
    if original_planning["status"] != source["planning_status"]:
        raise ValueError("Source episode and saved planning status disagree.")
    frozen = json.loads((source_dir / "execution_inputs.json").read_text())
    config = frozen["execution_config"]
    if (source["task_index"] != config["task_index"]
            or source["initial_state_index"] != config["initial_state_index"]
            or source["initial_state_sha256"] != frozen["initial_state_sha256"]):
        raise ValueError("Source episode and frozen inputs disagree.")
    result_path = main(config["task_index"], initial_state_index=config["initial_state_index"],
                       output_root=output_root, reference_source=source_dir)
    replay = json.loads(result_path.read_text())
    reference_planning = json.loads((result_path.parent / "llm_plan.json").read_text())
    same_executable_plan = (
        original_planning["status"] == "ready"
        and original_planning["plan"]["actions"] == reference_planning["plan"]["actions"]
    )
    if not replay["success"]:
        interpretation = "Cause remains unresolved; both plans failed."
    elif same_executable_plan:
        interpretation = "Cause remains unresolved; the same plan produced different outcomes."
    else:
        interpretation = "Reference plan succeeded with unchanged perception and controller."
    comparison = {
        "source_episode": str(source_dir / "episode.json"), "reference_episode": str(result_path),
        "original_success": source["success"], "reference_success": replay["success"],
        "same_perception_inputs_sha256": frozen["inputs_sha256"],
        "same_initial_state_sha256": frozen["initial_state_sha256"],
        "same_settled_state_sha256": frozen["settled_state_sha256"],
        "original_planning_status": original_planning["status"],
        "same_executable_plan": same_executable_plan,
        "planning_contribution_supported": bool(replay["success"] and not same_executable_plan),
        "interpretation": interpretation,
        "conditional_on_correct_upstream_outputs": True,
        "included_in_main_evaluation": False,
    }
    (result_path.parent / "reference_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8",
    )
    return result_path


def _write_pipeline_failure(
    output_root,
    task_index,
    target_slug,
    instruction,
    error,
    initial_state_index=INITIAL_STATE_INDEX,
    failure_stage=None,
    simulator_identity_used=False,
):
    """Preserve a pre-control pipeline failure without hiding the exception."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    episode_path = output_root / "episode.json"
    if episode_path.is_file():
        failure_path = output_root / "latest_pipeline_failure.json"
    else:
        failure_path = episode_path

    state_hash = None
    state_hash_error = None
    try:
        state_hash = _official_initial_state_sha256(task_index, initial_state_index)
    except Exception as hash_error:
        state_hash_error = str(hash_error)

    failure_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": METHOD_NAME,
                "run_status": "pipeline_error",
                "suite": SUITE_NAME,
                "task_index": task_index,
                "target_object": target_slug.replace("_", " "),
                "instruction": instruction,
                "success": False,
                "success_predicate": "LIBERO task environment check_success()",
                "termination_reason": "pipeline_error",
                "failure_observed_at": failure_stage,
                "causal_failure_stage": None,
                "execution_error": f"{type(error).__name__}: {error}",
                "simulator_object_identity_or_pose_used_by_method": simulator_identity_used,
                "assumed_correct_human_on_deferral": ASSUME_CORRECT_HUMAN,
                "execution_poses_from_simulator_ground_truth": False,
                "seed": RANDOM_SEED,
                "initial_state_index": initial_state_index,
                "initial_state_sha256": state_hash,
                "initial_state_hash_error": state_hash_error,
                "episode_horizon_steps": EPISODE_HORIZON_STEPS,
                "control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "control_mode": CONTROL_MODE,
                "observation_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "initial_physics_settle_steps": INITIAL_PHYSICS_SETTLE_STEPS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _official_initial_state_sha256(task_index, initial_state_index):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    states = np.asarray(suite.get_task_init_states(task_index))
    value = np.ascontiguousarray(states[int(initial_state_index)])
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def camera_point_to_world(point_camera_m, world_T_camera):
    point = np.asarray([*point_camera_m, 1.0], dtype=float)
    return (np.asarray(world_T_camera, dtype=float) @ point)[:3]


def task_associations_from_vlm_result(result_path, target_name="milk"):
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    associations = list(payload.get("associations", []))
    by_description = {
        str(item.get("target_description", "")).strip().lower(): item
        for item in associations
    }
    required = (target_name.lower(), "basket")
    if any(description not in by_description for description in required):
        raise RuntimeError(
            f"VLM result must contain independent associations for {target_name!r} "
            f"and 'basket'; received {sorted(by_description)}."
        )
    target_association = by_description[target_name.lower()]
    basket_association = by_description["basket"]
    resolved_states = {
        "vlm_accepted",
        "human_confirmed",
        "human_corrected",
        "assumed_human_confirmed",
        "assumed_human_corrected",
        "target_not_present",
    }
    unresolved = [
        item
        for item in (target_association, basket_association)
        if item.get("resolution") not in resolved_states
    ]
    if unresolved:
        descriptions = ", ".join(
            str(item.get("target_description")) for item in unresolved
        )
        raise RuntimeError(
            f"Semantic association requires human clarification for: {descriptions}."
        )
    target_object_id = target_association.get("final_object_id")
    basket_object_id = basket_association.get("final_object_id")
    if target_object_id is None or basket_object_id is None:
        missing = target_name if target_object_id is None else "basket"
        raise RuntimeError(f"Target not present: {missing!r}; task execution stopped.")
    if target_object_id == basket_object_id:
        raise RuntimeError(
            f"Semantic associations assigned {target_name} and basket to the same "
            "localized object."
        )
    return {
        "method": payload.get("method", "independent semantic association over localized RGB-D candidates"),
        "simulator_identity_used_for_assumed_human": payload.get("simulator_identity_used_for_assumed_human", False),
        "target_description": target_name,
        "target_object_id": target_object_id,
        "basket_description": "basket",
        "basket_object_id": basket_object_id,
        "associations": [target_association, basket_association],
    }


def save_video(frames, output_path, fps):
    if not frames:
        raise RuntimeError("No frames were captured for the LIBERO episode video.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create the episode video at {output_path}.")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Episode video frames have inconsistent resolutions.")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _agentview_rgb(raw_observation):
    image = np.asarray(raw_observation["agentview_image"], dtype=np.uint8)
    return np.ascontiguousarray(np.flip(image, axis=0))


if __name__ == "__main__":
    try:
        arguments = sys.argv[1:]
        if len(arguments) > 1:
            raise SystemExit("Usage: python -m scripts.libero_task_execution [task_index]")
        result_path = main(DEFAULT_TASK_INDEX if not arguments else int(arguments[0]))
        if not json.loads(result_path.read_text())["success"]:
            raise SystemExit("Episode was unsuccessful; see its saved report.")
    except LiberoIntegrationError as error:
        raise SystemExit(f"LIBERO task execution failed: {error}") from error
