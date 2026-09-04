"""Run one RGB-D/VLM/CAD LIBERO-Object episode."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from simulation.libero_cad import register_libero_cad_to_observation
from simulation.libero_control import (
    OPEN_GRIPPER,
    execute_top_grasp_and_place,
    hold_gripper,
    top_down_grasp_pose,
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
    associate_targets_from_localization,
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
RUN_VARIANT = "768x768_raw_likelihood_gate_grasp_pose_v6"
# Development operating point selected on the 500-trial LIBERO
# calibration/validation partitions. It is specific to this simulation
# experiment and is not a calibrated probability of correctness.
LIBERO_RAW_ASSOCIATION_THRESHOLD = 0.9999832372181827
CONTROL_FREQUENCY_HZ = 20
EPISODE_HORIZON_STEPS = 280
CONTROL_MODE = "relative"
RANDOM_SEED = 1000
INITIAL_STATE_INDEX = 0
INITIAL_PHYSICS_SETTLE_STEPS = 10
VIDEO_FRAME_STRIDE = 2
METHOD_NAME = "rgbd_vlm_cad_scripted_controller"


class _EpisodeSucceeded(RuntimeError):
    pass


class _EpisodeHorizonReached(RuntimeError):
    pass


class _EnvironmentTerminated(RuntimeError):
    pass


def main(task_index=DEFAULT_TASK_INDEX):
    task_index, target_slug, instruction = libero_object_task(task_index)
    output_root = episode_result_dir(
        PROPOSED_METHOD_FOLDER,
        task_index,
        INITIAL_STATE_INDEX,
        run_variant=RUN_VARIANT,
    )
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise SystemExit(
            f"Refusing to overwrite the existing proposed-method result: {output_root}. "
            "Move the directory aside before deliberately rerunning this task."
        )
    try:
        return _run_task(task_index)
    except Exception as error:
        _write_pipeline_failure(
            output_root,
            task_index,
            target_slug,
            instruction,
            error,
        )
        raise


def _run_task(task_index):
    task_index, target_slug, expected_instruction = libero_object_task(task_index)
    target_name = target_slug.replace("_", " ")
    output_root = episode_result_dir(
        PROPOSED_METHOD_FOLDER,
        task_index,
        INITIAL_STATE_INDEX,
        run_variant=RUN_VARIANT,
    )
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
            init_state_index=INITIAL_STATE_INDEX,
        )
        initial_state_sha256 = environment.last_init_state_sha256
        initial_state_count = environment.initial_state_count
        raw_observation = hold_gripper(
            environment,
            raw_observation,
            OPEN_GRIPPER,
            "initial_physics_settle",
            INITIAL_PHYSICS_SETTLE_STEPS,
        )
        environment.set_control_mode(CONTROL_MODE)
        sensor = LiberoRGBDSensor(environment, CAMERA_NAME)
        observation = sensor.capture(raw_observation)
        if observation.instruction != expected_instruction:
            raise RuntimeError(
                "LIBERO instruction does not match the experiment catalog: "
                f"expected {expected_instruction!r}, received {observation.instruction!r}."
            )
        capture_paths = save_libero_observation(
            observation,
            environment,
            perception_root / "capture",
        )
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
        )
        vlm_result_path = associate_targets_from_localization(
            [target_name, "basket"],
            image_path=vlm_visual_prompt_path,
            localization_path=localization_paths["localization"],
            output_path=output_root / "vlm_result.json",
            threshold=LIBERO_RAW_ASSOCIATION_THRESHOLD,
            human_resolver=opencv_human_resolver,
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
        video_frames.append(_agentview_rgb(raw_observation))

        def record_step(phase, phase_step, raw, action, reward, done, info):
            nonlocal first_success_step, step_count
            step_count += 1
            is_success = environment.check_success()
            action_log.append(
                {
                    "step": step_count,
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

        try:
            final_raw_observation = execute_top_grasp_and_place(
                environment,
                raw_observation,
                world_T_grasp,
                basket_world_m,
                callback=record_step,
            )
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

        success = environment.check_success()
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

    video_path = output_root / "episode.mp4"
    save_video(video_frames, video_path, fps=10.0)
    episode_path = output_root / "episode.json"
    episode_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": METHOD_NAME,
                "run_status": "completed",
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
                ],
                "simulator_object_identity_or_pose_used_by_method": False,
                "seed": RANDOM_SEED,
                "initial_state_index": INITIAL_STATE_INDEX,
                "initial_state_sha256": initial_state_sha256,
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
                "workspace_rgbd_pixels": int(workspace_mask.sum()),
                "localized_object_count": int(localization["object_count"]),
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
                    "localization": str(localization_paths["localization"]),
                    "annotation": str(localization_paths["annotated_rgb"]),
                    "task_associations": str(task_associations_path),
                    "vlm_visual_prompt": str(vlm_visual_prompt_path),
                    "vlm_result": str(vlm_result_path),
                    "target_cad_registration": target_cad_registration["result_path"],
                    "target_aligned_cad_cloud": target_cad_registration[
                        "aligned_cad_cloud_path"
                    ],
                    "target_augmented_cloud": target_cad_registration[
                        "augmented_cloud_path"
                    ],
                    "video": str(video_path),
                    "final_agentview_rgb": str(final_agentview_paths["rgb"]),
                    "final_wrist_rgb": str(final_wrist_paths["rgb"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Instruction: {observation.instruction}")
    print(f"Localized candidates: {localization['object_count']}")
    print(f"VLM providers: {', '.join(task_associations['providers'])}")
    print(
        f"Associated target ({target_name}): "
        f"{task_associations['target_object_id']}"
    )
    print(f"Associated basket: {task_associations['basket_object_id']}")
    print(
        "Target depth centroid in world frame (m): "
        f"{np.round(target_depth_centroid_world_m, 4).tolist()}"
    )
    print(
        "Target CAD-registered center in world frame (m): "
        f"{np.round(target_world_m, 4).tolist()}"
    )
    print(
        "Target CAD registration RMSE (m): "
        f"{target_cad_registration['constrained_rmse_m']:.6f}"
    )
    print(f"Basket centroid in world frame (m): {np.round(basket_world_m, 4).tolist()}")
    print(f"Action steps: {step_count}")
    print(f"Initial state: {INITIAL_STATE_INDEX} ({initial_state_sha256[:12]}...)")
    print(f"Termination: {termination_reason}")
    print(f"LIBERO task success: {success}")
    print(f"Episode video: {video_path}")
    print(f"Episode report: {episode_path}")
    print(
        "MuJoCo instance identity, segmentation, and ground-truth object pose "
        "were not used by the method"
    )
    if execution_error is not None:
        raise SystemExit(f"Task execution failed: {execution_error}")
    if not success:
        raise SystemExit("Task execution finished, but LIBERO reported failure.")


def _write_pipeline_failure(
    output_root,
    task_index,
    target_slug,
    instruction,
    error,
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
        state_hash = _official_initial_state_sha256(task_index, INITIAL_STATE_INDEX)
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
                "execution_error": f"{type(error).__name__}: {error}",
                "simulator_object_identity_or_pose_used_by_method": False,
                "seed": RANDOM_SEED,
                "initial_state_index": INITIAL_STATE_INDEX,
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
        "method": "independent semantic association over localized RGB-D candidates",
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
        main(DEFAULT_TASK_INDEX if not arguments else int(arguments[0]))
    except LiberoIntegrationError as error:
        raise SystemExit(f"LIBERO task execution failed: {error}") from error
