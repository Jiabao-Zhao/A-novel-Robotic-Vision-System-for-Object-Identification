import json
from pathlib import Path

import cv2
import numpy as np

from simulation.libero_cad import register_libero_cad_to_observation
from simulation.libero_control import (
    OPEN_GRIPPER,
    execute_top_grasp_and_place,
    hold_gripper,
)
from simulation.libero_env import LiberoIntegrationError, LiberoTaskEnvironment
from simulation.libero_experiment import PROPOSED_METHOD_FOLDER, episode_result_dir
from simulation.libero_io import save_libero_observation
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.perception_adapter import run_libero_localization
from vlm_module import classify_from_localization, create_roi_contact_sheet


SUITE_NAME = "libero_object"
TASK_INDEX = 7
CAMERA_NAME = "agentview"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CONTROL_FREQUENCY_HZ = 20
EPISODE_HORIZON_STEPS = 280
CONTROL_MODE = "relative"
RANDOM_SEED = 1000
INITIAL_STATE_INDEX = 0
INITIAL_PHYSICS_SETTLE_STEPS = 10
OUTPUT_ROOT = episode_result_dir(PROPOSED_METHOD_FOLDER, TASK_INDEX, INITIAL_STATE_INDEX)
VIDEO_FRAME_STRIDE = 2
METHOD_NAME = "rgbd_vlm_cad_scripted_controller"


class _EpisodeSucceeded(RuntimeError):
    pass


class _EpisodeHorizonReached(RuntimeError):
    pass


class _EnvironmentTerminated(RuntimeError):
    pass


def main():
    perception_root = OUTPUT_ROOT / "perception"
    action_log = []
    video_frames = []
    step_count = 0
    first_success_step = None
    termination_reason = "controller_completed"
    execution_error = None

    with LiberoTaskEnvironment(
        suite_name=SUITE_NAME,
        task_index=TASK_INDEX,
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
            OUTPUT_ROOT / "vlm_roi_contact_sheet.png",
        )
        vlm_result_path = classify_from_localization(
            observation.instruction,
            image_path=vlm_visual_prompt_path,
            localization_path=localization_paths["localization"],
            output_path=OUTPUT_ROOT / "vlm_result.json",
        )
        grounding = task_roles_from_vlm_result(vlm_result_path)
        objects_by_id = {
            item["object_id"]: item for item in localization["objects"]
        }
        milk = objects_by_id[grounding["milk_object_id"]]
        basket = objects_by_id[grounding["basket_object_id"]]
        milk_depth_centroid_world_m = camera_point_to_world(
            milk["centroid_3d_m"], observation.world_T_camera
        )
        basket_world_m = camera_point_to_world(
            basket["centroid_3d_m"], observation.world_T_camera
        )
        milk_cad_registration = register_libero_cad_to_observation(
            object_type=grounding["milk_object_type"],
            object_id=grounding["milk_object_id"],
            localization=localization,
            world_T_camera=observation.world_T_camera,
            output_dir=OUTPUT_ROOT / "cad_registration" / "milk",
        )
        milk_world_m = np.asarray(
            milk_cad_registration["registered_center_world_m"],
            dtype=float,
        )

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        grounding_path = OUTPUT_ROOT / "grounding.json"
        grounding_path.write_text(json.dumps(grounding, indent=2), encoding="utf-8")
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
                milk_world_m,
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
            OUTPUT_ROOT / "final_agentview",
        )
        final_wrist_paths = save_libero_observation(
            final_wrist,
            environment,
            OUTPUT_ROOT / "final_wrist",
        )

    video_path = OUTPUT_ROOT / "episode.mp4"
    save_video(video_frames, video_path, fps=10.0)
    episode_path = OUTPUT_ROOT / "episode.json"
    episode_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": METHOD_NAME,
                "suite": SUITE_NAME,
                "task_index": TASK_INDEX,
                "instruction": observation.instruction,
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
                    "known milk CAD prior selected from the VLM classification",
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
                "exact_rendering_cad_prior_used": milk_cad_registration[
                    "exact_rendering_cad_prior_used"
                ],
                "workspace_rgbd_pixels": int(workspace_mask.sum()),
                "localized_object_count": int(localization["object_count"]),
                "grounding": grounding,
                "milk_depth_centroid_world_m": milk_depth_centroid_world_m.tolist(),
                "milk_registered_center_world_m": milk_world_m.tolist(),
                "pick_position_source": "registered milk CAD axis-aligned-box center",
                "place_position_source": "depth-localized basket centroid",
                "basket_centroid_world_m": basket_world_m.tolist(),
                "milk_cad_registration": milk_cad_registration,
                "action_steps": step_count,
                "actions": action_log,
                "artifacts": {
                    "localization": str(localization_paths["localization"]),
                    "annotation": str(localization_paths["annotated_rgb"]),
                    "grounding": str(grounding_path),
                    "vlm_visual_prompt": str(vlm_visual_prompt_path),
                    "vlm_result": str(vlm_result_path),
                    "milk_cad_registration": milk_cad_registration["result_path"],
                    "milk_aligned_cad_cloud": milk_cad_registration[
                        "aligned_cad_cloud_path"
                    ],
                    "milk_augmented_cloud": milk_cad_registration["augmented_cloud_path"],
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
    print(f"VLM provider: {grounding['provider']}")
    print(f"Grounded milk: {grounding['milk_object_id']}")
    print(f"Grounded basket: {grounding['basket_object_id']}")
    print(
        "Milk depth centroid in world frame (m): "
        f"{np.round(milk_depth_centroid_world_m, 4).tolist()}"
    )
    print(
        "Milk CAD-registered center in world frame (m): "
        f"{np.round(milk_world_m, 4).tolist()}"
    )
    print(
        "Milk CAD registration RMSE (m): "
        f"{milk_cad_registration['constrained_rmse_m']:.6f}"
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


def camera_point_to_world(point_camera_m, world_T_camera):
    point = np.asarray([*point_camera_m, 1.0], dtype=float)
    return (np.asarray(world_T_camera, dtype=float) @ point)[:3]


def task_roles_from_vlm_result(result_path):
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    normalized = payload.get("normalized_result", {})
    if normalized.get("needs_human_clarification"):
        raise RuntimeError(
            "VLM grounding requires clarification: "
            f"{normalized.get('clarification_reason')}"
        )
    selected = list(normalized.get("selected_objects", []))
    milk = [
        item
        for item in selected
        if item.get("instruction_role") == "moved_object"
        and "milk" in str(item.get("object_type", "")).lower()
    ]
    basket = [
        item
        for item in selected
        if item.get("instruction_role") == "reference_object"
        and "basket" in str(item.get("object_type", "")).lower()
    ]
    if len(milk) != 1 or len(basket) != 1:
        raise RuntimeError(
            "VLM must select exactly one milk moved_object and one basket "
            f"reference_object; received selected objects: {selected}"
        )
    if milk[0]["object_id"] == basket[0]["object_id"]:
        raise RuntimeError("VLM assigned milk and basket to the same localized object.")
    return {
        "provider": payload.get("provider"),
        "method": "existing VLM module over enlarged depth-localized RGB crops",
        "milk_object_id": milk[0]["object_id"],
        "milk_object_type": milk[0]["object_type"],
        "basket_object_id": basket[0]["object_id"],
        "selected_objects": selected,
        "object_evaluations": normalized.get("object_evaluations", []),
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
        main()
    except LiberoIntegrationError as error:
        raise SystemExit(f"LIBERO task execution failed: {error}") from error
