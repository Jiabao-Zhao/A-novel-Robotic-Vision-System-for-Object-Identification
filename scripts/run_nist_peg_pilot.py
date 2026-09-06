"""One zero-shot episode per method, with supplied hole pose and unknown part pose."""

from datetime import datetime, timezone
import hashlib
import json

import cv2
import numpy as np

from simulation.nist_task_board_1 import NistTaskBoardEnvironment, OUTPUT_DIR
from simulation.nist_peg_task import (
    HORIZON, PegEvaluator, TARGET_DESCRIPTION, TASK_TEXT, WORLD_T_HOLE,
    estimate_peg_pose, grasp_and_insert, localize_parts,
)
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.libero_io import save_libero_observation


class EpisodeFinished(Exception):
    pass


def framework(environment, raw, output, callback, confirmed_trial=None):
    from vlm_module import associate_targets_from_localization, create_roi_contact_sheet, resolve_association

    observation = LiberoRGBDSensor(environment).capture(raw)
    paths = save_libero_observation(observation, environment, output / "capture")
    if confirmed_trial is None:
        localization, paths = localize_parts(observation, paths["rgb"], output)
        image = create_roi_contact_sheet(output / "capture/rgb.png", paths["localization"], output / "vlm_prompt.png")
        result_path = associate_targets_from_localization(
            [TARGET_DESCRIPTION], image, paths["localization"], output / "association.json",
            threshold=.75, human_resolver=None,
        )
        association = json.loads(result_path.read_text())["associations"][0]
    else:
        source_dir, confirmed_id = confirmed_trial
        localization = json.loads((source_dir / "point_cloud/point_cloud_localization.json").read_text())
        original = json.loads((source_dir / "association.json").read_text())["associations"][0]
        association = resolve_association(original, .75, human_resolver=lambda _: confirmed_id)
        association["diagnostics"]["human_response_time_s"] = None
        association["diagnostics"]["human_confirmation_collected_before_resume"] = True
        association["confirmation_source"] = "explicit user selection in Codex"
        association["original_trial"] = str(source_dir)
        (output / "association.json").write_text(json.dumps({"associations": [association]}, indent=2))
    if association["final_object_id"] is None:
        return f"association_{association['resolution']}"
    grasp, registration = estimate_peg_pose(localization, association["final_object_id"],
                                            observation.world_T_camera, output / "cad")
    (output / "grasp.json").write_text(json.dumps({"world_T_grasp": grasp.tolist(),
        "cad_alignment_rmse_m": registration["constrained_rmse_m"],
        "yaw_observable": False, "object_pose_source": "RGB-D and semantic CAD prior"}, indent=2))
    grasp_and_insert(environment, raw, grasp, callback)
    return "controller_finished"


def vla(environment, raw, output, callback):
    import torch
    from pathlib import Path
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import LiberoProcessorStep
    from transformers import AutoTokenizer

    checkpoint = Path.home() / ".cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    torch.manual_seed(1000)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    token_count = len(tokenizer(TASK_TEXT + "\n")["input_ids"])
    if token_count > policy.config.tokenizer_max_length:
        raise ValueError("Goal instruction would be truncated by the checkpoint tokenizer")
    (output / "policy.json").write_text(json.dumps({"checkpoint": str(checkpoint),
        "task": TASK_TEXT, "task_tokens": token_count, "fine_tuning": False,
        "hole_pose_input": "numeric world XYZ in language; vertical axis stated",
        "limitation": "checkpoint has no dedicated goal-pose channel; numeric goal following is unvalidated",
        "image_preprocessing": "native 256x256 RGB; official LIBERO 180-degree rotation",
        "state": "EEF XYZ + axis-angle + gripper qpos (8D)"}, indent=2))
    processor = LiberoProcessorStep()
    policy.reset()
    for index in range(HORIZON):
        batch = {"task": TASK_TEXT, "observation.robot_state": {
            "eef": {"pos": torch.tensor(raw["robot0_eef_pos"]).unsqueeze(0),
                    "quat": torch.tensor(raw["robot0_eef_quat"]).unsqueeze(0)},
            "gripper": {"qpos": torch.tensor(raw["robot0_gripper_qpos"]).unsqueeze(0)},
        }}
        for camera, key in (("agentview", "image"), ("robot0_eye_in_hand", "image2")):
            frame = raw[f"{camera}_image"]
            batch[f"observation.images.{key}"] = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
        with torch.inference_mode():
            action = post(policy.select_action(pre(processor.observation(batch)))).squeeze(0).cpu().numpy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("VLA returned an invalid action")
        action = np.clip(action, -1, 1)
        raw, reward, done, info = environment.step(action)
        callback("vla", index, raw, action, reward, done, info)
    return "horizon"


def run_method(method, output, confirmed_trial=None):
    if method not in {"framework", "framework_human", "vla"}:
        raise ValueError(f"Unknown method: {method}")
    if (method == "framework_human") != (confirmed_trial is not None):
        raise ValueError("Human-assisted runs require an explicitly confirmed source trial and object ID")
    output.mkdir(parents=True)
    # Preserve native checkpoint input resolution; record this asymmetry rather
    # than presenting the pilot as a resolution-controlled comparison.
    size = 256 if method == "vla" else 768
    with NistTaskBoardEnvironment(image_size=size) as environment:
        environment.instruction = TASK_TEXT
        raw = environment.reset(seed=1000)
        environment.set_control_mode("relative")
        for _ in range(40):
            raw, _, _, _ = environment.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
        state = np.r_[environment.sim.data.qpos, environment.sim.data.qvel]
        state_hash = hashlib.sha256(state.tobytes()).hexdigest()
        if confirmed_trial is not None:
            source_evaluation = json.loads((confirmed_trial[0] / "evaluation.json").read_text())
            if state_hash != source_evaluation["initial_state_sha256"]:
                raise ValueError("Scene state differs from the human-confirmed localization")
        evaluator = PegEvaluator(environment)
        log = []
        writer = cv2.VideoWriter(str(output / "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20., (size, size))
        if not writer.isOpened():
            raise RuntimeError("Could not create episode video")
        writer.write(cv2.cvtColor(raw["agentview_image"][::-1].copy(), cv2.COLOR_RGB2BGR))

        def record(phase, phase_step, raw, action, reward, done, info):
            scores = evaluator.score()
            log.append({"step": len(log) + 1, "phase": phase, "action": np.asarray(action).tolist(),
                        "eef_world_m": np.asarray(raw["robot0_eef_pos"]).tolist(), **scores})
            writer.write(cv2.cvtColor(raw["agentview_image"][::-1].copy(), cv2.COLOR_RGB2BGR))
            if len(log) % 50 == 0:
                print(f"{method}: step {len(log)}, phase={phase}, correct_part_lifted={scores['correct_part_lifted']}", flush=True)
            if scores["success"] or len(log) >= HORIZON or done:
                raise EpisodeFinished

        error = None
        try:
            termination = (vla(environment, raw, output, record) if method == "vla" else
                           framework(environment, raw, output, record, confirmed_trial))
        except EpisodeFinished:
            termination = "success" if evaluator.success else "horizon_or_environment_done"
        except Exception as exc:
            termination, error = "error", f"{type(exc).__name__}: {exc}"
            print(f"{method}: {error}", flush=True)
        finally:
            writer.release()
        result = {"method": method, "task": TASK_TEXT, "world_T_hole": WORLD_T_HOLE.tolist(),
                  "hole_pose_source": "supplied scene-design pose", "part_pose_supplied": False,
                  "initial_state_sha256": state_hash, "resolution": [size, size],
                  "steps": len(log), "termination": termination, "error": error,
                  "gate": (None if method == "vla" else
                           "raw generated-label likelihood >= 0.75; provisional, not NIST calibrated"),
                  "success_definition": "correct peg lifted, then >=5mm insertion with positive shaft clearance for 5 steps",
                  "evaluation_ground_truth_used_as_method_input": False, **evaluator.score(advance=False)}
        (output / "evaluation.json").write_text(json.dumps(result, indent=2))
        (output / "actions.json").write_text(json.dumps(log, indent=2))
        cv2.imwrite(str(output / "final.png"), cv2.cvtColor(environment.last_observation["agentview_image"][::-1].copy(), cv2.COLOR_RGB2BGR))
        print(method, json.dumps(result), flush=True)
        return result


def main():
    root = OUTPUT_DIR / "peg_pilot" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = [run_method(method, root / method) for method in ("framework", "vla")]
    summary = {"episodes": results,
               "matched_initial_state": len({r["initial_state_sha256"] for r in results}) == 1,
               "interpretation": "single-scene zero-shot pilot; different image resolutions and goal interfaces; not a method ranking"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved pilot: {root}")


if __name__ == "__main__":
    main()
