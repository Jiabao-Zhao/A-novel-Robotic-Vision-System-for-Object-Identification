"""Ground-truth-assisted diagnostic: execute the saved in-hand correction."""

from datetime import datetime, timezone
import hashlib
import json

import cv2
import numpy as np

from scripts.compute_nist_hole_alignment import TRIAL
from simulation.libero_control import move_eef_to_pose
from simulation.nist_peg_task import PegEvaluator
from simulation.nist_task_board_1 import NistTaskBoardEnvironment


def main():
    output = TRIAL / "corrected_insertion" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True)
    source = json.loads((TRIAL / "framework_human/actions.json").read_text())
    alignment = json.loads((TRIAL / "offset_diagnosis/hole_alignment.json").read_text())
    expected = json.loads((TRIAL / "framework_human/evaluation.json").read_text())["initial_state_sha256"]
    with NistTaskBoardEnvironment(image_size=512) as environment:
        raw = environment.reset(seed=1000)
        environment.set_control_mode("relative")
        for _ in range(40):
            raw, _, _, _ = environment.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
        state_hash = hashlib.sha256(np.r_[environment.sim.data.qpos, environment.sim.data.qvel].tobytes()).hexdigest()
        if state_hash != expected:
            raise RuntimeError("Initial state differs from saved trial")
        evaluator = PegEvaluator(environment)
        records = []
        writer = cv2.VideoWriter(str(output / "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20., (512, 512))
        if not writer.isOpened():
            raise RuntimeError("Cannot create video")

        def record(phase, index, observation, action, reward, done, info):
            scores = evaluator.score()
            records.append({"step": len(records) + 1, "phase": phase,
                            "action": np.asarray(action).tolist(),
                            "eef_world_m": np.asarray(observation["robot0_eef_pos"]).tolist(), **scores})
            writer.write(cv2.cvtColor(observation["agentview_image"][::-1].copy(), cv2.COLOR_RGB2BGR))
            if len(records) % 50 == 0:
                print(phase, len(records), scores, flush=True)

        error = None
        replay_error = 0.
        try:
            for item in source:
                if item["step"] > 192:
                    break
                raw, reward, done, info = environment.step(np.array(item["action"]))
                record(item["phase"], item["step"], raw, item["action"], reward, done, info)
                replay_error = max(replay_error, np.linalg.norm(np.array(raw["robot0_eef_pos"]) - item["eef_world_m"]),
                                   np.linalg.norm(environment.sim.data.body_xpos[evaluator.target_id] - item["evaluation_only_target_center_world_m"]))
            if replay_error > 1e-7:
                raise RuntimeError("Replay differs from saved above-hole state")
            target = np.array(alignment["targets"]["preinsert_tip_10mm_above"]["world_T_eef"])
            high = target.copy()
            high[2, 3] += .070
            for phase, goal in [("correct_axis_above_hole", high), ("corrected_preinsert", target)]:
                raw = move_eef_to_pose(environment, raw, goal, 1., phase, record,
                                       tolerance_m=.001, orientation_tolerance_rad=.003, max_steps=120)
            for tip_height_mm in (4, 1, -2, -5, -6):
                goal = target.copy()
                goal[2, 3] += (tip_height_mm - 10) / 1000
                raw = move_eef_to_pose(environment, raw, goal, 1., f"corrected_tip_{tip_height_mm}mm", record,
                                       tolerance_m=.001 if tip_height_mm > 0 else .0003,
                                       orientation_tolerance_rad=.003, max_steps=120)
                if evaluator.success:
                    break
        except RuntimeError as exc:
            error = str(exc)
        finally:
            writer.release()
        result = {"method": "one-shot ground-truth-assisted in-hand compensation",
                  "evaluation_ground_truth_used_as_method_input": True,
                  "contact_parameters_changed": False, "initial_state_sha256": state_hash,
                  "approach_tolerance_mm": 1., "insertion_tolerance_mm": .3,
                  "max_replay_error_m": float(replay_error), "error": error,
                  "steps": len(records), **evaluator.score(advance=False)}
        (output / "evaluation.json").write_text(json.dumps(result, indent=2))
        (output / "actions.json").write_text(json.dumps(records, indent=2))
        cv2.imwrite(str(output / "final.png"), cv2.cvtColor(environment.last_observation["agentview_image"][::-1].copy(), cv2.COLOR_RGB2BGR))
        print(json.dumps(result, indent=2), flush=True)
        print(f"Saved trial: {output}", flush=True)


if __name__ == "__main__":
    main()
