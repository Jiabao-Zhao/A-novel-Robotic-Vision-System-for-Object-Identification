"""Render NIST Task Board #1 and save calibrated RGB-D from both robot cameras."""

import json

import cv2
import numpy as np

from simulation.libero_io import save_libero_observation
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.nist_task_board_1 import NistTaskBoardEnvironment, OUTPUT_DIR, PARTS


def main():
    with NistTaskBoardEnvironment() as environment:
        raw = environment.reset(seed=1000)
        environment.set_control_mode("relative")
        for _ in range(40):
            raw, _, _, _ = environment.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
        if not np.all(np.isfinite(environment.sim.data.qpos)):
            raise RuntimeError("Non-finite simulator state after settling.")
        for camera in environment.camera_names:
            observation = LiberoRGBDSensor(environment, camera).capture(raw)
            save_libero_observation(observation, environment, OUTPUT_DIR / camera)
        rgb = environment.sim.render(width=3840, height=3840, camera_name="agentview")[::-1].copy()
        path = OUTPUT_DIR / "nist_task_board_1_3840.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not save {path}")
        cv2.imwrite(str(OUTPUT_DIR / "preview.jpg"),
                    cv2.cvtColor(cv2.resize(rgb, (1000, 1000)), cv2.COLOR_RGB2BGR))
        (OUTPUT_DIR / "scene_summary.json").write_text(json.dumps({
            "scene": environment.task_name, "loose_parts": list(PARTS),
            "loose_part_count": len(PARTS), "seed": 1000, "settling_steps": 40,
            "rgbd_resolution": [768, 768], "figure_resolution": [3840, 3840],
            "assembly_execution_validated": False,
        }, indent=2))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
