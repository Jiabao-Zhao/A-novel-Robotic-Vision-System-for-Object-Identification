"""Raw/filtered image handoff for fresh, ranked gear orientation estimates."""
import json
from pathlib import Path
import subprocess
import sys
import time

from ...plant import write
from ...preflight import estimate as original_estimate, observation_files
from ..known_static_filter import filter_observations


GEAR_YAW_STEPS_DEG = {"Gear_Large_m": 6., "Gear_Medium_m": 9.}


def estimate(mode, cad_path, observation_folder, destination, prior=None, lineage=None,
             prior_source="observed_initial_pose_transported_by_measured_robot_FK",
             planning_template=None):
    destination = Path(destination)
    started = time.perf_counter()
    record = {"mode": mode, "raw_observation_folder": str(observation_folder),
              "pose_core_or_original_acceptance_gates_changed": False,
              "part_state_or_estimated_CAD_used_for_filter": False}
    if mode == "table":
        fitted = original_estimate(mode, cad_path, observation_folder, destination)
        record["preprocessing_seconds"] = 0.
    elif mode == "held":
        if planning_template is None:
            raise ValueError("Held fitting requires the public known robot/static template")
        filtered = filter_observations(planning_template, observation_folder,
                                       destination/"filtered_observations")
        record.update(filtered)
        record["preprocessing_seconds"] = time.perf_counter()-started
        step = GEAR_YAW_STEPS_DEG.get(Path(cad_path).stem)
        if step is None:
            fitted = original_estimate(mode, cad_path, filtered["observation_folder"],
                destination, prior=prior, lineage=lineage, prior_source=prior_source)
        else:
            search_mode = "full_circle" if destination.name == "after_pickup" else "local_phase"
            filtered_files = observation_files(filtered["observation_folder"])
            for item in filtered_files:
                name = Path(item["calibration_path"]).name.removesuffix("_calibration.json")
                item["known_static_depth_m_path"] = str(Path(filtered["observation_folder"])
                    /"filter_diagnostics"/name/"known_static_depth.npy")
            packet = {"mode": mode, "cad_path": str(cad_path),
                "observations": observation_files(observation_folder),
                "filtered_observations": filtered_files,
                "prior_world_T_cad": prior, "prior_source": prior_source,
                "yaw_search_mode": search_mode, "yaw_step_deg": step}
            write(destination/"prior_lineage.json", lineage)
            packet_path = destination/"packet.json"
            write(packet_path, packet)
            completed = subprocess.run([sys.executable, "-m",
                "experiments.observed_held_assembly_20261025.board_feedback_followup_20261025.yaw_feedback.estimate_from_files",
                str(packet_path)], check=False, capture_output=True, text=True)
            (destination/"worker_stdout.txt").write_text(completed.stdout)
            (destination/"worker_stderr.txt").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError("Ranked observation worker failed: "+completed.stderr[-1200:])
            fitted = json.loads((destination/"estimate.json").read_text())
            record.update(yaw_search_mode=search_mode, yaw_step_deg=step,
                          ranking_uses="Raw depth; fitting uses known-static-filtered depth",
                          raw_forward_pixel_eligibility="Known robot/table/board occlusion excluded; differs from the original robot-only forward check")
    else:
        raise ValueError("Mode must be table or held")
    record["total_observation_processing_seconds"] = time.perf_counter()-started
    write(destination/"preprocessing.json", record)
    return fitted
