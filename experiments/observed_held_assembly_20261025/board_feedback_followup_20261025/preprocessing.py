"""Keep raw observations, remove explained static depth, run the frozen fitter."""
from pathlib import Path
import time

from ..plant import write
from ..preflight import estimate as original_estimate
from .known_static_filter import filter_observations


def estimate(mode, cad_path, observation_folder, destination, prior=None, lineage=None,
             prior_source="observed_initial_pose_transported_by_measured_robot_FK",
             planning_template=None):
    destination = Path(destination)
    started = time.perf_counter()
    record = {"mode": mode, "raw_observation_folder": str(observation_folder),
              "pose_core_or_acceptance_gates_changed": False,
              "part_state_or_estimated_CAD_used_for_filter": False}
    worker_folder = observation_folder
    if mode == "held":
        if planning_template is None:
            raise ValueError("Held fitting requires the public known robot/static template")
        filtered = filter_observations(planning_template, observation_folder,
                                       destination/"filtered_observations")
        worker_folder = filtered["observation_folder"]
        record.update(filtered)
    elif mode != "table":
        raise ValueError("Mode must be table or held")
    record["preprocessing_seconds"] = time.perf_counter()-started
    fitted = original_estimate(mode, cad_path, worker_folder, destination,
        prior=prior, lineage=lineage, prior_source=prior_source)
    record["total_observation_processing_seconds"] = time.perf_counter()-started
    write(destination/"preprocessing.json", record)
    return fitted
