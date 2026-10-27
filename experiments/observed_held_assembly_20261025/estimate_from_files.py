"""One-shot sensor-file boundary for the local observation-only CAD estimator."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from . import pose_estimation
from experiments.nist12_qwen_clip_reranking_20261023.depth_verification_20261023 import geometry
from scripts import surface_verification


PRIOR_SOURCE = "observed_initial_pose_transported_by_measured_robot_FK"
PRIOR_SOURCES = {PRIOR_SOURCE,
    "previous_observed_pose_transported_by_measured_robot_FK",
    "previous_observed_world_pose_after_release"}
CALIBRATION_FIELDS = {"camera_name", "width", "height", "intrinsics", "world_T_camera",
    "camera_axes", "depth_units", "world_units", "table_z_world_m", "calibration_source"}


def checked_calibration(path):
    calibration = json.loads(path.read_text())
    if set(calibration)-CALIBRATION_FIELDS:
        raise ValueError("Calibration contains fields outside the ordinary camera allowlist")
    for key, shape in (("intrinsics", (3, 3)), ("world_T_camera", (4, 4))):
        value = np.asarray(calibration[key], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Camera matrix must be a finite numeric array with its declared shape")
    for key in CALIBRATION_FIELDS-{"intrinsics", "world_T_camera"}:
        if key in calibration and not isinstance(calibration[key], (str, int, float)):
            raise ValueError("Camera metadata must be ordinary scalar calibration information")
    return calibration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate_packet(packet_path):
    packet_path = Path(packet_path).resolve()
    packet = json.loads(packet_path.read_text())
    allowed = {"mode", "cad_path", "observations", "prior_world_T_cad", "prior_source"}
    if set(packet)-allowed:
        raise ValueError("Estimator packet contains fields outside the sensor/CAD/prior boundary")
    if packet["mode"] not in ("table", "held"):
        raise ValueError("Packet mode must be table or held")
    inputs = {str(packet_path): digest(packet_path)}
    cad_path = Path(packet["cad_path"]).resolve()
    if cad_path.suffix.lower() not in {".stl", ".obj", ".ply"}:
        raise ValueError("Only native CAD mesh files are accepted")
    inputs[str(cad_path)] = digest(cad_path)
    observations = []
    for record in packet["observations"]:
        keys = {"rgb_path", "depth_m_path", "robot_depth_m_path", "calibration_path"}
        if set(record) != keys:
            raise ValueError("Each observation must contain exactly RGB, depth, robot depth and calibration paths")
        paths = {key: Path(value).resolve() for key, value in record.items()}
        if paths["depth_m_path"].suffix != ".npy" or paths["robot_depth_m_path"].suffix != ".npy":
            raise ValueError("Depth files must be plain NumPy arrays")
        if paths["calibration_path"].suffix != ".json":
            raise ValueError("Camera calibration must be JSON")
        if paths["rgb_path"].suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("RGB must be an ordinary camera image")
        for path in paths.values():
            inputs[str(path)] = digest(path)
        calibration = checked_calibration(paths["calibration_path"])
        observations.append({"rgb": np.asarray(Image.open(paths["rgb_path"]).convert("RGB")),
            "depth_m": np.load(paths["depth_m_path"], allow_pickle=False),
            "robot_depth_m": np.load(paths["robot_depth_m_path"], allow_pickle=False),
            "calibration": calibration})
    if not observations:
        raise ValueError("At least one permitted camera observation is required")
    if packet["mode"] == "table":
        if "prior_world_T_cad" in packet or "prior_source" in packet:
            raise ValueError("Initial table detection cannot receive an object-pose prior")
        estimate = pose_estimation.estimate_table_pose(cad_path, observations[0])
    else:
        if packet.get("prior_source") not in PRIOR_SOURCES:
            raise ValueError("Free-pose prior must be an observed pose transported by measured robot FK, or the previous observed world pose after release")
        estimate = pose_estimation.estimate_held_pose(cad_path, observations, packet["prior_world_T_cad"])
    output = packet_path.parent/"estimate.json"
    if output.exists():
        raise FileExistsError("Preserve prior estimate evidence; create a new sensor packet directory")
    if "observed_mask" in estimate:
        mask_path = packet_path.parent/"observed_mask.npy"
        if mask_path.exists():
            raise FileExistsError(mask_path)
        np.save(mask_path, estimate.pop("observed_mask"))
        estimate["observed_mask_path"] = str(mask_path)
    sources = {str(Path(module.__file__).resolve()): digest(module.__file__) for module in
               (pose_estimation, geometry, surface_verification)}
    sources[str(Path(__file__).resolve())] = digest(__file__)
    estimate["provenance"] = {"mode": packet["mode"], "input_files_sha256": inputs,
        "source_files_sha256": sources, "prior_source": packet.get("prior_source"),
        "method_scope": "New observation-only experiment reusing CAD raycasting and table ICP helpers; no claim of the full interactive runtime",
        "simulator_scene_trial_trajectory_result_files_read": False,
        "model_training_or_weights_used": False}
    output.write_text(json.dumps(estimate, indent=2)+"\n")
    return output, estimate


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m experiments.observed_held_assembly_20261025.estimate_from_files packet.json")
    output, estimate = estimate_packet(sys.argv[1])
    print(json.dumps({"status": estimate["status"], "estimate_path": str(output)}), flush=True)


if __name__ == "__main__":
    main()
