"""Strict raw/filtered sensor-file boundary for the fresh CAD yaw worker."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from . import yaw_search
from ... import estimate_from_files as original_worker
from ... import pose_estimation
from experiments.nist12_qwen_clip_reranking_20261023.depth_verification_20261023 import geometry
from scripts import surface_verification


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_observations(records, filtered, hashes):
    result = []
    keys = {"rgb_path", "depth_m_path", "robot_depth_m_path", "calibration_path"}
    if filtered:
        keys.add("known_static_depth_m_path")
    for record in records:
        if set(record) != keys:
            raise ValueError("Camera path record contains fields outside the raw/filtered sensor allowlist")
        paths = {key: Path(value).resolve() for key, value in record.items()}
        for key, path in paths.items():
            expected = {".json"} if key == "calibration_path" else {".png", ".jpg", ".jpeg"} if key == "rgb_path" else {".npy"}
            if path.suffix.lower() not in expected:
                raise ValueError("Camera input file type disagrees with its declared field")
            hashes[str(path)] = digest(path)
        observation = {"rgb": np.asarray(Image.open(paths["rgb_path"]).convert("RGB")),
            "depth_m": np.load(paths["depth_m_path"], allow_pickle=False),
            "robot_depth_m": np.load(paths["robot_depth_m_path"], allow_pickle=False),
            "calibration": original_worker.checked_calibration(paths["calibration_path"])}
        if filtered:
            observation["known_static_depth_m"] = np.load(paths["known_static_depth_m_path"], allow_pickle=False)
        result.append(observation)
    if not result:
        raise ValueError("At least one raw and filtered camera observation is required")
    return result


def estimate_packet(packet_path):
    packet_path = Path(packet_path).resolve()
    packet = json.loads(packet_path.read_text())
    allowed = {"mode", "cad_path", "observations", "filtered_observations", "prior_world_T_cad",
               "prior_source", "yaw_search_mode", "yaw_step_deg"}
    if set(packet) != allowed:
        raise ValueError("Yaw packet must contain exactly the permitted CAD/raw/filtered/prior/search fields")
    if packet["mode"] != "held" or packet["prior_source"] not in original_worker.PRIOR_SOURCES:
        raise ValueError("Yaw search requires a held/free observation-derived pose prior")
    yaw_search.yaw_offsets(packet["yaw_search_mode"], packet["yaw_step_deg"])
    output = packet_path.parent/"estimate.json"
    bank_path = packet_path.parent/"ranked_candidates.json"
    if output.exists() or bank_path.exists():
        raise FileExistsError("Preserve existing pose/ranking evidence; use a fresh sensor packet directory")
    cad_path = Path(packet["cad_path"]).resolve()
    if cad_path.suffix.lower() not in {".stl", ".obj", ".ply"}:
        raise ValueError("Only native CAD meshes are accepted")
    hashes = {str(packet_path): digest(packet_path), str(cad_path): digest(cad_path)}
    raw = _load_observations(packet["observations"], False, hashes)
    filtered = _load_observations(packet["filtered_observations"], True, hashes)
    estimate = yaw_search.estimate_yaw_pose(cad_path, filtered, raw, packet["prior_world_T_cad"],
                                           packet["yaw_search_mode"], packet["yaw_step_deg"])
    bank = estimate.pop("ranked_candidates")
    bank_path.write_text(json.dumps(bank, indent=2)+"\n")
    sources = {str(Path(module.__file__).resolve()): digest(module.__file__) for module in
               (yaw_search, original_worker, pose_estimation, geometry, surface_verification)}
    sources[str(Path(__file__).resolve())] = digest(__file__)
    estimate["ranked_candidates_path"] = str(bank_path)
    estimate["provenance"] = {"mode": packet["mode"], "input_files_sha256": hashes,
        "source_files_sha256": sources, "prior_source": packet["prior_source"],
        "ranked_candidates_sha256": digest(bank_path),
        "ranking_uses_raw_depth": True, "local_fitting_uses_filtered_depth": True,
        "diagnostic_chosen_pose_or_actual_orientation_used": False,
        "simulator_scene_trial_trajectory_result_files_read": False,
        "training_or_model_weights_used": False}
    output.write_text(json.dumps(estimate, indent=2)+"\n")
    return output, estimate


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Supply one fresh yaw packet.json path")
    output, estimate = estimate_packet(sys.argv[1])
    print(json.dumps({"status": estimate["status"], "estimate_path": str(output),
                      "bank_seconds": estimate["diagnostics"]["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
