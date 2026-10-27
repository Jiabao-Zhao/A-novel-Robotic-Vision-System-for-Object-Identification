"""Camera-only initial localization preflight; no robot movement or truth input."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .plant import PlantClient, ROOT, write
from .sensors import CAMERAS


BASE = ROOT/"outputs/observed_held_assembly_20261025"
PARTS = ("Gear_Large", "Gear_Medium", "M16_Hex_Nut")


def observation_files(folder):
    folder = Path(folder)
    return [{"rgb_path": str(folder/(name+"_rgb.png")),
             "depth_m_path": str(folder/(name+"_depth_m.npy")),
             "robot_depth_m_path": str(folder/(name+"_robot_depth_m.npy")),
             "calibration_path": str(folder/(name+"_calibration.json"))}
            for name, *_ in CAMERAS]


def estimate(mode, cad_path, observation_folder, destination, prior=None, lineage=None,
             prior_source="observed_initial_pose_transported_by_measured_robot_FK"):
    destination = Path(destination)
    packet = {"mode": mode, "cad_path": str(cad_path),
              "observations": observation_files(observation_folder)}
    if mode == "held":
        packet.update(prior_world_T_cad=prior, prior_source=prior_source)
        write(destination/"prior_lineage.json", lineage)
    path = destination/"packet.json"
    write(path, packet)
    completed = subprocess.run([sys.executable, "-m",
        "experiments.observed_held_assembly_20261025.estimate_from_files", str(path)],
        check=False, capture_output=True, text=True)
    (destination/"worker_stdout.txt").write_text(completed.stdout)
    (destination/"worker_stderr.txt").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("Observation-only estimation worker failed: "+completed.stderr[-1200:])
    return json.loads((destination/"estimate.json").read_text())


def main():
    output = BASE/"initial_preflight"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    print("OUTPUT="+str(output), flush=True)
    server_log = (output/"plant_server.log").open("w")
    process = subprocess.Popen([sys.executable, "-m", "experiments.observed_held_assembly_20261025.plant",
                                "--serve", str(output)], stdout=server_log, stderr=subprocess.STDOUT)
    plant = PlantClient(output)
    rows = []
    try:
        for part in PARTS:
            print("CAPTURE "+part, flush=True)
            plant.call("reset", part=part)
            packet = plant.call("observe", observation_folder=part+"/initial")
            result = estimate("table", ROOT/"outputs/simulation/nist_task_board_1/assets/meshes"/(part+"_m.stl"),
                              packet["observation_folder"], output/"estimates"/part)
            row = {"part": part, "estimate": result}
            rows.append(row)
            write(output/"results.json", rows)
            print(json.dumps({"part":part, "status":result["status"],
                              "pose":result.get("world_T_cad"),
                              "seconds":result.get("diagnostics",{}).get("seconds")}), flush=True)
            plant.call("finish")
    finally:
        try:
            plant.call("shutdown")
        finally:
            process.wait(timeout=30)
            server_log.close()
    print("COMPLETE="+str(output), flush=True)


if __name__ == "__main__":
    main()
