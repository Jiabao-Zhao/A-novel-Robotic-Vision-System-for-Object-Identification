"""Trusted image/robot execution server and an observation-only file client.

Only this server opens the physical simulation. Target state and contact labels
are saved under plant_private for later offline audit, never returned online.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


class PlantClient:
    """The controller receives no simulator instance or full state arrays."""
    def __init__(self, run):
        self.run = Path(run)
        self.queue = self.run/"plant_queue"
        self.queue.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def call(self, action, **arguments):
        token = uuid.uuid4().hex
        request = self.queue/(token+".request.json")
        response = self.queue/(token+".response.json")
        item = {"action": action, **arguments}
        temporary = request.with_suffix(".tmp")
        write(temporary, item)
        temporary.rename(request)
        started = time.perf_counter()
        while not response.exists():
            if time.perf_counter()-started > 600:
                raise TimeoutError("Image/robot execution server did not respond")
            time.sleep(.05)
        value = json.loads(response.read_text())
        self.calls.append({"action": action, "request": str(request.relative_to(self.run)),
                           "response": str(response.relative_to(self.run)),
                           "wall_seconds": time.perf_counter()-started})
        write(self.run/"client_rpc_log.json", self.calls)
        if not value.get("ok"):
            raise RuntimeError(value.get("error", "Execution adapter failed"))
        return value


def export_robot_static_model(trial, destination):
    """Known robot + fixed calibrated board CAD; no loose-product state."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(trial.scene_path)
    root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name", "").startswith("nist_part_"):
                parent.remove(child)
    for keyframes in list(root.findall("keyframe")):
        root.remove(keyframes)
    referenced = {geom.get("mesh") for geom in root.findall(".//geom") if geom.get("mesh")}
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh" and child.get("name") not in referenced:
                asset.remove(child)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    if "nist_part_" in destination.read_text():
        raise RuntimeError("Planning model retained loose-product metadata")


def serve(run):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial
    from experiments.runtime_antipodal_assembly_20261023.clarification_20261023.run_8mm_runtime_followup import restore_other_part_visibility
    from .sensors import capture_observation, robot_feedback

    run = Path(run).resolve()
    queue = run/"plant_queue"
    queue.mkdir(parents=True, exist_ok=True)
    trial, part, video_counter = None, None, 0

    def record_tick(goal):
        nonlocal video_counter
        original_tick(goal)
        if len(trial.states) % 10 == 0:
            frame = run/"recorded_camera"/part/(f"{video_counter:05d}.png")
            frame.parent.mkdir(parents=True, exist_ok=True)
            trial.render(frame)  # Fixed agentview; never aimed at an object pose.
            video_counter += 1

    def finish():
        nonlocal trial
        if trial is not None:
            trial.save()
            trial.close()
            trial = None

    try:
        active = True
        while active:
            requests = sorted(queue.glob("*.request.json"), key=lambda path: path.stat().st_mtime_ns)
            for request in requests:
                response = request.with_name(request.name.replace(".request.json", ".response.json"))
                if response.exists():
                    continue
                spec = json.loads(request.read_text())
                action = spec["action"]
                try:
                    if action == "reset":
                        finish()
                        part = spec["part"]
                        if part not in ("Gear_Large", "Gear_Medium", "M16_Hex_Nut"):
                            raise ValueError("Only the three requested parts are enabled")
                        trial = RobotTrial(SOURCE, run/"plant_private"/part, part)
                        restore_other_part_visibility(trial)
                        export_robot_static_model(trial, run/"known_static_model"/"robot_board.xml")
                        original_tick = trial._tick
                        trial._tick = record_tick
                        video_counter = 0
                        value = {"robot_feedback": robot_feedback(trial.m, trial.d),
                                 "planning_model": str(run/"known_static_model"/"robot_board.xml")}
                    elif action == "observe":
                        relative = Path(spec["observation_folder"])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("Observation folder must remain inside the observation inbox")
                        folder = run/"observations"/relative
                        capture_observation(trial.m, trial.d, folder)
                        value = {"observation_folder": str(folder), "robot_feedback": robot_feedback(trial.m, trial.d)}
                    elif action == "move":
                        frames = int(spec["frames"])
                        if not 2 <= frames <= 200:
                            raise ValueError("Move frame budget outside controlled trial limits")
                        trial.move(np.asarray(spec["world_T_grip"]), frames, str(spec["phase"]))
                        value = {"robot_feedback": robot_feedback(trial.m, trial.d)}
                    elif action == "hold":
                        frames = int(spec["frames"])
                        if not 1 <= frames <= 100:
                            raise ValueError("Hold frame budget outside controlled trial limits")
                        trial.hold(frames, str(spec["phase"]))
                        value = {"robot_feedback": robot_feedback(trial.m, trial.d)}
                    elif action == "gripper":
                        trial.gripper(float(spec["opening_m"]))
                        value = {"robot_feedback": robot_feedback(trial.m, trial.d)}
                    elif action == "feedback":
                        value = {"robot_feedback": robot_feedback(trial.m, trial.d)}
                    elif action == "finish":
                        finish()
                        value = {}
                    elif action == "shutdown":
                        finish()
                        active = False
                        value = {}
                    else:
                        raise ValueError("Unsupported camera/robot command")
                    write(response, {"ok": True, **value})
                except Exception as error:
                    # Error messages come from robot motion/camera IO, not object audit.
                    write(response, {"ok": False, "error": str(error)})
                    write(run/"plant_private"/"server_exception.json", {"error": str(error), "traceback": traceback.format_exc()})
            time.sleep(.02)
    finally:
        finish()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--serve":
        raise SystemExit("Use --serve RUN_FOLDER")
    serve(sys.argv[2])
