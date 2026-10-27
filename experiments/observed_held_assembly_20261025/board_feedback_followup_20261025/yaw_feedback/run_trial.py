"""Fresh three-part trial with camera-ranked gear phases and completed responses."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .. import run_trial as controller
from ..transport import CompleteResponseClient
from .preprocessing import estimate, GEAR_YAW_STEPS_DEG
from .yaw_search import protocol as yaw_protocol


HERE = Path(__file__).resolve().parent
BASE = controller.BASE/"yaw_feedback"


def main():
    controller.torch.set_num_threads(4)
    controller.estimate = estimate
    output = BASE/"runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    print("OUTPUT="+str(output), flush=True)
    goals = json.loads(controller.GOALS.read_text())["nominal_simulation_goals"]
    model = controller.GrConvNetAdapter(device="cpu")
    original = HERE.parents[1]
    sources = [original/name for name in ("run_trial.py", "plant.py", "planning.py",
        "sensors.py", "pose_estimation.py", "estimate_from_files.py", "preflight.py")]
    sources += [HERE.parent/name for name in ("run_trial.py", "known_static_filter.py", "transport.py")]
    sources += list(HERE.glob("*.py"))
    manifest = {"parts": controller.PARTS, "created_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "experiments.observed_held_assembly_20261025.board_feedback_followup_20261025.yaw_feedback.run_trial",
        "source_files_sha256": {str(path.relative_to(controller.ROOT)): controller.digest(path)
                                 for path in sorted(set(sources))},
        "pose_protocol": controller.pose_protocol(), "grasp_protocol": controller.grasp_protocol(),
        "yaw_protocol": yaw_protocol(), "gear_yaw_bank_steps_deg": GEAR_YAW_STEPS_DEG,
        "goals_file_sha256": controller.digest(controller.GOALS),
        "nominal_goals": {part: goals[part] for part in controller.PARTS},
        "camera_constants": controller.CAMERAS, "GR_model_provenance": model.provenance,
        "GR_model_load_seconds_once": model.model_setup_seconds,
        "controller_protocol": {"lowering_heights_above_goal_m": controller.LOWERING_HEIGHTS_M,
            "minimum_observed_pickup_rise_m": controller.MIN_PICKUP_RISE_M,
            "observed_stall_margin_m": controller.OBSERVED_STALL_M,
            "maximum_incremental_observed_part_in_grip_change_m": controller.MAX_PART_IN_GRIP_CHANGE_M,
            "maximum_incremental_observed_part_in_grip_rotation_change_deg": controller.MAX_PART_IN_GRIP_ROTATION_CHANGE_DEG,
            "nominal_final_position_tolerance_m": controller.FINAL_POSITION_TOLERANCE_M,
            "nominal_final_signed_CAD_Z_axis_tolerance_deg": controller.FINAL_AXIS_TOLERANCE_DEG,
            "nominal_final_full_CAD_rotation_tolerance_deg": controller.FINAL_CAD_ROTATION_TOLERANCE_DEG,
            "staged_exact_goal_route": controller.route_protocol(),
            "yaw_tooth_thread_phase_verified": False},
        "boundary": {"online_inputs": "Raw RGB-D, camera calibration, robot encoders/FK, native CAD and known static board/goals",
            "simulator_object_state_used_for_estimation_or_control": False,
            "renderer_target_segmentation_or_truth_crops_used": False,
            "robot_static_planner_plus_estimated_CAD_only": True,
            "plant_private_truth_read_only_after_online_estimates_and_commands_frozen": True,
            "prior_diagnostic_selected_poses_or_offline_errors_used": False,
            "response_parse_retry_resends_physical_command": False},
        "scope": "New observations on every reset; stepwise rendered RGB-D feedback; no hardware, gear engagement or nut threading verification"}
    controller.write(output/"manifest.json", manifest)
    server_log = (output/"plant_server.log").open("w")
    process = subprocess.Popen([sys.executable, "-m", "experiments.observed_held_assembly_20261025.plant",
                                "--serve", str(output)], stdout=server_log, stderr=subprocess.STDOUT)
    plant, results = CompleteResponseClient(output), []
    try:
        for part in controller.PARTS:
            print("START "+part, flush=True)
            result = controller.run_part(part, output, plant, model, goals[part])
            results.append(result)
            controller.write(output/"results.json", results)
            print(json.dumps({key: result.get(key) for key in ("part", "status", "physical_pickup_attempts",
                "observed_pickup_rise_m", "board_lowering_moves", "release_commanded", "wall_seconds", "error")}), flush=True)
    finally:
        try:
            plant.call("shutdown")
        finally:
            process.wait(timeout=30)
            server_log.close()
    print("COMPLETE="+str(output), flush=True)


if __name__ == "__main__":
    main()
