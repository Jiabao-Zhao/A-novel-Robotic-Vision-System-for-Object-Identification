"""Static replay of the saved M16 proposal's six table checks.

No inference, VLM request, physics step, robot movement, new grasp or selection.
Only new diagnostic JSON is written; frozen trial records remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .capture import rectangle_geometry, unproject_pixels
from experiments.runtime_antipodal_assembly_20261023.grasp_planning import (
    gripper_surface_local, transform_points,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = ROOT / "outputs/grconvnet_vlm_pickup_20261024/runs/20261024T135155Z"
DEPTHS_M = (.004, 0., -.004, -.008, .006, .008)
TABLE_MARGIN_M = .0001
PART = "M16_Hex_Nut"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_geom_names(model):
    """Match gripper_surface_local's exact point concatenation order."""
    names = []
    for index in range(model.ngeom):
        name = model.geom(index).name or ""
        if not name.startswith("gripper0_") or not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
            count = int(model.mesh_vertnum[model.geom_dataid[index]])
        elif model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX:
            count = 8
        else:
            raise ValueError("Unsupported collision geometry")
        names.extend([name]*count)
    return names


def diagnose_arm(run, arm):
    folder = run / arm / PART
    result = read(folder / "result.json")
    states = read(folder / "trajectory.json")
    assert result["physical_attempts"] == 0 and len(states) == 1
    depth = np.load(folder / "capture/depth_m.npy")
    mask = np.load(folder / "capture/target_mask.npy")
    calibration = read(folder / "capture/calibration.json")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = states[0]["qpos"]
    mujoco.mj_forward(model, data)
    original_qpos = data.qpos.copy()
    original_time = float(data.time)
    reference = np.asarray(states[0]["grip_site_pose"])[:3, :3]
    table_z = float(calibration["table_z_world_m"])
    pixels = np.argwhere(mask)
    target_points = unproject_pixels(pixels, depth[mask], calibration)
    names = point_geom_names(model)
    rows = []
    for proposal in result["raw_network_proposals"]:
        geometry = rectangle_geometry(proposal, depth, mask, calibration)
        axis = np.asarray(geometry["jaw_axis_world_unit"])
        axis[2] = 0.
        axis /= np.linalg.norm(axis)
        down = np.array([0., 0., -1.])
        options = [np.column_stack((axis*sign, np.cross(down, axis*sign), down)) for sign in (1., -1.)]
        rotation = min(options, key=lambda value: np.linalg.norm(Rotation.from_matrix(value @ reference.T).as_rotvec()))
        center = np.asarray(geometry["center_world_m"])
        opening = min(.08, float(geometry["predicted_width_m"])+.012)
        surfaces = gripper_surface_local(model, data, opening)
        assert len(names) == len(surfaces)
        variants = []
        for finger_depth in DEPTHS_M:
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = center-rotation[:, 2]*finger_depth
            points = transform_points(pose, surfaces)
            lowest = int(np.argmin(points[:, 2]))
            endpoint_gap = float(points[lowest, 2]-table_z)
            approach_adjustment = min(0., float(-pose[2, 2]*.10))
            gap = endpoint_gap+approach_adjustment
            variants.append({"finger_depth_m": finger_depth,
                "tool_reference_height_world_m": float(pose[2, 3]),
                "lowest_gripper_geom": names[lowest],
                "lowest_gripper_world_point_m": points[lowest].tolist(),
                "minimum_gripper_height_above_table_m": endpoint_gap,
                "approach_path_adjustment_m": approach_adjustment,
                "minimum_table_gap_m": gap,
                "required_margin_m": TABLE_MARGIN_M,
                "table_overlap_m": max(0., -gap),
                "additional_height_for_margin_m": max(0., TABLE_MARGIN_M-gap),
                "passed_same_table_check": gap >= TABLE_MARGIN_M,
                "world_T_grip": pose.tolist()})
        rows.append({"network_candidate_id": proposal["network_candidate_id"],
                     "raw_proposal": proposal, "saved_depth_geometry_recomputed": geometry,
                     "fixed_jaw_opening_m": opening,
                     "six_original_depth_checks": variants,
                     "all_six_rejected_reproduced": not any(row["passed_same_table_check"] for row in variants)})
    np.testing.assert_array_equal(data.qpos, original_qpos)
    assert float(data.time) == original_time
    return {"part": PART, "arm": arm, "saved_status": result["status"],
            "physical_attempts": result["physical_attempts"],
            "visible_target_surface_height_range_above_table_m": [float(target_points[:, 2].min()-table_z),
                                                                     float(target_points[:, 2].max()-table_z)],
            "proposal_count": len(rows), "proposals": rows,
            "physics_steps": 0, "model_calls": 0, "VLM_calls": 0,
            "qpos_unchanged": True, "scope": "Replay of this saved proposal and original height/depth mapping only; not a proof that the nut has no usable grasp."}


def main():
    run = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else DEFAULT_RUN
    output = run / "static_failure_diagnostics.json"
    if output.exists():
        raise FileExistsError(output)
    protected = [run / "manifest.json", run / "results.json",
                 HERE / "grconvnet_adapter.py", HERE / "run_trial.py", HERE / "capture.py",
                 ROOT / "experiments/runtime_antipodal_assembly_20261023/grasp_planning.py"]
    for arm in ("grconvnet_score", "grconvnet_vlm"):
        protected.extend(path for path in (run / arm / PART).rglob("*") if path.is_file())
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    rows = [diagnose_arm(run, arm) for arm in ("grconvnet_score", "grconvnet_vlm")]
    after = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    assert before == after
    diagnostic = {"scope": "Post-run static M16 rejection explanation; frozen results unmodified.",
                  "parts": rows, "all_original_rejections_reproduced": all(proposal["all_six_rejected_reproduced"]
                     for row in rows for proposal in row["proposals"]),
                  "source_sha256": before, "source_files_unchanged": True,
                  "diagnostic_source_sha256": digest(Path(__file__)),
                  "new_grasps_or_reselection": False, "model_calls": 0, "VLM_calls": 0, "physics_steps": 0}
    output.write_text(json.dumps(diagnostic, indent=2)+"\n")
    for row in rows:
        geometry = row["proposals"][0]["saved_depth_geometry_recomputed"]
        print(row["arm"], "surface_mm", geometry["estimated_height_m"]*1000,
              "middle_mm", geometry["center_world_m"][2]*1000,
              "gaps_mm", [round(value["minimum_table_gap_m"]*1000, 6)
                          for value in row["proposals"][0]["six_original_depth_checks"]])
    print(f"COMPLETE={output}")


if __name__ == "__main__":
    main()
