"""Independent vertex-vs-native-MuJoCo table-distance audit, no physics steps."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.runtime_antipodal_assembly_20261023.grasp_planning import (
    gripper_surface_local, transform_points,
)
from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/runtime_antipodal_assembly_20261023/20261023T213101Z"


def geom_vertices_world(model, data, geom):
    """Read compiled vertices/corners without any planner helper."""
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[first:first + count]
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                             for z in (-1, 1)]) * model.geom_size[geom]
    else:
        raise ValueError("Unexpected gripper geom type")
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs/runtime_antipodal_assembly_20261023/8mm_geometry_audit" / stamp
    out.mkdir(parents=True, exist_ok=False)
    trial = RobotTrial(RUN / "RGOCG8-50_8mm/scene.xml", out, "RGOCG8-50_8mm")
    source = json.loads((RUN / "RGOCG8-50_8mm/result.json").read_text())
    initial = np.asarray(source["world_T_cad_input"])
    center = initial[:3, 3]
    pin_axis = initial[:3, 2]
    pin_axis /= np.linalg.norm(pin_axis)
    x = np.cross(pin_axis, [0., 0., 1.])
    x /= np.linalg.norm(x)
    z = np.array([0., 0., -1.])
    y = np.cross(z, x)
    base = np.column_stack((x, y, z))
    table = trial.m.geom("workbench_top").id
    native_table_top = float(trial.d.geom_xpos[table, 2] + trial.m.geom_size[table, 2])
    gripper_ids = [geom for geom in range(trial.m.ngeom)
                   if (trial.m.geom(geom).name or "").startswith("gripper0_")
                   and (trial.m.geom_contype[geom] or trial.m.geom_conaffinity[geom])]
    pairs = np.array([center - .004*x, center + .004*x])
    rows = []
    for roll in (0., 20., -20., 45., -45., 90., -90.):
        rotation = Rotation.from_rotvec(x * np.deg2rad(roll)).as_matrix() @ base
        for depth in (.004, 0., -.004, -.008, .006, .008, .012, .016):
            command = np.eye(4)
            command[:3, :3] = rotation
            command[:3, 3] = center - rotation @ [0., 0., depth]
            item = {"roll_deg": roll, "pad_depth_m": depth,
                    "grip_world_z_m": float(command[2, 3]),
                    "pad_depth_in_frozen_range": bool(-.0116 <= depth <= .0044)}
            try:
                q = trial.solve_ik(command)
            except RuntimeError as error:
                item.update(status="ik_unavailable", error=str(error))
                rows.append(item)
                continue
            data = mujoco.MjData(trial.m)
            data.qpos[:] = trial.d.qpos
            data.qpos[trial.qa] = q
            data.qpos[trial.fqa] = [.01, -.01]
            mujoco.mj_forward(trial.m, data)
            actual = np.eye(4)
            actual[:3, :3] = data.site_xmat[trial.site].reshape(3, 3)
            actual[:3, 3] = data.site_xpos[trial.site]
            surface = gripper_surface_local(trial.m, trial.d, .02)
            planner_actual_min = float(transform_points(actual, surface)[:, 2].min() - native_table_top)
            geoms = []
            for geom in gripper_ids:
                vertex_min = float(geom_vertices_world(trial.m, data, geom)[:, 2].min() - native_table_top)
                segment = np.zeros(6)
                native_distance = float(mujoco.mj_geomDistance(
                    trial.m, data, geom, table, 1., segment))
                geoms.append({"geom": trial.m.geom(geom).name,
                              "vertex_minimum_over_table_m": vertex_min,
                              "native_geom_distance_to_table_m": native_distance,
                              "difference_m": vertex_min - native_distance})
            vertex_min = min(row["vertex_minimum_over_table_m"] for row in geoms)
            native_min = min(row["native_geom_distance_to_table_m"] for row in geoms)
            contacts = []
            for contact in data.contact:
                ids = [int(value) for value in contact.geom]
                if table in ids and any(value in gripper_ids for value in ids):
                    contacts.append({"geoms": [trial.m.geom(value).name for value in ids],
                                     "distance_m": float(contact.dist)})
            item.update(status="compared", world_T_grip=command.tolist(),
                        realized_site_error_m=float(np.linalg.norm(actual[:3, 3] - command[:3, 3])),
                        planner_minimum_over_table_m=planner_actual_min,
                        direct_vertex_minimum_over_table_m=vertex_min,
                        native_minimum_geom_distance_to_table_m=native_min,
                        planner_vs_direct_vertex_difference_m=planner_actual_min - vertex_min,
                        maximum_vertex_vs_native_distance_difference_m=max(abs(g["difference_m"]) for g in geoms),
                        table_contacts=contacts, geoms=geoms)
            rows.append(item)
    compared = [row for row in rows if row["status"] == "compared"]
    penetrating = [row for row in compared if row["planner_minimum_over_table_m"] < -.00005]
    native_zero_anomalies = [
        {"roll_deg": row["roll_deg"], "pad_depth_m": row["pad_depth_m"], **geom}
        for row in compared for geom in row["geoms"]
        if geom["native_geom_distance_to_table_m"] == 0.
        and abs(geom["vertex_minimum_over_table_m"]) > .0001]
    report = {"source_run": str(RUN), "source_result_sha256": hashlib.sha256(
                (RUN / "RGOCG8-50_8mm/result.json").read_bytes()).hexdigest(),
              "source_grasp_planning_sha256": hashlib.sha256(
                  ROOT.joinpath("experiments/runtime_antipodal_assembly_20261023/grasp_planning.py").read_bytes()).hexdigest(),
              "scope": "Static actual Panda IK poses; no dynamics, no model changes, other loose parts ignored.",
              "pin_world_T_cad": initial.tolist(), "center_cross_section_contacts_world_m": pairs.tolist(),
              "table_top_world_z_m": native_table_top, "opening_m": .02,
              "maximum_planner_vs_direct_vertex_error_m": max(
                  abs(row["planner_vs_direct_vertex_difference_m"]) for row in compared),
              "maximum_vertex_vs_native_geom_distance_error_m": max(
                  row["maximum_vertex_vs_native_distance_difference_m"] for row in compared),
              "maximum_penetrating_pose_minimum_distance_error_m": max(
                  abs(row["planner_minimum_over_table_m"]
                      - row["native_minimum_geom_distance_to_table_m"]) for row in penetrating),
              "table_penetration_agrees_with_actual_contact_detection": all(
                  (row["planner_minimum_over_table_m"] < -.00005)
                  == any(contact["distance_m"] < -.00005 for contact in row["table_contacts"])
                  for row in compared),
              "native_zero_distance_anomalies": native_zero_anomalies,
              "native_distance_caveat": "Some individual separated mesh/table distance queries return exactly zero despite positive vertex separation and no contact. Raw values are retained. Collision conclusions use negative minimum distances together with actual mj_forward contact rows.",
              "rows": rows}
    (out / "geometry_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "maximum_planner_vs_direct_vertex_error_m", "maximum_vertex_vs_native_geom_distance_error_m")}), flush=True)
    for row in compared:
        if row["roll_deg"] == 0.:
            print(json.dumps({key: row[key] for key in (
                "roll_deg", "pad_depth_m", "pad_depth_in_frozen_range",
                "planner_minimum_over_table_m", "native_minimum_geom_distance_to_table_m")}), flush=True)
    print(f"AUDIT={out}", flush=True)
    trial.close()


if __name__ == "__main__":
    main()
