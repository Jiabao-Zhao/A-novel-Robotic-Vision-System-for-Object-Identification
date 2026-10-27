"""Validate the saved 14-part grasp poses used by the static visualization.

This reads original saved selections (13) plus the separate automatic 8 mm
follow-up (1). It runs bounded IK and forward kinematics, without grasp search,
physics steps, target motion, rendering, or physical robot interfaces.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "outputs/runtime_antipodal_assembly_20261023"
ORIGINAL = BASE / "20261023T213101Z"
EIGHT_MM = BASE / "8mm_runtime_followup/20261024T012239Z"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_pose(data, body):
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    pose[:3, 3] = data.xpos[body]
    return pose


def main():
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    parts = manifest["parts"]
    source = Path(manifest["source_scene"])
    model = mujoco.MjModel.from_xml_path(str(source))
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "objects_centered")
    site = model.site("gripper0_grip_site").id
    joints = np.array([model.joint(f"robot0_joint{i}").id for i in range(1, 8)])
    qa, da = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    finger_joints = [model.joint(f"gripper0_finger_joint{i}").id for i in (1, 2)]
    fqa = model.jnt_qposadr[finger_joints]
    lower, upper = model.jnt_range[joints].T
    rows = []
    for part in parts:
        result_path = (EIGHT_MM / "result.json" if part == "RGOCG8-50_8mm"
                       else ORIGINAL / part / "result.json")
        result = json.loads(result_path.read_text())
        selected = result["selected_grasp"]
        desired = np.asarray(selected["world_T_grip"])
        contacts = np.asarray(selected["contacts_world_m"])
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        target = model.body("nist_part_" + part).id
        initial_pose = body_pose(data, target)
        target_joint = int(model.body_jntadr[target])
        target_qadr = int(model.jnt_qposadr[target_joint])
        target_q_before = data.qpos[target_qadr:target_qadr + 7].copy()
        proxy = SimpleNamespace(m=model, d=data, qa=qa, da=da, lower=lower, upper=upper,
                                site=site, site_body=int(model.site_bodyid[site]))
        row = {"part": part, "source_result": str(result_path),
               "source_result_sha256": digest(result_path), "source_status": result["status"],
               "source_condition": "automatic 8 mm generic-depth follow-up" if part == "RGOCG8-50_8mm"
                                   else "original fourteen-part run",
               "shown_gripper_state": "saved candidate opening before finger closure",
               "opening_m": selected["opening_m"], "pad_depth_m": selected["pad_depth_m"]}
        try:
            q = RobotTrial.solve_ik(proxy, desired)
            data.qpos[qa] = q
            data.qpos[fqa] = [selected["opening_m"] / 2, -selected["opening_m"] / 2]
            mujoco.mj_forward(model, data)
            realized = np.eye(4)
            realized[:3, :3] = data.site_xmat[site].reshape(3, 3)
            realized[:3, 3] = data.site_xpos[site]
            position_error = float(np.linalg.norm(realized[:3, 3] - desired[:3, 3]))
            rotation_error = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(
                desired[:3, :3] @ realized[:3, :3].T).as_rotvec())))
            contacts_local = contacts @ desired[:3, :3] - desired[:3, 3] @ desired[:3, :3]
            # Each candidate deliberately offsets the grip site from the contact
            # midpoint by its generic finger-depth sample.
            midpoint_error = float(np.linalg.norm(contacts_local.mean(axis=0)
                                                   - [0., 0., selected["pad_depth_m"]]))
            axis = contacts[1] - contacts[0]
            axis /= np.linalg.norm(axis)
            jaw_alignment = abs(float(axis @ desired[:3, 0]))
            mesh_path = ROOT / f"outputs/simulation/nist_task_board_1/assets/meshes/{part}_m.stl"
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            vertices = (np.asarray(mesh.vertices) @ initial_pose[:3, :3].T
                        + initial_pose[:3, 3]).astype(np.float32)
            tensor = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices),
                o3d.core.Tensor(np.asarray(mesh.triangles).astype(np.int32)))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tensor)
            distances = scene.compute_distance(o3d.core.Tensor(contacts.astype(np.float32))).numpy()
            unchanged = (np.array_equal(target_q_before, data.qpos[target_qadr:target_qadr + 7])
                         and np.allclose(initial_pose, body_pose(data, target), atol=1e-12, rtol=0.))
            input_pose_match = np.allclose(initial_pose, result["world_T_cad_input"], atol=1e-12, rtol=0.)
            passed = bool(position_error <= .0001 and rotation_error <= .1
                          and unchanged and input_pose_match and midpoint_error <= 1e-6
                          and abs(jaw_alignment - 1.) <= 1e-6
                          and np.max(distances) <= .00001)
            row.update(passed=passed, realized_position_error_m=position_error,
                       realized_rotation_error_deg=rotation_error,
                       target_state_unchanged=unchanged, initial_target_pose_match=input_pose_match,
                       contact_midpoint_tool_depth_error_m=midpoint_error,
                       absolute_contact_axis_tool_x_alignment=jaw_alignment,
                       maximum_contact_distance_from_native_cad_m=float(np.max(distances)),
                       world_T_grip_saved=desired.tolist(), world_T_grip_realized=realized.tolist(),
                       robot_qpos=q.tolist(), contacts_world_m=contacts.tolist())
        except Exception as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
        rows.append(row)
        print(f"POSE {part}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = BASE / "grasp_pose_visualization_validation" / stamp
    out.mkdir(parents=True, exist_ok=False)
    report = {"passed": all(row["passed"] for row in rows), "parts": rows,
              "source_scene_sha256": digest(source), "source_script_sha256": digest(__file__),
              "scope": "Static saved-pose/initial-CAD checks; no new grasp search or dynamics.",
              "tolerances": {"site_position_m": .0001, "site_rotation_deg": .1,
                             "contact_surface_distance_m": .00001},
              "mixed_source_condition": "13 original selected poses + one automatic 8 mm follow-up pose. This is not a rerun of the 14-part experiment."}
    (out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"VALIDATION={out} PASS={report['passed']}", flush=True)


if __name__ == "__main__":
    main()
