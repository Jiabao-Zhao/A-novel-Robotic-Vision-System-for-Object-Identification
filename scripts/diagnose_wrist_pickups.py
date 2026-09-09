"""Saved-state replay and robot measurements for causal diagnosis.

No model calls or automatic experiments. Simulator identities, poses, contacts,
and joint limits are evaluation evidence, not framework perception inputs.
"""

import hashlib

import numpy as np


def restore(environment, manifest, frozen):
    environment.reset(seed=manifest["seed"])
    raw = environment.env.set_init_state(frozen)
    environment.last_observation = raw
    environment.set_control_mode("relative")
    if "initial_gripper_action" in manifest:
        environment.robots[0].gripper.current_action = np.array(manifest["initial_gripper_action"])
    actual = np.ascontiguousarray(environment.sim.get_state().flatten())
    assert hashlib.sha256(actual.tobytes()).hexdigest() == manifest["initial_state_sha256"]
    return raw


def measure(environment, raw, evaluator):
    import mujoco
    from robosuite.utils.transform_utils import quat2mat

    sim = environment.sim
    robot = environment.robots[0]
    eef = np.asarray(raw["robot0_eef_pos"])
    rotation = quat2mat(raw["robot0_eef_quat"])
    target = sim.data.body_xpos[evaluator.ids[evaluator.target]].copy()
    pad_ids = sorted(set.union(*evaluator.fingers))
    pad_midpoint = np.mean(sim.data.geom_xpos[pad_ids], axis=0)
    contacts = []
    for index, contact in enumerate(sim.data.contact[:sim.data.ncon]):
        pair = (int(contact.geom1), int(contact.geom2))
        names = [sim.model.geom_id2name(g) or f"geom_{g}" for g in pair]
        if not (set(pair) & evaluator.part_geoms[evaluator.target] or
                any(n.startswith(("robot0", "gripper0")) for n in names)):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
        contacts.append({"geoms": names, "force_contact_frame_N": force[:3].tolist(),
            "geom_bodies": [sim.model.body_id2name(int(sim.model.geom_bodyid[g])) for g in pair],
            "contact_frame_world": contact.frame.reshape(3, 3).tolist(),
            "friction": contact.friction.tolist(), "penetration_mm": float(-contact.dist*1000),
            "contact_in_eef_mm": (rotation.T @ (contact.pos-eef)*1000).tolist()})
    joints = []
    for name in robot.robot_model.joints:
        index = sim.model.joint_name2id(name)
        q = float(sim.data.qpos[sim.model.jnt_qposadr[index]])
        limits = sim.model.jnt_range[index]
        joints.append({"name": name, "q_rad": q, "limits_rad": limits.tolist(),
                       "limit_margin_deg": float(np.rad2deg(min(q-limits[0], limits[1]-q)))})
    limits = [{"joint_id": int(sim.data.efc_id[i]), "force": float(sim.data.efc_force[i])}
              for i in range(sim.data.nefc)
              if sim.data.efc_type[i] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)]
    return {"eef_world_m": eef.tolist(), "eef_rotation_world": rotation.tolist(),
        "target_world_m": target.tolist(),
        "target_rotation_world": sim.data.body_xmat[evaluator.ids[evaluator.target]].reshape(3, 3).tolist(),
        "target_in_eef_mm": (rotation.T @ (target-eef)*1000).tolist(),
        "pad_midpoint_in_eef_mm": (rotation.T @ (pad_midpoint-eef)*1000).tolist(),
        "target_minus_pad_midpoint_in_eef_mm": (rotation.T @ (target-pad_midpoint)*1000).tolist(),
        "gripper_qpos_m": np.asarray(raw["robot0_gripper_qpos"]).tolist(),
        "controller_goal_world_m": np.asarray(robot.controller.goal_pos).tolist(),
        "controller_torques_Nm": np.asarray(robot.controller.torques).tolist(),
        "torque_limits_Nm": [np.asarray(v).tolist() for v in robot.torque_limits],
        "joints": joints, "active_joint_limits": limits, "contacts": contacts}
