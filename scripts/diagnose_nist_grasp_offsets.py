"""Evaluation-only replay and bounded interventions for the saved peg pilot.

Ground-truth poses/contact forces here are diagnostic measurements, never inputs
to the perception pipeline. Original actions and results are not overwritten.
"""

import hashlib
import json

import numpy as np

from simulation.nist_task_board_1 import NistTaskBoardEnvironment, OUTPUT_DIR
from simulation.nist_peg_task import TARGET_MESH


SOURCE = OUTPUT_DIR / "peg_pilot/20260905T142446744069Z/framework_human"
OUTPUT = SOURCE.parent / "offset_diagnosis"


def measure(environment, raw, target_id, target_geom, pad_ids):
    import mujoco
    from robosuite.utils.transform_utils import quat2mat

    sim = environment.sim
    eef = np.asarray(raw["robot0_eef_pos"])
    rotation = quat2mat(np.asarray(raw["robot0_eef_quat"]))
    peg = sim.data.body_xpos[target_id].copy()
    offset = peg - eef
    pads = [sim.data.geom_xpos[index].copy() for index in pad_ids]
    pad_midpoint = np.mean(pads, axis=0)
    contacts = []
    for index, contact in enumerate(sim.data.contact[:sim.data.ncon]):
        pair = [int(contact.geom1), int(contact.geom2)]
        if target_geom not in pair:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
        other = pair[1] if pair[0] == target_geom else pair[0]
        contacts.append({"other_geom": sim.model.geom_id2name(other),
                         "position_world_m": contact.pos.tolist(),
                         "penetration_mm": float(max(0, -contact.dist) * 1000),
                         "force_contact_frame_N": force[:3].tolist(),
                         "friction": contact.friction.tolist(),
                         "tangential_to_normal_ratio": float(np.linalg.norm(force[1:3]) / max(abs(force[0]), 1e-10))})
    controller = environment.robots[0].controller
    return {"eef_world_m": eef.tolist(), "eef_rotation_world": rotation.tolist(),
            "peg_world_m": peg.tolist(), "peg_rotation_world": sim.data.body_xmat[target_id].reshape(3, 3).tolist(),
            "peg_in_eef_mm": (rotation.T @ offset * 1000).tolist(),
            "peg_minus_eef_world_mm": (offset * 1000).tolist(),
            "pad_midpoint_in_eef_mm": (rotation.T @ (pad_midpoint - eef) * 1000).tolist(),
            "peg_minus_pad_midpoint_in_eef_mm": (rotation.T @ (peg - pad_midpoint) * 1000).tolist(),
            "gripper_qpos_m": np.asarray(raw["robot0_gripper_qpos"]).tolist(),
            "controller_goal_world_m": np.asarray(controller.goal_pos).tolist(),
            "contacts": contacts}


def main(variants=("baseline", "fixed_pose_closure", "no_finger_contact_closure",
                   "high_friction_transport", "static_hold_transport", "no_slip_transport",
                   "stiff_contacts_transport")):
    from robosuite.utils.control_utils import orientation_error
    from robosuite.utils.transform_utils import quat2mat

    OUTPUT.mkdir(parents=True, exist_ok=True)
    actions = json.loads((SOURCE / "actions.json").read_text())
    expected_hash = json.loads((SOURCE / "evaluation.json").read_text())["initial_state_sha256"]
    summary_path = OUTPUT / "summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for variant in variants:
        # Fresh environments also reset hidden controller/gripper state. A soft
        # reset after a rollout did not reproduce the saved initial state.
        with NistTaskBoardEnvironment(image_size=128) as environment:
            sim = environment.sim
            target_id = sim.model.body_name2id(f"nist_part_{TARGET_MESH}")
            target_geom = sim.model.geom_name2id(f"nist_part_{TARGET_MESH}_collision")
            geoms = environment.robots[0].gripper.important_geoms
            pad_ids = [sim.model.geom_name2id(name) for key in ("left_fingerpad", "right_fingerpad") for name in geoms[key]]
            gripper_ids = sorted({sim.model.geom_name2id(name) for names in geoms.values() for name in names})
            print("Baseline pad/peg friction:", sim.model.geom_friction[pad_ids + [target_geom]].tolist(), flush=True)
            raw = environment.reset(seed=1000)
            environment.set_control_mode("relative")
            for _ in range(40):
                raw, _, _, _ = environment.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
            state_hash = hashlib.sha256(np.r_[sim.data.qpos, sim.data.qvel].tobytes()).hexdigest()
            if state_hash != expected_hash:
                raise RuntimeError(f"{variant} initial state differs from saved episode")
            records = []
            anchor_pos = anchor_rot = None
            max_replay_error = 0.
            for item in actions:
                action = np.array(item["action"])
                step = item["step"]
                if step == 87:
                    anchor_pos = np.asarray(raw["robot0_eef_pos"]).copy()
                    anchor_rot = quat2mat(np.asarray(raw["robot0_eef_quat"]))
                    if variant == "no_finger_contact_closure":
                        # Remove only finger/peg collision. Keep each colliding
                        # with the floor and other normal (bit 1) scene geoms.
                        sim.model.geom_contype[target_geom] = 2
                        sim.model.geom_conaffinity[target_geom] = 1
                        for index in gripper_ids:
                            sim.model.geom_contype[index] = 4
                            sim.model.geom_conaffinity[index] = 1
                if variant == "fixed_pose_closure" and item["phase"] == "grasp":
                    controller = environment.robots[0].controller
                    action[:3] = np.clip((anchor_pos - raw["robot0_eef_pos"]) / controller.output_max[:3], -1, 1)
                    action[3:6] = np.clip(orientation_error(anchor_rot, quat2mat(raw["robot0_eef_quat"])) / controller.output_max[3:6], -1, 1)
                if variant == "high_friction_transport" and step == 112:
                    for index in pad_ids + [target_geom]:
                        sim.model.geom_friction[index, 0] *= 10
                if step == 112:
                    if variant == "static_hold_transport":
                        anchor_pos = np.asarray(raw["robot0_eef_pos"]).copy()
                        anchor_rot = quat2mat(np.asarray(raw["robot0_eef_quat"]))
                    if variant == "no_slip_transport":
                        sim.model.opt.noslip_iterations = 20
                    if variant == "stiff_contacts_transport":
                        for index in gripper_ids + [target_geom]:
                            sim.model.geom_solref[index] = [.004, 1.]
                            sim.model.geom_solimp[index, :3] = [.99, .999, .001]
                if variant == "static_hold_transport" and step >= 112:
                    controller = environment.robots[0].controller
                    action[:3] = np.clip((anchor_pos - raw["robot0_eef_pos"]) / controller.output_max[:3], -1, 1)
                    action[3:6] = np.clip(orientation_error(anchor_rot, quat2mat(raw["robot0_eef_quat"])) / controller.output_max[3:6], -1, 1)
                    action[-1] = 1.
                raw, _, _, _ = environment.step(action)
                row = {"step": step, "phase": item["phase"], **measure(environment, raw, target_id, target_geom, pad_ids)}
                records.append(row)
                if variant == "baseline":
                    max_replay_error = max(max_replay_error,
                        np.linalg.norm(np.array(row["eef_world_m"]) - item["eef_world_m"]),
                        np.linalg.norm(np.array(row["peg_world_m"]) - item["evaluation_only_target_center_world_m"]))
                if variant.endswith("closure") and step == 111:
                    break
                if variant.endswith("transport") and step == 192:
                    break
            if variant == "baseline" and max_replay_error > 1e-7:
                raise RuntimeError(f"Replay differs from saved episode: {max_replay_error} m")
            (OUTPUT / f"{variant}.json").write_text(json.dumps(records, indent=2))
            endpoints = {phase: [r for r in records if r["phase"] == phase][-1]
                         for phase in dict.fromkeys(r["phase"] for r in records)}
            summaries[variant] = {"max_replay_error_m": float(max_replay_error) if variant == "baseline" else None,
                                  "phase_endpoints": endpoints}
            for phase, row in endpoints.items():
                print(variant, phase, "peg/eef mm", np.round(row["peg_in_eef_mm"], 3),
                      "pad midpoint mm", np.round(row["pad_midpoint_in_eef_mm"], 3), flush=True)
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"Saved diagnosis: {OUTPUT}")


if __name__ == "__main__":
    main()
