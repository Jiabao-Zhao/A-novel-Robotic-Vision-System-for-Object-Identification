"""Contact-driven controller for the isolated runtime grasp experiment.

Controller provenance: adapted from the local round-pin-assembly-pilot's
``full_sequence.py`` (torque PD, dynamic fingers) and ``insertion_demo.py``
(damped Jacobian IK). Those frozen scripts are neither imported nor changed.
This module contains no robot hardware interfaces, attachment constraints,
object-pose correction, or candidate-specific control/contact settings.
"""

from pathlib import Path
import hashlib
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CONTACT_SETTINGS = {
    "solref": "0.006 1",
    "solimp": "0.95 0.99 0.001 0.5 2",
    "condim": "4",
    "priority": "2",
}
FRAME_DT_S = 0.04
KP = np.array([1600., 1600., 1400., 1400., 700., 500., 400.])
KD = np.array([80., 80., 70., 70., 35., 25., 20.])


def _validated_pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Expected a finite 4x4 world-frame pose in meters.")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous pose row.")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1., atol=1e-6
    ):
        raise ValueError("Pose rotation must be orthonormal with determinant +1.")
    return pose


class RobotTrial:
    """Panda torque simulation with one dynamic target and frozen other parts.

    Object poses exposed by ``target_pose`` and the recorded states are for the
    explicit perfect-initial-pose assumption and independent evaluation. Motion
    control uses robot joint/site state only. No target pose enters its feedback.
    """

    def __init__(self, scene_path, output_dir, target_name):
        source = Path(scene_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.target_name = (str(target_name) if str(target_name).startswith("nist_part_")
                            else "nist_part_" + str(target_name))
        tree = ET.parse(source)
        root = tree.getroot()
        for equality in list(root.findall("equality")):
            root.remove(equality)
        compiler = root.find("compiler")
        if compiler is not None:
            for attribute in ("meshdir", "texturedir"):
                relative = compiler.get(attribute)
                if relative and not Path(relative).is_absolute():
                    compiler.set(attribute, str((source.parent / relative).resolve()))
        # Resolve relative asset names before saving the experiment-local XML.
        for element in root.findall("asset/*"):
            file_name = element.get("file")
            if file_name and not Path(file_name).is_absolute():
                directory = "meshdir" if element.tag == "mesh" else "texturedir"
                base = (Path(compiler.get(directory)) if compiler is not None
                        and compiler.get(directory) else source.parent)
                element.set("file", str((base / file_name).resolve()))
        bodies = root.findall("worldbody//body")
        if not any(body.get("name") == self.target_name for body in bodies):
            raise ValueError(f"Target body not found: {self.target_name}")
        self.hidden_other_parts = []
        for body in bodies:
            body_name = body.get("name", "")
            if body_name.startswith("nist_part_"):
                hidden = body_name != self.target_name
                if hidden:
                    self.hidden_other_parts.append(body_name)
                for geom in body.findall(".//geom"):
                    if hidden:
                        geom.set("contype", "0")
                        geom.set("conaffinity", "0")
                        geom.set("rgba", "0 0 0 0")
                    elif int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
                        geom.attrib.update(CONTACT_SETTINGS)
        # Apply the same numerical contact settings to all physical fingertips.
        for body in bodies:
            if "finger" in body.get("name", ""):
                for geom in body.findall("geom"):
                    if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
                        geom.attrib.update(CONTACT_SETTINGS)
        self.scene_path = self.output_dir / "scene.xml"
        tree.write(self.scene_path, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.d = mujoco.MjData(self.m)
        key_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "objects_centered")
        if key_id < 0:
            raise ValueError("Source scene has no objects_centered keyframe.")
        mujoco.mj_resetDataKeyframe(self.m, self.d, key_id)
        mujoco.mj_forward(self.m, self.d)
        if self.m.neq:
            raise RuntimeError("Attachment/equality constraints must be absent.")
        self.joints = np.array([self.m.joint(f"robot0_joint{i}").id for i in range(1, 8)])
        self.qa = self.m.jnt_qposadr[self.joints]
        self.da = self.m.jnt_dofadr[self.joints]
        self.site = self.m.site("gripper0_grip_site").id
        self.site_body = int(self.m.site_bodyid[self.site])
        self.fjoints = np.array([self.m.joint(f"gripper0_finger_joint{i}").id for i in (1, 2)])
        self.fqa = self.m.jnt_qposadr[self.fjoints]
        self.fda = self.m.jnt_dofadr[self.fjoints]
        self.target_bid = self.m.body(self.target_name).id
        target_joint = int(self.m.body_jntadr[self.target_bid])
        if int(self.m.jnt_type[target_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Target must have a free joint.")
        self.target_qadr = int(self.m.jnt_qposadr[target_joint])
        self.target_dadr = int(self.m.jnt_dofadr[target_joint])
        self.arm_actuators = self._joint_actuators(self.joints)
        self.finger_actuators = self._joint_actuators(self.fjoints)
        self.fixed_qpos = self.d.qpos.copy()
        self.active_qpos = np.r_[self.qa, self.fqa,
                                np.arange(self.target_qadr, self.target_qadr + 7)]
        self.active_dofs = np.r_[self.da, self.fda,
                                np.arange(self.target_dadr, self.target_dadr + 6)]
        self.lower, self.upper = self.m.jnt_range[self.joints].T
        self.opening = .08
        self.phase = "initial"
        self.states = []
        self.contact_history = []
        self.audit = {key: 0. for key in ("robot_self", "robot_table", "robot_board",
                                        "target_board", "finger_target")}
        self.bilateral_frames = 0
        self.max_joint_frame_step_rad = 0.
        self._renderer = None
        self._record_frame()

    def _joint_actuators(self, joint_ids):
        result = []
        for joint_id in joint_ids:
            matches = np.flatnonzero(self.m.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError(f"Expected one actuator for joint {joint_id}.")
            result.append(int(matches[0]))
        return np.asarray(result)

    def pose(self):
        result = np.eye(4)
        result[:3, :3] = self.d.site_xmat[self.site].reshape(3, 3)
        result[:3, 3] = self.d.site_xpos[self.site]
        return result

    def target_pose(self):
        """Ground-truth target pose: initial assumed input or audit only."""
        result = np.eye(4)
        result[:3, :3] = self.d.xmat[self.target_bid].reshape(3, 3)
        result[:3, 3] = self.d.xpos[self.target_bid]
        return result

    def solve_ik(self, transform, seed=None):
        """Solve bounded pose IK in separate data; never move execution state."""
        target = _validated_pose(transform)
        data = mujoco.MjData(self.m)
        data.qpos[:] = self.d.qpos
        q = self.d.qpos[self.qa].copy() if seed is None else np.asarray(seed, dtype=float).copy()
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("IK seed must contain seven finite joint angles.")
        q = np.clip(q, self.lower + .003, self.upper - .003)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(350):
            data.qpos[self.qa] = q
            mujoco.mj_kinematics(self.m, data)
            mujoco.mj_comPos(self.m, data)
            pe = target[:3, 3] - data.site_xpos[self.site]
            re = Rotation.from_matrix(
                target[:3, :3] @ data.site_xmat[self.site].reshape(3, 3).T
            ).as_rotvec()
            if np.linalg.norm(pe) < .0001 and np.linalg.norm(re) < .0015:
                return q.copy()
            mujoco.mj_jac(self.m, data, jacp, jacr, data.site_xpos[self.site], self.site_body)
            jacobian = np.vstack((jacp[:, self.da], .3 * jacr[:, self.da]))
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + .00003 * np.eye(6), np.r_[pe, .3 * re]
            )
            delta *= min(1., .10 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, self.lower + .003, self.upper - .003)
        raise RuntimeError(f"IK failed: position_error_m={np.linalg.norm(pe):.6g}, "
                           f"rotation_error_rad={np.linalg.norm(re):.6g}")

    def _body_name(self, body_id):
        return self.m.body(int(body_id)).name or "world"

    def _robot_body(self, body_id):
        return self._body_name(body_id).startswith(("robot0_", "gripper0_"))

    def _fixed_mount(self, body_id):
        current = int(body_id)
        while current:
            if int(self.m.body_jntnum[current]):
                return False
            current = int(self.m.body_parentid[current])
        return True

    def _rigid_root(self, body_id):
        current = int(body_id)
        while current and not int(self.m.body_jntnum[current]):
            current = int(self.m.body_parentid[current])
        return current

    def _adjacent(self, first, second):
        first, second = self._rigid_root(first), self._rigid_root(second)
        return (first == second or self._rigid_root(self.m.body_parentid[first]) == second
                or self._rigid_root(self.m.body_parentid[second]) == first)

    def _contact_rows(self, data):
        rows = []
        sides = set()
        force = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geoms = [int(g) for g in contact.geom]
            bodies = [int(self.m.geom_bodyid[g]) for g in geoms]
            body_names = [self._body_name(b) for b in bodies]
            geom_names = [self.m.geom(g).name or f"geom_{g}" for g in geoms]
            robot = [self._robot_body(b) for b in bodies]
            table = [name == "workbench_top" or name == "floor" for name in geom_names]
            board = [name == "nist_board" or name.startswith("nist_fixture_")
                     for name in body_names]
            kind = None
            if all(robot) and not self._adjacent(*bodies):
                kind = "robot_self"
            elif any(robot) and any(table):
                robot_body = bodies[0] if robot[0] else bodies[1]
                if not self._fixed_mount(robot_body):
                    kind = "robot_table"
            elif any(robot) and any(board):
                kind = "robot_board"
            elif self.target_bid in bodies and any(board):
                kind = "target_board"
            elif self.target_bid in bodies and any("finger" in name for name in body_names):
                kind = "finger_target"
            if kind is None:
                continue
            mujoco.mj_contactForce(self.m, data, contact_id, force)
            if kind == "finger_target" and force[0] > .02:
                for name in body_names:
                    if "leftfinger" in name or "finger_joint1" in name:
                        sides.add("left")
                    if "rightfinger" in name or "finger_joint2" in name:
                        sides.add("right")
            rows.append({"kind": kind, "geoms": geom_names, "bodies": body_names,
                         "penetration_m": float(max(0., -contact.dist)),
                         "normal_force_n": float(force[0])})
        return rows, sorted(sides)

    def collision_report(self, q, opening, target_pose_override=None):
        """Planner report for nonadjacent robot self-contact and table contact.

        Board and target contact are separately audited during execution; this
        method alone is not a complete path/assembly collision certificate.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("Expected seven finite robot joint angles.")
        if not 0 <= float(opening) <= .08:
            raise ValueError("Panda jaw opening must be within 0–0.08 m.")
        data = mujoco.MjData(self.m)
        data.qpos[:] = self.d.qpos
        data.qpos[self.qa] = q
        data.qpos[self.fqa] = [float(opening) / 2, -float(opening) / 2]
        if target_pose_override is not None:
            target = _validated_pose(target_pose_override)
            xyzw = Rotation.from_matrix(target[:3, :3]).as_quat()
            data.qpos[self.target_qadr:self.target_qadr + 7] = np.r_[
                target[:3, 3], xyzw[[3, 0, 1, 2]]]
        mujoco.mj_forward(self.m, data)
        rows, _ = self._contact_rows(data)
        relevant = [row for row in rows if row["kind"] in ("robot_self", "robot_table")
                    and row["penetration_m"] > .00005]
        maximum = max((row["penetration_m"] for row in relevant), default=0.)
        limits_ok = bool(np.all(q >= self.lower + .001) and np.all(q <= self.upper - .001))
        return {"valid": limits_ok and not relevant, "joint_limits_ok": limits_ok,
                "max_penetration_m": maximum, "contacts": relevant}

    def gripper(self, opening):
        """Set actuated jaw opening in meters; call hold/move to step physics."""
        opening = float(opening)
        if not np.isfinite(opening) or not 0 <= opening <= .08:
            raise ValueError("Panda jaw opening must be within 0–0.08 m.")
        self.opening = opening

    def _tick(self, goal_q):
        start_q = self.d.qpos[self.qa].copy()
        delta = np.asarray(goal_q) - start_q
        self.max_joint_frame_step_rad = max(self.max_joint_frame_step_rad,
                                            float(np.max(np.abs(delta))))
        if np.max(np.abs(delta)) > .15:
            raise RuntimeError("Discontinuous joint waypoint exceeds 0.15 rad/frame.")
        substeps = max(1, int(round(FRAME_DT_S / self.m.opt.timestep)))
        duration = substeps * float(self.m.opt.timestep)
        for substep in range(substeps):
            active_qpos = self.d.qpos[self.active_qpos].copy()
            active_qvel = self.d.qvel[self.active_dofs].copy()
            self.d.qpos[:] = self.fixed_qpos
            self.d.qvel[:] = 0.
            self.d.qpos[self.active_qpos] = active_qpos
            self.d.qvel[self.active_dofs] = active_qvel
            mujoco.mj_forward(self.m, self.d)
            reference = start_q + delta * (substep + 1) / substeps
            torque = (self.d.qfrc_bias[self.da]
                      + KP * (reference - self.d.qpos[self.qa])
                      + KD * (delta / duration - self.d.qvel[self.da]))
            ranges = self.m.actuator_ctrlrange[self.arm_actuators]
            self.d.ctrl[:] = 0.
            self.d.ctrl[self.arm_actuators] = np.clip(torque, ranges[:, 0], ranges[:, 1])
            self.d.ctrl[self.finger_actuators] = [self.opening / 2, -self.opening / 2]
            mujoco.mj_step(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            rows, _ = self._contact_rows(self.d)
            for row in rows:
                self.audit[row["kind"]] = max(self.audit[row["kind"]], row["penetration_m"])
        self._record_frame()

    def _record_frame(self):
        rows, sides = self._contact_rows(self.d)
        if len(sides) == 2:
            self.bilateral_frames += 1
        self.states.append({"time_s": float(self.d.time), "phase": self.phase,
                            "robot_qpos": self.d.qpos[self.qa].tolist(),
                            "gripper_qpos": self.d.qpos[self.fqa].tolist(),
                            "target_pose_audit": self.target_pose().tolist(),
                            "grip_site_pose": self.pose().tolist(),
                            "grasp_contacts": sides, "commanded_opening_m": self.opening,
                            "qpos": self.d.qpos.tolist(), "qvel": self.d.qvel.tolist()})
        self.contact_history.append({"time_s": float(self.d.time), "phase": self.phase,
                                     "contacts": rows})

    def move(self, transform, frames=60, phase="move"):
        target = _validated_pose(transform)
        if int(frames) < 2:
            raise ValueError("A move requires at least two frames.")
        self.phase = str(phase)
        start = self.pose()
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))
        for fraction in np.linspace(0, 1, int(frames)):
            smooth = fraction * fraction * (3 - 2 * fraction)
            waypoint = np.eye(4)
            waypoint[:3, :3] = slerp(smooth).as_matrix()
            waypoint[:3, 3] = (1 - smooth) * start[:3, 3] + smooth * target[:3, 3]
            q = self.solve_ik(waypoint, self.d.qpos[self.qa])
            self._tick(q)
        return self.pose()

    def hold(self, frames, phase="hold"):
        self.phase = str(phase)
        goal_q = self.d.qpos[self.qa].copy()
        for _ in range(int(frames)):
            self._tick(goal_q)
        return self.pose()

    def render(self, path):
        """Render a snapshot without changing physics or camera calibration."""
        import cv2
        self.m.vis.global_.offwidth = max(960, self.m.vis.global_.offwidth)
        self.m.vis.global_.offheight = max(720, self.m.vis.global_.offheight)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, height=720, width=960)
        options = mujoco.MjvOption()
        options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
        options.sitegroup[:] = 0
        self._renderer.update_scene(self.d, camera="agentview", scene_option=options)
        rgb = self._renderer.render().copy()
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write render: {path}")
        return str(path)

    def save(self):
        """Save execution and audit evidence; never write into the source trial."""
        metadata = {"source_scene_sha256": self.source_sha256, "target_name": self.target_name,
                    "attachments_present": bool(self.m.neq),
                    "other_loose_parts_hidden_and_noncolliding": self.hidden_other_parts,
                    "other_objects_frozen": True, "target_feedback_in_controller": False,
                    "common_contact_settings": CONTACT_SETTINGS,
                    "friction_coefficients_changed": False,
                    "controller_provenance": "Adapted local round-pin-assembly-pilot full_sequence.py and insertion_demo.py",
                    "max_penetration_m": self.audit, "bilateral_contact_frames": self.bilateral_frames,
                    "max_joint_frame_step_rad": self.max_joint_frame_step_rad,
                    "physics_timestep_s": float(self.m.opt.timestep), "recording_frame_dt_s": FRAME_DT_S,
                    "trajectory_frames": len(self.states)}
        (self.output_dir / "controller_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (self.output_dir / "trajectory.json").write_text(json.dumps(self.states) + "\n")
        (self.output_dir / "contacts.json").write_text(json.dumps(self.contact_history) + "\n")
        np.savez_compressed(self.output_dir / "trajectory.npz",
                            time_s=np.asarray([row["time_s"] for row in self.states]),
                            qpos=np.asarray([row["qpos"] for row in self.states]),
                            qvel=np.asarray([row["qvel"] for row in self.states]),
                            phase=np.asarray([row["phase"] for row in self.states]))
        return metadata

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
