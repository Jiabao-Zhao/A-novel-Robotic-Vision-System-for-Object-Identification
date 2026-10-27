"""Robot/static-board planning with a fresh observation-fitted CAD surrogate.

The template must contain no loose products or saved state. Four CAD-Z convex
bands preserve stepped outlines conservatively; their hulls close CAD bores.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial, _validated_pose
from experiments.runtime_antipodal_assembly_20261023.grasp_planning import validate_path


SURROGATE_BODY = "estimated_cad_surrogate"
SURROGATE_JOINT = "estimated_cad_free_joint"
COLLISION_BANDS = 4
OVERLAP_LIMIT_M = .00005
ROBOT_PREFIXES = ("robot0_", "gripper0_")


def native_triangles(path):
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * 50:
        raise ValueError("Expected binary native-metre STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].astype(float)
    if not vertices.size or not np.all(np.isfinite(vertices)):
        raise ValueError("CAD has no finite triangles")
    return vertices


def band_vertices(triangles, count=COLLISION_BANDS):
    """Clip triangle edges to fixed CAD-Z slabs, then let MuJoCo hull each slab."""
    vertices = triangles.reshape(-1, 3)
    zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
    if zmax - zmin < 1e-6:
        raise ValueError("CAD must have nonzero native-Z thickness")
    first = triangles[:, [0, 1, 2]].reshape(-1, 3)
    last = triangles[:, [1, 2, 0]].reshape(-1, 3)
    dz = last[:, 2] - first[:, 2]
    bands = []
    edges = np.linspace(zmin, zmax, count + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        values = [vertices[(vertices[:, 2] >= lower - 1e-10) & (vertices[:, 2] <= upper + 1e-10)]]
        for level in (lower, upper):
            valid = (abs(dz) > 1e-12) & (np.minimum(first[:, 2], last[:, 2]) <= level) & (np.maximum(first[:, 2], last[:, 2]) >= level)
            fraction = (level - first[valid, 2]) / dz[valid]
            values.append(first[valid] + fraction[:, None] * (last[valid] - first[valid]))
        points = np.unique(np.round(np.concatenate(values), 10), axis=0)
        if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(0)) < 3:
            raise ValueError("CAD band has no three-dimensional collision hull")
        bands.append(points)
    return bands


class PlanningModel:
    """Known robot + static board + one explicitly estimated CAD body."""

    # Reuse the established bounded IK and robot topology, without its plant init.
    pose = RobotTrial.pose
    solve_ik = RobotTrial.solve_ik
    _body_name = RobotTrial._body_name
    _robot_body = RobotTrial._robot_body
    _fixed_mount = RobotTrial._fixed_mount
    _rigid_root = RobotTrial._rigid_root
    _adjacent = RobotTrial._adjacent

    def __init__(self, template_path, feedback, mesh_path, estimated_pose):
        template_path, mesh_path = Path(template_path).resolve(), Path(mesh_path).resolve()
        tree = ET.parse(template_path)
        root = tree.getroot()
        if root.find("keyframe") is not None:
            raise ValueError("Planning template must not contain saved keyframes")
        for body in root.findall(".//body"):
            if body.get("name", "").startswith("nist_part_"):
                raise ValueError("Planning template contains a loose-product body")
        for joint in root.findall(".//joint"):
            # Defaults can contain unnamed joint settings, not movable state.
            if joint.get("name") and not joint.get("name").startswith(ROBOT_PREFIXES):
                raise ValueError("Planning template contains a nonrobot movable joint")
        compiler = root.find("compiler")
        for element in root.findall("asset/*"):
            name = element.get("file")
            if not name or Path(name).is_absolute():
                continue
            directory = "meshdir" if element.tag == "mesh" else "texturedir"
            base = template_path.parent
            if compiler is not None and compiler.get(directory):
                base = Path(compiler.get(directory))
                if not base.is_absolute():
                    base = template_path.parent / base
            element.set("file", str((base / name).resolve()))
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", {"name": "estimated_cad_native", "file": str(mesh_path)})
        triangles = native_triangles(mesh_path)
        bands = band_vertices(triangles)
        body = ET.SubElement(root.find("worldbody"), "body", {"name": SURROGATE_BODY})
        ET.SubElement(body, "joint", {"name": SURROGATE_JOINT, "type": "free"})
        ET.SubElement(body, "geom", {"name": "estimated_cad_visual", "type": "mesh",
            "mesh": "estimated_cad_native", "contype": "0", "conaffinity": "0", "group": "1", "mass": "0"})
        for index, points in enumerate(bands):
            name = f"estimated_cad_band_{index}"
            ET.SubElement(asset, "mesh", {"name": name, "vertex": " ".join(f"{value:.10g}" for value in points.ravel())})
            ET.SubElement(body, "geom", {"name": name, "type": "mesh", "mesh": name,
                "group": "0", "density": "1000", "margin": "0", "gap": "0"})
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.joints = np.array([self.m.joint(f"robot0_joint{i}").id for i in range(1, 8)])
        self.qa, self.da = self.m.jnt_qposadr[self.joints], self.m.jnt_dofadr[self.joints]
        fingers = np.array([self.m.joint(f"gripper0_finger_joint{i}").id for i in (1, 2)])
        self.fqa, self.fda = self.m.jnt_qposadr[fingers], self.m.jnt_dofadr[fingers]
        self.site = self.m.site("gripper0_grip_site").id
        self.site_body = int(self.m.site_bodyid[self.site])
        self.lower, self.upper = self.m.jnt_range[self.joints].T
        self.target_bid = self.m.body(SURROGATE_BODY).id
        joint = self.m.joint(SURROGATE_JOINT).id
        self.target_qadr = int(self.m.jnt_qposadr[joint])
        self.target_dadr = int(self.m.jnt_dofadr[joint])
        self.provenance = {
            "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
            "native_CAD_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "surrogate_body": SURROGATE_BODY, "pose_source": "Observation-fitted CAD pose supplied by caller",
            "collision_model": "Four equal-height native-CAD-Z convex hull bands",
            "collision_limit": "Conservative hulls close bores and concavities within each band",
            "plant_state_used": False, "physics_steps": 0,
        }
        self.set_estimated_pose(estimated_pose)
        self.set_feedback(feedback)

    def set_estimated_pose(self, estimated_pose):
        pose = _validated_pose(estimated_pose)
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        self.d.qpos[self.target_qadr:self.target_qadr + 7] = np.r_[pose[:3, 3], quat[[3, 0, 1, 2]]]
        self.d.qvel[self.target_dadr:self.target_dadr + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def set_feedback(self, feedback):
        expected = [f"robot0_joint{i}" for i in range(1, 8)]
        if feedback["joint_names"] != expected:
            raise ValueError("Robot feedback joint names/order disagree")
        expected_fingers = [f"gripper0_finger_joint{i}" for i in (1, 2)]
        if feedback["finger_joint_names"] != expected_fingers:
            raise ValueError("Finger feedback names/order disagree")
        self.d.qpos[self.qa] = np.asarray(feedback["joint_position_rad"], float)
        self.d.qvel[self.da] = np.asarray(feedback["joint_velocity_rad_s"], float)
        self.d.qpos[self.fqa] = np.asarray(feedback["finger_position_m"], float)
        mujoco.mj_forward(self.m, self.d)
        if not np.allclose(self.pose(), np.asarray(feedback["world_T_grip"]), atol=1e-7):
            raise ValueError("Robot FK disagrees with measured robot feedback")

    def collision_report(self, q, opening, target_pose_override=None):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("Expected seven finite robot joint angles")
        if not 0 <= float(opening) <= .08:
            raise ValueError("Panda finger opening must lie within 0 to 80 mm")
        data = mujoco.MjData(self.m)
        data.qpos[:] = self.d.qpos
        data.qpos[self.qa] = q
        data.qpos[self.fqa] = [opening / 2, -opening / 2]
        if target_pose_override is not None:
            pose = _validated_pose(target_pose_override)
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            data.qpos[self.target_qadr:self.target_qadr + 7] = np.r_[pose[:3, 3], quat[[3, 0, 1, 2]]]
        mujoco.mj_forward(self.m, data)
        contacts, estimated_contacts = [], []
        for contact in data.contact:
            bodies = [int(self.m.geom_bodyid[int(geom)]) for geom in contact.geom]
            robot = [self._robot_body(body) for body in bodies]
            kind = None
            if all(robot) and not self._adjacent(*bodies):
                kind = "robot_self"
            elif any(robot) and self.target_bid not in bodies:
                robot_body = bodies[0] if robot[0] else bodies[1]
                if not self._fixed_mount(robot_body):
                    kind = "robot_static_table_or_board"
            row = {"kind": kind, "bodies": [self._body_name(body) for body in bodies],
                   "geoms": [self.m.geom(int(geom)).name for geom in contact.geom],
                   "penetration_m": float(max(0., -contact.dist))}
            if kind and row["penetration_m"] > OVERLAP_LIMIT_M:
                contacts.append(row)
            elif self.target_bid in bodies and not any(robot):
                estimated_contacts.append({**row, "kind": "estimated_CAD_static_contact"})
        limits = bool(np.all(q >= self.lower + .001) and np.all(q <= self.upper - .001))
        return {"valid": limits and not contacts, "joint_limits_ok": limits,
                "max_penetration_m": max((row["penetration_m"] for row in contacts), default=0.),
                "contacts": contacts, "estimated_CAD_static_contacts_diagnostic": estimated_contacts,
                "estimated_CAD_static_contacts_veto_motion": False,
                "physics_steps": 0}

    def check_path(self, poses, opening, *, seed=None, target_relative=None, check_open_target=False):
        """Sampled robot self/table/board checks; estimated open-hand CAD optional."""
        return validate_path(self, poses, opening, seed=seed,
                             target_relative=target_relative, check_open_target=check_open_target)
