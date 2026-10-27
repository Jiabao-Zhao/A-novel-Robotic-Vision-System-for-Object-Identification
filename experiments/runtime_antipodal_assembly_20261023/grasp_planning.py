"""Online mesh contacts from GraspKit; Panda frames and scene checks are local.

All lengths are metres. ``world_T_grip`` refers to gripper0_grip_site;
its X axis closes the jaws and its +Z axis approaches the object.
"""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch

from .vendor import AntipodalSampler, AntipodalSamplerCfg


@dataclass
class Candidate:
    world_T_grip: np.ndarray
    contacts_world_m: np.ndarray
    width_m: float
    opening_m: float
    score: float
    roll_deg: float
    pad_depth_m: float
    table_clearance_m: float = 0.

    def record(self):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()}


def transform_points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def sample_contacts(mesh_path, world_T_cad, n_sample=2048):
    """Call the pinned upstream generator afresh on the workspace-aligned CAD."""
    started = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    vertices = transform_points(world_T_cad, np.asarray(mesh.vertices)).astype(np.float32)
    triangles = np.asarray(mesh.triangles).astype(np.int64)
    if not len(vertices) or not len(triangles):
        raise ValueError(f"Empty CAD mesh: {mesh_path}")
    sampler = AntipodalSampler(AntipodalSamplerCfg(
        n_sample=n_sample, min_length=.002, max_length=.075,
        max_angle=np.deg2rad(15)))
    pairs = sampler.sample(torch.from_numpy(vertices), torch.from_numpy(triangles)).numpy()
    pairs = pairs[np.all(np.isfinite(pairs), axis=(1, 2))]
    raw_count = len(pairs)
    # The pinned sampler tests one hit normal using an absolute cosine. Qualify
    # both endpoint normals explicitly before calling a contact pair opposing.
    scene = o3d.t.geometry.RaycastingScene()
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices), o3d.core.Tensor(triangles.astype(np.int32)))
    scene.add_triangles(tensor_mesh)
    if len(pairs):
        normals = scene.compute_closest_points(o3d.core.Tensor(
            pairs.reshape(-1, 3).astype(np.float32)))["primitive_normals"].numpy().reshape(-1, 2, 3)
        axis = pairs[:, 1] - pairs[:, 0]
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        aligned = np.all(np.abs(np.einsum('nij,nj->ni', normals, axis))
                         >= np.cos(np.deg2rad(15)), axis=1)
        opposing = np.sum(normals[:, 0] * normals[:, 1], axis=1) <= -np.cos(np.deg2rad(30))
        pairs = pairs[aligned & opposing]
    # The upstream path repeats rays at two distances. Deduplicate those contacts.
    canonical = pairs.copy()
    for pair in canonical:
        if tuple(pair[1]) < tuple(pair[0]):
            pair[:] = pair[::-1].copy()
    keys = np.round(canonical.reshape(-1, 6), 5)
    if len(pairs):
        _, indices = np.unique(keys, axis=0, return_index=True)
        pairs = pairs[np.sort(indices)]
    return mesh, vertices, triangles, pairs, {
        "raw_upstream_pairs": raw_count, "qualified_unique_pairs": len(pairs),
        "sampling_seconds": time.perf_counter() - started,
        "two_endpoint_normal_angle_deg": 15., "opposing_normal_angle_deg": 30.}


def gripper_surface_local(model, reference_data, opening_m):
    """Actual collidable gripper surfaces, in grip-site coordinates at a jaw width.

    Include every mesh vertex and box corner, rather than a point-gripper proxy.
    MuJoCo has already incorporated the asset mesh transform in geom_xpose.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = reference_data.qpos
    for index, sign in ((1, 1), (2, -1)):
        joint = model.joint(f"gripper0_finger_joint{index}").id
        data.qpos[model.jnt_qposadr[joint]] = sign * opening_m / 2
    mujoco.mj_forward(model, data)
    site = model.site("gripper0_grip_site").id
    position = data.site_xpos[site]
    rotation = data.site_xmat[site].reshape(3, 3)
    result = []
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        if not name.startswith("gripper0_"):
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[first:first + count]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                              for z in (-1, 1)]) * model.geom_size[geom]
        else:
            raise ValueError(f"Unsupported gripper collision geometry: {name}, {kind}")
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        result.append((world - position) @ rotation)
    if not result:
        raise ValueError("No collidable gripper geometry")
    return np.concatenate(result)


def candidate_frames(pairs, reference_rotation, cad_center_world, limit_pairs=160):
    """Expand contact pairs into jaw frames; no part-specific pose annotations."""
    pairs = np.asarray(pairs)
    if not len(pairs):
        return []
    midpoint = pairs.mean(axis=1)
    # Bound runtime uniformly, spreading samples across the returned contacts.
    indices = np.linspace(0, len(pairs) - 1, min(limit_pairs, len(pairs)), dtype=int)
    candidates = []
    for index in indices:
        pair = pairs[index]
        axis = pair[1] - pair[0]
        width = np.linalg.norm(axis)
        if not .002 < width < .075:
            continue
        axis /= width
        down = np.array([0., 0., -1.])
        down -= axis * np.dot(axis, down)
        if np.linalg.norm(down) < .05:
            down = np.array([1., 0., 0.])
            down -= axis * np.dot(axis, down)
        down /= np.linalg.norm(down)
        for roll in (0., 45., -45., 90., -90., 180.):
            approach = Rotation.from_rotvec(axis * np.deg2rad(roll)).apply(down)
            if approach[2] > .1:
                continue
            for jaw_sign in (1., -1.):
                x_axis = axis * jaw_sign
                rotation = np.column_stack((x_axis, np.cross(approach, x_axis), approach))
                distance = np.linalg.norm(Rotation.from_matrix(
                    rotation @ reference_rotation.T).as_rotvec())
                for depth in (.004, 0., -.004, -.008):
                    # These are generic pad-depth samples within the actual Panda
                    # pad interval [-.0116,.0044], not object-specific grasp poses.
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = midpoint[index] - rotation @ [0., 0., depth]
                    center_cost = np.linalg.norm(midpoint[index] - cad_center_world) / .05
                    vertical_cost = 1. + approach[2]
                    score = center_cost + .25 * distance + .5 * vertical_cost
                    candidates.append(Candidate(pose, pair.copy(), float(width),
                                                min(.08, float(width + .012)),
                                                float(score), roll, depth))
    return sorted(candidates, key=lambda item: item.score)


def table_filter(candidates, model, data, table_z=0., margin_m=.0001):
    """Conservative gripper/table test using the complete collision surfaces."""
    cache = {}
    kept = []
    for candidate in candidates:
        key = round(candidate.opening_m, 8)
        if key not in cache:
            cache[key] = gripper_surface_local(model, data, candidate.opening_m)
        points = transform_points(candidate.world_T_grip, cache[key])
        minimum = float(points[:, 2].min() - table_z)
        # The open gripper moves backward along tool Z for its approach. Include
        # that path analytically for a horizontal support plane.
        minimum += min(0., float(-candidate.world_T_grip[2, 2] * .10))
        candidate.table_clearance_m = minimum
        if minimum >= margin_m:
            kept.append(candidate)
    return kept


def target_obstruction_report(trial, q, opening):
    """Reject unintended target penetration before contact closure.

    The open hand must clear the target, independently of the optional external
    obstacle policy. Intended fingertip contact is established during closing.
    """
    data = mujoco.MjData(trial.m)
    data.qpos[:] = trial.d.qpos
    data.qpos[trial.qa] = q
    data.qpos[trial.fqa] = [opening / 2, -opening / 2]
    mujoco.mj_forward(trial.m, data)
    contacts = []
    for contact in data.contact:
        first, second = [int(trial.m.geom_bodyid[g]) for g in contact.geom]
        if trial.target_bid not in (first, second):
            continue
        other = second if first == trial.target_bid else first
        name = trial.m.body(other).name or ""
        if name.startswith(("robot0_", "gripper0_")) and contact.dist < -.00005:
            contacts.append({"body": name, "penetration_m": float(-contact.dist)})
    return {"valid": not contacts, "contacts": contacts}


def linear_pose_path(first, last, count):
    from scipy.spatial.transform import Slerp
    rotations = Slerp([0, 1], Rotation.from_matrix([first[:3, :3], last[:3, :3]]))
    result = []
    for t in np.linspace(0, 1, count):
        pose = np.eye(4)
        pose[:3, :3] = rotations(t).as_matrix()
        pose[:3, 3] = first[:3, 3] * (1-t) + last[:3, 3] * t
        result.append(pose)
    return result


def validate_path(trial, poses, opening, *, seed=None, target_relative=None,
                  check_open_target=False):
    """Check sampled path IK and robot/table/self contacts; no loose obstacles."""
    q = trial.d.qpos[trial.qa].copy() if seed is None else seed.copy()
    qpath, report = [], {"checked_waypoints": 0, "max_penetration_m": 0.}
    for pose in poses:
        try:
            q = trial.solve_ik(pose, seed=q)
        except RuntimeError as error:
            return None, {**report, "status": "ik_failed", "reason": str(error)}
        target = None if target_relative is None else pose @ target_relative
        check = trial.collision_report(q, opening, target_pose_override=target)
        report["checked_waypoints"] += 1
        report["max_penetration_m"] = max(report["max_penetration_m"],
                                           check["max_penetration_m"])
        if not check["valid"]:
            return None, {**report, "status": "collision", "contacts": check["contacts"]}
        if check_open_target:
            obstruction = target_obstruction_report(trial, q, opening)
            if not obstruction["valid"]:
                return None, {**report, "status": "target_obstruction",
                              "contacts": obstruction["contacts"]}
        qpath.append(q.copy())
    return qpath, {**report, "status": "passed"}
