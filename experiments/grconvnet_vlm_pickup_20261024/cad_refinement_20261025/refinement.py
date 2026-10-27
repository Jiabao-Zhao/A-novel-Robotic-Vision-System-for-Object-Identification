"""Bounded CAD correction around GR-ConvNet seeds; no IK or physics search.

All geometry is in the existing metric world frame. Each network seed retains
its closing direction and yields at most one pose. No global contact sampler,
part-specific grasp, model retraining or execution feedback is used.
"""
import hashlib
from pathlib import Path
import time

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from experiments.grconvnet_vlm_pickup_20261024.capture import rectangle_geometry
from experiments.runtime_antipodal_assembly_20261023.grasp_planning import (
    Candidate, gripper_surface_local, transform_points,
)


ACROSS_OFFSETS_M = (0., -.002, .002, -.004, .004)
HEIGHT_FRACTIONS = (.5, .25, .75)
FINGER_DEPTHS_M = (.004, 0., -.004, -.008, .006, .008)
OPENING_PADDING_M = (.012, .016, .020)
MAX_SEED_SHIFT_M = .010
MAX_NORMAL_ANGLE_DEG = 30.
TABLE_MARGIN_M = .0001
TARGET_OVERLAP_TOLERANCE_M = .00005
MAX_OPENING_M = .080
MAX_CONTACT_SPAN_M = .075
XY_RANK_RESOLUTION_M = .0001


def protocol():
    return {"name": "GR-seeded bounded CAD refinement", "units": "metres, world frame",
        "local_offsets_across_jaw_m": ACROSS_OFFSETS_M,
        "local_offset_along_jaw_m": 0.,
        "aligned_CAD_world_z_height_fractions": HEIGHT_FRACTIONS,
        "finger_depths_m": FINGER_DEPTHS_M, "opening_padding_m": OPENING_PADDING_M,
        "maximum_final_contact_midpoint_xy_shift_from_GR_seed_m": MAX_SEED_SHIFT_M,
        "maximum_endpoint_normal_angle_deg": MAX_NORMAL_ANGLE_DEG,
        "table_margin_m": TABLE_MARGIN_M, "target_overlap_tolerance_m": TARGET_OVERLAP_TOLERANCE_M,
        "maximum_contact_span_m": MAX_CONTACT_SPAN_M, "maximum_opening_m": MAX_OPENING_M,
        "maximum_CAD_lines_per_seed": len(ACROSS_OFFSETS_M)*len(HEIGHT_FRACTIONS),
        "maximum_pose_opening_variants_per_seed": len(ACROSS_OFFSETS_M)*len(HEIGHT_FRACTIONS)*len(FINGER_DEPTHS_M)*len(OPENING_PADDING_M),
        "maximum_returned_poses_per_network_seed": 1, "maximum_network_seeds": 4,
        "selection": "First geometry-feasible variant sorted by XY shift rounded to 0.1 mm, distance from CAD mid-height, fixed finger-depth order, opening padding, normal agreement, stable CAD-line index",
        "static_target_check": "Move target into the unchanged initial robot hand frame, set jaw opening, mj_forward only; inspect target/gripper contacts",
        "jaw_direction": "Same horizontal closing line as each original GR seed; only equivalent 180-degree jaw exchange chosen nearest initial robot orientation",
        "no_global_sampler": True, "no_part_specific_grasp_annotations": True,
        "no_inverse_kinematics_in_refinement": True, "no_physics_steps": True,
        "no_VLM_or_execution_outcome_in_variant_selection": True}


def rigid_target_at_initial_hand(reference_grip, proposed_grip, world_T_cad):
    """Preserve hand/target relative pose while keeping the robot unchanged."""
    return reference_grip @ np.linalg.inv(proposed_grip) @ world_T_cad


def static_target_report(trial, pose, opening_m, world_T_cad):
    """Collision-model check with valid MuJoCo forward-position caches."""
    data = mujoco.MjData(trial.m)
    data.qpos[:] = trial.d.qpos
    data.qpos[trial.fqa] = [opening_m/2, -opening_m/2]
    moved_target = rigid_target_at_initial_hand(trial.pose(), pose, world_T_cad)
    xyzw = Rotation.from_matrix(moved_target[:3, :3]).as_quat()
    data.qpos[trial.target_qadr:trial.target_qadr+7] = np.r_[moved_target[:3, 3], xyzw[[3, 0, 1, 2]]]
    mujoco.mj_forward(trial.m, data)
    contacts = []
    for contact in data.contact:
        bodies = [int(trial.m.geom_bodyid[g]) for g in contact.geom]
        if trial.target_bid not in bodies:
            continue
        other = bodies[1] if bodies[0] == trial.target_bid else bodies[0]
        name = trial.m.body(other).name or ""
        if not name.startswith("gripper0_"):
            continue
        contacts.append({"gripper_body": name,
            "geoms": [trial.m.geom(int(g)).name for g in contact.geom],
            "signed_distance_m": float(contact.dist), "overlap_m": float(max(0., -contact.dist))})
    worst = max((contact["overlap_m"] for contact in contacts), default=0.)
    return {"passed": worst <= TARGET_OVERLAP_TOLERANCE_M,
            "maximum_gripper_target_overlap_m": worst, "contacts": contacts,
            "physics_steps": 0, "robot_joint_angles_changed": False}


class AlignedCAD:
    def __init__(self, mesh_path, world_T_cad):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.vertices = transform_points(world_T_cad, np.asarray(mesh.vertices))
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        if not len(self.vertices) or not len(faces):
            raise ValueError("Empty native CAD mesh")
        self.zmin, self.zmax = self.vertices[:, 2].min(), self.vertices[:, 2].max()
        self.scene = o3d.t.geometry.RaycastingScene()
        tensor = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(self.vertices.astype(np.float32)), o3d.core.Tensor(faces))
        self.scene.add_triangles(tensor)
        self.mesh_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

    def local_pairs(self, seed_center, axis):
        """Outer ray hits only on a fixed local grid around the supplied seed."""
        across = np.cross([0., 0., -1.], axis)
        starts, definitions = [], []
        for fraction in HEIGHT_FRACTIONS:
            z = self.zmin+fraction*(self.zmax-self.zmin)
            for offset in ACROSS_OFFSETS_M:
                center = np.array([seed_center[0], seed_center[1], z])+offset*across
                reach = float(np.linalg.norm(self.vertices-center, axis=1).max()+.02)
                starts.extend([np.r_[center-reach*axis, axis], np.r_[center+reach*axis, -axis]])
                definitions.append({"fraction": fraction, "along_m": 0., "across_m": offset})
        rays = np.asarray(starts, np.float32)
        hits = self.scene.cast_rays(o3d.core.Tensor(rays))
        distances = hits["t_hit"].numpy().reshape(-1, 2)
        normals = hits["primitive_normals"].numpy().reshape(-1, 2, 3)
        points = (rays[:, :3]+rays[:, 3:]*hits["t_hit"].numpy()[:, None]).reshape(-1, 2, 3)
        cosine = np.cos(np.deg2rad(MAX_NORMAL_ANGLE_DEG))
        kept, rejected = [], {"missed_CAD": 0, "normal_angle": 0, "contact_span": 0, "seed_shift": 0}
        for index, definition in enumerate(definitions):
            if not np.all(np.isfinite(distances[index])):
                rejected["missed_CAD"] += 1
                continue
            pair = points[index].astype(float)
            normal = normals[index].astype(float)
            agreement = np.abs(normal @ axis)
            if (np.any(agreement < cosine)
                    or np.dot(normal[0], normal[1]) > -np.cos(np.deg2rad(2*MAX_NORMAL_ANGLE_DEG))):
                rejected["normal_angle"] += 1
                continue
            span = float(np.linalg.norm(pair[1]-pair[0]))
            if not .002 < span < MAX_CONTACT_SPAN_M:
                rejected["contact_span"] += 1
                continue
            midpoint = pair.mean(0)
            shift = float(np.linalg.norm(midpoint[:2]-seed_center[:2]))
            if shift > MAX_SEED_SHIFT_M:
                rejected["seed_shift"] += 1
                continue
            kept.append({**definition, "line_index": index, "contacts_world_m": pair,
                         "midpoint_world_m": midpoint, "span_m": span,
                         "xy_shift_m": shift, "surface_normals_world_unit": normal.tolist(),
                         "normal_axis_cosines": agreement.tolist()})
        return kept, {"CAD_lines_tested": len(definitions), "qualified_CAD_lines": len(kept), "rejected_lines": rejected}


def refine_candidates(prediction, observation, trial, mesh_path, initial):
    """Return one CAD-corrected pose per GR seed, preserving network ordering."""
    started = time.perf_counter()
    mesh_path = Path(mesh_path)
    cad = AlignedCAD(mesh_path, initial)
    table_z = float(observation["calibration"]["table_z_world_m"])
    pool, records, rejected, per_seed = [], [], [], []
    surface_cache = {}
    original_qpos, original_time = trial.d.qpos.copy(), float(trial.d.time)
    for proposal in prediction["proposals"]:
        summary = {"network_candidate_id": proposal["network_candidate_id"], "proposal": proposal,
                   "table_checks": 0, "static_target_checks": 0, "table_rejections": 0,
                   "target_rejections": 0, "geometry_checks": []}
        try:
            before = rectangle_geometry(proposal, observation["depth_m"], observation["target_mask"], observation["calibration"])
            summary["before_depth_geometry"] = before
            center = np.asarray(before["center_world_m"])
            axis = np.asarray(before["jaw_axis_world_unit"])
            axis[2] = 0.
            axis /= np.linalg.norm(axis)
            down = np.array([0., 0., -1.])
            options = [np.column_stack((axis*sign, np.cross(down, axis*sign), down)) for sign in (1., -1.)]
            rotation = min(options, key=lambda value: np.linalg.norm(Rotation.from_matrix(value @ trial.pose()[:3, :3].T).as_rotvec()))
            pairs, ray_summary = cad.local_pairs(center, axis)
            summary.update(ray_summary)
            variants = []
            for pair in pairs:
                for depth_index, finger_depth in enumerate(FINGER_DEPTHS_M):
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = pair["midpoint_world_m"]-rotation[:, 2]*finger_depth
                    seen_openings = set()
                    for pad_index, padding in enumerate(OPENING_PADDING_M):
                        opening = min(MAX_OPENING_M, pair["span_m"]+padding)
                        if opening in seen_openings:
                            continue
                        seen_openings.add(opening)
                        rank = (round(pair["xy_shift_m"]/XY_RANK_RESOLUTION_M),
                                abs(pair["fraction"]-.5), depth_index, pad_index,
                                -min(pair["normal_axis_cosines"]), pair["line_index"])
                        variants.append((rank, pair, finger_depth, opening, pose))
            variants.sort(key=lambda value: value[0])
            summary["pose_opening_variants_formed"] = len(variants)
            selected = None
            for rank, pair, finger_depth, opening, pose in variants:
                key = round(opening, 10)
                if key not in surface_cache:
                    surface_cache[key] = gripper_surface_local(trial.m, trial.d, opening)
                points = transform_points(pose, surface_cache[key])
                minimum = float(points[:, 2].min()-table_z)
                minimum += min(0., float(-pose[2, 2]*.10))
                summary["table_checks"] += 1
                variant_check = {"geometry_rank": list(rank), "CAD_line_index": pair["line_index"],
                    "finger_depth_m": finger_depth, "opening_m": opening,
                    "contact_midpoint_world_m": pair["midpoint_world_m"].tolist(),
                    "minimum_gripper_table_gap_m": minimum,
                    "table_passed": minimum >= TABLE_MARGIN_M}
                summary["geometry_checks"].append(variant_check)
                if minimum < TABLE_MARGIN_M:
                    summary["table_rejections"] += 1
                    continue
                check = static_target_report(trial, pose, opening, initial)
                variant_check["open_hand_target_check"] = check
                summary["static_target_checks"] += 1
                if not check["passed"]:
                    summary["target_rejections"] += 1
                    continue
                candidate = Candidate(pose, pair["contacts_world_m"].copy(), pair["span_m"], opening,
                                      -proposal["quality"], 0., finger_depth, minimum)
                selected = candidate
                after = {"surface_center_world_m": pair["midpoint_world_m"].tolist(),
                         "center_world_m": pair["midpoint_world_m"].tolist(),
                         "jaw_axis_world_unit": axis.tolist(), "predicted_width_m": pair["span_m"],
                         "jaw_endpoints_world_m": pair["contacts_world_m"].tolist(),
                         "estimated_height_m": float(cad.zmax-cad.zmin),
                         "depth_method": "Aligned CAD correction; original image depth retained separately under before_depth_geometry",
                         "center_height_policy": "Aligned CAD horizontal slice, bounded around the original GR seed",
                         "object_pose_used": True}
                correction = {"xy_shift_m": pair["xy_shift_m"],
                    "contact_center_height_change_m": float(pair["midpoint_world_m"][2]-center[2]),
                    "contact_span_change_m": pair["span_m"]-before["predicted_width_m"],
                    "chosen_height_fraction": pair["fraction"], "CAD_line_index": pair["line_index"],
                    "local_along_offset_m": pair["along_m"], "local_across_offset_m": pair["across_m"],
                    "contact_normal_axis_cosines": pair["normal_axis_cosines"],
                    "surface_normals_world_unit": pair["surface_normals_world_unit"],
                    "chosen_geometry_rank": list(rank), "open_hand_target_check": check,
                    "seed_jaw_axis_world_unit": axis.tolist(), "jaw_direction_changed": False}
                summary["selected_correction"] = correction
                pool.append(candidate)
                records.append({"network_candidate_id": proposal["network_candidate_id"],
                    "proposal": proposal, "before_depth_geometry": before, "depth_geometry": after,
                    "CAD_correction": correction, "chosen_depth_policy": protocol()["selection"],
                    "predicted_endpoints_are_actual_surface_contacts": True,
                    "CAD_alignment_assumed_exact": True, **candidate.record()})
                break
            if selected is None:
                reason = "No geometry-feasible variant within the frozen local GR-seed bounds"
                rejected.append({"network_candidate_id": proposal["network_candidate_id"], "reason": reason,
                                 "proposal": proposal, "CAD_refinement": summary})
        except ValueError as error:
            rejected.append({"network_candidate_id": proposal["network_candidate_id"], "reason": str(error), "proposal": proposal})
        per_seed.append(summary)
    np.testing.assert_array_equal(trial.d.qpos, original_qpos)
    assert float(trial.d.time) == original_time
    return pool, records, rejected, {"protocol": protocol(), "mesh_sha256": cad.mesh_sha256,
        "aligned_CAD_world_z_range_m": [float(cad.zmin), float(cad.zmax)],
        "table_z_world_m": table_z,
        "per_network_seed": per_seed, "refinement_seconds": time.perf_counter()-started,
        "input_robot_state_unchanged": True, "physics_steps": 0, "model_calls": 0, "VLM_calls": 0}
