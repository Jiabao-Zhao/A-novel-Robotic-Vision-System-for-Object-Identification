"""Known-hole NIST round-peg pilot: perception, motion, and separate scoring."""

import json

import numpy as np

from .nist_task_board_1 import ASSET_DIR, BOARD_BOTTOM_Z_M, BOARD_CENTER_XY_M, BOARD_SIZE_M


TARGET_DESCRIPTION = "16 mm diameter round peg"
TARGET_MESH = "RGOCG16-50_16mm"
PEG_LENGTH_M = 0.050
PEG_RADIUS_M = 0.008
HOLE_RADIUS_M = 0.0081
# Supplied design pose, NOT read from the live simulator or inferred by the VLM.
# Local +Z points out of the board. Insertion proceeds along local -Z.
WORLD_T_HOLE = np.eye(4)
WORLD_T_HOLE[:2, 3] = (np.array([266.45, 122.176]) - 192) * .001 + BOARD_CENTER_XY_M
WORLD_T_HOLE[2, 3] = BOARD_BOTTOM_Z_M + BOARD_SIZE_M[2]
TASK_TEXT = "Insert the 16mm round peg into the vertical hole at world XYZ meters: " + ", ".join(
    f"{value:.6f}" for value in WORLD_T_HOLE[:3, 3]) + "."
HORIZON = 400


def localize_parts(observation, rgb_path, output_dir):
    """Depth-only localization within the parts work area, with no instance poses."""
    from point_cloud_localization import PointCloudConfig, PointCloudLocalization
    from .perception_adapter import mask_depth_to_world_workspace, pointcloud_localization_inputs

    depth, _ = mask_depth_to_world_workspace(observation, (-.34, -.36, -.04), (.36, -.16, .12))
    depth_path = output_dir / "workspace_depth.npy"
    np.save(depth_path, depth)
    config = PointCloudConfig(
        output_dir=output_dir / "point_cloud", annotation_dir=output_dir / "annotation",
        voxel_size_m=.0008, plane_distance_threshold_m=.0008,
        dbscan_eps_m=.006, dbscan_min_points=4, min_cluster_points=12,
        min_cluster_extent_m=.002, max_cluster_aspect_ratio=20,
        min_roi_width_px=3, min_roi_height_px=3,
        raw_cluster_dbscan_eps_m=.003, raw_cluster_dbscan_min_points=3,
        statistical_outlier_neighbors=8, radius_outlier_radius_m=.004,
        radius_outlier_min_neighbors=2,
    )
    paths = PointCloudLocalization(config).run_from_arrays(**pointcloud_localization_inputs(
        observation, rgb_path, depth_path, depth_m=depth))
    return json.loads(paths["localization"].read_text()), paths


def estimate_peg_pose(localization, object_id, world_T_camera, output_dir):
    """Use the semantic CAD prior and the selected observed cloud only."""
    from CADPointCloudRegistration import CADPointCloudRegistration, CADRegistrationConfig

    candidate = next(item for item in localization["objects"] if item["object_id"] == object_id)
    registrar = CADPointCloudRegistration(CADRegistrationConfig(
        voxel_size_m=.0008, cad_sample_points=6000, xy_step_m=.001,
        min_xy_step_m=.00005, constrained_iterations=20,
    ))
    result = registrar.run(
        ASSET_DIR / "meshes" / f"{TARGET_MESH}_m.stl", candidate["pointcloud_path"],
        plane_model=localization["plane_model"],
        cad_metadata={"scale_to_m": 1., "cad_up_axis": "Z", "yaw_candidates_deg": [0.]},
        output_dir=output_dir,
    )
    if result["warnings"] or result["constrained_rmse_m"] > .002:
        raise RuntimeError(f"Peg CAD alignment rejected: {result['warnings']}, RMSE={result['constrained_rmse_m']}")
    world_T_cad = world_T_camera @ np.array(result["T_observed_from_cad"])
    center = world_T_cad @ np.r_[result["cad_center_m"], 1.]
    # The round peg's yaw is unobservable and irrelevant to this grasp baseline.
    grasp = np.eye(4)
    grasp[:3, :3] = np.diag([1., -1., -1.])
    grasp[:3, 3] = center[:3]
    grasp[2, 3] += .005
    return grasp, result


def grasp_and_insert(environment, observation, grasp, callback):
    """Open-loop peg-to-gripper assumption; robot proprioception closes OSC loop.

    No target object state is read during motion. There is no contact search or
    force-based correction, so a failed insertion is an expected possible result.
    """
    from .libero_control import move_eef_to_pose, hold_gripper, OPEN_GRIPPER, CLOSE_GRIPPER

    above = grasp.copy()
    above[2, 3] += .12
    observation = move_eef_to_pose(environment, observation, above, OPEN_GRIPPER, "approach_part", callback)
    observation = move_eef_to_pose(environment, observation, grasp, OPEN_GRIPPER, "descend_to_part", callback,
                                   tolerance_m=.001, max_steps=100)
    observation = hold_gripper(environment, observation, CLOSE_GRIPPER, "grasp", 25, callback)
    lift = grasp.copy()
    lift[2, 3] += .10
    observation = move_eef_to_pose(environment, observation, lift, CLOSE_GRIPPER, "lift", callback)
    above_hole = grasp.copy()
    above_hole[:3, 3] = WORLD_T_HOLE[:3, 3] + [0., 0., .11]
    observation = move_eef_to_pose(environment, observation, above_hole, CLOSE_GRIPPER, "approach_hole", callback,
                                   tolerance_m=.001, max_steps=100)
    # Nominal peg tip is 30 mm below the grip-site target (25 mm half-length
    # plus 5 mm grasp offset). Stop 5 mm below the supplied entrance plane.
    for index, tip_height in enumerate((.010, .004, .001, -.002, -.005)):
        insertion = above_hole.copy()
        insertion[2, 3] = WORLD_T_HOLE[2, 3] + .030 + tip_height
        observation = move_eef_to_pose(environment, observation, insertion, CLOSE_GRIPPER,
                                       f"insert_{index}", callback, tolerance_m=.001 if tip_height > 0 else .0002,
                                       orientation_tolerance_rad=.015, max_steps=60)
    return hold_gripper(environment, observation, CLOSE_GRIPPER, "hold_inserted", 10, callback)


def insertion_metrics(center_world_m, rotation_world, hole_pose=WORLD_T_HOLE):
    """Evaluation geometry only. Score shaft clearance over the engaged segment."""
    center = hole_pose[:3, :3].T @ (np.asarray(center_world_m) - hole_pose[:3, 3])
    axis = hole_pose[:3, :3].T @ np.asarray(rotation_world)[:, 2]
    if axis[2] < 0:
        axis = -axis
    tip = center - axis * PEG_LENGTH_M / 2
    tilt = float(np.degrees(np.arccos(np.clip(axis[2], -1, 1))))
    at_entrance = center - axis * center[2] / max(axis[2], 1e-9)
    radial = float(max(np.linalg.norm(at_entrance[:2]), np.linalg.norm(tip[:2])))
    depth = float(max(0., -tip[2])) if radial < HOLE_RADIUS_M else 0.
    # Conservative cylinder envelope; equality at zero clearance is not success.
    clearance = HOLE_RADIUS_M - PEG_RADIUS_M / max(axis[2], 1e-9) - radial
    inserted = bool(depth >= .005 and depth <= .020 and tilt < 1. and clearance > 0.)
    return {"insertion_depth_mm": depth * 1000, "radial_error_mm": radial * 1000,
            "tilt_deg": tilt, "estimated_clearance_mm": float(clearance * 1000),
            "inserted": inserted}


class PegEvaluator:
    """Reads simulator ground truth only for scores, never for policy inputs."""

    def __init__(self, environment):
        from .nist_task_board_1 import PARTS

        self.environment = environment
        sim = environment.sim
        self.body_ids = {name: sim.model.body_name2id(f"nist_part_{name}") for name in PARTS}
        self.initial_positions = {name: sim.data.body_xpos[index].copy() for name, index in self.body_ids.items()}
        self.target_id = self.body_ids[TARGET_MESH]
        self.collision_id = sim.model.geom_name2id(f"nist_part_{TARGET_MESH}_collision")
        gripper = environment.robots[0].gripper
        self.finger_ids = [set(sim.model.geom_name2id(name) for name in gripper.important_geoms[key])
                           for key in ("left_fingerpad", "right_fingerpad")]
        self.grasped = self.lifted = self.approached = False
        self.inserted_steps = 0
        self.success = False
        self.wrong_parts_lifted = set()

    def score(self, advance=True):
        sim = self.environment.sim
        touching = set()
        for contact in sim.data.contact[:sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.collision_id in pair:
                touching.update(pair - {self.collision_id})
        bilateral = all(touching & finger for finger in self.finger_ids)
        position = sim.data.body_xpos[self.target_id].copy()
        rotation = sim.data.body_xmat[self.target_id].reshape(3, 3)
        lifted_now = position[2] - self.initial_positions[TARGET_MESH][2] > .015
        self.grasped |= bilateral
        self.lifted |= bool(bilateral and lifted_now)
        self.approached |= bool(self.lifted and bilateral and np.linalg.norm(position[:2] - WORLD_T_HOLE[:2, 3]) < .010)
        metrics = insertion_metrics(position, rotation)
        if advance:
            self.inserted_steps = self.inserted_steps + 1 if metrics["inserted"] and self.lifted else 0
        self.success |= self.inserted_steps >= 5
        for name, index in self.body_ids.items():
            if name != TARGET_MESH and sim.data.body_xpos[index, 2] - self.initial_positions[name][2] > .015:
                self.wrong_parts_lifted.add(name)
        return {**metrics, "correct_part_grasped": bool(self.grasped),
                "correct_part_lifted": bool(self.lifted), "correct_hole_approached": bool(self.approached),
                "success": bool(self.success), "wrong_parts_lifted": sorted(self.wrong_parts_lifted),
                "evaluation_only_target_center_world_m": position.tolist()}
