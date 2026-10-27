"""Observation-only native-metre CAD alignment before and after pickup.

Inputs are camera arrays/calibration and a robot-motion prior. No simulator,
target identity mask, current target pose, scene XML or execution interface is
imported. Held fitting changes all six local pose parameters; repeated geometry
can still make an absolute orientation ambiguous.
"""
import hashlib
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

from experiments.nist12_qwen_clip_reranking_20261023.depth_verification_20261023 import geometry
from scripts.surface_verification import CadSurface


ROBOT_DEPTH_TOLERANCE_M = .001
HELD_PRIOR_EXPANSION_M = .020
HELD_VOXEL_M = .0005
HELD_YAW_STARTS_DEG = (0., -5., 5., -10., 10.)
HELD_GATES_M = (.012, .006, .003)
HELD_ITERATIONS_PER_GATE = 15
HELD_MAX_TRANSLATION_M = .020
HELD_MAX_ROTATION_DEG = 20.
MIN_POINTS = 30
TABLE_MIN_SCORE = .55
TABLE_AMBIGUITY_GAP = .02
HELD_MIN_1MM_INLIER_FRACTION = .5
HELD_MIN_FORWARD_DEPTH_AGREEMENT = .5


def protocol():
    return {"units": "native CAD metres; world_T_cad maps CAD into calibrated world",
        "table_detection": "Depth above known calibrated table, 8-connected image components; no target identity mask or position seed",
        "table_height_band_m": [.0005, .15], "minimum_component_pixels": 12,
        "component_diameter_ratio_to_CAD_diagonal_band": [.3, 2.],
        "table_minimum_geometry_score": TABLE_MIN_SCORE,
        "table_ambiguous_if_distinct_component_score_gap_below": TABLE_AMBIGUITY_GAP,
        "table_orientation_prior": "Native CAD upright branch: support faces nearest the positive and negative shortest CAD bounding-box axis",
        "table_yaw_starts_deg": list(geometry.YAW_DEGREES),
        "table_fit": "Existing native-scale table-constrained ICP and full projected/observed silhouette/depth score",
        "robot_depth_match_tolerance_m": ROBOT_DEPTH_TOLERANCE_M,
        "held_foreground_prior_expansion_m": HELD_PRIOR_EXPANSION_M,
        "held_voxel_m": HELD_VOXEL_M, "held_yaw_starts_deg": HELD_YAW_STARTS_DEG,
        "held_correspondence_gates_m": HELD_GATES_M,
        "held_iterations_per_gate": HELD_ITERATIONS_PER_GATE,
        "held_max_translation_from_motion_prior_m": HELD_MAX_TRANSLATION_M,
        "held_max_rotation_from_motion_prior_deg": HELD_MAX_ROTATION_DEG,
        "held_Huber_point_to_plane_scale_m": .0015,
        "held_maximum_translation_step_m": .003, "held_maximum_rotation_step_deg": 3.,
        "held_line_search_step_fractions": [1., .5, .25, .125],
        "held_stop_translation_step_m": 1e-7, "held_stop_rotation_step_rad": 1e-5,
        "held_candidate_ranking_distance_clip_m": .003,
        "held_minimum_observed_points": MIN_POINTS,
        "held_minimum_observed_inlier_fraction_within_1mm": HELD_MIN_1MM_INLIER_FRACTION,
        "held_forward_depth_agreement_tolerance_m": .002,
        "held_minimum_forward_depth_agreement_fraction": HELD_MIN_FORWARD_DEPTH_AGREEMENT,
        "held_local_observability_singular_value_ratio": 1e-4,
        "held_fit": "Robust observed-to-exact-CAD point-to-plane ICP; free XYZ and local rotation, no table constraint",
        "symmetry": "Local observable alignment only; globally repeated CAD orientations are not uniquely identified",
        "training": False, "RGB_used_for_registration": False,
        "simulator_target_pose_or_segmentation_used": False}


def _load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or not len(mesh.triangles):
        raise ValueError("Readable native-metre CAD triangle mesh required")
    return mesh


def _check_observation(observation):
    forbidden = {"target_mask", "segmentation", "target_pose", "target_bbox", "world_T_target"}
    if forbidden.intersection(observation):
        raise ValueError("Sensor input must not contain simulator target masks, boxes or poses")
    depth = np.asarray(observation["depth_m"], float)
    robot = np.asarray(observation["robot_depth_m"], float)
    if depth.ndim != 2 or robot.shape != depth.shape:
        raise ValueError("Depth and robot-only predicted depth must be equal-size H by W arrays")
    valid = np.isfinite(depth) & (depth > 0)
    robot_match = np.isfinite(robot) & (robot > 0) & (np.abs(depth-robot) <= ROBOT_DEPTH_TOLERANCE_M)
    return depth, valid & ~robot_match, robot_match


def _points(depth, mask, calibration):
    pixels = np.argwhere(mask)
    K = np.asarray(calibration["intrinsics"], float)
    z = depth[mask]
    camera = np.column_stack(((pixels[:, 1]-K[0, 2])*z/K[0, 0],
                              (pixels[:, 0]-K[1, 2])*z/K[1, 1], z))
    world_T_camera = np.asarray(calibration["world_T_camera"], float)
    world = camera @ world_T_camera[:3, :3].T+world_T_camera[:3, 3]
    return pixels, camera, world


def predict_held_pose(world_T_cad_before_pickup, world_T_grip_at_pickup, world_T_grip_now):
    """Transport an observed initial pose using measured robot kinematics only."""
    return (np.asarray(world_T_grip_now) @ np.linalg.inv(world_T_grip_at_pickup)
            @ np.asarray(world_T_cad_before_pickup))


def estimate_table_pose(mesh_path, overhead_observation):
    """Detect supplied CAD anywhere in the raw tabletop depth image."""
    started = time.perf_counter()
    depth, valid, robot_match = _check_observation(overhead_observation)
    calibration = overhead_observation["calibration"]
    pixels, camera, world = _points(depth, valid, calibration)
    table_z = float(calibration["table_z_world_m"])
    above = (world[:, 2] > table_z+.0005) & (world[:, 2] < table_z+.15)
    foreground = np.zeros(depth.shape, bool)
    foreground[tuple(pixels[above].T)] = True
    labels, component_count = label(foreground, np.ones((3, 3)))
    mesh = _load_mesh(mesh_path)
    cad = geometry.prepare_cad(mesh)
    shortest_axis = int(np.argmin(np.ptp(cad.vertices, axis=0)))
    directions = [np.eye(3)[shortest_axis]*sign for sign in (1., -1.)]
    supports = []
    for direction in directions:
        face = max(cad.supports, key=lambda row: np.asarray(row["outward_normal"]) @ direction)
        if face not in supports:
            supports.append(face)
    cad.supports = supports
    world_T_camera = np.asarray(calibration["world_T_camera"], float)
    world_plane = np.array([0., 0., 1., -table_z])
    camera_plane = world_T_camera.T @ world_plane
    K = np.asarray(calibration["intrinsics"], float)
    cad_diameter = float(np.linalg.norm(np.ptp(cad.vertices, axis=0)))
    candidates, rejected = [], []
    for component_id in range(1, component_count+1):
        mask = labels == component_id
        count = int(mask.sum())
        if count < 12:
            continue
        _, observed, component_world = _points(depth, mask, calibration)
        diameter = float(np.linalg.norm(np.ptp(component_world, axis=0)))
        if not .3*cad_diameter <= diameter <= 2.*cad_diameter:
            rejected.append({"component_id": component_id, "pixels": count,
                             "observed_diameter_m": diameter, "reason": "Outside generic native-CAD size band"})
            continue
        fit = geometry.fit_pair(cad, observed, camera_plane, K, depth, mask, clearances=(0.,))
        arm = fit["arms"][0]
        best = arm["best_geometry"]
        if best is None:
            rejected.append({"component_id": component_id, "pixels": count,
                             "reason": "All native-CAD projections unavailable", "pose_hypotheses": arm["pose_bank"]})
            continue
        pose = world_T_camera @ np.asarray(best["T_camera_from_cad"])
        candidates.append({"component_id": component_id, "pixels": count,
            "world_T_cad": pose.tolist(), "score": best["verification"]["geometry_score"],
            "best_fit": best, "hypothesis_count": arm["hypothesis_count"],
            "pose_hypotheses": arm["pose_bank"],
            "observed_diameter_m": diameter})
    candidates.sort(key=lambda row: (-row["score"], row["component_id"]))
    diagnostics = {"protocol": protocol(), "components_above_plane": component_count,
        "robot_pixels_removed": int(robot_match.sum()), "candidate_components": candidates,
        "size_rejections": rejected, "CAD_native_extent_m": np.ptp(cad.vertices, axis=0).tolist(),
        "CAD_support_faces_tested": supports, "seconds": time.perf_counter()-started,
        "prior_object_pose_used": False, "target_mask_source": "Connected component of observed depth above calibrated plane",
        "absolute_symmetric_orientation_resolved": False}
    if not candidates or candidates[0]["score"] < TABLE_MIN_SCORE:
        return {"status": "unavailable_no_CAD_depth_match", "world_T_cad": None,
                "observed_mask": np.zeros(depth.shape, bool), "diagnostics": diagnostics}
    if len(candidates) > 1 and candidates[1]["score"] >= TABLE_MIN_SCORE and candidates[0]["score"]-candidates[1]["score"] < TABLE_AMBIGUITY_GAP:
        return {"status": "unavailable_ambiguous_CAD_components", "world_T_cad": None,
                "observed_mask": np.zeros(depth.shape, bool), "diagnostics": diagnostics}
    best = candidates[0]
    diagnostics["selected_component_id"] = best["component_id"]
    return {"status": "available", "world_T_cad": best["world_T_cad"],
            "observed_mask": labels == best["component_id"], "diagnostics": diagnostics}


def _closest(surface, points_world, pose):
    local = (points_world-pose[:3, 3]) @ pose[:3, :3]
    query = surface.scene.compute_closest_points(o3d.core.Tensor(local.astype(np.float32)))
    nearest_local = query["points"].numpy().astype(float)
    normals_world = query["primitive_normals"].numpy().astype(float) @ pose[:3, :3].T
    nearest_world = nearest_local @ pose[:3, :3].T+pose[:3, 3]
    difference = points_world-nearest_world
    distance = np.linalg.norm(difference, axis=1)
    residual = np.einsum("ij,ij->i", normals_world, difference)
    return nearest_world, normals_world, distance, residual


def _loss(surface, points, pose):
    distance = _closest(surface, points, pose)[2]
    return float(np.mean(np.minimum(distance, .003)**2))


def _free_icp(surface, points, initial, motion_prior):
    pose = initial.copy()
    history = []
    for gate in HELD_GATES_M:
        for iteration in range(HELD_ITERATIONS_PER_GATE):
            nearest, normals, distance, residual = _closest(surface, points, pose)
            use = distance < gate
            if int(use.sum()) < MIN_POINTS:
                break
            residual = residual[use]
            jacobian = np.column_stack((normals[use], np.cross(nearest[use]-pose[:3, 3], normals[use])))
            weight = np.minimum(1., .0015/(np.abs(residual)+1e-12))
            weighted = jacobian*np.sqrt(weight[:, None])
            delta = np.linalg.lstsq(weighted, residual*np.sqrt(weight), rcond=1e-6)[0]
            delta[:3] *= min(1., .003/(np.linalg.norm(delta[:3])+1e-12))
            delta[3:] *= min(1., np.deg2rad(3)/(np.linalg.norm(delta[3:])+1e-12))
            previous_loss = _loss(surface, points, pose)
            accepted = False
            for step in (1., .5, .25, .125):
                candidate = pose.copy()
                candidate[:3, :3] = Rotation.from_rotvec(delta[3:]*step).as_matrix() @ pose[:3, :3]
                candidate[:3, 3] += delta[:3]*step
                shift = np.linalg.norm(candidate[:3, 3]-motion_prior[:3, 3])
                angle = np.rad2deg(np.linalg.norm(Rotation.from_matrix(
                    candidate[:3, :3] @ motion_prior[:3, :3].T).as_rotvec()))
                if shift > HELD_MAX_TRANSLATION_M or angle > HELD_MAX_ROTATION_DEG:
                    continue
                loss = _loss(surface, points, candidate)
                if loss <= previous_loss+1e-14:
                    pose, accepted = candidate, True
                    history.append({"gate_m": gate, "iteration": iteration, "loss_m2": loss,
                                    "inlier_points": int(use.sum()), "step_fraction": step})
                    break
            if not accepted or np.linalg.norm(delta[:3]) < 1e-7 and np.linalg.norm(delta[3:]) < 1e-5:
                break
    return pose, history


def _forward_depth_check(surface, pose, observations):
    records = []
    for observation in observations:
        calibration = observation["calibration"]
        camera_T_cad = np.linalg.inv(np.asarray(calibration["world_T_camera"])) @ pose
        depth = np.asarray(observation["depth_m"])
        render = surface.render(camera_T_cad, np.asarray(calibration["intrinsics"]), depth.shape, pixel_offset=0.)
        if render is None:
            records.append({"assessable_CAD_pixels": 0, "depth_agreeing_CAD_pixels": 0})
            continue
        x1, y1, x2, y2 = render["roi"]
        expected = render["depth"]
        measured = depth[y1:y2, x1:x2]
        robot = np.asarray(observation["robot_depth_m"])[y1:y2, x1:x2]
        valid_cad = np.isfinite(expected) & (expected > 0)
        robot_occluded = valid_cad & np.isfinite(robot) & (robot < expected+ROBOT_DEPTH_TOLERANCE_M)
        assessable = valid_cad & ~robot_occluded & np.isfinite(measured) & (measured > 0)
        residual = np.abs(measured[assessable]-expected[assessable])
        records.append({"camera_name": calibration.get("camera_name"),
            "assessable_CAD_pixels": int(assessable.sum()),
            "robot_occluded_CAD_pixels": int(robot_occluded.sum()),
            "depth_agreeing_CAD_pixels": int(np.sum(residual < .002)),
            "depth_MAE_m": float(residual.mean()) if len(residual) else None,
            "image_clipped": bool(render["clipped"])})
    total = sum(row["assessable_CAD_pixels"] for row in records)
    agreeing = sum(row["depth_agreeing_CAD_pixels"] for row in records)
    return {"cameras": records, "assessable_CAD_pixels": total,
            "fraction_within_2mm": agreeing/total if total else 0.,
            "passed": total >= MIN_POINTS and agreeing/total >= HELD_MIN_FORWARD_DEPTH_AGREEMENT}


def estimate_held_pose(mesh_path, observations, predicted_world_T_cad):
    """Fit a held CAD to raw multi-view depth after removing known robot pixels."""
    started = time.perf_counter()
    mesh = _load_mesh(mesh_path)
    surface = CadSurface(mesh)
    prior = np.asarray(predicted_world_T_cad, float)
    if prior.shape != (4, 4) or not np.isfinite(prior).all():
        raise ValueError("Finite measured-motion CAD pose prior required")
    if isinstance(observations, dict):
        observations = list(observations.values())
    clouds, camera_records = [], []
    vertices = np.asarray(mesh.vertices)
    lower, upper = vertices.min(0)-HELD_PRIOR_EXPANSION_M, vertices.max(0)+HELD_PRIOR_EXPANSION_M
    for observation in observations:
        depth, valid, robot_match = _check_observation(observation)
        _, _, points = _points(depth, valid, observation["calibration"])
        local = (points-prior[:3, 3]) @ prior[:3, :3]
        keep = np.all((local >= lower) & (local <= upper), axis=1)
        selected = points[keep]
        if len(selected):
            clouds.append(geometry.voxel_means(selected, HELD_VOXEL_M))
        camera_records.append({"camera_name": observation["calibration"].get("camera_name"),
            "valid_nonrobot_depth_pixels": int(valid.sum()), "robot_pixels_removed": int(robot_match.sum()),
            "points_inside_motion_prior_volume": len(selected)})
    diagnostics = {"protocol": protocol(), "cameras": camera_records,
        "predicted_world_T_cad": prior.tolist(), "absolute_symmetric_orientation_resolved": False,
        "current_object_pose_or_simulator_segmentation_used": False,
        "CAD_sha256": hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest()}
    if not clouds or sum(map(len, clouds)) < MIN_POINTS:
        return {"status": "unavailable_insufficient_nonrobot_depth", "world_T_cad": None, "diagnostics": diagnostics}
    points = geometry.voxel_means(np.concatenate(clouds), HELD_VOXEL_M)
    bank = []
    for yaw in HELD_YAW_STARTS_DEG:
        initial = prior.copy()
        initial[:3, :3] = prior[:3, :3] @ Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        pose, history = _free_icp(surface, points, initial, prior)
        _, _, distance, _ = _closest(surface, points, pose)
        bank.append({"initial_local_yaw_offset_deg": yaw, "world_T_cad": pose.tolist(),
            "clipped_surface_loss_m2": _loss(surface, points, pose),
            "inlier_fraction_within_1mm": float(np.mean(distance < .001)),
            "inlier_rmse_within_3mm_m": float(np.sqrt(np.mean(distance[distance < .003]**2)))
                if np.any(distance < .003) else None,
            "iterations": history})
    bank.sort(key=lambda row: (row["clipped_surface_loss_m2"], abs(row["initial_local_yaw_offset_deg"])))
    best = bank[0]
    pose = np.asarray(best["world_T_cad"])
    nearest, normals, distance, _ = _closest(surface, points, pose)
    inliers = distance < .001
    radius = max(float(surface.radius), .001)
    matrix = np.column_stack((normals[inliers], np.cross(nearest[inliers]-pose[:3, 3], normals[inliers])/radius))
    singular_values = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.zeros(6)
    forward_check = _forward_depth_check(surface, pose, observations)
    diagnostics.update(observed_voxel_points=len(points), local_pose_candidates=bank,
        local_observability_singular_values=singular_values.tolist(),
        local_six_parameter_support=bool(len(singular_values) == 6 and singular_values[-1] > singular_values[0]*1e-4),
        forward_depth_check=forward_check,
        translation_change_from_motion_prior_m=float(np.linalg.norm(pose[:3, 3]-prior[:3, 3])),
        rotation_change_from_motion_prior_deg=float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(pose[:3, :3] @ prior[:3, :3].T).as_rotvec()))),
        seconds=time.perf_counter()-started)
    if best["inlier_fraction_within_1mm"] < HELD_MIN_1MM_INLIER_FRACTION or not forward_check["passed"]:
        return {"status": "unavailable_poor_CAD_depth_agreement", "world_T_cad": None, "diagnostics": diagnostics}
    return {"status": "available", "world_T_cad": best["world_T_cad"], "diagnostics": diagnostics}
