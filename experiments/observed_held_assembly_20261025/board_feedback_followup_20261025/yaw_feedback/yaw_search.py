"""Fresh-observation CAD yaw bank; fits filtered depth and ranks raw depth."""
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ... import pose_estimation as local
from experiments.nist12_qwen_clip_reranking_20261023.depth_verification_20261023 import geometry
from scripts.surface_verification import CadSurface


CLIP_M = .002
DISTINCT_ROTATION_DEG = 2.
ALLOWED_STEPS_DEG = (6., 9.)


def protocol():
    return {"scope": "Fresh raw/known-static-filtered RGB-D CAD yaw search; no stored actual orientation or diagnostic pose reuse",
        "full_circle_offsets_deg": "Uniform [0,360) at the declared CAD tooth-hypothesis pitch",
        "declared_CAD_tooth_hypothesis_pitch_options_deg": ALLOWED_STEPS_DEG,
        "local_phase_offsets_pitch_multipliers": (0., -1., 1., -2., 2.),
        "local_fitter": "Unchanged observation-only _free_icp, each yaw seed its own bounded local prior",
        "local_fitter_protocol": local.protocol(),
        "foreground": "Filtered observations inside measured-motion-prior CAD bounds expanded by20mm; fixed0.5mm voxel means",
        "ranking": "Lowest equal-weight mean of observed-to-exact-CAD clipped squared surface distance and full projected-CAD-to-RAW-depth clipped squared error",
        "error_clip_m": CLIP_M, "distinct_solution_rotation_distance_deg": DISTINCT_ROTATION_DEG,
        "forward_occlusion": "Exclude only known robot/table/board depth closer than projected CAD plus1mm",
        "quality_gates": "Keep original >=30 points, >=50% points within1mm and >=50% assessable projected CAD depth within2mm",
        "ranking_gap_policy": "Record uncalibrated gaps; no sensor-noise-based confidence threshold. Stop for a numerical exact tie between distinct orientations.",
        "exact_tie_numerical_tolerance": "64*float64_epsilon*max(abs(best_loss),abs(next_distinct_loss),1e-12), in square metres",
        "RGB_used_for_fit": False, "training_or_weights": False, "unique_absolute_orientation_claim": False}


def _camera_map(observations):
    if isinstance(observations, dict):
        return observations
    result = {}
    for observation in observations:
        name = observation["calibration"]["camera_name"]
        if name in result:
            raise ValueError("Camera names must be unique")
        result[name] = observation
    return result


def yaw_offsets(mode, step_deg):
    if mode not in ("full_circle", "local_phase") or float(step_deg) not in ALLOWED_STEPS_DEG:
        raise ValueError("Use full_circle/local_phase with a declared6 or9degree CAD tooth hypothesis")
    return (np.arange(0., 360., step_deg) if mode == "full_circle" else
            np.array([0., -1., 1., -2., 2.])*step_deg)


def _forward_score(surface, pose, raw, filtered):
    records = []
    for name, camera in raw.items():
        calibration = camera["calibration"]
        depth = np.asarray(camera["depth_m"])
        render = surface.render(np.linalg.inv(np.asarray(calibration["world_T_camera"])) @ pose,
            np.asarray(calibration["intrinsics"]), depth.shape, pixel_offset=0.)
        if render is None:
            records.append({"camera_name": name, "assessable_CAD_pixels": 0,
                            "clipped_squared_residual_sum_m2": 0., "depth_agreeing_CAD_pixels": 0})
            continue
        x1, y1, x2, y2 = render["roi"]
        expected = render["depth"]
        known = np.asarray(filtered[name]["known_static_depth_m"])[y1:y2, x1:x2]
        measured = depth[y1:y2, x1:x2]
        cad = np.isfinite(expected) & (expected > 0)
        hidden = cad & np.isfinite(known) & (known < expected+.001)
        assessable = cad & ~hidden & np.isfinite(measured) & (measured > 0)
        residual = np.abs(measured[assessable]-expected[assessable])
        records.append({"camera_name": name, "projected_CAD_pixels": int(cad.sum()),
            "known_geometry_occluded_CAD_pixels": int(hidden.sum()),
            "assessable_CAD_pixels": len(residual), "depth_agreeing_CAD_pixels": int(np.sum(residual < .002)),
            "clipped_squared_residual_sum_m2": float(np.sum(np.minimum(residual, CLIP_M)**2)),
            "mean_absolute_depth_error_m": float(residual.mean()) if len(residual) else None,
            "image_clipped": bool(render["clipped"])})
    count = sum(row["assessable_CAD_pixels"] for row in records)
    agree = sum(row["depth_agreeing_CAD_pixels"] for row in records)
    loss = sum(row["clipped_squared_residual_sum_m2"] for row in records)/count if count else CLIP_M**2
    return loss, {"cameras": records, "assessable_CAD_pixels": count,
        "fraction_within_2mm": agree/count if count else 0.,
        "passed": count >= local.MIN_POINTS and agree/count >= local.HELD_MIN_FORWARD_DEPTH_AGREEMENT}


def estimate_yaw_pose(mesh_path, filtered_observations, raw_observations, predicted_world_T_cad,
                      yaw_search_mode, yaw_step_deg):
    """Return an ordinary pose estimate plus its complete ranked candidate bank."""
    started = time.perf_counter()
    offsets = yaw_offsets(yaw_search_mode, yaw_step_deg)
    prior = np.asarray(predicted_world_T_cad, float)
    if prior.shape != (4, 4) or not np.isfinite(prior).all():
        raise ValueError("Finite observation/FK-derived CAD prior required")
    raw, filtered = _camera_map(raw_observations), _camera_map(filtered_observations)
    if set(raw) != set(filtered):
        raise ValueError("Raw and filtered camera names must match")
    mesh = local._load_mesh(mesh_path)
    surface = CadSurface(mesh)
    lower = surface.vertices.min(0)-local.HELD_PRIOR_EXPANSION_M
    upper = surface.vertices.max(0)+local.HELD_PRIOR_EXPANSION_M
    clouds, camera_points = [], []
    for name, observation in filtered.items():
        if observation["calibration"] != raw[name]["calibration"]:
            raise ValueError("Raw and filtered camera calibration must be identical")
        if np.asarray(observation["known_static_depth_m"]).shape != np.asarray(raw[name]["depth_m"]).shape:
            raise ValueError("Known-static predicted depth must match raw image dimensions")
        depth, valid, robot_match = local._check_observation(observation)
        _, _, points = local._points(depth, valid, observation["calibration"])
        native = (points-prior[:3, 3]) @ prior[:3, :3]
        selected = points[np.all((native >= lower) & (native <= upper), axis=1)]
        if len(selected):
            clouds.append(geometry.voxel_means(selected, local.HELD_VOXEL_M))
        camera_points.append({"camera_name": name, "points_inside_motion_prior_volume": len(selected),
                              "robot_pixels_removed": int(robot_match.sum())})
    diagnostics = {"protocol": protocol(), "yaw_search_mode": yaw_search_mode, "yaw_step_deg": float(yaw_step_deg),
        "predicted_world_T_cad": prior.tolist(), "cameras": camera_points,
        "CAD_sha256": hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),
        "current_object_pose_or_simulator_segmentation_used": False,
        "absolute_symmetric_orientation_resolved": False, "ranking_confidence_calibrated": False}
    if not clouds or sum(map(len, clouds)) < local.MIN_POINTS:
        diagnostics["seconds"] = time.perf_counter()-started
        return {"status": "unavailable_insufficient_nonrobot_depth", "world_T_cad": None,
                "diagnostics": diagnostics, "ranked_candidates": []}
    points = geometry.voxel_means(np.concatenate(clouds), local.HELD_VOXEL_M)
    bank = []
    for index, offset in enumerate(offsets):
        initial = prior.copy()
        initial[:3, :3] = prior[:3, :3] @ Rotation.from_euler("z", offset, degrees=True).as_matrix()
        pose, history = local._free_icp(surface, points, initial, initial)
        _, _, distance, _ = local._closest(surface, points, pose)
        surface_loss = float(np.mean(np.minimum(distance, CLIP_M)**2))
        forward_loss, forward = _forward_score(surface, pose, raw, filtered)
        bank.append({"candidate_id": index, "initial_local_Z_offset_deg": float(offset),
            "world_T_cad": pose.tolist(), "point_surface_loss_m2": surface_loss,
            "forward_depth_loss_m2": forward_loss, "combined_loss_m2": (surface_loss+forward_loss)/2,
            "observed_point_inlier_fraction_within_1mm": float(np.mean(distance < .001)),
            "forward_depth_check": forward, "local_refinement_history": history})
    ranked = sorted(bank, key=lambda row: (row["combined_loss_m2"], row["candidate_id"]))
    distinct = []
    for candidate in ranked:
        rotation = np.asarray(candidate["world_T_cad"])[:3, :3]
        if all(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rotation @ np.asarray(
                other["world_T_cad"])[:3, :3].T).as_rotvec())) > DISTINCT_ROTATION_DEG for other in distinct):
            distinct.append(candidate)
    best = ranked[0]
    gap = distinct[1]["combined_loss_m2"]-best["combined_loss_m2"] if len(distinct)>1 else None
    tolerance = (64*np.finfo(float).eps*max(abs(best["combined_loss_m2"]),
                 abs(distinct[1]["combined_loss_m2"]), 1e-12)) if len(distinct)>1 else None
    exact_tie = bool(gap is not None and gap <= tolerance)
    pose = np.asarray(best["world_T_cad"])
    nearest, normals, distance, _ = local._closest(surface, points, pose)
    inliers = distance < .001
    matrix = np.column_stack((normals[inliers], np.cross(nearest[inliers]-pose[:3, 3], normals[inliers])/max(surface.radius, .001)))
    singular = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.zeros(6)
    diagnostics.update(observed_voxel_points=len(points), candidate_count=len(bank),
        selected_candidate_id=best["candidate_id"], ranked_candidate_ids=[row["candidate_id"] for row in ranked],
        distinct_yaw_ranked_candidate_ids=[row["candidate_id"] for row in distinct],
        distinct_solution_combined_loss_gap_m2=gap, numerical_exact_tie_tolerance_m2=tolerance,
        numerical_exact_tie_between_distinct_rotations=exact_tie,
        selected_combined_loss_m2=best["combined_loss_m2"],
        forward_depth_check=best["forward_depth_check"],
        local_observability_singular_values=singular.tolist(),
        local_six_parameter_support=bool(len(singular)==6 and singular[-1] > singular[0]*1e-4),
        translation_change_from_motion_prior_m=float(np.linalg.norm(pose[:3, 3]-prior[:3, 3])),
        rotation_change_from_motion_prior_deg=float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(pose[:3, :3] @ prior[:3, :3].T).as_rotvec()))),
        seconds=time.perf_counter()-started)
    status = ("unavailable_exact_yaw_tie" if exact_tie else "unavailable_poor_CAD_depth_agreement" if
        best["observed_point_inlier_fraction_within_1mm"] < local.HELD_MIN_1MM_INLIER_FRACTION or not best["forward_depth_check"]["passed"]
        else "available")
    return {"status": status, "world_T_cad": best["world_T_cad"] if status == "available" else None,
            "diagnostics": diagnostics, "ranked_candidates": ranked}
