"""Calibrated, image-only RGB-D input for the controlled pickup experiment.

The camera is fixed independently of every target's CAD pose. Only the supplied
target identity selects the rendered segmentation mask. Optical camera axes are
right/down/forward; metric depth is optical Z. Pixel centres have integer indices.
No physics or object-pose estimator is invoked by this module.
"""
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image


IMAGE_SIZE = 1024
CAMERA_NAME = "birdview"
CAMERA_POSITION_WORLD_M = (.20, 0., 1.30)
CAMERA_VERTICAL_FOV_DEG = 45.
TABLE_Z_M = 0.
# Render native CAD appearances and table, not convex contact approximations.
VISUAL_GEOM_GROUPS = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)


def unproject_pixels(row_col, optical_depth_m, calibration):
    """Map pixel centre(s) and optical Z into metric world coordinates."""
    pixels = np.asarray(row_col, dtype=float)
    single = pixels.ndim == 1
    pixels = np.atleast_2d(pixels)
    depth = np.broadcast_to(np.asarray(optical_depth_m, float), (len(pixels),))
    if pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
        raise ValueError("Expected finite row/column pixel centres")
    if not np.all(np.isfinite(depth) & (depth > 0)):
        raise ValueError("Optical depths must be finite positive metres")
    intrinsics = np.asarray(calibration["intrinsics"])
    transform = np.asarray(calibration["world_T_camera"])
    camera_points = np.column_stack(((pixels[:, 1]-intrinsics[0, 2])*depth/intrinsics[0, 0],
                                     (pixels[:, 0]-intrinsics[1, 2])*depth/intrinsics[1, 1], depth))
    world = camera_points @ transform[:3, :3].T + transform[:3, 3]
    return world[0] if single else world


def project_world(points_world_m, calibration):
    """Inverse of unproject_pixels, returning row/column and optical depth."""
    points = np.atleast_2d(np.asarray(points_world_m, float))
    transform = np.asarray(calibration["world_T_camera"])
    intrinsics = np.asarray(calibration["intrinsics"])
    camera = (points-transform[:3, 3]) @ transform[:3, :3]
    column = intrinsics[0, 0]*camera[:, 0]/camera[:, 2] + intrinsics[0, 2]
    row = intrinsics[1, 1]*camera[:, 1]/camera[:, 2] + intrinsics[1, 2]
    return np.column_stack((row, column)), camera[:, 2]


def _depth_audit(trial, depth_m, mask, calibration):
    """Compare renderer depths with independent MuJoCo rays at pixel centres."""
    groups = VISUAL_GEOM_GROUPS.copy()
    transform = np.asarray(calibration["world_T_camera"])
    intrinsics = np.asarray(calibration["intrinsics"])
    target_pixels = np.argwhere(mask)
    table_depth = float(transform[2, 3]-calibration["table_z_world_m"])
    background_pixels = np.argwhere(~mask & np.isfinite(depth_m) & (abs(depth_m-table_depth) < .002))
    selected = []
    for available in (target_pixels, background_pixels):
        if len(available):
            selected.extend(available[np.linspace(0, len(available)-1, min(24, len(available)), dtype=int)])
    records = []
    for row, col in selected:
        optical_ray = np.array([(col-intrinsics[0, 2])/intrinsics[0, 0],
                                (row-intrinsics[1, 2])/intrinsics[1, 1], 1.])
        magnitude = np.linalg.norm(optical_ray)
        direction = transform[:3, :3] @ (optical_ray/magnitude)
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(trial.m, trial.d, transform[:3, 3], direction,
                                 groups, 1, -1, geom)
        if distance < 0:
            raise AssertionError(f"Depth pixel ray missed at {row},{col}")
        ray_z = float(distance/magnitude)
        observed = float(depth_m[row, col])
        records.append({"row": int(row), "col": int(col),
                        "on_target": bool(mask[row, col]), "ray_geom_id": int(geom[0]),
                        "render_depth_m": observed, "ray_depth_m": ray_z,
                        "absolute_difference_m": abs(observed-ray_z)})
    maximum = max(record["absolute_difference_m"] for record in records)
    # Rendered raster edges can disagree with a geometric ray on silhouette
    # boundaries. Record every difference, without changing input depth values.
    pixels = np.asarray([[record["row"], record["col"]] for record in records])
    depths = np.asarray([record["render_depth_m"] for record in records])
    points = unproject_pixels(pixels, depths, calibration)
    reprojected, recovered_depth = project_world(points, calibration)
    return {"sampled_pixel_count": len(records), "samples": records,
            "maximum_ray_depth_difference_m": maximum,
            "maximum_roundtrip_pixel_error": float(np.max(np.abs(reprojected-pixels))),
            "maximum_roundtrip_depth_error_m": float(np.max(abs(recovered_depth-depths)))}


def capture_scene(trial, output_dir=None):
    """Return RGB, optical-Z depth, perfect visible target mask and calibration.

    The known simulated camera calibration is explicit experiment input; no
    target position/orientation is used. Caller restores visibility of other
    loose parts before capture, leaving their contacts disabled as requested.
    """
    started = time.perf_counter()
    original_qpos = trial.d.qpos.copy()
    original_time = float(trial.d.time)
    camera_id = trial.m.camera(CAMERA_NAME).id
    if int(trial.m.cam_bodyid[camera_id]) != 0:
        raise ValueError("Capture camera must be fixed to world")
    trial.m.cam_pos[camera_id] = CAMERA_POSITION_WORLD_M
    trial.m.cam_quat[camera_id] = [1., 0., 0., 0.]
    trial.m.cam_fovy[camera_id] = CAMERA_VERTICAL_FOV_DEG
    trial.m.vis.global_.offwidth = max(IMAGE_SIZE, trial.m.vis.global_.offwidth)
    trial.m.vis.global_.offheight = max(IMAGE_SIZE, trial.m.vis.global_.offheight)
    # Integer segmentation IDs must not be blended at raster edges. This also
    # keeps RGB/depth/ID sampling aligned at the same single pixel centre.
    trial.m.vis.quality.offsamples = 0
    mujoco.mj_forward(trial.m, trial.d)
    option = mujoco.MjvOption()
    option.geomgroup[:] = VISUAL_GEOM_GROUPS
    option.sitegroup[:] = 0
    with mujoco.Renderer(trial.m, height=IMAGE_SIZE, width=IMAGE_SIZE) as renderer:
        renderer.update_scene(trial.d, camera=camera_id, scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(trial.d, camera=camera_id, scene_option=option)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(trial.d, camera=camera_id, scene_option=option)
        segmentation = renderer.render().copy()
    geom_ids = segmentation[:, :, 0]
    valid_geom = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_ids >= 0)
    mask = np.zeros_like(geom_ids, dtype=bool)
    mask[valid_geom] = trial.m.geom_bodyid[geom_ids[valid_geom]] == trial.target_bid
    if not np.any(mask):
        raise ValueError("Target has no visible pixels in the fixed camera")
    focal = IMAGE_SIZE/(2*np.tan(np.deg2rad(CAMERA_VERTICAL_FOV_DEG/2)))
    intrinsics = np.array([[focal, 0., (IMAGE_SIZE-1)/2],
                           [0., focal, (IMAGE_SIZE-1)/2], [0., 0., 1.]])
    world_T_camera = np.eye(4)
    world_T_camera[:3, :3] = trial.d.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
    world_T_camera[:3, 3] = trial.d.cam_xpos[camera_id]
    calibration = {"camera_name": CAMERA_NAME, "width": IMAGE_SIZE, "height": IMAGE_SIZE,
                   "intrinsics": intrinsics.tolist(), "world_T_camera": world_T_camera.tolist(),
                   "camera_axes": "OpenCV: +X right, +Y down, +Z forward",
                   "world_axes": "Existing objects_centered simulation world; +Z up",
                   "depth_units": "metres; optical camera Z, not ray range",
                   "pixel_convention": "Integer row/column coordinates are pixel centres; principal point=(size-1)/2",
                   "camera_vertical_fov_deg": CAMERA_VERTICAL_FOV_DEG,
                   "offscreen_msaa_samples": 0,
                   "table_z_world_m": TABLE_Z_M, "camera_calibration_source": "Known simulator camera parameters",
                   "object_pose_used_for_camera_or_grasp_generation": False,
                   "target_mask_source": "Perfect visible renderer geometry segmentation by supplied target identity",
                   "rendered_geom_groups": VISUAL_GEOM_GROUPS.tolist(),
                   "rendered_appearance": "Native CAD visual meshes and colors; convex contact approximations excluded"}
    audit = _depth_audit(trial, depth, mask, calibration)
    np.testing.assert_array_equal(trial.d.qpos, original_qpos)
    assert float(trial.d.time) == original_time
    record = {"capture_seconds": time.perf_counter()-started, "physics_steps": 0,
              "qpos_unchanged": True, "target_pixel_count": int(mask.sum()),
              "target_touches_frame_border": bool(np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]])),
              "target_depth_range_m": [float(depth[mask].min()), float(depth[mask].max())],
              "camera_ray_validation": audit}
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(folder / "rgb.png")
        Image.fromarray(mask.astype(np.uint8)*255).save(folder / "target_mask.png")
        np.save(folder / "depth_m.npy", depth)
        np.save(folder / "target_mask.npy", mask)
        (folder / "calibration.json").write_text(json.dumps(calibration, indent=2)+"\n")
        (folder / "capture_audit.json").write_text(json.dumps(record, indent=2)+"\n")
    return {"rgb": rgb, "depth_m": depth, "target_mask": mask,
            "calibration": calibration, "audit": record}


def rectangle_geometry(rectangle, depth_m, target_mask, calibration):
    """Convert an image grasp into top-down world geometry without CAD pose.

    Use centre depth if the centre sees the target. Otherwise use median target
    depth within five pixels, then the nearest target pixel if necessary. A grasp
    centre may lie over a part's hole; its XY remains the predicted image point.
    The estimated mid-height is halfway from the known table to observed surface.
    """
    center = np.asarray(rectangle["center_rc"], float)
    axis = np.asarray(rectangle["jaw_axis_rc_unit"], float)
    width_pixels = float(rectangle["width_pixels"])
    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-5) or width_pixels <= 0:
        raise ValueError("Expected a unit image jaw axis and positive opening")
    row, col = np.rint(center).astype(int)
    if not (0 <= row < depth_m.shape[0] and 0 <= col < depth_m.shape[1]):
        raise ValueError("Predicted grasp centre lies outside captured image")
    sample_pixels = None
    center_valid = bool(target_mask[row, col] and np.isfinite(depth_m[row, col]) and depth_m[row, col] > 0)
    if center_valid:
        measured_depth = float(depth_m[row, col])
        depth_method = "target_visible_at_rounded_center_pixel"
        sample_pixels = np.array([[row, col]])
    else:
        target_pixels = np.argwhere(target_mask)
        if not len(target_pixels):
            raise ValueError("No target depth near predicted grasp centre")
        distances = np.linalg.norm(target_pixels-center, axis=1)
        local_pixels = target_pixels[distances <= 5.]
        if len(local_pixels):
            sample_pixels = local_pixels
            depth_method = "median_visible_target_depth_within_five_pixels"
        else:
            sample_pixels = target_pixels[[int(np.argmin(distances))]]
            depth_method = "nearest_visible_target_pixel_depth_without_changing_predicted_xy"
        values = depth_m[sample_pixels[:, 0], sample_pixels[:, 1]]
        valid = np.isfinite(values) & (values > 0)
        if not np.any(valid):
            raise ValueError("No finite target depth near predicted grasp centre")
        measured_depth = float(np.median(values[valid]))
        sample_pixels = sample_pixels[valid]
    endpoints_rc = np.stack([center-axis*width_pixels/2, center+axis*width_pixels/2])
    surface = unproject_pixels(center, measured_depth, calibration)
    endpoints = unproject_pixels(endpoints_rc, measured_depth, calibration)
    jaw_axis = endpoints[1]-endpoints[0]
    predicted_width = float(np.linalg.norm(jaw_axis))
    jaw_axis /= predicted_width
    table_z = float(calibration["table_z_world_m"])
    estimated_height = float(surface[2]-table_z)
    if not estimated_height > 0:
        raise ValueError("Observed target surface is not above the known table")
    middle = surface.copy()
    middle[2] = table_z + estimated_height/2
    return {"surface_center_world_m": surface.tolist(), "center_world_m": middle.tolist(),
            "jaw_axis_world_unit": jaw_axis.tolist(), "jaw_endpoints_world_m": endpoints.tolist(),
            "jaw_endpoints_rc": endpoints_rc.tolist(), "predicted_width_m": predicted_width,
            "estimated_height_m": estimated_height, "optical_depth_used_m": measured_depth,
            "mask_center_valid": center_valid,
            "maximum_depth_sample_distance_pixels": float(np.max(np.linalg.norm(sample_pixels-center, axis=1))),
            "depth_method": depth_method, "depth_sample_pixels_rc": sample_pixels.tolist(),
            "center_height_policy": "Half observed target surface height above known horizontal table",
            "object_pose_used": False}
