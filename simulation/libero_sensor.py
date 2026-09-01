from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .libero_env import LiberoIntegrationError


@dataclass
class LiberoRGBDObservation:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    world_T_camera: np.ndarray
    camera_T_world: np.ndarray
    instruction: str
    robot_state: dict
    raw_observation: Mapping
    camera_name: str
    rgb_key: str
    depth_key: str
    orientation_correction: str


class LiberoRGBDSensor:
    """Expose a LIBERO camera as a conventional calibrated RGB-D sensor.

    LIBERO's raw camera arrays use the OpenGL bottom-left image origin. This
    class flips rows exactly once for a conventional top-left image origin.
    The same correction is applied to RGB and depth; columns are not flipped.
    """

    def __init__(self, environment, camera_name="agentview", camera_utils=None):
        self.environment = environment
        self.camera_name = str(camera_name)
        if camera_utils is None:
            try:
                from robosuite.utils import camera_utils as robosuite_camera_utils
            except ModuleNotFoundError as error:
                raise LiberoIntegrationError(
                    "robosuite camera utilities are unavailable. Install LeRobot's "
                    "LIBERO extra before capturing RGB-D."
                ) from error
            camera_utils = robosuite_camera_utils
        self.camera_utils = camera_utils

    def capture(self, raw_observation=None):
        raw = raw_observation
        if raw is None:
            raw = self.environment.last_observation
        if raw is None:
            raw = self.environment.reset()

        rgb_key, raw_rgb = _find_camera_observation(raw, self.camera_name, "_image")
        depth_key, normalized_depth = _find_camera_observation(raw, self.camera_name, "_depth")

        raw_rgb = np.asarray(raw_rgb)
        normalized_depth = _single_channel(np.asarray(normalized_depth), "normalized depth")
        if raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
            raise LiberoIntegrationError(
                f"Expected HxWx3 RGB at {rgb_key!r}; received shape {raw_rgb.shape}."
            )
        if raw_rgb.shape[:2] != normalized_depth.shape:
            raise LiberoIntegrationError(
                f"RGB shape {raw_rgb.shape[:2]} and depth shape {normalized_depth.shape} do not match."
            )
        if not np.issubdtype(raw_rgb.dtype, np.integer):
            raise LiberoIntegrationError(
                f"Expected integer RGB data at {rgb_key!r}; received dtype {raw_rgb.dtype}."
            )
        if not np.all(np.isfinite(normalized_depth)):
            raise LiberoIntegrationError("Normalized MuJoCo depth contains NaN or infinity.")
        minimum = float(normalized_depth.min())
        maximum = float(normalized_depth.max())
        if minimum < 0.0 or maximum > 1.0:
            raise LiberoIntegrationError(
                "MuJoCo depth must be normalized to [0, 1] before metric conversion; "
                f"received range [{minimum:.6f}, {maximum:.6f}]."
            )

        try:
            metric_depth = self.camera_utils.get_real_depth_map(
                self.environment.sim,
                normalized_depth,
            )
            height, width = normalized_depth.shape
            intrinsics = self.camera_utils.get_camera_intrinsic_matrix(
                self.environment.sim,
                self.camera_name,
                height,
                width,
            )
            world_T_camera = self.camera_utils.get_camera_extrinsic_matrix(
                self.environment.sim,
                self.camera_name,
            )
        except Exception as error:
            raise LiberoIntegrationError(
                f"Failed to convert depth or retrieve calibration for camera "
                f"{self.camera_name!r}: {error}"
            ) from error

        metric_depth = _single_channel(np.asarray(metric_depth, dtype=np.float32), "metric depth")
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        world_T_camera = np.asarray(world_T_camera, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise LiberoIntegrationError(f"Expected a 3x3 intrinsic matrix; received {intrinsics.shape}.")
        if world_T_camera.shape != (4, 4):
            raise LiberoIntegrationError(
                f"Expected a 4x4 camera pose matrix; received {world_T_camera.shape}."
            )
        if not np.all(np.isfinite(metric_depth)):
            raise LiberoIntegrationError("Metric depth contains NaN or infinity after conversion.")
        if not np.any(metric_depth > 0.0):
            raise LiberoIntegrationError("Metric depth contains no positive distances.")

        camera_T_world = np.linalg.inv(world_T_camera)
        identity_error = np.max(np.abs(world_T_camera @ camera_T_world - np.eye(4)))
        if identity_error > 1e-8:
            raise LiberoIntegrationError(
                f"Camera transform inversion check failed; max identity error is {identity_error:.3e}."
            )

        rgb = np.ascontiguousarray(np.flip(raw_rgb, axis=0), dtype=np.uint8)
        depth_m = np.ascontiguousarray(np.flip(metric_depth, axis=0), dtype=np.float32)

        return LiberoRGBDObservation(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            world_T_camera=world_T_camera,
            camera_T_world=camera_T_world,
            instruction=self.environment.instruction,
            robot_state=_extract_robot_state(raw),
            raw_observation=raw,
            camera_name=self.camera_name,
            rgb_key=rgb_key,
            depth_key=depth_key,
            orientation_correction="vertical_flip_once_from_opengl_bottom_left_to_image_top_left",
        )


def depth_statistics(depth_m):
    depth = np.asarray(depth_m)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        raise LiberoIntegrationError("Metric depth has no finite positive pixels.")
    return {
        "valid_pixels": int(valid.size),
        "minimum_m": float(valid.min()),
        "maximum_m": float(valid.max()),
        "median_m": float(np.median(valid)),
    }


def _single_channel(array, label):
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise LiberoIntegrationError(f"Expected HxW {label}; received shape {array.shape}.")
    return array


def _find_camera_observation(observation, camera_name, suffix):
    leaves = dict(_flatten_mapping(observation))
    exact = f"{camera_name}{suffix}"
    if exact in leaves:
        return exact, leaves[exact]

    exact_matches = [
        key for key in leaves if key.rsplit(".", 1)[-1] == exact
    ]
    if len(exact_matches) == 1:
        key = exact_matches[0]
        return key, leaves[key]

    camera_matches = [
        key for key in leaves if camera_name in key and key.endswith(suffix)
    ]
    if len(camera_matches) == 1:
        key = camera_matches[0]
        return key, leaves[key]

    suffix_matches = [key for key in leaves if key.endswith(suffix)]
    available = ", ".join(sorted(leaves))
    if not suffix_matches:
        raise LiberoIntegrationError(
            f"No {suffix[1:]} observation was returned for {camera_name!r}. "
            f"Available keys: {available}"
        )
    raise LiberoIntegrationError(
        f"Could not uniquely select the {camera_name!r} {suffix[1:]} observation. "
        f"Candidates: {', '.join(suffix_matches)}"
    )


def _flatten_mapping(value, prefix=""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_mapping(child, child_prefix)
    else:
        yield prefix, value


def _extract_robot_state(observation):
    robot_state = {}
    for key, value in _flatten_mapping(observation):
        leaf_name = key.rsplit(".", 1)[-1]
        if not leaf_name.startswith("robot0_"):
            continue
        if leaf_name.endswith(("_image", "_depth", "_segmentation")):
            continue
        array = np.asarray(value)
        if array.size <= 64 and np.issubdtype(array.dtype, np.number):
            robot_state[key] = array.copy()
    return robot_state
