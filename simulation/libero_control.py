import numpy as np


OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0
MINIMUM_GRIP_SITE_Z_M = 0.005
FINGER_ENGAGEMENT_DEPTH_M = 0.040
LOW_PROFILE_HALF_HEIGHT_M = 0.025
MINIMUM_RAISED_GRASP_OFFSET_M = 0.010


def top_down_grasp_pose(
    world_T_cad,
    cad_center_cad_m,
    cad_extent_m,
    cad_up_axis,
    reference_world_R_eef,
):
    """Build a reachable top-down Panda pose from a registered CAD pose.

    The grasp point is at the CAD center for low-profile objects and moves toward
    the top for tall objects so the Panda hand clears the object. The two poses
    separated by 180 degrees around tool Z are gripper-equivalent; the one closest
    to the current end-effector orientation is selected.
    """
    world_T_cad = np.asarray(world_T_cad, dtype=float)
    cad_center_cad_m = np.asarray(cad_center_cad_m, dtype=float)
    cad_extent_m = np.asarray(cad_extent_m, dtype=float)
    reference_world_R_eef = np.asarray(reference_world_R_eef, dtype=float)
    if world_T_cad.shape != (4, 4) or not np.all(np.isfinite(world_T_cad)):
        raise ValueError("world_T_cad must be a finite 4x4 matrix.")
    _validate_rotation(world_T_cad[:3, :3], "world_T_cad rotation")
    if cad_center_cad_m.shape != (3,) or not np.all(np.isfinite(cad_center_cad_m)):
        raise ValueError("cad_center_cad_m must be a finite three-vector.")
    if cad_extent_m.shape != (3,) or not np.all(np.isfinite(cad_extent_m)):
        raise ValueError("cad_extent_m must be a finite three-vector.")
    if np.any(cad_extent_m <= 0.0):
        raise ValueError("cad_extent_m values must be positive.")
    _validate_rotation(reference_world_R_eef, "reference_world_R_eef")

    axis = str(cad_up_axis).upper()
    if axis == "Z":
        up_index = 2
        cad_R_grasp = np.diag([1.0, -1.0, -1.0])
    elif axis == "Y":
        up_index = 1
        cad_R_grasp = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    else:
        raise ValueError(f"Unsupported cad_up_axis {cad_up_axis!r}; expected Y or Z.")

    cad_grasp_point_m = cad_center_cad_m.copy()
    center_to_top_m = 0.5 * cad_extent_m[up_index]
    if center_to_top_m <= LOW_PROFILE_HALF_HEIGHT_M:
        center_to_grasp_m = 0.0
    else:
        center_to_grasp_m = max(
            MINIMUM_RAISED_GRASP_OFFSET_M,
            center_to_top_m - FINGER_ENGAGEMENT_DEPTH_M,
        )
    cad_grasp_point_m[up_index] += center_to_grasp_m

    base_rotation = world_T_cad[:3, :3] @ cad_R_grasp
    yaw_equivalent_rotation = base_rotation @ np.diag([-1.0, -1.0, 1.0])
    candidates = (base_rotation, yaw_equivalent_rotation)
    grasp_rotation = min(
        candidates,
        key=lambda candidate: _rotation_distance(reference_world_R_eef, candidate),
    )

    world_T_grasp = np.eye(4)
    world_T_grasp[:3, :3] = grasp_rotation
    world_T_grasp[:3, 3] = _transform_point(world_T_cad, cad_grasp_point_m)
    return world_T_grasp


def _validate_rotation(rotation, name):
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} must have determinant +1.")


def _rotation_distance(first, second):
    cosine = (np.trace(first.T @ second) - 1.0) * 0.5
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _transform_point(transform, point):
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    return (np.asarray(transform, dtype=float) @ homogeneous)[:3]


def move_eef_to_pose(
    environment,
    observation,
    world_T_eef_target,
    gripper_action,
    phase,
    callback=None,
    tolerance_m=0.008,
    orientation_tolerance_rad=0.06,
    max_steps=180,
):
    """Move robosuite's grip-site frame to a world-frame SE(3) target."""
    from robosuite.utils.control_utils import orientation_error
    from robosuite.utils.transform_utils import quat2mat

    target_pose = np.asarray(world_T_eef_target, dtype=float)
    if target_pose.shape != (4, 4) or not np.all(np.isfinite(target_pose)):
        raise ValueError(f"Invalid {phase} target pose; expected a finite 4x4 matrix.")
    target_position = target_pose[:3, 3]
    target_rotation = target_pose[:3, :3]
    if not np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"Invalid {phase} target rotation; matrix is not orthonormal.")
    if not np.isclose(np.linalg.det(target_rotation), 1.0, atol=1e-5):
        raise ValueError(f"Invalid {phase} target rotation; determinant is not +1.")
    if not (
        -0.4 <= target_position[0] <= 0.4
        and -0.4 <= target_position[1] <= 0.4
        and MINIMUM_GRIP_SITE_Z_M <= target_position[2] <= 0.5
    ):
        raise ValueError(
            f"Unsafe or unreachable {phase} target position: {target_position.tolist()}"
        )

    action_dimension = int(environment.action_dim)
    if action_dimension != 7:
        raise RuntimeError(f"Expected a 7D OSC pose action; received {action_dimension}D.")
    output_limit = np.asarray(
        environment.robots[0].controller.output_max[:3],
        dtype=float,
    )
    rotation_limit = np.asarray(
        environment.robots[0].controller.output_max[3:6],
        dtype=float,
    )
    if output_limit.shape != (3,) or rotation_limit.shape != (3,):
        raise RuntimeError("LIBERO OSC_POSE controller must expose six output limits.")
    if np.any(output_limit <= 0.0) or np.any(rotation_limit <= 0.0):
        raise RuntimeError("LIBERO OSC_POSE controller output limits must be positive.")

    stable_steps = 0
    for step_index in range(max_steps):
        current = np.asarray(observation["robot0_eef_pos"], dtype=float)
        current_quaternion = np.asarray(observation["robot0_eef_quat"], dtype=float)
        if current_quaternion.shape != (4,) or not np.all(np.isfinite(current_quaternion)):
            raise RuntimeError("LIBERO observation is missing a valid robot0_eef_quat.")
        position_error = target_position - current
        rotation_error = orientation_error(target_rotation, quat2mat(current_quaternion))
        action = np.zeros(action_dimension, dtype=np.float32)
        action[:3] = np.clip(position_error / output_limit, -1.0, 1.0)
        action[3:6] = np.clip(rotation_error / rotation_limit, -1.0, 1.0)
        action[-1] = float(gripper_action)
        observation, reward, done, info = environment.step(action)
        if callback is not None:
            callback(phase, step_index, observation, action, reward, done, info)

        reached_position = np.asarray(observation["robot0_eef_pos"], dtype=float)
        reached_quaternion = np.asarray(observation["robot0_eef_quat"], dtype=float)
        position_error = target_position - reached_position
        rotation_error = orientation_error(
            target_rotation,
            quat2mat(reached_quaternion),
        )
        if (
            np.linalg.norm(position_error) <= tolerance_m
            and np.linalg.norm(rotation_error) <= orientation_tolerance_rad
        ):
            stable_steps += 1
            if stable_steps >= 3:
                return observation
        else:
            stable_steps = 0

    final_position = np.asarray(observation["robot0_eef_pos"], dtype=float)
    final_rotation = quat2mat(np.asarray(observation["robot0_eef_quat"], dtype=float))
    final_position_error = float(np.linalg.norm(target_position - final_position))
    final_rotation_error = float(np.linalg.norm(orientation_error(target_rotation, final_rotation)))
    raise RuntimeError(
        f"End effector did not reach {phase} target {target_position.tolist()} within "
        f"{max_steps} steps; final position was {final_position.tolist()} and "
        f"errors were {final_position_error:.4f} m and {final_rotation_error:.4f} rad."
    )


def hold_gripper(
    environment,
    observation,
    gripper_action,
    phase,
    steps,
    callback=None,
):
    action = np.zeros(int(environment.action_dim), dtype=np.float32)
    action[-1] = float(gripper_action)
    for step_index in range(int(steps)):
        observation, reward, done, info = environment.step(action)
        if callback is not None:
            callback(phase, step_index, observation, action, reward, done, info)
    return observation


def execute_top_grasp_and_place(
    environment,
    observation,
    world_T_grasp,
    place_xyz_m,
    callback=None,
):
    """Execute a top grasp using a perception-derived world-frame grip-site pose."""
    grasp_pose = np.asarray(world_T_grasp, dtype=float)
    if grasp_pose.shape != (4, 4) or not np.all(np.isfinite(grasp_pose)):
        raise ValueError("world_T_grasp must be a finite 4x4 matrix.")
    place = np.asarray(place_xyz_m, dtype=float)
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[2, 3] += 0.15
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] = 0.27
    preplace_pose = grasp_pose.copy()
    preplace_pose[:3, 3] = [place[0], place[1], max(0.29, place[2] + 0.16)]
    retreat_pose = preplace_pose.copy()
    retreat_pose[2, 3] = 0.35

    observation = move_eef_to_pose(
        environment,
        observation,
        pregrasp_pose,
        OPEN_GRIPPER,
        "move_above_target",
        callback,
    )
    observation = hold_gripper(
        environment, observation, OPEN_GRIPPER, "open_gripper", 12, callback
    )
    observation = move_eef_to_pose(
        environment,
        observation,
        grasp_pose,
        OPEN_GRIPPER,
        "descend_to_target",
        callback,
    )
    observation = hold_gripper(
        environment, observation, CLOSE_GRIPPER, "close_gripper", 30, callback
    )
    observation = move_eef_to_pose(
        environment, observation, lift_pose, CLOSE_GRIPPER, "lift_target", callback
    )
    observation = move_eef_to_pose(
        environment,
        observation,
        preplace_pose,
        CLOSE_GRIPPER,
        "move_above_basket",
        callback,
    )
    observation = hold_gripper(
        environment, observation, OPEN_GRIPPER, "release_target", 35, callback
    )
    observation = move_eef_to_pose(
        environment, observation, retreat_pose, OPEN_GRIPPER, "retreat", callback
    )
    observation = hold_gripper(
        environment, observation, OPEN_GRIPPER, "settle", 30, callback
    )
    return observation
