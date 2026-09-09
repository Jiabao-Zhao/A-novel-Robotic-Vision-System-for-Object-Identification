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
    *,
    free_yaw=False,
):
    """Build a candidate top-down Panda pose from a registered CAD pose.

    The grasp point is at the CAD center for low-profile objects and moves toward
    the top for tall objects so the Panda hand clears the object. The two poses
    separated by 180 degrees around tool Z are gripper-equivalent; the one closest
    to the current end-effector orientation is selected. This does not check
    inverse kinematics, collisions, or grasp stability. For a part that permits
    gripping at any angle around its up axis, free_yaw preserves the robot's
    current heading instead of imposing the registered CAD yaw.
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
    if free_yaw:
        down = -world_T_cad[:3, up_index]
        heading = reference_world_R_eef[:, 0]
        heading = heading - down * np.dot(heading, down)
        if np.linalg.norm(heading) < 1e-8:
            raise ValueError("Cannot project the reference gripper heading onto the grasp plane.")
        heading /= np.linalg.norm(heading)
        grasp_rotation = np.column_stack((heading, np.cross(down, heading), down))

    world_T_grasp = np.eye(4)
    world_T_grasp[:3, :3] = grasp_rotation
    world_T_grasp[:3, 3] = _transform_point(world_T_cad, cad_grasp_point_m)
    # A floor-level CAD estimate can sit slightly below the controller bound.
    # Correct only a shortfall up to 1 mm; larger violations stay invalid.
    if MINIMUM_GRIP_SITE_Z_M - .001 <= world_T_grasp[2, 3] < MINIMUM_GRIP_SITE_Z_M:
        world_T_grasp[2, 3] = MINIMUM_GRIP_SITE_Z_M + .0001
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


def select_grasp_orientation(environment, world_T_grasp):
    """Check equivalent jaw orientations against robot joint limits before motion.

    Uses robot kinematics and the supplied grasp pose, without object truth.
    The pregrasp/grasp checks are not a collision or complete trajectory check.
    """
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    robot, sim = environment.robots[0], environment.sim
    model = sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = sim.data.qpos
    joint_ids = [sim.model.joint_name2id(name) for name in robot.robot_model.joints]
    indices = model.jnt_qposadr[joint_ids]
    lower, upper = model.jnt_range[joint_ids].T
    initial = data.qpos[indices].copy()
    site = sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
    candidates = []
    for angle in (0, 180):
        pose = np.asarray(world_T_grasp, dtype=float).copy()
        if angle:
            pose[:3, :3] = pose[:3, :3] @ np.diag([-1., -1., 1.])
        q, checks = initial.copy(), []
        for height_offset in (.15, 0.):
            target = pose.copy()
            target[2, 3] += height_offset
            reference = q.copy()

            def residual(joints):
                data.qpos[indices] = joints
                mujoco.mj_kinematics(model, data)
                position = data.site_xpos[site] - target[:3, 3]
                rotation = Rotation.from_matrix(
                    target[:3, :3] @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
                return np.r_[position, .1*rotation, .0001*(joints-reference)]

            fit = least_squares(residual, np.clip(q, lower+1e-6, upper-1e-6),
                bounds=(lower+1e-6, upper-1e-6), max_nfev=200,
                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            q = fit.x
            error = residual(q)
            check = {"height_offset_m": height_offset,
                "position_error_mm": float(np.linalg.norm(error[:3])*1000),
                "rotation_error_deg": float(np.rad2deg(np.linalg.norm(error[3:6])/.1)),
                "joint_limit_margin_deg": float(np.rad2deg(min(np.min(q-lower), np.min(upper-q))))}
            checks.append(check)
            feasible = check["position_error_mm"] <= 1 and check["rotation_error_deg"] <= 1
            if not feasible:
                break
        candidates.append({"equivalent_rotation_deg": angle, "checks": checks})
        if len(checks) == 2 and feasible:
            return pose, {"selected_equivalent_rotation_deg": angle, "candidates": candidates}
    raise RuntimeError(f"No equivalent grasp passed the robot joint-limit check: {candidates}")


def _pose_action(environment, observation, position, rotation, gripper_action):
    from robosuite.utils.control_utils import orientation_error
    from robosuite.utils.transform_utils import quat2mat

    if int(environment.action_dim) != 7:
        raise RuntimeError("Expected a 7D OSC pose action.")
    limits = np.asarray(environment.robots[0].controller.output_max, dtype=float)
    if limits.shape != (6,) or np.any(limits <= 0):
        raise RuntimeError("OSC_POSE controller must expose six positive output limits.")
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("LIBERO observation is missing a valid robot0_eef_quat.")
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip((position-observation["robot0_eef_pos"])/limits[:3], -1., 1.)
    action[3:6] = np.clip(orientation_error(rotation, quat2mat(quaternion))/limits[3:6], -1., 1.)
    action[-1] = float(gripper_action)
    return action


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
            f"Outside configured translation bounds for {phase}: {target_position.tolist()}"
        )

    stable_steps = 0
    for step_index in range(max_steps):
        action = _pose_action(environment, observation, target_position, target_rotation, gripper_action)
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
    """Keep one fixed hand pose while opening, closing, or holding the fingers."""
    from robosuite.utils.transform_utils import quat2mat

    position = np.asarray(observation["robot0_eef_pos"], dtype=float).copy()
    rotation = quat2mat(observation["robot0_eef_quat"])
    for step_index in range(int(steps)):
        action = _pose_action(environment, observation, position, rotation, gripper_action)
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
    """Reference sequence using the same skills as the LLM plan executor."""
    observation = pick_object(environment, observation, world_T_grasp, callback)
    return place_object(environment, observation, world_T_grasp, place_xyz_m, callback)


def pick_object(environment, observation, world_T_grasp, callback=None):
    """Approach, close, and lift using a perception-derived grip-site pose."""
    grasp_pose = np.asarray(world_T_grasp, dtype=float)
    if grasp_pose.shape != (4, 4) or not np.all(np.isfinite(grasp_pose)):
        raise ValueError("world_T_grasp must be a finite 4x4 matrix.")
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[2, 3] += 0.15
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] = 0.27

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
    return observation


def place_object(environment, observation, world_T_grasp, place_xyz_m, callback=None):
    """Release into an open container; retains the original basket controller."""
    preplace_pose = np.asarray(world_T_grasp, dtype=float).copy()
    place = np.asarray(place_xyz_m, dtype=float)
    preplace_pose[:3, 3] = [place[0], place[1], max(0.29, place[2] + 0.16)]
    retreat_pose = preplace_pose.copy()
    retreat_pose[2, 3] = 0.35
    observation = move_eef_to_pose(
        environment,
        observation,
        preplace_pose,
        CLOSE_GRIPPER,
        "move_above_destination",
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
