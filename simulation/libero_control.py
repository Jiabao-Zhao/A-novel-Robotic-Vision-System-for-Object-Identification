import numpy as np


OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0


def move_eef_to(
    environment,
    observation,
    target_xyz_m,
    gripper_action,
    phase,
    callback=None,
    tolerance_m=0.008,
    max_steps=120,
):
    """Move the end effector with LIBERO's normalized OSC position action."""
    target = np.asarray(target_xyz_m, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError(f"Invalid {phase} target position: {target_xyz_m}")
    if not (-0.4 <= target[0] <= 0.4 and -0.4 <= target[1] <= 0.4 and 0.03 <= target[2] <= 0.5):
        raise ValueError(f"Unsafe or unreachable {phase} target position: {target.tolist()}")

    action_dimension = int(environment.action_dim)
    if action_dimension != 7:
        raise RuntimeError(f"Expected a 7D OSC pose action; received {action_dimension}D.")
    translation_limit = np.asarray(
        environment.robots[0].controller.output_max[:3],
        dtype=float,
    )

    stable_steps = 0
    for step_index in range(max_steps):
        current = np.asarray(observation["robot0_eef_pos"], dtype=float)
        error = target - current
        action = np.zeros(action_dimension, dtype=np.float32)
        action[:3] = np.clip(error / translation_limit, -1.0, 1.0)
        action[-1] = float(gripper_action)
        observation, reward, done, info = environment.step(action)
        if callback is not None:
            callback(phase, step_index, observation, action, reward, done, info)

        if np.linalg.norm(error) <= tolerance_m:
            stable_steps += 1
            if stable_steps >= 3:
                return observation
        else:
            stable_steps = 0

    final_position = np.asarray(observation["robot0_eef_pos"], dtype=float)
    final_error = float(np.linalg.norm(target - final_position))
    raise RuntimeError(
        f"End effector did not reach {phase} target {target.tolist()} within "
        f"{max_steps} steps; final position was {final_position.tolist()} and "
        f"the error was {final_error:.4f} m."
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
    pick_xyz_m,
    place_xyz_m,
    callback=None,
):
    """Execute a conservative top grasp and drop using perceived world points."""
    pick = np.asarray(pick_xyz_m, dtype=float)
    place = np.asarray(place_xyz_m, dtype=float)
    pregrasp = np.array([pick[0], pick[1], max(0.21, pick[2] + 0.15)])
    grasp = np.array([pick[0], pick[1], max(0.11, pick[2] + 0.055)])
    lift = np.array([pick[0], pick[1], 0.27])
    preplace = np.array([place[0], place[1], max(0.29, place[2] + 0.16)])
    retreat = np.array([place[0], place[1], 0.35])

    observation = move_eef_to(
        environment, observation, pregrasp, OPEN_GRIPPER, "move_above_target", callback
    )
    observation = hold_gripper(
        environment, observation, OPEN_GRIPPER, "open_gripper", 12, callback
    )
    observation = move_eef_to(
        environment, observation, grasp, OPEN_GRIPPER, "descend_to_target", callback
    )
    observation = hold_gripper(
        environment, observation, CLOSE_GRIPPER, "close_gripper", 30, callback
    )
    observation = move_eef_to(
        environment, observation, lift, CLOSE_GRIPPER, "lift_target", callback
    )
    observation = move_eef_to(
        environment, observation, preplace, CLOSE_GRIPPER, "move_above_basket", callback
    )
    observation = hold_gripper(
        environment, observation, OPEN_GRIPPER, "release_target", 35, callback
    )
    observation = move_eef_to(
        environment, observation, retreat, OPEN_GRIPPER, "retreat", callback
    )
    observation = hold_gripper(
        environment, observation, OPEN_GRIPPER, "settle", 30, callback
    )
    return observation
