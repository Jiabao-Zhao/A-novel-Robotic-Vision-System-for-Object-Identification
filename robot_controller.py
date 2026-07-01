import threading
import time
import logging

import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Rotation
import math


logger = logging.getLogger(__name__)

ZERO_TCP_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FIXED_TCP_OFFSET = [0.0, 0.0, 0.22438, 0.0, 0.0, 0.0]
DEFAULT_TCP_OFFSET = list(FIXED_TCP_OFFSET)
SAFE_HOME_CONFIG = {
    "enabled": True,
    "home_joints_rad": [
        -1.3946722189532679,
        -0.7461099785617371,
        -2.337867498397827,
        -1.6291781864561976,
        1.5701189041137695,
        -2.96495229402651,
    ],
    "joint_speed": 0.35,
    "joint_acceleration": 0.35,
    "settle_s": 0.05,
    "skip_if_within_rad": 0.005,
}
ROBOT_HOSTNAME = "192.168.1.172"
RTDE_CONTROL_FREQUENCY_HZ = 125.0
RTDE_RECEIVE_FREQUENCY_HZ = 125.0
RTDE_RECEIVE_VARIABLES = [
    "actual_TCP_pose",
    "actual_TCP_speed",
    "actual_TCP_force",
    "actual_q",
    "actual_qd",
]


def read_active_tcp_offset(rtde_interface, fallback=None):
    fallback = list(DEFAULT_TCP_OFFSET if fallback is None else fallback)
    getter = getattr(rtde_interface, "getTCPOffset", None)
    if not callable(getter):
        return fallback

    try:
        tcp = np.asarray(getter(), dtype=float).reshape(-1)
    except Exception:
        return fallback

    if tcp.size != 6 or not np.all(np.isfinite(tcp)):
        return fallback
    return [float(value) for value in tcp.tolist()]


class RTDEStateFeedback:
    def __init__(self, rate=RTDE_RECEIVE_FREQUENCY_HZ):
        self.rate = float(rate)
        self.nominal_dt = 1.0 / self.rate
        self._hostname = ROBOT_HOSTNAME
        self._variables = list(RTDE_RECEIVE_VARIABLES)

        self._rtde_r = self._connect_receive()
        self.lock = threading.Lock()
        self.first_sample_ready = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.TCP = list(FIXED_TCP_OFFSET)
        self._last_receive_reconnect_s = 0.0

        self.pose = None
        self.tcp_speed = None
        self.wrench = None
        self.q = None
        self.qd = None

        self.thread.start()

    def _connect_receive(self):
        return RTDEReceiveInterface(
            self._hostname,
            self.rate,
            self._variables,
        )

    def _reconnect_receive(self):
        now = time.monotonic()
        if now - self._last_receive_reconnect_s < 1.0:
            return
        self._last_receive_reconnect_s = now
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass
        try:
            self._rtde_r = self._connect_receive()
        except Exception:
            pass

    def _read_loop(self):
        while self.running:
            try:
                pose = np.asarray(self._rtde_r.getActualTCPPose(), dtype=float)
                tcp_speed = np.asarray(self._rtde_r.getActualTCPSpeed(), dtype=float)
                wrench = np.asarray(self._rtde_r.getActualTCPForce(), dtype=float)
                q = np.asarray(self._rtde_r.getActualQ(), dtype=float)
                qd = np.asarray(self._rtde_r.getActualQd(), dtype=float)

                with self.lock:
                    self.pose = pose
                    self.tcp_speed = tcp_speed
                    self.wrench = wrench
                    self.q = q
                    self.qd = qd

                self.first_sample_ready.set()
            except Exception:
                self._reconnect_receive()
            time.sleep(self.nominal_dt)

    def get_pose(self):
        self.first_sample_ready.wait()

        with self.lock:
            pose = self.pose.copy()

        rotvec = pose[3:6]
        r = Rotation.from_rotvec(rotvec)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)

        return [
            float(pose[0] * 1000),
            float(pose[1] * 1000),
            float(pose[2] * 1000),
            float(roll),
            float(pitch),
            float(yaw),
        ]

    def get_raw_pose(self):
        self.first_sample_ready.wait()
        with self.lock:
            return self.pose.copy()

    def get_tcp_speed(self):
        self.first_sample_ready.wait()
        with self.lock:
            return self.tcp_speed.copy()

    def get_wrench(self):
        self.first_sample_ready.wait()
        with self.lock:
            return self.wrench.copy()

    def get_tcp_offset(self):
        return list(self.TCP)

    def get_q(self):
        self.first_sample_ready.wait()
        with self.lock:
            return self.q.copy()

    def get_qd(self):
        self.first_sample_ready.wait()
        with self.lock:
            return self.qd.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self._rtde_r.disconnect()

class RTDECommander:
    def __init__(self, robot_state):
        self.hostname = ROBOT_HOSTNAME
        self.control_frequency_hz = RTDE_CONTROL_FREQUENCY_HZ
        self._rtde_c = self._connect_control()
        self.acc = 0.5
        self.vel = 0.3
        self.vz = -0.01
        self.fz_threshold = 5.0
        self.stable_samples = 5
        self.dt = 0.008
        self.timeout_s = 30.0
        self.stop_acc = 1.0
        self.motion_settle_timeout_s = 5.0
        self.motion_tcp_speed_tol = 0.002
        self.motion_joint_speed_tol = 0.01
        self.motion_settle_samples = 5
        self.last_gripper_width = None
        self.RobotState = robot_state
        self.TCP = list(FIXED_TCP_OFFSET)
        logger.info(
            "Fixed TCP offset: %s",
            self._format_tcp_offset(self.TCP),
        )
        self.task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.selection_vector = [0, 0, 1, 0, 0, 0]
        self.search_force_z = -4.0
        self.wrench = [0.0, 0.0, self.search_force_z, 0.0, 0.0, 0.0]
        self.limits = [0.02, 0.02, 0.02, 1.0, 1.0, 1.0]
        self.search_damping = 0.3
        self.force_gain_scaling = 0.8
        self.hole_fz_drop_threshold = 2.0
        self.hole_z_drop_distance_mm = 1.0
        self.hole_detect_samples = 3
        self.hole_min_search_time_s = 0.25
        self.hole_min_search_radius_m = 0.001
        self.search_timeout = 60.0
        self.force_type = 2
        self.spiral_pitch = 0.001
        self.spiral_speed = 0.005
        self.spiral_max_radius = 0.02
        self.spiral_dt = 0.008

    def _connect_control(self):
        return RTDEControlInterface(self.hostname, self.control_frequency_hz)
    # ═════════════════════════════════════════════════════════════════════════════
    #  Robot Dynamics
    # ═════════════════════════════════════════════════════════════════════════════

    @staticmethod
    def rpy_to_rotvec(rpy_deg):
        r = Rotation.from_euler('xyz', rpy_deg, degrees=True)
        return r.as_rotvec()

    def build_pose(self, pose_xyz_rpy):
        pos_m = [c / 1000.0 for c in pose_xyz_rpy[:3]]  # mm → m
        rotvec = self.rpy_to_rotvec(pose_xyz_rpy[3:])
        return list(pos_m) + list(rotvec)

    def inverse_kin(self, pose_xyz_rpy):
        pose = self.build_pose(pose_xyz_rpy)
        qnear = self.RobotState.get_q()
        try:
            raw = self._rtde_c.getInverseKinematics(pose, qnear)
            q = np.asarray(raw, dtype=float)
        except Exception:
            self._reconnect_control()
            return None

        if q.size != 6 or not np.all(np.isfinite(q)):
            self._reconnect_control()
            return None
        return q

    def refresh_tcp_offset(self, log=False):
        self.TCP = list(FIXED_TCP_OFFSET)
        return list(self.TCP)

    @staticmethod
    def _format_tcp_offset(tcp):
        xyz_mm = [float(value) * 1000.0 for value in tcp[:3]]
        rot = [float(value) for value in tcp[3:]]
        return (
            f"xyz_mm=[{xyz_mm[0]:.3f}, {xyz_mm[1]:.3f}, {xyz_mm[2]:.3f}], "
            f"rotvec_rad=[{rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}]"
        )

    def forward_kin(self, q):
        self.refresh_tcp_offset()
        return np.asarray(self._rtde_c.getForwardKinematics(list(q), self.TCP), dtype=float)

    def forward_kin_as_pose(self, q):
        fk_raw = self.forward_kin(q)
        r = Rotation.from_rotvec(fk_raw[3:6])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return [
            fk_raw[0] * 1000,
            fk_raw[1] * 1000,
            fk_raw[2] * 1000,
            float(roll),
            float(pitch),
            float(yaw),
            ]

    def get_jacobian(self, q):
        self.refresh_tcp_offset()
        J = np.asarray(self._rtde_c.getJacobian(list(q), self.TCP), dtype=float)
        return J.reshape(6, 6)

    def get_jacobian_time_derivative(self, q, qd):
        self.refresh_tcp_offset()
        J_dot = np.asarray(self._rtde_c.getJacobianTimeDerivative(list(q), list(qd), self.TCP), dtype=float)
        return J_dot.reshape(6, 6)

    def get_mass_matrix(self, q):
        M = np.asarray(self._rtde_c.getMassMatrix(list(q)), dtype=float)
        return M.reshape(6, 6)

    def get_coriolis_and_centrifugal_torques(self, q, qd):
        C = np.asarray(self._rtde_c.getCoriolisAndCentrifugalTorques(list(q), list(qd)), dtype=float)
        return C

    def direct_torque(self, tau):
        self._rtde_c.directTorque(list(np.asarray(tau, dtype=float)))

    # ═════════════════════════════════════════════════════════════════════════════
    #  Utility
    # ═════════════════════════════════════════════════════════════════════════════
    def zero_ft_sensor(self):
        self._rtde_c.zeroFtSensor()

    def wait_until_motion_complete(self, timeout_s=None):
        timeout_s = self.motion_settle_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        stable = 0

        while time.monotonic() < deadline:
            tcp_speed = self.RobotState.get_tcp_speed()
            qd = self.RobotState.get_qd()
            linear_speed = float(np.linalg.norm(tcp_speed[:3]))
            joint_speed = float(np.linalg.norm(qd))

            if (
                    linear_speed < self.motion_tcp_speed_tol
                    and joint_speed < self.motion_joint_speed_tol
            ):
                stable += 1
                if stable >= self.motion_settle_samples:
                    return True
            else:
                stable = 0

            time.sleep(self.dt)

        return False

    def _restore_control_script_after_custom_script(self, reason: str):
        reupload = getattr(self._rtde_c, "reuploadScript", None)
        if not callable(reupload):
            return
        try:
            reupload()
        except Exception:
            self._reconnect_control()

    def gripper_set(self, width, force, blocking=True, settle_s=None):
        self._ensure_control_connected()
        tool_index = 0
        body = f"""
      local rg = rpc_factory("xmlrpc","http://localhost:41414")
      local ret = rg.rg_grip({tool_index}, {float(width)}, {float(force)})
      textmsg("rg_grip returned: ", ret)
    """
        self._rtde_c.sendCustomScriptFunction("rg2_cmd", body)
        self._restore_control_script_after_custom_script(reason="gripper_set")
        if blocking:
            if settle_s is None:
                if self.last_gripper_width is None:
                    settle_s = 1.5
                else:
                    travel_mm = abs(float(width) - float(self.last_gripper_width))
                    settle_s = min(2.0, max(0.6, 0.03 * travel_mm))
            time.sleep(settle_s)
        self.last_gripper_width = float(width)

    def set_gripper_open(self, blocking=True):
        self.gripper_set(70, 10, blocking=blocking, settle_s=1.2)

    def set_gripper_close(self, blocking=True):
        self.gripper_set(10, 10, blocking=blocking, settle_s=2.0)

    def zeroFtSensor(self):
        self._ensure_control_connected()
        self._rtde_c.zeroFtSensor()

    # ═════════════════════════════════════════════════════════════════════════════
    #  Move in cartesian coordinate
    # ═════════════════════════════════════════════════════════════════════════════
    def go_home(self, open_gripper=True, pre_lift_z_mm=0.0):
        if open_gripper:
            self.set_gripper_open()
        pre_lift_z_mm = float(pre_lift_z_mm or 0.0)
        if abs(pre_lift_z_mm) > 1e-9:
            self._ensure_control_connected()
            self.move_in_z(pre_lift_z_mm)
        self._execute_safe_home_route()

    def _execute_safe_home_route(self):
        self._ensure_control_connected()
        config = SAFE_HOME_CONFIG
        if not bool(config.get("enabled", True)):
            raise RuntimeError("Safe home route is disabled.")
        speed = float(config.get("joint_speed", self.vel))
        acceleration = float(config.get("joint_acceleration", self.acc))
        settle_s = float(config.get("settle_s", 0.05))
        skip_tolerance_rad = float(config.get("skip_if_within_rad", 0.005))
        target_q = self._validate_joint_target(config["home_joints_rad"], label="home_joints_rad")

        if speed <= 0 or acceleration <= 0:
            raise ValueError("Safe home joint_speed and joint_acceleration must be positive.")

        current_q = self._current_joint_positions()
        joint_delta = target_q - current_q
        if float(np.max(np.abs(joint_delta))) <= skip_tolerance_rad:
            return

        self._rtde_c.moveJ(target_q.tolist(), speed, acceleration, asynchronous=False)
        self.wait_until_motion_complete()
        if settle_s > 0:
            time.sleep(settle_s)

    def _current_joint_positions(self):
        getter = getattr(self.RobotState, "get_q", None)
        if not callable(getter):
            raise RuntimeError("Cannot execute safe home route because robot joint feedback is unavailable.")
        q = np.asarray(getter(), dtype=float).reshape(-1)
        if q.size != 6 or not np.all(np.isfinite(q)):
            raise RuntimeError("Current robot joint feedback is invalid.")
        return q

    def is_at_home(self, tolerance_rad=None):
        config = SAFE_HOME_CONFIG
        target_q = self._validate_joint_target(
            config["home_joints_rad"],
            label="home_joints_rad",
        )
        tolerance = float(
            config.get("skip_if_within_rad", 0.005)
            if tolerance_rad is None
            else tolerance_rad
        )
        if tolerance <= 0:
            tolerance = 0.005
        current_q = self._current_joint_positions()
        return float(np.max(np.abs(target_q - current_q))) <= tolerance

    @staticmethod
    def _validate_joint_target(values, label="joint_target"):
        q = np.asarray(values, dtype=float).reshape(-1)
        if q.size != 6:
            raise ValueError(f"{label} must contain exactly 6 joint values in radians.")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{label} contains non-finite joint values.")
        if np.max(np.abs(q)) > (2.0 * math.pi + 1e-6):
            raise ValueError(f"{label} contains a joint outside +/- 2*pi rad.")
        return q

    def move_to_pose(self, pose):
        self.move_to_cartesian(pose[0], pose[1], pose[2], -180, 0, 0)

    def move_to_coordinate(self, coordinate):
        self.move_to_cartesian(*coordinate)

    def move_in_x(self, dx):
        """Move in X direction by dx mm"""
        p = self.RobotState.get_pose()
        p[0] += dx
        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def move_in_y(self, dy):
        """Move in Y direction by dy mm"""
        p = self.RobotState.get_pose()
        p[1] += dy
        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def move_in_z(self, dz):
        """Move in Z direction by dz mm"""
        p = self.RobotState.get_pose()
        p[2] += dz
        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def move_in_cartesian(self, dx, dy, dz, d_roll, d_pitch, d_yaw):
        p = self.RobotState.get_pose()
        p[0] += dx
        p[1] += dy
        p[2] += dz
        p[3] += d_roll
        p[4] += d_pitch
        p[5] += d_yaw

        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def orthogonal_regrasp_alignment(
            self,
            slide_open_width=70.0,
            final_close_width=10.0,
            yaw_delta_deg=90.0,
            approach_lift_mm=50.0,
            rotation_clearance_lift_mm=30.0,
            gentle_force=5.0,
            final_force=20.0,
            settle_s=0.8,
    ):
        """
        Table-supported orthogonal regrasp alignment.

        Preconditions:
        - The part is already grasped.
        - The robot can lift the part before rotating yaw.
        - A support/alignment surface is below the part.

        Sequence:
        1. Lift the grasped part to the approach height.
        2. Rotate TCP yaw by 90 degrees.
        3. Move down until contact so the part is supported.
        4. Open the gripper so the table-supported part can settle.
        5. Lift the gripper above the supported part.
        6. Rotate yaw back.
        7. Return to the supported grasp height.
        8. Close to the final grasp width.
        """
        params = self._validate_orthogonal_regrasp_params(
            slide_open_width=slide_open_width,
            final_close_width=final_close_width,
            yaw_delta_deg=yaw_delta_deg,
            approach_lift_mm=approach_lift_mm,
            rotation_clearance_lift_mm=rotation_clearance_lift_mm,
            gentle_force=gentle_force,
            final_force=final_force,
            settle_s=settle_s,
        )

        self.move_in_cartesian(0.0, 0.0, params["approach_lift_mm"], 0.0, 0.0, 0.0)
        self.move_in_cartesian(0.0, 0.0, 0.0, 0.0, 0.0, params["yaw_delta_deg"])

        contact_ok = self.move_down_until_contact()
        if not contact_ok:
            raise RuntimeError("orthogonal_regrasp_alignment failed before regrasp: no support contact detected.")

        self.gripper_set(
            params["slide_open_width"],
            params["gentle_force"],
            blocking=True,
            settle_s=params["settle_s"],
        )
        self.move_in_cartesian(
            0.0,
            0.0,
            params["rotation_clearance_lift_mm"],
            0.0,
            0.0,
            0.0,
        )
        self.move_in_cartesian(0.0, 0.0, 0.0, 0.0, 0.0, -params["yaw_delta_deg"])
        self.move_in_cartesian(
            0.0,
            0.0,
            -params["rotation_clearance_lift_mm"],
            0.0,
            0.0,
            0.0,
        )
        self.gripper_set(
            params["final_close_width"],
            params["final_force"],
            blocking=True,
            settle_s=params["settle_s"],
        )

        return True

    @staticmethod
    def _validate_orthogonal_regrasp_params(
            slide_open_width,
            final_close_width,
            yaw_delta_deg,
            approach_lift_mm,
            rotation_clearance_lift_mm,
            gentle_force,
            final_force,
            settle_s,
    ):
        params = {
            "slide_open_width": float(slide_open_width),
            "final_close_width": float(final_close_width),
            "yaw_delta_deg": float(yaw_delta_deg),
            "approach_lift_mm": float(approach_lift_mm),
            "rotation_clearance_lift_mm": float(rotation_clearance_lift_mm),
            "gentle_force": float(gentle_force),
            "final_force": float(final_force),
            "settle_s": float(settle_s),
        }

        if not 0.0 < params["slide_open_width"] <= 110.0:
            raise ValueError("slide_open_width must be in (0, 110] mm.")
        if not 0.0 < params["final_close_width"] <= params["slide_open_width"]:
            raise ValueError("final_close_width must be positive and no larger than slide_open_width.")
        if abs(params["yaw_delta_deg"]) < 45.0 or abs(params["yaw_delta_deg"]) > 135.0:
            raise ValueError("yaw_delta_deg should be near 90 degrees for orthogonal alignment.")
        if params["approach_lift_mm"] < 0.0:
            raise ValueError("approach_lift_mm must be non-negative.")
        if params["rotation_clearance_lift_mm"] < 0.0:
            raise ValueError("rotation_clearance_lift_mm must be non-negative.")
        if params["gentle_force"] <= 0.0 or params["final_force"] <= 0.0:
            raise ValueError("gentle_force and final_force must be positive.")
        if params["gentle_force"] > params["final_force"]:
            raise ValueError("gentle_force should not exceed final_force.")
        if params["settle_s"] < 0.0:
            raise ValueError("settle_s must be non-negative.")
        return params

    def move_to_cartesian(self, x, y, z, roll, pitch, yaw):
        p = [x, y, z, roll, pitch, yaw]
        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def rotate_single_joint(self, joint_index, delta_deg):
        current_q = self.RobotState.get_q()
        target_q = current_q.copy()
        target_q[joint_index] += math.radians(delta_deg)
        self._rtde_c.moveJ(target_q, self.vel, self.acc, asynchronous=False)

    def send_velocity(self, vel):
        self._rtde_c.speedL(vel.tolist(), self.acc, time=0.1)

    def stopL(self):
        self._rtde_c.stopL(2)

    def soft_stop_motion(self):
        """Best-effort stop for active RTDE motion modes."""
        stopped = False
        for stop_call in (
                lambda: self._rtde_c.speedStop(self.stop_acc),
                lambda: self._rtde_c.servoStop(),
                lambda: self._rtde_c.stopL(2),
        ):
            try:
                stop_call()
                stopped = True
            except Exception:
                pass
        return stopped

    def move_until_contact(self):
        self.zero_ft_sensor()
        t0 = time.time()
        over = 0

        try:
            self._rtde_c.speedL([0.0, 0.0, self.vz, 0.0, 0.0, 0.0], 0.3, 0.1)

            while True:
                if (time.time() - t0) > self.timeout_s:
                    self._rtde_c.speedStop(self.stop_acc)
                    print("[move_until_contact] Timeout - no contact detected")
                    return False

                wrench = self.RobotState.get_wrench()
                fz = wrench[2]

                if abs(fz) >= self.fz_threshold:
                    over += 1
                    if over >= self.stable_samples:
                        self._rtde_c.speedStop(self.stop_acc)
                        print(f"[move_until_contact] Contact detected! Fz={fz:.2f}N")
                        return True
                else:
                    over = 0
                time.sleep(self.dt)
        finally:
            try:
                self._rtde_c.speedStop(self.stop_acc)
            except Exception:
                pass

    def move_until_down_until(self):
        return self.move_until_contact()

    def move_down_until_contact(self):
        return self.move_until_contact()

    def _reconnect_control(self):
        try:
            self._rtde_c.disconnect()
        except Exception:
            pass
        try:
            self._rtde_c = self._connect_control()
            self.refresh_tcp_offset(log=True)
        except Exception:
            raise

    def disconnect(self):
        self._rtde_c.disconnect()

    def _ensure_control_connected(self):
        try:
            if self._rtde_c.isConnected():
                return
        except Exception:
            pass
        self._reconnect_control()

if __name__ == "__main__":
    state = RTDEStateFeedback()
    robot = RTDECommander(state)
    robot.go_home()

