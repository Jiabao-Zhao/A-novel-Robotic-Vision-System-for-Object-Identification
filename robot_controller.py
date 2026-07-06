import threading
import time
import logging
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Rotation
import math


logger = logging.getLogger(__name__)

ZERO_TCP_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FIXED_TCP_OFFSET = [0.0, 0.0, 0.22438, 0.0, 0.0, 0.0]
DEFAULT_TCP_OFFSET = list(FIXED_TCP_OFFSET)
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
T_EE_CAMERA = np.array([
    [-0.994932591781407, -0.095336836012590, 0.031937838221171, 25.592566657792],
    [0.096220059168056, -0.994983958303154, 0.027360974636959, 73.297417454375],
    [0.029169127940838, 0.030295386092556, 0.999115284417506, 18.666004062560674],
    [0.0, 0.0, 0.0, 1.0],
])
RG2_REFERENCE_OPEN_WIDTH_MM = 70.0
RG2_DEFAULT_OPEN_WIDTH_MM = 70.0
RG2_DEFAULT_CLOSE_WIDTH_MM = 10.0
RG2_MIN_WIDTH_MM = 1.0
RG2_MAX_WIDTH_MM = 110.0
RG2_OPEN_CLEARANCE_MM = 30.0
RG2_CLOSE_MARGIN_MM = 2.0
RG2_PICK_OPEN_FORCE = 10.0
RG2_PICK_CLOSE_FORCE = 10.0
RG2_PICK_CLEARANCE_Z_MM = 50.0
RG2_FINGERTIP_HEIGHT_CURVE_MM = (
    (10.0, 64.0), (15.0, 57.0), (20.0, 52.0), (25.0, 50.0),
    (40.0, 49.0), (50.0, 48.0), (60.0, 45.0), (70.0, 42.0),
    (80.0, 40.0), (90.0, 38.0), (100.0, 36.0), (110.0, 36.0),
)


def clamp(value, lower, upper):
    return min(float(upper), max(float(lower), float(value)))


def rg2_fingertip_height_mm(width_mm):
    width = clamp(
        width_mm,
        RG2_FINGERTIP_HEIGHT_CURVE_MM[0][0],
        RG2_FINGERTIP_HEIGHT_CURVE_MM[-1][0],
    )
    for (x0, y0), (x1, y1) in zip(
        RG2_FINGERTIP_HEIGHT_CURVE_MM,
        RG2_FINGERTIP_HEIGHT_CURVE_MM[1:],
    ):
        if x0 <= width <= x1:
            return y0 + ((width - x0) / (x1 - x0)) * (y1 - y0)
    return RG2_FINGERTIP_HEIGHT_CURVE_MM[-1][1]


def rg2_tcp_reference_z_offset_mm(width_mm):
    return abs(
        rg2_fingertip_height_mm(width_mm)
        - rg2_fingertip_height_mm(RG2_REFERENCE_OPEN_WIDTH_MM)
    )


def rg2_gripper_plan(grasp_width_mm=None):
    if grasp_width_mm is None:
        open_width = RG2_DEFAULT_OPEN_WIDTH_MM
        close_width = RG2_DEFAULT_CLOSE_WIDTH_MM
        effective_tip_width = RG2_REFERENCE_OPEN_WIDTH_MM
    else:
        open_width = clamp(
            grasp_width_mm + RG2_OPEN_CLEARANCE_MM,
            RG2_MIN_WIDTH_MM,
            RG2_MAX_WIDTH_MM,
        )
        close_width = clamp(
            grasp_width_mm - RG2_CLOSE_MARGIN_MM,
            RG2_MIN_WIDTH_MM,
            open_width,
        )
        effective_tip_width = grasp_width_mm

    return {
        "open_width_mm": float(open_width),
        "close_width_mm": float(close_width),
        "tcp_reference_z_offset_mm": float(rg2_tcp_reference_z_offset_mm(effective_tip_width)),
    }


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
        self.hostname = ROBOT_HOSTNAME
        self.rate = float(rate)
        self.nominal_dt = 1.0 / self.rate
        self.variables = list(RTDE_RECEIVE_VARIABLES)
        self.fixed_tcp_offset = list(FIXED_TCP_OFFSET)
        self.tcp_offset = list(FIXED_TCP_OFFSET)

        self._rtde_r = self._connect_receive()
        self.lock = threading.Lock()
        self.first_sample_ready = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self._last_receive_reconnect_s = 0.0

        self.pose = None
        self.tcp_speed = None
        self.wrench = None
        self.q = None
        self.qd = None

        self.thread.start()

    def _connect_receive(self):
        return RTDEReceiveInterface(
            self.hostname,
            self.rate,
            self.variables,
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
        return list(self.tcp_offset)

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
    def __init__(self, robot_state, rate=RTDE_CONTROL_FREQUENCY_HZ):
        self.hostname = ROBOT_HOSTNAME
        self.control_frequency_hz = float(rate)
        self.robot_state = robot_state
        self.fixed_tcp_offset = list(FIXED_TCP_OFFSET)
        self.tcp_offset = list(FIXED_TCP_OFFSET)
        self.safe_home_config = {
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
        logger.info(
            "Fixed TCP offset: %s",
            self._format_tcp_offset(self.tcp_offset),
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
        self._rtde_c = self._connect_control()

    def _connect_control(self):
        return RTDEControlInterface(self.hostname, self.control_frequency_hz)

    # Robot dynamics

    @staticmethod
    def rpy_to_rotvec(rpy_deg):
        r = Rotation.from_euler('xyz', rpy_deg, degrees=True)
        return r.as_rotvec()

    def build_pose(self, pose_xyz_rpy, nearest_to_current=True):
        pos_m = [c / 1000.0 for c in pose_xyz_rpy[:3]]  # mm to m
        rotvec = self.rpy_to_rotvec(pose_xyz_rpy[3:])
        if nearest_to_current:
            try:
                current_rotvec = self.robot_state.get_raw_pose()[3:6]
                rotvec = self.nearest_equivalent_rotvec(rotvec, current_rotvec)
            except Exception:
                pass
        return list(pos_m) + list(rotvec)

    @staticmethod
    def nearest_equivalent_rotvec(desired_rotvec, current_rotvec):
        desired = np.asarray(desired_rotvec, dtype=float).reshape(3)
        current = np.asarray(current_rotvec, dtype=float).reshape(3)
        theta = float(np.linalg.norm(desired))
        if theta < 1e-12:
            return desired

        axis = desired / theta
        candidates = [
            desired,
            desired + 2.0 * math.pi * axis,
            desired - 2.0 * math.pi * axis,
        ]
        return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - current)))

    def inverse_kin(self, pose_xyz_rpy):
        pose = self.build_pose(pose_xyz_rpy)
        qnear = self.robot_state.get_q()
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
        self.tcp_offset = list(self.fixed_tcp_offset)
        return list(self.tcp_offset)

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
        return np.asarray(
            self._rtde_c.getForwardKinematics(list(q), self.tcp_offset),
            dtype=float,
        )

    def camera_mount_pose_matrix_mm(self):
        """
        Return T_base_ee in millimeters for the camera mount.

        This deliberately uses ZERO_TCP_OFFSET, not the active RG2 picking TCP,
        so the gripper TCP offset is not applied twice when transforming the
        camera-frame object point.
        """
        q = self.robot_state.get_q()
        fk = np.asarray(
            self._rtde_c.getForwardKinematics(list(q), ZERO_TCP_OFFSET),
            dtype=float,
        )
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(fk[3:6]).as_matrix()
        transform[:3, 3] = fk[:3] * 1000.0
        return transform

    def camera_point_to_base_mm(self, point_camera_mm, T_base_ee=None):
        if T_base_ee is None:
            T_base_ee = self.camera_mount_pose_matrix_mm()
        point_camera = np.append(np.asarray(point_camera_mm, dtype=float).reshape(3), 1.0)
        point_base = np.asarray(T_base_ee, dtype=float).reshape(4, 4) @ T_EE_CAMERA @ point_camera
        return point_base[:3]

    def output_object_pose_base(
            self,
            registration_path="output/registered_point_cloud/cad_registration_result.json",
            output_path="output/robot_pose/object_pose_base.json",
    ):
        result = self.object_pose_base_from_registration(registration_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def object_pose_base_from_registration(self, registration_path):
        registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
        T_base_ee_mm = self.camera_mount_pose_matrix_mm()
        T_base_camera_mm = T_base_ee_mm @ T_EE_CAMERA

        T_camera_object_m = np.asarray(registration["T_observed_from_cad"], dtype=float).reshape(4, 4)
        T_camera_object_mm = T_camera_object_m.copy()
        T_camera_object_mm[:3, 3] *= 1000.0
        T_base_object_mm = T_base_camera_mm @ T_camera_object_mm

        center_camera_m = self.object_center_camera_m(registration["aligned_cad_cloud_path"])
        center_camera_mm = center_camera_m * 1000.0
        center_base_mm = (T_base_camera_mm @ np.array([*center_camera_mm, 1.0]))[:3]

        R_camera_object = T_camera_object_mm[:3, :3]
        R_base_object = T_base_object_mm[:3, :3]
        object_rpy_camera_deg = Rotation.from_matrix(R_camera_object).as_euler("xyz", degrees=True)
        object_rpy_base_deg = Rotation.from_matrix(R_base_object).as_euler("xyz", degrees=True)

        tcp_pick_rpy_base_deg = np.array([180.0, 0.0, object_rpy_base_deg[2]])
        desired_pick_rotvec = self.rpy_to_rotvec(tcp_pick_rpy_base_deg)
        current_rotvec = self.robot_state.get_raw_pose()[3:6]
        tcp_pick_rotvec = self.nearest_equivalent_rotvec(desired_pick_rotvec, current_rotvec)

        return {
            "object_center_camera_m": center_camera_m.tolist(),
            "object_center_camera_mm": center_camera_mm.tolist(),
            "object_center_base_mm": center_base_mm.tolist(),
            "object_rpy_camera_deg": object_rpy_camera_deg.tolist(),
            "object_rpy_base_deg": object_rpy_base_deg.tolist(),
            "object_pose_base_xyz_rpy_mm_deg": (
                np.concatenate([center_base_mm, object_rpy_base_deg]).tolist()
            ),
            "tcp_pick_rpy_base_deg": tcp_pick_rpy_base_deg.tolist(),
            "tcp_pick_rotvec_base_rad": tcp_pick_rotvec.tolist(),
            "tcp_pick_pose_base_xyz_rxryrz_mm_rad": (
                np.concatenate([center_base_mm, tcp_pick_rotvec]).tolist()
            ),
            "T_camera_from_object": T_camera_object_m.tolist(),
            "T_base_from_object_mm": T_base_object_mm.tolist(),
            "T_ee_from_camera_mm": T_EE_CAMERA.tolist(),
            "T_base_from_ee_zero_tcp_mm": T_base_ee_mm.tolist(),
            "T_base_from_camera_mm": T_base_camera_mm.tolist(),
        }

    @staticmethod
    def object_center_camera_m(cloud_path):
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        if cloud.is_empty():
            raise ValueError(f"Object cloud is empty: {cloud_path}")
        return np.asarray(cloud.get_axis_aligned_bounding_box().get_center(), dtype=float)

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
    # Utility
    def zero_ft_sensor(self):
        self._rtde_c.zeroFtSensor()

    def wait_until_motion_complete(self, timeout_s=None):
        timeout_s = self.motion_settle_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        stable = 0

        while time.monotonic() < deadline:
            tcp_speed = self.robot_state.get_tcp_speed()
            qd = self.robot_state.get_qd()
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

    # Move in cartesian coordinates
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
        config = self.safe_home_config
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
        getter = getattr(self.robot_state, "get_q", None)
        if not callable(getter):
            raise RuntimeError("Cannot execute safe home route because robot joint feedback is unavailable.")
        q = np.asarray(getter(), dtype=float).reshape(-1)
        if q.size != 6 or not np.all(np.isfinite(q)):
            raise RuntimeError("Current robot joint feedback is invalid.")
        return q

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

    def move_in_z(self, dz):
        p = self.robot_state.get_pose()
        p[2] += float(dz)
        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def move_to_cartesian(self, x, y, z, roll, pitch, yaw):
        p = [x, y, z, roll, pitch, yaw]
        pose = self.build_pose(p)
        self._rtde_c.moveL(pose, self.acc, self.vel, asynchronous=False)
        self.wait_until_motion_complete()

    def pick_object(
            self,
            point_camera_mm,
            object_rpy_base_deg=None,
            grasp_width_mm=None,
            object_height_mm=None,
            clearance_z_mm=RG2_PICK_CLEARANCE_Z_MM,
            open_force=RG2_PICK_OPEN_FORCE,
            close_force=RG2_PICK_CLOSE_FORCE,
            go_home_after=True,
    ):
        T_base_ee = self.camera_mount_pose_matrix_mm()
        x, y, z = self.camera_point_to_base_mm(point_camera_mm, T_base_ee)

        closed_grasp_z = float(z)
        if object_height_mm is not None:
            closed_grasp_z -= float(object_height_mm) / 2.0

        plan = rg2_gripper_plan(grasp_width_mm)
        pick_z = closed_grasp_z + plan["tcp_reference_z_offset_mm"]
        clearance_z_mm = float(clearance_z_mm)
        object_rpy_base_deg = (
            [180.0, 0.0, 0.0]
            if object_rpy_base_deg is None
            else list(np.asarray(object_rpy_base_deg, dtype=float).reshape(3))
        )
        pick_rpy = [180.0, 0.0, float(object_rpy_base_deg[2])]
        approach_pose = [float(x), float(y), pick_z + clearance_z_mm, *pick_rpy]

        self.gripper_set(plan["open_width_mm"], open_force)
        self.move_to_cartesian(*approach_pose)
        self.move_in_z(-clearance_z_mm)
        self.gripper_set(plan["close_width_mm"], close_force)
        self.move_in_z(clearance_z_mm)
        if go_home_after:
            self.go_home(open_gripper=False)

        return {
            "point_camera_mm": np.asarray(point_camera_mm, dtype=float).reshape(3).tolist(),
            "point_base_mm": [float(x), float(y), float(z)],
            "closed_grasp_z_mm": float(closed_grasp_z),
            "pick_z_mm": float(pick_z),
            "object_rpy_base_deg": object_rpy_base_deg,
            "pick_rpy_base_deg": pick_rpy,
            "approach_pose_xyz_rpy": approach_pose,
            "gripper_plan": plan,
            "T_base_ee_zero_tcp_mm": T_base_ee.tolist(),
        }

    def place_object(
            self,
            place_pose_xyz_rpy,
            open_width_mm=RG2_DEFAULT_OPEN_WIDTH_MM,
            open_force=RG2_PICK_OPEN_FORCE,
            clearance_z_mm=RG2_PICK_CLEARANCE_Z_MM,
            go_home_after=True,
    ):
        place_pose = list(np.asarray(place_pose_xyz_rpy, dtype=float).reshape(6))
        clearance_z_mm = float(clearance_z_mm)
        approach_pose = place_pose.copy()
        approach_pose[2] += clearance_z_mm

        self.move_to_cartesian(*approach_pose)
        self.move_in_z(-clearance_z_mm)
        self.gripper_set(open_width_mm, open_force)
        self.move_in_z(clearance_z_mm)
        if go_home_after:
            self.go_home(open_gripper=False)

        return {
            "place_pose_xyz_rpy": place_pose,
            "approach_pose_xyz_rpy": approach_pose,
            "open_width_mm": float(open_width_mm),
            "open_force": float(open_force),
            "clearance_z_mm": clearance_z_mm,
        }

    def report_pose(self, object_id=None, object_type=None, pose=None):
        return {
            "object_id": object_id,
            "object_type": object_type,
            "pose": pose,
        }
    

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

