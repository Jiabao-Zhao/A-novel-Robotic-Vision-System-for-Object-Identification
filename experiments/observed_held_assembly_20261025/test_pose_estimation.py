"""Synthetic sensor tests; known test poses never enter the estimator API."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from PIL import Image

from .pose_estimation import estimate_held_pose, estimate_table_pose, predict_held_pose
from . import estimate_from_files as worker_module
from .estimate_from_files import estimate_packet, PRIOR_SOURCE, PRIOR_SOURCES, checked_calibration
from scripts.surface_verification import CadSurface


def transform(translation, angles_deg=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    result[:3, 3] = translation
    return result


def observation(mesh, object_pose, camera_position, aim, size=180, focal=250.):
    forward = np.asarray(aim)-camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < .01:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    camera_pose[:3, 3] = camera_position
    K = np.array([[focal, 0., (size-1)/2], [0., focal, (size-1)/2], [0., 0., 1.]])
    render = CadSurface(mesh).render(np.linalg.inv(camera_pose) @ object_pose, K, (size, size), pixel_offset=0.)
    depth = np.full((size, size), 2., np.float64)
    x1, y1, x2, y2 = render["roi"]
    local = render["depth"]
    depth[y1:y2, x1:x2] = np.where(np.isfinite(local), local, 2.)
    return {"depth_m": depth, "robot_depth_m": np.full_like(depth, np.inf),
            "rgb": np.zeros((size, size, 3), np.uint8),
            "calibration": {"intrinsics": K.tolist(), "world_T_camera": camera_pose.tolist(),
                            "table_z_world_m": 0., "camera_name": "synthetic"}}


class ObservedPoseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)/"native.stl"
        self.mesh = o3d.geometry.TriangleMesh.create_box(.04, .028, .015)
        self.mesh.translate([-.02, -.014, -.0075])
        self.mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(self.path), self.mesh)

    def tearDown(self):
        self.folder.cleanup()

    def test_motion_prior_uses_only_rigid_hand_transform(self):
        before = transform([-.2, .1, .01], [0, 0, 20])
        grip_before = transform([-.2, .1, .02], [180, 0, 10])
        grip_after = transform([.05, .02, .3], [175, 3, 30])
        prior = predict_held_pose(before, grip_before, grip_after)
        np.testing.assert_allclose(np.linalg.inv(grip_after) @ prior,
                                   np.linalg.inv(grip_before) @ before, atol=1e-14)

    def test_held_multiview_corrects_translation_tilt_and_yaw(self):
        actual = transform([.07, -.02, .22], [7, -5, 34])
        prior = actual.copy()
        prior[:3, :3] = actual[:3, :3] @ Rotation.from_euler("xyz", [1, -2, -4.56], degrees=True).as_matrix()
        prior[:3, 3] += [.003, -.002, .002]
        views = [observation(self.mesh, actual, np.array(position), actual[:3, 3]) for position in
                 ((.07, -.02, .4), (.2, -.16, .34), (-.08, .1, .32))]
        fit = estimate_held_pose(self.path, views, prior)
        self.assertEqual(fit["status"], "available")
        estimated = np.asarray(fit["world_T_cad"])
        error_mm = np.linalg.norm(estimated[:3, 3]-actual[:3, 3])*1000
        error_deg = np.rad2deg(np.linalg.norm(Rotation.from_matrix(estimated[:3, :3] @ actual[:3, :3].T).as_rotvec()))
        self.assertLess(error_mm, .3)
        self.assertLess(error_deg, 1.)
        self.assertTrue(fit["diagnostics"]["local_six_parameter_support"])

    def test_robot_depth_removal_prevents_false_pose(self):
        actual = transform([0., 0., .2])
        view = observation(self.mesh, actual, np.array([0., 0., .4]), actual[:3, 3])
        view["robot_depth_m"] = view["depth_m"].copy()
        result = estimate_held_pose(self.path, [view], actual)
        self.assertEqual(result["status"], "unavailable_insufficient_nonrobot_depth")

    def test_rejects_target_mask_input(self):
        actual = transform([0., 0., .2])
        view = observation(self.mesh, actual, np.array([0., 0., .4]), actual[:3, 3])
        view["target_mask"] = np.ones(view["depth_m"].shape, bool)
        with self.assertRaises(ValueError):
            estimate_held_pose(self.path, [view], actual)

    def test_initial_table_detection_has_no_xy_or_yaw_seed(self):
        actual = transform([.04, -.02, .0075], [0, 0, 27])
        view = observation(self.mesh, actual, np.array([0., 0., .32]), np.array([0., 0., 0.]), size=240, focal=260.)
        distractor = o3d.geometry.TriangleMesh.create_box(.01, .01, .01)
        distractor.translate([-.005, -.005, -.005])
        other = observation(distractor, transform([-.055, .02, .005]), np.array([0., 0., .32]), np.zeros(3), size=240, focal=260.)
        view["depth_m"] = np.minimum(view["depth_m"], other["depth_m"])
        fit = estimate_table_pose(self.path, view)
        self.assertEqual(fit["status"], "available")
        estimated = np.asarray(fit["world_T_cad"])
        self.assertLess(np.linalg.norm(estimated[:3, 3]-actual[:3, 3]), .001)
        self.assertFalse(fit["diagnostics"]["prior_object_pose_used"])
        self.assertGreater(int(fit["observed_mask"].sum()), 100)

    def test_file_worker_hashes_only_declared_sensor_inputs(self):
        folder = Path(self.folder.name)
        actual = transform([.02, -.01, .2], [3, -2, 25])
        views = [observation(self.mesh, actual, np.array(position), actual[:3, 3]) for position in
                 ((.02, -.01, .4), (.17, -.15, .32))]
        records = []
        for index, view in enumerate(views):
            rgb, depth, robot, calibration = [folder/f"camera{index}_{suffix}" for suffix in
                                             ("rgb.png", "depth.npy", "robot.npy", "calibration.json")]
            Image.fromarray(view["rgb"]).save(rgb)
            np.save(depth, view["depth_m"])
            np.save(robot, view["robot_depth_m"])
            calibration.write_text(json.dumps(view["calibration"]))
            records.append({"rgb_path": str(rgb), "depth_m_path": str(depth),
                            "robot_depth_m_path": str(robot), "calibration_path": str(calibration)})
        packet = folder/"packet.json"
        packet.write_text(json.dumps({"mode": "held", "cad_path": str(self.path),
            "observations": records, "prior_world_T_cad": actual.tolist(), "prior_source": PRIOR_SOURCE}))
        output, fit = estimate_packet(packet)
        self.assertTrue(output.is_file())
        self.assertEqual(fit["status"], "available")
        self.assertEqual(len(fit["provenance"]["input_files_sha256"]), 10)
        self.assertFalse(fit["provenance"]["simulator_scene_trial_trajectory_result_files_read"])

    def test_file_calibration_rejects_nested_simulator_data(self):
        path = Path(self.folder.name)/"camera.json"
        path.write_text(json.dumps({"intrinsics": np.eye(3).tolist(), "world_T_camera": np.eye(4).tolist(),
                                    "audit": {"target_pose": np.eye(4).tolist()}}))
        with self.assertRaises(ValueError):
            checked_calibration(path)

    def test_worker_prior_source_allowlist(self):
        folder = Path(self.folder.name)
        actual = transform([0., 0., .2])
        view = observation(self.mesh, actual, np.array([0., 0., .4]), actual[:3, 3])
        rgb, depth, robot, calibration = [folder/suffix for suffix in
                                         ("rgb.png", "depth.npy", "robot.npy", "calibration.json")]
        Image.fromarray(view["rgb"]).save(rgb)
        np.save(depth, view["depth_m"])
        np.save(robot, view["robot_depth_m"])
        calibration.write_text(json.dumps(view["calibration"]))
        record = {"rgb_path": str(rgb), "depth_m_path": str(depth),
                  "robot_depth_m_path": str(robot), "calibration_path": str(calibration)}
        self.assertEqual(PRIOR_SOURCES, {"observed_initial_pose_transported_by_measured_robot_FK",
            "previous_observed_pose_transported_by_measured_robot_FK", "previous_observed_world_pose_after_release"})
        for index, prior_source in enumerate(sorted(PRIOR_SOURCES)):
            packet_folder = folder/f"allowed{index}"
            packet_folder.mkdir()
            packet = packet_folder/"packet.json"
            packet.write_text(json.dumps({"mode": "held", "cad_path": str(self.path),
                "observations": [record], "prior_world_T_cad": actual.tolist(), "prior_source": prior_source}))
            with patch.object(worker_module.pose_estimation, "estimate_held_pose",
                       return_value={"status": "test_only", "world_T_cad": None, "diagnostics": {}}) as estimator:
                _, result = estimate_packet(packet)
                self.assertEqual(result["provenance"]["prior_source"], prior_source)
                estimator.assert_called_once()
        rejected = folder/"rejected.json"
        rejected.write_text(json.dumps({"mode": "held", "cad_path": str(self.path),
            "observations": [record], "prior_world_T_cad": actual.tolist(), "prior_source": "simulator_target_pose"}))
        with self.assertRaises(ValueError):
            estimate_packet(rejected)


if __name__ == "__main__":
    unittest.main()
