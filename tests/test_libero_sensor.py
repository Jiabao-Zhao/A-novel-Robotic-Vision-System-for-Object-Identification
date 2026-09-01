import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulation.libero_env import LiberoIntegrationError
from simulation.libero_io import save_libero_observation
from simulation.libero_sensor import LiberoRGBDSensor
from simulation.perception_adapter import mask_depth_to_world_workspace, pointcloud_localization_inputs


class _FakeEnvironment:
    instruction = "pick up the test object"
    sim = object()
    last_observation = None
    suite_name = "libero_object"
    task_index = 0
    task_name = "test_task"


class _FakeCameraUtils:
    @staticmethod
    def get_real_depth_map(sim, depth):
        return np.asarray(depth, dtype=np.float32) + 0.5

    @staticmethod
    def get_camera_intrinsic_matrix(sim, camera_name, height, width):
        return np.array([[100.0, 0.0, width / 2], [0.0, 101.0, height / 2], [0.0, 0.0, 1.0]])

    @staticmethod
    def get_camera_extrinsic_matrix(sim, camera_name):
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        return matrix


class LiberoSensorTests(unittest.TestCase):
    def setUp(self):
        self.raw_rgb = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        self.raw_depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        self.raw_observation = {
            "agentview_image": self.raw_rgb,
            "agentview_depth": self.raw_depth[:, :, None],
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        }

    def test_capture_converts_and_flips_rgbd_once(self):
        sensor = LiberoRGBDSensor(
            _FakeEnvironment(),
            camera_name="agentview",
            camera_utils=_FakeCameraUtils,
        )
        observation = sensor.capture(self.raw_observation)

        np.testing.assert_array_equal(observation.rgb, self.raw_rgb[::-1])
        np.testing.assert_allclose(observation.depth_m, (self.raw_depth + 0.5)[::-1])
        np.testing.assert_allclose(
            observation.world_T_camera @ observation.camera_T_world,
            np.eye(4),
        )
        self.assertIn("robot0_eef_pos", observation.robot_state)

    def test_adapter_uses_metric_float_depth(self):
        sensor = LiberoRGBDSensor(
            _FakeEnvironment(),
            camera_name="agentview",
            camera_utils=_FakeCameraUtils,
        )
        observation = sensor.capture(self.raw_observation)
        inputs = pointcloud_localization_inputs(observation, "rgb.png", "depth.npy")

        self.assertEqual(inputs["depth_scale_m"], 1.0)
        self.assertEqual(inputs["camera_intrinsics"]["width"], 2)
        self.assertEqual(inputs["camera_intrinsics"]["height"], 2)
        self.assertIs(inputs["rgb"], observation.rgb)
        self.assertIs(inputs["depth"], observation.depth_m)

    def test_world_workspace_mask_uses_calibrated_depth(self):
        sensor = LiberoRGBDSensor(
            _FakeEnvironment(),
            camera_name="agentview",
            camera_utils=_FakeCameraUtils,
        )
        observation = sensor.capture(self.raw_observation)
        masked_depth, mask = mask_depth_to_world_workspace(
            observation,
            minimum_xyz_m=(0.9, 1.9, 3.5),
            maximum_xyz_m=(1.1, 2.1, 4.0),
        )

        self.assertTrue(mask.all())
        np.testing.assert_array_equal(masked_depth, observation.depth_m)

    def test_invalid_normalized_depth_fails(self):
        invalid = dict(self.raw_observation)
        invalid["agentview_depth"] = np.full((2, 2, 1), 1.1, dtype=np.float32)
        sensor = LiberoRGBDSensor(
            _FakeEnvironment(),
            camera_name="agentview",
            camera_utils=_FakeCameraUtils,
        )

        with self.assertRaisesRegex(LiberoIntegrationError, "normalized to \\[0, 1\\]"):
            sensor.capture(invalid)

    def test_capture_files_exclude_object_ground_truth(self):
        environment = _FakeEnvironment()
        sensor = LiberoRGBDSensor(
            environment,
            camera_name="agentview",
            camera_utils=_FakeCameraUtils,
        )
        observation = sensor.capture(self.raw_observation)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = save_libero_observation(observation, environment, temporary_directory)
            metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))

            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            self.assertEqual(np.load(paths["depth"]).dtype, np.float32)
            self.assertEqual(metadata["depth_unit"], "meter")
            self.assertFalse(metadata["ground_truth_object_state_saved"])


if __name__ == "__main__":
    unittest.main()
