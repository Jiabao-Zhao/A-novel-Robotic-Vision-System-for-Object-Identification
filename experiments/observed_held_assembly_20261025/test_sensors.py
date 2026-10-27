"""Focused camera-calibration and sensor-boundary checks; no physics steps."""
import unittest
from itertools import product

import mujoco
import numpy as np

from . import sensors


def fixture():
    links = []
    for index in range(1, 8):
        links.append(f'<body name="robot0_link{index}" pos="0 0 .025">'
                     f'<joint name="robot0_joint{index}" axis="0 1 0"/>'
                     '<geom type="capsule" fromto="0 0 0 0 0 .025" size=".008" group="1"/>')
    hand = ('<body name="gripper0_eef" pos="0 0 .025">'
            '<site name="gripper0_grip_site"/>'
            '<body name="gripper0_leftfinger" pos=".02 0 0">'
            '<joint name="gripper0_finger_joint1" type="slide" axis="1 0 0"/>'
            '<geom type="box" size=".005 .01 .02" group="1"/></body>'
            '<body name="gripper0_rightfinger" pos="-.02 0 0">'
            '<joint name="gripper0_finger_joint2" type="slide" axis="1 0 0"/>'
            '<geom type="box" size=".005 .01 .02" group="1"/></body></body>')
    xml = ('<mujoco><visual><global offwidth="1024" offheight="1024"/></visual>'
           '<worldbody><camera name="birdview"/><camera name="frontview"/>'
           '<camera name="sideview"/><geom type="plane" size="1 1 .01" group="1"/>'
           '<body name="ordinary_scene_item" pos=".05 .05 .03">'
           '<geom type="box" size=".03 .03 .03" group="1"/></body>'
           '<body name="robot0_base" pos="-.15 -.025 0">' + ''.join(links) + hand
           + '</body>' * 8 + '</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class SensorTests(unittest.TestCase):
    def test_calibration_round_trip(self):
        for name, _, position, lookat, fovy in sensors.CAMERAS:
            cal = sensors.calibration(position, lookat, fovy, name)
            pixels = np.array([[0., 0.], [200.25, 612.75], [1023., 1023.]])
            depth = np.array([.4, .7, 1.2])
            world = sensors.unproject_pixels(pixels, depth, cal)
            recovered, recovered_depth = sensors.project_world(world, cal)
            np.testing.assert_allclose(recovered, pixels, atol=1e-10)
            np.testing.assert_allclose(recovered_depth, depth, atol=1e-12)

    def test_capture_boundary_restoration_and_metric_depth(self):
        model, data = fixture()
        qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
        rgba, cameras = model.geom_rgba.copy(), model.cam_pos.copy()
        observation = sensors.capture_observation(model, data)
        self.assertEqual(set(observation), {"cameras", "robot_feedback"})
        self.assertEqual(set(observation["cameras"]), {camera[0] for camera in sensors.CAMERAS})
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        np.testing.assert_array_equal(model.geom_rgba, rgba)
        np.testing.assert_array_equal(model.cam_pos, cameras)
        self.assertEqual(data.time, time)
        self.assertTrue(any(np.isfinite(entry["robot_depth_m"]).any()
                            for entry in observation["cameras"].values()))
        for entry in observation["cameras"].values():
            self.assertEqual(set(entry), {"rgb", "depth_m", "robot_depth_m", "calibration"})
            self.assertEqual(entry["rgb"].shape, (1024, 1024, 3))
            self.assertTrue(np.isinf(entry["robot_depth_m"]).any())
            calibration = entry["calibration"]
            transform = np.asarray(calibration["world_T_camera"])
            intrinsics = np.asarray(calibration["intrinsics"])
            errors = []
            for row in (200, 400, 600, 800):
                for column in (200, 400, 600, 800):
                    ray = np.array([(column - intrinsics[0, 2]) / intrinsics[0, 0],
                                    (row - intrinsics[1, 2]) / intrinsics[1, 1], 1.])
                    norm = np.linalg.norm(ray)
                    distance = mujoco.mj_ray(model, data, transform[:3, 3], transform[:3, :3] @ (ray / norm),
                                             sensors.VISUAL_GROUPS, 1, -1, np.array([-1], np.int32))
                    if distance >= 0:
                        errors.append(abs(distance / norm - entry["depth_m"][row, column]))
            self.assertTrue(errors)
            self.assertLess(max(errors), 2e-4)

    def test_board_camera_covers_known_goals_and_160mm_standoff(self):
        camera = next(value for value in sensors.CAMERAS if value[0] == "board_inspection")
        name, _, position, lookat, fovy = camera
        cal = sensors.calibration(position, lookat, fovy, name)
        # User-approved nominal board goals and native CAD bounds, not product
        # measurements or simulator target states.
        goals = (([-.57055, -.419824, .043991603829], [.031, .031, .010]),
                 ([-.52055, -.419824, .043991603829], [.021, .021, .010]),
                 ([-.37055, -.419824, .035491600014], [.013857, .012, .0065]))
        for center, half_extent in goals:
            corners = np.asarray(list(product((-1., 1.), repeat=3))) * half_extent + center
            for rise in (0., .160):
                pixels, depth = sensors.project_world(corners + [0, 0, rise], cal)
                self.assertTrue(np.all(depth > 0))
                self.assertTrue(np.all(pixels >= 32))
                self.assertTrue(np.all(pixels < 1024 - 32))


if __name__ == "__main__":
    unittest.main()
