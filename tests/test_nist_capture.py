"""Small offline regressions for the extracted NIST capture helpers."""
from dataclasses import dataclass
import itertools
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from simulation import nist_capture as capture


class NistCaptureTests(unittest.TestCase):
    def setUp(self):
        self.cube = np.array(list(itertools.product((-.01, .01), repeat=3)))

    def test_support_polygon_and_cube_center_use_metric_cad(self):
        center = capture.hull_center(self.cube, ConvexHull(self.cube))
        np.testing.assert_allclose(center, np.zeros(3), atol=1e-18)
        audit = capture.supported_pose(self.cube, np.eye(3), center)
        self.assertAlmostEqual(audit['support_area_m2'], .0004)
        self.assertAlmostEqual(audit['com_proxy_margin_m'], .01)
        self.assertIsNone(capture.supported_pose(self.cube, np.eye(3), np.array([.02, 0., 0.])))

    def test_gear_and_usb_rest_restrictions_remain(self):
        for identity, count in [('Gear_Large', 2), ('USB_Male', 1), ('KET16_Square_16mm', 6)]:
            options = capture.rest_options(self.cube, identity)
            self.assertEqual(len(options), count)
            for option in options:
                self.assertGreater(option['audit']['support_area_m2'], 0.)
                np.testing.assert_allclose(option['rotation'] @ option['rotation'].T, np.eye(3), atol=1e-14)

    def test_layout_is_repeatable_without_mixed_objects(self):
        root = ET.fromstring('<mujoco><worldbody><body name="nist_part_KET16_Square_16mm"/></worldbody></mujoco>')
        items = {'KET16_Square_16mm': dict(body='nist_part_KET16_Square_16mm', points=self.cube,
            base_rotation=np.eye(3), extent_m=[.02, .02, .02],
            options=capture.rest_options(self.cube, 'KET16_Square_16mm'))}
        first, poses = capture.layout(root, items, 31, 2026102207, 0)
        second, again = capture.layout(root, items, 31, 2026102207, 0)
        self.assertEqual(ET.tostring(first), ET.tostring(second))
        self.assertEqual(poses, again)
        self.assertAlmostEqual(poses['KET16_Square_16mm']['minimum_world_z_m'], .001)
        self.assertIsNone(root.find('worldbody/body').get('pos'))
        self.assertTrue(capture.separated(np.array([[0., 0.], [.01, .01]]),
                                         [np.array([[.033, 0.], [.04, .01]])]))
        self.assertFalse(capture.separated(np.array([[0., 0.], [.01, .01]]),
                                          [np.array([[.032, 0.], [.04, .01]])]))

    def test_mesh_vertices_keep_geom_rotation_and_translation(self):
        model = SimpleNamespace(geom_dataid=[0], mesh_vertadr=[0], mesh_vertnum=[2],
                                mesh_vert=np.array([[1., 2., 3.], [4., 5., 6.]]))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        data = SimpleNamespace(geom_xmat=[rotation.ravel()], geom_xpos=np.array([[.1, .2, .3]]))
        np.testing.assert_array_equal(capture.vertices(model, data, 0),
                                      model.mesh_vert @ rotation.T + data.geom_xpos[0])

    def test_localization_uses_inclusive_native_crop_and_labels_only_after_clustering(self):
        @dataclass
        class Item:
            roi: dict

        calls = []

        class Config:
            def __init__(self, **values):
                self.__dict__.update(values)

        class Localizer:
            def __init__(self, config):
                self.config = config

            def rgbd_to_pointcloud(self, rgb, depth, scale, intrinsics):
                calls.append((rgb.copy(), depth.copy(), scale, dict(intrinsics)))
                return SimpleNamespace(voxel_down_sample=lambda size: 'cloud')

            def segment_table_plane(self, cloud):
                return 'objects', [0., 0., 1., 0.], None

            def cluster_objects(self, objects, intrinsics, plane):
                return 'clusters'

            def localize_clusters(self, clusters, intrinsics):
                return [Item(dict(x1=2, y1=3, x2=5, y2=8))]

        fake_modules = dict(open3d=SimpleNamespace(utility=SimpleNamespace(
            random=SimpleNamespace(seed=lambda value: None))),
            point_cloud_localization=SimpleNamespace(PointCloudConfig=Config, PointCloudLocalization=Localizer))
        rgb = np.arange(12*10*3, dtype=np.uint8).reshape(12, 10, 3)
        obs = SimpleNamespace(rgb=rgb, depth_m=np.full((12, 10), .1, dtype=np.float32),
                              intrinsics=np.array([[100., 0., 5.], [0., 100., 6.], [0., 0., 1.]]),
                              world_T_camera=np.eye(4))
        seg = np.zeros((12, 10, 2), dtype=np.int32)
        seg[3:9, 2:6, 0] = 7
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            folder = Path(temporary)
            for state, identity in [('first', 'pin'), ('second', 'changed-audit-name')]:
                (folder / 'scenes' / state).mkdir(parents=True)
                truth = {identity: dict(geom_id=7, visible_pixels=24)}
                with patch.dict(sys.modules, fake_modules):
                    rows = capture.localize(state, obs, seg, truth, output_dir=folder, raw_dir=folder/'raw')
                self.assertEqual(rows[0]['native_crop_wh'], [4, 6])
                self.assertEqual(rows[0]['sent_crop_wh'], [4, 6])
                self.assertFalse(rows[0]['background_removed'])
                self.assertEqual(rows[0]['posthoc_identity'], identity)
                with Image.open(folder / rows[0]['crop_file']) as crop:
                    np.testing.assert_array_equal(np.asarray(crop), rgb[3:9, 2:6])
                audit = capture.read(folder/'scenes'/state/'localization_audit.json')
                self.assertEqual(audit['config'], capture.LOCALIZATION_PARAMETERS)
            for first, second in zip(calls[0][:2], calls[1][:2]):
                np.testing.assert_array_equal(first, second)
            self.assertEqual(calls[0][2:], calls[1][2:])


if __name__ == '__main__':
    unittest.main()
