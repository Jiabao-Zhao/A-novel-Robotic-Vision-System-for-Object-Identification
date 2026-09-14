import unittest

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from CADPointCloudRegistration import CADPointCloudRegistration
from scripts.wrist_oracle_pose import fixed_pose_metrics, perspective_render
from scripts.run_wrist_oracle_pose_diagnostic import evaluate_pairs


class OraclePoseTests(unittest.TestCase):
    def test_known_plane_rays_preserve_depth_and_pixel_centers(self):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]),
            o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
        T = np.eye(4); T[2, 3] = 2.
        K = np.array([[100., 0, 2.], [0, 100., 2.], [0, 0, 1.]])
        rendered = perspective_render(mesh, T, K, (4, 4), np.array([.5, .5, .5]))
        self.assertTrue(rendered['mask'].all())
        np.testing.assert_allclose(rendered['depth_m'], 2., atol=1e-6)
        points = rendered['surface_points_camera_m']
        uv = (points @ K.T)[:, :2] / points[:, 2, None]
        y, x = np.indices((4, 4))
        np.testing.assert_allclose(uv, np.column_stack((x.ravel() + .5, y.ravel() + .5)), atol=1e-6)

    def test_transform_direction_and_fixed_pose_coverage(self):
        cad = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            [[0, 0, 0], [.02, 0, 0], [0, .03, 0], [0, 0, .04], [.01, .01, .01]]))
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [20, -30, 15], degrees=True).as_matrix()
        T[:3, 3] = [.14, -.05, .55]
        observed = o3d.geometry.PointCloud(cad).transform(T)
        metrics = fixed_pose_metrics(cad, observed, T, CADPointCloudRegistration())
        self.assertEqual(metrics['registration_fitness'], 1.)
        self.assertEqual(metrics['C_obs'], 1.)
        self.assertEqual(metrics['C_cad'], 1.)
        self.assertEqual(metrics['C_f1'], 1.)
        self.assertLess(metrics['observed_to_cad_rmse_m'], 1e-12)
        wrong = fixed_pose_metrics(cad, observed, np.linalg.inv(T), CADPointCloudRegistration())
        self.assertEqual(wrong['registration_fitness'], 0.)

    def test_geometry_ties_do_not_claim_positive_discrimination(self):
        pairs = [{'cad_id': 'red', 'object_id': oid, 'visual_raw': visual,
                  'full_cad': {'registration_fitness': 1.}}
                 for oid, visual in [('blue', .2), ('red', .9)]]
        result = evaluate_pairs(pairs, {'red': 'red'}, 'full_cad')
        target = result['per_target'][0]
        self.assertEqual(target['geometry']['margin'], 0.)
        self.assertEqual(target['geometry']['prediction'], 'blue')
        self.assertEqual(target['visual']['prediction'], 'red')
        self.assertEqual(target['fused']['prediction'], 'red')


if __name__ == '__main__':
    unittest.main()
