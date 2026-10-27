import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from CADPointCloudRegistration import CADPointCloudRegistration


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


class ConstrainedICPTests(unittest.TestCase):
    def setUp(self):
        self.registrar = CADPointCloudRegistration()
        rng = np.random.default_rng(11)
        xx, yy = np.meshgrid(np.arange(5) * .018, np.arange(4) * .021)
        self.points = np.column_stack((xx.ravel(), yy.ravel(), rng.uniform(0, .025, xx.size)))
        self.points[:, 2] -= self.points[:, 2].min()
        self.cad = cloud(self.points)
        normal = np.array([.3, -.4, np.sqrt(.75)])
        self.table = self.registrar.table_frame([*normal, -.35], cloud([normal * .4]))
        self.truth = np.eye(4)
        self.truth[:3, :3] = (
            self.registrar.rotation_about_axis(normal, np.deg2rad(36.237))
            @ self.registrar.rotation_between_vectors([0., 0., 1.], normal)
        )
        self.truth[:3, 3] = (.35 * normal + .13 * self.table['x_axis']
                            - .07 * self.table['y_axis'])
        # A partial observation: the hidden CAD points must not become residuals.
        self.observed = cloud(self.apply(self.points[::2], self.truth))

    @staticmethod
    def apply(points, transform):
        return points @ transform[:3, :3].T + transform[:3, 3]

    def test_recovers_non_grid_pose_from_partial_cloud_on_tilted_table(self):
        pivot = self.apply(self.points.mean(axis=0)[None], self.truth)[0]
        delta = self.registrar.rotation_about_point_transform(
            self.registrar.rotation_about_axis(self.table['normal'], np.deg2rad(1.373)), pivot)
        delta[:3, 3] += .001137 * self.table['x_axis'] - .000823 * self.table['y_axis']
        initial = delta @ self.truth
        fitted, score = self.registrar.refine_xy_yaw(self.cad, self.observed, self.table, initial)

        np.testing.assert_allclose(fitted, self.truth, atol=1e-10)
        np.testing.assert_allclose(fitted[:3, :3].T @ fitted[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted[:3, :3]), 1.)
        heights_before = self.apply(self.points, initial) @ self.table['normal'] + self.table['d']
        heights_after = self.apply(self.points, fitted) @ self.table['normal'] + self.table['d']
        np.testing.assert_allclose(heights_after, heights_before, atol=1e-12)
        self.assertAlmostEqual(heights_after.min(), 0.)
        self.assertLess(score['rmse_m'], 1e-10)
        self.assertTrue(np.all(np.diff(score['icp_rmse_history_m']) <= 0))
        self.assertEqual(score['rmse_m'], min(score['rmse_history_m']))
        self.assertAlmostEqual(score['rmse_m'], self.registrar.observed_to_cad_rmse(
            self.cad, self.observed, fitted), places=12)

    def test_cannot_correct_height_by_leaving_table(self):
        raised = cloud(np.asarray(self.observed.points) + .004 * self.table['normal'])
        fitted, score = self.registrar.refine_xy_yaw(self.cad, raised, self.table, self.truth)
        np.testing.assert_allclose(fitted, self.truth, atol=1e-10)
        self.assertAlmostEqual(score['rmse_m'], .004, places=10)

    def test_unobservable_yaw_keeps_orientation(self):
        single = cloud([[0., 0., 0.]])
        target = cloud([[.00117, -.00231, .003]])
        table = self.registrar.table_frame([0., 0., 1., 0.], target)
        fitted, score = self.registrar.refine_xy_yaw(single, target, table, np.eye(4))
        np.testing.assert_allclose(fitted[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(fitted[:3, 3], [.00117, -.00231, 0.], atol=1e-12)
        self.assertAlmostEqual(score['rmse_m'], .003)

    def test_all_216_hypotheses_are_ranked_by_unchanged_metric(self):
        best, records, warnings = self.registrar.find_best_tabletop_alignment(
            self.cad, self.observed, [*self.table['normal'], self.table['d']])
        self.assertFalse(warnings)
        self.assertEqual(len(records), 216)
        self.assertEqual(best['rmse_m'], min(row['rmse_m'] for row in records))
        for row in records:
            self.assertLessEqual(row['rmse_m'], row['score_breakdown']['initial_rmse_m'])
            self.assertTrue(np.all(np.diff(row['score_breakdown']['icp_rmse_history_m']) <= 0))
            self.assertEqual(row['rmse_m'], min(row['score_breakdown']['rmse_history_m']))
            self.assertAlmostEqual(row['rmse_m'], self.registrar.observed_to_cad_rmse(
                self.cad, self.observed, np.array(row['final_transform'])), places=12)
            heights = (self.apply(self.points, np.array(row['final_transform']))
                       @ self.table['normal'] + self.table['d'])
            self.assertAlmostEqual(heights.min(), 0., places=10)

    def test_empty_or_nonfinite_clouds_fail(self):
        for invalid in (cloud([]), cloud([[np.nan, 0., 0.]])):
            with self.assertRaisesRegex(ValueError, 'nonempty, finite'):
                self.registrar.refine_xy_yaw(self.cad, invalid, self.table, self.truth)

    def test_run_saves_transform_configuration_and_icp_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cad_path, observed_path = root / 'cad.pcd', root / 'observed.pcd'
            o3d.io.write_point_cloud(str(cad_path), self.cad)
            o3d.io.write_point_cloud(str(observed_path), self.observed)
            result = self.registrar.run(
                cad_path, observed_path, [*self.table['normal'], self.table['d']],
                {'scale_to_m': 1., 'cad_up_axis': 'Z', 'yaw_candidates_deg': [36.]},
                root / 'result')
            saved = json.loads(Path(result['result_path']).read_text())
            candidates = json.loads(Path(result['candidates_path']).read_text())
            self.assertEqual(len(candidates), 1)
            self.assertEqual(saved['registration_config']['constrained_iterations'], 12)
            self.assertIn('point-to-point ICP', saved['method'])
            self.assertIn('icp_rmse_history_m', saved['final_score_breakdown'])
            np.testing.assert_allclose(saved['T_observed_from_cad'], candidates[0]['final_transform'])


if __name__ == '__main__':
    unittest.main()
