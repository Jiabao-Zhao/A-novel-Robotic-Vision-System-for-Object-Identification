"""Native numerical parity on synthetic point clouds; no simulator identities."""
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial import cKDTree
from batched import native, refine_batch


class BatchTests(unittest.TestCase):
    def test_seedwise_stopping_and_pose_parity(self):
        rng = np.random.default_rng(318)
        points = rng.uniform([-.03, -.009, 0.], [.03, .009, .012], size=(800, 3))
        cad = SimpleNamespace(points=points, tree=cKDTree(points))
        for true_yaw in (0., .37, 1.2):
            rotation = native.axis_rotation([0., 0., 1.], true_yaw)
            observed = points[:170] @ rotation.T + [.11, -.05, .4]
            normal, basis = np.array([0., 0., 1.]), np.eye(3)[:, :2]
            seeds = []
            for yaw in (0., .1, .6, 1.4, 2.8, 4.2):
                pose = np.eye(4)
                pose[:3, :3] = native.axis_rotation(normal, yaw)
                pose[:3, 3] = [.10, -.06, .4]
                seeds.append(pose)
            poses, records = refine_batch(cad, observed, normal, basis, seeds)
            for i, seed in enumerate(seeds):
                reference_pose, reference = native.refine_pose(cad, observed, normal, basis, seed)
                np.testing.assert_allclose(poses[i], reference_pose, atol=1e-12, rtol=0.)
                self.assertEqual(records[i]['iterations'], reference['iterations'])
                self.assertEqual(records[i]['stop_reason'], reference['stop_reason'])
                np.testing.assert_allclose(records[i]['rmse_history_m'], reference['rmse_history_m'], atol=1e-12, rtol=0.)


if __name__ == '__main__':
    unittest.main()
