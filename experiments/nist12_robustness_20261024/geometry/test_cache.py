"""The exact cache omits only scene measurements unused by candidate scoring."""
import unittest
import numpy as np
from run_depth import observation_key, native


class CacheTests(unittest.TestCase):
    def test_outside_mask_depth_is_irrelevant_but_observed_depth_is_not(self):
        K = np.array([[200., 0., 15.], [0., 200., 10.], [0., 0., 1.]])
        plane = np.array([0., 0., -1., .6])
        depth = np.full((20, 30), .6, dtype=np.float32)
        mask = np.zeros(depth.shape, dtype=bool)
        mask[6:14, 10:20] = True
        depth[mask] = .58
        render = np.full(depth.shape, np.inf)
        render[5:15, 9:21] = .581
        indices = np.flatnonzero(mask)
        points = np.array([[.01, .02, .58], [.02, .03, .58]])
        original = observation_key(depth.shape, K, plane, points, indices, depth[mask])
        changed = depth.copy()
        changed[~mask] = .2
        self.assertEqual(original, observation_key(changed.shape, K, plane, points, indices, changed[mask]))
        self.assertEqual(native.score_projection(depth, mask, render), native.score_projection(changed, mask, render))
        changed[mask] += .001
        self.assertNotEqual(original, observation_key(changed.shape, K, plane, points, indices, changed[mask]))


if __name__ == '__main__':
    unittest.main()
