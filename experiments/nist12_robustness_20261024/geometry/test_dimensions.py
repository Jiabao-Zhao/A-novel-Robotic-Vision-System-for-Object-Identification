"""Synthetic metric checks; no scene audit labels or models are opened."""
import unittest
import numpy as np
from dimensions import footprint_sides, score_dimensions


class DimensionTests(unittest.TestCase):
    def test_rotation_preserves_rectangular_metric_sides(self):
        corners = np.array([[0., 0.], [.05, 0.], [.05, .008], [0., .008]])
        for angle in (0., .17, 1.1, 2.3):
            c, s = np.cos(angle), np.sin(angle)
            points = corners @ np.array([[c, -s], [s, c]]) + [2., -1.]
            np.testing.assert_allclose(footprint_sides(points), [.05, .008], atol=1e-12)

    def test_collinear_points_remain_a_defined_partial_measurement(self):
        np.testing.assert_allclose(footprint_sides([[0., 0.], [.03, .04]]), [.05, 0.], atol=1e-12)

    def test_size_score_is_metric_and_unavailable_stays_explicit(self):
        observed = {'status': 'available', 'dimensions_m': [.05, .008, .008]}
        exact = [{'support_index': 0, 'dimensions_m': [.05, .008, .008]}]
        wrong = [{'support_index': 0, 'dimensions_m': [.05, .012, .012]}]
        self.assertEqual(score_dimensions(observed, exact)['score'], 1.)
        self.assertLess(score_dimensions(observed, wrong)['score'], 1.)
        self.assertIsNone(score_dimensions({'status': 'unavailable_observed_points'}, exact)['score'])


if __name__ == '__main__':
    unittest.main()
