"""Focused preprocessing and ranking checks; no models, labels, or downloads."""
import unittest
import numpy as np
import run_dino as run


class Controls(unittest.TestCase):
    def test_inclusive_roi_and_native_plane_distance(self):
        depth = np.ones((4, 5))
        depth[1:3, 2:5] = [[1., 1.003, 0.], [1.004, float('nan'), 1.001]]
        mask = run.foreground(depth, np.eye(3), dict(x1=2, y1=1, x2=4, y2=2),
                              np.array([0., 0., 1., -1.]))
        np.testing.assert_array_equal(mask, [[False, True, False], [True, False, False]])

    def test_plane_normal_scaling_does_not_change_foreground(self):
        depth = np.array([[.999, 1.001, 1.005], [0., np.inf, np.nan]])
        roi = dict(x1=0, y1=0, x2=2, y2=1)
        first = run.foreground(depth, np.eye(3), roi, np.array([0., 0., 1., -1.]))
        second = run.foreground(depth, np.eye(3), roi, np.array([0., 0., 3., -3.]))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, [[False, False, True], [False, False, False]])

    def test_foreground_tight_crop_and_letterbox(self):
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        rgb[2:8, 3:15] = [30, 50, 70]
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 3:15] = True
        image, resized_mask, bounds = run.prepare(rgb, mask)
        self.assertEqual(bounds, [3, 2, 14, 7])
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(resized_mask.shape, (224, 224))
        self.assertTrue((image[:56] == 255).all())
        self.assertTrue((resized_mask[:56] == 0).all())
        self.assertTrue((resized_mask[56:168] == 1).all())
        with self.assertRaises(ValueError):
            run.prepare(rgb, np.zeros_like(mask))

    def test_ties_stay_tied_and_null_stays_unavailable(self):
        result = run.rank_candidates([dict(proposal_id=3, score=.8),
            dict(proposal_id=1, score=.8), dict(proposal_id=2, score=None)])
        self.assertEqual(result['top_proposal_ids'], [1, 3])
        self.assertIsNone(result['winner_proposal_id'])
        self.assertEqual(result['gap'], 0.)
        self.assertEqual(result['unavailable_ids'], [2])
        empty = run.rank_candidates([dict(proposal_id=2, score=None)])
        self.assertFalse(empty['available'])
        self.assertIsNone(empty['top_score'])

    def test_multiview_mean_and_real_best_view(self):
        query = [1., 0.]
        references = [[1., 0.], [0., 1.], [-1., 0.], [1., 1.], [1., 0.]]
        semantic = run.visual.semantic_score(query, references)
        expected = (1. + 0. + 0. + 1/np.sqrt(2) + 1.) / 5
        self.assertAlmostEqual(semantic['semantic_score'], expected, places=6)
        self.assertEqual(semantic['best_view_index_1based'], 1)
        self.assertAlmostEqual(run.visual.combined_visual_score(.6, .2), .4)


if __name__ == '__main__':
    unittest.main()
