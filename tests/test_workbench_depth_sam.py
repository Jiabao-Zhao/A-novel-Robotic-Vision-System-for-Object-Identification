import unittest

import numpy as np
import pytest

from scripts.run_workbench_depth_sam import (
    inverse_depth_rgb, paper_score, paper_visual_scores, select_candidates,
)


def test_changed_model_cannot_reuse_cached_workbench_masks(tmp_path, monkeypatch):
    from scripts import run_workbench_depth_sam as experiment

    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment, "model_provenance", lambda: {"model": "new SAM3 weights"})
    experiment.save_json(tmp_path / "sam_protocol.json", {"model": "old weights"})
    with pytest.raises(AssertionError, match="Frozen SAM3 generation protocol changed"):
        experiment.generate_proposals(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))


class WorkbenchDepthSAMTests(unittest.TestCase):
    def test_depth_preprocessing_preserves_invalid_pixels_and_near_contrast(self):
        image = inverse_depth_rgb(np.array([[0., .5, 1., 1.5, 2., np.nan]]))
        np.testing.assert_array_equal(image[..., 0], image[..., 1])
        np.testing.assert_array_equal(image[..., 0], image[..., 2])
        values = image[0, :, 0].astype(int)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 0)
        self.assertEqual(values[1], 255)
        self.assertEqual(values[4], 0)
        self.assertGreater(values[1] - values[2], values[3] - values[4])

    def test_no_identity_or_patch_override_of_cls_selected_view(self):
        cosines, view, appearance = paper_visual_scores(
            np.array([0., 1.]), np.array([1., 0.]), np.eye(2), np.eye(2))
        np.testing.assert_array_equal(cosines, [0., 1.])
        self.assertEqual(view, 1)
        self.assertEqual(appearance, 0.)

    def test_geometry_weight_is_bounded_by_one_third(self):
        self.assertEqual(paper_score(1., 1., 0., 0.), 1.)
        self.assertAlmostEqual(paper_score(1., 1., 0., 1.), 2 / 3)
        self.assertEqual(paper_score(.4, .4, .4, .2), .4)

    def test_early_exit_can_prefer_lower_scoring_depth_candidate(self):
        rows = [{"candidate_id": "rgb_001", "branch": "rgb", "score": .9},
                {"candidate_id": "depth_001", "branch": "depth", "score": .7}]
        selected, stopped = select_candidates(rows, "dual_depth_first_eo")
        self.assertTrue(stopped)
        self.assertEqual(selected[0]["candidate_id"], "depth_001")
        rows[1]["score"] = .69
        selected, stopped = select_candidates(rows, "dual_depth_first_eo")
        self.assertFalse(stopped)
        self.assertEqual(selected[0]["candidate_id"], "rgb_001")
        self.assertEqual(select_candidates([], "dual_depth_first_eo"), ([], False))


if __name__ == "__main__":
    unittest.main()
