"""Focused checks of the NIDS BOP soft-mask pooling and maximum-cosine score."""

import unittest

import numpy as np

from nids_matching import _bilinear_weights, nids_appearance, nids_embedding, nids_match, nids_score


def vector(values):
    return {"status": "available", "embedding": values}


class NidsMatchingTest(unittest.TestCase):
    def test_bilinear_samples_match_torch_centers(self):
        mask = np.arange(224 * 224, dtype=np.float64).reshape(224, 224)
        actual = _bilinear_weights(mask, 16)
        expected = np.empty((16, 16))
        for y in range(16):
            for x in range(16):
                expected[y, x] = mask[6 + 14*y:8 + 14*y, 6 + 14*x:8 + 14*x].mean()
        np.testing.assert_array_equal(actual, expected)

    def test_soft_mask_not_thresholded(self):
        mask = np.zeros((224, 224))
        mask[6, 6] = 1
        weights = _bilinear_weights(mask, 16)
        self.assertEqual(weights[0, 0], 0.25)
        self.assertEqual(weights.sum(), 0.25)

    def test_average_raw_tokens_before_normalization(self):
        tokens = np.zeros((256, 2))
        tokens[0] = [10, 0]
        tokens[1] = [0, 1]
        mask = np.zeros((224, 224))
        mask[6:8, 6:8] = 1
        mask[6:8, 20:22] = 1
        result = nids_embedding(tokens, mask)
        np.testing.assert_allclose(result["embedding"], [5, .5])
        self.assertFalse(result["metadata"]["tokenwise_l2_normalization"])

    def test_weighted_average_retains_fractional_foreground(self):
        tokens = np.zeros((16, 16, 2))
        tokens[0, 0] = [1, 0]
        tokens[0, 1] = [0, 1]
        mask = np.zeros((224, 224))
        mask[6:8, 6:8] = 1
        mask[6, 20] = 1
        result = nids_embedding(tokens, mask)
        np.testing.assert_allclose(result["embedding"], [.8, .2])
        self.assertEqual(result["metadata"]["mask_weight_sum"], 1.25)

    def test_empty_mask_is_explicit_unavailable(self):
        result = nids_embedding(np.ones((256, 2)), np.zeros((224, 224)))
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["embedding"])

    def test_zero_average_is_explicit_unavailable(self):
        result = nids_embedding(np.zeros((256, 2)), np.ones((224, 224)))
        self.assertEqual(result["status"], "unavailable")

    def test_view_aggregation_is_maximum(self):
        result = nids_score(vector([1, 0]), [vector([1, 0]), vector([0, 1])])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["per_view_score"], [1.0, 0.0])
        self.assertEqual(result["best_view_index"], 0)

    def test_cosine_uses_vector_norms(self):
        result = nids_score(vector([3, 0]), [vector([4, 0]), vector([1, 1])])
        np.testing.assert_allclose(result["per_view_score"], [1, 1 / np.sqrt(2)])

    def test_negative_cosine_clamps_to_zero(self):
        result = nids_score(vector([1, 0]), [vector([-1, 0])])
        self.assertEqual(result["per_view_cosine"], [-1.0])
        self.assertEqual(result["score"], 0.0)

    def test_bad_template_keeps_explicit_unavailable(self):
        result = nids_score(vector([1, 0]), [{"status": "unavailable", "embedding": None}])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["unavailable_view_indices"], [0])

    def test_tied_views_are_recorded(self):
        result = nids_score(vector([1, 0]), [vector([1, 0]), vector([2, 0])])
        self.assertEqual(result["tied_best_view_indices"], [0, 1])

    def test_invalid_grid_raises(self):
        with self.assertRaises(ValueError):
            nids_embedding(np.zeros((255, 2)), np.ones((224, 224)))

    def test_mask_range_is_checked(self):
        with self.assertRaises(ValueError):
            nids_embedding(np.ones((256, 2)), np.full((224, 224), 255))

    def test_reference_appearance_uses_strict_majority(self):
        tokens = np.ones((256, 2))
        qmask = np.ones((224, 224))
        rmask = np.zeros((224, 224))
        rmask[:7, :14] = 1
        result = nids_appearance(tokens, qmask, tokens, rmask)
        self.assertEqual(result["status"], "unavailable")
        rmask[7, 0] = 1
        result = nids_appearance(tokens, qmask, tokens, rmask)
        self.assertEqual(result["reference_majority_foreground_patch_count"], 1)
        self.assertAlmostEqual(result["score"], 1.0, places=6)

    def test_appearance_is_directional(self):
        query = np.tile([1.0, 0.0], (256, 1))
        reference = np.tile([0.0, 1.0], (256, 1))
        reference[0] = [1, 0]
        mask = np.ones((224, 224))
        forward = nids_appearance(query, mask, reference, mask)
        reverse = nids_appearance(reference, mask, query, mask)
        self.assertAlmostEqual(forward["score"], 1, places=6)
        self.assertAlmostEqual(reverse["score"], 1 / 256, places=6)

    def test_full_score_adds_appearance_at_ffa_selected_view(self):
        mask = np.ones((224, 224))
        query = {"patch_tokens": np.tile([1.0, 0.0], (256, 1)), "mask224": mask}
        bad = {"patch_tokens": np.tile([0.0, 1.0], (256, 1)), "mask224": mask}
        result = nids_match(query, [bad, query])
        self.assertEqual(result["best_view_index"], 1)
        self.assertAlmostEqual(result["score"], 1, places=6)
        self.assertEqual(result["score"], (result["ffa_only_score"] + result["appearance_score"]) / 2)

    def test_soft_pooling_matches_direct_torch(self):
        import torch
        import torch.nn.functional as functional
        rng = np.random.default_rng(7)
        tokens = rng.normal(size=(256, 32)).astype(np.float32)
        mask = (rng.uniform(size=(224, 224)) > .4).astype(np.float32)
        actual = nids_embedding(tokens, mask)
        grid = torch.from_numpy(tokens).reshape(1, 16, 16, 32)
        weights = functional.interpolate(torch.from_numpy(mask)[None, None], size=(16, 16), mode="bilinear")
        pooled = (grid * weights.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / weights.sum()
        expected = pooled.numpy()[0]
        np.testing.assert_allclose(actual["embedding"], expected, atol=1e-7, rtol=1e-6)

    def test_appearance_matches_pinned_source_arithmetic(self):
        import torch
        import torch.nn.functional as functional
        rng = np.random.default_rng(9)
        rawquery = rng.normal(size=(256, 32)).astype(np.float32)
        rawreference = rng.normal(size=(256, 32)).astype(np.float32)
        querymask = (rng.uniform(size=(224, 224)) > .25).astype(np.float32)
        referencemask = (rng.uniform(size=(224, 224)) > .35).astype(np.float32)
        actual = nids_appearance(rawquery, querymask, rawreference, referencemask)
        grid = torch.from_numpy(rawquery).reshape(1, 16, 16, 32)
        softmask = functional.interpolate(torch.from_numpy(querymask)[None, None], size=(16, 16), mode="bilinear")
        q = functional.normalize((grid * softmask.permute(0, 2, 3, 1)).reshape(1, 256, -1), dim=-1)
        hardmask = functional.avg_pool2d(torch.from_numpy(referencemask)[None, None], kernel_size=14, stride=14).flatten(-2) > .5
        r = functional.normalize(torch.from_numpy(rawreference)[None] * hardmask.permute(0, 2, 1), dim=-1)
        sim = torch.matmul(q, r.permute(0, 2, 1)).max(dim=-1).values
        factor = torch.count_nonzero(q.sum(dim=-1), dim=-1) + 1e-6
        expected = (sim.sum(dim=-1) / factor).clamp(0, 1).item()
        self.assertAlmostEqual(actual["score"], expected, places=7)


if __name__ == "__main__":
    unittest.main()
