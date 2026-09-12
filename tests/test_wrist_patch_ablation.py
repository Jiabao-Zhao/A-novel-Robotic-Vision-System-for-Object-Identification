import unittest
from unittest.mock import patch

import numpy as np
import torch

import cad_object_association as association
from scripts.run_wrist_patch_ablation import (
    evaluate_variant, extract_tokens, letterbox_mask, patch_occupancy, score_margins, score_pair,
)


class PatchMaskTests(unittest.TestCase):
    def test_mask_letterbox_matches_rgb_extent_with_zero_padding(self):
        mask = np.ones((20, 80), dtype=np.uint8)
        boxed = letterbox_mask(mask)
        rgb = association.letterbox_rgb(np.zeros((20, 80, 3), dtype=np.uint8))
        np.testing.assert_array_equal(boxed > 0, np.all(rgb == 0, axis=-1))
        self.assertEqual(boxed.sum(), 56 * 224)
        self.assertEqual(set(np.unique(boxed)), {0, 1})

    def test_patch_grid_has_row_major_positions_and_fractional_occupancy(self):
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[:7, :14] = 1
        mask[28:42, 42:56] = 1
        mask[223, 223] = 1
        occupancy = patch_occupancy(mask)
        self.assertEqual(occupancy.shape, (16, 16))
        self.assertEqual(occupancy[0, 0], .5)
        self.assertEqual(occupancy[2, 3], 1.)
        self.assertEqual(occupancy[15, 15], 1 / 196)
        self.assertEqual(np.count_nonzero(occupancy), 3)
        self.assertAlmostEqual(occupancy.sum() * 196, mask.sum())


class TokenExtractionTests(unittest.TestCase):
    def test_cls_matches_production_and_both_token_types_are_unit_normalized(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                self.batches = []

            def forward_features(self, batch):
                self.batches.append(batch.clone())
                cls = batch.mean(dim=(2, 3)).repeat(1, 128)
                patches = torch.zeros((len(batch), 256, 384))
                patches[:, :, 0] = torch.arange(1, 257)
                patches[:, :, 1] = 1
                return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}

            def forward(self, batch):
                return self.forward_features(batch)["x_norm_clstoken"]

        model = Encoder().eval()
        image = np.random.default_rng(4).integers(0, 256, (41, 29, 3), dtype=np.uint8)
        with patch.object(association, "load_dino_encoder", return_value=model):
            old_cls = association.extract_dino_features([image]).astype(float)
            cls, patches = extract_tokens([image])
        np.testing.assert_array_equal(cls, old_cls / np.linalg.norm(old_cls, axis=1, keepdims=True))
        torch.testing.assert_close(model.batches[0], model.batches[1], rtol=0, atol=0)
        np.testing.assert_allclose(np.linalg.norm(patches, axis=-1), 1, atol=1e-15)
        self.assertEqual(patches.shape, (1, 16, 16, 384))
        np.testing.assert_allclose(patches[0, 2, 3, :2], np.array([36, 1]) / np.sqrt(36**2 + 1))


class PatchMatchingTests(unittest.TestCase):
    def setUp(self):
        self.observed_cls = np.array([1., 0.])
        self.observed = np.zeros((16, 16, 2))
        self.observed[0, 0] = [1, 0]
        self.observed[0, 1] = [0, 1]
        self.weights = np.zeros((16, 16))
        self.weights[0, :2] = [.25, .75]
        self.cad_cls = np.tile([0., 1.], (14, 1))
        self.cad_cls[1] = [1, 0]
        self.cad = np.zeros((14, 16, 16, 2))
        self.cad[:, 0, 0] = [1, 0]
        self.cad[:, 0, 1] = [0, 1]
        self.cad_weights = np.zeros((14, 16, 16))
        self.cad_weights[:, 0, :2] = 1
        self.cad_weights[1, 0, 1] = 0  # Perfect second match is background in the CLS-selected view.

    def score(self):
        return score_pair(self.observed_cls, self.observed, self.weights,
                          self.cad_cls, self.cad, self.cad_weights)

    def test_cls_selects_view_and_patch_matching_excludes_background_and_weights_occupancy(self):
        result = self.score()
        self.assertEqual(result["selected_view_index_1based"], 2)
        self.assertEqual(len(result["cls_view_cosines"]), 14)
        self.assertEqual(result["cls_score"], 1)
        self.assertEqual(result["patch_score"], .25)
        self.assertEqual(result["cls_patch_score"], .625)
        self.assertEqual(result["observed_foreground_patch_count"], 2)
        self.assertEqual(result["selected_cad_foreground_patch_count"], 1)

    def test_many_observed_patches_can_match_the_same_cad_patch(self):
        self.observed[0, 1] = [1, 0]
        self.assertEqual(self.score()["patch_score"], 1)

    def test_negative_similarity_is_preserved(self):
        self.cad[1, 0, 0] = [-1, 0]
        self.assertEqual(self.score()["patch_score"], -.25)

    def test_missing_foreground_fails_instead_of_fabricating_a_score(self):
        self.weights[:] = 0
        with self.assertRaises(ValueError):
            self.score()

    def test_margin_uses_the_strongest_incorrect_candidate_and_preserves_sign(self):
        rows = [{"object_id": "correct", "score": .5}, {"object_id": "wrong_a", "score": .8},
                {"object_id": "wrong_b", "score": .1}]
        result = score_margins(rows, "correct", "score")
        self.assertAlmostEqual(result["correct_minus_best_incorrect"], -.3)
        self.assertEqual(result["strongest_incorrect_object_id"], "wrong_a")
        self.assertAlmostEqual(result["correct_minus_mean_incorrect"], .05)

    def test_each_visual_variant_uses_existing_fusion_normalization(self):
        pairs = [{"cad_id": "cad", "object_id": "object_001", "cls_score": -.5,
                  "patch_score": .5, "cls_patch_score": 0., "geometry_score": .8,
                  "runtime": {"cls_comparison_time_s": .01, "patch_matching_time_s": .02}},
                 {"cad_id": "cad", "object_id": "object_002", "cls_score": -.8,
                  "patch_score": -.8, "cls_patch_score": -.8, "geometry_score": 0.,
                  "runtime": {"cls_comparison_time_s": .01, "patch_matching_time_s": .02}}]
        timing = {"observed_image_loading_time_s": 0., "observed_joint_feature_extraction_time_s": .1,
                  "observed_mask_preparation_time_s": .01}
        audit = [{"object_id": "object_001", "simulator_instance": "cad", "status": "matched"}]
        for name, raw in (("cls_only", -.5), ("patch_only", .5), ("cls_patch", 0.)):
            result = evaluate_variant(name, pairs, {"cad": "target"}, audit, timing)
            winner = result["associations"][0]["candidate_ranking"][0]
            self.assertEqual(winner["visual_score"], (raw + 1) / 2)
            self.assertEqual(winner["fused_score"], .5 * ((raw + 1) / 2) + .5 * .8)
            self.assertEqual(result["accuracy"]["fused"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
