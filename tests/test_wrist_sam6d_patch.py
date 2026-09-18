import ast
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.sam6d_patch_matching import (
    appearance_score, combined_visual_score, masked_patch_descriptors, semantic_score,
)


CACHE = Path(__file__).resolve().parents[1] / "outputs/cache/sam6d_matching_source"


def upstream_scoring_classes():
    """Load only reviewed pure scoring classes, avoiding SAM/Lightning imports."""
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np}
    for filename, names in (("model/utils.py", {"BatchedData"}),
                            ("model/loss.py", {"PairwiseSimilarity", "MaskedPatch_MatrixSimilarity"})):
        path = CACHE / filename
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class SAM6DPatchTests(unittest.TestCase):
    def test_strict_mask_threshold_and_equal_patch_weight(self):
        tokens = np.zeros((256, 2)); tokens[:3] = [[1, 0], [0, 1], [-1, 0]]
        occupancy = np.zeros(256); occupancy[:3] = [1, .51, .5]
        query = masked_patch_descriptors(tokens, occupancy)
        reference = masked_patch_descriptors(tokens, np.r_[1., np.zeros(255)])
        self.assertEqual(int(torch.count_nonzero(query.sum(-1))), 2)
        result = appearance_score(query, reference)
        self.assertAlmostEqual(result["appearance_score"], .5, places=6)
        self.assertNotAlmostEqual(result["appearance_score"], 1 / 1.51, places=3)

    def test_zero_background_columns_and_empty_query(self):
        query = torch.zeros(256, 2); query[0] = torch.tensor([1., 0.])
        reference = torch.zeros_like(query); reference[0] = torch.tensor([-1., 0.])
        self.assertEqual(appearance_score(query, reference)["appearance_score"], 0.)
        result = appearance_score(torch.zeros_like(query), reference)
        self.assertEqual(result["appearance_score"], 0.)
        self.assertEqual(result["visible_ratio_diagnostic"], 0.)

    def test_global_top_five_clamping_and_single_best_view(self):
        cosines = np.array([-.8, .9, .8, .7, .6, -.5])
        cad = np.stack([cosines, np.sqrt(1 - cosines ** 2)], axis=1)
        result = semantic_score([1., 0.], cad)
        self.assertEqual(result["best_view_index_1based"], 2)
        self.assertAlmostEqual(result["semantic_score"], .6, places=6)
        self.assertAlmostEqual(combined_visual_score(result["semantic_score"], .2), .4, places=6)

    def test_upstream_count_uses_sum_not_norm(self):
        query = torch.zeros(256, 2); query[0] = torch.tensor([1., -1.])
        reference = query.clone()
        # Preserve this upstream edge case rather than silently replacing its denominator.
        self.assertEqual(appearance_score(query, reference)["observed_patch_count"], 0)
        self.assertEqual(appearance_score(query, reference)["appearance_score"], 1.)

    @unittest.skipUnless((CACHE / "model/loss.py").exists(), "Pinned upstream source not cached")
    def test_numerical_parity_with_actual_upstream_code(self):
        upstream = upstream_scoring_classes()
        metric = upstream["MaskedPatch_MatrixSimilarity"]()
        rng = np.random.default_rng(67)
        for dimension in (384, 1024):
            query = masked_patch_descriptors(rng.normal(size=(256, dimension)), rng.random(256))
            reference = masked_patch_descriptors(rng.normal(size=(256, dimension)), rng.random(256))
            result = appearance_score(query, reference)
            self.assertEqual(result["appearance_score"], float(metric.compute_straight(query[None], reference[None])[0]))
            self.assertEqual(result["visible_ratio_diagnostic"], float(metric.compute_visible_ratio(query[None], reference[None])[0]))
            q = torch.tensor(rng.normal(size=(1, dimension)), dtype=torch.float32)
            r = torch.tensor(rng.normal(size=(1, 42, dimension)), dtype=torch.float32)
            expected = upstream["PairwiseSimilarity"]()(q, r)[0, 0]
            actual = semantic_score(q[0], r[0])
            np.testing.assert_array_equal(actual["clamped_cls_view_scores"], expected.numpy())


if __name__ == "__main__":
    unittest.main()
