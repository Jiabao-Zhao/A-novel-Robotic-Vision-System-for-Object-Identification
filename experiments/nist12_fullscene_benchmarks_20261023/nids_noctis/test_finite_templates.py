"""CPU test of the secondary numerical guard, without model inference or GT."""

from pathlib import Path
import unittest

import torch

from run_finite_templates import ValidTemplateMetric, common


class DotProduct(torch.nn.Module):
    def forward(self, query, reference):
        return torch.einsum("bf,otf->bot", query, reference)


class GuardTest(unittest.TestCase):
    def test_invalid_view_exclusion_preserves_valid_values_and_all_classes(self):
        reference = torch.tensor([[[1., 0.], [float("nan"), float("nan")]],
                                  [[0., 1.], [.5, .5]]])
        query = torch.tensor([[1., 0.], [0., 1.]])
        valid = torch.isfinite(reference).all(-1)
        original = DotProduct()(query, reference)
        guard = ValidTemplateMetric(DotProduct(), valid)
        score = guard(query, reference)
        expanded = valid[None].expand_as(score)
        self.assertTrue(torch.equal(original[expanded], score[expanded]))
        self.assertTrue(torch.isneginf(score[~expanded]).all())
        self.assertEqual(score.max(-1).values.max(-1).indices.tolist(), [0, 1])
        self.assertEqual(tuple(score.shape), (2, 2, 2))

    def test_unexpected_nonfinite_valid_score_is_rejected(self):
        guard = ValidTemplateMetric(DotProduct(), torch.ones((1, 1), dtype=torch.bool))
        with self.assertRaises(AssertionError):
            guard(torch.tensor([[float("nan"), 1.]]), torch.ones((1, 1, 2)))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTest))
    common.save(Path(__file__).parent / "finite_templates_CPU_tests.json", dict(tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful(), labels_read=False,
        GPU_inference=False, source_sha256=common.digest(Path(__file__))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
