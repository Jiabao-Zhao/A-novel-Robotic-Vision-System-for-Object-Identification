"""Synthetic regression checks only; no captured labels or model outputs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from evaluate_native import (edges, iou, select, physical_identity, failure,
                             conditional_boxes, pin_confusion, cluster_ci)


class EvaluationTests(unittest.TestCase):
    def test_native_box_conventions(self):
        self.assertEqual(edges([0,0,9,9], 'cnos'), [0.,0.,10.,10.])
        self.assertEqual(edges([0,0,10,10], 'nemo'), [0.,0.,10.,10.])
        self.assertEqual(edges([0,0,10,10], 'noctis'), [0.,0.,10.,10.])
        self.assertEqual(iou([0,0,10,10], [10,0,20,10]), 0.)
        self.assertAlmostEqual(iou([0,0,10,10], [0,0,20,10]), .5)

    def test_requested_identity_and_exact_ties(self):
        rows = [dict(object_id=1, proposal_id=3, score=.2), dict(object_id=2, proposal_id=4, score=.9),
                dict(object_id=1, proposal_id=5, score=.4)]
        chosen, status = select(rows, 1)
        self.assertEqual((chosen['proposal_id'], status), (5, 'ok'))
        self.assertEqual(select(rows, 3), (None, 'no_retained_detection'))
        self.assertEqual(select(rows+[dict(object_id=1, proposal_id=6, score=.4)], 1),
                         (None, 'unavailable_top_score_tie'))

    def test_mask_identity_uses_full_union(self):
        labels = np.array([[0,1,1], [0,2,2]])
        mask = np.array([[0,1,0], [0,1,1]], bool)
        selected, maximum, overlaps = physical_identity(mask, labels,
            [dict(label_id=1,target='a'), dict(label_id=2,target='b')])
        self.assertEqual((selected, maximum), ('b', 2/3))
        self.assertEqual(overlaps['a'], 1/4)

    def test_stage_attribution_and_unavailable(self):
        self.assertEqual(failure(False, 'no_retained_detection', False, False, None, 'x'), 'own_proposal_localization_miss')
        self.assertEqual(failure(False, 'no_retained_detection', True, False, None, 'x'), 'native_size_filter_miss')
        self.assertEqual(failure(None, 'unavailable_reference_features', None, None, None, 'x'), 'method_reference_features_invalid')
        self.assertEqual(failure(False, 'no_retained_detection', None, None, None, 'x', True, False),
                         'conditional_component_size_filter_miss')

    def test_nemo_exact_native_component_area_boundary(self):
        mask = np.zeros((80, 100), bool)
        mask[2:34, 2:34] = True
        mask[45:76, 50:81] = True
        with TemporaryDirectory() as folder:
            np.savez_compressed(Path(folder)/'raw_obj_000001.npz', amodal_mask=mask)
            before, after = conditional_boxes(Path(folder), 1)
        self.assertEqual(before, [[2,2,34,34], [50,45,81,76]])
        self.assertEqual(after, [[2,2,34,34]])

    def test_primary_bbox_correctness_prevents_diagnostic_pin_error(self):
        target, other = 'RGOCG8-50_8mm', 'RGOCG12-50_12mm'
        self.assertIsNone(pin_confusion(True, target, other))
        self.assertEqual(pin_confusion(False, target, other), 'same_shape_wrong_size')
        self.assertEqual(pin_confusion(False, target, 'KET8_Square_8mm'), 'round_rectangular_confusion')

    def test_paired_bootstrap_resamples_layouts_not_object_rows(self):
        rows = [dict(base_scene_id='a', correct=True, delta_vs_ours=1, eligible=True)]*12
        rows += [dict(base_scene_id='b', correct=False, delta_vs_ours=-1, eligible=True)]*12
        self.assertEqual(cluster_ci(rows, repeats=200), [0., 1.])
        self.assertEqual(cluster_ci(rows, value='delta_vs_ours', repeats=200), [-1., 1.])
        self.assertIsNone(cluster_ci([dict(base_scene_id='a', correct=None, delta_vs_ours=None, eligible=False)]))


if __name__ == '__main__':
    unittest.main()
