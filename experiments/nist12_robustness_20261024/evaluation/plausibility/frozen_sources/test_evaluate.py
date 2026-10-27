"""Synthetic regression tests only: no dataset annotation or GPU/model access."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import numpy as np
import evaluate
from evaluate import (arm_decisions, bbox_half_open, box_iou, cluster_ci,
                      decision_correct, pin_family, proposal_audit, query_state, rank_available,
                      paired_condition_effects, relative_nominal_pixel_group, diagnostic_identity,
                      hash_prediction_inventory)


class EvaluatorTests(unittest.TestCase):
    def test_box_conventions_match_one_pixel_and_rectangles(self):
        self.assertEqual(bbox_half_open([3, 4, 3, 4]), [3., 4., 4., 5.])
        self.assertEqual(box_iou(bbox_half_open([3, 4, 8, 9]),
                                 bbox_half_open([3, 4, 6, 6], 'bop_xywh')), 1.)
        self.assertEqual(box_iou([0, 0, 1, 1], [1, 0, 2, 1]), 0.)

    def test_null_nan_and_top_ties_remain_explicit(self):
        result = rank_available([(0, None), (1, float('nan')), (2, .7)])
        self.assertEqual(result['proposal_id'], 2)
        result = rank_available([(0, .5), (1, .5+5e-13), (2, None)])
        self.assertIsNone(result['proposal_id'])
        self.assertEqual(result['tied_ids'], [1, 0])
        self.assertEqual(rank_available([])['status'], 'unavailable_no_finite_scores')

    def test_top_five_restriction_and_invalid_distractor(self):
        task = dict(status='ok', candidate_ids=[1, 2], scores=[[.8, 1], [.7, 2], [.6, 3]])
        pairs = [dict(proposal_id=1, geometry_score=.6, size_score=.9),
                 dict(proposal_id=2, geometry_score=None, size_score=None),
                 dict(proposal_id=3, geometry_score=.99, size_score=.8)]
        result = arm_decisions(task, pairs)
        self.assertEqual(result['clip_top5_depth']['proposal_id'], 1)
        self.assertEqual(result['depth_all']['proposal_id'], 3)
        self.assertEqual(result['size_only']['proposal_id'], 1)

    def test_cutoff_tie_does_not_block_clip_only(self):
        task = dict(status='unavailable_cutoff_tie', candidate_ids=[], scores=[[.9, 0], [.7, 1]])
        result = arm_decisions(task, [dict(proposal_id=0, geometry_score=.9, size_score=.5)])
        self.assertEqual(result['clip_only']['proposal_id'], 0)
        self.assertEqual(result['clip_top5_depth']['status'], 'unavailable_cutoff_tie')

    def test_absence_and_unobservability_are_distinct(self):
        hidden = [dict(object_id=2, visible_pixels=0)]
        self.assertEqual(query_state(hidden, 1), ('absent', 0, 0))
        self.assertEqual(query_state(hidden, 2), ('present_unobservable', 1, 0))
        self.assertFalse(decision_correct(rank_available([]), [], 1))

    def test_duplicate_any_instance_correct_but_requires_clarification(self):
        objects = [dict(object_id=1, visible_pixels=20), dict(object_id=1, visible_pixels=30)]
        self.assertEqual(query_state(objects, 1), ('present_visible', 2, 2))
        self.assertTrue(decision_correct(dict(proposal_id=4),
            [dict(object_id=1, box_iou=.1), dict(object_id=1, box_iou=.6)], 1))

    def test_cluster_bootstrap_resamples_scenes_not_object_rows(self):
        rows = [dict(base_scene_id='a', correct=True, present=True)]*12
        rows += [dict(base_scene_id='b', correct=False, present=True)]*12
        self.assertEqual(cluster_ci(rows, repeats=200), [0., 1.])
        self.assertIsNone(cluster_ci([dict(base_scene_id='a', correct=False, present=False)]))

    def test_pin_family_has_precise_shape_names(self):
        self.assertEqual(pin_family('RGOCG16-50_16mm'), 'round')
        self.assertEqual(pin_family('KET16_Square_16mm'), 'rectangular')
        self.assertIsNone(pin_family('Gear_Large'))

    def test_hidden_instance_label_above_visible_label_max(self):
        instances = [dict(instance_id='hidden', object_id=1, target='part',
                          label_id=8, box=None, visible_pixels=0)]
        result = proposal_audit([dict(proposal_id=0, mask_index=0, bbox_xyxy=[0, 0, 1, 1])],
            np.ones((1, 2, 2), dtype=bool), instances, np.zeros((2, 2), dtype=np.uint16))
        self.assertEqual(result[0][0]['mask_iou'], 0.)
        self.assertEqual(result[0][0]['box_iou'], 0.)

    def test_paired_conditions_keep_fully_hidden_present_and_exclude_absent(self):
        common = dict(object_id=1, target='part', method='depth_all', proposal_id=0,
                      target_pixel_fraction_relative_nominal=1.)
        nominal = dict(common, scene_id='R00', base_scene_id='R00', cohort='simulation',
                       condition='nominal', present=True, correct=True, state='present_visible')
        hidden = dict(common, scene_id='R00_occ', base_scene_id='R00', cohort='simulation',
                      condition='occlusion_heavy', present=True, correct=False, state='present_unobservable')
        absent = dict(common, scene_id='R00_abs', base_scene_id='R00', cohort='simulation',
                      condition='absent', present=False, correct=False, state='absent')
        effects, summaries = paired_condition_effects([nominal, hidden, absent])
        by_condition = {r['condition']: r for r in summaries}
        self.assertEqual(by_condition['occlusion_heavy']['introduced'], 1)
        self.assertEqual(by_condition['occlusion_heavy']['accuracy_difference'], -1.)
        self.assertEqual(by_condition['absent']['excluded_absent_pairs'], 1)
        self.assertIsNone(by_condition['absent']['accuracy_difference'])
        absent_effect = next(r for r in effects if r['condition'] == 'absent')
        self.assertFalse(absent_effect['comparison_eligible'])
        self.assertIsNone(absent_effect['delta'])
        self.assertEqual(absent_effect['change'], 'not_comparable_not_present')
        self.assertEqual(relative_nominal_pixel_group(0., 0), 'fully_hidden')
        self.assertEqual(relative_nominal_pixel_group(.5, 5), '(0.25,0.5]')

    def test_mask_identity_ties_remain_unresolved_without_changing_primary(self):
        matches = [dict(instance_id='a', object_id=1, target='round8', mask_iou=.5, box_iou=.7),
                   dict(instance_id='b', object_id=2, target='round12', mask_iou=.5+5e-13, box_iou=.1)]
        result = diagnostic_identity(matches)
        self.assertIsNone(result['match'])
        self.assertEqual(result['status'], 'unresolved_max_mask_iou_tie')
        self.assertEqual([r['instance_id'] for r in result['tied_identities']], ['a', 'b'])
        self.assertTrue(decision_correct(dict(proposal_id=0), matches, 1))

    def test_unique_diagnostic_identity_and_zero_overlap(self):
        matches = [dict(instance_id='a', object_id=1, target='part', mask_iou=.3)]
        self.assertEqual(diagnostic_identity(matches)['match']['target'], 'part')
        matches.append(dict(instance_id='b', object_id=2, target='other', mask_iou=.1))
        for row in matches:
            row['mask_iou'] = 0.
        result = diagnostic_identity(matches)
        self.assertIsNone(result['match'])
        self.assertEqual(result['status'], 'unassigned_low_mask_overlap')
        self.assertEqual(len(result['tied_identities']), 2)

    def test_report_interval_fields_preserve_unavailable(self):
        from render_report import percentage_interval
        self.assertEqual(percentage_interval(None, None), 'N/A')
        self.assertEqual(percentage_interval(.1, [-.2, .3], ' pp'), '10.0 pp [-20.0, 30.0]')

    def test_hypothesis_bank_payload_tamper_is_rejected(self):
        with TemporaryDirectory() as folder:
            bank = Path(folder)/'bank.json.gz'
            bank.write_bytes(b'original synthetic bank')
            expected = evaluate.digest(bank)
            bank.write_bytes(b'tampered synthetic bank')
            with self.assertRaisesRegex(AssertionError, 'Hypothesis bank SHA256 mismatch'):
                hash_prediction_inventory([bank], {bank: expected})

    def test_repeated_bank_references_are_hashed_once_and_locked(self):
        with TemporaryDirectory() as folder:
            bank = Path(folder)/'bank.json.gz'
            bank.write_bytes(b'synthetic bank')
            expected = evaluate.digest(bank)
            with mock.patch.object(evaluate, 'digest', wraps=evaluate.digest) as hashed, \
                 mock.patch.object(evaluate, 'relative', side_effect=lambda file: Path(file).name):
                result = hash_prediction_inventory([bank, bank, bank.parent/'.'/bank.name], {bank: expected})
            self.assertEqual(hashed.call_count, 1)
            self.assertEqual(result, {'bank.json.gz': expected})


if __name__ == '__main__':
    unittest.main()
