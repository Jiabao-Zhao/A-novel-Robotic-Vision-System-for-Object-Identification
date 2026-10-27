"""Synthetic supplementary-diagnostic checks; no data annotations opened."""
import unittest
import copy
import numpy as np
from plausibility import development_threshold, in_scope, overlay, summarize_overlay


class PlausibilityTests(unittest.TestCase):
    def test_maximum_negative_including_gripper_can_leave_zero_target_coverage(self):
        rows = [dict(bbox_target=True,score=.9), dict(bbox_target=True,score=None),
                dict(bbox_target=False,score=.8), dict(bbox_target=False,score=.95,kind='gripper')]
        result = development_threshold(rows)
        self.assertEqual(result['tau'], float(np.nextafter(.95, np.inf)))
        self.assertEqual(result['target_pairs_passing'], 0)
        self.assertEqual(result['target_coverage_all_pairs'], 0.)
        self.assertEqual(result['non_target_pairs_passing'], 0)
        self.assertEqual(result['unavailable_target_pairs'], 1)

    def test_threshold_is_not_clipped_when_a_negative_reaches_one(self):
        result = development_threshold([dict(bbox_target=False,score=1.), dict(bbox_target=True,score=1.)])
        self.assertGreater(result['tau'], 1.)
        self.assertEqual(result['target_pairs_passing'], 0)
        self.assertIsNone(development_threshold([dict(bbox_target=True,score=.9)])['tau'])

    def test_zero_one_multiple_and_unavailable_preserve_candidates(self):
        task = dict(status='ok',candidate_ids=[1,2,3])
        calibration = dict(status='available',tau=.8,known_targets=['part'])
        pairs = [dict(proposal_id=1,geometry_score=.8,status='available'),
                 dict(proposal_id=2,geometry_score=.7,status='available'),
                 dict(proposal_id=3,geometry_score=None,status='invalid_fit')]
        self.assertEqual(overlay(task,pairs,calibration)['flag'], 'one_passing_proposal')
        self.assertEqual(overlay(task,pairs,calibration)['unavailable_candidates'], [dict(proposal_id=3,status='invalid_fit')])
        pairs[1]['geometry_score'] = .9
        self.assertEqual(overlay(task,pairs,calibration)['flag'], 'multiple_passing_proposals')
        pairs[0]['geometry_score'] = .1; pairs[1]['geometry_score'] = .2
        self.assertEqual(overlay(task,pairs,calibration)['flag'], 'zero_passing_proposals')
        pairs[0]['geometry_score'] = None; pairs[1]['geometry_score'] = None
        self.assertEqual(overlay(task,pairs,calibration)['flag'], 'unavailable')
        self.assertEqual(overlay(dict(status='unavailable_cutoff_tie',candidate_ids=[]),pairs,calibration)['unavailable_reason'], 'unavailable_cutoff_tie')

    def test_scope_excludes_new_cads_real_data_and_other_arms(self):
        calibration = dict(known_targets=['known'])
        self.assertTrue(in_scope('simulation','clip_top5_depth','known',calibration))
        self.assertTrue(in_scope('depth_stress','clip_top5_depth','known',calibration))
        self.assertFalse(in_scope('new_cads','clip_top5_depth','known',calibration))
        self.assertFalse(in_scope('real_data','clip_top5_depth','known',calibration))
        self.assertFalse(in_scope('simulation','depth_all','known',calibration))
        self.assertFalse(in_scope('simulation','clip_top5_depth','new',calibration))

    def test_overlay_cannot_change_primary_rankings_or_inputs(self):
        from evaluate import arm_decisions
        task = dict(status='ok',candidate_ids=[1,2],scores=[[.9,1],[.8,2]])
        pairs = [dict(proposal_id=1,geometry_score=.3,size_score=.9,status='available'),
                 dict(proposal_id=2,geometry_score=.8,size_score=.5,status='available')]
        original = copy.deepcopy((task,pairs))
        before = arm_decisions(task,pairs)
        result = overlay(task,pairs,dict(status='available',tau=.9))
        self.assertEqual(result['flag'], 'zero_passing_proposals')
        self.assertEqual(arm_decisions(task,pairs), before)
        self.assertEqual((task,pairs), original)

    def test_diagnostic_summary_does_not_certify_absence_or_duplicates(self):
        rows = [dict(plausibility_eligible=True,plausibility=dict(flag='zero_passing_proposals'),
                     present=True,correct=True,plausibility_winner_passes=False),
                dict(plausibility_eligible=True,plausibility=dict(flag='multiple_passing_proposals'),
                     present=False,correct=False,plausibility_winner_passes=True)]
        result = summarize_overlay(rows)
        self.assertEqual(result['present_correct_winner_passing'], 0)
        self.assertEqual(result['absent_primary_winner_passing'], 1)
        self.assertEqual(result['certified_absence'], 0)
        self.assertEqual(result['certified_duplicate_detection'], 0)


if __name__ == '__main__':
    unittest.main()
