"""Offline ranking and report regression checks; no models or dataset labels."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import analysis_helpers as helper


class AnalysisHelpers(unittest.TestCase):
    def test_ties_keep_every_candidate_without_a_winner(self):
        rank = helper.rank_candidates([{'proposal_id': 7, 'score': .8},
            {'proposal_id': 3, 'score': .8-5e-13}], expected_ids=[3, 7])
        self.assertEqual(rank['top_proposal_ids'], [7, 3])
        self.assertIsNone(rank['winner_proposal_id'])
        self.assertEqual(helper.outcome(rank, 7), 'tie')

    def test_missing_score_keeps_decision_unavailable(self):
        rank = helper.rank_candidates([{'proposal_id': 1, 'score': .9},
                                       {'proposal_id': 2, 'score': None}])
        self.assertFalse(rank['available'])
        self.assertEqual(helper.outcome(rank, 1), 'unavailable')

    def test_complete_unique_competitors_are_required(self):
        for candidates, expected in [([], None),
            ([{'proposal_id': 1, 'score': 1}], [1, 2]),
            ([{'proposal_id': 1, 'score': 1}, {'proposal_id': 1, 'score': 0}], None)]:
            with self.subTest(candidates=candidates), self.assertRaises(AssertionError):
                helper.rank_candidates(candidates, expected)

    def test_singleton_and_correctness_are_separate(self):
        rank = helper.rank_candidates([{'proposal_id': 4, 'score': 0}])
        self.assertIsNone(rank['gap'])
        self.assertTrue(rank['nonpositive_maximum'])
        self.assertEqual(helper.outcome(rank, 4), 'correct')
        self.assertEqual(helper.outcome(rank, 6), 'wrong')

    def test_json_and_csv_preserve_nested_fields(self):
        rows = [{'proposal_id': 2, 'views': [1, 3], 'detail': {'value': .5}}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            helper.export(folder, 'rows', rows)
            self.assertEqual(helper.read(folder/'rows.json'), rows)
            with (folder/'rows.csv').open(newline='', encoding='utf-8') as stream:
                record, = csv.DictReader(stream)
            self.assertEqual(json.loads(record['views']), [1, 3])
            self.assertEqual(json.loads(record['detail']), {'value': .5})

    def test_nonfinite_json_is_not_silently_written(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            helper.save(Path(directory)/'invalid.json', {'score': float('nan')})


if __name__ == '__main__':
    unittest.main()
