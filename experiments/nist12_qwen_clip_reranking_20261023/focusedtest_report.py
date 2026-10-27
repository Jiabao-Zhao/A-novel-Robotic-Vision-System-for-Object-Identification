"""Small synthetic checks; never opens model outputs or audit identities."""
import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).with_name('analyze.py')
spec = importlib.util.spec_from_file_location('report_under_test', FILE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportDecisionTests(unittest.TestCase):
    def test_missing_competitor_prevents_unique_answer(self):
        result = report.rank([dict(id=0, score=.9, available=True), dict(id=1, score=None, available=False)])
        self.assertEqual(result['top_ids'], [])
        self.assertEqual(report.outcome(result, 0), 'unavailable')

    def test_identical_scores_remain_tied(self):
        result = report.rank([dict(id=0, score=.9, available=True), dict(id=1, score=.9, available=True)])
        self.assertEqual(result['top_ids'], [0, 1])
        self.assertEqual(report.outcome(result, 0), 'tied')

    def test_top5_boundary_tie_is_not_arbitrarily_resolved(self):
        rows = [dict(id=i, score=score, available=True) for i, score in enumerate([9, 8, 7, 6, 5, 5])]
        self.assertEqual(report.shortlist(rows)['status'], 'unavailable_ambiguous_top5')
        self.assertEqual(report.shortlist(rows)['candidate_ids'], [])

    def test_fixed_and_introduced_are_distinct_from_missing(self):
        self.assertEqual(report.paired_change('wrong', 'correct'), 'fixed_error')
        self.assertEqual(report.paired_change('correct', 'wrong'), 'introduced_error')
        self.assertEqual(report.paired_change('correct', 'unavailable'), 'lost_correct_to_unavailable')
        self.assertEqual(report.paired_change('tied', 'correct'), 'resolved_tied')

    def test_retained_dino_winner_cannot_change(self):
        rows = [dict(id=i, score=score, available=True) for i, score in enumerate([.1, .9, .3, .4, .5, .6])]
        base = report.rank(rows)
        restricted = report.rank([r for r in rows if r['id'] in [0, 1, 2, 3, 4]])
        self.assertEqual(base['top_ids'], restricted['top_ids'])

    def test_pin_family_does_not_conflate_round_and_rectangular(self):
        self.assertEqual(report.family('RGOCG8-50_8mm'), report.family('RGOCG16-50_16mm'))
        self.assertNotEqual(report.family('RGOCG8-50_8mm'), report.family('KET8_Square_8mm'))


if __name__ == '__main__':
    unittest.main()
