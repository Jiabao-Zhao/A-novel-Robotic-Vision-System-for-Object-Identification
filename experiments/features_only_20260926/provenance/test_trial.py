"""Checks for the frozen prompts, request controls and comparison mathematics."""
import copy
import math
import unittest
from run_trial import HERE, SOURCE, MODELS, read, audit, request_for, runner, validate_inputs
from analyze import auc, count


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = read(HERE / 'manifest.json')

    def test_baseline_and_inputs_unchanged(self):
        validate_inputs(self.m)

    def test_no_category_names_and_identical_family_prompts(self):
        self.assertFalse(audit(self.m)['product_name_leakage'])

    def test_leakage_is_rejected(self):
        m = copy.deepcopy(self.m)
        m['prompts']['Gear_Large']['system_prompt'] += ' This is a gear.'
        with self.assertRaises(AssertionError):
            audit(m)

    def test_exact_model_controls_and_images(self):
        old = read(SOURCE / 'manifest.json')
        for model in MODELS:
            for crops in self.m['crops'].values():
                for crop in crops:
                    for target in self.m['names']:
                        req = request_for(self.m, model, target, crop)
                        prior = runner.request_for(old, model, target, crop)
                        self.assertEqual({k:v for k,v in req.items() if k != 'messages'},
                                         {k:v for k,v in prior.items() if k != 'messages'})
                        self.assertEqual(req['messages'][1]['images'], prior['messages'][1]['images'])

    def test_raw_probability_not_renormalized(self):
        response = dict(done=True, message=dict(content='A'), logprobs=[dict(token='A', logprob=math.log(.5),
            top_logprobs=[dict(token=c, logprob=math.log(p)) for c,p in zip('ABC', (.5,.2,.1))])])
        result = runner.previous.extract('local', response)
        self.assertAlmostEqual(result['p_A'], .5)
        self.assertAlmostEqual(sum(result['p_' + c] for c in 'ABC'), .8)

    def test_auc_ties_and_threshold_inclusive(self):
        rows = [dict(same_family=True, p_A=.7), dict(same_family=False, p_A=.7)]
        self.assertEqual(auc(rows), (.5,.5))
        self.assertEqual(count(rows, .7), (2,2))

    def test_auc_uncertainty_is_not_zero_imputation(self):
        rows = [dict(same_family=True, p_A=None, p_A_lower=0., p_A_upper=.1),
                dict(same_family=False, p_A=.05)]
        self.assertEqual(auc(rows), (0.,1.))

    def test_dataset_shape(self):
        rows = read(HERE / 'baseline_scores.json')
        for model in MODELS:
            rs = [r for r in rows if r['model'] == model]
            self.assertEqual(len(rs), 484)
            self.assertEqual(sum(r['same_family'] for r in rs), 100)
            self.assertEqual(sum(r['own_object_match'] for r in rs), 44)
            self.assertEqual(sum(r['assisted_fallback'] for r in rs), 22)


if __name__ == '__main__':
    unittest.main()
