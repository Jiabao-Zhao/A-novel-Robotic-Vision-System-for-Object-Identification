"""Validation of unchanged inputs and probability intervals without API calls."""
import math
import unittest

from run_models import HERE, ROOT, SOURCE, read, digest, token_scores, extract, request_for
from analyze import count_pass, cell, auc_bounds


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=read(HERE/'manifest.json')

    def test_prompts_and_crops_unchanged(self):
        previous=read(SOURCE/'manifest.json')
        self.assertEqual(self.manifest['prompts'],previous['prompts'])
        self.assertEqual(self.manifest['crops'],previous['crops'])
        for path,sha in self.manifest['immutable_inputs'].items():
            self.assertEqual(digest(ROOT/path),sha,path)

    def test_requests_have_no_observation_identity(self):
        crop=self.manifest['crops']['state_101'][-2]
        for kind in ('gpt','local'):
            request,meta=request_for(kind,self.manifest,'Gear_Large',crop)
            if kind=='gpt':
                self.assertFalse(request['store'])
                self.assertEqual(request['reasoning']['effort'],'none')
                self.assertEqual(request['input'][1]['content'][0]['text'],self.manifest['prompts']['Gear_Large']['user_prompt'])
                self.assertEqual(set(request['input'][1]['content'][1]),{'type','detail','image_url'})
                self.assertNotIn('posthoc_identity',str(meta))
            else:
                self.assertEqual(request['messages'][1]['content'],self.manifest['prompts']['Gear_Large']['user_prompt'])
                self.assertFalse(request['think'])

    def test_probability_is_not_renormalized(self):
        entry=dict(token='B',logprob=math.log(.8),top_logprobs=[dict(token='A',logprob=math.log(.1)),dict(token='B',logprob=math.log(.8))])
        value=token_scores(entry)
        self.assertAlmostEqual(value['p_A'],.1)
        self.assertNotAlmostEqual(value['p_A'],.1/.9)

    def test_unknown_stays_unknown_with_rounding_allowance(self):
        value=token_scores(dict(token='B',logprob=0.0,top_logprobs=[dict(token='B',logprob=0.0)]))
        self.assertIsNone(value['p_A'])
        self.assertEqual(value['p_A_lower'],0)
        self.assertGreater(value['p_A_upper'],0)
        self.assertEqual(count_pass([value],.7),(0,0))
        self.assertEqual(cell(value),'<=0.001')

    def test_uncertain_gate_not_counted_as_rejection(self):
        row=dict(p_A=None,p_A_lower=0,p_A_upper=.75)
        self.assertEqual(count_pass([row],.7),(0,1))

    def test_auc_interval_contains_possible_orderings(self):
        rows=[dict(same_family=True,p_A=None,p_A_lower=.4,p_A_upper=.6),
              dict(same_family=False,p_A=.5)]
        self.assertEqual(auc_bounds(rows),(0,1))

    def test_saved_gpt_response_and_usage(self):
        paths=[p for p in (HERE/'gpt6-sol/responses').rglob('*.json') if not p.stem.endswith('_error')]
        self.assertEqual(len(paths),484)
        for path in paths:
            value=read(path)
            self.assertEqual(value['response']['model'],'gpt-6-sol')
            self.assertEqual(value['response']['service_tier'],'default')
            self.assertIn(extract('gpt',value['response'])['answer'],'ABC')

    def test_saved_local_responses_complete(self):
        paths=[p for p in (HERE/'gemma4-12b/responses').rglob('*.json') if not p.stem.endswith('_error')]
        self.assertEqual(len(paths),484)
        for path in paths:
            value=read(path)
            self.assertEqual(value['response']['model'],'gemma4:12b')
            score=extract('local',value['response'])
            self.assertIn(score['answer'],('A','B','C'))
            self.assertIsNotNone(score['p_A'])


if __name__=='__main__':
    unittest.main(verbosity=2)
