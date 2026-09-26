"""Focused validation of the family-association experiment contract."""
import math
import os
import unittest
from run_trial import HERE,ROOT,MODELS,read,request_for,prepare,family,previous,validate_inputs
from analyze import auc,count


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=read(HERE/'manifest.json')

    def test_baseline_inputs_preserved(self):
        validate_inputs(self.m)
        old=read(ROOT/'gpt6-gemma4-four-scene-trial/manifest.json')
        self.assertEqual(self.m['crops'],old['crops'])

    def test_labels_and_local_scope(self):
        self.assertEqual(set(self.m['labels']),set('ABC'))
        self.assertEqual(self.m['labels']['A'],'FAMILY MATCH SUPPORTED')
        self.assertTrue(self.m['local_only'])
        self.assertNotIn('gpt-6-sol',MODELS)

    def test_family_prompts_identical(self):
        for fam in ('gear','hex_nut','round_pin'):
            prompts=[p for t,p in self.m['prompts'].items() if family(t)==fam]
            self.assertTrue(all(p==prompts[0] for p in prompts))

    def test_controls_unchanged(self):
        crop=self.m['crops']['original'][0]
        for model,control in self.m['controls'].items():
            request=request_for(self.m,model,'Gear_Large',crop)
            expected={k:v for k,v in control.items() if k not in ('messages','expected_model_digest','expected_runtime_version')}
            self.assertEqual({k:v for k,v in request.items() if k!='messages'},expected)
            self.assertNotIn('Gear_Large',request['messages'][1]['content'])
            self.assertEqual(request['messages'][1]['images'],[dict(file=crop['crop_file'],sha256=crop['crop_sha256'])])

    def test_descriptions_preserved(self):
        self.assertEqual(self.m['descriptions']['gear'],['White circular gear with teeth around the outer edge','Central hole with raised hub'])
        self.assertEqual(self.m['descriptions']['hex_nut'],['Gray hexagonal nut with central through-hole','Six flat outer sides with no perimeter teeth'])
        self.assertEqual(self.m['descriptions']['round_pin'],['Long gray cylindrical pin','Smooth curved side with circular end faces'])

    def test_score_not_normalized(self):
        r=previous.token_scores(dict(token='B',logprob=math.log(.8),top_logprobs=[dict(token='A',logprob=math.log(.1)),dict(token='B',logprob=math.log(.8))]))
        self.assertAlmostEqual(r['p_A'],.1)
        self.assertNotAlmostEqual(r['p_A'],.1/.9)

    def test_auc_ties_and_unknown_bounds(self):
        self.assertEqual(auc([dict(same_family=True,p_A=.5),dict(same_family=False,p_A=.5)]),(.5,.5))
        self.assertEqual(auc([dict(same_family=True,p_A=None,p_A_lower=.4,p_A_upper=.6),dict(same_family=False,p_A=.5)]),(0,1))
        self.assertEqual(count([dict(p_A=None,p_A_lower=0,p_A_upper=.8)],.7),(0,1))

    @unittest.skipUnless(os.environ.get('CHECK_COMPLETE')=='1','Run after inference completes')
    def test_all_raw_results_complete_and_paired(self):
        baseline=read(HERE/'baseline_scores.json')
        index={(r['model'],r['state'],r['target'],r['posthoc_identity']):r for r in baseline}
        for model,slug in MODELS.items():
            files=list((HERE/slug/'responses').rglob('*.json'))
            self.assertEqual(len(files),484)
            for path in files:
                raw=read(path)
                old=index[model,raw['state'],raw['target'],raw['identity']]
                self.assertEqual(raw['crop_sha256'],old['crop_sha256'])
                self.assertEqual(raw['response']['model'],model)
                crop=next(c for c in self.m['crops'][raw['state']] if c['identity']==raw['identity'])
                self.assertEqual(raw['request'],request_for(self.m,model,raw['target'],crop))
                value=previous.extract('local',raw['response'])
                self.assertIn(value['answer'],('A','B','C'))
                if value['p_A'] is not None:
                    self.assertAlmostEqual(value['p_A'],math.exp(value['logp_A']))


if __name__=='__main__':
    unittest.main(verbosity=2)
