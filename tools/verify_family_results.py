"""Verify published family or features-only evidence offline, without inference."""
import hashlib
import json
import math
import re
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    root=Path(sys.argv[1]).resolve()
    def path(name):
        value=(root/name).resolve()
        assert value.is_relative_to(root.parent) and value.is_file(),name
        return value
    for name,sha in read(root/'SHA256SUMS.json').items():
        assert hashlib.sha256(path(name).read_bytes()).hexdigest()==sha,name
    prompts=read(root/'selected_prompts.json')
    manifest=read(root/'manifest.json')
    condition=manifest.get('condition','family_ablation')
    baseline_condition=manifest.get('baseline_condition','baseline')
    baseline_package=manifest.get('baseline_package','../vlm_four_scene_20260925')
    baseline_prompts=read(path(baseline_package+'/selected_prompts.json'))
    if condition=='features_only':
        forbidden=re.compile(r'\b(gears?|nuts?|pins?|USB|DSUB|connectors?|waterproof)\b',re.I)
        for prompt in prompts.values():
            text=prompt['system_prompt']+'\n'+prompt['user_prompt']
            assert not forbidden.search(text)
            assert all(target not in text for target in manifest['names'])
    rows=read(root/'all_scores_comparison.json')
    assert len(rows)==2904
    index={(r['condition'],r['model'],r['state'],r['target'],r['posthoc_identity']):r for r in rows}
    assert len(index)==2904
    assert {r['condition'] for r in rows}=={condition,baseline_condition}
    for row in rows:
        raw=read(path(row['response_file']));response=raw['response'];request=raw['request']
        assert response['model']==row['model'] and response['done']
        assert hashlib.sha256(path(row['crop_file']).read_bytes()).hexdigest()==row['crop_sha256']==raw['crop_sha256']
        expected=(prompts if row['condition']==condition else baseline_prompts)[row['target']]
        assert request['messages'][0]['content']==expected['system_prompt']
        assert request['messages'][1]['content']==expected['user_prompt']
        assert request['messages'][1]['images'][0]['sha256']==row['crop_sha256']
        assert response['message']['content'].strip()==row['answer'] and row['answer'] in ('A','B','C')
        logs=[v for v in response['logprobs'] if v['token'] in ('A','B','C')]
        assert len(logs)==1
        entry=logs[0];alternatives={v['token']:v['logprob'] for v in entry['top_logprobs']}
        alternatives[entry['token']]=entry['logprob']
        for letter in 'ABC':
            value=row['p_'+letter]
            if letter in alternatives:
                assert value is not None and math.isclose(value,math.exp(alternatives[letter]),abs_tol=1e-10)
            else:
                assert value is None
        if row['condition']==condition:
            old=index[baseline_condition,row['model'],row['state'],row['target'],row['posthoc_identity']]
            old_raw=read(path(old['response_file']))
            old_request=old_raw['request']
            if condition=='features_only' or 'model_digest' in old_raw:
                assert raw['model_digest']==old_raw['model_digest']
            assert row['crop_sha256']==old['crop_sha256']
            assert {k:v for k,v in request.items() if k!='messages'}=={k:v for k,v in old_request.items() if k!='messages'}
    for check in read(root/'threshold_comparison.json'):
        subset=[r for r in rows if r['condition']==check['condition'] and r['model']==check['model']
                and (check['state']=='all' or r['state']==check['state'])
                and (check['scope']=='all_crops' or not r['assisted_fallback'])]
        for category in ('family','unrelated','own'):
            group=[r for r in subset if (r['same_family'] if category=='family' else not r['same_family'] if category=='unrelated' else r['own_object_match'])]
            assert len(group)==check[category+'_total']
            for side in ('lower','upper'):
                n=sum((r['p_A'] if r['p_A'] is not None else r['p_A_'+side])>=check['threshold'] for r in group)
                assert n==check[category+'_retained_'+side]
    print('PASS: package hashes, 2904 paired records, crop hashes, exact prompts, unchanged controls, raw token scores, and threshold counts.')


if __name__=='__main__':
    main()
