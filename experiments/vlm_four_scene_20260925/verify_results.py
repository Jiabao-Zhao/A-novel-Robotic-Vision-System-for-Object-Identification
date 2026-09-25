"""Offline audit of the published classification evidence. Python stdlib only."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(name):
    result=(ROOT/name).resolve()
    assert result.is_relative_to(ROOT.resolve()), name
    assert result.is_file(), name
    return result


def family(target):
    if target.startswith('Gear_'):
        return 'gear'
    if 'Hex_Nut' in target:
        return 'hex_nut'
    if target.startswith('RGOCG'):
        return 'round_pin'
    return target


def scores(response):
    if response['model']=='gpt-6-sol':
        assert response['status']=='completed'
        blocks=[c for o in response['output'] if o.get('type')=='message'
                for c in o['content'] if c.get('type')=='output_text']
        answer=''.join(c['text'] for c in blocks).strip()
        logs=[t for c in blocks for t in c.get('logprobs',[])]
    else:
        assert response['done'] and not response['message'].get('thinking')
        answer=response['message']['content'].strip()
        logs=response['logprobs']
    assert answer in ('A','B','C')
    labels=[v for v in logs if v['token'] in ('A','B','C')]
    assert len(labels)==1 and labels[0]['token']==answer
    entry=labels[0]
    alternatives={v['token']:v['logprob'] for v in entry['top_logprobs']}
    alternatives[entry['token']]=entry['logprob']
    assert all(math.isfinite(v) and v<=.0001 for v in alternatives.values())
    mass=sum(math.exp(min(0,v)) for v in alternatives.values())
    assert mass<=1.0001
    upper=min(1,max(0,1-mass)+.0001,min(math.exp(min(0,v)) for v in alternatives.values())+.0001)
    result={'answer':answer}
    for label in 'ABC':
        lp=alternatives.get(label)
        probability=math.exp(min(0,lp)) if lp is not None else None
        result['p_'+label]=probability
        result['p_'+label+'_lower']=probability if probability is not None else 0
        result['p_'+label+'_upper']=probability if probability is not None else upper
    return result


def main():
    hashes=read(ROOT/'SHA256SUMS.json')
    for name,sha in hashes.items():
        assert hashlib.sha256(resolve(name).read_bytes()).hexdigest()==sha,name
    manifest=read(ROOT/'manifest.json')
    prompts=read(ROOT/'selected_prompts.json')
    rows=read(ROOT/'data/all_model_scores.json')
    assert len(rows)==1936
    assert len({(r['model'],r['state'],r['target'],r['posthoc_identity']) for r in rows})==1936
    assert len(list((ROOT/'raw').rglob('*.json')))==1936
    with (ROOT/'data/all_model_scores.csv').open(encoding='utf-8-sig',newline='') as f:
        csvrows=list(csv.DictReader(f))
    assert len(csvrows)==1936
    for row,textrow in zip(rows,csvrows):
        for key in ('model','state','target','posthoc_identity','crop_file','response_file'):
            assert row[key]==textrow[key]
        if row['p_A'] is None:
            assert textrow['p_A']==''
        else:
            assert float(textrow['p_A'])==row['p_A']
        raw=read(resolve(row['response_file']))
        request=raw['request']
        assert raw['response']['model']==row['model']==request['model']
        assert (raw['state'],raw['target'],raw['identity'])==(row['state'],row['target'],row['posthoc_identity'])
        assert row['own_object_match']==(row['target']==row['posthoc_identity'])
        assert row['same_family']==(family(row['target'])==family(row['posthoc_identity']))
        crop=next(c for c in manifest['crops'][row['state']] if c['identity']==row['posthoc_identity'])
        assert crop['crop_file']==row['crop_file']
        assert crop['assisted_fallback']==row['assisted_fallback']
        assert crop['crop_sha256']==raw['crop_sha256']==row['crop_sha256']
        assert hashlib.sha256(resolve(row['crop_file']).read_bytes()).hexdigest()==row['crop_sha256']
        expected=prompts[row['target']]
        if row['model']=='gpt-6-sol':
            assert request['input'][0]['content']==expected['system_prompt']
            assert request['input'][1]['content'][0]['text']==expected['user_prompt']
            image=request['input'][1]['content'][1]['image_url']
            assert request['reasoning']['effort']=='none' and not request['store']
        else:
            assert request['messages'][0]['content']==expected['system_prompt']
            assert request['messages'][1]['content']==expected['user_prompt']
            image=request['messages'][1]['images'][0]
            assert not request['think']
        assert image['sha256']==row['crop_sha256']
        computed=scores(raw['response'])
        assert computed['answer']==row['answer']
        for label in 'ABC':
            value=row['p_'+label]
            if computed['p_'+label] is None:
                assert value is None
                assert math.isclose(row['p_'+label+'_upper'],computed['p_'+label+'_upper'],abs_tol=1e-10)
            else:
                assert value is not None and math.isclose(value,computed['p_'+label],abs_tol=1e-10)
    for model in manifest['models']:
        subset=[r for r in rows if r['model']==model]
        assert len(subset)==484 and sum(r['own_object_match'] for r in subset)==44
        assert sum(r['same_family'] for r in subset)==100
    def bounds(row):
        return (row['p_A'],row['p_A']) if row['p_A'] is not None else (row['p_A_lower'],row['p_A_upper'])
    checks=read(ROOT/'data/model_threshold_comparison.json')
    for check in checks:
        subset=[r for r in rows if r['model']==check['model']]
        assert check['state']=='all' and check['scope']=='all_crops'
        for key,flag in (('intended','own_object_match'),('family','same_family'),('unrelated',None)):
            group=[r for r in subset if r[flag]] if flag else [r for r in subset if not r['same_family']]
            assert len(group)==check[key+'_total']
            assert sum(bounds(r)[0]>=check['cutoff'] for r in group)==check[key+'_retained_min']
            assert sum(bounds(r)[1]>=check['cutoff'] for r in group)==check[key+'_retained_max']
        if check['cutoff'] in (.6,.7,.95,.98):
            print(check['model'],'cutoff',check['cutoff'],'own',check['intended_retained_min'],
                  'family',check['family_retained_min'],'unrelated',check['unrelated_retained_max'])
    print(f'PASS: {len(hashes)} file hashes; 1936 raw scores and prompts; 44 crops; all published aggregate threshold checks.')


if __name__=='__main__':
    main()
