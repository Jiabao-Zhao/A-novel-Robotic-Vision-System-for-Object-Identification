"""Offline standard-library verification; works on the extracted results archive."""
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(name):
    return ROOT / name.replace('\\', '/')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m = read(HERE / 'manifest.json')
    prior = read(ROOT / m['baseline_directory'] / 'manifest.json')
    assert m['crops'] == prior['crops'] and m['controls'] == prior['controls']
    rows = read(HERE / 'all_scores_comparison.json')
    assert len(rows) == 2904
    forbidden = re.compile(r'\b(gears?|nuts?|pins?|USB|DSUB|connectors?|waterproof)\b', re.I)
    for p in m['prompts'].values():
        text = p['system_prompt'] + '\n' + p['user_prompt']
        assert not forbidden.search(text)
        assert all(t not in text for t in m['names'])
    seen = set()
    for r in rows:
        key = tuple(r[k] for k in ('condition','model','state','target','posthoc_identity'))
        assert key not in seen
        seen.add(key)
        condition = m if r['condition'] == 'features_only' else prior
        raw = read(resolve(r['response_file']))
        assert digest(resolve(r['crop_file'])) == r['crop_sha256'] == raw['crop_sha256']
        expected = condition['controls'][r['model']]
        actual = raw['request']
        assert raw['model_digest'] == expected['expected_model_digest']
        assert {k:v for k,v in actual.items() if k != 'messages'} == {
            k:v for k,v in expected.items() if k not in ('messages','expected_model_digest','expected_runtime_version')}
        prompt = condition['prompts'][r['target']]
        assert actual['messages'][0]['content'] == prompt['system_prompt']
        assert actual['messages'][1]['content'] == prompt['user_prompt']
        assert actual['messages'][1]['images'] == [dict(file=r['crop_file'], sha256=r['crop_sha256'])]
        response = raw['response']
        assert response['done'] and not response['message'].get('thinking')
        assert response['message']['content'].strip() == r['answer'] in ('A','B','C')
        entries = [t for t in response['logprobs'] if t['token'] in ('A','B','C')]
        assert len(entries) == 1
        entry = entries[0]
        assert entry['token'] == r['answer']
        for token in response['logprobs']:
            if token is not entry:
                assert token['token'].strip() == '' or token['token'].startswith('<')
        probs = {t['token']: t['logprob'] for t in entry['top_logprobs']}
        probs[entry['token']] = entry['logprob']
        for c in 'ABC':
            if c in probs:
                assert r['p_' + c] is not None
                assert math.isclose(r['p_' + c], math.exp(min(0., probs[c])), rel_tol=1e-12, abs_tol=1e-15)
            else:
                assert r['p_' + c] is None
    validation = read(HERE / 'validation.json')
    for path, sha in validation['raw_response_hashes'].items():
        assert digest(resolve(path)) == sha
    for condition in ('family_association', 'features_only'):
        for model in m['models']:
            rs = [r for r in rows if r['condition'] == condition and r['model'] == model]
            assert len(rs) == 484
            assert sum(r['same_family'] for r in rs) == 100
            assert sum(r['own_object_match'] for r in rs) == 44
    for r in read(HERE / 'threshold_sweep.json'):
        rs = [s for s in rows if s['model'] == r['model'] and s['condition'] == r['condition']
              and (r['state'] == 'all' or s['state'] == r['state'])
              and (r['scope'] == 'all_crops' or not s['assisted_fallback'])]
        for name, selected in [('family', [s for s in rs if s['same_family']]),
                               ('unrelated', [s for s in rs if not s['same_family']]),
                               ('own', [s for s in rs if s['own_object_match']])]:
            assert len(selected) == r[name + '_total']
            for side in ('lower', 'upper'):
                n = sum((s['p_A'] if s['p_A'] is not None else s['p_A_' + side]) >= r['threshold'] for s in selected)
                assert n == r[name + '_retained_' + side]
    print('Verified 2,904 raw responses, 1,452 new comparisons, 44 crop hashes, controls, prompts, unnormalized probabilities and all aggregate threshold counts.')


if __name__ == '__main__':
    main()
