"""Frozen four-scene classification trial with explicitly assisted USB crops."""
import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATES = ('original', 'state_101', 'state_202', 'state_303')
OPTIONS = dict(temperature=0, seed=0, num_ctx=4096, num_predict=1,
               repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
               top_k=0, top_p=1.0, min_p=0.0)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    save(path.with_suffix('.json'), rows)
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def api(path, data=None):
    request = Request('http://127.0.0.1:11434' + path,
                      data=json.dumps(data).encode() if data is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def family(target):
    if target.startswith('Gear_'):
        return 'gear'
    if 'Hex_Nut' in target:
        return 'hex_nut'
    if target.startswith('RGOCG'):
        return 'round_pin'
    return target


def prepare():
    source = ROOT / 'classification-score-charts/local-model'
    prompts = read(source / 'selected_prompts.json')
    summaries = read(source / 'object_summary.json')
    names = {r['target']: r['target_name'] for r in summaries}
    assert set(names) == set(prompts) and len(names) == 11
    crops = {}
    inputs = {str((source / 'selected_prompts.json').relative_to(ROOT)):
              digest(source / 'selected_prompts.json')}
    for state in STATES:
        folder = ROOT / 'shadow-pin-usb-trial'
        if state != 'original':
            folder /= state
        audit_path = folder / 'proposal_audit.json'
        audit = read(audit_path)
        inputs[str(audit_path.relative_to(ROOT))] = digest(audit_path)
        scene_file = folder / 'scenes/baseline/wrist_rgb.png'
        inputs[str(scene_file.relative_to(ROOT))] = digest(scene_file)
        selected = []
        for item in audit:
            if item['posthoc_identity'] not in names:
                continue
            path = ROOT / item['crop_workspace_path']
            assert digest(path) == item['crop_sha256']
            selected.append(dict(identity=item['posthoc_identity'], proposal_id=item['proposal_id'],
                                 crop_file=str(path.relative_to(ROOT)), crop_sha256=digest(path),
                                 roi=item['roi'], assisted_fallback=False,
                                 localization_method='accepted_depth_rectangle'))
        if state in ('state_101', 'state_303'):
            number = 5 if state == 'state_101' else 2
            path = folder / f'usb_rejected_cluster_{number}.png'
            diagnostic = folder / 'rejected_usb_clusters.json'
            cluster = next(r for r in read(diagnostic)['clusters'] if r['raw_cluster_label'] == number)
            inputs[str(diagnostic.relative_to(ROOT))] = digest(diagnostic)
            selected.append(dict(identity='USB_Male', proposal_id=f'recovered_cluster_{number}',
                                 crop_file=str(path.relative_to(ROOT)), crop_sha256=digest(path),
                                 roi=cluster['roi'], assisted_fallback=True,
                                 localization_method='manually_verified_rejected_depth_cluster',
                                 rejection_reasons=cluster['rejection_reasons']))
        assert len(selected) == 11 and {r['identity'] for r in selected} == set(names)
        selected.sort(key=lambda r: list(names).index(r['identity']))
        for item in selected:
            inputs[item['crop_file']] = item['crop_sha256']
        crops[state] = selected
    manifest = dict(names=names, prompts=prompts, crops=crops, immutable_inputs=inputs,
                    scope='11 targets x 11 observed crops x 4 fixed scenes; no new render views',
                    expected_calls_per_model=484, options=OPTIONS, think=False,
                    labels=dict(A='match supported', B='mismatch supported', C='insufficient evidence'),
                    score='exp(raw log probability of token A), not normalized over A/B/C',
                    threshold_before_collection=None,
                    fallback_policy='USB in scenes 101/303: visually verified complete crop from a rejected depth cluster. Reference identity assisted cluster selection; not autonomous localization or SAM.',
                    exclusions=['M8_Hex_Nut', 'KET16_Square_16mm', 'KET12_Square_12mm', 'KET8_Square_8mm'],
                    ground_truth_policy='Identity and ROI metadata are audit-only, never sent to the VLM. Exclusions are also applied to candidate analysis.',
                    comparison_warning='Previous 4B chart combines saved scene revisions. A strict model comparison requires 4B scores on these exact crop bytes and prompts.')
    old = HERE / 'manifest.json'
    if old.exists():
        assert read(old) == manifest, 'Frozen manifest changed; inspect before resuming.'
    save(old, manifest)
    save(HERE / 'selected_prompts.json', prompts)
    print(json.dumps({'prepared': True, 'crops': {k: len(v) for k,v in crops.items()},
                      'calls_per_model': 484, 'assisted_USB_crops': 2}), flush=True)


def request_for(model, prompt, crop):
    return dict(model=model, stream=False, think=False, logprobs=True, top_logprobs=20,
                keep_alive='10m', options=OPTIONS,
                messages=[dict(role='system', content=prompt['system_prompt']),
                          dict(role='user', content=prompt['user_prompt'],
                               images=[base64.b64encode((ROOT / crop['crop_file']).read_bytes()).decode('ascii')])])


def probabilities(result):
    logs = result['logprobs']
    assert len(logs) == 1 and logs[0]['token'] in 'ABC'
    assert result['message']['content'] == logs[0]['token']
    assert not result['message'].get('thinking')
    alternatives = {r['token']: r['logprob'] for r in logs[0]['top_logprobs']}
    assert all(c in alternatives and alternatives[c] <= 0 for c in 'ABC'), 'Missing label likelihood, not zero.'
    scores = {f'p_{c}': math.exp(alternatives[c]) for c in 'ABC'}
    assert sum(scores.values()) <= 1.000001
    return dict(answer=logs[0]['token'], **scores,
                **{f'logp_{c}': alternatives[c] for c in 'ABC'},
                other_token_probability=1-sum(scores.values()))


def classify(model, limit):
    manifest = read(HERE / 'manifest.json')
    for name, sha in manifest['immutable_inputs'].items():
        assert digest(ROOT / name) == sha, name
    dest = HERE / model.split(':')[-1]
    runtime = dict(model=model, version=api('/api/version'), tags=api('/api/tags'),
                   show=api('/api/show', dict(model=model)), options=OPTIONS)
    model_info = next(r for r in runtime['tags']['models'] if r['name'] == model)
    runtime['model_digest'] = model_info['digest']
    if (dest / 'runtime.json').exists():
        assert read(dest / 'runtime.json')['model_digest'] == runtime['model_digest']
    save(dest / 'runtime.json', runtime)
    n = 0
    try:
        # Candidate-first order reuses the image encoding across target queries.
        for state in STATES:
            for crop in manifest['crops'][state]:
                for target, prompt in manifest['prompts'].items():
                    out = dest / 'responses' / state / target / f"{crop['identity']}.json"
                    if out.exists():
                        saved = read(out)
                        assert saved['crop_sha256'] == crop['crop_sha256']
                        probabilities(saved['response'])
                        continue
                    request = request_for(model, prompt, crop)
                    start = perf_counter()
                    result = api('/api/chat', request)
                    elapsed = perf_counter()-start
                    request['messages'][-1]['images'] = [dict(file=crop['crop_file'], sha256=crop['crop_sha256'])]
                    save(out, dict(state=state, target=target, identity=crop['identity'],
                                   crop_sha256=crop['crop_sha256'], model_digest=runtime['model_digest'],
                                   request=request, response=result, elapsed_seconds=elapsed))
                    scores = probabilities(result)
                    n += 1
                    if n == 1:
                        save(dest / 'loaded_runtime.json', api('/api/ps'))
                    print(json.dumps(dict(model=model, new_completed=n, scene=state,
                                          target=target, candidate=crop['identity'],
                                          p_A=round(scores['p_A'],6), answer=scores['answer'],
                                          seconds=round(elapsed,2))), flush=True)
                    if limit and n >= limit:
                        return
    finally:
        api('/api/generate', dict(model=model, keep_alive=0))


def collect(model):
    manifest = read(HERE / 'manifest.json')
    dest = HERE / model.split(':')[-1]
    rows = []
    for state in STATES:
        for crop in manifest['crops'][state]:
            for target in manifest['prompts']:
                path = dest / 'responses' / state / target / f"{crop['identity']}.json"
                saved = read(path)
                assert saved['crop_sha256'] == digest(ROOT / crop['crop_file'])
                rows.append(dict(model=model, state=state, target=target,
                                 target_name=manifest['names'][target], posthoc_identity=crop['identity'],
                                 own_object_match=target == crop['identity'],
                                 same_family=family(target) == family(crop['identity']),
                                 assisted_fallback=crop['assisted_fallback'],
                                 crop_file=crop['crop_file'], crop_sha256=crop['crop_sha256'],
                                 response_file=str(path.relative_to(ROOT)),
                                 elapsed_seconds=saved['elapsed_seconds'],
                                 **probabilities(saved['response'])))
    assert len(rows) == 484
    table(dest / 'all_scores', rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'classify', 'collect'])
    parser.add_argument('--model', default='qwen3.5:9b')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'classify':
        classify(args.model, args.limit)
    else:
        collect(args.model)
