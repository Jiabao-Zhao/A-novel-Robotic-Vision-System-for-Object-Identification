"""Two additional models evaluated on the unchanged 484-pair dataset."""
import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'qwen9b-four-scene-trial'
spec = importlib.util.spec_from_file_location('qwen_trial_base', SOURCE / 'run_trial.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
read, save, digest, table = base.read, base.save, base.digest, base.table
MODELS = {'gpt': 'gpt-6-sol', 'local': 'gemma4:12b'}
FOLDERS = {'gpt': 'gpt6-sol', 'local': 'gemma4-12b'}


def prepare():
    manifest = read(SOURCE / 'manifest.json')
    manifest['source_manifest'] = str((SOURCE / 'manifest.json').relative_to(ROOT))
    manifest['source_manifest_sha256'] = digest(SOURCE / 'manifest.json')
    manifest['new_models'] = MODELS
    manifest['model_storage'] = 'D:/AI/models'
    manifest['comparison_warning'] = 'Same image bytes and prompt text for every model. Native tokenizers, chat templates, image preprocessing and supported inference controls differ. Existing 4B/9B results are reused without selecting per-scene maxima.'
    manifest['gpt_options'] = dict(reasoning={'effort':'none'}, temperature=0,
                                   max_output_tokens=16, image_detail='high', top_logprobs=20,
                                   store=False, service_tier='default')
    manifest['local_options'] = {**base.OPTIONS, 'num_predict':16}
    manifest['local_output_budget_note'] = '16 tokens allows native end/control tokens; score the sole A/B/C label token. No explanation, nonempty thinking, or extra visible text allowed.'
    manifest['missing_likelihood_policy'] = 'Unknown exact probabilities stay null; lower=0, upper=min(unreported probability mass + 0.0001 rounding allowance, smallest reported top-token probability + 0.0001). No score fabricated as zero.'
    manifest['reference_urls'] = ['https://developers.openai.com/api/docs/models/gpt-6-sol',
                                  'https://ollama.com/library/gemma4:12b',
                                  'https://ai.google.dev/gemma/docs/core/model_card_4']
    if (HERE / 'manifest.json').exists():
        assert read(HERE / 'manifest.json') == manifest
    save(HERE / 'manifest.json', manifest)
    save(HERE / 'selected_prompts.json', manifest['prompts'])
    print('Prepared 484 unchanged target/crop pairs per model.', flush=True)


def token_scores(entry):
    alternatives = {v['token']: v['logprob'] for v in entry['top_logprobs']}
    alternatives[entry['token']] = entry['logprob']
    assert all(math.isfinite(lp) and lp <= 1e-4 for lp in alternatives.values())
    mass = sum(math.exp(min(0,lp)) for lp in alternatives.values())
    assert mass <= 1.0001, 'Invalid probability distribution.'
    # API log probabilities can be rounded to zero for near-certain tokens.
    upper = min(1.0, max(0.0, 1.0-mass)+1e-4,
                min(math.exp(min(0,lp)) for lp in alternatives.values())+1e-4)
    values = {}
    for label in 'ABC':
        lp = alternatives.get(label)
        p = math.exp(min(0,lp)) if lp is not None else None
        values.update({f'logp_{label}':lp, f'p_{label}':p,
                       f'p_{label}_lower':p if p is not None else 0.0,
                       f'p_{label}_upper':p if p is not None else upper})
    return values


def extract(kind, response):
    if kind == 'gpt':
        assert response['status']=='completed', 'Incomplete response retained; inspect it.'
        blocks = [c for o in response['output'] if o.get('type')=='message'
                  for c in o['content'] if c.get('type')=='output_text']
        answer = ''.join(c['text'] for c in blocks).strip()
        logs = [t for c in blocks for t in c.get('logprobs',[])]
        assert response['usage'].get('output_tokens_details',{}).get('reasoning_tokens',0)==0
    else:
        assert response['done']
        assert not response['message'].get('thinking'), 'Thinking must be disabled.'
        answer = response['message']['content'].strip()
        logs = response['logprobs']
    assert answer in ('A','B','C'), 'Not a single A/B/C answer.'
    labels = [entry for entry in logs if entry['token'] in ('A','B','C')]
    assert len(labels)==1 and labels[0]['token']==answer, 'Label token ambiguous or absent.'
    for entry in logs:
        if entry is not labels[0]:
            assert entry['token'].strip()=='' or entry['token'].startswith('<'), 'Unexpected visible output token.'
    return dict(answer=answer, **token_scores(labels[0]))


def request_for(kind, manifest, target, crop):
    prompt=manifest['prompts'][target]
    if kind=='local':
        request=base.request_for(MODELS[kind],prompt,crop)
        request['options']=manifest['local_options']
        metadata=copy.deepcopy(request)
        metadata['messages'][-1]['images']=[dict(file=crop['crop_file'],sha256=crop['crop_sha256'])]
    else:
        request=dict(model=MODELS[kind], reasoning={'effort':'none'},temperature=0,
                     max_output_tokens=16,store=False,service_tier='default',
                     include=['message.output_text.logprobs'],top_logprobs=20,
                     input=[dict(role='system',content=prompt['system_prompt']),
                            dict(role='user',content=[dict(type='input_text',text=prompt['user_prompt']),
                                 dict(type='input_image',detail='high',image_url='data:image/png;base64,'+
                                      base64.b64encode((ROOT/crop['crop_file']).read_bytes()).decode('ascii'))])])
        metadata=copy.deepcopy(request)
        metadata['input'][-1]['content'][-1]['image_url']=dict(file=crop['crop_file'],sha256=crop['crop_sha256'])
    return request,metadata


def infer(kind, request):
    if kind=='local':
        return base.api('/api/chat',request)
    http=Request('https://api.openai.com/v1/responses',data=json.dumps(request).encode(),
                 headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],'Content-Type':'application/json'})
    with urlopen(http,timeout=180) as response:
        return json.load(response)


def gpt_parallel(manifest, dest):
    jobs=[]
    for state,crops in manifest['crops'].items():
        for crop in crops:
            for target in manifest['prompts']:
                out=dest/'responses'/state/target/(crop['identity']+'.json')
                if out.exists():
                    saved=read(out)
                    _,metadata=request_for('gpt',manifest,target,crop)
                    assert saved['request']==metadata
                    extract('gpt',saved['response'])
                else:
                    jobs.append((state,crop,target,out))

    def worker(job):
        state,crop,target,out=job
        request,metadata=request_for('gpt',manifest,target,crop)
        start=perf_counter()
        try:
            result=infer('gpt',request)
        except HTTPError as error:
            save(out.with_name(out.stem+'_error.json'),dict(http_status=error.code,request=metadata))
            raise RuntimeError(f'HTTP {error.code}; no automatic retry.') from None
        saved=dict(state=state,target=target,identity=crop['identity'],crop_sha256=crop['crop_sha256'],
                   request=metadata,response=result,elapsed_seconds=perf_counter()-start)
        save(out,saved)
        scores=extract('gpt',result)
        return dict(state=state,target=target,candidate=crop['identity'],answer=scores['answer'],
                    p_A=scores['p_A'],p_A_upper=scores['p_A_upper'],seconds=round(saved['elapsed_seconds'],2))

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Only one bounded group is submitted at a time, so failures stop new calls.
        for offset in range(0,len(jobs),4):
            files=[p for p in (dest/'responses').rglob('*.json') if not p.stem.endswith('_error')]
            billed=[read(p)['response']['usage'] for p in files]
            conservative_cost=sum((r['input_tokens']*2.5+r['output_tokens']*10)/1e6 for r in billed)
            assert conservative_cost<1.98, 'Spend guard reached; stop before issuing more calls.'
            for i,result in enumerate(executor.map(worker,jobs[offset:offset+4]),offset+1):
                print(json.dumps(dict(model=MODELS['gpt'],new_completed=i,total_new=len(jobs),**result)),flush=True)


def classify(kind, limit):
    manifest=read(HERE/'manifest.json')
    for name,sha in manifest['immutable_inputs'].items():
        assert digest(ROOT/name)==sha,name
    dest=HERE/FOLDERS[kind]
    runtime=dict(model=MODELS[kind],backend='OpenAI Responses' if kind=='gpt' else 'Ollama',
                 reasoning=False,source_manifest_sha256=manifest['source_manifest_sha256'])
    if kind=='local':
        runtime.update(version=base.api('/api/version'),show=base.api('/api/show',{'model':MODELS[kind]}),
                       tags=base.api('/api/tags'))
        runtime['model_digest']=next(m['digest'] for m in runtime['tags']['models'] if m['name']==MODELS[kind])
    else:
        assert os.environ.get('OPENAI_API_KEY'), 'Existing API credential is unavailable.'
    if (dest/'runtime.json').exists():
        assert read(dest/'runtime.json')==runtime
    save(dest/'runtime.json',runtime)
    if kind=='gpt' and not limit:
        return gpt_parallel(manifest,dest)
    completed=0
    try:
        for state,crops in manifest['crops'].items():
            for crop in crops:
                for target in manifest['prompts']:
                    out=dest/'responses'/state/target/(crop['identity']+'.json')
                    request,metadata=request_for(kind,manifest,target,crop)
                    if out.exists():
                        saved=read(out)
                        assert saved['request']==metadata
                        extract(kind,saved['response'])
                        continue
                    start=perf_counter()
                    try:
                        result=infer(kind,request)
                    except HTTPError as error:
                        save(out.with_name(out.stem+'_error.json'),dict(http_status=error.code,request=metadata))
                        raise RuntimeError(f'HTTP {error.code}; metadata saved, no automatic retry.') from None
                    saved=dict(state=state,target=target,identity=crop['identity'],crop_sha256=crop['crop_sha256'],
                               request=metadata,response=result,elapsed_seconds=perf_counter()-start)
                    save(out,saved)
                    scores=extract(kind,result)
                    completed+=1
                    if kind=='local' and completed==1:
                        save(dest/'loaded_runtime.json',base.api('/api/ps'))
                    print(json.dumps(dict(model=MODELS[kind],new_completed=completed,state=state,target=target,
                                          candidate=crop['identity'],answer=scores['answer'],p_A=scores['p_A'],
                                          p_A_upper=scores['p_A_upper'],seconds=round(saved['elapsed_seconds'],2))),flush=True)
                    if limit and completed>=limit:
                        return
    finally:
        if kind=='local':
            base.api('/api/generate',dict(model=MODELS[kind],keep_alive=0))


def collect(kind):
    manifest=read(HERE/'manifest.json')
    dest=HERE/FOLDERS[kind]
    rows,usage=[],[]
    for state,crops in manifest['crops'].items():
        for crop in crops:
            assert digest(ROOT/crop['crop_file'])==crop['crop_sha256']
            for target,name in manifest['names'].items():
                path=dest/'responses'/state/target/(crop['identity']+'.json')
                saved=read(path)
                _,metadata=request_for(kind,manifest,target,crop)
                assert saved['request']==metadata
                row=dict(model=MODELS[kind],state=state,target=target,target_name=name,
                         posthoc_identity=crop['identity'],own_object_match=target==crop['identity'],
                         same_family=base.family(target)==base.family(crop['identity']),
                         assisted_fallback=crop['assisted_fallback'],crop_file=crop['crop_file'],
                         crop_sha256=crop['crop_sha256'],response_file=str(path.relative_to(ROOT)),
                         elapsed_seconds=saved['elapsed_seconds'],**extract(kind,saved['response']))
                rows.append(row)
                if kind=='gpt':
                    u=saved['response']['usage']
                    cache=u.get('input_tokens_details',{})
                    cache_read=cache.get('cached_tokens',0)
                    cache_write=cache.get('cache_write_tokens',0)
                    cost=((u['input_tokens']-cache_read-cache_write)*2+cache_read*.2+cache_write*2.5+u['output_tokens']*10)/1e6
                    usage.append(dict(state=state,target=target,identity=crop['identity'],
                                      input_tokens=u['input_tokens'],cached_tokens=cache_read,
                                      cache_write_tokens=cache_write,output_tokens=u['output_tokens'],
                                      reasoning_tokens=u.get('output_tokens_details',{}).get('reasoning_tokens',0),
                                      estimated_standard_cost_usd=cost,response_file=row['response_file']))
    assert len(rows)==484
    table(dest/'all_scores',rows)
    if usage:
        table(dest/'api_usage',usage)
        save(dest/'usage_total.json',dict(requests=484,**{k:sum(r[k] for r in usage) for k in
             ('input_tokens','cached_tokens','cache_write_tokens','output_tokens','reasoning_tokens','estimated_standard_cost_usd')}))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','classify','collect'])
    parser.add_argument('--kind',choices=list(MODELS),default='gpt')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    elif args.action=='classify':
        classify(args.kind,args.limit)
    else:
        collect(args.kind)
