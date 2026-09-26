"""Local-only family prompt ablation; preserve baseline inputs and controls."""
import argparse
import base64
import copy
import importlib.util
import json
from pathlib import Path
from time import perf_counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'gpt6-gemma4-four-scene-trial'
spec=importlib.util.spec_from_file_location('baseline_models',SOURCE/'run_models.py')
previous=importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
read,save,digest,table=previous.read,previous.save,previous.digest,previous.table
api,family=previous.base.api,previous.base.family
MODELS={'qwen3.5:4b':'qwen3.5-4b','qwen3.5:9b':'qwen3.5-9b','gemma4:12b':'gemma4-12b'}


def prepare():
    baseline=read(SOURCE/'manifest.json')
    baseline_rows=[r for r in read(SOURCE/'all_model_scores.json') if r['model'] in MODELS]
    assert len(baseline_rows)==1452
    descriptions=read(HERE/'descriptions.json')
    system=(HERE/'system_prompt.txt').read_text(encoding='utf-8')
    template=(HERE/'user_prompt.txt').read_text(encoding='utf-8')
    prompts={t:dict(system_prompt=system,user_prompt=template.format(
        descriptions='\n'.join('- '+s for s in descriptions[family(t)]))) for t in baseline['names']}
    controls={}
    for model,slug in MODELS.items():
        sample=next(r for r in baseline_rows if r['model']==model)
        raw=read(ROOT/sample['response_file'])
        controls[model]=copy.deepcopy(raw['request'])
        runtime_path=(ROOT/'qwen9b-four-scene-trial'/model.split(':')[1]/'runtime.json'
                      if model.startswith('qwen') else SOURCE/slug/'runtime.json')
        controls[model]['expected_model_digest']=read(runtime_path)['model_digest']
        controls[model]['expected_runtime_version']=read(runtime_path)['version']['version']
    protected={}
    for folder in (SOURCE,ROOT/'qwen9b-four-scene-trial'):
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                protected[path.relative_to(ROOT).as_posix()]=digest(path)
    immutable={**baseline['immutable_inputs'],**protected}
    for name in ('system_prompt.txt','user_prompt.txt','descriptions.json'):
        path=HERE/name
        immutable[path.relative_to(ROOT).as_posix()]=digest(path)
    render=ROOT/'cad-view-review/Waterproof_Male/review_sheet.png'
    immutable[render.relative_to(ROOT).as_posix()]=digest(render)
    manifest=dict(names=baseline['names'],crops=baseline['crops'],prompts=prompts,
        descriptions=descriptions,models=MODELS,controls=controls,immutable_inputs=immutable,
        baseline_scores='gpt6-gemma4-four-scene-trial/all_model_scores.json',
        baseline_commit='b0011c57544d43883d57a0fc5e58137936bf91d1',
        labels=dict(A='FAMILY MATCH SUPPORTED',B='FAMILY MISMATCH SUPPORTED',C='INSUFFICIENT EVIDENCE'),
        score='exp(raw log probability of token A); no A/B/C renormalization',
        local_only=True,expected_calls_per_model=484,threshold_before_collection=None,
        render_inspection=dict(file=render.relative_to(ROOT).as_posix(),
            finding='Block-shaped body with grouped round openings visible on opposing connector faces; no exact opening count added.'),
        scope='Prompt and description ablation together, not an isolated wording-only effect. All 11 target IDs retained; size-family prompts identical. No target IDs or sizes appended to family descriptions.',
        exclusions=baseline['exclusions'],fallback_policy=baseline['fallback_policy'])
    if (HERE/'manifest.json').exists():
        assert read(HERE/'manifest.json')==manifest,'Frozen ablation changed.'
    save(HERE/'manifest.json',manifest)
    save(HERE/'selected_prompts.json',prompts)
    table(HERE/'baseline_scores',baseline_rows)
    print('Prepared 3 local models x 484 calls; no cloud API; baseline protected.',flush=True)


def validate_inputs(manifest):
    for name,sha in manifest['immutable_inputs'].items():
        assert digest(ROOT/name)==sha,name


def request_for(manifest,model,target,crop):
    request=copy.deepcopy(manifest['controls'][model])
    request.pop('expected_model_digest')
    request.pop('expected_runtime_version')
    prompt=manifest['prompts'][target]
    request['messages']=[dict(role='system',content=prompt['system_prompt']),
        dict(role='user',content=prompt['user_prompt'],images=[dict(file=crop['crop_file'],sha256=crop['crop_sha256'])])]
    return request


def classify(model,limit=0):
    assert model in MODELS,'Local models only.'
    m=read(HERE/'manifest.json')
    validate_inputs(m)
    dest=HERE/MODELS[model]
    version=api('/api/version')
    tag=next(t for t in api('/api/tags')['models'] if t['name']==model)
    assert tag['digest']==m['controls'][model]['expected_model_digest'],'Model weights changed.'
    assert version['version']==m['controls'][model]['expected_runtime_version'],'Ollama version changed.'
    save(dest/'runtime.json',dict(model=model,model_digest=tag['digest'],version=version,
        controls=m['controls'][model],show=api('/api/show',dict(model=model))))
    start_run=perf_counter()
    new=0
    try:
        for state,crops in m['crops'].items():
            for crop in crops:
                for target in m['names']:
                    path=dest/'responses'/state/target/(crop['identity']+'.json')
                    meta=request_for(m,model,target,crop)
                    if path.exists():
                        stored=read(path)
                        assert stored['request']==meta and stored['model_digest']==tag['digest']
                        previous.extract('local',stored['response'])
                        continue
                    request=copy.deepcopy(meta)
                    request['messages'][1]['images']=[base64.b64encode((ROOT/crop['crop_file']).read_bytes()).decode('ascii')]
                    started=perf_counter()
                    response=api('/api/chat',request)
                    save(path,dict(state=state,target=target,identity=crop['identity'],
                        crop_sha256=crop['crop_sha256'],model_digest=tag['digest'],request=meta,
                        response=response,elapsed_seconds=perf_counter()-started))
                    result=previous.extract('local',response)
                    new+=1
                    if new==1:
                        save(dest/'loaded_runtime.json',api('/api/ps'))
                    if new==1 or new%11==0:
                        print(json.dumps(dict(model=model,new_completed=new,state=state,
                            candidate=crop['identity'],last_target=target,p_A=result['p_A'],
                            answer=result['answer'],wall_seconds=round(perf_counter()-start_run,1))),flush=True)
                    if limit and new>=limit:
                        return
    finally:
        api('/api/generate',dict(model=model,keep_alive=0))
        save(dest/'last_session.json',dict(new_calls=new,wall_seconds=perf_counter()-start_run))
    validate_inputs(m)


def collect(model):
    m=read(HERE/'manifest.json')
    rows=[]
    for state,crops in m['crops'].items():
        for crop in crops:
            for target,name in m['names'].items():
                path=HERE/MODELS[model]/'responses'/state/target/(crop['identity']+'.json')
                value=read(path)
                assert value['request']==request_for(m,model,target,crop)
                assert value['response']['model']==model
                rows.append(dict(model=model,state=state,target=target,target_name=name,
                    target_family=family(target),posthoc_identity=crop['identity'],
                    observed_family=family(crop['identity']),own_object_match=target==crop['identity'],
                    same_family=family(target)==family(crop['identity']),assisted_fallback=crop['assisted_fallback'],
                    crop_file=crop['crop_file'],crop_sha256=crop['crop_sha256'],
                    response_file=path.relative_to(ROOT).as_posix(),elapsed_seconds=value['elapsed_seconds'],
                    **previous.extract('local',value['response'])))
    assert len(rows)==484
    table(HERE/MODELS[model]/'all_scores',rows)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','classify','collect'])
    parser.add_argument('--model',choices=list(MODELS))
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        for model in ([args.model] if args.model else MODELS):
            if args.action=='classify':
                classify(model,args.limit)
            else:
                collect(model)
