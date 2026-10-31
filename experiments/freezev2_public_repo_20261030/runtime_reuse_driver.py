"""Versioned DINO model reuse around the immutable public inference runner.

This adds execution provenance; it does not replace the public pose algorithms.
The original configuration, source snapshot, numerical outputs and old receipts
remain unchanged. Activation requires a separate measured equivalence receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

BASE_MANIFEST_SHA = 'adaafb2c3525d3202f612e4c88c01dff28534476f7f3e207056a6a14e1307b2b'
CONFIGURATION_SHA = '7c88a0303a9da06388b68ec50c63502dada02689e0bb06cce20a13e70c77c567'
FILES = ('runtime_reuse_driver.py','launch_runtime_reuse.py','dino_reuse_trial.py')
TERMINAL = ('complete','failed','blocked')


def local(value):
    value = str(value)
    if os.name != 'nt' and len(value)>2 and value[1]==':':
        value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name == 'nt' and value.startswith('/mnt/') and len(value)>7:
        value=value[5].upper()+':/'+value[7:]
    return Path(value).expanduser().resolve()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def configuration(config_path):
    config=read(config_path);out=local(config['paths']['output'])
    if digest(config)!=CONFIGURATION_SHA:raise ValueError('Canonical method/configuration changed')
    saved=out/'configuration.json'
    if read(saved)!=config:raise ValueError('Saved configuration differs')
    original=out/'source_snapshot/source_manifest.json'
    if sha256(original)!=BASE_MANIFEST_SHA:raise ValueError('Original sealed manifest changed')
    return config,out,saved


def seal_revision(config_path,source_dir=None):
    config,out,saved=configuration(config_path)
    source=Path(source_dir or Path(__file__).parent).resolve()
    folder=out/'auxiliary_source/runtime_reuse_v1'
    files={name:sha256(source/name) for name in FILES}
    manifest={'schema':'runtime-reuse-source-v1','configuration_sha256':digest(config),
              'configuration_file_sha256':sha256(saved),'base_source_manifest_sha256':BASE_MANIFEST_SHA,
              'files':files,'cache_policy':'DINO local giant model reused within each localization process; GeDi unchanged',
              'method_boundary':'Execution revision only; no original source, settings, sampling, precision, registration or scoring changes'}
    target=folder/'execution_manifest.json'
    if target.exists():
        if read(target)!=manifest:raise ValueError('Existing execution revision differs; preserve it and use a new version')
        for name,wanted in files.items():
            if sha256(folder/name)!=wanted:raise ValueError('Sealed execution helper changed: '+name)
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Unidentified execution revision folder already exists')
        folder.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            with (folder/name).open('xb') as f:f.write((source/name).read_bytes())
        atomic_json(target,manifest)
    return target


def verify_revision(config_path,manifest_path,*,require_sealed_self=True):
    config,out,saved=configuration(config_path);manifest_path=local(manifest_path)
    if manifest_path!=out/'auxiliary_source/runtime_reuse_v1/execution_manifest.json':
        raise ValueError('Execution manifest is outside its versioned run directory')
    m=read(manifest_path)
    if m.get('schema')!='runtime-reuse-source-v1' or m.get('configuration_sha256')!=digest(config) or m.get('base_source_manifest_sha256')!=BASE_MANIFEST_SHA:
        raise ValueError('Execution source/configuration identity differs')
    if m.get('configuration_file_sha256')!=sha256(saved) or set(m.get('files',{}))!=set(FILES):
        raise ValueError('Execution manifest file identities differ')
    for name,wanted in m['files'].items():
        if sha256(manifest_path.parent/name)!=wanted:raise ValueError('Execution helper changed: '+name)
    if require_sealed_self and Path(__file__).resolve()!=manifest_path.parent/'runtime_reuse_driver.py':
        raise ValueError('Execute the sealed runtime driver')
    return config,out,m


def load_base(out):
    folder=Path(out)/'source_snapshot'
    if sha256(folder/'source_manifest.json')!=BASE_MANIFEST_SHA:raise ValueError('Base source manifest changed')
    for name,wanted in read(folder/'source_manifest.json')['files'].items():
        if sha256(folder/name)!=wanted:raise ValueError('Base source payload changed: '+name)
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('sealed_runtime_reuse_base_suite',folder/'suite_runner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verified_transition(config_path,manifest_path,transition_path,*,verify_payloads=True,require_sealed_self=True):
    config,out,_=verify_revision(config_path,manifest_path,require_sealed_self=require_sealed_self)
    path=local(transition_path)
    if not path.is_relative_to(out):raise ValueError('Transition manifest escaped own output')
    v=read(path)
    if v.get('schema')!='runtime-reuse-transition-v1' or v.get('status')!='ready_for_handoff':raise ValueError('Transition is not ready')
    wanted={'base_configuration_sha256':digest(config),'base_source_manifest_sha256':BASE_MANIFEST_SHA,
            'execution_revision_sha256':sha256(manifest_path)}
    if any(v.get(k)!=x for k,x in wanted.items()):raise ValueError('Transition source/configuration differs')
    legacy=v.get('legacy_receipts')
    if not isinstance(legacy,dict):raise ValueError('Transition legacy receipt ledger is absent')
    for relative,wanted in legacy.items():
        p=out/relative
        if p.resolve()!=p or not p.is_relative_to(out) or p.name!='receipt.json' or p.parent.parent.name!='targets':
            raise ValueError('Invalid legacy target receipt path')
        if verify_payloads and sha256(p)!=wanted:raise ValueError('Legacy target receipt changed: '+relative)
    return v


def require_validation(config_path,manifest_path,path):
    config,out,_=verify_revision(config_path,manifest_path,require_sealed_self=False)
    p=local(path)
    if not p.is_relative_to(out):raise ValueError('Validation receipt escaped own output')
    value=read(p)
    if value.get('status')!='passed' or value.get('execution_revision_sha256')!=sha256(manifest_path) or value.get('configuration_sha256')!=digest(config):
        raise ValueError('Measured equivalence and speed validation did not pass for this execution revision')
    if value.get('descriptor_equivalence') is not True or value.get('pose_equivalence') is not True or value.get('measured_speedup') is not True:
        raise ValueError('Actual descriptor/pose equivalence and speedup are required')
    if value.get('helper_sha256')!=read(manifest_path)['files']['dino_reuse_trial.py']:
        raise ValueError('Validation tested a different model reuse helper')
    artifacts=value.get('artifacts')
    if not isinstance(artifacts,dict) or not artifacts:raise ValueError('Measured validation artifacts are required')
    for name,wanted in artifacts.items():
        artifact=local(name)
        if not artifact.is_relative_to(out) or not artifact.is_file() or sha256(artifact)!=wanted:
            raise ValueError('Measured validation artifact identity differs: '+name)
    return value


def audit_provenance(out,dataset,revision_sha,transition,transition_sha,*,receipt_paths=None):
    out=Path(out);legacy=transition['legacy_receipts'];counts={'legacy':0,'runtime_reuse':0}
    paths=(out/'datasets'/dataset/'targets').glob('*/receipt.json') if receipt_paths is None else receipt_paths
    for p in paths:
        saved=read(p)
        if saved.get('status') not in TERMINAL:raise ValueError('Pending target in execution provenance audit')
        relative=p.relative_to(out).as_posix()
        if relative in legacy:
            if sha256(p)!=legacy[relative]:raise ValueError('Legacy target receipt changed')
            counts['legacy']+=1;continue
        if saved.get('execution_revision_sha256')!=revision_sha or saved.get('transition_manifest_sha256')!=transition_sha:
            raise ValueError('Unidentified target execution revision: '+relative)
        if saved.get('status')=='blocked' and saved.get('execution_report') is None and saved.get('original_csv') is None:
            counts['runtime_reuse']+=1;continue
        report=local(saved.get('execution_report',''))
        if report.parent!=p.parent or report.name!='runtime_reuse_report.json' or sha256(report)!=saved.get('execution_report_sha256'):
            raise ValueError('Target execution report identity differs')
        data=read(report)
        if data.get('execution_revision_sha256')!=revision_sha or data.get('original_request_sha256')!=sha256(p.parent/'original_command.json'):
            raise ValueError('Runtime report source/request identity differs')
        if data.get('configuration_sha256')!=CONFIGURATION_SHA or data.get('helper_sha256')!=read(out/'auxiliary_source/runtime_reuse_v1/execution_manifest.json')['files']['dino_reuse_trial.py']:
            raise ValueError('Runtime report helper/configuration identity differs')
        if data.get('integrity_error'):raise ValueError('Runtime cache integrity failed')
        counts['runtime_reuse']+=1
    return counts


def internal_original(config_path,manifest_path,request_path):
    config,out,_=verify_revision(config_path,manifest_path);base=load_base(out)
    request=local(request_path)
    if not request.is_relative_to(out/'datasets') or request.name!='original_command.json':raise ValueError('Original request escaped target output')
    value=read(request)
    if len(value.get('command',[]))<4 or value['command'][3]!='run-localization':raise ValueError('DINO reuse only supports original localization calls')
    report={'schema':'dino-runtime-reuse-report-v1','execution_revision_sha256':sha256(manifest_path),
            'configuration_sha256':digest(config),'original_request_sha256':sha256(request),
            'helper_sha256':sha256(Path(__file__).with_name('dino_reuse_trial.py')),'status':'running'}
    started=time.perf_counter();stats={};error=None
    try:
        import torch
        from dino_reuse_trial import dino_reuse_context
        with dino_reuse_context(torch,config['paths']['dinov2_root'],enabled=True,stats=stats):
            try:base.internal_original(request)
            except SystemExit as exc:
                if exc.code not in (None,0):raise
        if any(event.get('status')=='failed' for event in stats.get('events',[]) if event.get('component')=='torch.hub.load'):
            report['integrity_error']=True
            raise RuntimeError('A model load failed inside original candidate handling; runtime cache cannot silently count this as a candidate rejection')
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));error=exc
    finally:
        report.update(runtime_s=time.perf_counter()-started,statistics=stats)
        atomic_json(request.parent/'runtime_reuse_report.json',report)
    if error is not None:raise error
    return report


def run(config_path,manifest_path,transition_path,validation_path):
    config,out,_=verify_revision(config_path,manifest_path);require_validation(config_path,manifest_path,validation_path)
    transition=verified_transition(config_path,manifest_path,transition_path)
    revision_sha=sha256(manifest_path);transition_sha=sha256(transition_path);base=load_base(out)
    original_wrapped=base.wrapped_original;original_target=base.target_stage
    def wrapped(ctx,folder,command):
        if command[3]!='run-localization':return original_wrapped(ctx,folder,command)
        try:verify_revision(config_path,manifest_path)
        except ValueError as exc:raise base.IdentityError(str(exc)) from exc
        original_wrapped(ctx,folder,command)
        return [str(base.path_of(ctx['config'],'inference_python')),'-B',str(Path(__file__).resolve()),
                '--internal-original',str(Path(folder)/'original_command.json'),'--config',str(config_path),
                '--revision-manifest',str(manifest_path)]
    def target(ctx,dataset,item,compute,blocked=None):
        folder=base.target_folder(ctx,dataset,item);p=folder/'receipt.json'
        relative=p.relative_to(out).as_posix()
        existed=p.exists() and read(p).get('status') in TERMINAL
        record=original_target(ctx,dataset,item,compute,blocked)
        try:verify_revision(config_path,manifest_path)
        except ValueError as exc:raise base.IdentityError(str(exc)) from exc
        if existed:
            if relative in transition['legacy_receipts']:
                if sha256(p)!=transition['legacy_receipts'][relative]:raise base.IdentityError('Legacy receipt changed during resume')
            else:
                try:audit_provenance(out,dataset,revision_sha,transition,transition_sha,receipt_paths=[p])
                except ValueError as exc:raise base.IdentityError(str(exc)) from exc
            return record
        report=folder/'runtime_reuse_report.json'
        record.update(execution_revision_sha256=revision_sha,transition_manifest_sha256=transition_sha,
                      execution_report=str(report) if report.exists() else None,
                      execution_report_sha256=sha256(report) if report.exists() else None)
        if not report.exists() and not blocked:raise base.IdentityError('Runtime execution report is absent')
        if report.exists() and read(report).get('integrity_error'):raise base.IdentityError('Runtime cache integrity failed; preserve target and halt')
        ctx['durable'].atomic_json(p,record)
        return record
    base.wrapped_original=wrapped;base.target_stage=target
    import evaluate_own
    original_evaluate=evaluate_own.evaluate_dataset
    def evaluate(root,output,dataset,*args,**kwargs):
        try:
            verified_transition(config_path,manifest_path,transition_path)
            audit_provenance(output,dataset,revision_sha,transition,transition_sha)
        except ValueError as exc:raise base.IdentityError('Execution provenance audit failed: '+str(exc)) from exc
        return original_evaluate(root,output,dataset,*args,**kwargs)
    evaluate_own.evaluate_dataset=evaluate
    try:return base.run(config_path)
    finally:
        base.wrapped_original=original_wrapped;base.target_stage=original_target
        evaluate_own.evaluate_dataset=original_evaluate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--revision-manifest',required=True,type=Path);p.add_argument('--transition-manifest',type=Path)
    p.add_argument('--validation-receipt',type=Path);p.add_argument('--internal-original',type=Path)
    args=p.parse_args(argv)
    if args.internal_original:return internal_original(args.config,args.revision_manifest,args.internal_original)
    if not args.transition_manifest or not args.validation_receipt:p.error('A transition and passed validation receipt are required')
    return run(args.config,args.revision_manifest,args.transition_manifest,args.validation_receipt)

if __name__=='__main__':main()