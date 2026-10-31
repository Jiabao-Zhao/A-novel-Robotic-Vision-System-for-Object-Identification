"""Pinned run-control gate after full LM-O evaluation, before another dataset."""
from __future__ import annotations
import argparse
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

CONFIGURATION_SHA='7c88a0303a9da06388b68ec50c63502dada02689e0bb06cce20a13e70c77c567'
BASE_MANIFEST_SHA='adaafb2c3525d3202f612e4c88c01dff28534476f7f3e207056a6a14e1307b2b'
EXECUTION_REVISION_SHA='f0ecc7957d9862b647677c57a9e34df438a8f642c07b07970437ab3e126d7e4b'
FILES=('runtime_gate_driver.py','launch_runtime_gate.py')
EXPECTED_STOPS=('stopped_lmo_not_aligned','stopped_lmo_awaiting_alignment_tolerance','stopped_lmo_evaluation_unavailable')
METRIC='bop19_average_recall'


def local(value):
    value=str(value)
    if os.name!='nt' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name=='nt' and value.startswith('/mnt/') and len(value)>7:value=value[5].upper()+':/'+value[7:]
    return Path(value).expanduser().resolve()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def load_prior(out,manifest_path=None):
    wanted=Path(out)/'auxiliary_source/runtime_reuse_v1/execution_manifest.json'
    path=local(manifest_path or wanted)
    if path!=wanted or sha256(path)!=EXECUTION_REVISION_SHA:raise ValueError('Validated DINO execution revision changed')
    value=read(path)
    for name,wanted_hash in value['files'].items():
        if name not in ('runtime_reuse_driver.py','launch_runtime_reuse.py','dino_reuse_trial.py') or sha256(path.parent/name)!=wanted_hash:
            raise ValueError('Validated DINO execution payload changed')
    sys.path.insert(0,str(path.parent))
    # The old launcher's absolute import must resolve this exact sealed module.
    prior=load_module(path.parent/'runtime_reuse_driver.py','sealed_lmo_gate_prior')
    sys.modules['runtime_reuse_driver']=prior
    return prior,path


def validate_policy(value):
    if value.get('schema')!='lmo-paper-stop-policy-v1' or value.get('status') not in ('configured','awaiting_user'):
        raise ValueError('Unknown LM-O stop policy')
    if isinstance(value.get('paper_ar'),bool) or value.get('paper_ar')!=0.694:
        raise ValueError('Paper comparator must be Table II row11 LM-O AR69.4 percent')
    if value['status']=='awaiting_user':return value
    if value.get('mode')=='rounded_one_decimal':return value
    tolerance=value.get('tolerance_pp')
    if value.get('mode')!='absolute_percentage_points' or isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or not 0<=tolerance<=100:
        raise ValueError('An explicit finite nonnegative percentage-point tolerance is required')
    return value


def configuration(config_path):
    config=read(config_path);out=local(config['paths']['output']);saved=out/'configuration.json'
    if digest(config)!=CONFIGURATION_SHA or read(saved)!=config:raise ValueError('Canonical inference configuration changed')
    if sha256(out/'source_snapshot/source_manifest.json')!=BASE_MANIFEST_SHA:raise ValueError('Original sealed inference source changed')
    return config,out,saved


def seal_control(config_path,policy_path,source_dir=None):
    config,out,saved=configuration(config_path);policy=local(policy_path)
    if not policy.is_relative_to(out/'run_control'):raise ValueError('Keep the immutable policy inside own/run_control')
    validate_policy(read(policy));prior,execution=load_prior(out)
    prior.verify_revision(saved,execution,require_sealed_self=False)
    source=Path(source_dir or Path(__file__).parent).resolve();folder=out/'auxiliary_source/runtime_reuse_v3_lmo_gate'
    target=folder/'control_manifest.json'
    value={'schema':'lmo-paper-run-control-source-v1','configuration_sha256':digest(config),
           'base_source_manifest_sha256':BASE_MANIFEST_SHA,'execution_revision_sha256':sha256(execution),
           'policy_path':str(policy),'policy_sha256':sha256(policy),'files':{name:sha256(source/name) for name in FILES},
           'boundary':'Scheduling only: existing sealed DINO runtime and original inference/evaluation stay unchanged; compare full LM-O before any next dataset.'}
    if target.exists():
        if read(target)!=value:raise ValueError('Sealed control revision differs; preserve it and use a new control revision')
        for name,wanted in value['files'].items():
            if sha256(folder/name)!=wanted:raise ValueError('Sealed run-control payload changed')
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Unknown control revision folder already exists')
        folder.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            with (folder/name).open('xb') as f:f.write((source/name).read_bytes())
        prior.atomic_json(target,value)
    return target


def verify_control(config_path,control_manifest,policy_path=None,revision_manifest=None,*,require_sealed_self=True):
    config,out,saved=configuration(config_path);path=local(control_manifest)
    if path!=out/'auxiliary_source/runtime_reuse_v3_lmo_gate/control_manifest.json':raise ValueError('Control manifest is outside the sealed revision')
    value=read(path)
    if value.get('schema')!='lmo-paper-run-control-source-v1' or value.get('configuration_sha256')!=CONFIGURATION_SHA or value.get('base_source_manifest_sha256')!=BASE_MANIFEST_SHA or value.get('execution_revision_sha256')!=EXECUTION_REVISION_SHA:
        raise ValueError('Run-control source/configuration pins differ')
    if set(value.get('files',{}))!=set(FILES):raise ValueError('Run-control file set differs')
    for name,wanted in value['files'].items():
        if sha256(path.parent/name)!=wanted:raise ValueError('Run-control source payload changed')
    if require_sealed_self and Path(__file__).resolve()!=path.parent/'runtime_gate_driver.py':raise ValueError('Execute the sealed run-control driver')
    policy=local(policy_path or value['policy_path'])
    if policy!=local(value['policy_path']) or not policy.is_relative_to(out/'run_control') or sha256(policy)!=value.get('policy_sha256'):
        raise ValueError('Bound alignment policy changed')
    validate_policy(read(policy));prior,execution=load_prior(out,revision_manifest)
    prior.verify_revision(saved,execution,require_sealed_self=False)
    return config,out,value,prior


def decide(ar,policy):
    validate_policy(policy)
    if isinstance(ar,bool) or not isinstance(ar,(int,float)) or not math.isfinite(ar) or not 0<=ar<=1:
        raise ValueError('Official own LM-O AR is unavailable or invalid')
    result={'paper_ar':0.694,'paper_ar_percent':69.4,'own_ar':ar,'own_ar_percent':ar*100,
            'difference_percentage_points':float((Decimal(str(ar))-Decimal('0.694'))*100),
            'paper_reference':'FreeZeV2 paper Table II row11 CNOS-only base; exact mask archive parity remains unresolved'}
    if policy['status']=='awaiting_user':return {**result,'aligned':None,'continue':False,'status':'stopped_lmo_awaiting_alignment_tolerance'}
    if policy['mode']=='rounded_one_decimal':
        rounded=(Decimal(str(ar))*100).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
        aligned=rounded==Decimal('69.4');result.update(mode=policy['mode'],own_rounded_percent=str(rounded))
    else:
        difference=abs((Decimal(str(ar))-Decimal('0.694'))*100)
        aligned=difference<=Decimal(str(policy['tolerance_pp']));result.update(mode=policy['mode'],tolerance_percentage_points=policy['tolerance_pp'])
    return {**result,'aligned':aligned,'continue':aligned,'status':'lmo_aligned_continue' if aligned else 'stopped_lmo_not_aligned'}


def audit_completed_lmo(ctx):
    """Reopen actual original evaluator evidence; no predictions or GT tuning."""
    import evaluate_own as evaluation
    out=ctx['output'];receipt_path=out/'datasets/lmo/own_evaluation_receipt.json';receipt=read(receipt_path)
    if receipt.get('schema')!='unchanged-public-own-prediction-evaluation-v1' or receipt.get('dataset')!='lmo' or receipt.get('status')!='complete' or receipt.get('authors_predictions_used') is not False:
        raise ValueError('A complete official own-prediction LM-O receipt is required')
    ec=evaluation.context(ctx['root'],out,'lmo',ctx['config'])
    csv=out/'datasets/lmo/freezev2public_lmo-test.csv';ledger=out/'datasets/lmo/target_ledger.json'
    score=out/'datasets/lmo/official_evaluation'/csv.stem/'scores_bop19.json'
    if local(receipt.get('CSV',''))!=csv or local(receipt.get('ledger',''))!=ledger or local(receipt.get('score_file',''))!=score:
        raise ValueError('LM-O evaluator payload paths differ')
    if receipt.get('protocol',{}).get('visib_gt_min')!=-1 or receipt.get('protocol',{}).get('task')!='BOP19 model-based 6D localization':raise ValueError('LM-O evaluation protocol differs')
    audit=evaluation.audit_ledger(out,'lmo',csv,ledger,ec['targets'],ec['pins'],ec['source'])
    if audit!=receipt.get('coverage') or audit.get('official_object_image_targets')!=1445 or audit.get('recorded_targets')!=1445 or audit.get('requested_instances')!=1445 or audit.get('missing_targets')!=0 or audit.get('failed_and_blocked_remain_in_denominator') is not True:
        raise ValueError('Full failure-inclusive LM-O target accounting is not verified')
    pins={**ec['pins'],'CSV_sha256':sha256(csv),'target_ledger_sha256':sha256(ledger),
          'ledger_artifacts_sha256':digest({'receipts':audit['target_receipts_sha256'],'original_CSVs':audit['original_generated_CSVs_sha256']})}
    if receipt.get('source_pins')!=pins or receipt.get('score_sha256')!=sha256(score):raise ValueError('Official LM-O metric/source/artifact identity differs')
    metrics=evaluation.pose_metrics(read(score))
    if receipt.get('metrics')!=metrics:raise ValueError('Saved own LM-O score differs from its official score file')
    return metrics[METRIC],{'receipt':str(receipt_path),'receipt_sha256':sha256(receipt_path),'score_file':str(score),'score_sha256':sha256(score),'CSV_sha256':sha256(csv),'ledger_sha256':sha256(ledger),'coverage':{k:audit[k] for k in ('official_object_image_targets','requested_instances','recorded_targets','missing_targets','terminal_status_counts','targets_without_saved_poses','saved_poses')}}


def comparison_gate(ctx,control_manifest,policy_path):
    # Audit source + immutable policy again at the actual dataset boundary.
    _,out,manifest,prior=verify_control(ctx['output']/'configuration.json',control_manifest,policy_path)
    result={'schema':'lmo-paper-comparison-gate-v1','control_manifest_sha256':sha256(control_manifest),
            'policy_path':str(local(policy_path)),'policy_sha256':manifest['policy_sha256'],
            'comparison_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'next_dataset_started_by_gate':False}
    try:
        ar,evidence=audit_completed_lmo(ctx);result.update(decide(ar,read(local(policy_path))),evidence=evidence)
    except Exception as exc:
        result.update(status='stopped_lmo_evaluation_unavailable',aligned=None,**{'continue':False},reason=type(exc).__name__+': '+str(exc))
    prior.atomic_json(out/'run_control/lmo_comparison_gate.json',result)
    return result


def install_gate(base,control_manifest,policy_path):
    original=base.run_dataset
    def execute(ctx,dataset,suite_lock):
        if dataset!='lmo':
            try:
                _,_,control,_=verify_control(ctx['output']/'configuration.json',control_manifest,policy_path)
                record=read(ctx['output']/'run_control/lmo_comparison_gate.json')
            except Exception as exc:raise base.IdentityError('LM-O comparison gate evidence is absent or invalid') from exc
            if record.get('continue') is not True or record.get('control_manifest_sha256')!=sha256(control_manifest) or record.get('policy_sha256')!=control['policy_sha256']:
                raise base.IdentityError('LM-O comparison gate did not authorize another dataset')
            return original(ctx,dataset,suite_lock)
        try:
            result=original(ctx,dataset,suite_lock)
        except Exception as exc:
            # Never let the original suite's dataset-error fallback proceed to
            # T-LESS when LM-O did not finish with a usable official score.
            gate={'schema':'lmo-paper-comparison-gate-v1','status':'stopped_lmo_evaluation_unavailable','continue':False,
                  'reason':type(exc).__name__+': '+str(exc),'control_manifest_sha256':sha256(control_manifest),
                  'comparison_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            ctx['durable'].atomic_json(ctx['output']/'run_control/lmo_comparison_gate.json',gate)
        else:
            try:gate=comparison_gate(ctx,control_manifest,policy_path)
            except Exception as exc:raise base.IdentityError('LM-O gate source/policy integrity failed: '+str(exc)) from exc
        if gate.get('continue') is not True:
            import evaluate_own
            evaluate_own.write_summary(ctx['root'],ctx['output'],durable=ctx['durable'])
            final={'status':gate['status'],'dataset':'lmo','comparison_gate':str(ctx['output']/'run_control/lmo_comparison_gate.json'),
                   'control_manifest_sha256':sha256(control_manifest),'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                   'next_datasets_started':False,'remaining_datasets_not_run_per_user_gate':['tless','tudl','icbin','itodd','hb','ycbv'],
                   'seven_dataset_aggregate':None}
            ctx['durable'].atomic_json(ctx['output']/'status.json',final)
            ctx['durable'].atomic_json(ctx['output']/'results_summary.json',final)
            report=['# LM-O comparison and user stop gate','',
                'Status: '+gate['status'],
                'Paper comparator: Table II row11 CNOS-only base LM-O pose AR69.4 percent.',
                'Own pose AR: '+(str(gate['own_ar_percent'])+' percent' if gate.get('own_ar_percent') is not None else 'unavailable; evaluation not verified'),
                'Difference in percentage points: '+str(gate.get('difference_percentage_points','unavailable')),
                'Aligned under the user policy: '+str(gate.get('aligned')),
                'Reason: '+gate.get('reason','User-requested comparison stop rule.'),'',
                'The other six datasets were not started after this gate; this is not a completed seven-dataset reproduction.',
                'All recorded LM-O failures remain in the official denominator. No predictions, settings or test-GT tuning were changed.',
                'The closest CNOS-only paper comparator is used; exact SAM/FastSAM mask archive parity remains unresolved.',
                'Machine-readable evidence: run_control/lmo_comparison_gate.json; official own metrics: datasets/lmo/own_evaluation_receipt.json.']
            base.atomic_text(ctx['output']/'run_control/LMO_PAPER_COMPARISON.md','\n'.join(report)+'\n')
            raise SystemExit(0)
        return result
    def guarded(ctx,dataset,suite_lock):
        if dataset!='lmo':return execute(ctx,dataset,suite_lock)
        try:return execute(ctx,dataset,suite_lock)
        except (SystemExit,base.IdentityError):raise
        except Exception as exc:raise base.IdentityError('LM-O run-control boundary failed; next datasets are forbidden: '+str(exc)) from exc
    base.run_dataset=guarded
    return original


def run(config_path,control_manifest,policy_path,revision_manifest,transition_manifest,validation_receipt):
    config,out,manifest,prior=verify_control(config_path,control_manifest,policy_path,revision_manifest)
    prior.require_validation(config_path,revision_manifest,validation_receipt)
    prior.verified_transition(config_path,revision_manifest,transition_manifest,require_sealed_self=False)
    original_load=prior.load_base
    def gated_load(folder):
        base=original_load(folder);install_gate(base,control_manifest,policy_path);return base
    prior.load_base=gated_load
    prior.atomic_json(out/'run_control/active_control.json',{'status':'lmo_comparison_gate_active','control_manifest':str(control_manifest),
                      'control_manifest_sha256':sha256(control_manifest),'policy_sha256':manifest['policy_sha256'],
                      'execution_revision_sha256':manifest['execution_revision_sha256'],'pid':os.getpid()})
    try:return prior.run(config_path,revision_manifest,transition_manifest,validation_receipt)
    finally:prior.load_base=original_load


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--control-manifest',required=True,type=Path);p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--revision-manifest',required=True,type=Path);p.add_argument('--transition-manifest',required=True,type=Path)
    p.add_argument('--validation-receipt',required=True,type=Path);a=p.parse_args(argv)
    return run(a.config,a.control_manifest,a.policy,a.revision_manifest,a.transition_manifest,a.validation_receipt)

if __name__=='__main__':main()