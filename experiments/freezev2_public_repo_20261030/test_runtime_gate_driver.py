import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import runtime_gate_driver as gate


def policy(**kw):return {'schema':'lmo-paper-stop-policy-v1','status':'configured','paper_ar':0.694,'mode':'absolute_percentage_points','tolerance_pp':1,**kw}

def atomic(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))

@pytest.mark.parametrize('ar,aligned',[(.694,True),(.684,True),(.704,True),(.683999,False),(.704001,False)])
def test_percentage_point_boundary(ar,aligned):
    v=gate.decide(ar,policy());assert v['aligned'] is aligned;assert v['continue'] is aligned

@pytest.mark.parametrize('ar,aligned',[(.6935,True),(.69449,True),(.69349,False),(.6945,False)])
def test_explicit_rounded_comparison(ar,aligned):
    assert gate.decide(ar,policy(mode='rounded_one_decimal'))['aligned'] is aligned


def test_two_point_policy_is_explicit():
    assert gate.decide(.674,policy(tolerance_pp=2))['continue'] is True
    assert gate.decide(.67399,policy(tolerance_pp=2))['continue'] is False


def test_pending_policy_never_defaults():
    v=gate.decide(.694,policy(status='awaiting_user'));assert v['continue'] is False;assert v['aligned'] is None
    assert v['status']=='stopped_lmo_awaiting_alignment_tolerance'

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,1.1,True,None])
def test_invalid_score_cannot_continue(value):
    with pytest.raises(ValueError):gate.decide(value,policy())

@pytest.mark.parametrize('value',[float('nan'),-1,True,None])
def test_invalid_tolerance_rejected(value):
    with pytest.raises(ValueError):gate.validate_policy(policy(tolerance_pp=value))


def test_author_accurate_comparator_refused():
    with pytest.raises(ValueError):gate.decide(.771,policy(paper_ar=.771))


@pytest.fixture
def fake_base(tmp_path,monkeypatch):
    class IdentityError(Exception):pass
    calls=[]
    def original(ctx,dataset,lock):calls.append(dataset);return {'dataset':dataset}
    base=SimpleNamespace(run_dataset=original,IdentityError=IdentityError,atomic_text=lambda p,v:(Path(p).parent.mkdir(parents=True,exist_ok=True),Path(p).write_text(v)))
    ctx={'output':tmp_path,'root':tmp_path,'durable':SimpleNamespace(atomic_json=atomic)}
    monkeypatch.setitem(sys.modules,'evaluate_own',SimpleNamespace(write_summary=lambda *a,**k:calls.append('summary')))
    monkeypatch.setattr(gate,'verify_control',lambda *a,**k:({},tmp_path,{'policy_sha256':'policy'},None))
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    return base,ctx,calls,manifest

@pytest.mark.parametrize('status',gate.EXPECTED_STOPS)
def test_gate_stops_before_any_other_dataset(fake_base,monkeypatch,status):
    base,ctx,calls,m=fake_base
    monkeypatch.setattr(gate,'comparison_gate',lambda *a:{'status':status,'continue':False,'aligned':False,'own_ar_percent':50.0})
    gate.install_gate(base,m,ctx['output']/'policy.json')
    with pytest.raises(SystemExit) as e:
        for dataset in ['lmo','tless']:base.run_dataset(ctx,dataset,None)
    assert e.value.code==0;assert calls==['lmo','summary']
    assert gate.read(ctx['output']/'status.json')['status']==status
    assert gate.read(ctx['output']/'results_summary.json')['next_datasets_started'] is False
    assert (ctx['output']/'run_control/LMO_PAPER_COMPARISON.md').exists()


def test_lmo_exception_cannot_fall_through_to_tless(fake_base):
    base,ctx,calls,m=fake_base
    def broken(*a):raise ValueError('official evaluation failed')
    base.run_dataset=broken;gate.install_gate(base,m,ctx['output']/'policy.json')
    with pytest.raises(SystemExit):base.run_dataset(ctx,'lmo',None)
    assert gate.read(ctx['output']/'status.json')['status']=='stopped_lmo_evaluation_unavailable'
    assert calls==['summary']


def test_missing_gate_is_fatal_before_tless(fake_base):
    base,ctx,calls,m=fake_base;gate.install_gate(base,m,ctx['output']/'policy.json')
    with pytest.raises(base.IdentityError):base.run_dataset(ctx,'tless',None)
    assert calls==[]


def test_unaligned_or_wrong_control_cannot_start_tless(fake_base):
    base,ctx,calls,m=fake_base;gate.install_gate(base,m,ctx['output']/'policy.json')
    atomic(ctx['output']/'run_control/lmo_comparison_gate.json',{'continue':True,'control_manifest_sha256':'wrong'})
    with pytest.raises(base.IdentityError):base.run_dataset(ctx,'tless',None)
    assert calls==[]


def test_aligned_gate_continues_next_dataset(fake_base,monkeypatch):
    base,ctx,calls,m=fake_base
    def compared(*args):
        v={'status':'lmo_aligned_continue','continue':True,'control_manifest_sha256':gate.sha256(m),'policy_sha256':'policy'}
        atomic(ctx['output']/'run_control/lmo_comparison_gate.json',v);return v
    monkeypatch.setattr(gate,'comparison_gate',compared);gate.install_gate(base,m,ctx['output']/'policy.json')
    for dataset in ['lmo','tless']:base.run_dataset(ctx,dataset,None)
    assert calls==['lmo','tless']


def test_policy_or_source_integrity_error_is_fatal(fake_base,monkeypatch):
    base,ctx,calls,m=fake_base
    def invalid(*args):raise ValueError('bound policy changed')
    monkeypatch.setattr(gate,'comparison_gate',invalid);gate.install_gate(base,m,ctx['output']/'policy.json')
    with pytest.raises(base.IdentityError):base.run_dataset(ctx,'lmo',None)
    assert calls==['lmo']


@pytest.fixture
def receipt_context(tmp_path,monkeypatch):
    folder=tmp_path/'datasets/lmo';folder.mkdir(parents=True)
    csv=folder/'freezev2public_lmo-test.csv';csv.write_text('saved own predictions')
    ledger=folder/'target_ledger.json';ledger.write_text('saved full ledger')
    score=folder/'official_evaluation'/csv.stem/'scores_bop19.json';score.parent.mkdir(parents=True)
    metrics={'bop19_average_recall':.694,'bop19_average_recall_vsd':.6,'bop19_average_recall_mssd':.7,'bop19_average_recall_mspd':.782}
    score.write_text(json.dumps(metrics));pins={'configuration_sha256':gate.CONFIGURATION_SHA}
    audit={'official_object_image_targets':1445,'recorded_targets':1445,'requested_instances':1445,'missing_targets':0,
        'failed_and_blocked_remain_in_denominator':True,'target_receipts_sha256':{},'original_generated_CSVs_sha256':{},
        'terminal_status_counts':{'complete':1445},'targets_without_saved_poses':20,'saved_poses':1425}
    sp={**pins,'CSV_sha256':gate.sha256(csv),'target_ledger_sha256':gate.sha256(ledger),'ledger_artifacts_sha256':gate.digest({'receipts':{},'original_CSVs':{}})}
    v={'schema':'unchanged-public-own-prediction-evaluation-v1','dataset':'lmo','status':'complete','authors_predictions_used':False,
        'CSV':str(csv),'ledger':str(ledger),'score_file':str(score),'score_sha256':gate.sha256(score),'source_pins':sp,
        'metrics':metrics,'coverage':audit,'protocol':{'visib_gt_min':-1,'task':'BOP19 model-based 6D localization'}}
    rp=folder/'own_evaluation_receipt.json';atomic(rp,v)
    fake=SimpleNamespace(context=lambda *a:{'targets':tmp_path/'targets.json','pins':pins,'source':tmp_path/'source'},
                         audit_ledger=lambda *a:copy.deepcopy(audit),pose_metrics=lambda value:value)
    monkeypatch.setitem(sys.modules,'evaluate_own',fake)
    return {'output':tmp_path,'root':tmp_path,'config':{}},v,rp,audit,csv,score


def test_actual_metric_receipt_is_bound(receipt_context):
    ctx,v,rp,audit,csv,score=receipt_context
    ar,evidence=gate.audit_completed_lmo(ctx);assert ar==.694;assert evidence['coverage']['recorded_targets']==1445

@pytest.mark.parametrize('change',['not_complete','author','score_hash','csv_hash','missing_targets','filtered_failures','protocol'])
def test_metric_gate_rejects_unverified_score(receipt_context,change):
    ctx,v,rp,audit,csv,score=receipt_context
    if change=='not_complete':v['status']='running'
    elif change=='author':v['authors_predictions_used']=True
    elif change=='score_hash':v['score_sha256']='wrong'
    elif change=='csv_hash':csv.write_text('changed')
    elif change=='missing_targets':audit['recorded_targets']=1444
    elif change=='filtered_failures':audit['failed_and_blocked_remain_in_denominator']=False
    elif change=='protocol':v['protocol']['visib_gt_min']=.1
    atomic(rp,v)
    with pytest.raises(ValueError):gate.audit_completed_lmo(ctx)


def test_expected_supervisor_stop_states_are_explicit():
    assert 'stopped_lmo_not_aligned' in gate.EXPECTED_STOPS
    assert 'suite_pass_finished' not in gate.EXPECTED_STOPS

def test_stale_continue_cannot_bypass_tampered_policy(fake_base,monkeypatch):
    base,ctx,calls,m=fake_base
    atomic(ctx['output']/'run_control/lmo_comparison_gate.json',{'continue':True,'control_manifest_sha256':gate.sha256(m),'policy_sha256':'policy'})
    def changed(*a,**k):raise ValueError('bound policy changed')
    monkeypatch.setattr(gate,'verify_control',changed);gate.install_gate(base,m,ctx['output']/'policy.json')
    with pytest.raises(base.IdentityError):base.run_dataset(ctx,'tless',None)
    assert calls==[]

def test_summary_failure_cannot_fall_through(fake_base,monkeypatch):
    base,ctx,calls,m=fake_base
    monkeypatch.setattr(gate,'comparison_gate',lambda *a:{'status':'stopped_lmo_not_aligned','continue':False})
    def broken(*a,**k):raise OSError('report disk error')
    monkeypatch.setitem(sys.modules,'evaluate_own',SimpleNamespace(write_summary=broken))
    gate.install_gate(base,m,ctx['output']/'policy.json')
    with pytest.raises(base.IdentityError):base.run_dataset(ctx,'lmo',None)
    assert calls==['lmo']


def test_sealed_folder_original_launcher_bootstrap_isolated(tmp_path):
    import shutil,subprocess
    control=tmp_path/'sealed_control';control.mkdir();snapshot=tmp_path/'run/source_snapshot';snapshot.mkdir(parents=True)
    study=Path(gate.__file__).parent
    for name in ['runtime_gate_driver.py','launch_runtime_gate.py']:shutil.copyfile(study/name,control/name)
    (snapshot/'reference_gate.py').write_text('MARKER="original snapshot reference"\n')
    (snapshot/'launch_inference.py').write_text('import reference_gate as ref\nGUARD_IMPORT=ref.MARKER\n')
    code="""
import pathlib,sys,types
control=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2])
sys.path.insert(0,str(control))
import launch_runtime_gate as launcher
assert 'reference_gate' not in sys.modules
calls=[]
def load_base(root):
 calls.append(str(root));sys.path.insert(0,str(pathlib.Path(root)/'source_snapshot'))
prior=types.SimpleNamespace(load_base=load_base)
original=launcher.load_original_launcher(out,prior)
assert calls==[str(out)]
assert original.GUARD_IMPORT=='original snapshot reference'
assert pathlib.Path(original.ref.__file__).parent==out/'source_snapshot'
print('isolated sealed-folder original launcher bootstrap passed')
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(control),str(snapshot.parent)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'bootstrap passed' in result.stdout
