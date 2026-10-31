from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_own as m

FIELDS = ['scene_id','im_id','obj_id','score','R','t','time']


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def save_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)


@pytest.fixture
def case(tmp_path,monkeypatch):
    output=tmp_path/'outputs/freezev2_public_repo_20261030/own_inference_v1'
    dataset='lmo';folder=output/'datasets'/dataset;folder.mkdir(parents=True)
    source=tmp_path/'public';source.mkdir()
    pins={'configuration_sha256':'configuration','public_revision':m.ref.PUBLIC_REVISION,'public_tracked_tree_sha256':'publictree'}
    targets=[{'scene_id':1,'im_id':0,'obj_id':1,'inst_count':2},{'scene_id':1,'im_id':1,'obj_id':2,'inst_count':1}]
    target_path=tmp_path/'official_targets.json';dump(target_path,targets)
    pose={'score':.7,'R':np.eye(3).tolist(),'t_mm':[10.,20.,500.],'time':.125}
    csv_row={'scene_id':1,'im_id':0,'obj_id':1,'score':.7,'R':'1 0 0 0 1 0 0 0 1','t':'10 20 500','time':.125}
    csv_path=folder/'freezev2public_lmo-test.csv';save_csv(csv_path,[csv_row])
    rows=[]
    for index,target in enumerate(targets):
        sub=folder/'targets'/str(index);sub.mkdir(parents=True)
        path=sub/'receipt.json';original_csv=sub/'freezev2public_lmo-test.csv' if index==0 else None
        if original_csv:save_csv(original_csv,[csv_row])
        row={**target,'status':'complete' if index==0 else 'failed','poses':[pose] if index==0 else [],
             'selected_count':1 if index==0 else 0,'target_receipt':str(path),
             'original_csv':str(original_csv) if original_csv else None,'original_result':None,
             'original_csv_sha256':m.ref.sha256(original_csv) if original_csv else None}
        receipt={'target':target,'source_pins':dict(pins),'runtime_s':.25,**{k:row[k] for k in ('status','poses','selected_count','original_csv','original_result','original_csv_sha256')}}
        dump(path,receipt);row['target_receipt_sha256']=m.ref.sha256(path);rows.append(row)
    ledger_path=folder/'target_ledger.json'
    ledger={'dataset':dataset,'status':'all_targets_recorded','expected_object_image_targets':2,'requested_instances':3,
        'targets':rows,'csv':str(csv_path),'csv_sha256':m.ref.sha256(csv_path),'source_pins':dict(pins)}
    dump(ledger_path,ledger)
    read_calls=[]
    def original_reader(source,path):
        read_calls.append(str(path))
        with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))
    monkeypatch.setattr(m,'checked_original_rows',original_reader)
    return {'root':tmp_path,'output':output,'source':source,'pins':pins,'targets':target_path,
            'csv':csv_path,'ledger_path':ledger_path,'ledger':ledger,'read_calls':read_calls}


def audit(c):
    return m.audit_ledger(c['output'],'lmo',c['csv'],c['ledger_path'],c['targets'],c['pins'],c['source'])


def test_all_failed_targets_remain_in_exact_denominator(case):
    result=audit(case)
    assert result['official_object_image_targets']==2 and result['requested_instances']==3
    assert result['recorded_targets']==2 and result['saved_poses']==1
    assert result['terminal_status_counts']=={'complete':1,'failed':1}
    assert result['targets_without_saved_poses']==1 and result['failed_and_blocked_remain_in_denominator']
    assert len(case['read_calls'])==2


def test_missing_target_cannot_count_as_full_dataset(case):
    case['ledger']['targets'].pop();dump(case['ledger_path'],case['ledger'])
    with pytest.raises(ValueError,match='Missing official'):audit(case)


@pytest.mark.parametrize('defect',['duplicate','unknown','instances','pending','wrong_config','CSV_hash','receipt_hash','receipt_poses'])
def test_exact_target_and_source_identity_is_required(case,defect):
    ledger=case['ledger'];row=ledger['targets'][0]
    if defect=='duplicate':ledger['targets'].append(dict(row))
    if defect=='unknown':row['obj_id']=99
    if defect=='instances':row['inst_count']=1
    if defect=='pending':row['status']='pending'
    if defect=='wrong_config':ledger['source_pins']['configuration_sha256']='different'
    if defect=='CSV_hash':ledger['csv_sha256']='changed'
    if defect=='receipt_hash':row['target_receipt_sha256']='changed'
    if defect=='receipt_poses':row['poses'][0]['t_mm'][2]+=1
    dump(case['ledger_path'],ledger)
    with pytest.raises(ValueError):audit(case)


def test_changed_aggregate_pose_rejected_even_when_ledger_hash_updated(case):
    with case['csv'].open(newline='') as stream:rows=list(csv.DictReader(stream))
    rows[0]['t']='10 20 999';save_csv(case['csv'],rows)
    case['ledger']['csv_sha256']=m.ref.sha256(case['csv']);dump(case['ledger_path'],case['ledger'])
    with pytest.raises(ValueError,match='Aggregate CSV poses'):audit(case)


def test_authors_submission_is_rejected(case):
    author=case['output']/'author_predictions/freezev21_lmo-test.csv';save_csv(author,[])
    with pytest.raises(ValueError,match='Author/reference'):
        m.audit_ledger(case['output'],'lmo',author,case['ledger_path'],case['targets'],case['pins'],case['source'])
    with pytest.raises(ValueError,match='inside'):
        m.own_path(case['root']/'earlier_results.csv',case['output'],'CSV')


def test_public_nine_significant_digit_export_precision_is_used():
    a=m.pose_signature((1,2,3),.123456789012,np.eye(3),[1.,2.,3.],.111111111122)
    b=m.pose_signature((1,2,3),float('.123456789'),np.eye(3),[1.,2.,3.],float('.111111111'))
    assert a==b
    with pytest.raises(ValueError,match='Nonfinite'):m.pose_signature((1,2,3),float('nan'),np.eye(3),[1,2,3],1)
    with pytest.raises(ValueError,match='proper rigid'):m.pose_signature((1,2,3),1,np.diag([1,1,-1]),[1,2,3],1)


@pytest.fixture
def evaluation_case(case,monkeypatch):
    toolkit=case['root']/'toolkit';toolkit.mkdir()
    source_id={'tracked_tree_sha256':'publictree'};toolkit_id={'tracked_tree_sha256':'toolkittree'}
    runtime={'python':'/recorded/venv/bin/python','dataset_view':str(case['root']/'GT_evaluation_only'),
        'environment':{'EGL_PLATFORM':'surfaceless'},'datasets':{'lmo':{'ready':True}}}
    ctx={'root':case['root'],'output':case['output'],'dataset':'lmo','source':case['source'],'toolkit':toolkit,
         'runtime':runtime,'targets':case['targets'],'pins':case['pins'],'source_identity':source_id,'toolkit_identity':toolkit_id}
    monkeypatch.setattr(m,'context',lambda *args:ctx)
    monkeypatch.setattr(m.ref,'git_identity',lambda path,revision:source_id if Path(path)==case['source'] else toolkit_id)
    class Durable:
        atomic_json=staticmethod(dump)
        def require_disk(self,path,minimum):assert minimum==15
    class Owned:
        calls=0;returncode=0
        def start_ticks(self,pid):return 'actual-start-ticks'
        def run_owned(self,command,env,folder,lock,heartbeat):
            self.calls+=1;self.command=command;self.env=env;self.lock=lock
            assert not lock.closed
            heartbeat(444)
            scores={name:.25 for name in m.METRICS};scores['bop19_average_time_per_image']=.125
            dump(folder/'official_evaluation'/case['csv'].stem/'scores_bop19.json',scores)
            return self.returncode
    lock_path=case['root']/'outputs/freezev2_public_repo_20261030/base_public_v1/compute.lock'
    lock_path.parent.mkdir(parents=True);stream=lock_path.open('a+b')
    yield case,ctx,Durable(),Owned(),stream
    stream.close()


def evaluate(c,durable,owned,lock):
    return m.evaluate_dataset(c['root'],c['output'],'lmo',c['csv'],{},c['ledger_path'],lock,durable,owned)


def test_original_own_submission_evaluator_uses_inherited_lock(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case
    receipt=evaluate(c,durable,owned,lock)
    assert receipt['status']=='complete' and receipt['authors_predictions_used'] is False
    assert owned.command[0]=='/recorded/venv/bin/python'
    assert owned.command[2]=='evaluate-submission' and 'evaluate-reference' not in owned.command
    assert owned.command[owned.command.index('--num-workers')+1]=='1'
    assert owned.lock is lock and owned.env['BOP_PATH']==ctx['runtime']['dataset_view']
    assert receipt['coverage']['requested_instances']==3 and receipt['metrics'][m.METRICS[0]]==.25


def test_matching_completed_receipt_is_reused(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case
    first=evaluate(c,durable,owned,lock);second=evaluate(c,durable,owned,lock)
    assert first==second and owned.calls==1


def test_private_GT_still_requires_complete_ownCSV_and_ledger(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case
    ctx['runtime']['datasets']['lmo']={'GT_withheld':True}
    result=evaluate(c,durable,owned,lock)
    assert result['status']=='metrics_unavailable_private_GT' and result['metrics'] is None
    assert result['coverage']['recorded_targets']==2 and owned.calls==0


def test_evaluator_failure_is_recorded_without_dropping_targets(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case;owned.returncode=9
    with pytest.raises(RuntimeError):evaluate(c,durable,owned,lock)
    result=m.ref.read(c['output']/'datasets/lmo/own_evaluation_receipt.json')
    assert result['status']=='failed_original_evaluator' and result['metrics'] is None
    assert result['coverage']['recorded_targets']==2


def test_changed_evaluation_source_cannot_reuse_scores(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case;evaluate(c,durable,owned,lock)
    ctx['pins']['evaluation_helper_sha256']='newhelper'
    with pytest.raises(ValueError,match='separate output'):evaluate(c,durable,owned,lock)
    assert owned.calls==1


def test_summary_has_no_false_seven_dataset_aggregate(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case;evaluate(c,durable,owned,lock)
    result=m.write_summary(c['root'],c['output'],durable)
    assert result['status']=='incomplete' and result['seven_dataset_aggregate'] is None
    assert result['published_performance_reproduced'] is False
    assert result['datasets']['tless']['status']=='incomplete'


def test_configuration_must_match_canonical_manifest(tmp_path):
    path=tmp_path/'experiments/freezev2_public_repo_20261030/public_inference_v1.json'
    dump(path,{'seed':0})
    result=m.configuration_pins(tmp_path,{'seed':0})
    assert result['configuration_sha256']==m.ref.digest({'seed':0})
    with pytest.raises(ValueError,match='canonical'):m.configuration_pins(tmp_path,{'seed':1})

def test_image_time_includes_zero_pose_failed_attempts(case):
    ledger=case['ledger'];target_rows=m.ref.read(case['targets']);target_rows[1]['im_id']=0;dump(case['targets'],target_rows)
    failed=ledger['targets'][1];failed['im_id']=0
    receipt_path=Path(failed['target_receipt']);record=m.ref.read(receipt_path);record['target']['im_id']=0
    dump(receipt_path,record);failed['target_receipt_sha256']=m.ref.sha256(receipt_path)
    ledger['targets'][0]['poses'][0]['time']=.375
    with case['csv'].open(newline='') as stream:rows=list(csv.DictReader(stream))
    rows[0]['time']=.375;save_csv(case['csv'],rows);ledger['csv_sha256']=m.ref.sha256(case['csv'])
    dump(case['ledger_path'],ledger)
    result=audit(case)
    assert result['image_runtime_s_by_scene_image']=={'000001/000000':.375}
    assert result['saved_poses']==1 and result['requested_instances']==3
    ledger['targets'][0]['poses'][0]['time']=.125;dump(case['ledger_path'],ledger)
    with pytest.raises(ValueError,match='Aggregate image runtime'):audit(case)


def test_original_multimask_selected_geometry_is_verified(case):
    row=case['ledger']['targets'][0];result=Path(row['target_receipt']).with_name('original_result.json')
    pose=row['poses'][0]
    dump(result,{'selected':[{'final_score':pose['score'],'R':pose['R'],'t_mm':pose['t_mm']}]})
    row['original_result']=str(result)
    receipt=Path(row['target_receipt']);record=m.ref.read(receipt);record['original_result']=str(result);dump(receipt,record)
    row['target_receipt_sha256']=m.ref.sha256(receipt);dump(case['ledger_path'],case['ledger'])
    assert audit(case)['saved_poses']==1
    dump(result,{'selected':[]})
    with pytest.raises(ValueError,match='selected geometry'):audit(case)


def test_private_GT_cannot_skip_missing_target_audit(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case
    ctx['runtime']['datasets']['lmo']={'GT_withheld':True}
    c['ledger']['targets'].pop();dump(c['ledger_path'],c['ledger'])
    with pytest.raises(ValueError,match='Missing official'):evaluate(c,durable,owned,lock)
    assert owned.calls==0


def test_invalid_official_scores_are_recorded_as_validation_failure(evaluation_case):
    c,ctx,durable,owned,lock=evaluation_case
    run=owned.run_owned
    def invalid(*args):
        code=run(*args)
        score=c['output']/'datasets/lmo/official_evaluation'/c['csv'].stem/'scores_bop19.json'
        values=m.ref.read(score);values[m.METRICS[0]]=float('nan');dump(score,values)
        return code
    owned.run_owned=invalid
    with pytest.raises(ValueError,match='nonfinite'):evaluate(c,durable,owned,lock)
    receipt=m.ref.read(c['output']/'datasets/lmo/own_evaluation_receipt.json')
    assert receipt['status']=='failed_evaluation_validation' and receipt['metrics'] is None