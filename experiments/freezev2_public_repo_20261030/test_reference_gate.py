from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_gate_under_test',HERE/'reference_gate.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

EXPECTED={'ar':.771,'vsd':.623,'mssd':.829,'mspd':.861,'time':29.805}
SCORES={g.METRICS[k]:v for k,v in EXPECTED.items() if k!='time'}
SCORES['bop19_average_time_per_image']=123.456


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Durable:
    def __init__(self):self.lock_active=False;self.events=[];self.token=object()
    atomic_json=staticmethod(dump)
    def require_disk(self,path,minimum):self.events.append(('disk',minimum))
    @contextmanager
    def worker_lock(self,path):
        if self.lock_active:raise RuntimeError('Duplicate GPU/evaluator owner')
        self.lock_active=True;self.events.append(('lock_enter',str(path)))
        try:yield self.token
        finally:self.events.append(('lock_exit',str(path)));self.lock_active=False


@pytest.fixture
def case(tmp_path,monkeypatch):
    out=tmp_path/'output';out.mkdir()
    source=tmp_path/'public';source.mkdir();(source/'run_bop.py').write_text('original marker')
    toolkit=tmp_path/'toolkit';toolkit.mkdir()
    view=tmp_path/'view';view.mkdir()
    csv_path=out/'author_predictions/freezev21_lmo-test.csv';csv_path.parent.mkdir();csv_path.write_text('untouched author marker')
    runtime={'python':str(Path(sys.prefix)/'bin/python'),'toolkit_root':str(toolkit),'dataset_view':str(view),
             'toolkit_revision':g.TOOLKIT_REVISION,'renderer_type':'vispy','protocol':{'visib_gt_min':-1},
             'environment':{'EGL_PLATFORM':'surfaceless','KEEP_ENTRY':'preserve'},'datasets':{'lmo':{'ready':True}}}
    source_id={'revision':g.PUBLIC_REVISION,'tracked_tree_sha256':'publictree'}
    toolkit_id={'revision':g.TOOLKIT_REVISION,'tracked_tree_sha256':'toolkittree'}
    ctx={'root':tmp_path,'output':out,'source':source,'runtime':runtime,'source_identity':source_id,
         'toolkit_identity':toolkit_id,'specs':{'lmo':{'filename':csv_path.name,'expected':EXPECTED}}}
    pins={'CSV_sha256':g.sha256(csv_path),'public':'source pin','runtime':'runtime pin'}
    def pin_reader(current,dataset):return csv_path,view/'targets.json',dict(pins),{'requested_instances':7},{}
    monkeypatch.setattr(g,'dataset_pins',pin_reader)
    monkeypatch.setattr(g,'require_runtime',lambda runtime:None)
    monkeypatch.setattr(g,'git_identity',lambda path,revision:source_id if Path(path)==source else toolkit_id)
    durable=Durable()
    class Owned:
        calls=0;returncode=0;scores=dict(SCORES)
        @staticmethod
        def start_ticks(pid):return 'original-start-ticks'
        def run_owned(self,command,env,folder,lock,heartbeat):
            assert durable.lock_active and lock is durable.token
            self.calls+=1;durable.events.append(('owned',command,dict(env)))
            heartbeat(900)
            score=Path(folder)/'official_eval'/csv_path.stem/'scores_bop19.json'
            dump(score,self.scores)
            return self.returncode
    owned=Owned()
    return ctx,durable,owned,pins,csv_path


def test_original_command_and_inherited_environment(case):
    ctx,durable,owned,_,csv_path=case
    command,env=g.command_and_environment(ctx['runtime'],ctx['source'],'lmo',csv_path,ctx['output']/'reference/lmo')
    assert command[1]==str(ctx['source']/'run_bop.py')
    assert command[2]=='evaluate-reference'
    assert command[command.index('--result-csv')+1]==str(csv_path)
    assert command[command.index('--num-workers')+1]=='1'
    assert 'cohort' not in ' '.join(command)
    assert env['KEEP_ENTRY']=='preserve' and env['EGL_PLATFORM']=='surfaceless'
    assert env['PATH'].split(g.os.pathsep)[0]==str(Path(command[0]).parent)


@pytest.mark.parametrize('field',['ar','vsd','mssd','mspd'])
def test_each_pose_metric_is_checked(field):
    scores=dict(SCORES);scores[g.METRICS[field]]+=.001
    comparison=g.metric_comparison(scores,EXPECTED)
    assert not comparison[g.METRICS[field]]['passed']
    assert sum(not row['passed'] for row in comparison.values())==1


def test_rounding_tolerance_and_timing_is_not_accuracy_gate():
    scores={key:value+.0005 for key,value in SCORES.items() if key in g.METRICS.values()}
    scores['bop19_average_time_per_image']=99999
    assert all(row['passed'] for row in g.metric_comparison(scores,EXPECTED).values())
    scores[g.METRICS['ar']]+=.00001
    assert not g.metric_comparison(scores,EXPECTED)[g.METRICS['ar']]['passed']


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,1.01])
def test_nonfinite_or_impossible_score_rejected(bad):
    scores=dict(SCORES);scores[g.METRICS['ar']]=bad
    with pytest.raises(ValueError):g.metric_comparison(scores,EXPECTED)


def test_guard_holds_compute_lock_and_valid_receipt_is_reused(case,monkeypatch):
    ctx,durable,owned,pins,csv_path=case
    monkeypatch.setattr(g,'context',lambda root,output:ctx)
    monkeypatch.setattr(g,'pinned_guards',lambda root:(durable,owned))
    first=g.run(ctx['root'],ctx['output'],('lmo',))
    assert first['results'][0]['status']=='complete' and owned.calls==1
    assert not durable.lock_active
    assert [event[0] for event in durable.events].index('lock_enter')<[event[0] for event in durable.events].index('owned')<[event[0] for event in durable.events].index('lock_exit')
    result=g.run(ctx['root'],ctx['output'],('lmo',))
    assert owned.calls==1 and result['results'][0]['metrics']=={k:SCORES[k] for k in g.METRICS.values()}
    assert result['results'][0]['timing_tolerance_applied'] is False


def test_changed_pins_cannot_reuse_or_merge_prior_evaluation(case):
    ctx,durable,owned,pins,_=case
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:g.run_dataset(ctx,'lmo',durable,owned,lock)
    pins['CSV_sha256']='different author CSV'
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:
        with pytest.raises(ValueError,match='incompatible'):g.run_dataset(ctx,'lmo',durable,owned,lock)
    assert owned.calls==1


def test_changed_score_bytes_are_not_accepted_as_a_valid_receipt(case):
    ctx,durable,owned,pins,csv_path=case
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:receipt=g.run_dataset(ctx,'lmo',durable,owned,lock)
    score=ctx['output']/'reference/lmo/official_eval'/csv_path.stem/'scores_bop19.json'
    altered=dict(SCORES);altered[g.METRICS['ar']]=0;dump(score,altered)
    assert not g.reusable(receipt,pins,score,EXPECTED)


def test_failed_original_child_is_recorded(case):
    ctx,durable,owned,_,_=case;owned.returncode=2
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:
        with pytest.raises(RuntimeError,match='Original reference'):g.run_dataset(ctx,'lmo',durable,owned,lock)
    receipt=g.read(ctx['output']/'reference/lmo/metric_receipt.json')
    assert receipt['status']=='failed_original_evaluator' and receipt['evaluator_returncode']==2


def test_wrong_reference_score_is_saved_and_blocks_gate(case):
    ctx,durable,owned,_,_=case;owned.scores=dict(SCORES);owned.scores[g.METRICS['mssd']]=.4
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:
        with pytest.raises(RuntimeError,match='differs'):g.run_dataset(ctx,'lmo',durable,owned,lock)
    receipt=g.read(ctx['output']/'reference/lmo/metric_receipt.json')
    assert receipt['status']=='reference_score_mismatch' and not receipt['comparison'][g.METRICS['mssd']]['passed']


@pytest.mark.parametrize('dataset',['hb','itodd'])
def test_private_gt_is_skipped_with_null_metrics(case,dataset):
    ctx,durable,owned,_,_=case
    with durable.worker_lock(ctx['output']/'compute.lock') as lock:receipt=g.run_dataset(ctx,dataset,durable,owned,lock)
    assert receipt['status']=='skipped_private_GT' and receipt['metrics'] is None and owned.calls==0


def test_original_specs_are_literal_and_not_executed(tmp_path):
    source=tmp_path/'public';(source/'freezev2').mkdir(parents=True)
    refs={dataset:{'filename':dataset+'.csv','expected':EXPECTED} for dataset in g.PUBLIC_DATASETS+g.PRIVATE_DATASETS}
    (source/'freezev2/bop.py').write_text("raise RuntimeError('must not execute')\nBOP_TOOLKIT_COMMIT="+repr(g.TOOLKIT_REVISION)+'\nREFERENCE_SUBMISSIONS='+repr(refs))
    assert g.original_specs(source)==refs


def test_targets_cannot_drop_duplicate_or_change_instance_denominator(tmp_path):
    path=tmp_path/'targets.json';row={'scene_id':1,'im_id':2,'obj_id':3,'inst_count':2}
    dump(path,[row]);assert g.target_counts(path)=={'object_image_targets':1,'requested_instances':2,'images':1}
    dump(path,[row,row])
    with pytest.raises(ValueError):g.target_counts(path)
@pytest.mark.parametrize('physical,canonical,expected',[
    (b'x = 1\n',b'x = 1\n','byte_identical'),
    (b'x = 1\r\ny = 2\r\n',b'x = 1\ny = 2\n','LF_git_blob_to_CRLF_checkout_only'),
    (b'\x00\r\n',b'\x00\r\n','byte_identical'),
])
def test_only_exact_or_checkout_crlf_conversion_is_accepted(physical,canonical,expected):
    assert g.verify_blob_content(physical,canonical)==expected


@pytest.mark.parametrize('physical,canonical',[
    (b'x = 2\r\n',b'x = 1\n'),
    (b'x = 1\r',b'x = 1\n'),
    (b'\x00\r\n',b'\x00\n'),
    (b'x = 1\n',b'x = 1\r\n'),
])
def test_crlf_policy_never_hides_semantic_or_binary_changes(physical,canonical):
    with pytest.raises(ValueError,match='beyond checkout'):g.verify_blob_content(physical,canonical)


def test_real_git_crlf_checkout_is_pinned_without_source_or_config_changes(tmp_path):
    import subprocess
    repo=tmp_path/'source';repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-c','core.autocrlf=false','-C',str(repo),*args],stderr=subprocess.STDOUT,text=True)
    git('init','-q')
    file=repo/'example.py';file.write_bytes(b'value = 1\n')
    git('add','example.py')
    git('-c','user.name=Reference gate test','-c','user.email=reference-gate@example.invalid','commit','-qm','fixture')
    revision=git('rev-parse','HEAD').strip()
    file.write_bytes(b'value = 1\r\n')
    physical_before=file.read_bytes();config_before=(repo/'.git/config').read_bytes()
    identity=g.git_identity(repo,revision)
    assert identity['checkout_line_ending_only_files']==['example.py']
    assert identity['tracked_files']['example.py']!=identity['canonical_git_blob_files']['example.py']
    assert file.read_bytes()==physical_before and (repo/'.git/config').read_bytes()==config_before
    file.write_bytes(b'value = 2\r\n')
    with pytest.raises(ValueError):g.git_identity(repo,revision)

def test_command_preserves_venv_executable_even_when_resolve_would_follow_symlink(case,monkeypatch):
    ctx,_,_,_,csv_path=case
    interpreter=Path(ctx['runtime']['python'])
    original=Path.resolve
    def resolve(path,*args,**kwargs):
        if path==interpreter:return Path('/system/python')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',resolve)
    command,env=g.command_and_environment(ctx['runtime'],ctx['source'],'lmo',csv_path,ctx['output'])
    assert command[0]==str(interpreter)
    assert env['PATH'].split(g.os.pathsep)[0]==str(interpreter.parent)


def test_runtime_guard_compares_venv_prefix_before_following_executable_symlink(tmp_path,monkeypatch):
    from types import SimpleNamespace
    interpreter=tmp_path/'venv/bin/python'
    monkeypatch.setattr(g,'evaluation_python',lambda runtime:interpreter)
    monkeypatch.setattr(g,'os',SimpleNamespace(name='posix'))
    monkeypatch.setattr(g,'sys',SimpleNamespace(prefix=str(interpreter.parent.parent)))
    g.require_runtime({})
    monkeypatch.setattr(g,'sys',SimpleNamespace(prefix=str(tmp_path/'another_venv')))
    with pytest.raises(RuntimeError,match='recorded WSL evaluation Python'):g.require_runtime({})
