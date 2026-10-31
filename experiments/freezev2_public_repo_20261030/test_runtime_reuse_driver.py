"""Offline tests for the additive execution revision and unchanged legacy receipts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import runtime_reuse_driver as driver
import launch_runtime_reuse as launcher


@pytest.fixture
def sealed(tmp_path,monkeypatch):
    out=tmp_path/'run';out.mkdir();base=out/'source_snapshot';base.mkdir()
    original={'files':{},'configuration_sha256':'fixture'}
    driver.atomic_json(base/'source_manifest.json',original)
    monkeypatch.setattr(driver,'BASE_MANIFEST_SHA',driver.sha256(base/'source_manifest.json'))
    config={'paths':{'output':str(out),'compute_lock':str(tmp_path/'compute.lock'),'inference_python':'/venv/bin/python'},'resources':{'auto_restart_limit':2}}
    monkeypatch.setattr(driver,'CONFIGURATION_SHA',driver.digest(config))
    driver.atomic_json(out/'configuration.json',config)
    source=tmp_path/'source';source.mkdir()
    for name in driver.FILES:(source/name).write_text('# fixture '+name)
    manifest=driver.seal_revision(out/'configuration.json',source)
    monkeypatch.setattr(driver,'__file__',str(manifest.parent/'runtime_reuse_driver.py'))
    return SimpleNamespace(out=out,config=config,config_path=out/'configuration.json',source=source,manifest=manifest)


def transition(s,legacy=None):
    path=s.out/'execution_revisions/transition.json'
    value={'schema':'runtime-reuse-transition-v1','status':'ready_for_handoff',
           'base_configuration_sha256':driver.digest(s.config),'base_source_manifest_sha256':driver.BASE_MANIFEST_SHA,
           'execution_revision_sha256':driver.sha256(s.manifest),'legacy_receipts':legacy or {}}
    driver.atomic_json(path,value);return path,value


def receipt(s,key='s000002_i000003_o000001',value=None):
    p=s.out/'datasets/lmo/targets'/key/'receipt.json'
    driver.atomic_json(p,value or {'status':'complete','selected_count':1})
    return p


def test_seal_reuses_exact_revision(sealed):
    before=sealed.manifest.read_bytes()
    assert driver.seal_revision(sealed.config_path,sealed.source)==sealed.manifest
    assert sealed.manifest.read_bytes()==before


def test_seal_refuses_mutable_source_change(sealed):
    (sealed.source/driver.FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='differs'):driver.seal_revision(sealed.config_path,sealed.source)


def test_seal_refuses_changed_payload(sealed):
    (sealed.manifest.parent/driver.FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='changed'):driver.verify_revision(sealed.config_path,sealed.manifest)


def test_configuration_change_rejected(sealed):
    c=copy.deepcopy(sealed.config);c['changed']=True
    driver.atomic_json(sealed.config_path,c)
    with pytest.raises(ValueError,match='configuration changed'):driver.verify_revision(sealed.config_path,sealed.manifest)


def test_base_manifest_change_rejected(sealed):
    (sealed.out/'source_snapshot/source_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest changed'):driver.verify_revision(sealed.config_path,sealed.manifest)


def test_driver_must_run_from_sealed_path(sealed,monkeypatch):
    monkeypatch.setattr(driver,'__file__',str(sealed.source/'runtime_reuse_driver.py'))
    with pytest.raises(ValueError,match='sealed runtime driver'):driver.verify_revision(sealed.config_path,sealed.manifest)


def test_transition_verifies_legacy_without_rewriting(sealed):
    p=receipt(sealed);before=p.read_bytes();relative=p.relative_to(sealed.out).as_posix()
    path,v=transition(sealed,{relative:driver.sha256(p)})
    assert driver.verified_transition(sealed.config_path,sealed.manifest,path)==v
    assert p.read_bytes()==before


def test_transition_rejects_legacy_mutation(sealed):
    p=receipt(sealed);path,_=transition(sealed,{p.relative_to(sealed.out).as_posix():driver.sha256(p)})
    p.write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):driver.verified_transition(sealed.config_path,sealed.manifest,path)


def test_transition_rejects_path_escape(sealed):
    path,_=transition(sealed,{'../outside/receipt.json':'x'})
    with pytest.raises(ValueError,match='Invalid legacy'):driver.verified_transition(sealed.config_path,sealed.manifest,path)


def test_transition_rejects_wrong_revision(sealed):
    path,v=transition(sealed);v['execution_revision_sha256']='wrong';driver.atomic_json(path,v)
    with pytest.raises(ValueError,match='differs'):driver.verified_transition(sealed.config_path,sealed.manifest,path)


def test_validation_requires_actual_equivalence_and_speedup(sealed):
    p=sealed.out/'validation.json';v={'status':'passed','execution_revision_sha256':driver.sha256(sealed.manifest),'configuration_sha256':driver.digest(sealed.config),
                                   'descriptor_equivalence':True,'pose_equivalence':True,'measured_speedup':True,
                                   'helper_sha256':driver.read(sealed.manifest)['files']['dino_reuse_trial.py'],
                                   'artifacts':{str(sealed.manifest):driver.sha256(sealed.manifest)}}
    driver.atomic_json(p,v);assert driver.require_validation(sealed.config_path,sealed.manifest,p)==v
    for field in ['descriptor_equivalence','pose_equivalence','measured_speedup']:
        changed=dict(v);changed[field]=False;driver.atomic_json(p,changed)
        with pytest.raises(ValueError,match='required'):driver.require_validation(sealed.config_path,sealed.manifest,p)


def test_audit_legacy_unchanged(sealed):
    p=receipt(sealed);path,v=transition(sealed,{p.relative_to(sealed.out).as_posix():driver.sha256(p)})
    assert driver.audit_provenance(sealed.out,'lmo',driver.sha256(sealed.manifest),v,driver.sha256(path))=={'legacy':1,'runtime_reuse':0}


def test_audit_new_execution_report(sealed):
    path,v=transition(sealed);p=receipt(sealed);request=p.parent/'original_command.json';driver.atomic_json(request,{'command':['a']})
    report=p.parent/'runtime_reuse_report.json';revision=driver.sha256(sealed.manifest)
    driver.atomic_json(report,{'execution_revision_sha256':revision,'original_request_sha256':driver.sha256(request),
                               'configuration_sha256':driver.CONFIGURATION_SHA,'helper_sha256':driver.read(sealed.manifest)['files']['dino_reuse_trial.py']})
    saved=driver.read(p);saved.update(execution_revision_sha256=revision,transition_manifest_sha256=driver.sha256(path),execution_report=str(report),execution_report_sha256=driver.sha256(report));driver.atomic_json(p,saved)
    assert driver.audit_provenance(sealed.out,'lmo',revision,v,driver.sha256(path))=={'legacy':0,'runtime_reuse':1}
    report.write_text('{}')
    with pytest.raises(ValueError,match='report identity'):driver.audit_provenance(sealed.out,'lmo',revision,v,driver.sha256(path))


def test_audit_blocked_target_needs_no_execution_report(sealed):
    path,v=transition(sealed);revision=driver.sha256(sealed.manifest)
    receipt(sealed,value={'status':'blocked','original_csv':None,'execution_report':None,'execution_revision_sha256':revision,'transition_manifest_sha256':driver.sha256(path)})
    assert driver.audit_provenance(sealed.out,'lmo',revision,v,driver.sha256(path))['runtime_reuse']==1


def test_audit_unidentified_execution_rejected(sealed):
    path,v=transition(sealed);receipt(sealed)
    with pytest.raises(ValueError,match='Unidentified'):driver.audit_provenance(sealed.out,'lmo',driver.sha256(sealed.manifest),v,driver.sha256(path))


def test_audit_pending_execution_rejected(sealed):
    path,v=transition(sealed);receipt(sealed,value={'status':'running'})
    with pytest.raises(ValueError,match='Pending'):driver.audit_provenance(sealed.out,'lmo',driver.sha256(sealed.manifest),v,driver.sha256(path))


def test_audit_rejects_report_escape(sealed,tmp_path):
    path,v=transition(sealed);revision=driver.sha256(sealed.manifest);report=tmp_path/'runtime_reuse_report.json';report.write_text('{}')
    receipt(sealed,value={'status':'complete','execution_revision_sha256':revision,'transition_manifest_sha256':driver.sha256(path),'execution_report':str(report),'execution_report_sha256':driver.sha256(report)})
    with pytest.raises(ValueError,match='report identity'):driver.audit_provenance(sealed.out,'lmo',revision,v,driver.sha256(path))


def test_lock_probe_checks_both_real_locks(sealed,monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher.subprocess,'run',lambda command,**kw:calls.append(command) or SimpleNamespace(returncode=0,stdout='',stderr=''))
    launcher.probe_locks(sealed.config,sealed.out)
    assert calls[0][-1]==sealed.config['paths']['compute_lock']
    assert calls[0][-2].endswith('/suite.lock')


def test_lock_probe_failure_refuses_launch(sealed,monkeypatch):
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='',stderr='locked'))
    with pytest.raises(RuntimeError,match='real OS lock'):launcher.probe_locks(sealed.config,sealed.out)


def test_exact_native_identity():
    p=SimpleNamespace(create_time=lambda:5.0,cmdline=lambda:['python','supervisor'])
    ps=SimpleNamespace(Process=lambda pid:p,NoSuchProcess=LookupError,ZombieProcess=RuntimeError)
    assert launcher.exact_live({'pid':5,'create_time':5.0,'actual_cmdline':['python','supervisor']},ps)
    assert not launcher.exact_live({'pid':5,'create_time':6.0,'actual_cmdline':['python','supervisor']},ps)


def test_access_denied_is_not_death():
    def denied(pid):raise PermissionError('denied')
    ps=SimpleNamespace(Process=denied,NoSuchProcess=LookupError,ZombieProcess=RuntimeError)
    with pytest.raises(PermissionError):launcher.exact_live({'pid':5,'create_time':5.0,'actual_cmdline':[]},ps)

def test_resume_can_audit_one_terminal_receipt_before_later_pending(sealed):
    p=receipt(sealed);path,v=transition(sealed,{p.relative_to(sealed.out).as_posix():driver.sha256(p)})
    receipt(sealed,key='s000002_i000007_o000001',value={'status':'running'})
    assert driver.audit_provenance(sealed.out,'lmo',driver.sha256(sealed.manifest),v,driver.sha256(path),receipt_paths=[p])['legacy']==1
    with pytest.raises(ValueError,match='Pending'):
        driver.audit_provenance(sealed.out,'lmo',driver.sha256(sealed.manifest),v,driver.sha256(path))


def test_validation_requires_bound_artifacts_and_exact_helper(sealed):
    p=sealed.out/'validation.json';artifact=sealed.out/'comparison.json';artifact.write_text('{"status":"passed"}')
    v={'status':'passed','execution_revision_sha256':driver.sha256(sealed.manifest),'configuration_sha256':driver.digest(sealed.config),
       'descriptor_equivalence':True,'pose_equivalence':True,'measured_speedup':True,
       'helper_sha256':driver.read(sealed.manifest)['files']['dino_reuse_trial.py'],'artifacts':{str(artifact):driver.sha256(artifact)}}
    driver.atomic_json(p,v);driver.require_validation(sealed.config_path,sealed.manifest,p)
    artifact.write_text('{}')
    with pytest.raises(ValueError,match='artifact identity'):driver.require_validation(sealed.config_path,sealed.manifest,p)
    v['artifacts']={};driver.atomic_json(p,v)
    with pytest.raises(ValueError,match='artifacts are required'):driver.require_validation(sealed.config_path,sealed.manifest,p)
    v['helper_sha256']='wrong';driver.atomic_json(p,v)
    with pytest.raises(ValueError,match='different model reuse helper'):driver.require_validation(sealed.config_path,sealed.manifest,p)


def test_validation_artifact_escape_rejected(sealed,tmp_path):
    artifact=tmp_path/'outside.json';artifact.write_text('{}');p=sealed.out/'validation.json'
    v={'status':'passed','execution_revision_sha256':driver.sha256(sealed.manifest),'configuration_sha256':driver.digest(sealed.config),
       'descriptor_equivalence':True,'pose_equivalence':True,'measured_speedup':True,
       'helper_sha256':driver.read(sealed.manifest)['files']['dino_reuse_trial.py'],'artifacts':{str(artifact):driver.sha256(artifact)}}
    driver.atomic_json(p,v)
    with pytest.raises(ValueError,match='artifact identity'):driver.require_validation(sealed.config_path,sealed.manifest,p)
