"""Offline tests only: no live controller, process stop or launch."""
import ast
from pathlib import Path
import pytest
import perform_lmo_gate_handoff as handoff
import runtime_reuse_driver as revision
from test_runtime_reuse_driver import sealed


@pytest.fixture
def control(sealed):
    folder=sealed.out/'auxiliary_source/runtime_reuse_v2_lmo_gate';folder.mkdir(parents=True)
    files={}
    for name in handoff.CONTROL_FILES:
        (folder/name).write_text('# fixture '+name);files[name]=revision.sha256(folder/name)
    policy=sealed.out/'run_control/policy.json';revision.atomic_json(policy,{'schema':'lmo-paper-stop-policy-v1','status':'awaiting_user','paper_ar':.694})
    manifest=folder/'control_manifest.json'
    revision.atomic_json(manifest,{'schema':'lmo-paper-run-control-source-v1','configuration_sha256':revision.digest(sealed.config),
        'base_source_manifest_sha256':revision.BASE_MANIFEST_SHA,'execution_revision_sha256':revision.sha256(sealed.manifest),
        'files':files,'policy_path':str(policy),'policy_sha256':revision.sha256(policy)})
    return sealed,manifest,policy


def test_controller_reuses_verified_lock_and_idle_logic():
    code=handoff.controller_source();compile(code,'control-handoff','exec')
    assert "'schema':'runtime-gate-cutover-v1'" in code
    assert "'schema':'runtime-reuse-transition-v1'" not in code
    assert "'previous_transition_sha256':request['previous_transition_sha256']" in code
    assert "'control_manifest_sha256':request['control_manifest_sha256']" in code
    assert 'fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)' in code
    assert 'if not check_idle()' in code
    assert 'SIGKILL' not in code and 'unlink' not in code


def test_original_handoff_dependency_is_immutable():
    assert revision.sha256(handoff.original.__file__)==handoff.ORIGINAL_HANDOFF_SHA


def test_control_verification_accepts_exact_awaiting_user_policy(control):
    s,m,p=control
    config,out,saved=handoff.verify_control_files(s.config_path,m,p,s.manifest)
    assert config==s.config and out==s.out and saved==s.config_path


def test_control_source_change_rejected(control):
    s,m,p=control;(m.parent/'runtime_gate_driver.py').write_text('changed')
    with pytest.raises(ValueError,match='control source changed'):handoff.verify_control_files(s.config_path,m,p,s.manifest)


def test_policy_change_rejected(control):
    s,m,p=control;revision.atomic_json(p,{'status':'configured','tolerance_pp':5})
    with pytest.raises(ValueError,match='policy identity'):handoff.verify_control_files(s.config_path,m,p,s.manifest)


def test_wrong_prior_execution_revision_rejected(control):
    s,m,p=control;v=revision.read(m);v['execution_revision_sha256']='other';revision.atomic_json(m,v)
    with pytest.raises(ValueError,match='identity differs'):handoff.verify_control_files(s.config_path,m,p,s.manifest)


def test_policy_must_match_explicit_cli_path(control,tmp_path):
    s,m,p=control;alternate=tmp_path/'different_policy.json';alternate.write_bytes(p.read_bytes())
    with pytest.raises(ValueError,match='policy identity'):handoff.verify_control_files(s.config_path,m,alternate,s.manifest)


def test_source_file_set_cannot_expand_unrecorded(control):
    s,m,p=control;v=revision.read(m);v['files']['extra.py']='x';revision.atomic_json(m,v)
    with pytest.raises(ValueError,match='source file set'):handoff.verify_control_files(s.config_path,m,p,s.manifest)


def test_control_location_is_fixed(control,tmp_path):
    s,m,p=control;copy=tmp_path/'control_manifest.json';copy.write_bytes(m.read_bytes())
    with pytest.raises(ValueError,match='control location'):handoff.verify_control_files(s.config_path,copy,p,s.manifest)


def test_controller_structure_change_fails_closed(monkeypatch):
    monkeypatch.setattr(handoff.original,'CONTROLLER','print("changed")')
    with pytest.raises(ValueError,match='structure changed'):handoff.controller_source()