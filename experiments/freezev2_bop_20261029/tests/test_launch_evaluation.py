"""Native auxiliary-launcher helpers; no subprocess, WSL or GPU is launched."""
import json
import os

import psutil
import pytest

import launch_evaluation as launcher


def _sources(tmp_path):
    root = tmp_path / 'synthetic_run'
    primary = root / 'source_snapshot' / 'source_manifest.json'
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps({'files': {'fixture_only.py': 'synthetic-source-sha'}}))
    source = tmp_path / 'synthetic_source' / 'complete_evaluation.py'
    source.parent.mkdir()
    source.write_text('"""Unit fixture only; never execute."""\n')
    return root, primary, source


def test_helper_seal_is_immutable_and_reuse_preserves_receipt_bytes(tmp_path):
    root, primary, source = _sources(tmp_path)
    helper, manifest, binding = launcher.seal_helper(root, source)
    receipt_before = manifest.read_bytes()
    assert helper.read_bytes() == source.read_bytes()
    assert binding['files'] == {'complete_evaluation.py': launcher.sha256(source)}
    assert binding['primary_source_manifest_sha256'] == launcher.sha256(primary)
    assert helper.parent == root / 'auxiliary_source' / 'evaluation_v1'
    assert launcher.seal_helper(root, source) == (helper, manifest, binding)
    assert manifest.read_bytes() == receipt_before


def test_helper_seal_rejects_changed_primary_manifest(tmp_path):
    root, primary, source = _sources(tmp_path)
    helper, manifest, _ = launcher.seal_helper(root, source)
    receipt_before, helper_before = manifest.read_bytes(), helper.read_bytes()
    primary.write_text('{"files":{"fixture_only.py":"changed-primary-source"}}')
    with pytest.raises(RuntimeError, match='source binding differs'):
        launcher.seal_helper(root, source)
    assert manifest.read_bytes() == receipt_before and helper.read_bytes() == helper_before


@pytest.mark.parametrize('changed_file', ['sealed_helper', 'new_source'])
def test_helper_seal_rejects_tampered_or_replacement_script(tmp_path, changed_file):
    root, _, source = _sources(tmp_path)
    helper, manifest, _ = launcher.seal_helper(root, source)
    receipt_before = manifest.read_bytes()
    modified = helper if changed_file == 'sealed_helper' else source
    modified.write_text('"""Different fixture script; never execute."""\n')
    with pytest.raises(RuntimeError, match='changed|differs'):
        launcher.seal_helper(root, source)
    assert manifest.read_bytes() == receipt_before


def test_exact_actual_test_process_ownership_refuses_duplicate_bridge(tmp_path):
    owner = psutil.Process(os.getpid())
    bridge = tmp_path / 'evaluation_bridge.json'
    bridge.write_text(json.dumps({'pid': owner.pid, 'create_time': owner.create_time(),
                                  'actual_cmdline': owner.cmdline()}))
    with pytest.raises(RuntimeError, match='exact existing evaluation bridge is live'):
        launcher.refuse_live_bridge(bridge)


@pytest.mark.parametrize('mismatch', ['create_time', 'actual_cmdline'])
def test_pid_alone_does_not_establish_bridge_ownership(tmp_path, mismatch):
    owner = psutil.Process(os.getpid())
    prior = {'pid': owner.pid, 'create_time': owner.create_time(), 'actual_cmdline': owner.cmdline()}
    prior[mismatch] = prior[mismatch] - 1 if mismatch == 'create_time' else ['different-unit-fixture-command']
    bridge = tmp_path / 'evaluation_bridge.json'
    bridge.write_text(json.dumps(prior))
    launcher.refuse_live_bridge(bridge)


def test_absent_or_nonexistent_bridge_owner_is_allowed(tmp_path):
    bridge = tmp_path / 'evaluation_bridge.json'
    launcher.refuse_live_bridge(bridge)
    nonexistent_pid = 2147483647
    assert not psutil.pid_exists(nonexistent_pid)
    bridge.write_text(json.dumps({'pid': nonexistent_pid, 'create_time': 0,
                                  'actual_cmdline': ['synthetic-nonexistent-process']}))
    launcher.refuse_live_bridge(bridge)
