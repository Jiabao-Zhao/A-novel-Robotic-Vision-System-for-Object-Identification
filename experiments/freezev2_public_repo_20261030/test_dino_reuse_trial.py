"""CPU-only contracts for an isolated model-reuse execution trial."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dino_reuse_trial_under_test', HERE / 'dino_reuse_trial.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def call(cache, repo):
    return cache(str(repo), t.MODEL, pretrained=True, source='local')


def test_exact_local_model_is_reused(tmp_path):
    calls = []
    model = object()
    cache = t.DINOModelReuse(lambda *a, **k: calls.append((a, k)) or model, tmp_path)
    assert call(cache, tmp_path) is model
    assert call(cache, tmp_path) is model
    assert len(calls) == 1
    assert cache.stats['dino_model_loads'] == cache.stats['dino_cache_hits'] == 1
    assert [e['cache_hit'] for e in cache.stats['events']] == [False, True]


def test_disabled_cache_loads_each_time(tmp_path):
    calls = []
    cache = t.DINOModelReuse(lambda *a, **k: calls.append(1) or object(), tmp_path, enabled=False)
    assert call(cache, tmp_path) is not call(cache, tmp_path)
    assert len(calls) == 2
    assert cache.stats['dino_cache_hits'] == 0


def test_wrong_repo_is_rejected_instead_of_wrong_reuse(tmp_path):
    cache = t.DINOModelReuse(lambda *a, **k: object(), tmp_path)
    call(cache, tmp_path)
    with pytest.raises(ValueError, match='exact pinned'):
        call(cache, tmp_path / 'different')
    assert cache.stats['dino_cache_hits'] == 0


@pytest.mark.parametrize('kwargs', [dict(pretrained=False, source='local'),
                                   dict(pretrained=True, source='github'),
                                   dict(pretrained=True, source='local', force_reload=True),
                                   dict(pretrained=True)])
def test_changed_dino_signature_rejected(tmp_path, kwargs):
    cache = t.DINOModelReuse(lambda *a, **k: pytest.fail('must not load'), tmp_path)
    with pytest.raises(ValueError):
        cache(str(tmp_path), t.MODEL, **kwargs)


def test_other_model_names_delegate_unchanged(tmp_path):
    calls = []
    cache = t.DINOModelReuse(lambda *a, **k: calls.append((a, k)) or object(), tmp_path)
    first = cache('other', 'another_model', pretrained=False)
    second = cache('other', 'another_model', pretrained=False)
    assert first is not second
    assert calls == [(('other', 'another_model'), {'pretrained': False})] * 2
    assert cache.stats['other_hub_calls'] == 2
    assert cache.stats['dino_model_loads'] == cache.stats['dino_cache_hits'] == 0


def test_failed_load_is_not_cached(tmp_path):
    calls = []
    model = object()
    def load(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('first load failed')
        return model
    cache = t.DINOModelReuse(load, tmp_path)
    with pytest.raises(RuntimeError):
        call(cache, tmp_path)
    assert call(cache, tmp_path) is model
    assert call(cache, tmp_path) is model
    assert len(calls) == 2
    assert cache.stats['events'][0]['status'] == 'failed'


def test_context_restores_load_and_drops_cached_reference_on_error(tmp_path):
    original = lambda *a, **k: object()
    torch = SimpleNamespace(hub=SimpleNamespace(load=original))
    with pytest.raises(RuntimeError):
        with t.dino_reuse_context(torch, tmp_path) as stats:
            call(torch.hub.load, tmp_path)
            cache = torch.hub.load
            raise RuntimeError('trial error')
    assert torch.hub.load is original
    assert cache.model is None
    assert stats['dino_model_loads'] == 1


def test_timed_attribute_restores_on_error():
    def fail():
        raise RuntimeError('expected')
    owner = SimpleNamespace(run=fail)
    events = []
    with pytest.raises(RuntimeError):
        with t.timed_attribute(owner, 'run', events, 'fixture'):
            owner.run()
    assert owner.run is fail
    assert events[0]['component'] == 'fixture'
    assert events[0]['status'] == 'failed'


@pytest.fixture
def trial(tmp_path):
    out, source = tmp_path / 'own', tmp_path / 'public'
    source.mkdir()
    (source / 'run_bop.py').write_text('pass\n')
    root = out / 'auxiliary_diagnostics/trial'
    config = {'paths': {'output': str(out), 'public_source': str(source), 'inference_python': sys.executable,
                        'compute_lock': str(tmp_path / 'compute.lock'), 'dinov2_root': str(tmp_path / 'dino')},
              'resources': {'cpu_threads': 4},
              'settings': {'dino_model': t.MODEL, 'precision': 'FP32', 'TF32': False}}
    config_path = write(out / 'configuration.json', config)
    source_identity = {'source': 'fixture-exact'}
    ref = out / 'source_snapshot/reference_gate.py'
    ref.parent.mkdir(parents=True)
    ref.write_text('def git_identity(path, revision):\n    return ' + repr(source_identity) + '\n')
    manifest_path = write(out / 'source_snapshot/source_manifest.json',
                          {'configuration_sha256': t.digest(config), 'files': {'reference_gate.py': t.sha256(ref)}})
    pins = {'configuration_sha256': t.digest(config), 'orchestration_source_manifest_sha256': t.sha256(manifest_path),
            'public_revision': t.PUBLIC_REVISION}
    write(out / 'run_manifest.json', {'source_pins': pins, 'public_source_identity': source_identity})
    baseline_command = [sys.executable, '-B', str(source / 'run_bop.py'), 'run-localization', '--dataset', 'lmo',
                        '--work-dir', str(out / 'datasets/lmo/targets/case/original'),
                        '--output', str(out / 'datasets/lmo/targets/case/original.csv')]
    receipt = write(out / 'datasets/lmo/targets/case/receipt.json',
                    {'status': 'complete', 'source_pins': pins, 'command': baseline_command, 'outputs': {}})
    command = baseline_command.copy()
    command[command.index('--work-dir') + 1] = str(root / 'original')
    command[command.index('--output') + 1] = str(root / 'original.csv')
    request = {'configuration': str(config_path), 'source_pins': pins, 'baseline_receipt': str(receipt),
               'command': command, 'public_revision': t.PUBLIC_REVISION, 'source': str(source)}
    request_path = write(root / 'request.json', request)
    return {'request': request_path, 'report': root / 'report.json', 'value': request, 'receipt': receipt,
            'config_path': config_path, 'out': out}


def test_request_allows_only_new_auxiliary_output_paths(trial):
    result = t.validate_request(trial['request'], trial['report'])
    assert result['baseline']['status'] == 'complete'
    assert result['request_sha256'] == t.sha256(trial['request'])


def test_request_rejects_scientific_command_change(trial):
    trial['value']['command'][trial['value']['command'].index('--dataset') + 1] = 'tless'
    write(trial['request'], trial['value'])
    with pytest.raises(ValueError, match='Only trial work/output'):
        t.validate_request(trial['request'], trial['report'])


def test_request_rejects_mismatched_source_pins(trial):
    trial['value']['source_pins']['configuration_sha256'] = 'different'
    write(trial['request'], trial['value'])
    with pytest.raises(ValueError, match='Sealed configuration/source'):
        t.validate_request(trial['request'], trial['report'])


def test_request_rejects_existing_original_output(trial):
    path = Path(trial['value']['command'][-1])
    path.write_text('preserve prior trial')
    with pytest.raises(FileExistsError, match='already exists'):
        t.validate_request(trial['request'], trial['report'])
    assert path.read_text() == 'preserve prior trial'


def test_request_rejects_non_auxiliary_report(trial):
    with pytest.raises(ValueError, match='auxiliary_diagnostics'):
        t.validate_request(trial['request'], trial['out'] / 'report.json')


def test_request_rejects_failed_baseline(trial):
    value = t.read(trial['receipt'])
    value['status'] = 'failed'
    write(trial['receipt'], value)
    with pytest.raises(ValueError, match='complete matching own baseline'):
        t.validate_request(trial['request'], trial['report'])


def test_execution_exception_is_reported_without_overwriting_original(trial, monkeypatch):
    monkeypatch.setattr(t, 'require_inherited_lock', lambda *a: {'fixture': True})
    monkeypatch.setattr(t, 'parent_death_guard', lambda: 123)
    baseline_hash = t.sha256(trial['receipt'])
    def fail(ctx, **kwargs):
        raise RuntimeError('expected model failure')
    with pytest.raises(RuntimeError, match='expected model failure'):
        t.run_trial(trial['request'], trial['report'], reuse_dino=True, lock_fd=99, executor=fail)
    report = t.read(trial['report'])
    assert report['status'] == 'failed'
    assert report['error'] == 'RuntimeError: expected model failure'
    assert report['baseline_receipt_sha256'] == baseline_hash == t.sha256(trial['receipt'])
    assert report['reuse_dino'] is True


def test_completed_trial_reports_nested_timings(trial, monkeypatch):
    monkeypatch.setattr(t, 'require_inherited_lock', lambda *a: {'fixture': True})
    monkeypatch.setattr(t, 'parent_death_guard', lambda: 123)
    def fake(ctx, *, reuse_dino, stats):
        stats['events'].append({'component': 'fixture-original', 'status': 'complete', 'seconds': 1.0})
    report = t.run_trial(trial['request'], trial['report'], reuse_dino=False, lock_fd=99, executor=fake)
    assert report['status'] == 'complete'
    assert t.read(trial['report'])['statistics']['events'][0]['component'] == 'fixture-original'
    assert report['helper_sha256'] == t.sha256(HERE / 'dino_reuse_trial.py')


def test_source_identity_change_after_execution_fails(trial, monkeypatch):
    monkeypatch.setattr(t, 'require_inherited_lock', lambda *a: {'fixture': True})
    monkeypatch.setattr(t, 'parent_death_guard', lambda: 123)
    calls = []
    def identity(*a):
        calls.append(1)
        return {'source': 'fixture-exact' if len(calls) == 1 else 'changed'}
    monkeypatch.setattr(t, '_load_module', lambda *a: SimpleNamespace(git_identity=identity))
    with pytest.raises(ValueError, match='after trial'):
        t.run_trial(trial['request'], trial['report'], reuse_dino=True, lock_fd=99,
                    executor=lambda *a, **k: None)
    assert t.read(trial['report'])['status'] == 'failed'


@pytest.mark.skipif(sys.platform != 'linux', reason='Real inherited flock contract requires Linux')
def test_inherited_lock_must_be_the_recorded_file(tmp_path):
    import fcntl
    first, other = tmp_path / 'lock', tmp_path / 'other'
    with first.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        other.touch()
        with pytest.raises(ValueError, match='not the recorded'):
            t.require_inherited_lock(stream.fileno(), other)
        assert t.require_inherited_lock(stream.fileno(), first)['inode'] == first.stat().st_ino

def test_original_command_alias_is_normalized(trial):
    trial['value']['original_command'] = trial['value'].pop('command')
    write(trial['request'], trial['value'])
    result = t.validate_request(trial['request'], trial['report'])
    assert result['request']['command'] == trial['value']['original_command']
