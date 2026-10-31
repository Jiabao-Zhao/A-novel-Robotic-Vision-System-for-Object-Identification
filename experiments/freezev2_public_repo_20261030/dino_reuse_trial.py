"""Isolated, lock-guarded trial of reusing the public target DINO model.

This changes execution only in a NEW auxiliary trial. It does not edit the public
source, sealed runner, live configuration, or saved benchmark outputs. A cache
hit skips original DINO construction (including its RNG consumption). Numerical
parity must be measured before using this helper in a benchmark driver.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

MODEL = 'dinov2_vitg14'
PUBLIC_REVISION = '12f82a2f8afd435466a87cf2dafce404087aeb3f'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def local(value):
    value = str(value)
    if os.name != 'nt' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    elif os.name == 'nt' and value.startswith('/mnt/'):
        value = value[5].upper() + ':/' + value[7:]
    return Path(value).expanduser().resolve()


def _stats(value=None):
    value = {} if value is None else value
    for key in ('hub_load_calls', 'dino_cache_hits', 'dino_model_loads', 'other_hub_calls'):
        value.setdefault(key, 0)
    value.setdefault('events', [])
    return value


class DINOModelReuse:
    """One exact local model key; all other model names delegate unchanged."""

    def __init__(self, original_load, expected_repo, *, enabled=True, stats=None, clock=time.perf_counter):
        self.original_load = original_load
        self.expected_repo = local(expected_repo)
        self.enabled = bool(enabled)
        self.stats = _stats(stats)
        self.clock = clock
        self.model = None

    def __call__(self, *args, **kwargs):
        self.stats['hub_load_calls'] += 1
        model_name = args[1] if len(args) > 1 else kwargs.get('model')
        is_dino = model_name == MODEL
        hit = False
        started = self.clock()
        outcome = 'failed'
        try:
            if not is_dino:
                self.stats['other_hub_calls'] += 1
                result = self.original_load(*args, **kwargs)
            else:
                if (len(args) != 2 or set(kwargs) != {'pretrained', 'source'}
                        or kwargs['pretrained'] is not True or kwargs['source'] != 'local'
                        or local(args[0]) != self.expected_repo):
                    raise ValueError('DINO load differs from the exact pinned local model signature')
                hit = self.enabled and self.model is not None
                if hit:
                    self.stats['dino_cache_hits'] += 1
                    result = self.model
                else:
                    result = self.original_load(*args, **kwargs)
                    self.stats['dino_model_loads'] += 1
                    if self.enabled:
                        self.model = result
            outcome = 'complete'
            return result
        finally:
            self.stats['events'].append({'component': 'torch.hub.load', 'model_name': str(model_name),
                                         'seconds': self.clock() - started, 'status': outcome,
                                         'cache_hit': hit})


@contextmanager
def dino_reuse_context(torch, expected_repo, *, enabled=True, stats=None):
    """Yield counters and temporarily replace only the exact local hub load.

    The original DinoExtractor constructor, model.to/eval, image preprocessing,
    feature forward, GeDi models, RNG calls, sampling, RANSAC and ICP are untouched.
    A reused object retains its original loaded weights and GPU allocation.
    """
    original = torch.hub.load
    cache = DINOModelReuse(original, expected_repo, enabled=enabled, stats=stats)
    torch.hub.load = cache
    try:
        yield cache.stats
    finally:
        torch.hub.load = original
        cache.model = None


@contextmanager
def timed_attribute(owner, name, events, component):
    original = getattr(owner, name)

    def timed(*args, **kwargs):
        started = time.perf_counter()
        status = 'failed'
        label = component(*args, **kwargs) if callable(component) else component
        try:
            result = original(*args, **kwargs)
            status = 'complete'
            return result
        finally:
            events.append({'component': label, 'seconds': time.perf_counter() - started, 'status': status})

    setattr(owner, name, timed)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _flags(command):
    if len(command) < 4 or command[1] != '-B' or command[3] != 'run-localization':
        raise ValueError('Trial accepts only a recorded original run-localization command')
    tokens = command[4:]
    if len(tokens) % 2:
        raise ValueError('Malformed original command flags')
    values = {}
    for key, value in zip(tokens[::2], tokens[1::2]):
        if not key.startswith('--') or key in values:
            raise ValueError('Malformed or duplicate original command flag')
        values[key] = value
    return values


def _under(path, root):
    path, root = local(path), local(root)
    if path == root or not path.is_relative_to(root):
        raise ValueError('Trial output must be a new child of own auxiliary_diagnostics')
    return path


def validate_request(request_path, report_path):
    """Bind the trial to an exact completed own target; output paths alone vary."""
    request_path, report_path = local(request_path), local(report_path)
    value = read(request_path)
    config_path = local(value['configuration'])
    config = read(config_path)
    out = local(config['paths']['output'])
    if config_path != out / 'configuration.json':
        raise ValueError('Use the saved own configuration')
    trial_root = out / 'auxiliary_diagnostics'
    _under(report_path, trial_root)
    if report_path.exists():
        raise FileExistsError('A trial report already exists')
    manifest_path = out / 'source_snapshot/source_manifest.json'
    manifest = read(manifest_path)
    pins = value['source_pins']
    if (digest(config) != pins['configuration_sha256']
            or sha256(manifest_path) != pins['orchestration_source_manifest_sha256']
            or manifest['configuration_sha256'] != pins['configuration_sha256']):
        raise ValueError('Sealed configuration/source pins differ')
    for name, wanted in manifest['files'].items():
        path = local(manifest_path.parent / name)
        if path.parent != manifest_path.parent or sha256(path) != wanted:
            raise ValueError('Sealed orchestration source changed: ' + name)
    run = read(out / 'run_manifest.json')
    if run['source_pins'] != pins:
        raise ValueError('Run source pins differ')
    baseline_path = _under(value['baseline_receipt'], out / 'datasets')
    baseline = read(baseline_path)
    if baseline.get('status') != 'complete' or baseline.get('source_pins') != pins:
        raise ValueError('A complete matching own baseline receipt is required')
    for path, wanted in baseline.get('outputs', {}).items():
        if sha256(local(path)) != wanted:
            raise ValueError('Saved baseline artifact changed')
    command = value.get('command', value.get('original_command'))
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError('Missing original command')
    value['command'] = command  # Normalize the supported original_command alias.
    before, after = _flags(baseline['command']), _flags(command)
    for key in ('--output', '--work-dir'):
        if key not in before or key not in after:
            raise ValueError('Trial output/work directory flags are required')
        target = _under(after[key], trial_root)
        if target.exists():
            raise FileExistsError('Trial output/work directory already exists')
        if target.parent != report_path.parent:
            raise ValueError('Trial payloads and report must share one new trial directory')
    if command[:4] != baseline['command'][:4] or {k: v for k, v in before.items() if k not in ('--output', '--work-dir')} != {
            k: v for k, v in after.items() if k not in ('--output', '--work-dir')}:
        raise ValueError('Only trial work/output paths may differ from the recorded original command')
    source = local(value['source'])
    if (source != local(config['paths']['public_source']) or command[2] != str(source / 'run_bop.py')
            or command[0] != config['paths']['inference_python'] or value['public_revision'] != PUBLIC_REVISION
            or pins['public_revision'] != PUBLIC_REVISION):
        raise ValueError('Original public executable/source revision differs')
    settings = config['settings']
    if (settings['dino_model'] != MODEL or settings['precision'] != 'FP32' or settings['TF32'] is not False
            or config['resources']['cpu_threads'] != 4):
        raise ValueError('Unsupported scientific precision/model/resource settings')
    return {'request': value, 'configuration': config, 'output': out, 'source': source,
            'run_manifest': run, 'baseline': baseline, 'report': report_path,
            'request_sha256': sha256(request_path), 'configuration_file_sha256': sha256(config_path),
            'baseline_receipt_sha256': sha256(baseline_path)}


def require_inherited_lock(fd, lock_path):
    if os.name != 'posix':
        raise RuntimeError('Trial requires the recorded WSL runtime')
    import fcntl
    actual, wanted = os.fstat(fd), os.stat(lock_path)
    if (actual.st_dev, actual.st_ino) != (wanted.st_dev, wanted.st_ino):
        raise ValueError('Inherited descriptor is not the recorded shared compute lock')
    # Re-locking this inherited open-file description is harmless. An unrelated
    # competing owner causes failure; the inherited descriptor remains open.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {'fd': fd, 'device': actual.st_dev, 'inode': actual.st_ino, 'path': str(lock_path)}


def parent_death_guard():
    parent = os.getppid()
    if parent <= 1:
        raise RuntimeError('A live lock-owning trial controller is required')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != parent:
        raise RuntimeError('Trial controller exited before parent-death guard was installed')
    return parent


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_original(ctx, *, reuse_dino, stats):
    """Call original public main; profiling events are nested, not additive."""
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    source = ctx['source']
    old_path, old_argv = list(sys.path), sys.argv
    sys.path.insert(0, str(source))
    try:
        original = _load_module(source / 'run_bop.py', 'dino_reuse_trial_public_original')
        events = stats['events']
        with dino_reuse_context(torch, ctx['configuration']['paths']['dinov2_root'], enabled=reuse_dino, stats=stats), \
                timed_attribute(original, 'extract_target_cache', events, 'extract_target_cache'), \
                timed_attribute(original.GediExtractor, 'encode', events, 'GediExtractor.encode'), \
                timed_attribute(original, 'main', events, lambda *a, **k: 'main:' + (sys.argv[1] if len(sys.argv) > 1 else 'unknown')):
            sys.argv = ctx['request']['command'][2:]
            original.main()
        if torch.cuda.is_available():
            stats['max_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
    finally:
        sys.path[:] = old_path
        sys.argv = old_argv


def run_trial(request_path, report_path, *, reuse_dino, lock_fd, executor=execute_original):
    ctx = validate_request(request_path, report_path)
    lock = require_inherited_lock(lock_fd, local(ctx['configuration']['paths']['compute_lock']))
    parent_pid = parent_death_guard()
    report = {'schema': 'public-dino-model-reuse-trial-v1', 'status': 'running',
              'reuse_dino': bool(reuse_dino), 'source_pins': ctx['request']['source_pins'],
              'request_sha256': ctx['request_sha256'], 'configuration_file_sha256': ctx['configuration_file_sha256'],
              'baseline_receipt_sha256': ctx['baseline_receipt_sha256'], 'helper_sha256': sha256(__file__),
              'helper_path': str(Path(__file__).resolve()), 'command': ctx['request']['command'],
              'lock': lock, 'parent_pid': parent_pid, 'pid': os.getpid(),
              'limits': ['Trial only; no claim of numerical parity or benchmark speedup.',
                         'A cache hit skips original DINO constructor RNG consumption; downstream GeDi seeding is unchanged.',
                         'Events are nested wall times without extra CUDA synchronization; do not add them.',
                         'No GeDi cache, descriptor repair, sampling, precision, pose, ranking, or public-source change.'],
              'statistics': _stats()}
    started = time.perf_counter()
    source_ref = _load_module(ctx['output'] / 'source_snapshot/reference_gate.py', 'dino_reuse_trial_reference_gate')
    wanted = ctx['run_manifest']['public_source_identity']
    try:
        if source_ref.git_identity(ctx['source'], PUBLIC_REVISION) != wanted:
            raise ValueError('Public source differs before trial')
        executor(ctx, reuse_dino=reuse_dino, stats=report['statistics'])
        if source_ref.git_identity(ctx['source'], PUBLIC_REVISION) != wanted:
            raise ValueError('Public source differs after trial')
        report['status'] = 'complete'
        return report
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['runtime_s'] = time.perf_counter() - started
        ctx['report'].parent.mkdir(parents=True, exist_ok=True)
        with ctx['report'].open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reuse-dino', action='store_true')
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_trial(args.request, args.report, reuse_dino=args.reuse_dino, lock_fd=args.lock_fd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())