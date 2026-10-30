"""Windows-only launcher for the separately sealed evaluation queue.

This auxiliary process never launches inference or clears a lock. Its Linux
queue waits for the original experiment owners to release their OS locks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def windows_path(value, *, relative_to=None):
    """Resolve protocol paths on Windows, including /mnt/<drive>/ references."""
    value = str(value)
    if value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
        value = value[5].upper() + ':/' + value[7:]
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = Path(relative_to) / path
    return path.resolve()


def mount_path(value):
    path = str(Path(value).resolve()).replace('\\', '/')
    if len(path) < 3 or path[1:3] != ':/' or not path[0].isalpha():
        raise ValueError('Evaluation paths must use a local Windows drive mounted in WSL')
    return '/mnt/' + path[0].lower() + path[2:]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def refuse_live_bridge(bridge):
    """PID text alone is insufficient: compare the live creation time/command."""
    import psutil

    if not bridge.exists():
        return
    previous = read_json(bridge)
    try:
        owner = psutil.Process(int(previous['pid']))
        if (owner.is_running()
                and owner.create_time() == float(previous['create_time'])
                and owner.cmdline() == previous['actual_cmdline']):
            raise RuntimeError('The exact existing evaluation bridge is live; refusing a duplicate')
    except psutil.NoSuchProcess:
        return
    except psutil.AccessDenied as exc:
        raise RuntimeError('Cannot verify existing evaluation bridge ownership; refusing a duplicate') from exc


def seal_helper(root, helper_source):
    """Keep evaluation source distinct from the immutable inference snapshot."""
    primary = root / 'source_snapshot' / 'source_manifest.json'
    if not primary.is_file():
        raise FileNotFoundError('The original immutable inference source manifest is missing')
    if not helper_source.is_file():
        raise FileNotFoundError(f'Evaluation queue source is missing: {helper_source}')
    source_bytes = helper_source.read_bytes()
    helper_sha = hashlib.sha256(source_bytes).hexdigest()
    primary_sha = sha256(primary)
    folder = root / 'auxiliary_source' / 'evaluation_v1'
    helper = folder / 'complete_evaluation.py'
    manifest = folder / 'source_manifest.json'
    expected = {
        'schema': 'freezev2-auxiliary-evaluation-source-v1',
        'files': {'complete_evaluation.py': helper_sha},
        'primary_source_manifest': 'source_snapshot/source_manifest.json',
        'primary_source_manifest_sha256': primary_sha,
    }
    if manifest.exists():
        previous = read_json(manifest)
        if any(previous.get(key) != value for key, value in expected.items()):
            raise RuntimeError('The existing sealed evaluation helper/source binding differs; use a new version')
        if not helper.is_file() or sha256(helper) != helper_sha:
            raise RuntimeError('The existing sealed evaluation helper is missing or changed')
        return helper, manifest, expected
    if helper.exists() and sha256(helper) != helper_sha:
        raise RuntimeError('Refusing to replace a differing already sealed evaluation helper')
    folder.mkdir(parents=True, exist_ok=True)
    if not helper.exists():
        # Exclusive creation prevents one launcher from replacing another's
        # selected helper. An interrupted partial write is rejected on retry.
        try:
            with helper.open('xb') as stream:
                stream.write(source_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
    if sha256(helper) != helper_sha:
        raise RuntimeError('Evaluation helper changed while sealing; nothing was launched')
    if sha256(primary) != primary_sha:
        raise RuntimeError('Original inference source manifest changed while sealing')
    atomic_json(manifest, {**expected, 'sealed_epoch': time.time()})
    return helper, manifest, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('protocol.json')))
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This launcher is Windows-only; invoke the sealed evaluation queue directly on WSL')
    import psutil

    config = windows_path(args.config)
    protocol = read_json(config)
    root = windows_path(protocol['paths']['suite_output_root'], relative_to=config.parent)
    if not root.is_dir():
        raise FileNotFoundError(f'The existing experiment run root is missing: {root}')
    bridge = root / 'evaluation_bridge.json'
    refuse_live_bridge(bridge)
    runtime_path = windows_path(protocol['paths']['evaluation_runtime'], relative_to=config.parent)
    runtime = read_json(runtime_path)
    python = runtime['python']
    if not isinstance(python, str) or not python.startswith('/'):
        raise ValueError('evaluation_runtime.python must name the configured absolute WSL evaluation interpreter')
    helper, source_manifest, source_binding = seal_helper(root, Path(__file__).with_name('complete_evaluation.py'))
    common = ['wsl.exe', '--exec', python, mount_path(helper), '--run-root', mount_path(root)]
    # This probe checks only post_evaluation.lock. Main inference/evaluation
    # locks are intentionally allowed to remain live while this queue starts.
    checked = subprocess.run(common + ['--probe-queue-lock'], capture_output=True, text=True, timeout=45)
    if checked.returncode:
        raise RuntimeError('Evaluation queue lock is owned or its probe failed: '
                           + (checked.stdout + checked.stderr)[-6000:])
    refuse_live_bridge(bridge)
    if sha256(helper) != source_binding['files']['complete_evaluation.py']:
        raise RuntimeError('Sealed evaluation helper changed after the queue probe')
    if sha256(root / 'source_snapshot' / 'source_manifest.json') != source_binding['primary_source_manifest_sha256']:
        raise RuntimeError('Original source manifest changed after the queue probe')
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    log_path = root / 'evaluation_bridge.log'
    launched = time.time()
    with log_path.open('ab', buffering=0) as log:
        process = subprocess.Popen(common, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=flags)
    try:
        owner = psutil.Process(process.pid)
        created, actual = owner.create_time(), owner.cmdline()
    except psutil.NoSuchProcess as exc:
        raise RuntimeError(f'Evaluation bridge exited before ownership could be recorded; inspect {log_path}') from exc
    value = {
        'schema': 'freezev2-native-evaluation-bridge-v1',
        'pid': process.pid, 'create_time': created, 'actual_cmdline': actual,
        'command': common, 'launched_epoch': launched,
        'run_root': str(root), 'helper_path': str(helper),
        'helper_sha256': source_binding['files']['complete_evaluation.py'],
        'auxiliary_source_manifest_sha256': sha256(source_manifest),
        'primary_source_manifest_sha256': source_binding['primary_source_manifest_sha256'],
        'evaluation_python': python, 'evaluation_runtime_sha256': sha256(runtime_path),
        'log_path': str(log_path),
    }
    atomic_json(bridge, value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
