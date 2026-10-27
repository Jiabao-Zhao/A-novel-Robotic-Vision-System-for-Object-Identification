"""One-off GPU queue; exact-image caching lets descriptions overlap localization."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import HERE, read, save, digest, relative, ACTIVE_INPUTS, ACTIVE_SCENES, ACTIVE_QUERIES


def run(*arguments):
    command = [sys.executable, '-u', str(HERE/'run_visual.py'), *arguments]
    if arguments[0] == 'describe' and sys.platform != 'win32':
        # The existing Ollama service binds Windows localhost; CLIP uses WSL CUDA.
        script = str(HERE/'run_visual.py')
        assert script.startswith('/mnt/d/')
        command = ['/mnt/d/Python/python.exe', '-u', 'D:'+script[6:], *arguments]
    print('START '+str(arguments), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def describe_snapshot():
    run('describe')
    signature = digest(HERE/'visual/inputs.json')
    destination = HERE/'visual/description_runs'/signature
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('inputs.json', 'protocol.json', 'description_completion.json'):
        shutil.copyfile(HERE/'visual'/name, destination/name)
    return signature


def main():
    baseline = HERE/'baselines/completion.json'
    release = HERE/'baselines/gpu_release.json'
    while not release.exists():
        time.sleep(10)
    assert read(baseline)['complete']
    save(HERE/'visual/queue_protocol.json', {
        'source_sha256': digest(__file__), 'baseline_completion_sha256': digest(baseline),
        'gpu_release_sha256': digest(release),
        'description_cache': 'Exact pixels and original prompt/model controls only.',
        'labels_read': False})
    first = describe_snapshot()
    manifests = [(manifest, catalog, name) for name, manifest, catalog, _ in ACTIVE_INPUTS]
    expected = [row['scene_id'] for manifest, _, _ in manifests
                for row in read(HERE/manifest)['scenes']]
    assert len(expected) == len(set(expected)) == ACTIVE_SCENES
    while True:
        missing = [sid for sid in expected
                   if not (HERE/'geometry/localization'/sid/'localization.json').exists()]
        if not missing:
            break
        save(HERE/'visual/waiting_localization.json', {'missing': missing, 'labels_read': False})
        time.sleep(10)
    # Additional candidates may have appeared while the first GPU pass ran.
    import run_visual
    current = run_visual.inputs_now()
    if current != read(HERE/'visual/inputs.json'):
        describe_snapshot()
    run('encode')
    for manifest, catalog, name in manifests:
        run('score', relative(HERE/manifest), relative(HERE/catalog), name)
    save(HERE/'visual/queue_completion.json', {
        'complete': True, 'scenes': ACTIVE_SCENES, 'queries': ACTIVE_QUERIES, 'labels_read': False,
        'first_input_sha256': first, 'final_input_sha256': digest(HERE/'visual/inputs.json'),
        'source_sha256': digest(__file__)})


if __name__ == '__main__':
    main()
