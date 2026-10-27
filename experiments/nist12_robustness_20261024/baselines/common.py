"""Label-free 20-layout inputs; immutable prior baseline dependencies."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIOR = REPO / 'experiments/nist12_fullscene_benchmarks_20261023'
SOURCES = PRIOR / 'sources'
WEIGHTS = PRIOR / 'weights'
DEPS = PRIOR / 'deps'


def read(filename):
    return json.loads(Path(filename).read_text(encoding='utf-8-sig'))


def save(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(filename):
    result = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def path(value):
    text = str(value).replace('\\', '/')
    if len(text) > 1 and text[1] == ':' and sys.platform != 'win32':
        text = '/mnt/' + text[0].lower() + text[2:]
    filename = Path(text)
    return filename if filename.is_absolute() else REPO / filename


def relative(value):
    return Path(value).resolve().relative_to(REPO).as_posix()


def setup():
    sys.path.insert(0, str(DEPS))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'ultralytics_config'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    return read(HERE / 'inputs.json')
