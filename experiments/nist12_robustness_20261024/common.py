"""Local paths and immutable input bookkeeping for the robustness study."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PRIOR=HERE.parent/'nist12_fullscene_benchmarks_20261023'
OLD=HERE.parent/'nist12_qwen_clip_reranking_20261023'
DEPS=PRIOR/'deps'

# Retain only NIST simulation; synthetic depth corruption and T-LESS were retired.
ACTIVE_INPUTS = (
    ('simulation', 'capture_manifest.json', 'catalog.json', 200),
    ('new_cads', 'new_cads/inputs.json', 'new_cads/catalog.json', 20),
)
ACTIVE_SCENES = 220
ACTIVE_QUERIES = 2720


def read(file):
    return json.loads(Path(file).read_text(encoding='utf-8-sig'))


def save(file,value):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(file):
    with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def path(value):
    value=str(value).replace('\\','/')
    if sys.platform!='win32' and len(value)>1 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    p=Path(value)
    return p if p.is_absolute() else REPO/p


def relative(file):return Path(file).resolve().relative_to(REPO).as_posix()


def setup():
    sys.path.insert(0,str(DEPS))
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    return read(HERE/'inputs.json') if (HERE/'inputs.json').exists() else read(PRIOR/'inputs.json')
