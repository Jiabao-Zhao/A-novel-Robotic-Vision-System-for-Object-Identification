"""Small atomic checkpoints and real OS locks for the unattended experiment."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import uuid
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=json_value,allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,default=json_value,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    _replace(temporary,path)


def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    _replace(temporary,path)


def _replace(source,destination):
    for attempt in range(8):
        try: os.replace(source,destination); return
        except PermissionError:
            if attempt==7: raise
            time.sleep(.05*(attempt+1))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ensure_identity(folder, identity):
    """A partial stage may resume only after its scientific/input identity matches."""
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    file=folder/'identity.json'
    if file.exists():
        if read_json(file)!=identity: raise ValueError(f'Incompatible cache identity: {folder}; use a new run version')
    else:
        if any(folder.iterdir()): raise ValueError(f'Unidentified partial cache: {folder}')
        atomic_json(file,identity)


def require_disk(path,minimum_gb):
    path=Path(path)
    while not path.exists():path=path.parent
    free=shutil.disk_usage(path).free
    if free<minimum_gb*1024**3: raise RuntimeError(f'Disk below conservative reserve: {free/1024**3:.2f}GiB free, need{minimum_gb}GiB')
    return free


@contextmanager
def worker_lock(path):
    """fcntl/msvcrt handle owns liveness; a JSON PID file alone is not a lock."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close();raise RuntimeError('Another live suite worker owns the OS lock')
    try:yield stream
    finally:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        # POSIX close releases flock only after every inherited duplicate is
        # closed. An evaluator guard inherits this descriptor deliberately, so
        # even abrupt worker death cannot unlock a still-running GPU child.
        stream.close()
