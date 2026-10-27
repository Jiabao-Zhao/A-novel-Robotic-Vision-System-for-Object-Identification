"""Pause only this trial's nominal fit process and its controlled serial queue."""
from pathlib import Path
from datetime import datetime, timezone
import os
import signal
import sys
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, digest, relative

out = HERE/'geometry'
snapshot = out/'runs'/datetime.now(timezone.utc).strftime('fit_pause_%Y%m%dT%H%M%SZ')
snapshot.mkdir(parents=True, exist_ok=True)
for name in ('run_depth.py', 'queue_cohorts.py', 'execution_protocol.json', 'progress.json', 'pair_index.json'):
    file = out/name
    if file.exists():
        shutil.copyfile(file, snapshot/name)
processes = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        command = (directory/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        status = (directory/'status').read_text()
        ppid = int(next(row for row in status.splitlines() if row.startswith('PPid:')).split()[1])
    except (OSError, StopIteration, ValueError):
        continue
    processes.append({'pid': int(directory.name), 'ppid': ppid, 'command': command})
manifests = [out/'nominal_inputs.json', HERE/'capture_manifest.json',
             HERE/'new_cads/inputs.json', HERE/'real_data/inputs.json']
allowed = [value for file in manifests for value in (str(file), relative(file))]
fitting = [row for row in processes if str(out/'run_depth.py')+' ' in row['command']
           and any(value+' ' in row['command'] for value in allowed)]
queues = [row for row in processes if str(out/'queue_cohorts.py') in row['command']]
assert len(fitting) == 1 and len(queues) == 1, (fitting, queues)
children = [row for row in processes if row['ppid'] == fitting[0]['pid']]
targets = queues + children + fitting
save(snapshot/'pause_receipt.json', {'utc': datetime.now(timezone.utc).isoformat(),
     'reason': 'Parent-authorized execution-only worker change; cache-preserving resource restart.',
     'targets': targets, 'completed_cache_preserved': True, 'labels_read': False})
for row in targets:
    os.kill(row['pid'], signal.SIGTERM)
for row in targets:
    try:
        command = (Path('/proc')/str(row['pid'])/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if command == row['command']:
            os.kill(row['pid'], signal.SIGCONT)
    except (FileNotFoundError, ProcessLookupError):
        pass
print({'terminated_pids': [row['pid'] for row in targets], 'cache_preserved': True})
