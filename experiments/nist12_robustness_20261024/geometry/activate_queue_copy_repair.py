"""One authorized, cache-preserving receipt-copy repair activation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

OUT = Path(__file__).resolve().parent
QUEUE_PID, QUEUE_START = 3167404, 2552788
RUNNER_PID, RUNNER_START = 3167429, 2552827
EXPECTED_QUEUE = 'eba0309144349622c6a36543cd306f63950b5eeb2c794ef81421b3e4799c6df0'
EXPECTED_RUNNER = '3d5797f1ebf18270425dee10b3a4b73a7696ffb317b9255eaafe4b07526faba6'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def identity(pid):
    folder = Path('/proc')/str(pid)
    try:
        raw = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        argv = [part.decode() for part in (folder/'cmdline').read_bytes().split(b'\0') if part]
        return {'pid': pid, 'ppid': int(raw[1]), 'start_ticks': int(raw[19]),
                'state': raw[0], 'argv': argv}
    except FileNotFoundError:
        return None


def same(row):
    current = identity(row['pid'])
    return current and current['start_ticks'] == row['start_ticks'] and current['argv'] == row['argv']


assert sha(OUT/'queue_cohorts.py') == EXPECTED_QUEUE
assert sha(OUT/'run_depth.py') == EXPECTED_RUNNER
assert json.loads((OUT/'queue_copy_repair_independent_review.json').read_text())
queue, runner = identity(QUEUE_PID), identity(RUNNER_PID)
assert queue['start_ticks'] == QUEUE_START and queue['argv'][1] == str(OUT/'queue_cohorts.py'), queue
assert runner['start_ticks'] == RUNNER_START and runner['ppid'] == QUEUE_PID, runner
assert runner['argv'][1:] == [str(OUT/'run_depth.py'), str(OUT.parent/'capture_manifest.json'),
                              str(OUT.parent/'catalog.json')], runner
children = [row for entry in Path('/proc').iterdir() if entry.name.isdigit()
            if (row := identity(int(entry.name))) and row['ppid'] == RUNNER_PID]
workers = [row for row in children if len(row['argv']) == 4 and row['argv'][0] == runner['argv'][0]
           and row['argv'][1] == '-c' and row['argv'][2].startswith('from multiprocessing.spawn import spawn_main;')
           and row['argv'][3] == '--multiprocessing-fork']
trackers = [row for row in children if len(row['argv']) == 3 and row['argv'][0] == runner['argv'][0]
            and row['argv'][1] == '-c' and row['argv'][2].startswith('from multiprocessing.resource_tracker import main;')]
assert len(children) == 9 and len(workers) == 8 and len(trackers) == 1, children
targets = [queue, runner]+children
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = OUT/'runs/paired_simulation/attempts'/('copy_repair_activation_'+stamp)
archive.mkdir(parents=True)
receipt = {'utc': now(), 'root_authorized': True, 'reason': 'Activate independently reviewed control-only receipt-copy repair.',
           'targets': targets, 'old_loaded_queue_sha256': sha(OUT/'queue_cohorts_before_copy_repair.py'),
           'new_queue_sha256': EXPECTED_QUEUE, 'unchanged_numeric_runner_sha256': EXPECTED_RUNNER,
           'committed_outputs_modified': False, 'labels_read': False}
save(archive/'activation.json', receipt)
stopped = []
try:
    for row in targets:
        assert same(row), row
        os.kill(row['pid'], signal.SIGSTOP)
        stopped.append(row)
    for name in ('pair_index.json', 'execution_protocol.json', 'progress.json', 'completion.json', 'run_depth.py',
                 'queue_cohorts.py', 'queue_cohorts_before_copy_repair.py', 'queue_copy_repair_independent_review.json'):
        source = OUT/name
        if source.exists():
            value = source.read_bytes()
            (archive/name).write_bytes(value)
            assert sha(archive/name) == hashlib.sha256(value).hexdigest()
    cohort = OUT/'runs/paired_simulation'
    for source in cohort.iterdir():
        if source.is_file():
            destination = archive/('cohort_'+source.name)
            value = source.read_bytes()
            destination.write_bytes(value)
            assert sha(destination) == hashlib.sha256(value).hexdigest()
    for row in targets:
        assert same(row), row
        os.kill(row['pid'], signal.SIGTERM)
finally:
    for row in stopped:
        if same(row):
            os.kill(row['pid'], signal.SIGCONT)
deadline = time.monotonic()+45
while any(same(row) for row in targets) and time.monotonic() < deadline:
    time.sleep(.25)
assert not any(same(row) for row in targets), [identity(row['pid']) for row in targets if same(row)]
orphans = []
for source in (OUT/'fits').rglob('*'):
    if source.is_file() and (source.name.endswith('.part') or
                            (source.name.endswith('.json.gz') and not source.with_suffix('').exists())):
        destination = archive/'orphan_files'/source.relative_to(OUT/'fits')
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = source.read_bytes()
        destination.write_bytes(value)
        digest = hashlib.sha256(value).hexdigest()
        assert sha(destination) == digest
        orphans.append({'path': str(source.relative_to(OUT)), 'archive': str(destination.relative_to(OUT)),
                        'bytes': len(value), 'sha256': digest})
receipt.update(verified_exit_utc=now(), all_exact_target_processes_exited=True, orphan_files=orphans,
               orphan_count=len(orphans), original_orphans_deleted=False, restart_started=False)
save(archive/'activation.json', receipt)
save(OUT/'queue_copy_repair_activation.json', receipt | {'archive': str(archive.relative_to(OUT))})
print(json.dumps({'all_target_processes_exited': True, 'archive': str(archive), 'orphan_count': len(orphans)}), flush=True)
