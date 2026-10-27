"""Fail-closed resource admission for this trial's first >2M-voxel CPU pool."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
AUTH_SHA = '9726a0713c7744ee44537c0200e17d460e938d9158887ee18a00bdae44792b15'


def digest(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def validate(value, expected, now):
    if value.get('root_execution_authorized') is not True:
        raise ValueError('Explicit conditional-controller admission required')
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError('Admission identity differs: '+key)
    if value.get('actual_windows_physical') is not True or value.get('measurement_source') != 'Windows GlobalMemoryStatusEx':
        raise ValueError('Actual Windows physical memory sample required')
    available = value.get('host_available_bytes')
    if type(available) is not int or available < 7*1024**3:
        raise ValueError('At least7GiB actual physical availability required')
    age = (now-datetime.fromisoformat(value['utc'])).total_seconds()
    if not -5 <= age <= 120:
        raise ValueError('Admission sample must be fresh within120seconds')


def save(file, value):
    temporary = file.with_suffix(file.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(file)


def wait_for_admission(output, manifest, runner, first_job, first_voxels, max_voxels):
    if sys.flags.optimize:
        raise RuntimeError('Very-dense resource admission requires unoptimized Python')
    if Path(output).resolve() != OUT or first_voxels <= 2_000_000:
        raise RuntimeError('Unexpected controlled-trial admission request')
    if digest(OUT/'conditional_controller_authorization_20261025.json') != AUTH_SHA:
        raise RuntimeError('Conditional controller authorization changed')
    expected = dict(manifest_sha256=digest(manifest), runner_sha256=digest(runner),
        admission_guard_sha256=digest(__file__), first_cache_key=first_job[3],
        first_observation_key=first_job[4], first_observed_voxels=int(first_voxels),
        max_group_observed_voxels=int(max_voxels), workers=1, query_threads=8,
        conditional_authorization_sha256=AUTH_SHA)
    marker = OUT/'very_dense_host_admission.json'
    while True:
        if (digest(manifest) != expected['manifest_sha256'] or digest(runner) != expected['runner_sha256']
                or digest(__file__) != expected['admission_guard_sha256']):
            raise RuntimeError('Source or manifest changed while awaiting resource admission')
        try:
            value = json.loads(marker.read_text(encoding='utf-8'))
            validate(value, expected, datetime.now(timezone.utc))
        except (OSError, ValueError, KeyError, TypeError) as error:
            save(OUT/'very_dense_admission_waiting.json', dict(
                utc=datetime.now(timezone.utc).isoformat(), expected=expected,
                reason=str(error), status='Required full fits remain pending; no very-dense pool started.',
                conservative_increment_budget_GiB=4, increment_budget_is_measured_bound=False,
                actual_host_reserve_to_protect_GiB=2, labels_read=False))
            time.sleep(15)
            continue
        save(OUT/'very_dense_admission_used.json', dict(
            utc=datetime.now(timezone.utc).isoformat(), expected=expected,
            admission_sha256=digest(marker), actual_host_available_bytes=value['host_available_bytes'],
            conservative_increment_budget_GiB=4, increment_budget_is_measured_bound=False,
            actual_host_reserve_to_protect_GiB=2, labels_read=False))
        return
