"""Serial controlled-trial queue; preserve each cohort's execution receipts."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import hashlib
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, digest, relative

OUT = HERE/'geometry'
RECEIPTS = ('pair_index.json', 'execution_protocol.json', 'progress.json', 'completion.json')
COHORTS = (
    ('paired_simulation', HERE/'capture_manifest.json', HERE/'catalog.json'),
    ('depth_stress', HERE/'depth_stress_manifest.json', HERE/'catalog.json'),
    ('new_query_cads', HERE/'new_cads/inputs.json', HERE/'new_cads/catalog.json'),
    ('real_tless', HERE/'real_data/inputs.json', HERE/'real_data/catalog.json'),
)
_COPIED, _INDEX_STATE = {}, {}
COPY_ATTEMPTS = 3


def now():
    return datetime.now(timezone.utc).isoformat()


def copy_receipt(source, destination):
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name+'.part')
    for attempt in range(COPY_ATTEMPTS):
        try:
            value = source.read_bytes()
            temporary.write_bytes(value)
            if hashlib.sha256(temporary.read_bytes()).digest() != hashlib.sha256(value).digest():
                raise OSError('Receipt copy byte hash mismatch')
            temporary.replace(destination)
            return
        except OSError as error:
            if error.errno != 61 or attempt+1 == COPY_ATTEMPTS:
                raise
            time.sleep(.05)


def preserve(destination, manifest):
    destination.mkdir(parents=True, exist_ok=True)
    index = OUT/'pair_index.json'
    if not index.exists():
        return
    stat = index.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    if _INDEX_STATE.get('signature') != signature:
        _INDEX_STATE.update(signature=signature, scenes=set(read(index)['scene_pairs']))
    if _INDEX_STATE['scenes'] != {row['scene_id'] for row in read(manifest)['scenes']}:
        return
    for filename in RECEIPTS:
        source = OUT/filename
        if source.exists():
            if filename == 'completion.json' and read(source)['manifest_sha256'] != digest(manifest):
                continue
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if _COPIED.get((str(destination), filename)) == signature:
                continue
            copy_receipt(source, destination/filename)
            _COPIED[str(destination), filename] = signature


def preserve_while_running(process, destination, manifest):
    try:
        while process.poll() is None:
            preserve(destination, manifest)
            time.sleep(5)
    except Exception as error:
        failure = {'utc': now(), 'child_pid': process.pid, 'error_type': type(error).__name__,
                   'message': str(error), 'errno': getattr(error, 'errno', None), 'labels_read': False}
        try:
            save(destination/'receipt_copy_failure_before_drain.json', dict(failure, child_drained=False))
        except Exception as diagnostic_error:
            print(f'Could not save copy failure diagnostics: {diagnostic_error!r}', file=sys.stderr, flush=True)
        process.wait()
        try:
            save(destination/'receipt_copy_failure_after_drain.json', dict(failure, child_drained=True,
                 child_returncode=process.returncode, drained_utc=now()))
        except Exception as diagnostic_error:
            print(f'Could not save drained-child diagnostics: {diagnostic_error!r}', file=sys.stderr, flush=True)
        raise


def finished(manifest):
    file = OUT/'completion.json'
    return file.exists() and read(file)['manifest_sha256'] == digest(manifest)


def durable_finished(destination, manifest):
    file = destination/'completion.json'
    return file.exists() and read(file)['manifest_sha256'] == digest(manifest)


def archive_attempt(destination):
    existing = [file for file in destination.iterdir() if file.is_file()]
    if not existing:
        return
    archive = destination/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive.mkdir(parents=True)
    for file in existing:
        copy_receipt(file, archive/file.name)


def missing_localizations(manifest):
    return [row['scene_id'] for row in read(manifest)['scenes']
            if not (OUT/'localization'/row['scene_id']/'localization.json').exists()]


def main():
    nominal = OUT/'nominal_inputs.json'
    nominal_receipt = OUT/'runs/nominal'
    nominal_receipt.mkdir(parents=True, exist_ok=True)
    if not durable_finished(nominal_receipt, nominal):
        if not (nominal_receipt/'run_depth.py').exists():
            copy_receipt(OUT/'run_depth.py', nominal_receipt/'run_depth.py')
        save(nominal_receipt/'queue_receipt.json', {'manifest': relative(nominal),
             'manifest_sha256': digest(nominal), 'observed_existing_run_utc': now(),
             'labels_read': False})
        while not finished(nominal):
            preserve(nominal_receipt, nominal)
            time.sleep(5)
        preserve(nominal_receipt, nominal)
    print('Nominal cohort receipt preserved.', flush=True)
    for name, manifest, catalog in COHORTS:
        destination = OUT/'runs'/name
        destination.mkdir(parents=True, exist_ok=True)
        if durable_finished(destination, manifest):
            print(f'{name} already complete; durable receipt retained.', flush=True)
            continue
        archive_attempt(destination)
        save(destination/'queue_receipt.json', {'manifest': relative(manifest),
             'manifest_sha256': digest(manifest), 'catalog': relative(catalog),
             'catalog_sha256': digest(catalog), 'queued_utc': now(), 'labels_read': False})
        while missing_localizations(manifest):
            save(destination/'waiting.json', {'utc': now(), 'missing_localizations': missing_localizations(manifest),
                 'labels_read': False})
            time.sleep(10)
        command = [sys.executable, str(OUT/'run_depth.py'), str(manifest), str(catalog)]
        copy_receipt(OUT/'run_depth.py', destination/'run_depth.py')
        save(destination/'launch.json', {'command': command, 'started_utc': now(),
             'runner_sha256': digest(OUT/'run_depth.py'), 'labels_read': False})
        print(f'Starting {name}.', flush=True)
        with (destination/'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            preserve_while_running(process, destination, manifest)
        preserve(destination, manifest)
        save(destination/'exit.json', {'returncode': process.returncode, 'finished_utc': now(),
             'completed_expected_manifest': finished(manifest), 'labels_read': False})
        if process.returncode or not finished(manifest):
            raise RuntimeError(f'{name} failed; preserved log and execution receipts in {destination}')
        print(f'{name} finished.', flush=True)
    save(OUT/'queue_completion.json', {'finished_utc': now(),
         'cohorts': ['nominal']+[row[0] for row in COHORTS], 'labels_read': False})


if __name__ == '__main__':
    main()
