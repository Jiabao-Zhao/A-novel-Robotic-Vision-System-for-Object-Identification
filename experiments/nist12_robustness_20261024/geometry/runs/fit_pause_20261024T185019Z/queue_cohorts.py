"""Serial controlled-trial queue; preserve each cohort's execution receipts."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
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


def now():
    return datetime.now(timezone.utc).isoformat()


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
            temporary = destination/(filename+'.part')
            shutil.copyfile(source, temporary)
            temporary.replace(destination/filename)
            _COPIED[str(destination), filename] = signature


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
        shutil.copyfile(file, archive/file.name)


def missing_localizations(manifest):
    return [row['scene_id'] for row in read(manifest)['scenes']
            if not (OUT/'localization'/row['scene_id']/'localization.json').exists()]


def main():
    nominal = OUT/'nominal_inputs.json'
    nominal_receipt = OUT/'runs/nominal'
    nominal_receipt.mkdir(parents=True, exist_ok=True)
    if not durable_finished(nominal_receipt, nominal):
        if not (nominal_receipt/'run_depth.py').exists():
            shutil.copyfile(OUT/'run_depth.py', nominal_receipt/'run_depth.py')
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
        shutil.copyfile(OUT/'run_depth.py', destination/'run_depth.py')
        save(destination/'launch.json', {'command': command, 'started_utc': now(),
             'runner_sha256': digest(OUT/'run_depth.py'), 'labels_read': False})
        print(f'Starting {name}.', flush=True)
        with (destination/'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            while process.poll() is None:
                preserve(destination, manifest)
                time.sleep(5)
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
