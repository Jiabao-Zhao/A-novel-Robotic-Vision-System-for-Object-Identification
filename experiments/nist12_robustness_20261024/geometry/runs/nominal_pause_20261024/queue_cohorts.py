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


def now():
    return datetime.now(timezone.utc).isoformat()


def preserve(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for filename in RECEIPTS:
        source = OUT/filename
        if source.exists():
            temporary = destination/(filename+'.part')
            shutil.copyfile(source, temporary)
            temporary.replace(destination/filename)


def finished(manifest):
    file = OUT/'completion.json'
    return file.exists() and read(file)['manifest_sha256'] == digest(manifest)


def missing_localizations(manifest):
    return [row['scene_id'] for row in read(manifest)['scenes']
            if not (OUT/'localization'/row['scene_id']/'localization.json').exists()]


def main():
    nominal = OUT/'nominal_inputs.json'
    nominal_receipt = OUT/'runs/nominal'
    nominal_receipt.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT/'run_depth.py', nominal_receipt/'run_depth.py')
    save(nominal_receipt/'queue_receipt.json', {'manifest': relative(nominal),
         'manifest_sha256': digest(nominal), 'observed_existing_run_utc': now(),
         'labels_read': False})
    while not finished(nominal):
        preserve(nominal_receipt)
        time.sleep(5)
    preserve(nominal_receipt)
    print('Nominal cohort receipt preserved.', flush=True)
    for name, manifest, catalog in COHORTS:
        destination = OUT/'runs'/name
        destination.mkdir(parents=True, exist_ok=True)
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
                preserve(destination)
                time.sleep(5)
        preserve(destination)
        save(destination/'exit.json', {'returncode': process.returncode, 'finished_utc': now(),
             'completed_expected_manifest': finished(manifest), 'labels_read': False})
        if process.returncode or not finished(manifest):
            raise RuntimeError(f'{name} failed; preserved log and execution receipts in {destination}')
        print(f'{name} finished.', flush=True)
    save(OUT/'queue_completion.json', {'finished_utc': now(),
         'cohorts': ['nominal']+[row[0] for row in COHORTS], 'labels_read': False})


if __name__ == '__main__':
    main()
