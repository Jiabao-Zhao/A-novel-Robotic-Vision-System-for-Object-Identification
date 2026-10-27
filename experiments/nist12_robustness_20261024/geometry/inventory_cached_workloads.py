"""Label-free inventory from saved exact observation hashes, without preprocessing/fits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRIAL = HERE.parent

def read(file):
    return json.loads(file.read_text(encoding='utf-8-sig'))

def digest(file):
    h = hashlib.sha256()
    with file.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()

def main():
    now = datetime.now(timezone.utc)
    observations_file = HERE/'observation_workloads.json'
    observations = read(observations_file)['observations']
    execution = hashlib.sha256((digest(HERE/'batched.py')+digest(HERE/'dimensions.py')).encode()).hexdigest()
    catalogs = dict(nominal=TRIAL/'catalog.json', paired_simulation=TRIAL/'catalog.json',
        new_query_cads=TRIAL/'new_cads/catalog.json',
        real_tless=TRIAL/'real_data/catalog.json')
    cache = {f.stem for f in (HERE/'fits').glob('*/*.json')}
    mesh_hashes, all_keys, cohorts = {}, set(), []
    for cohort, catalog_file in catalogs.items():
        catalog = read(catalog_file)['objects']
        for item in catalog:
            if item['mesh'] not in mesh_hashes:
                mesh_hashes[item['mesh']] = digest(REPO/item['mesh'])
            if 'mesh_sha256' in item: assert item['mesh_sha256'] == mesh_hashes[item['mesh']]
        keys = {}
        for obs in observations:
            if obs['cohort'] != cohort: continue
            assert obs['catalog_count'] == len(catalog)
            for item in catalog:
                key = hashlib.sha256((execution+mesh_hashes[item['mesh']]+obs['observation_key']).encode()).hexdigest()
                if key in keys: assert keys[key] == obs['observed_voxels']
                keys[key] = obs['observed_voxels']
        pending = set(keys)-cache
        globally_new = pending-all_keys
        bands = dict(ordinary=0, dense=0, very_dense=0)
        for key in globally_new:
            count = keys[key]
            bands['ordinary' if count <= 131072 else 'dense' if count <= 2000000 else 'very_dense'] += 1
        cohorts.append(dict(cohort=cohort, unique_keys=len(keys), cached_keys=len(set(keys)&cache),
            pending_keys=len(pending), pending_not_in_prior_cohorts=len(globally_new),
            unique_pending_resource_jobs=bands, catalog_sha256=digest(catalog_file)))
        all_keys.update(keys)
    assert len(all_keys) == 26920
    progress = read(HERE/'runs/paired_simulation/progress.json')
    paired_rate = 60*progress['completed_new_fits']/progress['seconds']
    reference_file = HERE/'cache_inventory_20261024T214507.json'
    reference_rate = read(reference_file)['current_core_mean_fits_per_minute']
    bank_sizes = [f.stat().st_size for f in (HERE/'fits').glob('*/*.json.gz') if f.name[:-8] in cache]
    pending_total = len(all_keys-cache)
    ordinary_pending = sum(row['unique_pending_resource_jobs']['ordinary'] for row in cohorts)
    report = dict(utc=now.isoformat(), unique_required_keys=len(all_keys),
        cached_required_summaries=len(all_keys&cache), pending_required_keys=pending_total,
        cache_inventory_is_commit_presence_not_full_bank_hash_validation=True,
        cohorts=cohorts, observed_voxel_hash_source_sha256=digest(observations_file),
        execution_hash=execution, mesh_sha256=mesh_hashes,
        current_core_progress=progress, current_core_mean_fits_per_minute=paired_rate,
        current_core_rate_includes_index_and_restart_startup=True,
        prior_warm_8x2_reference_fits_per_minute=reference_rate,
        prior_warm_reference_sha256=digest(reference_file),
        ordinary_pending_at_prior_warm_reference_rate_hours=ordinary_pending/reference_rate/60,
        rate_note='Conditional ordinary-only projection from prior three-hour 8x2 core rate, not total ETA. Current recovery includes index/startup. Other cohorts/CADs and dense/very-dense jobs may have different rates; no aggregate GPU speed is inferred.',
        free_D_disk_bytes=shutil.disk_usage(HERE).free,
        median_current_complete_bank_bytes=statistics.median(bank_sizes),
        projected_additional_bank_bytes_at_current_median=pending_total*statistics.median(bank_sizes),
        labels_read=False, fits_run=0, preprocessing_calls=0)
    output = HERE/('cache_inventory_'+now.strftime('%Y%m%dT%H%M%S')+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('mesh_sha256',)}))
    print(str(output))

if __name__ == '__main__':
    main()
