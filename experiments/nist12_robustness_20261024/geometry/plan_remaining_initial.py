"""Read-only exact cache inventory and workload plan; never score or read labels."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import hashlib
import os
import shutil
import sys
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, path, digest
from run_depth import observation, observation_key

OUT = HERE/'geometry'
COHORTS = (
    ('nominal', OUT/'nominal_inputs.json', HERE/'catalog.json'),
    ('paired_simulation', HERE/'capture_manifest.json', HERE/'catalog.json'),
    ('depth_stress', HERE/'depth_stress_manifest.json', HERE/'catalog.json'),
    ('new_query_cads', HERE/'new_cads/inputs.json', HERE/'new_cads/catalog.json'),
    ('real_tless', HERE/'real_data/inputs.json', HERE/'real_data/catalog.json'),
)


def main():
    started = perf_counter()
    cache = {file.stem for file in (OUT/'fits').glob('*/*.json')}
    bank_sizes = [file.stat().st_size for file in (OUT/'fits').glob('*/*.json.gz') if file.name[:-8] in cache]
    indexed = read(OUT/'pair_index.json')['scene_pairs']
    execution = hashlib.sha256((digest(OUT/'batched.py')+digest(OUT/'dimensions.py')).encode()).hexdigest()
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'cached_unique_fits_at_start': len(cache),
               'disk_free_bytes_at_start': shutil.disk_usage(OUT).free,
               'existing_fit_bank_bytes': sum(bank_sizes),
               'bank_bytes_q50_q95_max': np.quantile(bank_sizes, [.5,.95,1]).tolist() if bank_sizes else [],
               'cohorts': [], 'labels_read': False, 'scoring_calls': 0, 'complete': False}
    all_keys, memo, mesh_hashes = set(), {}, {}
    for name, manifest_file, catalog_file in COHORTS:
        catalog = read(catalog_file)['objects']
        for row in catalog:
            mesh_hashes.setdefault(row['mesh'], digest(path(row['mesh'])))
        keys, pixels, voxels, waiting = set(), [], [], []
        pairs = 0
        scenes = read(manifest_file)['scenes']
        for scene in scenes:
            file = OUT/'localization'/scene['scene_id']/'localization.json'
            if not file.exists():
                waiting.append(scene['scene_id'])
                continue
            loc = read(file)
            depth = np.load(path(scene['depth']), mmap_mode='r')
            K = np.load(path(scene['intrinsics']))
            plane = np.asarray(loc['plane_camera_m'])
            scene_keys = set()
            for proposal in loc['proposals']:
                signature = (loc['input_sha256']['depth'], loc['input_sha256']['intrinsics'],
                             plane.dtype.str, plane.tobytes(),
                             tuple(proposal['roi'][key] for key in ('x1', 'y1', 'x2', 'y2')))
                if signature not in memo:
                    points, voxel, indices, values = observation(depth, K, plane, proposal)
                    memo[signature] = (observation_key(depth.shape, K, plane, voxel, indices, values), len(indices), len(voxel))
                    del points, voxel, indices, values
                obs_key, pixel_count, voxel_count = memo[signature]
                pixels.append(pixel_count)
                voxels.append(voxel_count)
                for item in catalog:
                    key = hashlib.sha256((execution+mesh_hashes[item['mesh']]+obs_key).encode()).hexdigest()
                    keys.add(key)
                    scene_keys.add(key)
                    pairs += 1
            if scene['scene_id'] in indexed:
                assert scene_keys == {row['fit_key'] for row in indexed[scene['scene_id']]}
            del depth
        uncached = keys-cache
        new_global = keys-all_keys
        all_keys.update(keys)
        receipt['cohorts'].append({'name': name, 'scenes': len(scenes), 'waiting_for_localization': waiting,
             'scene_candidate_pairs': pairs, 'unique_fit_keys': len(keys), 'cached_fit_keys': len(keys & cache),
             'uncached_fit_keys': len(uncached), 'uncached_not_in_prior_cohorts': len(new_global-cache),
             'observation_pixels_q50_q95_max': np.quantile(pixels, [.5,.95,1]).tolist() if pixels else [],
             'observed_voxels_q50_q95_max': np.quantile(voxels, [.5,.95,1]).tolist() if voxels else []})
        receipt['unique_uncached_across_all_cohorts'] = len(all_keys-cache)
        if bank_sizes:
            receipt['projected_additional_bank_bytes_using_current_q50_q95'] = [
                int(len(all_keys-cache)*value) for value in np.quantile(bank_sizes, [.5,.95])]
            receipt['storage_estimate_note'] = 'Uses complete current banks; future hypothesis histories may differ. No banks removed or reduced.'
        receipt['seconds'] = perf_counter()-started
        save(OUT/'remaining_plan.json', receipt)
        print(receipt['cohorts'][-1], flush=True)
    receipt['complete'] = not any(row['waiting_for_localization'] for row in receipt['cohorts'])
    receipt['seconds'] = perf_counter()-started
    save(OUT/'remaining_plan.json', receipt)


if __name__ == '__main__':
    main()
