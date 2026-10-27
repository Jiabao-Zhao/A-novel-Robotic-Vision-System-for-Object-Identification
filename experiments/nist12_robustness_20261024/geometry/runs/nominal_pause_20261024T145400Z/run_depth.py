"""All-CAD/all-proposal geometry and metric-size scores, with exact-input reuse."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import hashlib
import json
import gzip
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, path, digest, relative
from batched import native, fit_pair
from dimensions import observed_dimensions, cad_dimensions, score_dimensions

_CAD, _SCENES, _ITEMS, _OBS, _OUT = {}, {}, {}, {}, None
WORKERS = 2


def atomic_save(destination, value):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(destination)


def observation(depth, K, plane, proposal):
    roi = proposal['roi']
    x1, y1, x2, y2 = [roi[key] for key in ('x1', 'y1', 'x2', 'y2')]
    yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
    z = depth[y1:y2+1, x1:x2+1]
    xyz = np.stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z), axis=-1)
    foreground = np.isfinite(z) & (z > 0) & (np.abs(xyz @ plane[:3] + plane[3])/np.linalg.norm(plane[:3]) > .002)
    indices = ((yy[foreground]*depth.shape[1]) + xx[foreground]).astype(np.int64)
    points = xyz[foreground]
    voxel = native.voxel_means(points) if len(points) else points
    return points, voxel, indices, z[foreground]


def observation_key(depth_shape, K, plane, voxel, indices, observed_depth):
    hash_value = hashlib.sha256()
    for value in (np.asarray(depth_shape, dtype=np.int64), K, plane, voxel, indices, observed_depth):
        array = np.ascontiguousarray(value)
        hash_value.update(str((array.dtype.str, array.shape)).encode())
        hash_value.update(array.tobytes())
    return hash_value.hexdigest()


def initialize(catalog, scenes, out):
    global _ITEMS, _SCENES, _OUT
    _ITEMS = {row['object_id']: row for row in catalog}
    _SCENES = {row['scene_id']: row for row in scenes}
    _OUT = Path(out)


def fit_job(job):
    scene_id, object_id, pid, key, obs_key = job
    result_file = _OUT/'fits'/key[:2]/(key+'.json')
    if result_file.exists():
        record = read(result_file)
        assert record['cache_key'] == key and record['observation_key'] == obs_key
        return key
    start = perf_counter()
    item, scene = _ITEMS[object_id], _SCENES[scene_id]
    if object_id not in _CAD:
        import open3d as o3d
        cad = native.prepare_cad(o3d.io.read_triangle_mesh(str(path(item['mesh']))))
        _CAD[object_id] = cad, cad_dimensions(cad, native)
    cad, expected_dimensions = _CAD[object_id]
    if scene_id not in _OBS:
        _OBS.clear()
        loc = read(_OUT/'localization'/scene_id/'localization.json')
        _OBS[scene_id] = (np.load(path(scene['depth']), mmap_mode='r'), np.load(path(scene['intrinsics'])), loc, {})
    depth, K, loc, prepared = _OBS[scene_id]
    plane = np.asarray(loc['plane_camera_m'])
    if pid not in prepared:
        proposal = next(row for row in loc['proposals'] if row['proposal_id'] == pid)
        points, voxel, indices, values = observation(depth, K, plane, proposal)
        assert observation_key(depth.shape, K, plane, voxel, indices, values) == obs_key
        prepared[pid] = (points, voxel, indices)
    points, voxel, indices = prepared[pid]
    mask = np.zeros(depth.shape, dtype=bool)
    mask.ravel()[indices] = True
    result = fit_pair(cad, points, plane, K, depth, mask, clearances=(0.,))
    arm = result['arms'][0] if result['arms'] else None
    best = None if arm is None else arm['best_geometry']
    size = score_dimensions(observed_dimensions(voxel, plane, native), expected_dimensions)
    archive = result_file.with_suffix('.json.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.gz.part')
    temporary.write_bytes(gzip.compress(json.dumps(result, separators=(',', ':'), allow_nan=False).encode(), compresslevel=1, mtime=0))
    temporary.replace(archive)
    summary = {'cache_key': key, 'observation_key': obs_key, 'first_scene_id': scene_id,
               'object_id': object_id, 'target': item['target'], 'first_proposal_id': pid,
               'status': result['status'] if arm is None else arm['status'],
               'geometry_score': None if best is None else best['verification']['geometry_score'],
               'best_geometry': best, 'best_geometry_index': None if arm is None else arm['best_geometry_index'],
               'hypothesis_count': 0 if arm is None else arm['hypothesis_count'],
               'available_projections': 0 if arm is None else arm['available_projections'],
               'size_score': size['score'], 'size': size, 'bank_file': relative(archive),
               'bank_sha256': digest(archive), 'seconds': perf_counter()-start, 'labels_read': False}
    atomic_save(result_file, summary)
    return key


def scene_scores(scene, rows, output):
    pairs = []
    for row in rows:
        fit_file = output/'fits'/row['fit_key'][:2]/(row['fit_key']+'.json')
        if not fit_file.exists():
            return False
        record = read(fit_file)
        pairs.append(dict(row, fit_file=relative(fit_file), status=record['status'],
                          geometry_score=record['geometry_score'], size_score=record['size_score']))
    atomic_save(output/'scores'/(scene['scene_id']+'.json'), {'scene_id': scene['scene_id'], 'pairs': pairs,
                'labels_read': False, 'availability': 'Unavailable candidates remain explicit; no task-level all-candidates veto.'})
    return True


def main():
    started = perf_counter()
    manifest_file = path(sys.argv[1]) if len(sys.argv) > 1 else HERE/'capture_manifest.json'
    catalog_file = path(sys.argv[2]) if len(sys.argv) > 2 else HERE/'catalog.json'
    output = path(sys.argv[3]) if len(sys.argv) > 3 else HERE/'geometry'
    manifest, catalog = read(manifest_file), read(catalog_file)['objects']
    parity = read(HERE/'geometry/batch_parity_completion.json')
    assert parity['passed'] and parity['batched_source_sha256'] == digest(Path(__file__).with_name('batched.py'))
    execution_hash = hashlib.sha256((digest(Path(__file__).with_name('batched.py'))+digest(Path(__file__).with_name('dimensions.py'))).encode()).hexdigest()
    mesh_hashes = {row['object_id']: digest(path(row['mesh'])) for row in catalog}
    scenes, waiting = [], []
    for scene in manifest['scenes']:
        localization = output/'localization'/scene['scene_id']/'localization.json'
        (scenes if localization.exists() else waiting).append(scene)
    jobs, unique_jobs, index = [], {}, {}
    for scene in scenes:
        loc = read(output/'localization'/scene['scene_id']/'localization.json')
        depth, K = np.load(path(scene['depth']), mmap_mode='r'), np.load(path(scene['intrinsics']))
        plane, rows = np.asarray(loc['plane_camera_m']), []
        for proposal in loc['proposals']:
            points, voxel, indices, values = observation(depth, K, plane, proposal)
            obs_key = observation_key(depth.shape, K, plane, voxel, indices, values)
            for item in catalog:
                key = hashlib.sha256((execution_hash+mesh_hashes[item['object_id']]+obs_key).encode()).hexdigest()
                job = (scene['scene_id'], item['object_id'], proposal['proposal_id'], key, obs_key)
                rows.append({'object_id': item['object_id'], 'target': item['target'],
                             'proposal_id': proposal['proposal_id'], 'fit_key': key})
                if not (output/'fits'/key[:2]/(key+'.json')).exists():
                    unique_jobs.setdefault(key, job)
        index[scene['scene_id']] = rows
        scene_scores(scene, rows, output)
    jobs = list(unique_jobs.values())
    atomic_save(output/'pair_index.json', {'scene_pairs': index, 'new_unique_fit_jobs': len(jobs),
                'waiting_for_localization': [row['scene_id'] for row in waiting], 'labels_read': False})
    print(f'Depth: {len(scenes)} ready scenes, {len(jobs)} unique pending fits, {len(waiting)} waiting scenes', flush=True)
    atomic_save(output/'execution_protocol.json', {'frozen_geometry_sha256': native.EXPECTED_SHA256 if hasattr(native, 'EXPECTED_SHA256') else digest(Path(native.__file__)),
                'wrapper_sha256': digest(__file__), 'batched_sha256': digest(Path(__file__).with_name('batched.py')),
                'dimension_sha256': digest(Path(__file__).with_name('dimensions.py')), 'workers': WORKERS,
                'cache': 'Exact CAD, geometry, mask indices and observed-depth inputs; RGB and irrelevant outside-mask depth do not affect the score.',
                'labels_read': False, 'parity_sha256': digest(HERE/'geometry/batch_parity_completion.json')})
    if jobs:
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
             initializer=initialize, initargs=(catalog, scenes, str(output))) as pool:
            futures = [pool.submit(fit_job, job) for job in jobs]
            for number, future in enumerate(as_completed(futures), 1):
                key = future.result()
                atomic_save(output/'progress.json', {'completed_new_fits': number, 'total_new_fits': len(jobs),
                            'seconds': perf_counter()-started, 'last_key': key})
                if number % 20 == 0:
                    print(f'Depth {number}/{len(jobs)}, {perf_counter()-started:.1f}s', flush=True)
                if number % 100 == 0:
                    for scene in scenes:
                        scene_scores(scene, index[scene['scene_id']], output)
    finished = sum(scene_scores(scene, index[scene['scene_id']], output) for scene in scenes)
    atomic_save(output/'completion.json', {'scenes_scored': finished, 'scene_candidate_pairs': sum(len(v) for v in index.values()),
                'unique_new_fit_jobs': len(jobs), 'manifest_sha256': digest(manifest_file), 'catalog_sha256': digest(catalog_file),
                'seconds': perf_counter()-started, 'waiting_for_localization': [row['scene_id'] for row in waiting], 'labels_read': False})


if __name__ == '__main__':
    main()
