"""Replay one frozen ordinary cached pair with threaded queries, ignoring time only."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import gzip
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, path, digest
from run_depth import observation, observation_key
from batched import native, fit_pair
from query_workers import QueryWorkerTree


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != 'seconds'}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def main():
    out = HERE/'geometry'
    scene = read(out/'nominal_inputs.json')['scenes'][0]
    item = read(HERE/'catalog.json')['objects'][0]
    loc = read(out/'localization'/scene['scene_id']/'localization.json')
    proposal = loc['proposals'][0]
    depth, K = np.load(path(scene['depth']), mmap_mode='r'), np.load(path(scene['intrinsics']))
    plane = np.asarray(loc['plane_camera_m'])
    points, voxel, indices, values = observation(depth, K, plane, proposal)
    obs_key = observation_key(depth.shape, K, plane, voxel, indices, values)
    execution = hashlib.sha256((digest(out/'batched.py')+digest(out/'dimensions.py')).encode()).hexdigest()
    key = hashlib.sha256((execution+digest(path(item['mesh']))+obs_key).encode()).hexdigest()
    summary = read(out/'fits'/key[:2]/(key+'.json'))
    expected = json.loads(gzip.decompress(path(summary['bank_file']).read_bytes()))
    save(out/'query_workers_parity_protocol.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'scene_id': scene['scene_id'], 'object_id': item['object_id'], 'proposal_id': proposal['proposal_id'],
         'cache_key': key, 'baseline_bank_sha256': summary['bank_sha256'], 'observed_voxels': len(voxel),
         'workers': 4, 'ignored_fields': ['seconds'], 'selection': 'First nominal scene, first CAD and proposal; no labels.',
         'batched_sha256': digest(out/'batched.py'), 'query_proxy_sha256': digest(out/'query_workers.py'), 'labels_read': False})
    import open3d as o3d
    cad = native.prepare_cad(o3d.io.read_triangle_mesh(str(path(item['mesh']))))
    cad.tree = QueryWorkerTree(cad.tree, 4)
    mask = np.zeros(depth.shape, dtype=bool)
    mask.ravel()[indices] = True
    started = perf_counter()
    actual = fit_pair(cad, points, plane, K, depth, mask, clearances=(0.,))
    same = without_timing(actual) == without_timing(expected)
    save(out/'query_workers_parity_completion.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'passed': same, 'whole_pose_bank_exact_equal_ignoring_seconds_only': same,
         'hypotheses': sum(arm['hypothesis_count'] for arm in actual['arms']),
         'maximum_numeric_difference': 0. if same else None, 'seconds': perf_counter()-started,
         'cached_reference_key': key, 'labels_read': False})
    assert same, 'Threaded-query replay differs from the frozen entire bank.'
    print(read(out/'query_workers_parity_completion.json'), flush=True)


if __name__ == '__main__':
    main()
