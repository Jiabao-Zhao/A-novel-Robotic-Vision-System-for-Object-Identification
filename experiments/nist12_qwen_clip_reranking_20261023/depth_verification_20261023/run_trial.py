"""Label-free depth verification of the frozen CLIP five-crop shortlists."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO))
WORKERS = 4
CONFIG = {
    'voxel_m': .001,
    'cad_sample_points': 40000,
    'random_seed': 0,
    'iterations': 30,
    'icp_rmse_tolerance_m': 1e-8,
    'yaw_step_deg': 10.,
    'max_support_directions': 12,
    'clearances_m': [0., .001],
    'depth_residual_scale_m': .005,
    'foreground_plane_threshold_m': .002,
    'pixel_offset': 0.,
    'tie_atol': 1e-12,
}
_CAD_CACHE = {}
_SCENE_CACHE = {}
_INPUTS = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(value):
    value = str(value).replace('\\', '/')
    if len(value) > 1 and value[1] == ':' and sys.platform != 'win32':
        value = '/mnt/' + value[0].lower() + value[2:]
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def relative(path):
    path = local(path)
    try:
        return str(path.relative_to(REPO)).replace('\\', '/')
    except ValueError:
        value = str(path).replace('\\', '/')
        if value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
            return value[5].upper() + ':' + value[6:]
        return value


def write_once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Refusing to change saved evidence: {path}'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare():
    import geometry
    assert CONFIG['voxel_m'] == geometry.VOXEL_M
    assert CONFIG['cad_sample_points'] == geometry.SAMPLE_POINTS
    assert CONFIG['iterations'] == geometry.ITERATIONS
    assert CONFIG['max_support_directions'] == geometry.MAX_SUPPORT_FACES
    assert CONFIG['depth_residual_scale_m'] == geometry.DEPTH_SCALE_M
    manifest = read(STUDY / 'data/capture_manifest.json')
    cads = read(STUDY / 'data/cad_inputs.json')
    shortlists = [r for r in read(STUDY / 'clip/shortlist.json') if r['method'] == 'clip_full']
    dino = {(r['state'], r['target'], r['proposal_id']): r
            for r in read(STUDY / 'dino/scores.json')}
    assert len(shortlists) == 96 and len(cads) == 12
    states, models, tasks, pairs = {}, {}, [], []
    source_paths = [STUDY / name for name in ('data/capture_manifest.json',
        'data/cad_inputs.json', 'clip/shortlist.json', 'dino/scores.json')]
    for state, capture in manifest['states'].items():
        assert len(capture['crops']) == 12
        states[state] = {
            'depth_file': capture['depth_file'],
            'intrinsics_file': capture['intrinsics_file'],
            'plane_model_camera': capture['plane_model_camera'],
            'crops': {str(c['proposal_id']): {'roi': c['roi']} for c in capture['crops']},
        }
        source_paths += [local(capture[k]) for k in ('depth_file', 'intrinsics_file')]
    for target, cad in cads.items():
        path = local(cad['mesh_file'])
        assert digest(path) == cad['mesh_sha256'], path
        models[target] = {'mesh_file': relative(path), 'mesh_sha256': cad['mesh_sha256'],
                          'native_extent_m': cad['native_extent_m']}
        source_paths.append(path)
    for row in sorted(shortlists, key=lambda r: (r['state'], r['target'])):
        assert row['status'] == 'ok' and len(row['candidate_ids']) == 5
        tasks.append({'state': row['state'], 'target': row['target'],
                      'candidate_ids': row['candidate_ids']})
        for pid in row['candidate_ids']:
            saved = dino[row['state'], row['target'], pid]
            assert saved['status'] == 'available'
            pairs.append({'state': row['state'], 'target': row['target'], 'proposal_id': pid,
                'global_score': saved['global_score'], 'patch_score': saved['patch_score'],
                'coverage': saved['coverage'], 'dino_score': saved['score'],
                'DINO_CAD_view': saved['selected_CAD_view']})
    assert len(pairs) == len({pair_key(p) for p in pairs}) == 480
    inputs = {'states': states, 'cads': models, 'tasks': tasks, 'pairs': pairs}
    write_once(HERE / 'inputs.json', inputs)
    source_paths += [HERE / name for name in ('run_trial.py', 'geometry.py', 'test_geometry.py', 'select_results.py', 'inputs.json')]
    source_paths += [REPO / 'CADPointCloudRegistration.py',
        REPO / 'scripts/surface_verification.py',
        REPO / 'simulation/nist_capture.py']
    old_hashes = read(STUDY / 'pin_failure_audit/source_hashes.json')
    for name, expected in old_hashes.items():
        assert digest(local(name)) == expected, name
    protected = dict(old_hashes)
    protected.update({relative(p): digest(p) for p in sorted(set(source_paths))})
    protocol = {
        'task': 'One supplied CAD, independently fitted to its five frozen CLIP crops.',
        'scenes': sorted(states), 'products': 12, 'tasks': 96, 'candidate_pairs': 480,
        'configuration': CONFIG,
        'primary_arm': 'zero-contact table fit; documented DINO/geometry fusion',
        'controls': ['zero-contact geometry alone', 'zero-contact one-sided ICP RMSE alone',
                     'declared 1 mm simulator-clearance geometry and fusion sensitivity'],
        'orientation': 'CAD-only stable support directions; 36 yaw initializations per direction; continuous table x/y/yaw refinement.',
        'scale': 'Native CAD meters, no fitted scale, no object pose or support choice from simulator.',
        'observations': 'Original single-camera depth and intrinsics; existing observed plane and inclusive ROIs; same 2 mm plane foreground mask as DINO.',
        'geometry': 'G=1-(P_obs+P_cad+P_depth)/3; both silhouette mismatches divided by full projected/observed union; P_depth=mean(min(abs(Z_observed-Z_CAD)/0.005m,1)) on valid object overlap.',
        'geometry_visibility': 'Raycast self-visible CAD; full projected CAD silhouette, no external-occluder carving or table-penetration acceptance gate.',
        'pose_selection': 'Maximize G over the fitted orientation hypotheses separately for each crop and clearance; retain raw-RMSE alternative and all poses.',
        'fusion': '(global + patch + r*G)/(2+r), with exactly the frozen DINO scores and patch-derived coverage r.',
        'unavailable': 'Empty or invalid observed depth and no valid CAD/object depth overlap remain explicit unavailable; no fabricated zero scores or fallback.',
        'task_selection': 'Require all five candidate scores available for an arm; choose unique maximum (RMSE control unique minimum), ties within 1e-12 remain unresolved.',
        'clearance': 'Zero contact primary; 1 mm known synthetic capture offset is a separate sensitivity condition, not estimated physical calibration.',
        'evaluation': 'Freeze all label-free fits and predictions before a separate evaluation process reads audit identities. No parameter tuning by target or outcome.',
        'new_model_inference': False, 'new_weights': False, 'active_runtime_modified': False,
        'source_sha256': protected,
    }
    write_once(HERE / 'protocol.json', protocol)
    print(json.dumps({'prepared_pairs': len(pairs), 'protected_sources': len(protected),
        'protocol_sha256': digest(HERE / 'protocol.json')}), flush=True)
    return inputs


def pair_key(pair):
    return f"{pair['state']}__{pair['target']}__p{pair['proposal_id']:03d}"


def check_sources():
    for name, expected in read(HERE / 'protocol.json')['source_sha256'].items():
        assert digest(local(name)) == expected, name


def observation(pair):
    global _SCENE_CACHE
    state = pair['state']
    capture = _INPUTS['states'][state]
    if _SCENE_CACHE.get('state') != state:
        _SCENE_CACHE = {'state': state, 'depth': np.load(local(capture['depth_file'])),
                        'K': np.load(local(capture['intrinsics_file']))}
    depth, K = _SCENE_CACHE['depth'], _SCENE_CACHE['K']
    roi = capture['crops'][str(pair['proposal_id'])]['roi']
    x1, y1, x2, y2 = [roi[k] for k in ('x1', 'y1', 'x2', 'y2')]
    yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
    z = depth[y1:y2+1, x1:x2+1]
    xyz = np.stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z), axis=-1)
    plane = np.asarray(capture['plane_model_camera'], dtype=np.float64)
    distance = np.abs(xyz @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3])
    foreground = np.isfinite(z) & (z > 0) & (distance > CONFIG['foreground_plane_threshold_m'])
    mask = np.zeros(depth.shape, dtype=bool)
    mask[y1:y2+1, x1:x2+1] = foreground
    points = xyz[foreground].astype(np.float64)
    return points, plane, K, depth, mask


def worker_init():
    global _INPUTS
    _INPUTS = read(HERE / 'inputs.json')


def fit_one(pair):
    from geometry import prepare_cad, fit_pair
    import open3d as o3d
    output = HERE / 'fits' / (pair_key(pair) + '.json')
    if output.exists():
        return {'key': pair_key(pair), 'cached': True}
    start = perf_counter()
    target = pair['target']
    if target not in _CAD_CACHE:
        mesh = o3d.io.read_triangle_mesh(str(local(_INPUTS['cads'][target]['mesh_file'])))
        _CAD_CACHE[target] = prepare_cad(mesh)
    points, plane, K, depth, mask = observation(pair)
    result = fit_pair(_CAD_CACHE[target], points, plane, K, depth, mask,
                      clearances=tuple(CONFIG['clearances_m']))
    row = {'state': pair['state'], 'target': target, 'proposal_id': pair['proposal_id'],
        'baseline': {k: pair[k] for k in ('global_score', 'patch_score', 'coverage', 'dino_score', 'DINO_CAD_view')},
        'observed_points': len(points), 'geometry': result, 'wall_seconds': perf_counter()-start,
        'protocol_sha256': digest(HERE / 'protocol.json')}
    write_once(output, row)
    return {'key': pair_key(pair), 'seconds': row['wall_seconds']}


def run(smoke=False):
    check_sources()
    inputs = read(HERE / 'inputs.json')
    pairs = inputs['pairs'][:1] if smoke else inputs['pairs']
    started = perf_counter()
    if smoke:
        worker_init()
        print(json.dumps(fit_one(pairs[0])), flush=True)
        return
    pending = [p for p in pairs if not (HERE / 'fits' / (pair_key(p)+'.json')).exists()]
    print(json.dumps({'pending_pairs': len(pending), 'workers': WORKERS}), flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as pool:
        futures = {pool.submit(fit_one, pair): pair for pair in pending}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps({'completed': len(pairs)-len(pending)+number, 'total': len(pairs),
                              'wall_seconds': perf_counter()-started, **result}), flush=True)
    check_sources()
    paths = [HERE / 'fits' / (pair_key(p)+'.json') for p in pairs]
    assert all(p.exists() for p in paths)
    write_once(HERE / 'fitting_completion.json', {
        'candidate_pairs': len(paths), 'all_sources_unchanged': True,
        'protocol_sha256': digest(HERE / 'protocol.json'),
        'fit_sha256': {p.name: digest(p) for p in paths},
        'evaluation_labels_read': False, 'new_model_inference': False,
    })
    print(json.dumps({'fitting_complete': len(paths)}), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'prepare'
    if mode == 'prepare':
        prepare()
    elif mode in ('run', 'smoke'):
        run(smoke=mode == 'smoke')
    else:
        raise ValueError('Use prepare, smoke, or run.')
