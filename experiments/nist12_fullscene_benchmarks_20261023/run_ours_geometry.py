"""Frozen native-scale geometry fitting on fresh depth proposals and CLIP top five."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import multiprocessing
import sys
from time import perf_counter
from common import HERE, REPO, read, save, path, digest
import numpy as np

OLD = HERE.parent / 'nist12_qwen_clip_reranking_20261023/depth_verification_20261023'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OLD))
from geometry import prepare_cad, fit_pair

_CAD = {}
_OBS = {}


def fit(pair):
    import open3d as o3d
    scene_id, target, proposal_id = pair
    destination = HERE / 'ours/fits' / f'{scene_id}__{target}__p{proposal_id:03d}.json'
    if destination.exists():
        return destination.name
    start = perf_counter()
    inputs = read(HERE / 'inputs.json')
    if target not in _CAD:
        item = next(r for r in inputs['objects'] if r['target'] == target)
        _CAD[target] = prepare_cad(o3d.io.read_triangle_mesh(str(path(item['mesh']))))
    if scene_id not in _OBS:
        scene = next(r for r in inputs['scenes'] if r['scene_id'] == scene_id)
        depth = np.load(path(scene['depth']))
        K = np.load(path(scene['intrinsics']))
        loc = read(HERE / 'ours' / scene_id / 'localization.json')
        _OBS.clear()
        _OBS[scene_id] = depth, K, loc
    depth, K, loc = _OBS[scene_id]
    roi = loc['proposals'][proposal_id]['roi']
    x1, y1, x2, y2 = [roi[k] for k in ('x1','y1','x2','y2')]
    yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
    z = depth[y1:y2+1, x1:x2+1]
    xyz = np.stack(((xx-K[0,2])*z/K[0,0], (yy-K[1,2])*z/K[1,1], z), axis=-1)
    plane = np.asarray(loc['plane_camera_m'])
    distance = np.abs(xyz @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3])
    foreground = np.isfinite(z) & (z > 0) & (distance > .002)
    mask = np.zeros(depth.shape, bool)
    mask[y1:y2+1,x1:x2+1] = foreground
    result = fit_pair(_CAD[target], xyz[foreground], plane, K, depth, mask, clearances=(0.,))
    save(destination, {'scene_id': scene_id, 'target': target, 'proposal_id': proposal_id,
         'geometry': result, 'seconds': perf_counter()-start,
         'geometry_source_sha256': digest(OLD / 'geometry.py'), 'labels_read': False})
    return destination.name


def main():
    start = perf_counter()
    inputs = read(HERE / 'inputs.json')
    shortlists = read(HERE / 'ours/shortlists.json')
    pairs = [(r['scene_id'],r['target'],p) for r in shortlists if r['status']=='ok' for p in r['candidate_ids']]
    save(HERE / 'ours/geometry_protocol.json', {'clearances_m':[0.], 'geometry_source_sha256':digest(OLD / 'geometry.py'),
         'shortlists_sha256':digest(HERE / 'ours/shortlists.json'), 'pairs':len(pairs), 'workers':8, 'labels_read':False})
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(fit, p) for p in pairs]
        for index, future in enumerate(as_completed(futures),1):
            name = future.result()
            save(HERE / 'ours/geometry_progress.json', {'completed':index, 'total':len(pairs), 'seconds':perf_counter()-start, 'last':name})
            if index % 10 == 0:
                print(f'Geometry {index}/{len(pairs)}, {perf_counter()-start:.0f}s', flush=True)
    for scene in inputs['scenes']:
        folder = HERE / 'ours' / scene['scene_id']
        proposals = read(folder / 'localization.json')['proposals']
        rows, decisions = [], []
        for item in inputs['objects']:
            target = item['target']
            shortlist = next(r for r in shortlists if r['scene_id']==scene['scene_id'] and r['target']==target)
            available = []
            for proposal_id in shortlist['candidate_ids']:
                record = read(HERE / 'ours/fits' / f"{scene['scene_id']}__{target}__p{proposal_id:03d}.json")
                arm = record['geometry']['arms'][0]
                best = arm.get('best_geometry')
                if best is not None:
                    available.append((best['verification']['geometry_score'],proposal_id))
            available.sort(reverse=True)
            status = 'unavailable'
            if len(available)==len(shortlist['candidate_ids']) and available:
                status = 'tie' if len(available)>1 and abs(available[0][0]-available[1][0])<=1e-12 else 'ok'
            if status == 'ok':
                score, pid = available[0]
                proposal = proposals[pid]
                rows.append({'proposal_id':pid, 'object_id':item['object_id'], 'score':score,
                             'bbox_xyxy':proposal['bbox_xyxy'], 'mask_index':pid})
            decisions.append({'object_id':item['object_id'], 'target':target, 'status':status, 'scores':available,
                              'shortlist_ids':shortlist['candidate_ids']})
        save(folder / 'detections.json', rows)
        save(folder / 'decisions.json', decisions)
    save(HERE / 'ours/completion.json', {'scenes':8, 'tasks':96, 'fit_pairs':len(pairs), 'seconds':perf_counter()-start,
         'own_proposals':True, 'labels_read':False})


if __name__ == '__main__':
    main()
