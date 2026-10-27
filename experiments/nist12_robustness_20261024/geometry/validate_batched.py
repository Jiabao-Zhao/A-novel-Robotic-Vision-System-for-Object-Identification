"""Pre-label parity/profile using only the prior frozen CAD/candidate fit records."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, PRIOR, read, save, path, digest
from batched import native, fit_pair

TOLERANCE = 1e-6


def main():
    save(HERE/'geometry/batch_parity_protocol.json', {'labels_read': False, 'source': 'Prior N00 saved CAD/candidate scores only',
         'selection': 'Highest saved available geometry score per CAD; no scene identities used.',
         'max_geometry_score_tolerance': TOLERANCE, 'method_changes': False,
         'batched_source_sha256': digest(Path(__file__).with_name('batched.py'))})
    inputs = read(PRIOR/'inputs.json')
    scene = next(row for row in inputs['scenes'] if row['scene_id'] == 'N00')
    depth, K = np.load(path(scene['depth'])), np.load(path(scene['intrinsics']))
    localization = read(PRIOR/'ours/N00/localization.json')
    plane = np.asarray(localization['plane_camera_m'])
    records = []
    for item in inputs['objects']:
        options = []
        for file in sorted((PRIOR/'ours/fits').glob('N00__'+item['target']+'__p*.json')):
            record = read(file)
            best = record['geometry']['arms'][0]['best_geometry']
            if best is not None:
                options.append((best['verification']['geometry_score'], file, record))
        _, source, reference = max(options, key=lambda row: row[0])
        pid = reference['proposal_id']
        roi = localization['proposals'][pid]['roi']
        x1, y1, x2, y2 = [roi[key] for key in ('x1', 'y1', 'x2', 'y2')]
        yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
        z = depth[y1:y2+1, x1:x2+1]
        xyz = np.stack(((xx-K[0,2])*z/K[0,0], (yy-K[1,2])*z/K[1,1], z), axis=-1)
        foreground = np.isfinite(z) & (z > 0) & (np.abs(xyz @ plane[:3] + plane[3])/np.linalg.norm(plane[:3]) > .002)
        mask = np.zeros(depth.shape, dtype=bool)
        mask[y1:y2+1, x1:x2+1] = foreground
        cad = native.prepare_cad(o3d.io.read_triangle_mesh(str(path(item['mesh']))))
        start = perf_counter()
        actual = fit_pair(cad, xyz[foreground], plane, K, depth, mask, clearances=(0.,))
        batched_seconds = perf_counter()-start
        a, b = actual['arms'][0], reference['geometry']['arms'][0]
        assert a['hypothesis_count'] == b['hypothesis_count'] and a['available_projections'] == b['available_projections']
        score_errors, pose_errors = [], []
        stop_matches = 0
        for current, previous in zip(a['pose_bank'], b['pose_bank']):
            assert current['support_index'] == previous['support_index'] and current['initial_yaw_deg'] == previous['initial_yaw_deg']
            current_score, previous_score = current['verification']['geometry_score'], previous['verification']['geometry_score']
            assert (current_score is None) == (previous_score is None)
            if current_score is not None:
                score_errors.append(abs(current_score-previous_score))
            pose_errors.append(float(np.max(np.abs(np.asarray(current['T_camera_from_cad'])-previous['T_camera_from_cad']))))
            stop_matches += current['fitting']['stop_reason'] == previous['fitting']['stop_reason'] and current['fitting']['iterations'] == previous['fitting']['iterations']
        max_error = max(score_errors, default=0.)
        assert max_error < TOLERANCE, (item['target'], max_error)
        fresh_native_seconds = None
        if item['target'] in ('Gear_Large', 'KET8_Square_8mm'):
            start = perf_counter()
            fresh = native.fit_pair(cad, xyz[foreground], plane, K, depth, mask, clearances=(0.,))
            fresh_native_seconds = perf_counter()-start
            assert abs(fresh['arms'][0]['best_geometry']['verification']['geometry_score']-a['best_geometry']['verification']['geometry_score']) < TOLERANCE
        row = {'target': item['target'], 'proposal_id': pid, 'reference_fit_sha256': digest(source),
               'hypotheses': a['hypothesis_count'], 'maximum_geometry_score_difference': max_error,
               'maximum_pose_element_difference': max(pose_errors), 'native_stop_and_iteration_matches': stop_matches,
               'best_score_difference': abs(a['best_geometry']['verification']['geometry_score']-b['best_geometry']['verification']['geometry_score']),
               'batched_seconds': batched_seconds, 'fresh_native_seconds': fresh_native_seconds}
        records.append(row)
        save(HERE/'geometry/batch_parity_progress.json', {'records': records, 'labels_read': False})
        print(item['target'], row, flush=True)
    save(HERE/'geometry/batch_parity_completion.json', {'passed': True, 'labels_read': False, 'CAD_candidate_pairs': len(records),
         'records': records, 'maximum_geometry_score_difference': max(row['maximum_geometry_score_difference'] for row in records),
         'hypotheses_verified': sum(row['hypotheses'] for row in records), 'batched_source_sha256': digest(Path(__file__).with_name('batched.py'))})


if __name__ == '__main__':
    main()
