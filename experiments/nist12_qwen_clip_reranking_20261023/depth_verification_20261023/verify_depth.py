"""Independent full-union ray replay of deterministic frozen depth-fit samples."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import json
import hashlib

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
from run_trial import read, local, digest, check_sources, write_once


def masks_and_depth(inputs, row):
    capture = inputs['states'][row['state']]
    depth = np.load(local(capture['depth_file']))
    K = np.load(local(capture['intrinsics_file']))
    box = capture['crops'][str(row['proposal_id'])]['roi']
    x1, y1, x2, y2 = [box[k] for k in ('x1', 'y1', 'x2', 'y2')]
    yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
    z = depth[y1:y2+1, x1:x2+1]
    xyz = np.stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z), axis=-1)
    plane = np.asarray(capture['plane_model_camera'])
    fg = np.isfinite(z) & (z > 0) & (np.abs(xyz @ plane[:3]+plane[3])/np.linalg.norm(plane[:3]) > .002)
    mask = np.zeros(depth.shape, bool)
    mask[y1:y2+1, x1:x2+1] = fg
    return depth, K, mask


def replay(scene, vertices, pose, K, depth, mask):
    rotation, translation = pose[:3, :3], pose[:3, 3]
    camera_vertices = vertices @ rotation.T + translation
    assert np.all(camera_vertices[:, 2] > 0)
    uv = camera_vertices[:, :2] / camera_vertices[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]
    yy, xx = np.nonzero(mask)
    x1 = max(0, int(np.floor(min(uv[:, 0].min(), xx.min()))) - 2)
    y1 = max(0, int(np.floor(min(uv[:, 1].min(), yy.min()))) - 2)
    x2 = min(depth.shape[1], int(np.ceil(max(uv[:, 0].max(), xx.max()))) + 3)
    y2 = min(depth.shape[0], int(np.ceil(max(uv[:, 1].max(), yy.max()))) + 3)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    directions = np.stack(((xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones(xx.shape)), axis=-1)
    cad_directions = directions @ rotation
    cad_origins = np.broadcast_to(-translation @ rotation, cad_directions.shape)
    rays = np.concatenate((cad_origins, cad_directions), axis=-1).astype(np.float32)
    rendered = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
    observed = mask[y1:y2, x1:x2]
    measured = depth[y1:y2, x1:x2]
    cad = np.isfinite(rendered) & (rendered > 0)
    union, overlap = observed | cad, observed & cad
    valid = overlap & np.isfinite(measured) & (measured > 0)
    counts = {'observed_pixels': int(observed.sum()), 'projected_pixels': int(cad.sum()),
              'union_pixels': int(union.sum()), 'intersection_pixels': int(overlap.sum()),
              'valid_depth_overlap_pixels': int(valid.sum()),
              'observed_only_pixels': int((observed & ~cad).sum()),
              'cad_only_pixels': int((cad & ~observed).sum())}
    if not valid.any():
        return dict(counts, geometry_score=None)
    residual = np.abs(measured[valid] - rendered[valid])
    p_obs = counts['observed_only_pixels']/counts['union_pixels']
    p_cad = counts['cad_only_pixels']/counts['union_pixels']
    p_depth = float(np.minimum(residual/.005, 1).mean())
    return dict(counts, p_obs=p_obs, p_cad=p_cad, p_depth=p_depth,
                mask_iou=counts['intersection_pixels']/counts['union_pixels'],
                geometry_score=1-(p_obs+p_cad+p_depth)/3,
                depth_mae_m=float(residual.mean()))


def main():
    check_sources()
    complete = read(HERE/'selection_completion.json')
    assert complete['selections_sha256'] == digest(HERE/'selections.json')
    inputs = read(HERE/'inputs.json')
    fits = sorted((HERE/'fits').glob('*.json'))
    assert len(fits) == 480
    chosen = np.linspace(0, len(fits)-1, 16).astype(int)
    models, checks = {}, []
    for index in chosen:
        path = fits[index]
        row = read(path)
        target = row['target']
        if target not in models:
            mesh = o3d.io.read_triangle_mesh(str(local(inputs['cads'][target]['mesh_file'])))
            scene = o3d.t.geometry.RaycastingScene(nthreads=1)
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            models[target] = (scene, np.asarray(mesh.vertices))
        depth, K, mask = masks_and_depth(inputs, row)
        for arm in row['geometry']['arms']:
            indices = {0, len(arm['pose_bank'])//2, arm['best_rmse_index']}
            if arm['best_geometry_index'] is not None:
                indices.add(arm['best_geometry_index'])
            for pose_index in sorted(indices):
                saved = arm['pose_bank'][pose_index]
                expected = saved['verification']
                if 'observed_pixels' not in expected:
                    assert expected['geometry_score'] is None
                    checks.append({'fit': path.name, 'arm': arm['arm_name'], 'pose_index': pose_index,
                                   'structurally_unavailable': expected['status']})
                    continue
                actual = replay(*models[target], np.asarray(saved['T_camera_from_cad']), K, depth, mask)
                count_errors = {k: actual[k]-expected[k] for k in (
                    'observed_pixels', 'projected_pixels', 'union_pixels', 'intersection_pixels',
                    'valid_depth_overlap_pixels', 'observed_only_pixels', 'cad_only_pixels')}
                # At most one float32 silhouette-boundary ray may change between
                # algebraically equivalent camera/CAD transformations.
                assert max(map(abs, count_errors.values())) <= 1, (path, pose_index, count_errors)
                assert (actual['geometry_score'] is None) == (expected['geometry_score'] is None)
                differences = {}
                if actual['geometry_score'] is not None:
                    for key in ('p_obs', 'p_cad', 'p_depth', 'geometry_score', 'mask_iou', 'depth_mae_m'):
                        differences[key] = abs(actual[key]-expected[key])
                    assert differences['depth_mae_m'] < 2e-6
                    assert max(differences[k] for k in differences if k != 'depth_mae_m') < .002
                checks.append({'fit': path.name, 'arm': arm['arm_name'], 'pose_index': pose_index,
                               'count_errors': count_errors, 'absolute_score_differences': differences})
        print(json.dumps({'replayed_fits': len({r['fit'] for r in checks}), 'poses': len(checks)}), flush=True)
    write_once(HERE/'independent_ray_verification.json', {
        'status': 'passed', 'fit_files': len(chosen), 'poses': len(checks),
        'full_union_replay': True, 'uses_geometry_score_implementation': False,
        'uses_observation_labels_or_poses': False,
        'source_sha256': digest(__file__), 'checks': checks,
    })


if __name__ == '__main__':
    main()
