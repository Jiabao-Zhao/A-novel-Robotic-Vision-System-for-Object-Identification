"""Executable plumbing diagnostic; descriptors are synthetic, never benchmarked."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .data import write_json
from .pipeline import local_path, config_hash, source_hashes
from .registration import feature_ransac, refine_icp, score_pose, joint_score, rank_candidates
from .evaluation import PosePrediction, write_bop_csv, validate_bop_csv
from .features import fuse_features


def run_synthetic(config):
    output = local_path(config['paths']['output_root']) / 'synthetic_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_manifest.json').exists():
        raise ValueError('Preserve previous synthetic diagnostic; choose separate output_root')
    rng = np.random.default_rng(config['seed'])
    cad = rng.uniform(-.035, .035, (200, 3))
    v, g = rng.normal(size=(200, 64)), rng.normal(size=(200, 64))
    cad_features = fuse_features(v, g)
    R = Rotation.from_euler('xyz', [117, -73, 158], degrees=True).as_matrix()
    t = np.array([.05, -.03, .65])
    indices = rng.choice(len(cad), 80, replace=False)
    sparse = np.concatenate((cad[indices] @ R.T + t, rng.uniform([-.1,-.1,.5],[.1,.1,.8],(20,3))))
    features = np.concatenate((cad_features[indices], fuse_features(rng.normal(size=(20,64)), rng.normal(size=(20,64)))))
    dense = cad[indices] @ R.T + t
    diameter = float(np.linalg.norm(np.ptp(cad, axis=0)))
    start = time.perf_counter()
    coarse = feature_ransac(cad, sparse, cad_features, features, diameter, config['registration'])
    if coarse.status != 'ok':
        raise RuntimeError(f'Synthetic large-rotation registration failed: {coarse.status}')
    refined = refine_icp(cad, dense, coarse.R, coarse.t, diameter, config['registration'])
    scores = score_pose(cad, sparse, refined.R, refined.t, diameter, coarse.correspondences, dense_xyz=dense, config=config['registration'])
    total = joint_score(coarse.score, scores.feature_score, scores.geometry_score)
    candidates = [dict(candidate_id='synthetic_correct', obj_id=1, R=refined.R,t=refined.t,score=total,status='ok'),
                  dict(candidate_id='synthetic_duplicate',obj_id=1,R=refined.R,t=refined.t+[.00001,0,0],score=total*.9,status='ok'),
                  dict(candidate_id='synthetic_unavailable',obj_id=1,R=None,t=None,score=None,status='no_valid_hypothesis')]
    ranking = rank_candidates(candidates, translation_nms_threshold_m=diameter*.1,max_instances=1)
    elapsed = time.perf_counter() - start
    csv = output / 'synthetic_tless-test.csv'
    write_bop_csv([PosePrediction(0,0,1,total,refined.R,refined.t,elapsed)],csv)
    official = validate_bop_csv(csv,local_path(config['paths']['bop_toolkit']),extra_python_paths=[local_path(config['paths']['bop_eval_deps'])])
    angle = float(Rotation.from_matrix(refined.R @ R.T).magnitude())
    translation_error = float(np.linalg.norm(refined.t-t))
    coarse_angle = float(Rotation.from_matrix(coarse.R @ R.T).magnitude())
    coarse_translation_error = float(np.linalg.norm(coarse.t-t))
    # One-sided partial-overlap ICP can move an exact coarse pose: record both.
    if coarse_angle > 1e-4 or coarse_translation_error > 1e-5 or angle > np.deg2rad(.5) or translation_error > .001:
        raise RuntimeError('Synthetic coarse recovery or bounded partial-overlap ICP tolerance failed')
    np.savez_compressed(output/'trace.npz',cad_m=cad,observed_sparse_m=sparse,dense_m=dense,coarse_R=coarse.R,coarse_t_m=coarse.t,
                        refined_R=refined.R,refined_t_m=refined.t,known_R=R,known_t_m=t)
    result={'status':'passed','scope':'synthetic_plumbing_only','trained_DINOv2_or_GeDi':False,
            'claim':'Not real BOP inference, mask accuracy, or published-score reproduction.',
            'config_sha256':config_hash(config),'source_sha256':source_hashes(),'seed':config['seed'],
            'cad_points':200,'observed_points':100,'observed_outliers':20,'CAD_overlap_fraction':.4,
            'large_rotation_euler_degrees':[117,-73,158], 'rotation_error_rad':angle,'translation_error_m':translation_error,
            'coarse_rotation_error_rad':coarse_angle,'coarse_translation_error_m':coarse_translation_error,
            'ICP_observation':'One-sided ICP shifts the exact coarse pose slightly under partial overlap; both errors retained, no tuning on BOP GT.',
            'coarse_feature_score':coarse.score,'refined_scores':asdict(scores),'joint_score':total,
            'selected':[p['candidate_id'] for p in ranking.selected],'suppressed':ranking.suppressed,'rejected':ranking.rejected,
            'runtime_s':elapsed,'official_csv_check':official,'registration':coarse.diagnostics,'icp_status':refined.status}
    write_json(output/'run_manifest.json',result)
    return result
