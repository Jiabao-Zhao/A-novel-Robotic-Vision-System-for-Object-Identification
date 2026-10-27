"""Read-only replay from frozen inputs/features, without importing trial scorers.

Only this verifier's own report is written. Observed identities are never loaded.
Run after completion.json has frozen every score and selected proposal ID.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = ('cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_noadapter', 'nids_ffa_only')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(name):
    name = str(name).replace('\\', '/')
    if len(name) > 1 and name[1] == ':' and sys.platform != 'win32':
        name = '/mnt/' + name[0].lower() + name[2:]
    path = Path(name)
    return path if path.is_absolute() else REPO / path


def verify():
    torch.set_num_threads(1)
    complete = read(HERE / 'completion.json')
    protocol = read(HERE / 'protocol.json')
    adjustment = read(HERE / 'execution_adjustment.json')
    assert adjustment['adapter_sha256'] == digest(HERE / 'run_cached.py')
    assert adjustment['features_sha256'] == digest(HERE / 'features.npz')
    assert adjustment['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert adjustment['scoring_rules_changed'] is False
    assert adjustment['feature_values_or_dtypes_changed'] is False
    assert adjustment['stopped_before_any_scores_or_selections_written'] is True
    for name, expected in protocol['source_sha256'].items():
        assert digest(local(name)) == expected, name
    for filename, field in [('candidate_scores.json', 'scores_sha256'),
                            ('selections.json', 'selections_sha256'),
                            ('features.npz', 'features_sha256'),
                            ('protocol.json', 'protocol_sha256')]:
        assert digest(HERE / filename) == complete[field], filename
    records = read(HERE / 'input_records.json')
    inputs = read(HERE / 'inputs.json')
    pools = read(HERE / 'template_pool.json')['eligible_record_indices']
    source = np.load(HERE / 'preprocessed.npz')
    features = np.load(HERE / 'features.npz')
    np.testing.assert_array_equal(features['encoded_record_indices'], np.arange(336))
    cls = torch.from_numpy(features['cls'])
    raw = torch.from_numpy(features['patches'])
    masks = source['masks']
    assert cls.shape == (336, 1024) and raw.shape == (336, 256, 1024)
    hard, soft, pooled, sam_patches, nids_patches = [], [], [], [], []
    for index, mask in enumerate(masks):
        occupancy = mask.reshape(16, 14, 16, 14).mean(axis=(1, 3)).ravel()
        hard.append(torch.from_numpy(occupancy > .5))
        weights = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                                size=(16, 16), mode='bilinear', align_corners=False)[0, 0]
        soft.append(weights.reshape(-1))
        if weights.sum() > 0:
            pooled.append((raw[index].reshape(16, 16, 1024) * weights[..., None]).sum((0, 1)) /
                          weights.sum())
        else:
            pooled.append(None)
        sam_patches.append(F.normalize(raw[index] * hard[-1][:, None], dim=-1))
        nids_patches.append(F.normalize(raw[index] * soft[-1][:, None], dim=-1))
    for target in protocol['targets']:
        independent = [i for i, row in enumerate(records) if row['kind'] == 'cad' and
                       row['target'] == target and hard[i].any() and soft[i].sum() > 0]
        assert independent == pools[target], target
    observations = {}
    for state, scene in inputs['states'].items():
        z = np.load(local(scene['depth_file']))
        intrinsic = np.load(local(scene['intrinsics_file']))
        plane = np.asarray(scene['plane_model_camera'])
        for crop in scene['crops']:
            a, b, c, d = [crop['roi'][k] for k in ('x1', 'y1', 'x2', 'y2')]
            yy, xx = np.mgrid[b:d+1, a:c+1]
            sub = z[b:d+1, a:c+1]
            xyz = np.stack([(xx-intrinsic[0, 2])*sub/intrinsic[0, 0],
                            (yy-intrinsic[1, 2])*sub/intrinsic[1, 1], sub], axis=-1)
            valid = np.isfinite(sub) & (sub > 0) & (
                np.abs(xyz @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3]) > .002)
            fy, fx = np.nonzero(valid)
            assert len(fy)
            bounding = [int(fx.min()+a), int(fy.min()+b), int(fx.max()+a), int(fy.max()+b)]
            # A separate float64 mean checks the physical translation, without
            # duplicating the scorer's float32 full-image reduction implementation.
            center = xyz[valid].astype(np.float64).mean(axis=0)
            observations[(state, crop['proposal_id'])] = {
                'bbox': bounding, 'center': center, 'pixels': int(valid.sum()),
                'K': intrinsic.astype(np.float32), 'shape': z.shape}
    catalog = read(HERE.parent / 'nist12_qwen_clip_reranking_20261023/data/cad_inputs.json')
    samples = np.load(HERE / 'cad_surface_samples.npz')
    rows = read(HERE / 'candidate_scores.json')
    selections = read(HERE / 'selections.json')
    shortlists = {(r['state'], r['target']): r['candidate_ids'] for r in
                  read(HERE.parent/'nist12_qwen_clip_reranking_20261023/clip/shortlist.json')
                  if r['method'] == 'clip_full'}
    assert len(shortlists) == 96
    assert len(rows) == 1152 and len(selections) == 960
    max_error = {'semantic': 0., 'appearance': 0., 'coverage': 0., 'nids_ffa': 0.,
                 'nids_appearance': 0., 'translation_m': 0., 'score': 0.}
    rebuilt = {}
    for number, row in enumerate(rows):
        state, target, proposal = row['state'], row['target'], row['proposal_id']
        index, reference = row['record_index'], pools[target]
        assert records[index]['kind'] == 'crop'
        assert records[index]['state'] == state and records[index]['proposal_id'] == proposal
        query_cls = F.normalize(cls[index], dim=-1)
        ref_cls = F.normalize(cls[reference], dim=-1)
        cosine = F.cosine_similarity(query_cls[None], ref_cls, dim=-1)
        clipped = cosine.clamp(0, 1)
        semantic = float(torch.topk(clipped, 5).values.mean())
        best = reference[int(clipped.argmax())]
        np.testing.assert_allclose(cosine.numpy(), row['semantic']['cls_view_cosines'], atol=2e-7, rtol=0)
        assert records[best]['view_index'] == row['sam6d_best_CAD_view']
        similarities = sam_patches[index] @ sam_patches[best].T
        best_forward = similarities.max(dim=1).values
        appearance = float((best_forward.sum() /
                            (torch.count_nonzero(sam_patches[index].sum(-1)) + 1e-6)).clamp(0, 1))
        best_reverse = similarities.max(dim=0).values
        coverage = float(torch.count_nonzero(best_reverse * (best_reverse > .5)) /
                         (torch.count_nonzero(best_reverse) + 1e-6))
        np.testing.assert_allclose(best_forward.numpy(), row['appearance']['observed_patch_maxima'], atol=2e-7, rtol=0)
        assert pooled[index] is not None and all(pooled[j] is not None for j in reference)
        qffa = F.normalize(pooled[index], dim=-1)
        rffa = F.normalize(torch.stack([pooled[j] for j in reference]), dim=-1)
        ffa_views = F.cosine_similarity(qffa[None], rffa, dim=-1).clamp(0, 1)
        nids_ffa = float(ffa_views.max())
        nids_best = reference[int(ffa_views.argmax())]
        assert int(ffa_views.argmax()) == row['nids']['best_view_index']
        np.testing.assert_allclose(ffa_views.numpy(), row['nids']['per_view_score'], atol=2e-7, rtol=0)
        nids_sim = nids_patches[index] @ sam_patches[nids_best].T
        nids_count = int(torch.count_nonzero(nids_patches[index].sum(-1)))
        nids_appearance = float((nids_sim.max(dim=1).values.sum() / (nids_count + 1e-6)).clamp(0, 1))
        assert nids_count == row['nids_appearance']['query_nonzero_channel_sum_count']
        observed = observations[(state, proposal)]
        geometry = row['geometry']
        assert geometry['status'] == 'available'
        assert geometry['observed_mask_pixels'] == geometry['observed_valid_depth_pixels'] == observed['pixels']
        assert geometry['observed_bbox_xyxy'] == observed['bbox']
        translation = np.asarray(geometry['translation_camera_m'], dtype=np.float32)
        translation_error = float(np.max(np.abs(translation-observed['center'])))
        assert translation_error < 1e-6, translation_error
        max_error['translation_m'] = max(max_error['translation_m'], translation_error)
        view = next(v for v in catalog[target]['views'] if v['view_index'] == records[best]['view_index'])
        backward = np.asarray(view['camera_direction_cad'], dtype=np.float64)
        backward /= np.linalg.norm(backward)
        hint = [0., 1., 0.] if abs(backward[2]) > .99 else [0., 0., 1.]
        right = np.cross(hint, backward)
        right /= np.linalg.norm(right)
        rotation = np.stack([right, -np.cross(backward, right), -backward]).astype(np.float32)
        np.testing.assert_allclose(rotation, inputs['cads'][target]['rotations'][str(view['view_index'])], atol=1e-7)
        transformed = (rotation @ samples[target].T).T + translation
        assert transformed.shape == (2048, 3) and (transformed[:, 2] > 0).all()
        projected = (observed['K'] @ transformed.T).T
        uv = (projected[:, :2]/projected[:, 2:]).astype(np.int32)
        uv[:, 0] = np.clip(uv[:, 0], 0, observed['shape'][1]-1)
        uv[:, 1] = np.clip(uv[:, 1], 0, observed['shape'][0]-1)
        box = np.r_[uv.min(0), uv.max(0)].tolist()
        assert box == geometry['predicted_bbox_xyxy']
        box = np.asarray(box, np.float32)
        other = np.asarray(observed['bbox'], np.float32)
        wh = np.minimum(box[2:], other[2:])-np.maximum(box[:2], other[:2])
        inter = np.prod(wh) if np.all(wh > 0) else np.float32(0)
        union = np.prod(box[2:]-box[:2])+np.prod(other[2:]-other[:2])-inter
        iou = float(inter/union) if union > 0 else 0.
        assert iou == geometry['geometry_bbox_iou']
        expected = {'cnos': semantic, 'sam6d_rgb': (semantic+appearance)/2,
                    'sam6d_rgbd': (semantic+appearance+coverage*iou)/(2+coverage),
                    'nids_noadapter': (nids_ffa+nids_appearance)/2, 'nids_ffa_only': nids_ffa}
        for name, value in [('semantic', semantic), ('appearance', appearance), ('coverage', coverage),
                            ('nids_ffa', nids_ffa), ('nids_appearance', nids_appearance)]:
            actual = {'semantic': row['semantic']['semantic_score'],
                      'appearance': row['appearance']['appearance_score'],
                      'coverage': row['appearance']['visible_ratio_diagnostic'],
                      'nids_ffa': row['nids']['score'],
                      'nids_appearance': row['nids_appearance']['score']}[name]
            error = abs(value-actual)
            assert error < 2e-7, (state, target, proposal, name, error)
            max_error[name] = max(max_error[name], error)
        for method, value in expected.items():
            error = abs(value-row['scores'][method])
            assert error < 2e-7, (method, error)
            max_error['score'] = max(max_error['score'], error)
            rebuilt.setdefault((method, state, target), []).append((proposal, value))
        if number % 144 == 143:
            print(f'Independently verified {number+1}/1152 candidate pairs', flush=True)
    for result in selections:
        method = result['method']
        base_method = method.removeprefix('clip5_')
        candidates = rebuilt[(base_method, result['state'], result['target'])]
        expected_count = 12
        if method.startswith('clip5_'):
            shortlist = shortlists[(result['state'], result['target'])]
            assert len(shortlist) == 5 and len(set(shortlist)) == 5
            candidates = [r for r in candidates if r[0] in shortlist]
            expected_count = 5
        assert len(candidates) == expected_count
        assert result['candidate_ids'] == [p for p, _ in candidates]
        ordered = sorted(candidates, key=lambda p: (-p[1], p[0]))
        tied = [p for p, score in ordered if abs(score-ordered[0][1]) <= 1e-12]
        assert result['ranked_ids'] == [p for p, _ in ordered]
        assert result['selected_ids'] == tied
        assert result['status'] == ('selected' if len(tied) == 1 else 'tied')
        np.testing.assert_allclose([result['top_score'], result['gap']],
                                   [ordered[0][1], ordered[0][1]-ordered[1][1]], atol=2e-7, rtol=0)
    for name, expected in protocol['source_sha256'].items():
        assert digest(local(name)) == expected, name
    report = dict(passed=True, source_hashes_checked=len(protocol['source_sha256']),
                  images=336, candidate_pairs=1152, method_scores=5760, selections=960,
                  all_pair_arithmetic_and_rankings_replayed=True,
                  all_projected_boxes_and_IoU_exact=True,
                  foreground_masks_rebuilt_from_depth=True, native_template_rotations_rebuilt=True,
                  primary_method_candidates=12, matched_shortlist_controls=5,
                  frozen_CLIP_shortlists_verified=96, labels_or_simulator_poses_read=False,
                  imported_trial_scorers=False, maximum_absolute_errors=max_error,
                  verifier_sha256=digest(__file__), protocol_sha256=digest(HERE/'protocol.json'),
                  features_sha256=digest(HERE/'features.npz'),
                  execution_adapter_sha256=digest(HERE/'run_cached.py'),
                  execution_adjustment_sha256=digest(HERE/'execution_adjustment.json'),
                  NPZ_cache_changes_values_dtypes_or_scoring=False,
                  scores_sha256=digest(HERE/'candidate_scores.json'),
                  selections_sha256=digest(HERE/'selections.json'))
    target = HERE/'independent_verification.json'
    if target.exists():
        assert read(target) == report, 'Existing independent verification evidence differs.'
    else:
        target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    verify()
