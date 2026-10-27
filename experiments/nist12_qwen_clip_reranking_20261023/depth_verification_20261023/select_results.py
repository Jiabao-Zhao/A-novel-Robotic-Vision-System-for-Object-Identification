"""Freeze every depth-based crop selection before reading evaluation identities."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_trial import read, write_once, digest, pair_key, check_sources, CONFIG


def candidate_score(pair, fitted, clearance, mode):
    raw = {'proposal_id': pair['proposal_id'], 'status': 'available', 'score': None,
           'fit_file': 'fits/' + pair_key(pair) + '.json', 'arm': clearance}
    arms = [a for a in fitted['geometry'].get('arms', []) if a['arm_name'] == clearance]
    if not arms:
        return dict(raw, status=fitted['geometry']['status'])
    arm, = arms
    if mode == 'rmse':
        pose = arm['best_rmse']
        raw.update(score=pose['fitting']['rmse_m'], pose_index=arm['best_rmse_index'])
    else:
        pose = arm['best_geometry']
        if pose is None:
            return dict(raw, status=arm['status'])
        verification = pose['verification']
        geometry = verification['geometry_score']
        if geometry is None:
            return dict(raw, status=verification['status'])
        if mode == 'geometry':
            score = geometry
        elif mode == 'fusion':
            r = pair['coverage']
            score = (pair['global_score'] + pair['patch_score'] + r * geometry) / (2 + r)
        else:
            raise ValueError(mode)
        raw.update(score=float(score), pose_index=arm['best_geometry_index'])
    verification = pose['verification']
    raw.update(geometry_score=verification.get('geometry_score'),
               mask_iou=verification.get('mask_iou'), p_depth=verification.get('p_depth'),
               depth_mae_m=verification.get('depth_mae_m'),
               fitting_rmse_m=pose['fitting']['rmse_m'],
               global_score=pair['global_score'], patch_score=pair['patch_score'],
               coverage=pair['coverage'], dino_score=pair['dino_score'])
    assert raw['score'] is not None and np.isfinite(raw['score'])
    return raw


def select(task, method, candidates, minimize=False):
    row = {'state': task['state'], 'target': task['target'], 'method': method,
           'candidate_ids': task['candidate_ids'], 'candidate_scores': candidates,
           'selected_ids': [], 'status': None, 'top_score': None, 'gap': None}
    if len(candidates) != 5 or any(c['status'] != 'available' for c in candidates):
        return dict(row, status='unavailable_incomplete_shortlist')
    ranked = sorted(candidates, key=lambda c: c['score'], reverse=not minimize)
    best = ranked[0]['score']
    tied = [c['proposal_id'] for c in ranked if abs(c['score']-best) <= CONFIG['tie_atol']]
    row.update(selected_ids=tied, top_score=best, gap=abs(best-ranked[1]['score']),
               status='selected' if len(tied) == 1 else 'tied',
               ranked_ids=[c['proposal_id'] for c in ranked])
    return row


def main():
    check_sources()
    complete = read(HERE / 'fitting_completion.json')
    assert complete['candidate_pairs'] == 480 and not complete['evaluation_labels_read']
    assert complete['protocol_sha256'] == digest(HERE / 'protocol.json')
    for name, expected in complete['fit_sha256'].items():
        assert digest(HERE / 'fits' / name) == expected, name
    inputs = read(HERE / 'inputs.json')
    pairs = {(p['state'], p['target'], p['proposal_id']): p for p in inputs['pairs']}
    selections = []
    for task in inputs['tasks']:
        members = [pairs[task['state'], task['target'], pid] for pid in task['candidate_ids']]
        baseline = [{'proposal_id': p['proposal_id'], 'score': p['dino_score'], 'status': 'available'}
                    for p in members]
        selections.append(select(task, 'clip_dino_baseline', baseline))
        fitted = {p['proposal_id']: read(HERE / 'fits' / (pair_key(p)+'.json')) for p in members}
        for clearance in ('contact_0mm', 'clearance_1mm'):
            for mode in ('geometry', 'fusion', 'rmse'):
                scores = [candidate_score(p, fitted[p['proposal_id']], clearance, mode) for p in members]
                selections.append(select(task, clearance+'_'+mode, scores, minimize=mode == 'rmse'))
    assert len(selections) == 96*7
    write_once(HERE / 'selections.json', selections)
    write_once(HERE / 'selection_completion.json', {
        'tasks': 96, 'methods': 7, 'selections': len(selections),
        'selections_sha256': digest(HERE / 'selections.json'),
        'protocol_sha256': digest(HERE / 'protocol.json'),
        'evaluation_labels_read': False, 'all_fitting_hashes_verified': True,
    })
    print(json.dumps({'selections_frozen': len(selections), 'evaluation_labels_read': False}), flush=True)


if __name__ == '__main__':
    main()
