"""Post-selection audit; labels are inaccessible until selections verify.

No model inference, registration, or selection occurs here. The existing audit
identities enter only after every frozen, label-free selection hash is checked.
"""
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]
OUT = HERE / 'analysis'
RAW_FIT_AUDIT = {}
METHODS = ('clip_dino_baseline', 'contact_0mm_fusion', 'contact_0mm_geometry',
           'contact_0mm_rmse', 'clearance_1mm_fusion', 'clearance_1mm_geometry',
           'clearance_1mm_rmse')
LABELS = {
    'clip_dino_baseline': 'CLIP top 5 + DINO',
    'contact_0mm_fusion': 'Table contact: DINO + depth',
    'contact_0mm_geometry': 'Table contact: geometry only',
    'contact_0mm_rmse': 'Table contact: ICP RMSE only',
    'clearance_1mm_fusion': '1 mm clearance: DINO + depth',
    'clearance_1mm_geometry': '1 mm clearance: geometry only',
    'clearance_1mm_rmse': '1 mm clearance: ICP RMSE only',
}


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


def write_once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Refusing to change saved evidence: {path}'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':'))
                             if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def family(name):
    return ('round_pin' if name.startswith('RGOCG') else
            'rectangular_pin' if name.startswith('KET') else name)


def pretty(name):
    if name.startswith('RGOCG'):
        return 'Round ' + name.rsplit('_', 1)[-1]
    if name.startswith('KET'):
        return 'Rectangular ' + name.rsplit('_', 1)[-1]
    return name.replace('_Male', '').replace('_', ' ')


def native_vertices(path):
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    assert len(raw) == 84 + 50 * count, 'Expected the audited binary metric STL.'
    record = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.unique(np.frombuffer(raw, dtype=record, count=count, offset=84)
                     ['vertices'].reshape(-1, 3).astype(float), axis=0)


def verify_raw_fit_arithmetic():
    """Stream frozen poses; replay masks, scores, table height and rigid scale."""
    inputs = read(HERE / 'inputs.json')
    mesh_vertices = {target: native_vertices(local(item['mesh_file']))
                     for target, item in inputs['cads'].items()}
    support_minima, expected, pose_count, available_count = {}, {}, 0, 0
    for pair in inputs['pairs']:
        name = f"{pair['state']}__{pair['target']}__p{pair['proposal_id']:03d}.json"
        fitted = read(HERE / 'fits' / name)
        assert (fitted['state'], fitted['target'], fitted['proposal_id']) == (pair['state'], pair['target'], pair['proposal_id'])
        plane = np.asarray(inputs['states'][pair['state']]['plane_model_camera'], float)
        normal, distance = plane[:3] / np.linalg.norm(plane[:3]), plane[3] / np.linalg.norm(plane[:3])
        # The observing camera is above these audited table planes; no world pose.
        if distance < 0:
            normal, distance = -normal, -distance
        for arm in fitted['geometry']['arms']:
            bank = arm['pose_bank']
            scores, errors, available = [], [], []
            for index, pose in enumerate(bank):
                pose_count += 1
                T = np.asarray(pose['T_camera_from_cad'], float)
                assert T.shape == (4, 4) and np.isfinite(T).all()
                assert np.max(np.abs(T[3] - [0, 0, 0, 1])) <= 1e-12
                R = T[:3, :3]
                assert np.max(np.abs(R.T @ R - np.eye(3))) <= 1e-9
                assert abs(np.linalg.det(R) - 1) <= 1e-9
                local_normal = R.T @ normal
                cache_key = pair['target'], tuple(np.round(local_normal, 10))
                if cache_key not in support_minima:
                    support_minima[cache_key] = float(np.min(mesh_vertices[pair['target']] @ local_normal))
                height = support_minima[cache_key] + T[:3, 3] @ normal + distance
                assert abs(height - arm['clearance_m']) <= 1e-8
                assert abs(height - pose['minimum_table_height_m']) <= 1e-8
                history = pose['fitting']['rmse_history_m']
                assert len(history) and np.isfinite(history).all()
                assert all(b <= a + 1e-12 for a, b in zip(history, history[1:]))
                errors.append(pose['fitting']['rmse_m'])
                verification = pose['verification']
                score = verification['geometry_score']
                scores.append(score)
                if score is None:
                    continue
                available_count += 1
                available.append(index)
                union = verification['union_pixels']
                assert union > 0 and verification['valid_depth_overlap_pixels'] > 0
                assert verification['intersection_pixels'] + verification['observed_only_pixels'] + verification['cad_only_pixels'] == union
                assert abs(verification['p_obs'] - verification['observed_only_pixels'] / union) <= 1e-12
                assert abs(verification['p_cad'] - verification['cad_only_pixels'] / union) <= 1e-12
                assert abs(verification['mask_iou'] - verification['intersection_pixels'] / union) <= 1e-12
                assert 0 <= verification['p_depth'] <= 1
                recomputed = 1 - (verification['p_obs'] + verification['p_cad'] + verification['p_depth']) / 3
                assert abs(score - recomputed) <= 1e-12
            geometry_index = max(available, key=lambda i: scores[i]) if available else None
            rmse_index = min(range(len(bank)), key=lambda i: errors[i])
            assert geometry_index == arm['best_geometry_index']
            assert rmse_index == arm['best_rmse_index']
            assert arm['best_geometry'] == (bank[geometry_index] if geometry_index is not None else None)
            assert arm['best_rmse'] == bank[rmse_index]
            for mode in ('geometry', 'fusion', 'rmse'):
                index = rmse_index if mode == 'rmse' else geometry_index
                key = pair['state'], pair['target'], pair['proposal_id'], arm['arm_name'] + '_' + mode
                if index is None:
                    expected[key] = None
                    continue
                selected = bank[index]
                G = selected['verification']['geometry_score']
                score = errors[index] if mode == 'rmse' else G
                if mode == 'fusion':
                    r = pair['coverage']
                    score = (pair['global_score'] + pair['patch_score'] + r * G) / (2 + r)
                expected[key] = dict(score=score, pose_index=index, geometry_score=G,
                    fitting_rmse_m=errors[index],
                    mask_iou=selected['verification'].get('mask_iou'),
                    p_depth=selected['verification'].get('p_depth'),
                    depth_mae_m=selected['verification'].get('depth_mae_m'))
    return expected, dict(candidate_pairs=len(inputs['pairs']), pose_records=pose_count,
        available_projection_records=available_count, native_scale_rigid_transforms_verified=True,
        native_mesh_table_heights_verified=True, all_pose_union_arithmetic_verified=True,
        max_geometry_and_min_rmse_pose_selection_verified=True,
        full_depth_reprojection='Separate deterministic projection audit; not claimed here.',
        evaluation_labels_read=False)


def verify_before_labels(replay_raw=True):
    global RAW_FIT_AUDIT
    completion_path = HERE / 'selection_completion.json'
    if not completion_path.exists():
        raise RuntimeError('Selections are not frozen; evaluation labels will not be read.')
    completion = read(completion_path)
    assert completion['tasks'] == 96 and completion['methods'] == 7
    assert completion['selections'] == 672 and not completion['evaluation_labels_read']
    assert completion['all_fitting_hashes_verified']
    assert completion['selections_sha256'] == digest(HERE / 'selections.json')
    assert completion['protocol_sha256'] == digest(HERE / 'protocol.json')
    protocol = read(HERE / 'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert digest(local(name)) == expected, name
    fitted = read(HERE / 'fitting_completion.json')
    assert fitted['candidate_pairs'] == 480 and not fitted['evaluation_labels_read']
    assert fitted['protocol_sha256'] == completion['protocol_sha256']
    for name, expected in fitted['fit_sha256'].items():
        assert digest(HERE / 'fits' / name) == expected, name
    adjustment_path = HERE / 'execution_adjustment.json'
    if adjustment_path.exists():
        adjustment = read(adjustment_path)
        assert not adjustment['scoring_or_fitting_configuration_changed']
        assert not adjustment['evaluation_labels_read']
        assert adjustment['protocol_sha256'] == completion['protocol_sha256']
        assert adjustment['adapter_sha256'] == digest(HERE / 'run_parallel.py')
        for name, expected in adjustment['retained_completed_fit_sha256'].items():
            assert digest(HERE / 'fits' / name) == expected, name
    selections = read(HERE / 'selections.json')
    expected = None
    if replay_raw:
        expected, RAW_FIT_AUDIT = verify_raw_fit_arithmetic()
    assert len(selections) == 672
    assert set(row['method'] for row in selections) == set(METHODS)
    for method in METHODS:
        rows = [r for r in selections if r['method'] == method]
        assert len(rows) == len({(r['state'], r['target']) for r in rows}) == 96
        assert all(len(r['candidate_ids']) == len(r['candidate_scores']) == 5 for r in rows)
        assert all(set(r['selected_ids']) <= set(r['candidate_ids']) for r in rows)
        for row in rows:
            members = row['candidate_scores']
            if expected is not None and method != 'clip_dino_baseline':
                for candidate in members:
                    stored = expected[row['state'], row['target'], candidate['proposal_id'], method]
                    if stored is None:
                        assert candidate['status'] != 'available' and candidate['score'] is None
                        continue
                    assert candidate['status'] == 'available'
                    for key, value in stored.items():
                        if value is None:
                            assert candidate[key] is None
                        else:
                            assert abs(candidate[key] - value) <= 1e-12, (row['state'], row['target'], method, key)
            if any(c['status'] != 'available' for c in members):
                assert row['status'] == 'unavailable_incomplete_shortlist'
                assert row['selected_ids'] == []
                continue
            scores = [c['score'] for c in members]
            assert np.isfinite(scores).all()
            best = min(scores) if method.endswith('_rmse') else max(scores)
            tied = [c['proposal_id'] for c in members if abs(c['score'] - best) <= 1e-12]
            assert set(tied) == set(row['selected_ids'])
            assert row['status'] == ('selected' if len(tied) == 1 else 'tied')
            assert abs(best - row['top_score']) <= 1e-12
            for candidate in members:
                if method.endswith('_fusion'):
                    r, g = candidate['coverage'], candidate['geometry_score']
                    expected_fusion = (candidate['global_score'] + candidate['patch_score'] + r * g) / (2 + r)
                    assert abs(candidate['score'] - expected_fusion) <= 1e-12
                elif method.endswith('_geometry'):
                    assert abs(candidate['score'] - candidate['geometry_score']) <= 1e-12
                elif method.endswith('_rmse'):
                    assert abs(candidate['score'] - candidate['fitting_rmse_m']) <= 1e-12
    return selections, completion


def evaluate(selections):
    # This is the first evaluation-label read. No simulator poses are used.
    prior = [row for row in read(STUDY / 'analysis/decisions.json')
             if row['method'] == 'clip_full_dino']
    assert len(prior) == 96 and sum(r['outcome'] == 'correct' for r in prior) == 66
    baseline = {(r['state'], r['target']): r for r in prior}
    identities = {(r['state'], r['truth_proposal_id']): r['target'] for r in prior}
    assert len(identities) == 96
    evaluated = []
    for saved in selections:
        key = saved['state'], saved['target']
        reference = baseline[key]
        selected = saved['selected_ids']
        truth = reference['truth_proposal_id']
        if saved['status'].startswith('unavailable'):
            outcome = 'unavailable'
        elif len(selected) != 1:
            outcome = 'tied'
        else:
            outcome = 'correct' if selected[0] == truth else 'wrong'
        predicted = identities[saved['state'], selected[0]] if len(selected) == 1 else None
        same_size_error = (outcome == 'wrong' and family(saved['target'])
                           in ('round_pin', 'rectangular_pin')
                           and family(predicted) == family(saved['target']))
        row = dict(saved, outcome=outcome, truth_proposal_id=truth,
                   truth_in_top5=truth in saved['candidate_ids'],
                   predicted_product=predicted, family=family(saved['target']),
                   same_family_wrong_size=same_size_error,
                   baseline_outcome=reference['outcome'],
                   baseline_selected_ids=reference['selected_ids'],
                   baseline_predicted_product=reference['predicted_product'])
        if saved['method'] == 'clip_dino_baseline':
            assert selected == reference['selected_ids'] and outcome == reference['outcome']
            assert abs(saved['top_score'] - reference['top_score']) <= 1e-12
        evaluated.append(row)
    assert sum(r['truth_in_top5'] for r in evaluated
               if r['method'] == 'clip_dino_baseline') == 95
    return evaluated


def summarize(rows):
    summaries, objects, changes = [], [], []
    for method in METHODS:
        chosen = [r for r in rows if r['method'] == method]
        summary = dict(method=method, label=LABELS[method], tasks=len(chosen))
        summary.update({outcome: sum(r['outcome'] == outcome for r in chosen)
                        for outcome in ('correct', 'wrong', 'tied', 'unavailable')})
        summary.update(correct_percent=100 * summary['correct'] / len(chosen),
            errors_fixed=sum(r['baseline_outcome'] == 'wrong' and r['outcome'] == 'correct' for r in chosen),
            errors_introduced=sum(r['baseline_outcome'] == 'correct' and r['outcome'] == 'wrong' for r in chosen),
            correct_lost_to_tie_or_unavailable=sum(r['baseline_outcome'] == 'correct'
                and r['outcome'] in ('tied', 'unavailable') for r in chosen),
            same_family_wrong_size=sum(r['same_family_wrong_size'] for r in chosen),
            pin_correct=sum(r['outcome'] == 'correct' and r['family'] in ('round_pin', 'rectangular_pin') for r in chosen),
            pin_tasks=sum(r['family'] in ('round_pin', 'rectangular_pin') for r in chosen),
            nonpin_correct=sum(r['outcome'] == 'correct' and r['family'] not in ('round_pin', 'rectangular_pin') for r in chosen),
            fixed_cases=[f"{r['state']}:{r['target']}" for r in chosen
                         if r['baseline_outcome'] == 'wrong' and r['outcome'] == 'correct'],
            introduced_cases=[f"{r['state']}:{r['target']}" for r in chosen
                              if r['baseline_outcome'] == 'correct' and r['outcome'] == 'wrong'])
        assert sum(summary[k] for k in ('correct', 'wrong', 'tied', 'unavailable')) == 96
        assert summary['correct'] == 66 + summary['errors_fixed'] - summary['errors_introduced'] - summary['correct_lost_to_tie_or_unavailable']
        summaries.append(summary)
        for target in sorted({r['target'] for r in chosen}):
            members = [r for r in chosen if r['target'] == target]
            objects.append(dict(method=method, target=target, family=family(target), tasks=8,
                **{outcome: sum(r['outcome'] == outcome for r in members)
                   for outcome in ('correct', 'wrong', 'tied', 'unavailable')},
                fixed=sum(r['baseline_outcome'] == 'wrong' and r['outcome'] == 'correct' for r in members),
                introduced=sum(r['baseline_outcome'] == 'correct' and r['outcome'] == 'wrong' for r in members),
                same_family_wrong_size=sum(r['same_family_wrong_size'] for r in members)))
        if method != 'clip_dino_baseline':
            for row in chosen:
                if row['selected_ids'] != row['baseline_selected_ids'] or row['outcome'] != row['baseline_outcome']:
                    change = dict(row)
                    change['change'] = ('fixed' if row['baseline_outcome'] == 'wrong' and row['outcome'] == 'correct' else
                        'introduced' if row['baseline_outcome'] == 'correct' and row['outcome'] == 'wrong' else
                        'lost_to_tie_or_unavailable' if row['baseline_outcome'] == 'correct' else 'changed_but_not_fixed')
                    changes.append(change)
    return summaries, objects, changes


def plots(summaries, objects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    colors = ['#627386', '#166d79', '#3a8791', '#95acaf', '#d6a044', '#ddba76', '#d9c9a7']
    y = np.arange(len(summaries))
    values = [r['correct'] for r in summaries]
    axes[0].barh(y, values, color=colors)
    for i, row in enumerate(summaries):
        axes[0].text(row['correct'] + .8, i,
                     f"{row['correct']}/96  (+{row['errors_fixed']}, −{row['errors_introduced']})", va='center', fontsize=10)
    axes[0].set(yticks=y, yticklabels=[LABELS[r['method']] for r in summaries], xlim=(0, 116),
                xlabel='Correct crop selections; parentheses: fixed, introduced', title='Same 96 tasks · same frozen CLIP top five')
    axes[0].invert_yaxis()
    axes[0].axvline(95, color='#9aa6ac', lw=1, ls='--')
    axes[0].text(95, -0.65, 'Retrieval ceiling 95', fontsize=9, ha='right', color='#617078')
    targets = sorted({r['target'] for r in objects})
    main = METHODS[:3]
    matrix = np.array([[next(r['correct'] for r in objects if r['method'] == m and r['target'] == t)
                        for m in main] for t in targets])
    axes[1].imshow(matrix, cmap='Blues', vmin=0, vmax=8, aspect='auto')
    for (i, j), value in np.ndenumerate(matrix):
        axes[1].text(j, i, str(value), ha='center', va='center', color='white' if value > 4 else '#163645')
    axes[1].set(xticks=range(3), xticklabels=['CLIP + DINO', '+ depth fusion', 'Geometry only'],
                yticks=range(len(targets)), yticklabels=[pretty(t) for t in targets],
                title='Per object: correct / 8 scenes\nZero-contact depth conditions')
    axes[1].tick_params(axis='both', length=0)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Single-camera depth verification', fontsize=21, x=.025, ha='left')
    fig.text(.025, .015, 'Known-CAD development simulation. No training or new camera views. 1 mm clearance arms are sensitivity controls.', color='#617078', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .93), w_pad=3)
    fig.savefig(OUT / 'comparison.png', dpi=150, facecolor='white')
    plt.close(fig)


def relative_link(path):
    import os
    return os.path.relpath(path, OUT).replace('\\', '/')


def html_report(summaries, objects, changes):
    esc = html.escape
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Single-camera depth verification</title>',
             '<style>body{font:16px/1.5 Segoe UI,Arial,sans-serif;color:#22323c;background:#f5f7f8;margin:0}main{max-width:1400px;margin:auto;padding:36px}h1{font-size:32px;margin:0 0 8px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #dfe5e8}th{background:#e8eef1}.case{background:white;border:1px solid #dae2e6;border-radius:7px;padding:18px;margin:14px 0}.pair{display:flex;gap:18px}.crop{max-width:300px;max-height:210px;object-fit:contain}a{color:#236d8b}.fixed{color:#13715c}.introduced,.lost_to_tie_or_unavailable{color:#a43c27}small{color:#617078}.scores{font-size:13px}.scores td,.scores th{padding:5px 8px}.overview{width:100%}code{font-size:12px}</style></head><body><main>',
             '<h1>Single-camera depth verification</h1><p>One supplied CAD → the same five CLIP crops → table-constrained CAD alignment and verification. All fits and selections were frozen before audit labels were read.</p>',
             '<p><b>Zero contact is primary.</b> The 1 mm clearance conditions are declared simulator sensitivity controls. Correct selections, wrong selections, ties and unavailable results remain separate. Similarities are not probabilities.</p>',
             '<img class="overview" src="comparison.png" alt="Comparison of all methods">',
             '<h2>All 96 tasks</h2><table><tr><th>Method</th><th>Correct</th><th>Wrong</th><th>Tied</th><th>Unavailable</th><th>Fixed</th><th>Introduced</th><th>Correct lost to tie/unavailable</th></tr>']
    for row in summaries:
        parts.append('<tr><td>' + esc(LABELS[row['method']]) + '</td>' + ''.join(f'<td>{row[k]}</td>' for k in ('correct', 'wrong', 'tied', 'unavailable', 'errors_fixed', 'errors_introduced', 'correct_lost_to_tie_or_unavailable')) + '</tr>')
    parts.append('</table><p><a href="per_task.csv">All task outcomes CSV</a> · <a href="per_object.csv">Per-object CSV</a> · <a href="paired_changes.csv">Every changed selection CSV</a> · <a href="summary.json">Exact summary JSON</a> · <a href="../figures/index.html">Projected CAD, observed depth and partial-cloud figures</a></p>')
    parts.append('<h2>Changed selections</h2><p>All changes are shown, including changed wrong choices that remain wrong. Crop identities below are post-hoc audit labels only.</p>')
    for row in changes:
        title = f"{row['state']} · {pretty(row['target'])} · {LABELS[row['method']]}"
        baseline_id = row['baseline_selected_ids'][0] if len(row['baseline_selected_ids']) == 1 else None
        parts.append(f'<section class="case"><h3>{esc(title)}</h3><p class="{row["change"]}"><b>{esc(row["change"].replace("_", " "))}</b>: {esc(row["baseline_outcome"])} → {esc(row["outcome"])}. True crop: {row["truth_proposal_id"]}.</p><div class="pair">')
        for label, ids, product in [('CLIP + DINO baseline', row['baseline_selected_ids'], row['baseline_predicted_product']), ('Depth condition', row['selected_ids'], row['predicted_product'])]:
            parts.append('<div><b>' + esc(label) + '</b><br>')
            for pid in ids:
                image = STUDY / f'data/scenes/{row["state"]}/crops/candidate_{pid:03d}.png'
                parts.append(f'<img class="crop" src="{esc(relative_link(image))}" alt="Observed crop {pid}">')
            parts.append(f'<br>{esc(str(product))} · crop IDs {esc(str(ids))}</div>')
        parts.append('</div><table class="scores"><tr><th>Crop</th><th>Post-hoc identity</th><th>Selected</th><th>Method score</th><th>DINO</th><th>G</th><th>Mask IoU</th><th>Mean depth error, mm</th><th>ICP RMSE, mm</th><th>Frozen fit</th></tr>')
        for score in sorted(row['candidate_scores'], key=lambda x: x['proposal_id']):
            pid = score['proposal_id']
            product = next(r['target'] for r in ALL_ROWS if r['method'] == 'clip_dino_baseline' and r['state'] == row['state'] and r['truth_proposal_id'] == pid)
            def fmt(key, factor=1):
                value = score.get(key)
                return '—' if value is None else f'{value*factor:.4f}'
            link = relative_link(HERE / score['fit_file']) if score.get('fit_file') else None
            parts.append(f'<tr><td>{pid}</td><td>{esc(pretty(product))}</td><td>{"yes" if pid in row["selected_ids"] else ""}</td><td>{fmt("score")}</td><td>{fmt("dino_score")}</td><td>{fmt("geometry_score")}</td><td>{fmt("mask_iou")}</td><td>{fmt("depth_mae_m",1000)}</td><td>{fmt("fitting_rmse_m",1000)}</td><td>' + (f'<a href="{esc(link)}">JSON</a>' if link else '—') + '</td></tr>')
        parts.append('</table></section>')
    parts.append('<p><small>Depth comes from the existing single camera. Table support and native CAD dimensions are assumptions of this verification test. No simulator labels or poses were used in fitting. These reused known-CAD simulation scenes do not establish real-sensor accuracy, held-out generalization, full-6D observability or assembly success.</small></p></main></body></html>')
    (OUT / 'index.html').write_text('\n'.join(parts), encoding='utf-8')


ALL_ROWS = []


def main():
    global ALL_ROWS
    selections, completion = verify_before_labels()
    OUT.mkdir(exist_ok=True)
    source_hashes = {str(p.relative_to(REPO)).replace('\\', '/'): digest(p) for p in
                     (Path(__file__), HERE / 'selections.json', HERE / 'selection_completion.json',
                      HERE / 'fitting_completion.json', HERE / 'protocol.json', STUDY / 'analysis/decisions.json')}
    for path in (HERE / 'execution_adjustment.json', HERE / 'run_parallel.py'):
        if path.exists():
            source_hashes[str(path.relative_to(REPO)).replace('\\', '/')] = digest(path)
    write_once(OUT / 'source_lock.json', {'source_sha256': source_hashes,
        'selection_hashes_verified_before_labels': True})
    write_once(OUT / 'verification.json', dict(RAW_FIT_AUDIT,
        selection_hashes_verified_before_labels=True,
        selections_sha256=completion['selections_sha256'],
        all_candidate_score_and_ranking_arithmetic_verified=True))
    ALL_ROWS = evaluate(selections)
    summaries, objects, changes = summarize(ALL_ROWS)
    write_once(OUT / 'per_task.json', ALL_ROWS)
    write_once(OUT / 'per_object.json', objects)
    write_once(OUT / 'paired_changes.json', changes)
    write_once(OUT / 'summary.json', {'tasks': 96, 'baseline_correct': 66,
        'strict_retrieval_ceiling': 95, 'methods': summaries,
        'primary': 'contact_0mm_fusion',
        'geometry_verification_diagnostic': 'contact_0mm_geometry',
        'simulation_clearance_controls': [m for m in METHODS if m.startswith('clearance')],
        'selection_hashes_verified_before_labels': True})
    for name, rows in [('per_task', ALL_ROWS), ('per_object', objects), ('paired_changes', changes)]:
        write_csv(OUT / (name + '.csv'), rows)
    plots(summaries, objects)
    html_report(summaries, objects, changes)
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != 'completion.json']
    write_once(OUT / 'completion.json', {'task_method_outcomes': len(ALL_ROWS),
        'selections_sha256': completion['selections_sha256'],
        'selected_before_evaluation': True,
        'artifact_sha256': {p.name: digest(p) for p in sorted(artifacts)}})
    print(json.dumps({'tasks': 96, 'outcomes': len(ALL_ROWS), 'summary': summaries}), flush=True)


if __name__ == '__main__':
    main()
