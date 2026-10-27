"""Standalone scientific figures and an evidence-linked failure gallery.

Execute only after the coordinating agent's GO and completed locked evaluation.
All displayed metrics/descriptions come from saved artifacts; no model calls.
"""
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ARMS, HERE, LOCK, OUT, cluster_ci, digest, lock_predictions, path, read, save
from plausibility import DIRECTORY as PLAUSIBILITY

REPORT = OUT/'report'
LABELS = dict(clip_only='CLIP alone', clip_top5_depth='CLIP top-5 + depth',
              depth_all='Depth, all crops', size_only='Size only')
COLORS = dict(clip_only='#64748b', clip_top5_depth='#0077bb', depth_all='#ee7733', size_only='#228833')
CONDITIONS = dict(nominal='Nominal', light_shadow='Harder shadows', background='Background',
    finish='Surface finish', camera_far='Farther camera', resolution_low='Lower resolution',
    occlusion_light='Occluder: light', occlusion_heavy='Occluder: heavy',
    absent='One CAD absent', duplicate='One CAD duplicated')


def safe(value):
    return html.escape(str(value))


def percentage_interval(value, interval, unit='%'):
    if value is None:
        return 'N/A'
    point = f'{100*value:.1f}{unit}'
    return point if interval is None else point+f' [{100*interval[0]:.1f}, {100*interval[1]:.1f}]'


def link(file, label=None):
    file = path(file)
    target = os.path.relpath(file, REPORT).replace('\\', '/')
    return f'<a href="{safe(target)}">{safe(label or file.name)}</a>'


def picture(file, label):
    if file is None:
        return f'<figure><div class="unavailable">Unavailable</div><figcaption>{safe(label)}</figcaption></figure>'
    target = os.path.relpath(path(file), REPORT).replace('\\', '/')
    return f'<figure><a href="{safe(target)}"><img src="{safe(target)}" loading="lazy"></a><figcaption>{safe(label)}</figcaption></figure>'


def export(fig, name):
    fig.savefig(REPORT/(name+'.png'), dpi=200, bbox_inches='tight')
    fig.savefig(REPORT/(name+'.pdf'), bbox_inches='tight')


def condition_plot(table, cohort, conditions, name, title):
    import matplotlib.pyplot as plt
    selected = {(r['condition'], r['method']): r for r in table if r['cohort'] == cohort}
    fig, ax = plt.subplots(figsize=(11., max(4., .47*len(conditions)+1.7)), layout='constrained')
    y = np.arange(len(conditions))
    for i, method in enumerate(ARMS):
        positions = y+(i-1.5)*.15
        for position, condition in zip(positions, conditions):
            row = selected[condition, method]
            value, ci = row['present_accuracy'], row['present_accuracy_cluster_ci95']
            if value is None:
                continue
            x = 100*value
            errors = None if ci is None else [[max(0., x-100*ci[0])], [max(0., 100*ci[1]-x)]]
            ax.errorbar(x, position, xerr=errors, fmt='o', ms=5, color=COLORS[method], capsize=2,
                        label=LABELS[method] if condition == conditions[0] else None)
    ax.set_yticks(y, [CONDITIONS.get(c, c.replace('_', ' ')) for c in conditions])
    ax.invert_yaxis()
    ax.set_xlim(-1, 101)
    ax.set_xlabel('Correct selections / physically present CAD queries (%)')
    ax.set_title(title, loc='left', fontweight='bold', pad=16)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(.5, -.1), frameon=False)
    fig.text(.01, -.04, 'Bars: 95% intervals from paired base-scene cluster bootstrap. Fully hidden present targets remain in the denominator.', fontsize=9)
    export(fig, name)
    plt.close(fig)


def object_heatmap(rows, cohort, name, title, targets=None, condition=None):
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['cohort'] == cohort and (condition is None or r['condition'] == condition)]
    if targets is None:
        targets = list(dict.fromkeys(r['target'] for r in selected))
    groups = defaultdict(list)
    for row in selected:
        groups[row['target'], row['method']].append(row)
    values = np.full((len(targets), len(ARMS)), np.nan)
    counts = {}
    for i, target in enumerate(targets):
        for j, method in enumerate(ARMS):
            present = [r for r in groups[target, method] if r['present']]
            n, correct = len(present), sum(r['correct'] for r in present)
            if n:
                values[i, j] = correct/n
            counts[i, j] = correct, n
    fig, ax = plt.subplots(figsize=(8.8, max(4., .35*len(targets)+1.9)), layout='constrained')
    image = ax.imshow(np.ma.masked_invalid(values), cmap='Blues', vmin=0., vmax=1., aspect='auto')
    ax.set_xticks(range(len(ARMS)), [LABELS[m] for m in ARMS], rotation=15, ha='right')
    ax.set_yticks(range(len(targets)), targets)
    for (i, j), (correct, n) in counts.items():
        ax.text(j, i, f'{correct}/{n}' if n else 'No present query', ha='center', va='center',
                fontsize=9 if n else 7, color='white' if n and correct/n > .55 else '#172033')
    ax.set_title(title, loc='left', fontweight='bold', pad=16)
    fig.colorbar(image, ax=ax, shrink=.6, label='Fraction correct among physically present CAD queries')
    export(fig, name)
    plt.close(fig)


def transfer_plot(table):
    import matplotlib.pyplot as plt
    selected = {(r['cohort'], r['method']): r for r in table}
    fig, axes = plt.subplots(1, 2, figsize=(11., 4.4), layout='constrained')
    for ax, metric, title in zip(axes, ('present_accuracy', 'absent_false_selections'),
                                 ('Real pilot: present CAD queries', 'Real pilot: absent CAD queries')):
        for j, method in enumerate(ARMS):
            row = selected['real_data', method]
            if metric == 'present_accuracy':
                value = row[metric]
                ci = row['present_accuracy_cluster_ci95']
                label = f'{row["correct"]}/{row["physically_present"]}'
            else:
                n = row['absent_queries']
                value = row[metric]/n if n else None
                ci = None
                label = f'{row[metric]}/{n}'
            if value is None:
                continue
            ax.bar(j, value*100, color=COLORS[method], width=.64)
            ax.text(j, value*100+2, label, ha='center', fontsize=9)
            if ci is not None:
                ax.errorbar(j, value*100, yerr=[[max(0., value-ci[0])*100], [max(0., ci[1]-value)*100]],
                            fmt='none', color='#111827', capsize=3)
        ax.set_ylim(0, 112)
        ax.set_xticks(range(4), [LABELS[m] for m in ARMS], rotation=25, ha='right')
        ax.set_ylabel('Correct selections (%)' if metric == 'present_accuracy' else 'Returned a crop despite absent CAD (%)')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Twenty cached T-LESS frames; thirty CAD identities', fontweight='bold')
    export(fig, 'real_pilot_summary')
    plt.close(fig)


def deterministic_cases(rows):
    primary = [r for r in rows if r['method'] == 'clip_top5_depth' and r['present'] and not r['correct']]
    primary.sort(key=lambda r: (r['cohort'] != 'simulation', r['condition'] != 'nominal', r['scene_id'], r['target']))
    categories = [
        ('Round/rectangular pin: wrong size', lambda r: r['pin_error'] == 'same_shape_wrong_size'),
        ('Round versus rectangular pin', lambda r: r['pin_error'] == 'round_rectangular_confusion'),
        ('CLIP excluded every matching proposal', lambda r: r['candidate_recall'] and not r['clip_top5_recall']),
        ('No matching automatic proposal', lambda r: not r['candidate_recall'] and r['state'] == 'present_visible'),
        ('Selected mask contains pixels from multiple instances', lambda r: len(r['selected_mask_instance_pixels']) >= 2),
        ('Occlusion condition', lambda r: r['condition'] in ('occlusion_light', 'occlusion_heavy')),
        ('Real RGB-D transfer failure', lambda r: r['cohort'] == 'real_data')]
    cases, used, missing = [], set(), []
    for title, predicate in categories:
        found = next((r for r in primary if predicate(r) and (r['scene_id'], r['object_id']) not in used), None)
        if found is None:
            missing.append(title)
        else:
            cases.append((title, found))
            used.add((found['scene_id'], found['object_id']))
    return cases, missing


def raw_descriptions(image_id, descriptions):
    content = []
    for mode in ('keywords', 'phrases'):
        row = descriptions.get((image_id, mode))
        if row is None:
            content.append(f'<p>{safe(mode)}: unavailable, no saved description.</p>')
        else:
            content.append(f'<details><summary>{safe(mode)} — {safe(row["status"])}</summary>'
                           f'<pre>{safe(row.get("raw_output", ""))}</pre>'
                           f'<p>Saved status reason: {safe(row.get("reason"))}</p></details>')
    return ''.join(content)


def failure_gallery(rows):
    cases, missing = deterministic_cases(rows)
    inputs = read(HERE/'visual/inputs.json')
    cad_inputs = {(r['target'], r['view_index']): r for r in inputs if r['kind'] == 'CAD'}
    crop_inputs = {(r['state'], r['proposal_id']): r for r in inputs if r['kind'] == 'crop'}
    descriptions = {(r['input_id'], r['mode']): r for r in read(HERE/'visual/descriptions.json')}
    shortlists, pairs = {}, {}
    for file in sorted((HERE/'visual').glob('*/shortlists.json')):
        shortlists.update({(r['scene_id'], r['object_id']): r for r in read(file)})
        pair_file = file.with_name('pairs.json')
        pairs.update({(r['scene_id'], r['object_id'], r['proposal_id']): r for r in read(pair_file)})
    indexed = {(r['scene_id'], r['object_id'], r['method']): r for r in rows}
    sections = []
    for title, row in cases:
        sid, oid, target = row['scene_id'], row['object_id'], row['target']
        task = shortlists[sid, oid]
        geometry_file = HERE/'geometry/scores'/(sid+'.json')
        geometry = {r['proposal_id']: r for r in read(geometry_file)['pairs'] if r['object_id'] == oid}
        selected_pair = pairs.get((sid, oid, row['proposal_id']))
        view = selected_pair['best_CAD_view'] if selected_pair else 1
        cad = cad_inputs[target, view]
        image_panels = [picture(row['rgb'], f'{sid}: complete observed RGB'),
                        picture(cad['file'], f'Requested CAD: {target}, view {view}' + (' (first view; no final selection)' if selected_pair is None else ''))]
        for method in ('clip_only', 'clip_top5_depth', 'depth_all'):
            decision = indexed[sid, oid, method]
            image_panels.append(picture(decision['crop'], f'{LABELS[method]}: p{decision["proposal_id"]}; '
                                         f'{"correct" if decision["correct"] else decision["status"] if decision["proposal_id"] is None else "wrong"}'))
        rank_rows = []
        display_ids = [pid for _, pid in task['scores'][:5]]
        matching_ids = [int(pid) for pid, iou in row['candidate_target_bbox_iou'].items() if iou >= .5]
        display_ids.extend(pid for pid in matching_ids if pid not in display_ids)
        rank_map = {pid: (rank+1, score) for rank, (score, pid) in enumerate(task['scores'])}
        for pid in display_ids:
            rank, score = rank_map.get(pid, (None, None))
            pair, fit = pairs.get((sid, oid, pid)), geometry.get(pid)
            crop = crop_inputs.get((sid, pid))
            cad_view = None if pair is None else pair['best_CAD_view']
            crop_image = picture(crop['file'], f'p{pid}') if crop else 'No crop record'
            fit_link = link(fit['fit_file'], 'component scores / hypotheses') if fit else 'Unavailable'
            reference_image = picture(cad_inputs[target, cad_view]['file'], f'CAD view {cad_view}') if cad_view is not None else 'Unavailable'
            rank_rows.append(f'<tr><td>{safe(rank)}</td><td>{crop_image}</td><td>{reference_image}</td><td>{safe(score)}</td>'
                f'<td>{safe(pid in task["candidate_ids"])}</td><td>{safe(row["candidate_target_bbox_iou"].get(str(pid)))}</td>'
                f'<td>{safe(None if fit is None else fit["geometry_score"])}</td>'
                f'<td>{safe(None if fit is None else fit["size_score"])}</td><td>{safe(cad_view)}</td><td>{fit_link}</td></tr>')
        selected_input = crop_inputs.get((sid, row['proposal_id']))
        text = '<h4>Exact saved Qwen output: CAD view</h4>'+raw_descriptions(cad['input_id'], descriptions)
        if selected_input is not None:
            text += '<h4>Exact saved Qwen output: selected crop</h4>'+raw_descriptions(selected_input['input_id'], descriptions)
        text += '<h4>Exact saved Qwen output: ranked candidate crops</h4>'
        for pid in display_ids:
            candidate = crop_inputs.get((sid, pid))
            if candidate is not None:
                text += f'<details><summary>Proposal {pid}</summary>'+raw_descriptions(candidate['input_id'], descriptions)+'</details>'
        sections.append(f'<section><h2>{safe(title)}</h2><p><b>{safe(sid)} · {safe(target)}</b> — '
            f'{safe(row["condition"])}. State: {safe(row["state"])}. '
            f'Visible target pixels: {row["target_visible_pixels"]}; nominal target pixels: {safe(row["nominal_target_visible_pixels"])}; '
            f'ratio: {safe(row["target_pixel_fraction_relative_nominal"])}.</p>'
            f'<div class="panels">{"".join(image_panels)}</div>'
            '<p>Primary matching rule: selected box IoU ≥ 0.5 with any visible instance of the requested CAD. '
            'Multiple-instance mask evidence lists raw labeled pixel contributions; it does not prove multiple detected objects.</p>'
            f'<p>Selected observed-mask instance pixels: <code>{safe(json.dumps(row["selected_mask_instance_pixels"]))}</code></p>'
            f'<p>Diagnostic mask identity: {safe(row["selected_physical_identity_status"])}. '
            f'Tied maximum-overlap identities: <code>{safe(json.dumps(row["selected_physical_tied_identities"]))}</code>. '
            'Tied diagnostic identities remain unresolved; primary bbox correctness is unchanged.</p>'
            '<table><thead><tr><th>CLIP rank</th><th>Observed crop</th><th>Corresponding CAD view</th><th>CLIP score</th><th>In top-five</th>'
            '<th>Target box IoU</th><th>Depth score</th><th>Size score</th><th>Best CLIP CAD view</th><th>Evidence</th></tr></thead>'
            f'<tbody>{"".join(rank_rows)}</tbody></table>{text}'
            f'<p>{link(geometry_file, "All geometric scores")} · {link(HERE/"geometry/localization"/sid/"localization.json", "Automatic proposals")} · '
            f'{link(HERE/"visual/descriptions.json", "Verbatim description records")}</p></section>')
    body = '<h1>Deterministic failure gallery</h1><p>Primary arm: CLIP top-five retrieval + depth selection. '
    body += 'Each category uses the first unused eligible query in the declared ordering: simulation first, nominal first, then scene ID and CAD name. '
    body += 'All evidence below is saved output. Unavailable predictions are not certified absence.</p>'
    protocol = read(HERE/'visual/protocol.json')
    body += '<details><summary>Exact system prompt and two independent user prompts</summary>'
    body += '<h3>System prompt</h3><pre>'+safe(protocol['system'])+'</pre>'
    for mode, prompt in protocol['prompts'].items():
        body += '<h3>'+safe(mode)+'</h3><pre>'+safe(prompt)+'</pre>'
    body += '</details>'
    body += '<p>CAD and crops are raw saved images. CLIP scores are the equal mean of image/image, text/text, image/text and text/image similarities, maximized over CAD views. '
    body += 'Depth/size scores are uncalibrated; component links preserve their actual units and hypothesis records.</p>'
    body += ''.join(sections)
    if missing:
        body += '<h2>Categories without an unused eligible failure</h2><ul>'+''.join(f'<li>{safe(x)}</li>' for x in missing)+'</ul>'
    save(REPORT/'gallery_cases.json', dict(cases=[dict(category=title, scene_id=r['scene_id'], target=r['target']) for title, r in cases],
        missing_categories=missing, selection='Deterministic saved-evidence order; no cherry-picked score cutoff.'))
    return body


def native_baseline_table(rows):
    """Called after GO; native outputs use their own masks and frozen scores."""
    folder = OUT/'native_baselines'
    if not (folder/'summary.json').exists():
        return ''
    completion = read(folder/'completion.json')
    assert completion['prediction_lock_sha256'] == digest(LOCK)
    summary = read(folder/'summary.json')
    nominal = [r for r in rows if r['cohort'] == 'simulation' and r['condition'] == 'nominal'
               and r['method'] == 'clip_top5_depth']
    assert len(nominal) == 240 and all(r['present'] for r in nominal)
    correct = sum(r['correct'] for r in nominal)
    recall = sum(r['candidate_recall'] for r in nominal)
    our_accuracy = percentage_interval(correct/240, cluster_ci(nominal))
    table_rows = [f'<tr><td>Our CLIP top-5 + depth (nominal reference)</td><td>{correct}/240</td>'
                  f'<td>{safe(our_accuracy)}</td><td>{recall}/240</td><td>Reference</td><td>Reference</td><td>Reference</td></tr>']
    names = dict(cnos='CNOS', sam6d_rgb='SAM-6D RGB', sam6d_rgbd='SAM-6D RGB-D',
                 nids_basic_finite_templates='NIDS-Net basic — finite-reference control',
                 noctis='NOCTIS', nemo='NeMO',
                 nids_basic_native_unguarded='NIDS-Net basic — native reference failure (N/A)')
    for row in summary:
        assert row['queries'] == 240
        available = row['available']
        total = f'{row["correct"]}/{row["queries"]}' if available else 'N/A'
        accuracy = percentage_interval(row['accuracy'], row['accuracy_cluster_ci95'])
        difference = percentage_interval(row['accuracy_difference_vs_ours'], row['paired_difference_cluster_ci95'], ' pp')
        raw_recall = (f'{row["own_raw_candidate_recalled"]}/{row["own_raw_candidate_queries"]}'
                      if available and row['own_raw_candidate_queries'] else 'N/A')
        fixed = row['fixed_vs_ours'] if available else 'N/A'
        introduced = row['introduced_vs_ours'] if available else 'N/A'
        table_rows.append('<tr>'+''.join(f'<td>{safe(value)}</td>' for value in
            (names.get(row['method'], row['method']), total, accuracy, raw_recall, fixed, introduced, difference))+'</tr>')
    body = '<h2>Native baselines · twenty nominal scenes</h2>'
    body += '<p>Each method generates its own masks from the same full RGB-D scenes and CAD catalog. '
    body += 'The 240 queries are the twelve requested CADs across twenty layouts. This compares proposal generation and identification together.</p>'
    body += '<table><thead><tr><th>Method</th><th>Correct / queries</th><th>Accuracy [95% CI]</th><th>Raw proposal recall</th>'
    body += '<th>Our errors fixed</th><th>Our correct matches made wrong</th><th>Difference vs ours [95% CI], pp</th></tr></thead><tbody>'+''.join(table_rows)+'</tbody></table>'
    body += '<p>Native basic NIDS-Net is unavailable because its native reference features are invalid; '
    body += 'it is not counted as zero accuracy. The separately declared finite-reference control is a completed, distinct variant. '
    body += 'NeMO uses CAD-conditioned masks, so class-agnostic raw proposal recall is N/A. '
    body += 'Accuracy and paired-difference intervals resample the twenty base layouts together, with 2,000 draws.</p>'
    available_links = [link(folder/(name+'.csv'), label) for name, label in
        [('summary', 'Native summary CSV'), ('per_object', 'Native per-object CSV'),
         ('paired_changes', 'Native fixed / introduced errors CSV'), ('paired_summary', 'Native paired intervals CSV'),
         ('per_query', 'Every native query CSV')]
        if (folder/(name+'.csv')).exists()]
    return body+'<p>'+' · '.join(available_links)+'</p>'


def plausibility_table():
    calibration = read(PLAUSIBILITY/'calibration.json')
    table = read(OUT/'results/plausibility_diagnostic.json')
    body = '<h2>Supplementary development-fixed plausibility</h2>'
    body += '<p>Historical N00–N07 labels and 480 saved top-five fits fixed the threshold before NEW robustness accuracy annotation access. '
    body += 'All historical negatives, including gripper regions, were retained. No model reruns, new fitting, primary ranking changes or held-out coverage tuning.</p>'
    body += f'<p>Threshold τ: <code>{safe(calibration["tau"])}</code>. Development target pairs passing: '
    body += f'{calibration["target_pairs_passing"]}/{calibration["bbox_target_pairs"]} (finite: {calibration["finite_target_pairs"]}); '
    body += f'non-target pairs passing: {calibration["non_target_pairs_passing"]}/{calibration["bbox_non_target_pairs"]} '
    body += f'(finite: {calibration["finite_non_target_pairs"]}). Target coverage, including unavailable pairs: '
    body += percentage_interval(calibration['target_coverage_all_pairs'], None)+'. Zero development non-target passes is not a generalization guarantee.</p>'
    body += '<table><thead><tr><th>Cohort / condition</th><th>Queries</th><th>0 passing</th><th>1 passing</th><th>Multiple passing</th>'
    body += '<th>Unavailable</th><th>Correct present winners passing / correct present winners</th><th>Absent winners passing / absent queries</th></tr></thead><tbody>'
    for row in table:
        values = (row['cohort']+' / '+row['condition'], row['queries'], row['zero_passing_proposals'],
                  row['one_passing_proposal'], row['multiple_passing_proposals'], row['unavailable'],
                  f'{row["present_correct_winner_passing"]}/{row["present_primary_correct"]}',
                  f'{row["absent_primary_winner_passing"]}/{row["absent_queries"]}')
        body += '<tr>'+''.join(f'<td>{safe(value)}</td>' for value in values)+'</tr>'
    body += '</tbody></table><p>Confirmatory scope: known twelve CADs, CLIP top-five plus depth, simulation and depth-stress cohorts only. '
    body += 'U/T cohorts and other arms are excluded. Counts refer to proposals, which can overlap or describe one physical object. '
    body += 'Zero passing proposals never certifies absence; multiple passing proposals never certifies duplicates. Original unavailable candidates and score ties remain explicit.</p>'
    return body+'<p>'+link(PLAUSIBILITY/'protocol.json', 'Frozen supplementary protocol')+' · '+link(PLAUSIBILITY/'calibration.json', 'Development coverage')+' · '+link(OUT/'results/plausibility_diagnostic.csv', 'Diagnostic CSV')+'</p>'


def page(title, body):
    return '<!doctype html><html><head><meta charset="utf-8"><title>'+safe(title)+'</title><style>'\
        'body{font:16px/1.5 system-ui,sans-serif;color:#172033;margin:40px auto;max-width:1420px;padding:0 24px}'\
        'h1,h2{line-height:1.15}h2{margin-top:0}section{border-top:1px solid #cbd5e1;padding:28px 0}'\
        '.panels{display:flex;gap:14px;align-items:start;flex-wrap:wrap}figure{margin:0;flex:1;min-width:160px}'\
        'img{width:100%;max-height:310px;object-fit:contain;background:#f1f5f9}figcaption{font-size:13px}'\
        'table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:8px;border:1px solid #cbd5e1;text-align:left}'\
        'td figure{min-width:100px}td img{height:100px;max-width:180px}pre{white-space:pre-wrap;background:#f1f5f9;padding:14px}'\
        '.unavailable{padding:60px 8px;background:#f1f5f9;text-align:center}details{margin:8px 0}a{color:#0077bb}'\
        '.chart{max-width:1120px;display:block;margin:26px 0}code{overflow-wrap:anywhere}</style></head><body>'+body+'</body></html>'


def main():
    assert sys.argv[1:] == ['--after-go'], 'Run only after GO and completed evaluation.'
    lock_predictions(create=False)
    result = OUT/'results'
    completion = read(result/'completion.json')
    assert completion['prediction_lock_sha256'] == digest(LOCK)
    rows = read(result/'per_query.json')
    table = read(result/'per_condition.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                               'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    conditions = [c for c in CONDITIONS if any(r['cohort'] == 'simulation' and r['condition'] == c for r in table)]
    condition_plot(table, 'simulation', conditions, 'controlled_conditions', 'Controlled simulation conditions · twenty paired layouts')
    stresses = list(dict.fromkeys(r['condition'] for r in table if r['cohort'] == 'depth_stress'))
    condition_plot(table, 'depth_stress', stresses, 'depth_stresses', 'Depth corruption · twenty paired layouts')
    object_heatmap(rows, 'simulation', 'nominal_per_object', 'Nominal simulation: correct / present queries', condition='nominal')
    females = [r['target'] for r in read(HERE/'new_cads/catalog.json')['objects'] if r['target'].endswith('_Female')]
    object_heatmap(rows, 'new_cads', 'new_query_CADs', 'Four newly queried connector CADs · twenty new layouts', targets=females)
    object_heatmap(rows, 'real_data', 'real_pilot_per_object', 'T-LESS real pilot: correct / present queries')
    transfer_plot(read(result/'per_cohort.json'))
    gallery = failure_gallery(rows)
    (REPORT/'failure_gallery.html').write_text(page('Classification failure gallery', gallery), encoding='utf-8')
    summaries = read(result/'per_cohort.json')
    summary_rows = ''.join('<tr>'+''.join(f'<td>{safe(value)}</td>' for value in
        (r['cohort'], LABELS[r['method']], f'{r["correct"]}/{r["physically_present"]}',
         r['unavailable_present'], r['absent_false_selections'], r['absent_unavailable']))+'</tr>' for r in summaries)
    body = '<h1>CAD-to-scene classification robustness</h1><p>Four frozen ranking arms; automatic depth proposals; native metric CADs. '
    body += 'Predictions were locked before classification-accuracy annotation access. Earlier simulator capture QA used truth for counts, projection and paired visibility input checks, without classification scoring or selection feedback. This evaluates recognition, not grasping or assembly.</p>'
    body += '<table><thead><tr><th>Cohort</th><th>Method</th><th>Correct / present</th><th>Unavailable present</th>'
    body += '<th>Absent: false crop selection</th><th>Absent: unavailable</th></tr></thead><tbody>'+summary_rows+'</tbody></table>'
    body += native_baseline_table(rows)
    body += plausibility_table()
    for name, title in [('controlled_conditions', 'Controlled conditions'), ('depth_stresses', 'Depth stresses'),
                        ('nominal_per_object', 'Nominal per-object results'), ('new_query_CADs', 'Newly queried CADs'),
                        ('real_pilot_summary', 'Real RGB-D pilot'), ('real_pilot_per_object', 'Real pilot per-object results')]:
        body += f'<h2>{safe(title)}</h2><img class="chart" src="{name}.png"><p>{link(REPORT/(name+".pdf"), "Vector PDF")}</p>'
    body += '<h2>How to read these results</h2><ul><li>Present denominators include fully hidden targets; observable-only accuracy is available in the tables.</li>'
    body += '<li>Occlusion ratios divide target visible pixel counts by same-layout nominal counts. They are not physical visibility or amodal estimates.</li>'
    body += '<li>No returned crop means unavailable. None of these rankers certifies absence or resolves multiple plausible instances.</li>'
    body += '<li>Absent-target condition pairs are explicitly non-comparable for fixed/introduced correctness changes. Maximum mask-overlap identity ties remain unresolved and retain every tied instance.</li>'
    body += '<li>Female connectors are new classification queries, but appeared earlier as fixed fixtures.</li>'
    body += '<li>The real transfer pilot uses twenty deterministic cached T-LESS frames and all thirty CAD identities. It is not the full BOP benchmark; occlusion and support conditions may violate the table model.</li>'
    body += '<li>Confidence intervals resample base scenes, preserving object and condition pairs. Threshold curves use a fixed grid. The separate plausibility threshold was fixed using historical development labels only, without held-out tuning.</li></ul>'
    body += '<p>'+link(REPORT/'failure_gallery.html', 'Failure gallery with exact prompts, descriptions and scores')+' · '
    body += link(result/'per_query.csv', 'Every query')+' · '+link(result/'paired_changes.csv', 'Fixed and introduced errors')+' · '
    body += link(result/'paired_condition_effects.csv', 'Paired condition effects')+' · '+link(result/'occlusion_by_relative_nominal_pixels.csv', 'Measured occlusion groups')+'</p>'
    (REPORT/'index.html').write_text(page('CAD recognition robustness', body), encoding='utf-8')
    save(REPORT/'completion.json', dict(evaluation_completion_sha256=digest(result/'completion.json'),
        prediction_lock_sha256=digest(LOCK), figures=6, formats=['PNG', 'PDF'],
        gallery='failure_gallery.html', index='index.html', model_calls=0, labels_read='Only previously evaluated result tables; prediction lock verified first.'))
    print(str(REPORT/'index.html'))


if __name__ == '__main__':
    main()
