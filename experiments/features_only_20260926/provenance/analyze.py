"""Paired analysis against the immediately preceding family-association trial."""
import importlib.util
import math
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from run_trial import HERE, ROOT, SOURCE, MODELS, read, save, table, digest, collect, family, validate_inputs, audit

spec = importlib.util.spec_from_file_location('family_analysis_helpers', SOURCE / 'analyze.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
interval, count, auc = helpers.interval, helpers.count, helpers.auc
summarize, thresholds = helpers.summarize, helpers.thresholds
STATES, FAMILIES, CHECKS = helpers.STATES, helpers.FAMILIES, helpers.CHECKS
CONDITIONS = ('family_association', 'features_only')


def heatmaps(rows, model, names):
    dest = HERE / MODELS[model]
    order = list(names)
    idx = {(r['state'], r['target'], r['posthoc_identity']): r for r in rows}
    with PdfPages(dest / 'four_scene_heatmaps.pdf') as pdf:
        for state in STATES:
            matrix = np.array([[interval(idx[state, t, c])[1] for c in order] for t in order])
            records = []
            fig, ax = plt.subplots(figsize=(15.5, 11))
            im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(order)), [names[c] + (' *' if idx[state, order[0], c]['assisted_fallback'] else '') for c in order], rotation=55, ha='right', fontsize=10)
            ax.set_yticks(range(len(order)), list(names.values()), fontsize=10)
            for i, t in enumerate(order):
                record = {'target': t}
                for j, c in enumerate(order):
                    r = idx[state, t, c]
                    value = f"{r['p_A']:.3f}" if r['p_A'] is not None else f"<={math.ceil(interval(r)[1] * 1000) / 1000:.3f}"
                    record[c] = r['p_A']
                    ax.text(j, i, value, ha='center', va='center', fontsize=9, color='white' if matrix[i, j] > .6 else '#222222')
                records.append(record)
            table(dest / (state + '_p_A_matrix'), records)
            ax.set(title=f'{model}: features-only association | {state}', xlabel='Candidate identity (evaluation only)', ylabel='Target identity (evaluation only; not sent to model)')
            fig.colorbar(im, ax=ax, fraction=.027, pad=.025, label='Raw P(A)')
            fig.subplots_adjust(left=.20, right=.94, top=.91, bottom=.25)
            fig.text(.025, .025, '* Assisted USB rectangle. Names shown only for evaluation. No inference cutoff; no probability normalization.', fontsize=10)
            fig.savefig(dest / (state + '_p_A_matrix.png'), dpi=180)
            pdf.savefig(fig)
            plt.close(fig)


def plots(groups, sweep):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, model in enumerate(MODELS):
        for condition, color in zip(CONDITIONS, ('#3672a3', '#b55432')):
            points = [r for r in sweep if r['model'] == model and r['condition'] == condition and r['scope'] == 'all_crops' and r['state'] == 'all']
            for ax, key in ((axes[0, col], 'family'), (axes[1, col], 'unrelated')):
                x = [r['threshold'] for r in points]
                ax.plot(x, [r[key + '_rate_lower'] for r in points], label=condition, color=color)
                ax.fill_between(x, [r[key + '_rate_lower'] for r in points], [r[key + '_rate_upper'] for r in points], color=color, alpha=.15)
                ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Raw P(A) threshold')
                ax.grid(alpha=.2)
            axes[0, col].set_title(model)
            axes[0, col].legend(fontsize=8)
    axes[0, 0].set_ylabel('Family retention')
    axes[1, 0].set_ylabel('Unrelated acceptance')
    fig.suptitle('Same crops and inference settings: family association vs features-only')
    fig.tight_layout()
    fig.savefig(HERE / 'threshold_comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, model in zip(axes, MODELS):
        for condition, color in zip(CONDITIONS, ('#3672a3', '#b55432')):
            rows = groups[model, condition]
            ts = sorted({v for r in rows for v in interval(r)} | {0., 1., math.nextafter(1., math.inf)}, reverse=True)
            pos = [r for r in rows if r['same_family']]
            neg = [r for r in rows if not r['same_family']]
            ax.plot([count(neg, t)[1] / len(neg) for t in ts], [count(pos, t)[0] / len(pos) for t in ts], color=color, label=condition)
        ax.set(title=model, xlabel='Unrelated acceptance (upper bound)', ylabel='Family retention (lower bound)', xlim=(0, .2), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Comparable false-match rates; development scenes, not held-out validation')
    fig.tight_layout()
    fig.savefig(HERE / 'family_retention_vs_false_matches.png', dpi=180)
    plt.close(fig)


def main():
    m = read(HERE / 'manifest.json')
    audit(m)
    validate_inputs(m)
    baseline = read(HERE / 'baseline_scores.json')
    groups, all_rows, paired = {}, [], []
    for model in MODELS:
        old = [r for r in baseline if r['model'] == model]
        new = collect(model)
        for condition, rows in zip(CONDITIONS, (old, new)):
            groups[model, condition] = rows
            all_rows.extend(dict(condition=condition, **r) for r in rows)
        idx = {(r['state'], r['target'], r['posthoc_identity']): r for r in old}
        for r in new:
            b = idx[r['state'], r['target'], r['posthoc_identity']]
            assert b['crop_sha256'] == r['crop_sha256'] and b['same_family'] == r['same_family']
            record = {k: r[k] for k in ('model', 'state', 'target', 'target_family', 'posthoc_identity', 'observed_family', 'same_family', 'own_object_match', 'assisted_fallback')}
            for prefix, row in zip(CONDITIONS, (b, r)):
                record[prefix + '_answer'] = row['answer']
                record[prefix + '_response_file'] = row['response_file']
                for label in 'ABC':
                    record[prefix + '_p_' + label] = row['p_' + label]
                    record[prefix + '_p_' + label + '_lower'], record[prefix + '_p_' + label + '_upper'] = interval(row, label)
            record['p_A_change_lower'] = interval(r)[0] - interval(b)[1]
            record['p_A_change_upper'] = interval(r)[1] - interval(b)[0]
            paired.append(record)
        heatmaps(new, model, m['names'])
    table(HERE / 'all_scores_comparison', all_rows)
    table(HERE / 'paired_scores', paired)
    metrics, sweep, fm, ft = [], [], [], []
    for (model, condition), rows in groups.items():
        for scope in ('all_crops', 'autonomous_depth_only'):
            eligible = [r for r in rows if scope == 'all_crops' or not r['assisted_fallback']]
            for state in (*STATES, 'all'):
                group = [r for r in eligible if state == 'all' or r['state'] == state]
                metrics.append(summarize(group, condition, model, scope, state, 'all'))
                sweep.extend(thresholds(group, condition, model, scope, state, 'all'))
                for fam in FAMILIES:
                    fr = [r for r in group if r['target_family'] == fam]
                    if not any(r['same_family'] for r in fr):
                        continue
                    fm.append(summarize(fr, condition, model, scope, state, fam))
                    ft.extend(r for r in thresholds(fr, condition, model, scope, state, fam) if r['threshold'] in CHECKS)
    for r in metrics:
        fs = [f for f in fm if all(f[k] == r[k] for k in ('condition', 'model', 'scope', 'state'))]
        r['macro_family_auc_lower'] = float(np.mean([f['family_vs_unrelated_auc_lower'] for f in fs]))
        r['macro_family_auc_upper'] = float(np.mean([f['family_vs_unrelated_auc_upper'] for f in fs]))
        r['macro_family_count'] = len(fs)
    for name, rows in [('metrics', metrics), ('threshold_sweep', sweep), ('per_family_metrics', fm), ('per_family_thresholds', ft)]:
        table(HERE / name, rows)
    table(HERE / 'threshold_comparison', [r for r in sweep if r['threshold'] in CHECKS])
    comparison = []
    for model in MODELS:
        for scope in ('all_crops', 'autonomous_depth_only'):
            for state in (*STATES, 'all'):
                a, b = [next(r for r in metrics if r['model'] == model and r['condition'] == c and r['scope'] == scope and r['state'] == state) for c in CONDITIONS]
                comparison.append(dict(model=model, scope=scope, state=state,
                    family_association_auc_lower=a['family_vs_unrelated_auc_lower'],
                    family_association_auc_upper=a['family_vs_unrelated_auc_upper'],
                    features_only_auc_lower=b['family_vs_unrelated_auc_lower'],
                    features_only_auc_upper=b['family_vs_unrelated_auc_upper'],
                    auc_change_lower=b['family_vs_unrelated_auc_lower'] - a['family_vs_unrelated_auc_upper'],
                    auc_change_upper=b['family_vs_unrelated_auc_upper'] - a['family_vs_unrelated_auc_lower'],
                    family_C_change=b['family_C_count'] - a['family_C_count'],
                    unrelated_C_change=b['unrelated_C_count'] - a['unrelated_C_count']))
    table(HERE / 'baseline_comparison', comparison)
    operating = []
    for model in MODELS:
        base = groups[model, CONDITIONS[0]]
        base_budget = count([r for r in base if not r['same_family']], .7)[1]
        for condition in CONDITIONS:
            rows = groups[model, condition]
            pos = [r for r in rows if r['same_family']]
            neg = [r for r in rows if not r['same_family']]
            own = [r for r in rows if r['own_object_match']]
            # Include an explicit reject-all point and all score breakpoints.
            ts = {v for r in rows for v in interval(r)} | {0., 1., math.nextafter(1., math.inf)}
            choices = [(count(pos, t)[0], -count(neg, t)[1], count(own, t)[0], t) for t in ts]
            for budget in sorted({0, 2, 5, 8, 14, 20, 40, base_budget}):
                best = max(v for v in choices if -v[1] <= budget)
                operating.append(dict(model=model, condition=condition, false_match_budget=budget,
                    family_total=len(pos), unrelated_total=len(neg), own_total=len(own),
                    family_retained_lower=best[0], unrelated_accepted_upper=-best[1], own_retained_lower=best[2],
                    threshold=best[3], family_retention=best[0] / len(pos), unrelated_acceptance=-best[1] / len(neg),
                    selection='Development-selected maximum family retention under the same false-match budget; optimistic, not validated.'))
    table(HERE / 'matched_false_match_operating_points', operating)
    known, known_pairs = [], []
    definitions = [('Waterproof <- DSUB', 'Waterproof_Male', 'DSUB_Male'), ('DSUB <- USB', 'DSUB_Male', 'USB_Male'), ('Gear <- hex nut', 'gear', 'hex_nut'), ('True nut family crops', 'hex_nut', 'hex_nut')]
    for label, tf, of in definitions:
        known_pairs.extend(dict(case=label, **r) for r in paired if r['target_family'] == tf and r['observed_family'] == of)
        for (model, condition), rows in groups.items():
            rs = [r for r in rows if r['target_family'] == tf and r['observed_family'] == of]
            rec = dict(case=label, model=model, condition=condition, pairs=len(rs), C_count=sum(r['answer'] == 'C' for r in rs))
            for letter in 'AC':
                for i, side in enumerate(('lower', 'upper')):
                    rec[f'mean_p_{letter}_{side}'] = float(np.mean([interval(r, letter)[i] for r in rs]))
            for t in CHECKS:
                rec[f'pass_{t}_lower'], rec[f'pass_{t}_upper'] = count(rs, t)
            known.append(rec)
    table(HERE / 'known_failures', known)
    table(HERE / 'known_failure_pairs', known_pairs)
    ranked, consistency = [], []
    for (model, condition), rows in groups.items():
        ordered = sorted((r for r in rows if not r['same_family']), key=lambda r: -interval(r)[1])
        ranked.extend(dict(condition=condition, rank=i, **r) for i, r in enumerate(ordered, 1))
        for state in STATES:
            for observed in m['names']:
                for fam in ('gear', 'hex_nut', 'round_pin'):
                    rs = [r for r in rows if r['state'] == state and r['posthoc_identity'] == observed and r['target_family'] == fam]
                    consistency.append(dict(model=model, condition=condition, state=state, observed=observed, target_family=fam,
                        repeated_queries=len(rs), p_A_range_upper=max(interval(r)[1] for r in rs) - min(interval(r)[0] for r in rs),
                        answer_agreement=len({r['answer'] for r in rs}) == 1))
    table(HERE / 'ranked_unrelated_pairs', ranked)
    table(HERE / 'top_unrelated_pairs', [r for r in ranked if r['rank'] <= 20])
    table(HERE / 'repeated_family_query_consistency', consistency)
    plots(groups, sweep)
    report(metrics, sweep, fm, ft, known, operating)
    validate_inputs(m)
    save(HERE / 'validation.json', dict(baseline_unchanged=True, calls=1452, paired_comparisons=1452,
        local_only=True, **audit(m), raw_response_hashes={r['response_file']: digest(ROOT / r['response_file']) for r in all_rows}))
    # Retain original workspace-relative paths so raw records resolve after extraction.
    paths = {p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.png', '.pdf', '.py', '.txt', '.md') and '__pycache__' not in p.parts}
    paths.update(ROOT / r['response_file'] for r in baseline)
    paths.update(ROOT / c['crop_file'] for crops in m['crops'].values() for c in crops)
    paths.update(SOURCE / name for name in ('manifest.json', 'descriptions.json', 'system_prompt.txt', 'user_prompt.txt', 'run_trial.py', 'analyze.py'))
    paths.update((ROOT / 'gpt6-gemma4-four-scene-trial/run_models.py', ROOT / 'qwen9b-four-scene-trial/run_trial.py'))
    with zipfile.ZipFile(HERE / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(HERE / 'results.zip') as z:
        assert z.testzip() is None
    print((HERE / 'README.md').read_text(encoding='utf-8'))


def report(metrics, sweep, fm, ft, known, operating):
    def span(a, b, digits=None):
        fmt = (lambda v: f'{v:.{digits}f}') if digits is not None else str
        return fmt(a) if a == b else f'{fmt(a)}-{fmt(b)}'
    lines = ['# Features-Only Association: Paired Local Experiment', '',
        '1,452 new calls: three local models, four frozen scenes, 11 targets x 11 candidates per scene. Direct baseline: the immediately preceding family-association condition, not the older exact-object trial.', '',
        '## Controlled Comparison',
        '- Same 44 RGB rectangle crops, crop hashes, target/candidate ordering, model digests, Ollama version, inference settings and raw probability extraction.',
        '- Only task wording, label meanings and descriptions change together. This does not isolate the causal effect of removing names alone.',
        '- Exact supplied descriptions are preserved in descriptions.json. No names, IDs, sizes or category headers appear in either model-visible prompt. Names in charts are evaluation labels only.',
        '- All size variants share identical full prompts, but were executed independently to preserve pairing. These are correlated repeated queries, not independent trials.',
        '- P(A)=exp(raw logp_A), without normalization, calibration or filtering. P(C) is a token probability; emitted-C frequency is reported separately.',
        '- Missing exact top-logprob values stay null with conservative bounds, never imputed as zero.',
        '- Two USB crops (states 101 and 303) remain assisted recoveries. Autonomous-depth-only tables exclude all comparisons against them. M8 and square pins remain excluded.',
        '- Original family results and earlier baseline files remain unchanged, checked by SHA-256.', '',
        '## Results at P(A) >= 0.70',
        '| Model | Condition | Family AUC | Family /100 | Unrelated /384 | Own /44 | C labels /484 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        for condition in CONDITIONS:
            r = next(r for r in metrics if r['model'] == model and r['condition'] == condition and r['scope'] == 'all_crops' and r['state'] == 'all')
            s = next(r for r in sweep if r['model'] == model and r['condition'] == condition and r['scope'] == 'all_crops' and r['state'] == 'all' and r['threshold'] == .7)
            cells = [span(s[k + '_retained_lower'], s[k + '_retained_upper']) for k in ('family', 'unrelated', 'own')]
            lines.append(f"| {model} | {condition} | {span(r['family_vs_unrelated_auc_lower'], r['family_vs_unrelated_auc_upper'], 4)} | {' | '.join(cells)} | {r['all_C_count']} |")
    lines.extend(['', '## Matched False-Match Limits',
        'Both conditions select their own threshold under the same maximum unrelated-acceptance count. These are optimistic development-set operating points, not validated gates.',
        '| Model | Maximum unrelated /384 | Family: previous -> features | Actual unrelated: previous -> features |',
        '|---|---:|---:|---:|'])
    for model in MODELS:
        for budget in (0, 5, 14, 20):
            rs = [next(r for r in operating if r['model'] == model and r['condition'] == c and r['false_match_budget'] == budget) for c in CONDITIONS]
            lines.append(f"| {model} | {budget} | {rs[0]['family_retained_lower']} -> {rs[1]['family_retained_lower']} | {rs[0]['unrelated_accepted_upper']} -> {rs[1]['unrelated_accepted_upper']} |")
    lines.extend(['', '## Specific Confusions',
        'At .70, the first three rows per model are unrelated false matches; true-nut rows should be retained.',
        '| Model | Case | Accepted: previous -> features | Pairs | C labels: previous -> features |',
        '|---|---|---:|---:|---:|'])
    for model in MODELS:
        for case in dict.fromkeys(r['case'] for r in known):
            a, b = [next(r for r in known if r['model'] == model and r['condition'] == c and r['case'] == case) for c in CONDITIONS]
            lines.append(f"| {model} | {case} | {span(a['pass_0.7_lower'], a['pass_0.7_upper'])} -> {span(b['pass_0.7_lower'], b['pass_0.7_upper'])} | {a['pairs']} | {a['C_count']} -> {b['C_count']} |")
    lines.extend(['', '## Per-Family Results', '| Model | Family | Family retained: previous -> features | Family pairs | Unrelated accepted: previous -> features | Unrelated pairs |', '|---|---|---:|---:|---:|---:|'])
    for model in MODELS:
        for fam in FAMILIES:
            a, b = [next(r for r in ft if r['model'] == model and r['condition'] == c and r['state'] == 'all' and r['scope'] == 'all_crops' and r['target_family'] == fam and r['threshold'] == .7) for c in CONDITIONS]
            lines.append(f"| {model} | {fam} | {a['family_retained_lower']} -> {b['family_retained_lower']} | {a['family_total']} | {a['unrelated_retained_upper']} -> {b['unrelated_retained_upper']} | {a['unrelated_total']} |")
    lines.extend(['', '## Insufficient Evidence', '| Model | Condition | Family mean P(C) | Family C /100 | Unrelated mean P(C) | Unrelated C /384 |', '|---|---|---:|---:|---:|---:|'])
    for r in metrics:
        if r['scope'] == 'all_crops' and r['state'] == 'all':
            values = [span(r[k + '_mean_p_C_lower'], r[k + '_mean_p_C_upper'], 5) for k in ('family', 'unrelated')]
            lines.append(f"| {r['model']} | {r['condition']} | {values[0]} | {r['family_C_count']} | {values[1]} | {r['unrelated_C_count']} |")
    lines.extend(['', '## Saved Evidence',
        '- all_scores_comparison.csv/json: all 2,904 old/new comparisons with P(A/B/C), log probabilities, labels, crop hashes, raw-response paths and evaluation identities.',
        '- paired_scores.csv/json: 1,452 matched comparisons and score differences.',
        '- metrics.csv/json and per_family_metrics.csv/json: AUC, macro-family AUC, mean P(C), C frequency, per-scene and assisted-crop sensitivity.',
        '- baseline_comparison.csv/json: paired aggregate AUC differences and changes in family/unrelated C counts for each scene and both crop scopes.',
        '- threshold_sweep.csv/json, per_family_thresholds.csv/json: family, unrelated and own-object retained counts and rates.',
        '- matched_false_match_operating_points.csv/json: thresholds and retention at common false-match budgets.',
        '- known_failures.csv/json, known_failure_pairs.csv/json: requested connector/gear-nut confusion directions.',
        '- top_unrelated_pairs.csv/json, ranked_unrelated_pairs.csv/json: top 20 and complete false-match rankings per model/condition.',
        '- repeated_family_query_consistency.csv/json: score variation among identical size-family prompts in both conditions.',
        '- Each model directory contains 484 raw responses, runtime provenance, four annotated heatmaps and a four-page PDF.',
        '- selected_prompts.json and prompt_audit.json: exact model-visible text and category-name leakage checks.',
        '- results.zip contains the analysis, new and paired baseline raw responses, 44 input crops and source helpers under their workspace-relative paths.', '',
        'Offline verification after extraction: `python features-only-ablation/verify_results.py`. This validates the bundled raw data without models, a network connection, or the entire older workspace.', '',
        '## Interpretation Limits',
        'These scenes were already used to develop the prompts. AUC and optimized operating points are descriptive development results, not held-out estimates. Scores are not calibrated probabilities of physical correctness. Own-object retention does not prove exact-size recognition. No threshold is deployed, and no existing condition is replaced.', '',
        '![Threshold comparison](threshold_comparison.png)', '',
        '![Family retention versus unrelated acceptance](family_retention_vs_false_matches.png)'])
    (HERE / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
