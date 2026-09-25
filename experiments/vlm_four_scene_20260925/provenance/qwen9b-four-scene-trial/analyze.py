"""Audit raw probabilities, summarize gates, and draw four-scene heatmaps."""
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from run_trial import HERE, ROOT, STATES, read, save, table, collect, digest

LABELS = ('Baseline', 'Scene 101', 'Scene 202', 'Scene 303')
CHECKS = (.6, .65, .7, .8, .9, .95, .97, .98, .99)


def summarize(rows, dest, names):
    cases = []
    for target, name in names.items():
        for state in STATES:
            group = [r for r in rows if r['target'] == target and r['state'] == state]
            own = next(r for r in group if r['own_object_match'])
            wrong = max((r for r in group if not r['same_family']), key=lambda r: r['p_A'])
            winner = max(group, key=lambda r: r['p_A'])
            cases.append(dict(target=target, target_name=name, scene=state,
                              intended_p_A=own['p_A'], intended_answer=own['answer'],
                              intended_assisted_fallback=own['assisted_fallback'],
                              highest_unrelated_p_A=wrong['p_A'],
                              highest_unrelated_object=wrong['posthoc_identity'],
                              margin_to_highest_unrelated=own['p_A']-wrong['p_A'],
                              winner=winner['posthoc_identity'], winner_p_A=winner['p_A'],
                              winner_same_family=winner['same_family'],
                              intended_response=own['response_file'],
                              highest_unrelated_response=wrong['response_file']))
    table(dest / 'per_object_per_scene', cases)
    summary = []
    for target, name in names.items():
        group = [r for r in cases if r['target'] == target]
        summary.append(dict(target=target, target_name=name,
                            mean_intended_p_A=float(np.mean([r['intended_p_A'] for r in group])),
                            minimum_intended_p_A=min(r['intended_p_A'] for r in group),
                            maximum_unrelated_p_A=max(r['highest_unrelated_p_A'] for r in group),
                            family_winner_cases=sum(r['winner_same_family'] for r in group),
                            **{f'intended_retained_{t}':sum(r['intended_p_A'] >= t for r in group)
                               for t in CHECKS}))
    table(dest / 'object_summary', summary)
    thresholds = []
    for scope in ('all_crops', 'autonomous_depth_only'):
        subset = [r for r in rows if scope == 'all_crops' or not r['assisted_fallback']]
        for state in (*STATES, 'all'):
            group = [r for r in subset if state == 'all' or r['state'] == state]
            own = [r for r in group if r['own_object_match']]
            compatible = [r for r in group if r['same_family']]
            unrelated = [r for r in group if not r['same_family']]
            for threshold in np.arange(0, 1.0001, .01):
                threshold = round(float(threshold), 2)
                wrong = [r for r in unrelated if r['p_A'] >= threshold]
                thresholds.append(dict(model=rows[0]['model'], scope=scope, scene=state, threshold=threshold,
                                       intended_retained=sum(r['p_A'] >= threshold for r in own),
                                       intended_total=len(own),
                                       family_retained=sum(r['p_A'] >= threshold for r in compatible),
                                       family_total=len(compatible),
                                       unrelated_retained=len(wrong), unrelated_total=len(unrelated),
                                       cases_with_unrelated_retained=len({(r['state'],r['target']) for r in wrong})))
    table(dest / 'threshold_sweep', thresholds)
    table(dest / 'threshold_check', [r for r in thresholds if r['threshold'] in CHECKS])
    table(dest / 'unrelated_passes_0_7', sorted([r for r in rows if not r['same_family'] and r['p_A'] >= .7], key=lambda r:-r['p_A']))
    return cases, thresholds


def heatmaps(rows, dest, names, model):
    order = list(names)
    matrices = {}
    for state in STATES:
        group = [r for r in rows if r['state'] == state]
        matrix = np.full((11, 11), np.nan)
        for r in group:
            matrix[order.index(r['target']),order.index(r['posthoc_identity'])] = r['p_A']
        assert np.isfinite(matrix).all()
        matrices[state] = matrix
        table(dest / (state + '_p_A_matrix'), [dict(target=target, **{identity:float(matrix[i,j])
              for j,identity in enumerate(order)}) for i,target in enumerate(order)])

    def draw(ax, state, label, compact=False):
        matrix = matrices[state]
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
        observed = [names[t] + (' *' if t == 'USB_Male' and state in ('state_101','state_303') else '') for t in order]
        ax.set_xticks(range(11), observed, rotation=55, ha='right', fontsize=9 if compact else 11)
        ax.set_yticks(range(11), list(names.values()), fontsize=9 if compact else 11)
        for i in range(11):
            for j in range(11):
                value = matrix[i,j]
                ax.text(j,i,f'{value:.3f}',ha='center',va='center',fontsize=8 if compact else 10,
                        color='white' if value >= .60 else '#202020')
        ax.set_title(f'{label}: {model} raw P(A)', loc='left', pad=14, fontsize=15 if compact else 20)
        ax.set_xlabel('Observed object crop (identity used only for evaluation)', labelpad=12)
        ax.set_ylabel('Target descriptions', labelpad=12)
        return im

    with PdfPages(dest / 'four_scene_heatmaps.pdf') as pdf:
        for state, label in zip(STATES,LABELS):
            fig,ax=plt.subplots(figsize=(15.5,11))
            im=draw(ax,state,label)
            fig.colorbar(im,ax=ax,fraction=.026,pad=.023,label='Raw P(A)')
            fig.subplots_adjust(left=.20,right=.95,top=.90,bottom=.25)
            fig.text(.025,.029,'All target/crop pairs; no score cutoff. M8 and square pins excluded. P(A) is not calibrated identity correctness.',fontsize=10)
            fig.text(.025,.009,'* USB in scenes 101/303: manually verified crop recovered from a rejected depth cluster; not autonomous localization.',fontsize=9)
            fig.savefig(dest / (state+'_p_A_matrix.png'),dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(23,18))
    for ax,state,label in zip(axes.flat,STATES,LABELS):
        im=draw(ax,state,label,True)
    fig.subplots_adjust(left=.12,right=.93,top=.94,bottom=.18,wspace=.39,hspace=.70)
    fig.colorbar(im,cax=fig.add_axes([.95,.33,.012,.4]),label='Raw P(A)')
    fig.suptitle(f'{model} | Every target against every crop | Four fixed scenes',x=.025,ha='left',fontsize=23,y=.99)
    fig.text(.025,.025,'Same 0-1 scale. No cutoff. * Assisted USB crop. Same-family matches are plausible alternatives, not exact instance identification.',fontsize=12)
    fig.savefig(dest / 'four_scene_overview.png',dpi=180)
    plt.close(fig)


def main():
    manifest=read(HERE / 'manifest.json')
    names=manifest['names']
    models=['qwen3.5:9b']
    if len(list((HERE / '4b/responses').rglob('*.json'))) == 484:
        models.insert(0,'qwen3.5:4b')
    all_rows=[]
    checks=[]
    for model in models:
        dest=HERE / model.split(':')[-1]
        rows=collect(model)
        cases,thresholds=summarize(rows,dest,names)
        heatmaps(rows,dest,names,model)
        all_rows.extend(rows)
        checks.extend(r for r in thresholds if r['scene']=='all' and r['threshold'] in CHECKS)
    table(HERE / 'model_threshold_comparison',checks)
    if len(models)==2:
        index={(r['state'],r['target'],r['posthoc_identity']):r for r in all_rows if r['model']==models[0]}
        pairs=[]
        for r in all_rows:
            if r['model']!=models[1]:
                continue
            old=index[(r['state'],r['target'],r['posthoc_identity'])]
            assert old['crop_sha256']==r['crop_sha256']
            a=read(ROOT / old['response_file'])['request']
            b=read(ROOT / r['response_file'])['request']
            assert a['messages']==b['messages'] and a['options']==b['options']
            pairs.append(dict(state=r['state'],target=r['target'],posthoc_identity=r['posthoc_identity'],
                              own_object_match=r['own_object_match'],same_family=r['same_family'],
                              assisted_fallback=r['assisted_fallback'],p_A_4b=old['p_A'],p_A_9b=r['p_A'],
                              difference_9b_minus_4b=r['p_A']-old['p_A']))
        table(HERE / 'paired_model_scores',pairs)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for model,color in zip(models,('#238a8d','#b34b27')):
            sweep=read(HERE / model.split(':')[-1] / 'threshold_sweep.json')
            sweep=[r for r in sweep if r['scope']=='all_crops' and r['scene']=='all']
            x=[r['threshold'] for r in sweep]
            axes[0].plot(x,[r['intended_retained']/r['intended_total'] for r in sweep],label=model,color=color,lw=2)
            axes[1].plot(x,[r['unrelated_retained']/r['unrelated_total'] for r in sweep],label=model,color=color,lw=2)
        for ax,title in zip(axes,('Intended object retention (44 cases)','Unrelated object acceptance (384 pairs)')):
            ax.set(title=title,xlabel='P(A) cutoff',ylabel='Fraction',xlim=(0,1),ylim=(0,1.02))
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle('Matched crops and prompts; exploratory thresholds, not held-out validation',fontsize=13)
        fig.tight_layout()
        fig.savefig(HERE / 'threshold_comparison.png',dpi=180)
        plt.close(fig)
        ranking=[]
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for model,color in zip(models,('#238a8d','#b34b27')):
            rows=[r for r in all_rows if r['model']==model]
            negatives=np.array([r['p_A'] for r in rows if not r['same_family']])
            own=np.array([r['p_A'] for r in rows if r['own_object_match']])
            positive=np.array([r['p_A'] for r in rows if r['same_family']])
            cutoffs=sorted({r['p_A'] for r in rows}|{1.0,0.0},reverse=True)
            fpr=[float(np.mean(negatives>=t)) for t in cutoffs]
            for ax,values in zip(axes,(own,positive)):
                ax.plot(fpr,[float(np.mean(values>=t)) for t in cutoffs],label=model,color=color,lw=2)
            delta=positive[:,None]-negatives[None,:]
            ranking.append(dict(model=model,family_vs_unrelated_auc=float(np.mean((delta>0)+.5*(delta==0))),
                                intended_mean_p_A=float(np.mean(own)),
                                total_inference_seconds=sum(r['elapsed_seconds'] for r in rows)))
        for ax,title in zip(axes,('Intended object retention','All same-family match retention')):
            ax.set(title=title,xlabel='Unrelated acceptance fraction',ylabel='Retained fraction',xlim=(0,.2),ylim=(0,1.02))
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle('Compare at equal false-match rates, not equal numerical cutoffs',fontsize=13)
        fig.tight_layout()
        fig.savefig(HERE / 'retention_vs_false_matches.png',dpi=180)
        plt.close(fig)
        table(HERE / 'ranking_metrics',ranking)
    validation=dict(models=models,comparisons_per_model=484,unique_crops=44,
                    intended_cases_per_model=44,family_pairs_per_model=100,unrelated_pairs_per_model=384,
                    assisted_crops=2,missing_label_probabilities=0,
                    hashes_verified=all(digest(ROOT / p)==sha for p,sha in manifest['immutable_inputs'].items()),
                    matched_model_inputs_verified=len(models)==2,
                    raw_response_hashes={r['response_file']:digest(ROOT / r['response_file']) for r in all_rows})
    assert validation['hashes_verified']
    save(HERE / 'validation.json',validation)
    text=['# Four-Scene Local Qwen Classification Trial','',
          '11 selected target objects, each compared independently with all 11 retained object crops in each of four fixed scenes. No new scene views. M8 and all square pins excluded from targets and candidate analysis.',
          '', '## Reproducibility',
          'Exact previous selected descriptions and prompts are frozen in selected_prompts.json. The USB query retains its existing shadow-aware paragraph. No description edits or inference score threshold. One image and all saved target descriptions per request. Ground-truth labels are not in model inputs.',
          'Qwen3.5 9B Q4_K_M is saved on D: via Ollama. Runtime versions, model digests, inference options, raw requests/responses and log probabilities are saved under each model folder. Thinking disabled; one greedy output token A/B/C.',
          '', '## USB Fallback',
          'Depth accepted the cable in baseline and scene 202. In scenes 101/303 its complete cluster was rejected by the fixed aspect-ratio filter. We recovered the existing native RGB rectangles, visually verified them, and verified pixel-for-pixel equality with the corresponding scene crop. Simulation reference boxes assisted identification of those rejected clusters. This is explicitly assisted localization, not SAM or autonomous success. No inpainting, new view, or image enhancement.',
          '', '## Score Meaning',
          'P(A) = exp(raw model log probability of token A). It is not renormalized over A/B/C and is not a calibrated probability of correct object identity. A means match supported, B mismatch supported, C insufficient evidence. All three probabilities and raw logs are retained.',
          'Gear sizes, hex-nut sizes, and round-pin sizes form separate product families. Same-family alternatives are plausible matches, not proof of exact size identity. An unrelated acceptance means a crop outside the target family passes the cutoff.',
          '', '## Threshold Results',
          '| Model | Cutoff | Intended retained / 44 | Unrelated retained / 384 |',
          '|---|---:|---:|---:|']
    for r in checks:
        if r['scope']=='all_crops':
            text.append(f"| {r['model']} | {r['threshold']:.2f} | {r['intended_retained']} | {r['unrelated_retained']} |")
    text.extend(['', 'Threshold sweeps include both all crops and autonomous-depth-only results; the latter excludes every pair involving either assisted USB crop. The four reused development scenes are not an independent test set. Do not select a final deployment threshold from these data alone.',
                 '', 'If both model folders are present, the 4B control was freshly rerun on the same 44 image files and exact prompts/options as 9B. paired_model_scores.csv verifies the comparison. Earlier 4B chart data mixed scene revisions and are not used as this matched control.',
                 '', '## Files',
                 '- 9b/four_scene_overview.png and four_scene_heatmaps.pdf: all 484 values across four scenes.',
                 '- 9b/*_p_A_matrix.png/csv/json: individual scene matrices.',
                 '- Each model: all_scores, per_object_per_scene, object_summary, threshold_sweep, threshold_check, unrelated_passes_0_7 (CSV and JSON).',
                 '- model_threshold_comparison.csv/json: selected cutoffs, with and without assisted crops.',
                 '- paired_model_scores.csv/json and threshold_comparison.png: matched model comparison when both runs are complete.',
                 '- retention_vs_false_matches.png and ranking_metrics.csv/json: threshold-independent comparison; same-family alternatives count as compatible.',
                 '- manifest.json, selected_prompts.json, validation.json and responses: full audit trail.',
                 '- results.zip includes all data, charts, raw responses and the 44 input crops; no model weights.'])
    (HERE / 'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    with zipfile.ZipFile(HERE / 'results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.png','.pdf','.md','.py') and '__pycache__' not in path.parts:
                archive.write(path,str(path.relative_to(HERE)))
        for state,crops in manifest['crops'].items():
            for crop in crops:
                archive.write(ROOT / crop['crop_file'],f"input_crops/{state}/{crop['identity']}.png")
    with zipfile.ZipFile(HERE / 'results.zip') as archive:
        assert archive.testzip() is None
    print(json.dumps([r for r in checks if r['scope']=='all_crops'],indent=2))


if __name__=='__main__':
    main()
