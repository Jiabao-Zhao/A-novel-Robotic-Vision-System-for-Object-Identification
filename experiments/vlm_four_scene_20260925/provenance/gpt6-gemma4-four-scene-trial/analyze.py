"""Four-model gate comparison, preserving missing probabilities as intervals."""
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from run_models import HERE, ROOT, SOURCE, FOLDERS, read, save, table, collect, digest

STATES = ('original','state_101','state_202','state_303')
LABELS = ('Baseline','Scene 101','Scene 202','Scene 303')
CHECKS = (.6,.65,.7,.8,.9,.95,.98,.99,.995,.999,.9999,.99999,.999999)


def interval(row):
    if row['p_A'] is not None:
        return row['p_A'],row['p_A']
    return row['p_A_lower'],row['p_A_upper']


def count_pass(rows, cutoff):
    minimum = sum(interval(r)[0]>=cutoff for r in rows)
    maximum = sum(interval(r)[1]>=cutoff for r in rows)
    return minimum,maximum


def cell(row):
    if row['p_A'] is not None:
        return f"{row['p_A']:.3f}"
    upper=math.ceil(row['p_A_upper']*1000)/1000
    return f"<={upper:.3f}"


def summarize(rows,dest,names):
    cases=[]
    for target,name in names.items():
        for state in STATES:
            group=[r for r in rows if r['state']==state and r['target']==target]
            own=next(r for r in group if r['own_object_match'])
            wrong=[r for r in group if not r['same_family']]
            best_wrong=max(wrong,key=lambda r:interval(r)[1])
            cases.append(dict(model=rows[0]['model'],target=target,target_name=name,state=state,
                              intended_p_A=own['p_A'],intended_p_A_lower=interval(own)[0],
                              intended_p_A_upper=interval(own)[1],intended_answer=own['answer'],
                              assisted_intended_crop=own['assisted_fallback'],
                              highest_unrelated_lower=max(interval(r)[0] for r in wrong),
                              highest_unrelated_upper=max(interval(r)[1] for r in wrong),
                              highest_unrelated_upper_object=best_wrong['posthoc_identity'],
                              intended_response=own['response_file']))
    table(dest/'per_object_per_scene',cases)
    objects=[]
    for target,name in names.items():
        group=[r for r in cases if r['target']==target]
        objects.append(dict(target=target,target_name=name,
                            mean_intended_lower=float(np.mean([r['intended_p_A_lower'] for r in group])),
                            mean_intended_upper=float(np.mean([r['intended_p_A_upper'] for r in group])),
                            max_unrelated_upper=max(r['highest_unrelated_upper'] for r in group),
                            **{f'intended_retained_min_{c}':sum(r['intended_p_A_lower']>=c for r in group) for c in CHECKS},
                            **{f'intended_retained_max_{c}':sum(r['intended_p_A_upper']>=c for r in group) for c in CHECKS}))
    table(dest/'object_summary',objects)
    sweep=[]
    cutoffs=sorted(set(round(float(t),2) for t in np.arange(0,1.001,.01))|set(CHECKS))
    for scope in ('all_crops','autonomous_depth_only'):
        eligible=[r for r in rows if scope=='all_crops' or not r['assisted_fallback']]
        for state in (*STATES,'all'):
            group=[r for r in eligible if state=='all' or r['state']==state]
            categories=dict(intended=[r for r in group if r['own_object_match']],
                            family=[r for r in group if r['same_family']],
                            unrelated=[r for r in group if not r['same_family']])
            for cutoff in cutoffs:
                record=dict(model=rows[0]['model'],scope=scope,state=state,cutoff=cutoff)
                for key,values in categories.items():
                    lower,upper=count_pass(values,cutoff)
                    record.update({key+'_retained_min':lower,key+'_retained_max':upper,
                                   key+'_total':len(values),key+'_gate_unknown':upper-lower})
                sweep.append(record)
    table(dest/'threshold_sweep',sweep)
    table(dest/'threshold_check',[r for r in sweep if r['cutoff'] in CHECKS])
    false=[r for r in rows if not r['same_family'] and interval(r)[1]>=.7]
    table(dest/'unrelated_passes_or_possible_0_7',sorted(false,key=lambda r:-interval(r)[1]))
    return sweep


def heatmaps(rows,dest,names):
    order=list(names)
    index={(r['state'],r['target'],r['posthoc_identity']):r for r in rows}
    assert len(index)==484
    for state in STATES:
        table(dest/(state+'_p_A_matrix'),[dict(target=t,**{c:cell(index[state,t,c]) for c in order}) for t in order])

    def draw(ax,state,label,compact=False):
        # Missing exact values are colored by their explicitly marked upper bound.
        matrix=np.array([[interval(index[state,t,c])[1] for c in order] for t in order])
        im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
        observed=[names[c]+(' *' if c=='USB_Male' and state in ('state_101','state_303') else '') for c in order]
        ax.set_xticks(range(11),observed,rotation=55,ha='right',fontsize=9 if compact else 11)
        ax.set_yticks(range(11),list(names.values()),fontsize=9 if compact else 11)
        for i,t in enumerate(order):
            for j,c in enumerate(order):
                r=index[state,t,c]
                ax.text(j,i,cell(r),ha='center',va='center',fontsize=6.5 if compact else 9,
                        color='white' if matrix[i,j]>=.6 else '#202020')
        ax.set_title(f"{label}: {rows[0]['model']} P(A)",loc='left',pad=14,fontsize=15 if compact else 20)
        ax.set_xlabel('Observed object crop (identity used only for evaluation)',labelpad=12)
        ax.set_ylabel('Target descriptions',labelpad=12)
        return im

    with PdfPages(dest/'four_scene_heatmaps.pdf') as pdf:
        for state,label in zip(STATES,LABELS):
            fig,ax=plt.subplots(figsize=(15.5,11))
            im=draw(ax,state,label)
            fig.colorbar(im,ax=ax,fraction=.026,pad=.023,label='P(A); upper bound when marked <=')
            fig.subplots_adjust(left=.20,right=.95,top=.90,bottom=.25)
            fig.text(.025,.03,'<= marks an upper bound when the API omitted exact P(A); not a zero or an estimated exact score. No score cutoff applied.',fontsize=9)
            fig.text(.025,.01,'* Assisted USB crop. M8 and square pins excluded. Raw P(A) is not calibrated correctness; all four models use identical crop files and prompts.',fontsize=9)
            fig.savefig(dest/(state+'_p_A_matrix.png'),dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(23,18))
    for ax,state,label in zip(axes.flat,STATES,LABELS):
        im=draw(ax,state,label,True)
    fig.subplots_adjust(left=.12,right=.93,top=.94,bottom=.18,wspace=.39,hspace=.70)
    fig.colorbar(im,cax=fig.add_axes([.95,.33,.012,.4]),label='P(A); upper bound when marked <=')
    fig.suptitle(f"{rows[0]['model']} | All four scenes | Every target against every crop",x=.025,ha='left',fontsize=23,y=.99)
    fig.text(.025,.025,'<= cells are upper bounds, rounded upward. No zero imputation. * Assisted USB crop. No threshold filtering.',fontsize=12)
    fig.savefig(dest/'four_scene_overview.png',dpi=180)
    plt.close(fig)


def auc_bounds(rows):
    positive=[interval(r) for r in rows if r['same_family']]
    negative=[interval(r) for r in rows if not r['same_family']]
    lower=upper=0
    for pl,pu in positive:
        for nl,nu in negative:
            if pl==pu and nl==nu:
                value=1 if pl>nl else .5 if pl==nl else 0
                lower+=value
                upper+=value
            else:
                lower+=int(pl>nu)
                upper+=int(pu>=nl)
    total=len(positive)*len(negative)
    return lower/total,upper/total


def main():
    manifest=read(HERE/'manifest.json')
    names=manifest['names']
    models={}
    for tag in ('4b','9b'):
        rows=read(SOURCE/tag/'all_scores.json')
        models[rows[0]['model']]=rows
    for kind,folder in FOLDERS.items():
        files=list((HERE/folder/'responses').rglob('*.json'))
        files=[p for p in files if not p.stem.endswith('_error')]
        if len(files)==484:
            rows=collect(kind)
            models[rows[0]['model']]=rows
            heatmaps(rows,HERE/folder,names)
    sweeps=[]
    metrics=[]
    all_rows=[]
    for model,rows in models.items():
        dest=HERE/(FOLDERS['gpt'] if model=='gpt-6-sol' else FOLDERS['local'] if model=='gemma4:12b' else model.replace(':','-'))
        dest.mkdir(exist_ok=True)
        table(dest/'all_scores',rows)
        sweeps.extend(summarize(rows,dest,names))
        lower,upper=auc_bounds(rows)
        metrics.append(dict(model=model,family_vs_unrelated_auc_lower=lower,family_vs_unrelated_auc_upper=upper,
                            missing_exact_p_A=sum(r['p_A'] is None for r in rows),
                            mean_intended_p_A_lower=float(np.mean([interval(r)[0] for r in rows if r['own_object_match']])),
                            mean_intended_p_A_upper=float(np.mean([interval(r)[1] for r in rows if r['own_object_match']])),
                            summed_request_seconds=sum(r['elapsed_seconds'] for r in rows)))
        all_rows.extend(rows)
    table(HERE/'ranking_metrics',metrics)
    checks=[r for r in sweeps if r['scope']=='all_crops' and r['state']=='all' and r['cutoff'] in CHECKS]
    table(HERE/'model_threshold_comparison',checks)
    table(HERE/'all_model_scores',all_rows)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    colors=('#238a8d','#b34b27','#773a94','#4d7c20')
    for (model,rows),color in zip(models.items(),colors):
        group=[r for r in sweeps if r['model']==model and r['scope']=='all_crops' and r['state']=='all']
        x=[r['cutoff'] for r in group]
        for ax,key in zip(axes,('intended','unrelated')):
            low=[r[key+'_retained_min']/r[key+'_total'] for r in group]
            high=[r[key+'_retained_max']/r[key+'_total'] for r in group]
            ax.plot(x,low,label=model,color=color,lw=2)
            ax.fill_between(x,low,high,color=color,alpha=.15)
    for ax,title in zip(axes,('Intended object retention (44 cases)','Unrelated acceptance (384 pairs)')):
        ax.set(title=title,xlabel='P(A) cutoff',ylabel='Fraction',xlim=(0,1),ylim=(0,1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('Same crops and prompts; shading = uncertainty from omitted label probabilities',fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE/'threshold_comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for (model,rows),color in zip(models.items(),colors):
        thresholds=sorted({value for r in rows for value in interval(r)}|{0.0,1.0},reverse=True)
        negatives=[r for r in rows if not r['same_family']]
        fpr=[count_pass(negatives,t)[1]/len(negatives) for t in thresholds]
        for ax,key in zip(axes,('own_object_match','same_family')):
            positive=[r for r in rows if r[key]]
            lower=[count_pass(positive,t)[0]/len(positive) for t in thresholds]
            ax.plot(fpr,lower,label=model,color=color,lw=2)
    for ax,title in zip(axes,('Intended object retention','All same-family match retention')):
        ax.set(title=title,xlabel='Unrelated acceptance fraction (upper bound)',
               ylabel='Retained fraction (lower bound)',xlim=(0,.2),ylim=(0,1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('Equal-error-rate comparison; conservative bounds where exact probabilities are missing',fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE/'retention_vs_false_matches.png',dpi=180)
    plt.close(fig)
    if 'gemma4:12b' in models:
        storage_path=Path('D:/AI/models/manifests/registry.ollama.ai/library/gemma4/12b')
        stored=read(storage_path)
        layers=[]
        for entry in [stored['config'],*stored['layers']]:
            blob=Path('D:/AI/models/blobs')/entry['digest'].replace(':','-')
            assert blob.is_file() and blob.stat().st_size==entry['size']
            layers.append(dict(path=str(blob),bytes=blob.stat().st_size,digest=entry['digest']))
        save(HERE/'gemma4-12b/storage.json',dict(manifest=str(storage_path),layers=layers,
                                             verified='All manifest-referenced files exist on D: at expected sizes; download hashes verified by Ollama.'))
    reference={(r['state'],r['target'],r['posthoc_identity']):r['crop_sha256'] for r in next(iter(models.values()))}
    for rows in models.values():
        assert len(rows)==484
        for r in rows:
            assert r['crop_sha256']==reference[r['state'],r['target'],r['posthoc_identity']]
    hashes={r['response_file']:digest(ROOT/r['response_file']) for r in all_rows}
    validation=dict(complete_models=list(models),comparisons_per_model=484,unique_crops=44,
                    exact_prompt_text_reused=True,crop_bytes_matched=True,assisted_USB_crops=2,
                    source_inputs_unchanged=all(digest(ROOT/p)==sha for p,sha in manifest['immutable_inputs'].items()),
                    source_manifest_unchanged=digest(SOURCE/'manifest.json')==manifest['source_manifest_sha256'],
                    raw_response_hashes=hashes)
    assert validation['source_inputs_unchanged'] and validation['source_manifest_unchanged']
    save(HERE/'validation.json',validation)
    lines=['# GPT-6 Sol and Gemma 4 12B: Four-Scene Classification','',
           'The frozen 11-target, 11-crop, four-scene dataset is unchanged. Each model has 484 independent comparisons. The previous Qwen3.5-4B and 9B runs are reused as controls; no per-scene best-score selection. M8 and square pins remain excluded from both targets and candidate analysis.',
           '', '## Method',
           'All models receive exactly the same system/user prompt text and crop bytes. They receive no observation identity, crop ID, ROI coordinates or ground-truth metadata. All saved target descriptions are included together. No new render views, prompt changes, inference score thresholds, or forced workspace winner.',
           'GPT-6 Sol uses OpenAI Responses, standard service, reasoning none, temperature 0, high image detail, max output 16 and top logprobs 20. Four requests run concurrently. The user explicitly approved uploading these simulation crops. Requests use store=false.',
           'Gemma 4 12B Q4_K_M is installed in D:/AI/models and runs through the existing D:/AI/Ollama runtime, with GPU/CPU offload. Native image preprocessing and tokenizers differ across model families. Gemma permits 16 generated tokens for end/control tags; only its single A/B/C answer token is scored. Reasoning is disabled.',
           'P(A) is exp(raw token log probability), not a written confidence and not normalized over A/B/C. It is not calibrated identity correctness. GPT may omit exact P(A); those entries remain null with conservative lower/upper bounds. A 0.0001 allowance covers rounded API log probabilities. Heatmap <= labels show upper bounds rounded upward, never zero imputation.',
           '', '## USB Recovery and Evaluation',
           'Baseline/202 USB rectangles were accepted by depth localization. Scenes 101/303 use visually verified recovered depth-cluster rectangles rejected by the aspect-ratio filter. These are assisted crops, not SAM or autonomous localization. Their pixels match the original scene rectangles exactly. Separate autonomous-depth-only statistics exclude all pairs involving those two crops.',
           'Same-family gear, nut and round-pin sizes count as plausible alternatives, not exact instance identification. Unrelated means outside that family. Four development scenes are not held-out validation. No final model or cutoff is automatically deployed.',
           '', '## Threshold Comparison',
           '| Model | Cutoff | Intended retained / 44 | Unrelated accepted / 384 |',
           '|---|---:|---:|---:|']
    def count_text(r,key):
        lo,hi=r[key+'_retained_min'],r[key+'_retained_max']
        return str(lo) if lo==hi else f'{lo}-{hi}'
    for r in checks:
        if r['cutoff'] in (.6,.65,.7,.9,.95,.98):
            lines.append(f"| {r['model']} | {r['cutoff']:.2f} | {count_text(r,'intended')} | {count_text(r,'unrelated')} |")
    if (HERE/'gpt6-sol/usage_total.json').exists():
        u=read(HERE/'gpt6-sol/usage_total.json')
        lines.extend(['','## API Usage',f"{u['input_tokens']:,} input tokens; {u['output_tokens']:,} output tokens; {u['reasoning_tokens']:,} reasoning tokens. Estimated standard API cost: ${u['estimated_standard_cost_usd']:.4f}, based on saved usage, not a billing invoice.",
                      'Rates: $2/M uncached input, $0.20/M cached input, $2.50/M cache writes, $10/M output. https://developers.openai.com/api/docs/models/gpt-6-sol'])
    lines.extend(['','## Files',
                  '- Each new model: four_scene_heatmaps.pdf, four_scene_overview.png, individual scene PNG/CSV/JSON matrices.',
                  '- Each model: all_scores, per_object_per_scene, object_summary, threshold_sweep, threshold_check, and unrelated_passes_or_possible_0_7 CSV/JSON.',
                  '- all_model_scores.csv/json and model_threshold_comparison.csv/json: combined comparisons.',
                  '- retention_vs_false_matches.png: comparison at equal unrelated-acceptance rates, using conservative bounds.',
                  '- ranking_metrics.csv/json: family-versus-unrelated AUC intervals, not exact-size correctness.',
                  '- selected_prompts.json, manifest.json, runtime.json, validation.json, responses/: complete provenance.',
                  '- results.zip: scripts, data, charts, raw responses for all complete models, and the 44 input crops. No weights included.'])
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(HERE/'results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.png','.pdf','.md','.py') and '__pycache__' not in path.parts:
                archive.write(path,str(path.relative_to(HERE)))
        for path in hashes:
            if not (ROOT/path).is_relative_to(HERE):
                archive.write(ROOT/path,'prior_evidence/'+path)
        for state,crops in manifest['crops'].items():
            for crop in crops:
                archive.write(ROOT/crop['crop_file'],f"input_crops/{state}/{crop['identity']}.png")
    with zipfile.ZipFile(HERE/'results.zip') as archive:
        assert archive.testzip() is None
    print('Completed models:',list(models))
    print('Threshold .7:',[r for r in checks if r['cutoff']==.7])


if __name__=='__main__':
    main()
