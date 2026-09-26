"""Paired family-discrimination analysis; no inference or baseline modification."""
import math
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from run_trial import HERE, ROOT, MODELS, read, save, table, digest, collect, family, validate_inputs

STATES=('original','state_101','state_202','state_303')
CHECKS=(.4,.5,.6,.65,.7,.8,.9,.95,.97,.98,.99,.995,.999,.9999,.99999,.999999)
FAMILIES=('gear','hex_nut','round_pin','Waterproof_Male','USB_Male','DSUB_Male')


def interval(row,label='A'):
    p=row['p_'+label]
    return (p,p) if p is not None else (row['p_'+label+'_lower'],row['p_'+label+'_upper'])


def count(rows,t):
    return sum(interval(r)[0]>=t for r in rows),sum(interval(r)[1]>=t for r in rows)


def auc(rows):
    positive=[interval(r) for r in rows if r['same_family']]
    negative=[interval(r) for r in rows if not r['same_family']]
    assert positive and negative
    lo=hi=0
    for pl,pu in positive:
        for nl,nu in negative:
            if pl==pu and nl==nu:
                v=1 if pl>nl else .5 if pl==nl else 0
                lo+=v; hi+=v
            else:
                lo+=int(pl>nu); hi+=int(pu>=nl)
    total=len(positive)*len(negative)
    return lo/total,hi/total


def summarize(rows,condition,model,scope,state,target_family):
    a,b=auc(rows)
    out=dict(condition=condition,model=model,scope=scope,state=state,target_family=target_family,
        comparisons=len(rows),family_pairs=sum(r['same_family'] for r in rows),
        unrelated_pairs=sum(not r['same_family'] for r in rows),
        family_vs_unrelated_auc_lower=a,family_vs_unrelated_auc_upper=b,
        missing_exact_p_A=sum(r['p_A'] is None for r in rows))
    for name,group in [('all',rows),('family',[r for r in rows if r['same_family']]),
                       ('unrelated',[r for r in rows if not r['same_family']]),
                       ('own',[r for r in rows if r['own_object_match']])]:
        out[name+'_count']=len(group)
        out[name+'_C_count']=sum(r['answer']=='C' for r in group)
        out[name+'_C_frequency']=out[name+'_C_count']/len(group) if group else None
        for label in 'AC':
            for i,side in enumerate(('lower','upper')):
                out[f'{name}_mean_p_{label}_{side}']=float(np.mean([interval(r,label)[i] for r in group])) if group else None
    return out


def thresholds(rows,condition,model,scope,state,target_family):
    result=[]
    for t in sorted(set(round(i/100,2) for i in range(101))|set(CHECKS)):
        out=dict(condition=condition,model=model,scope=scope,state=state,target_family=target_family,threshold=t)
        for name,group in [('family',[r for r in rows if r['same_family']]),
                           ('unrelated',[r for r in rows if not r['same_family']]),
                           ('own',[r for r in rows if r['own_object_match']])]:
            lo,hi=count(group,t)
            out.update({name+'_total':len(group),name+'_retained_lower':lo,name+'_retained_upper':hi,
                        name+'_rate_lower':lo/len(group) if group else None,
                        name+'_rate_upper':hi/len(group) if group else None})
        result.append(out)
    return result


def heatmaps(rows,model,names):
    order=list(names)
    dest=HERE/MODELS[model]
    idx={(r['state'],r['target'],r['posthoc_identity']):r for r in rows}
    with PdfPages(dest/'four_scene_heatmaps.pdf') as pdf:
        for state in STATES:
            matrix=np.array([[interval(idx[state,t,c])[1] for c in order] for t in order])
            records=[]
            fig,ax=plt.subplots(figsize=(15.5,11))
            im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
            ax.set_xticks(range(11),[names[c]+(' *' if state in ('state_101','state_303') and c=='USB_Male' else '') for c in order],rotation=55,ha='right',fontsize=10)
            ax.set_yticks(range(11),list(names.values()),fontsize=10)
            for i,t in enumerate(order):
                record={'target':t}
                for j,c in enumerate(order):
                    r=idx[state,t,c]
                    value=f"{r['p_A']:.3f}" if r['p_A'] is not None else f"<={math.ceil(interval(r)[1]*1000)/1000:.3f}"
                    record[c]=value
                    ax.text(j,i,value,ha='center',va='center',fontsize=9,color='white' if matrix[i,j]>.6 else '#222222')
                records.append(record)
            table(dest/(state+'_p_A_matrix'),records)
            ax.set(title=f'{model}: family-association prompt | {state}',xlabel='Observed crop (identity used only for evaluation)',ylabel='Target ID; size variants use identical family prompts')
            fig.colorbar(im,ax=ax,fraction=.027,pad=.025,label='Raw P(A)')
            fig.subplots_adjust(left=.20,right=.94,top=.91,bottom=.25)
            fig.text(.025,.025,'* Assisted USB rectangle. M8 and square pins excluded. No inference cutoff. Repeated family queries are correlated.',fontsize=10)
            fig.savefig(dest/(state+'_p_A_matrix.png'),dpi=180)
            pdf.savefig(fig)
            plt.close(fig)


def plots(metrics,sweep,groups):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,(model,slug) in enumerate(MODELS.items()):
        for condition,color in [('baseline','#3672a3'),('family_ablation','#b55432')]:
            points=[r for r in sweep if r['model']==model and r['condition']==condition and r['scope']=='all_crops' and r['state']=='all' and r['target_family']=='all']
            for ax,key in [(axes[0,col],'family'),(axes[1,col],'unrelated')]:
                ax.plot([r['threshold'] for r in points],[r[key+'_rate_lower'] for r in points],label=condition,color=color)
                ax.fill_between([r['threshold'] for r in points],[r[key+'_rate_lower'] for r in points],[r[key+'_rate_upper'] for r in points],color=color,alpha=.15)
                ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='P(A) threshold')
                ax.grid(alpha=.2)
            axes[0,col].set_title(model)
            axes[0,col].legend(fontsize=8)
    axes[0,0].set_ylabel('Family retention')
    axes[1,0].set_ylabel('Unrelated acceptance')
    fig.suptitle('Matched crops and model settings; baseline vs family prompt + descriptions')
    fig.tight_layout()
    fig.savefig(HERE/'threshold_comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(model,slug) in zip(axes,MODELS.items()):
        for condition,color in [('baseline','#3672a3'),('family_ablation','#b55432')]:
            rows=groups[model,condition]
            ts=sorted({v for r in rows for v in interval(r)}|{0.,1.},reverse=True)
            pos=[r for r in rows if r['same_family']]; neg=[r for r in rows if not r['same_family']]
            ax.plot([count(neg,t)[1]/len(neg) for t in ts],[count(pos,t)[0]/len(pos) for t in ts],color=color,label=condition)
        ax.set(title=model,xlabel='Unrelated acceptance (upper bound)',ylabel='Family retention (lower bound)',xlim=(0,.2),ylim=(0,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Family retention at comparable false-match rates; development data only')
    fig.tight_layout();fig.savefig(HERE/'family_retention_vs_false_matches.png',dpi=180);plt.close(fig)


def main():
    manifest=read(HERE/'manifest.json')
    validate_inputs(manifest)
    baseline=read(HERE/'baseline_scores.json')
    groups={}
    all_rows=[]
    paired=[]
    for model in MODELS:
        old=[dict(r,target_family=family(r['target']),observed_family=family(r['posthoc_identity'])) for r in baseline if r['model']==model]
        new=collect(model)
        for condition,rows in [('baseline',old),('family_ablation',new)]:
            groups[model,condition]=rows
            all_rows.extend(dict(condition=condition,**r) for r in rows)
        oldidx={(r['state'],r['target'],r['posthoc_identity']):r for r in old}
        for r in new:
            b=oldidx[r['state'],r['target'],r['posthoc_identity']]
            assert b['crop_sha256']==r['crop_sha256'] and b['same_family']==r['same_family']
            record={k:r[k] for k in ('model','state','target','target_family','posthoc_identity','observed_family','same_family','own_object_match','assisted_fallback')}
            for prefix,row in [('baseline',b),('ablation',r)]:
                record[prefix+'_answer']=row['answer']
                record[prefix+'_response_file']=row['response_file']
                for label in 'AC':
                    record[prefix+'_p_'+label]=row['p_'+label]
                    record[prefix+'_p_'+label+'_lower'],record[prefix+'_p_'+label+'_upper']=interval(row,label)
            record['p_A_change_lower']=interval(r)[0]-interval(b)[1]
            record['p_A_change_upper']=interval(r)[1]-interval(b)[0]
            paired.append(record)
        heatmaps(new,model,manifest['names'])
    table(HERE/'all_scores_comparison',all_rows)
    table(HERE/'paired_scores',paired)
    metrics=[];sweep=[];family_metrics=[];family_thresholds=[]
    for (model,condition),rows in groups.items():
        for scope in ('all_crops','autonomous_depth_only'):
            eligible=[r for r in rows if scope=='all_crops' or not r['assisted_fallback']]
            for state in (*STATES,'all'):
                group=[r for r in eligible if state=='all' or r['state']==state]
                metrics.append(summarize(group,condition,model,scope,state,'all'))
                sweep.extend(thresholds(group,condition,model,scope,state,'all'))
                # Some per-scene autonomous subsets lack any USB positive.
                for fam in FAMILIES:
                    fr=[r for r in group if r['target_family']==fam]
                    if not any(r['same_family'] for r in fr):
                        continue
                    family_metrics.append(summarize(fr,condition,model,scope,state,fam))
                    family_thresholds.extend(r for r in thresholds(fr,condition,model,scope,state,fam) if r['threshold'] in CHECKS)
    for r in metrics:
        eligible=[f for f in family_metrics if all(f[k]==r[k] for k in ('condition','model','scope','state'))]
        r['macro_family_auc_lower']=float(np.mean([f['family_vs_unrelated_auc_lower'] for f in eligible]))
        r['macro_family_auc_upper']=float(np.mean([f['family_vs_unrelated_auc_upper'] for f in eligible]))
        r['macro_family_count']=len(eligible)
    table(HERE/'metrics',metrics)
    table(HERE/'threshold_sweep',sweep)
    table(HERE/'threshold_comparison',[r for r in sweep if r['threshold'] in CHECKS])
    table(HERE/'per_family_metrics',family_metrics)
    table(HERE/'per_family_thresholds',family_thresholds)
    comparison=[]
    for model in MODELS:
        for scope in ('all_crops','autonomous_depth_only'):
            old=next(r for r in metrics if r['model']==model and r['condition']=='baseline' and r['scope']==scope and r['state']=='all')
            new=next(r for r in metrics if r['model']==model and r['condition']=='family_ablation' and r['scope']==scope and r['state']=='all')
            comparison.append(dict(model=model,scope=scope,
                baseline_auc_lower=old['family_vs_unrelated_auc_lower'],baseline_auc_upper=old['family_vs_unrelated_auc_upper'],
                revised_auc_lower=new['family_vs_unrelated_auc_lower'],revised_auc_upper=new['family_vs_unrelated_auc_upper'],
                auc_change_lower=new['family_vs_unrelated_auc_lower']-old['family_vs_unrelated_auc_upper'],
                auc_change_upper=new['family_vs_unrelated_auc_upper']-old['family_vs_unrelated_auc_lower'],
                baseline_family_C_count=old['family_C_count'],revised_family_C_count=new['family_C_count'],
                baseline_unrelated_C_count=old['unrelated_C_count'],revised_unrelated_C_count=new['unrelated_C_count']))
    table(HERE/'baseline_comparison',comparison)
    operating=[]
    for (model,condition),rows in groups.items():
        positives=[r for r in rows if r['same_family']]
        negatives=[r for r in rows if not r['same_family']]
        own=[r for r in rows if r['own_object_match']]
        choices=[]
        for t in sorted({v for r in rows for v in interval(r)}|{0.,1.}):
            choices.append((count(positives,t)[0],-count(negatives,t)[1],count(own,t)[0],t))
        for budget in (0,2,5,8,14,20):
            feasible=[v for v in choices if -v[1]<=budget]
            if not feasible:
                continue
            best=max(feasible)
            operating.append(dict(model=model,condition=condition,false_match_budget=budget,
                family_retained_lower=best[0],unrelated_accepted_upper=-best[1],own_retained_lower=best[2],
                threshold=best[3],selection='Optimistic development-set selection: maximize family retention, then minimize unrelated accepts, then maximize own retention, then higher cutoff. Not held-out validation.'))
    table(HERE/'development_operating_points',operating)
    false=[]
    for (model,condition),rows in groups.items():
        ordered=sorted([r for r in rows if not r['same_family']],key=lambda r:-interval(r)[1])
        false.extend(dict(condition=condition,rank=i,passes_0_7_definitely=interval(r)[0]>=.7,**r) for i,r in enumerate(ordered,1))
    table(HERE/'ranked_unrelated_pairs',false)
    table(HERE/'top_unrelated_pairs',[r for r in false if r['rank']<=20])
    # Inspect requested directions explicitly: target family <- observed family.
    definitions=[('Waterproof <- DSUB','Waterproof_Male','DSUB_Male'),('DSUB <- USB','DSUB_Male','USB_Male'),
                 ('Gear <- hex nut','gear','hex_nut'),('True nut family crops','hex_nut','hex_nut')]
    known=[];known_rows=[]
    for label,tf,of in definitions:
        known_rows.extend(dict(case=label,**r) for r in paired if r['target_family']==tf and r['observed_family']==of)
        for (model,condition),rows in groups.items():
            rs=[r for r in rows if r['target_family']==tf and r['observed_family']==of]
            record=dict(case=label,model=model,condition=condition,pairs=len(rs),C_count=sum(r['answer']=='C' for r in rs))
            for letter in 'AC':
                record[f'mean_p_{letter}_lower']=float(np.mean([interval(r,letter)[0] for r in rs]))
                record[f'mean_p_{letter}_upper']=float(np.mean([interval(r,letter)[1] for r in rs]))
            for t in (.6,.65,.7,.9,.95,.98):
                record[f'pass_{t}_lower'],record[f'pass_{t}_upper']=count(rs,t)
            known.append(record)
    table(HERE/'known_failures',known)
    table(HERE/'known_failure_pairs',known_rows)
    consistency=[]
    for model in MODELS:
        rows=groups[model,'family_ablation']
        for state in STATES:
            for observed in manifest['names']:
                for fam in ('gear','hex_nut','round_pin'):
                    rs=[r for r in rows if r['state']==state and r['posthoc_identity']==observed and r['target_family']==fam]
                    consistency.append(dict(model=model,state=state,observed=observed,target_family=fam,
                        repeated_queries=len(rs),p_A_range_upper=max(interval(r)[1] for r in rs)-min(interval(r)[0] for r in rs),
                        answer_agreement=len({r['answer'] for r in rs})==1))
    table(HERE/'repeated_family_query_consistency',consistency)
    plots(metrics,sweep,groups)
    lines=['# Local Product-Family Association Ablation','',
        'Three local models, four frozen scenes, 484 calls/model (1,452 new). No GPT/cloud API calls. Baseline untouched.',
        'This jointly changes the task, labels, comparison rules and descriptions. It is not an isolated description-only effect.',
        '', '## Inputs and Controls',
        '- Same 44 crop files, 11 target IDs and 11 candidates per scene; same model digests, Ollama version and exact per-model inference controls.',
        '- Size-family descriptions AND full request prompts are identical for all gear sizes, both nuts and all round pins. Each original target ID is still run independently for direct pairing, not supplied to the model.',
        '- Those repeated queries are correlated, not independent evidence. Per-family metrics and unweighted macro-family AUC supplement the original 100-family/384-unrelated pair-weighted metric.',
        '- M8 and square pins remain excluded. Two USB rectangles remain assisted recoveries (101/303); autonomous-depth-only tables exclude all comparisons against those crops, not just own-object rows.',
        '- Waterproof CAD review_sheet.png was visually inspected: a block body with grouped round openings. Descriptions are saved verbatim in descriptions.json. No new render views or localization passes.',
        '- P(A)=exp(raw logp_A), no renormalization or inference gate. A/B/C only. P(C) is raw token probability; C frequency counts emitted C labels, not a threshold on P(C).',
        '- Exact unavailable probabilities remain null, with conservative bounds as in the baseline. Charts use marked upper bounds; no zero imputation.',
        '', '## Aggregate Results',
        '| Model | Condition | Family AUC | Family at .70 /100 | Unrelated at .70 /384 | Own at .70 /44 | C labels /484 |',
        '|---|---|---:|---:|---:|---:|---:|']
    def span(lo,hi):
        return str(lo) if lo==hi else f'{lo}-{hi}'
    for model in MODELS:
        for condition in ('baseline','family_ablation'):
            m=next(r for r in metrics if r['model']==model and r['condition']==condition and r['state']=='all' and r['scope']=='all_crops')
            s=next(r for r in sweep if r['model']==model and r['condition']==condition and r['state']=='all' and r['scope']=='all_crops' and r['threshold']==.7)
            auc_text=f"{m['family_vs_unrelated_auc_lower']:.4f}" if m['family_vs_unrelated_auc_lower']==m['family_vs_unrelated_auc_upper'] else f"{m['family_vs_unrelated_auc_lower']:.4f}-{m['family_vs_unrelated_auc_upper']:.4f}"
            counts=[span(s[k+'_retained_lower'],s[k+'_retained_upper']) for k in ('family','unrelated','own')]
            lines.append(f"| {model} | {condition} | {auc_text} | {' | '.join(counts)} | {m['all_C_count']} |")
    lines.extend(['','## Requested Failure Cases','At P(A) >= 0.70. First three cases should have fewer passes; true-nut cases should have more.',
        '| Case | Model | Condition | Passes | Total | C labels |','|---|---|---|---:|---:|---:|'])
    for r in known:
        lines.append(f"| {r['case']} | {r['model']} | {r['condition']} | {span(r['pass_0.7_lower'],r['pass_0.7_upper'])} | {r['pairs']} | {r['C_count']} |")
    lines.extend(['','## Repeated-Query Diagnostic',
        f"Identical size-family prompts for the same crop had a maximum P(A) spread of {max(r['p_A_range_upper'] for r in consistency):.6f}; {sum(not r['answer_agreement'] for r in consistency)} groups disagreed on the emitted A/B/C label. These are separately executed calls, not copied scores. The cause of likelihood variation was not isolated; fixed settings do not establish bitwise reproducibility. See repeated_family_query_consistency.csv."])
    lines.extend(['','## Insufficient Evidence',
        'Mean raw P(C) and emitted-C frequency are different quantities. Family and unrelated pairs are separated below.',
        '| Model | Condition | Family mean P(C) | Family C /100 | Unrelated mean P(C) | Unrelated C /384 |',
        '|---|---|---:|---:|---:|---:|'])
    for r in metrics:
        if r['scope']!='all_crops' or r['state']!='all':
            continue
        values=[]
        for category in ('family','unrelated'):
            lo,hi=r[category+'_mean_p_C_lower'],r[category+'_mean_p_C_upper']
            values.append(f'{lo:.6g}' if lo==hi else f'{lo:.6g}-{hi:.6g}')
        lines.append(f"| {r['model']} | {r['condition']} | {values[0]} | {r['family_C_count']} | {values[1]} | {r['unrelated_C_count']} |")
    lines.extend(['','## Per-Family Retention',
        'Counts at .70, baseline -> revised. Denominators include repeated size-family query IDs, not independent physical trials.',
        '| Model | Target family | Family retained | Family pairs | Unrelated accepted | Unrelated pairs |',
        '|---|---|---:|---:|---:|---:|'])
    for model in MODELS:
        for fam in FAMILIES:
            rs=[next(r for r in family_thresholds if r['model']==model and r['condition']==c and r['scope']=='all_crops' and r['state']=='all' and r['target_family']==fam and r['threshold']==.7) for c in ('baseline','family_ablation')]
            pos=[span(r['family_retained_lower'],r['family_retained_upper']) for r in rs]
            neg=[span(r['unrelated_retained_lower'],r['unrelated_retained_upper']) for r in rs]
            lines.append(f"| {model} | {fam} | {pos[0]} -> {pos[1]} | {rs[0]['family_total']} | {neg[0]} -> {neg[1]} | {rs[0]['unrelated_total']} |")
    lines.extend(['','## Files',
        '- selected_prompts.json, descriptions.json, manifest.json: exact prompts, provenance and preserved baseline hashes.',
        '- all_scores_comparison.csv/json and paired_scores.csv/json: all baseline and revised scores, including P(A), P(B), P(C), labels, input hashes and raw-response paths.',
        '- metrics.csv/json: family AUC, macro-family AUC, P(C) means/bounds, C frequency by family/unrelated/own and scene.',
        '- baseline_comparison.csv/json: directly paired aggregate AUC changes and family/unrelated C counts.',
        '- development_operating_points.csv/json: matched false-match budgets with development-selected cutoffs. Optimistic analysis, NOT validated thresholds.',
        '- threshold_sweep.csv/json: every .01 cutoff plus high-score checks, with family/unrelated/own retained counts and rates.',
        '- per_family_metrics.csv/json, per_family_thresholds.csv/json: each target family separately.',
        '- ranked_unrelated_pairs.csv/json: all unrelated pairs sorted within each model/condition, not merely those above .70.',
        '- top_unrelated_pairs.csv/json: the top 20 unrelated pairs per model/condition, retaining repeated size-family queries explicitly.',
        '- known_failures.csv/json and known_failure_pairs.csv/json: requested confusion directions, paired raw scores and emitted labels.',
        '- repeated_family_query_consistency.csv/json: diagnostic score spread across identical size-family requests.',
        '- Model folders: raw responses, exact runtime metadata, per-scene heatmaps and four-page PDF.',
        '', '## Interpretation Limits',
        'These are reused development scenes, not a held-out validation set. Prompt changes were designed with knowledge of baseline failures. Do not claim calibrated identity correctness, statistical superiority, or a deployment-ready gate. Compare thresholds at matched false-match rates, not just at a universal .70. Own-object retention is secondary and does not imply exact-size identification.',
        'The earlier name-and-color-only proposal was not run. This family ablation is a different, explicitly requested experiment. No model or threshold is deployed automatically.'])
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    validate_inputs(manifest)
    save(HERE/'validation.json',dict(baseline_unchanged=True,local_only=True,calls=1452,
        paired_crops_match=True,model_controls_unchanged=True,
        complete_models=list(MODELS),raw_response_hashes={r['response_file']:digest(ROOT/r['response_file']) for rows in groups.values() for r in rows}))
    with zipfile.ZipFile(HERE/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in HERE.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.csv','.png','.pdf','.py','.txt','.md') and '__pycache__' not in p.parts:
                z.write(p,p.relative_to(HERE).as_posix())
        for r in baseline:
            p=ROOT/r['response_file'];z.write(p,'baseline_raw/'+r['response_file'].replace('\\','/'))
        for state,crops in manifest['crops'].items():
            for c in crops:
                z.write(ROOT/c['crop_file'],f"input_crops/{state}/{c['identity']}.png")
        for filename in ('run_models.py','run_trial.py'):
            p=(ROOT/'qwen9b-four-scene-trial'/filename if filename=='run_trial.py' else ROOT/'gpt6-gemma4-four-scene-trial'/filename)
            z.write(p,'baseline_code/'+p.parent.name+'/'+filename)
    with zipfile.ZipFile(HERE/'results.zip') as z:
        assert z.testzip() is None
    print('\n'.join(lines[:24]))


if __name__=='__main__':
    main()
