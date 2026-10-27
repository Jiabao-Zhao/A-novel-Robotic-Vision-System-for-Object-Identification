"""Post-inference audit only: never imported by a proposal or matching runner."""
import csv
import html
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
from common import HERE, REPO, read, save, path, digest, relative

METHODS = ('ours','cnos','sam6d_rgb','sam6d_rgbd','nids_basic','nids_basic_finite_templates','noctis','nemo')
AUDIT = REPO / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures'


def bbox_iou(a,b):
    # Continuous half-open edge coordinates; normalize each source explicitly.
    x0,y0 = max(a[0],b[0]), max(a[1],b[1])
    x1,y1 = min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,x1-x0)*max(0,y1-y0)
    area = lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    union = area(a)+area(b)-intersection
    return intersection/union if union>0 else 0.


def box_edges(box, method):
    result=list(map(float,box))
    if method in ('audit','ours','cnos','sam6d_rgb','sam6d_rgbd'):
        result[2]+=1.;result[3]+=1.
    return result


def mask_overlaps(mask, labels, truth_pixels, truth_ids):
    mask_count=int(np.count_nonzero(mask))
    intersect_counts=np.bincount(labels[mask],minlength=int(labels.max())+1)
    result=[]
    for name,visible_pixels in truth_pixels.items():
        intersection=int(intersect_counts[truth_ids[name]+1])
        union=mask_count+visible_pixels-intersection
        result.append((intersection/union if union else 0.,name))
    return sorted(result,reverse=True)


def select(rows, object_id):
    candidates = sorted((r for r in rows if r['object_id']==object_id),key=lambda r:-r['score'])
    if not candidates:
        return None,'no_retained_detection'
    if len(candidates)>1 and abs(candidates[0]['score']-candidates[1]['score'])<=1e-12:
        return None,'tie'
    return candidates[0],'ok'


def lock_predictions(manifest):
    hashes = {}
    for method in METHODS:
        if method=='ours':
            complete=HERE/'ours/completion.json'
            assert read(complete)['tasks']==96
            hashes[relative(complete)]=digest(complete)
        for scene in manifest['scenes']:
            folder = HERE/method/scene['scene_id']
            if method!='ours':
                complete=folder/'completion.json'
                if not complete.exists():
                    raise RuntimeError(f'Incomplete method run: {complete}')
                hashes[relative(complete)]=digest(complete)
            for name in ('detections.json','masks.npz'):
                file = folder/name
                if not file.exists():
                    raise RuntimeError(f'Incomplete predictions: {file}')
                hashes[relative(file)] = digest(file)
    for method in ('cnos','sam6d','nids_basic','nids_basic_finite_templates','noctis','ours'):
        for file in sorted((HERE/method).rglob('*.json')):
            if file.name in ('proposals.json','proposal_filter.json','raw_proposals.json','localization.json','proposal_completion.json','decisions.json'):
                hashes[relative(file)]=digest(file)
    # Record the CAD-only failure diagnosis and declared control before reading labels.
    for file in sorted((HERE/'nids_noctis').glob('*.json')):
        hashes[relative(file)]=digest(file)
    hashes[relative(HERE/'protocol.json')]=digest(HERE/'protocol.json')
    hashes[relative(HERE/'inputs.json')]=digest(HERE/'inputs.json')
    for method in METHODS:
        for file in sorted((HERE/method).glob('*.json')):
            hashes[relative(file)]=digest(file)
        for file in sorted((HERE/method).glob('N*/matching_scores.npz')):
            hashes[relative(file)]=digest(file)
    for file in sorted((HERE/'nemo').glob('N*/conditional_components.json')):
        hashes[relative(file)]=digest(file)
    for file in sorted((HERE/'nemo').glob('N*/raw_obj_*.npz')):
        hashes[relative(file)]=digest(file)
    destination = HERE/'predictions_lock.json'
    if destination.exists():
        frozen=read(destination)['sha256']
        assert all(hashes.get(name)==value for name,value in frozen.items())
        required={name for name in hashes if Path(name).name in ('detections.json','masks.npz','completion.json','matching_scores.npz')}
        assert required.issubset(frozen)
    else:
        save(destination,{'sha256':hashes,'frozen_before_audit_labels':True,
                          'locked_at_UTC':datetime.now(timezone.utc).isoformat()})


def raw_boxes(method, scene_id):
    folder = HERE/method/scene_id
    if method=='ours':
        return [r['bbox_xyxy'] for r in read(folder/'localization.json')['proposals']]
    if method in ('cnos','sam6d_rgb','sam6d_rgbd'):
        source = 'cnos' if method=='cnos' else 'sam6d'
        return [r['bbox_xyxy'] for r in read(HERE/source/'raw_proposals'/scene_id/'proposals.json')]
    if method in ('nids_basic','nids_basic_finite_templates','noctis'):
        source='nids_basic' if method=='nids_basic_finite_templates' else method
        return [r['bbox_xyxy'] for r in read(HERE/source/scene_id/'raw_proposals.json')]
    return None  # NeMO directly generates CAD-conditioned masks rather than a shared proposal bank.


def filtered_boxes(method, scene_id, raw):
    source='sam6d' if method in ('sam6d_rgb','sam6d_rgbd') else method
    if method=='nids_basic_finite_templates':source='nids_basic'
    file=HERE/source/scene_id/'proposal_filter.json'
    if file.exists() and raw is not None:
        indices=read(file)['retained_proposal_ids']
        return [raw[i] for i in indices]
    return None


def draw_case(scene_rgb, true_box, chosen, mask, target, method, outcome, destination):
    frame = Image.open(scene_rgb).convert('RGB')
    boxes = [true_box] + ([] if chosen is None else [chosen['bbox_xyxy']])
    lo = np.min(np.asarray(boxes)[:,:2],axis=0)-80
    hi = np.max(np.asarray(boxes)[:,2:],axis=0)+80
    region = [max(0,int(lo[0])),max(0,int(lo[1])),min(frame.width,int(hi[0])+1),min(frame.height,int(hi[1])+1)]
    frame=frame.crop(region)
    if chosen is not None:
        overlay = np.zeros((frame.height,frame.width,4),dtype=np.uint8)
        overlay[mask[region[1]:region[3],region[0]:region[2]]]=[220,40,40,75]
        frame = Image.alpha_composite(frame.convert('RGBA'),Image.fromarray(overlay)).convert('RGB')
    draw = ImageDraw.Draw(frame)
    local_box=lambda b:[b[0]-region[0],b[1]-region[1],b[2]-region[0],b[3]-region[1]]
    draw.rectangle(local_box(true_box),outline=(0,150,60),width=5)
    if chosen is not None:
        draw.rectangle(local_box(chosen['bbox_xyxy']),outline=(220,30,40),width=4)
    frame.thumbnail((1100,600))
    canvas = Image.new('RGB',(max(650,frame.width),frame.height+65),'white')
    canvas.paste(frame,(0,65))
    draw=ImageDraw.Draw(canvas)
    draw.text((12,10),f'{method} | {target} | {outcome}',fill='black')
    draw.text((12,32),'Green: target audit box. Red: selected box and mask.',fill='black')
    destination.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(destination)


def main():
    manifest = read(HERE/'inputs.json')
    lock_predictions(manifest)
    output = HERE/'analysis'
    output.mkdir(exist_ok=True)
    rows, audit_hashes = [], {}
    for scene in manifest['scenes']:
        scene_id = scene['scene_id']
        capture = AUDIT/'scenes'/scene_id/'capture_audit.json'
        seg_file = AUDIT/'raw'/scene_id/'audit_segmentation.npy'
        audit_hashes[relative(capture)] = digest(capture)
        audit_hashes[relative(seg_file)] = digest(seg_file)
        truths = read(capture)['audit_only_objects']
        segmentation = np.load(seg_file)
        labels=segmentation[:,:,0].astype(np.int64)+1
        label_counts=np.bincount(labels.ravel())
        gt_pixels={item['target']:int(label_counts[truths[item['target']]['geom_id']+1]) for item in manifest['objects']}
        assert all(gt_pixels[name]==truths[name]['visible_pixels'] for name in gt_pixels)
        for method in METHODS:
            folder = HERE/method/scene_id
            detections = read(folder/'detections.json')
            ours_decisions={r['object_id']:r for r in read(folder/'decisions.json')} if method=='ours' else {}
            conditional=read(folder/'conditional_components.json')['objects'] if method=='nemo' else {}
            masks = np.load(folder/'masks.npz')['masks']
            assert masks.dtype==bool and masks.shape[1:]==segmentation.shape[:2]
            assert all(0<=r['mask_index']<len(masks) and np.isfinite(r['score']) for r in detections)
            raw = raw_boxes(method,scene_id)
            filtered=filtered_boxes(method,scene_id,raw)
            for item in manifest['objects']:
                target = item['target']
                chosen,status = select(detections,item['object_id'])
                if method=='ours':
                    decision=ours_decisions[item['object_id']]
                    if decision['status']!='ok':
                        assert chosen is None
                        status=decision['status']
                if method=='nids_basic':
                    diagnostic=read(HERE/'nids_noctis/nids_zero_mask_diagnostic.json')
                    assert diagnostic['finite_scene_query_vectors'] and len(diagnostic['invalid_views'])==6
                    assert chosen is None and not detections
                    status='unavailable_reference_features'
                true_box=truths[target]['bbox']
                target_edges=box_edges(true_box,'audit')
                iou=0. if chosen is None else bbox_iou(box_edges(chosen['bbox_xyxy'],method),target_edges)
                mask_iou=0.
                selected_physical=None
                highest_mask_iou=0.
                if chosen is not None:
                    mask=masks[chosen['mask_index']]
                    overlaps=mask_overlaps(mask,labels,gt_pixels,{n:truths[n]['geom_id'] for n in gt_pixels})
                    highest_mask_iou=overlaps[0][0]
                    if highest_mask_iou>=.1:
                        selected_physical=overlaps[0][1]
                    mask_iou=next(value for value,name in overlaps if name==target)
                correct=status=='ok' and iou>=.5
                raw_recall=None if raw is None else max((bbox_iou(box_edges(b,method),target_edges) for b in raw),default=0.)>=.5
                filtered_recall=None if filtered is None else max((bbox_iou(box_edges(b,method),target_edges) for b in filtered),default=0.)>=.5
                retained_recall=max((bbox_iou(box_edges(r['bbox_xyxy'],method),target_edges) for r in detections),default=0.)>=.5
                conditional_pre=conditional_post=None
                if method=='nemo':
                    c=conditional[str(item['object_id'])]
                    conditional_pre=max((bbox_iou(b,target_edges) for b in c['pre1024_boxes_xyxy']),default=0.)>=.5
                    conditional_post=max((bbox_iou(b,target_edges) for b in c['postfilter_boxes_xyxy']),default=0.)>=.5
                if correct:
                    error='correct'
                elif status=='unavailable_reference_features':
                    error='method_reference_features_invalid'
                elif status=='tie':
                    error='tie'
                elif raw_recall is False:
                    error='proposal_localization_miss'
                elif filtered_recall is False:
                    error='native_size_filter_miss'
                elif chosen is None and method=='ours':
                    error='depth_verification_unavailable'
                elif chosen is None and method=='nemo':
                    error='conditional_component_size_filter_miss' if conditional_pre and not conditional_post else 'no_target_conditioned_detection'
                elif chosen is None:
                    error='target_identity_missing_after_matching'
                elif selected_physical==target:
                    error='target_selected_but_box_IoU_below_0.5'
                elif selected_physical is None:
                    error='selected_region_unassigned_or_poor_localization'
                else:
                    error='wrong_instance_selected'
                row={'scene_id':scene_id,'target':target,'method':method,'correct':correct,'status':status,'error':error,
                     'selected_physical_object':selected_physical,'highest_mask_IoU':highest_mask_iou,
                     'physical_identity_audit_rule':'Highest mask IoU >= 0.1; diagnostic only; weaker overlap remains unassigned',
                     'score':None if chosen is None else chosen['score'],
                     'proposal_id':None if chosen is None else chosen['proposal_id'],
                     'target_bbox_IoU':iou,'target_mask_IoU':mask_iou,'raw_proposal_bbox_recall':raw_recall,
                     'after_native_size_filter_bbox_recall':filtered_recall,
                     'conditional_mask_bbox_recall_before_area_filter':conditional_pre,
                     'conditional_mask_bbox_recall_after_area_filter':conditional_post,
                     'retained_box_recall_any_identity':retained_recall,'retained_detections':len(detections),
                     'raw_proposals':None if raw is None else len(raw)}
                if not correct:
                    image=output/'cases'/method/f'{scene_id}__{target}.png'
                    draw_case(path(scene['rgb']),true_box,chosen,None if chosen is None else masks[chosen['mask_index']],
                              target,method,error,image)
                    row['case_image']=image.relative_to(output).as_posix()
                else:
                    row['case_image']=None
                rows.append(row)
            del masks
    save(output/'per_task.json',rows)
    fields=list(rows[0])
    with (output/'per_task.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={m:{'correct':sum(r['correct'] for r in rows if r['method']==m),'total':96,
                  'available':sum(r['status']=='ok' for r in rows if r['method']==m),
                  'unavailable':sum(str(r['status']).startswith('unavailable') for r in rows if r['method']==m),
                  'no_retained_detection':sum(r['status']=='no_retained_detection' for r in rows if r['method']==m),
                  'ties':sum(r['status']=='tie' for r in rows if r['method']==m),
                  'errors':dict(Counter(r['error'] for r in rows if r['method']==m and not r['correct']))} for m in METHODS}
    def family(target):
        return 'round_pin' if str(target).startswith('RGOCG') else 'rectangular_pin' if str(target).startswith('KET') else str(target)
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        summary[method]['run_status']='invalid_reference_features' if method=='nids_basic' else 'completed'
        summary[method]['pin_correct']=sum(r['correct'] for r in selected if family(r['target']) in ('round_pin','rectangular_pin'))
        summary[method]['nonpin_correct']=sum(r['correct'] for r in selected if family(r['target']) not in ('round_pin','rectangular_pin'))
        summary[method]['wrong_size_within_pin_family']=sum(r['error']=='wrong_instance_selected' and family(r['target']) in ('round_pin','rectangular_pin')
            and family(r['target'])==family(r['selected_physical_object']) for r in selected)
    per_object=[{'target':o['target'],**{m:None if m=='nids_basic' else sum(r['correct'] for r in rows if r['target']==o['target'] and r['method']==m) for m in METHODS}} for o in manifest['objects']]
    object_errors={o['target']:{m:[r for r in rows if r['target']==o['target'] and r['method']==m and not r['correct']]
                               for m in METHODS} for o in manifest['objects']}
    save(output/'per_object_errors.json',object_errors)
    paired=[]
    for m in METHODS[1:]:
        if m=='nids_basic':
            paired.append({'method':m,'comparison_available':False,
                           'reason':'Native CAD reference NaNs; classification comparison is unavailable.'})
            continue
        fixed=[];introduced=[]
        for r in rows:
            if r['method']!=m: continue
            ours=next(v for v in rows if v['method']=='ours' and v['scene_id']==r['scene_id'] and v['target']==r['target'])
            key={'scene_id':r['scene_id'],'target':r['target'],'ours':ours['error'],'comparison':r['error']}
            if not ours['correct'] and r['correct']:fixed.append(key)
            if ours['correct'] and not r['correct']:introduced.append(key)
        summary[m].update(fixed_vs_ours=len(fixed),introduced_vs_ours=len(introduced))
        paired.append({'method':m,'comparison_available':True,'fixed':fixed,'introduced':introduced})
    save(output/'summary.json',summary);save(output/'per_object.json',per_object);save(output/'paired_changes.json',paired)
    with (output/'per_object.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['target',*METHODS]);writer.writeheader();writer.writerows(per_object)
    table='<tr><th>Object (correct / 8)</th>'+''.join(f'<th>{m}</th>' for m in METHODS)+'</tr>'
    for row in per_object:
        table+='<tr><td>'+html.escape(row['target'])+'</td>'+''.join('<td>N/A</td>' if row[m] is None else f'<td>{row[m]}/8</td>' for m in METHODS)+'</tr>'
    cases=''
    for method in METHODS:
        heading='unavailable: native reference features contain NaNs' if method=='nids_basic' else f'{summary[method]["correct"]}/96 correct'
        cases+=f'<h2>{method}: {heading}</h2>'
        for row in rows:
            if row['method']==method and not row['correct']:
                cases+=f'<details data-target="{html.escape(row["target"])}"><summary>{html.escape(row["scene_id"]+" "+row["target"]+": "+row["error"])}; selected {html.escape(str(row["selected_physical_object"]))}</summary><img loading="lazy" src="{row["case_image"]}"><pre>{html.escape(str(row))}</pre></details>'
    page=f'<!doctype html><meta charset="utf-8"><title>Full-scene CAD identification benchmark</title><style>body{{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#152333}}table{{border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ccd}}td{{text-align:center}}td:first-child{{text-align:left}}img{{max-width:100%}}details{{margin:10px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap}}</style><h1>Full-scene CAD identification</h1><p>Eight reused simulation scenes, 12 objects. Every method generates its own proposals. Highest-scored retained detection for each supplied CAD identity; correct requires target box IoU ≥ 0.5. Training-free on target parts. Custom rendered CAD reference condition.</p><table>{table}</table><p>Native proposal counts vary. All predictions were frozen before audit identities were read. This is development simulation, not held-out real-depth validation.</p>{cases}'
    (output/'index.html').write_text(page,encoding='utf-8')
    save(output/'completion.json',{'requests':len(rows),'predictions_locked_before_audit':True,'audit_sha256':audit_hashes,
          'prediction_lock_sha256':digest(HERE/'predictions_lock.json'),'evaluation_source_sha256':digest(__file__)})
    print(summary)


if __name__=='__main__':
    main()
