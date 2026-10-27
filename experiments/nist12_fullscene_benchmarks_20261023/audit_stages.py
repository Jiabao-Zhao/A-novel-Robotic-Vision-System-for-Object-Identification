"""Post-lock error attribution; never used for candidate generation or selection."""
from common import HERE, read, save, path, digest
from evaluate import AUDIT, bbox_iou, box_edges


def main():
    lock=read(HERE/'predictions_lock.json')
    assert lock['frozen_before_audit_labels']
    rows=read(HERE/'analysis/per_task.json')
    stages={}
    for method in dict.fromkeys(r['method'] for r in rows):
        selected=[r for r in rows if r['method']==method]
        def count_if_present(key):
            return None if all(r[key] is None for r in selected) else sum(r[key] is True for r in selected)
        stages[method]={
            'requests':len(selected),
            'raw_target_box_present':count_if_present('raw_proposal_bbox_recall'),
            'target_box_after_native_size_filter':count_if_present('after_native_size_filter_bbox_recall'),
            'CAD_conditioned_target_box_before_area_filter':count_if_present('conditional_mask_bbox_recall_before_area_filter'),
            'CAD_conditioned_target_box_after_area_filter':count_if_present('conditional_mask_bbox_recall_after_area_filter'),
            'unique_answer':sum(r['status']=='ok' for r in selected),
            'correct':None if method=='nids_basic' else sum(r['correct'] for r in selected)}
    our_rows=[]
    for scene in read(HERE/'inputs.json')['scenes']:
        scene_id=scene['scene_id']
        folder=HERE/'ours'/scene_id
        truth=read(AUDIT/'scenes'/scene_id/'capture_audit.json')['audit_only_objects']
        proposals=read(folder/'localization.json')['proposals']
        for decision in read(folder/'decisions.json'):
            target=decision['target']
            edges=box_edges(truth[target]['bbox'],'audit')
            available={pid for score,pid in decision['scores']}
            candidates=[]
            for pid in decision['shortlist_ids']:
                proposal=proposals[pid]
                candidates.append({'proposal_id':pid,'bbox_xyxy':proposal['bbox_xyxy'],
                    'target_bbox_IoU':bbox_iou(box_edges(proposal['bbox_xyxy'],'ours'),edges),
                    'depth_fit_available':pid in available,'crop':proposal['crop']})
            row=next(r for r in rows if r['method']=='ours' and r['scene_id']==scene_id and r['target']==target)
            our_rows.append({'scene_id':scene_id,'target':target,'status':decision['status'],
                'error':row['error'],'selected_physical_object':row['selected_physical_object'],
                'correct_target_in_CLIP_top5':any(c['target_bbox_IoU']>=.5 for c in candidates),'candidates':candidates})
    stages['ours']['target_box_in_CLIP_top5']=sum(r['correct_target_in_CLIP_top5'] for r in our_rows)
    stages['ours']['unavailable_with_correct_target_in_top5']=sum(r['status']=='unavailable' and r['correct_target_in_CLIP_top5'] for r in our_rows)
    save(HERE/'analysis/stage_counts.json',stages)
    save(HERE/'analysis/ours_shortlist_audit.json',{'posthoc_diagnostic':True,
        'no_predictions_changed':True,'prediction_lock_sha256':digest(HERE/'predictions_lock.json'),'tasks':our_rows})
    print(stages)


if __name__=='__main__':main()
