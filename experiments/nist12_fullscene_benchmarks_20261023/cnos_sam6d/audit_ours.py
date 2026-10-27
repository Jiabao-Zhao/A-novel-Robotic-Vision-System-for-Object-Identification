"""Independent saved-output replay of CLIP/depth, without audit identities."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import HERE as ROOT,read,save,path,digest,relative


def main():
    inputs=read(ROOT/'inputs.json'); base=ROOT/'ours'
    shortlists=read(base/'shortlists.json'); pairs=read(base/'clip_pair_scores.json')
    geometry_protocol=read(base/'geometry_protocol.json')
    source=ROOT.parent/'nist12_qwen_clip_reranking_20261023/depth_verification_20261023/geometry.py'
    assert digest(base/'shortlists.json')==geometry_protocol['shortlists_sha256']
    assert digest(source)==geometry_protocol['geometry_source_sha256']
    assert geometry_protocol['clearances_m']==[0.] and geometry_protocol['labels_read'] is False
    assert len(shortlists)==96 and len(pairs)==1344
    hashes={relative(p):digest(p) for p in [base/'shortlists.json',base/'clip_pair_scores.json',
        base/'geometry_protocol.json',base/'completion.json',ROOT/'inputs.json',source]}
    observed_counts={}; observations={}; proposals={}
    for scene in inputs['scenes']:
        sid=scene['scene_id']; folder=base/sid
        loc=read(folder/'localization.json'); masks=np.load(folder/'masks.npz')['masks']
        depth=np.load(path(scene['depth'])); K=np.load(path(scene['intrinsics']))
        plane=np.asarray(loc['plane_camera_m'],np.float64)
        assert masks.dtype==np.bool_ and masks.shape[1:]==depth.shape
        assert loc['labels_read'] is False and len(loc['proposals'])==len(masks)==14
        assert 0<np.nanmedian(depth)<5 and np.isfinite(plane).all()
        for row in loc['proposals']:
            pid=row['proposal_id']; roi=row['roi']
            a,b,c,d=[roi[k] for k in ('x1','y1','x2','y2')]
            assert row['mask_index']==pid and row['bbox_xyxy']==[a,b,c,d]
            yy,xx=np.mgrid[b:d+1,a:c+1]; z=depth[b:d+1,a:c+1]
            points=np.stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z],-1)
            foreground=np.isfinite(z)&(z>0)&(np.abs(points@plane[:3]+plane[3])/np.linalg.norm(plane[:3])>.002)
            assert np.array_equal(masks[pid,b:d+1,a:c+1],foreground)
            assert int(masks[pid].sum())==int(foreground.sum())
            assert digest(path(row['crop']))==row['crop_sha256']
            observed_counts[sid,pid]=int(foreground.sum());proposals[sid,pid]=row
        observations[sid]=loc
        for filename in ('localization.json','masks.npz','decisions.json','detections.json'):
            hashes[relative(folder/filename)]=digest(folder/filename)
    # The four-term view score and maximum over CAD views are reconstructed
    # from independently saved components, never from correctness labels.
    indexed={}
    for row in pairs:
        assert row['score_available'] and len(row['views'])==20
        values=[]
        for view in row['views']:
            score=sum(view['components'].values())/4
            assert abs(score-view['score'])<1e-14
            values.append(score)
        assert abs(max(values)-row['score'])<1e-14
        indexed.setdefault((row['scene_id'],row['target']),[]).append((row['proposal_id'],row['score']))
    selected_count=0; statuses=Counter(); pose_count=0; fit_count=0
    expected_files=set(); max_rotation_error=0.; max_height_error=0.
    for shortlist in shortlists:
        sid,target=shortlist['scene_id'],shortlist['target']
        ordered=sorted(indexed[sid,target],key=lambda p:(-p[1],p[0]))
        assert len(ordered)==14 and shortlist['status']=='ok' and shortlist['complete']
        assert shortlist['ranked_candidate_ids']==[p for p,_ in ordered]
        assert shortlist['candidate_ids']==[p for p,_ in ordered[:5]]
        np.testing.assert_allclose(shortlist['ranked_scores'],[s for _,s in ordered],atol=1e-14,rtol=0)
        assert ordered[4][1]-ordered[5][1]>1e-12
        scores=[]
        for pid in shortlist['candidate_ids']:
            file=base/'fits'/f'{sid}__{target}__p{pid:03d}.json'
            expected_files.add(file.name);record=read(file);fit_count+=1
            hashes[relative(file)]=digest(file)
            assert (record['scene_id'],record['target'],record['proposal_id'])==(sid,target,pid)
            assert record['labels_read'] is False and record['geometry_source_sha256']==digest(source)
            fit=record['geometry'];assert len(fit['arms'])==1
            obj=next(o for o in inputs['objects'] if o['target']==target)
            np.testing.assert_allclose(fit['cad_native_extent_m'],obj['native_extent_m'],atol=1e-8,rtol=0)
            arm=fit['arms'][0];assert arm['clearance_m']==0 and arm['arm_name']=='contact_0mm'
            records=arm['pose_bank']
            assert records, f'Missing pose bank {file}'
            pose_count+=len(records)
            rotations=np.asarray([p['T_camera_from_cad'] for p in records])[:,:3,:3]
            error=float(np.max(np.abs(rotations@rotations.transpose(0,2,1)-np.eye(3))))
            max_rotation_error=max(max_rotation_error,error);assert error<1e-8
            np.testing.assert_allclose(np.linalg.det(rotations),1.,atol=1e-8)
            available=[]
            for pose in records:
                height=abs(pose['minimum_table_height_m'])
                max_height_error=max(max_height_error,height);assert height<1e-7
                v=pose['verification']
                if 'observed_pixels' in v:assert v['observed_pixels']==observed_counts[sid,pid]
                if v['status']!='available':continue
                union=v['union_pixels']; inter=v['intersection_pixels']
                assert union==v['observed_pixels']+v['projected_pixels']-inter
                assert v['observed_only_pixels']==v['observed_pixels']-inter
                assert v['cad_only_pixels']==v['projected_pixels']-inter
                assert abs(v['p_obs']-v['observed_only_pixels']/union)<1e-14
                assert abs(v['p_cad']-v['cad_only_pixels']/union)<1e-14
                g=1-(v['p_obs']+v['p_cad']+v['p_depth'])/3
                assert abs(g-v['geometry_score'])<1e-14
                available.append(g)
            best=arm.get('best_geometry')
            if best is not None:
                assert available and abs(best['verification']['geometry_score']-max(available))<1e-14
                index=arm['best_geometry_index']
                assert records[index]==best
                scores.append((best['verification']['geometry_score'],pid))
            else:assert not available
        scores.sort(reverse=True)
        status='unavailable'
        if len(scores)==5:
            status='tie' if abs(scores[0][0]-scores[1][0])<=1e-12 else 'ok'
        decisions=read(base/sid/'decisions.json')
        decision=next(d for d in decisions if d['target']==target)
        assert decision['status']==status and decision['scores']==[list(v) for v in scores]
        assert decision['shortlist_ids']==shortlist['candidate_ids'];statuses[status]+=1
        detected=[d for d in read(base/sid/'detections.json') if d['object_id']==obj['object_id']]
        if status=='ok':
            selected_count+=1;assert len(detected)==1
            row=detected[0];score,pid=scores[0];proposal=proposals[sid,pid]
            assert row['proposal_id']==row['mask_index']==pid
            assert row['bbox_xyxy']==proposal['bbox_xyxy'] and row['score']==score
        else:assert not detected
    assert fit_count==480 and {p.name for p in (base/'fits').glob('*.json')}==expected_files
    for filename,expected in hashes.items():assert digest(path(filename))==expected
    save(HERE/'ours_replay_audit.json',dict(passed=True,scenes=8,requests=96,
        fresh_depth_proposals=112,CLIP_candidate_pairs=1344,CLIP_CAD_views_per_pair=20,
        exact_shortlist_size=5,fit_pairs=480,pose_hypotheses_verified=pose_count,
        decision_statuses=dict(statuses),exported_unique_geometry_winners=selected_count,
        mask_frame='native observed pinhole camera; xright,ydown,zforward; integerpixels,depthmeters',
        ROI_depth_masks_rebuilt_exact=True,clip_four_terms_and_ranking_replayed=True,
        every_best_pose_and_candidate_selection_replayed=True,
        max_rotation_orthogonality_error=max_rotation_error,max_contact_height_error_m=max_height_error,
        source_sha256=digest(__file__),sha256=hashes,ground_truth_or_accuracy_computed=False))
    print(dict(passed=True,fit_pairs=fit_count,pose_hypotheses=pose_count,statuses=dict(statuses)),flush=True)


if __name__=='__main__':main()
