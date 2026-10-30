"""Real predicted-mask inference and durable per-image failure/prediction ledger."""
from __future__ import annotations
from pathlib import Path
import copy
import time
import traceback
import numpy as np
from PIL import Image
from .data import sha256
from .durable import atomic_json,atomic_npz,digest,read_json,require_disk,ensure_identity
from .geometry import observed_clouds
from .features import sample_feature_map,fuse_features
from .registration import feature_ransac,refine_icp,score_pose,joint_score,rank_candidates
from .pipeline import _draw_pose


def _pose(R,t):return {'R_m2c':np.asarray(R).tolist(),'t_m':np.asarray(t).tolist()}


def _oom(error):return 'out of memory' in str(error).lower()


class InvalidCUDAContext(RuntimeError):
    """The process must restart before any other GPU case is valid."""


def infer_candidate(candidate,frame,bundle,recipe,backends,folder,resources,heartbeat=None):
    """Never substitute descriptors/poses. OOM only changes parallel evaluation."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity={'recipe':digest(recipe),'candidate':candidate.provenance(),'source':frame['source'],
              'CAD_feature_identity':bundle['feature_identity']}
    checkpoint=folder/'diagnostic.json'
    prior_runtime_s=0.;prior_blocks=[]
    if checkpoint.exists():
        previous=read_json(checkpoint)
        if previous.get('identity')!=identity:raise ValueError('Candidate checkpoint belongs to different source/features/method')
        for name,checksum in previous.get('artifact_files',{}).items():
            if sha256(folder/name)!=checksum:raise ValueError('Candidate checkpoint artifact changed')
        if previous['status']!='blocked':
            previous['_checkpoint_reused']=True
            return previous
        prior_runtime_s=previous['runtime_s']
        prior_blocks=previous.get('previous_blocked_attempts',[])+[{'type':previous.get('failure_type'),'reason':previous.get('reason'),'runtime_s':previous['runtime_s']}]
    trace={'candidate':candidate.provenance(),'source':frame['source'],'status':'started',
           'identity':identity,
           'pose_direction':'original_CAD_to_camera','units':'metres','model_sha256':bundle['model']['sha256'],
           'frozen_CAD_feature_identity':bundle['feature_identity'],'resource_attempts':[],
           'previous_blocked_attempts':prior_blocks}
    Image.fromarray(candidate.mask.astype(np.uint8)*255).save(folder/'mask.png')
    started=time.perf_counter();invalid_cuda_context=False
    config=copy.deepcopy(recipe['registration'])
    config['hypothesis_batch_size']=resources['ransac_batch_size']
    dino,gedi=backends
    for attempt in range(resources.get('oom_retries',3)+1):
        try:
            if heartbeat:heartbeat(stage='candidate_features',candidate_id=candidate.candidate_id,attempt=attempt)
            if candidate.mask.shape!=frame['depth_m'].shape:raise ValueError('Predicted-mask/sensor dimensions differ')
            cloud=observed_clouds(frame['depth_m'],candidate.mask,frame['K'],grid=recipe['observed']['grid'],dense_count=recipe['observed']['dense_points'],seed=recipe['seed'])
            crop=cloud['crop'].extract_rgb(frame['rgb'])
            Image.fromarray(crop).save(folder/'crop.png')
            visual=sample_feature_map(dino.extract(crop),cloud['feature_uv'],image_size=crop.shape[:2])
            geometric=gedi.compute(cloud['sparse'],cloud['dense'],bundle['model']['diameter_m'])
            fused=fuse_features(bundle['pca'].transform(visual),geometric)
            if fused.shape!=(len(cloud['sparse']),128) or not np.isfinite(fused).all():raise ValueError('Real observed frozen feature contract failed')
            if heartbeat:heartbeat(stage='global_registration',candidate_id=candidate.candidate_id)
            coarse=feature_ransac(bundle['points_m'],cloud['sparse'],bundle['features'],fused,bundle['model']['diameter_m'],config)
            trace['registration']=coarse.diagnostics
            if coarse.status!='ok':
                trace.update(status='failed',failure_type='no_valid_global_hypothesis',reason=coarse.status)
                break
            refined=refine_icp(bundle['points_m'],cloud['dense'],coarse.R,coarse.t,bundle['model']['diameter_m'],config)
            scores=score_pose(bundle['points_m'],cloud['sparse'],refined.R,refined.t,bundle['model']['diameter_m'],coarse.correspondences,dense_xyz=cloud['dense'],config=config)
            score=joint_score(coarse.score,scores.feature_score,scores.geometry_score)
            if not np.isfinite(score):raise ValueError('Nonfinite full base pose score')
            trace.update(status='complete',coarse_pose=_pose(coarse.R,coarse.t),refined_pose=_pose(refined.R,refined.t),
                         coarse_feature_score=coarse.score,refined_feature_score=scores.feature_score,geometry_score=scores.geometry_score,
                         joint_score=score,icp_status=refined.status,icp=refined.diagnostics,
                         sparse_points=len(cloud['sparse']),dense_points=len(cloud['dense']),
                         visual_shape=list(visual.shape),geometric_shape=list(geometric.shape),fused_shape=list(fused.shape))
            atomic_npz(folder/'observed.npz',sparse_m=cloud['sparse'],dense_m=cloud['dense'],fused=fused,
                       correspondence_indices=coarse.correspondences.cad_indices,correspondence_cosines=coarse.correspondences.similarities)
            _draw_pose(frame['rgb'],bundle['points_m'],frame['K'],coarse,refined,folder/'pose_overlay.png')
            trace['pose']={'candidate_id':candidate.candidate_id,'obj_id':candidate.object_id,'R':refined.R.tolist(),'t':refined.t.tolist(),'score':score,'status':'ok'}
            break
        except (ValueError,RuntimeError,MemoryError) as exc:
            trace['resource_attempts'].append({'attempt':attempt,'hypothesis_batch_size':config['hypothesis_batch_size'],'type':type(exc).__name__,'reason':str(exc)})
            if _oom(exc) and attempt<resources.get('oom_retries',3):
                # Only reduce RANSAC hypothesis parallelism. Never alter top-k,
                # iteration budget, thresholds, precision or GeDi sampling batches.
                config['hypothesis_batch_size']=max(1,config['hypothesis_batch_size']//2)
                if hasattr(gedi,'set_forward_microbatch_size'):
                    current=gedi.provenance.get('resources',{}).get('network_forward_microbatch_size',32)
                    gedi.set_forward_microbatch_size(max(16,current//2))
                import torch
                torch.cuda.empty_cache()
                continue
            infrastructure=_oom(exc) or any(s in str(exc).lower() for s in ('illegal memory access','device-side assert','undefined symbol','no kernel image','cuda driver','cudnn_status','cublas_status'))
            invalid_cuda_context=any(s in str(exc).lower() for s in ('illegal memory access','device-side assert'))
            trace.update(status='blocked' if infrastructure else 'failed',failure_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
            break
    trace['runtime_s']=time.perf_counter()-started+prior_runtime_s
    trace['artifact_files']={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='diagnostic.json'}
    atomic_json(folder/'diagnostic.json',trace)
    if invalid_cuda_context:raise InvalidCUDAContext('CUDA context invalid; durable candidate failure saved, fresh process required')
    trace['_prior_runtime_s']=prior_runtime_s
    return trace


def infer_image(dataset,scene_id,image_id,targets,bundles,index,recipe,backends,folder,identity,resources,heartbeat=None):
    """Transaction boundary is an entire image; all its CSV times match."""
    folder=Path(folder)
    complete=folder/'complete.json'
    expected_identity={'run':identity,'scene_id':scene_id,'image_id':image_id,
                       'target_rows':[[t.object_id,t.instance_count] for t in targets]}
    if complete.exists():
        value=read_json(complete)
        if value['identity']!=expected_identity:raise ValueError('Saved image result belongs to a different method/target set')
        for target in value['targets']:
            for candidate in target.get('candidates',[]):
                directory=folder/candidate['directory']
                if sha256(directory/'diagnostic.json')!=candidate['diagnostic_sha256']:raise ValueError('Saved candidate trace changed')
        return value
    require_disk(folder,resources['minimum_free_disk_gb'])
    ensure_identity(folder,expected_identity)
    started=time.perf_counter();results=[];predictions=[];reused_runtime_s=0.
    try:frame=dataset.frame(scene_id,image_id)
    except (ValueError,OSError,RuntimeError) as exc:
        for target in targets:results.append({'object_id':target.object_id,'requested_instances':target.instance_count,'status':'failed','valid_predictions':0,'reason':f'RGBD_input:{type(exc).__name__}:{exc}','candidates':[]})
        value={'identity':expected_identity,'status':'complete_with_failures','targets':results,'predictions':[],
               'runtime_s':time.perf_counter()-started,'GT_pose_access':False}
        atomic_json(complete,value);return value
    for target in targets:
        if heartbeat:heartbeat(stage='image_target',scene_id=scene_id,image_id=image_id,object_id=target.object_id)
        if target.object_id not in bundles:
            results.append({'object_id':target.object_id,'requested_instances':target.instance_count,'status':'blocked',
                            'valid_predictions':0,'reason':'CAD_onboarding_unavailable','candidates':[]});continue
        bundle=bundles[target.object_id]
        try:pool=index.candidates(scene_id,image_id,target.object_id,target.instance_count)
        except (ValueError,OSError) as exc:
            results.append({'object_id':target.object_id,'requested_instances':target.instance_count,'status':'failed',
                            'valid_predictions':0,'reason':f'predicted_masks:{exc}','candidates':[]});continue
        traces=[];poses=[]
        for candidate in pool:
            relative=Path(f'obj_{target.object_id:06d}')/f'{candidate.segmentor}_{candidate.record_index:06d}'
            trace=infer_candidate(candidate,frame,bundle,recipe,backends,folder/relative,resources,heartbeat)
            if trace.get('_checkpoint_reused'):reused_runtime_s+=trace['runtime_s']
            else:reused_runtime_s+=trace.get('_prior_runtime_s',0.)
            traces.append({'candidate_id':candidate.candidate_id,'directory':str(relative),'status':trace['status'],
                           'diagnostic_sha256':sha256(folder/relative/'diagnostic.json')})
            if trace['status']=='complete':poses.append(trace['pose'])
        ranked=rank_candidates(poses,translation_nms_threshold_m=recipe['candidate_ranking']['translation_nms_diameter_fraction']*bundle['model']['diameter_m'],max_instances=target.instance_count)
        for pose in ranked.selected:
            predictions.append({'scene_id':scene_id,'image_id':image_id,'object_id':target.object_id,
                                'R_m2c':pose['R'],'t_m':pose['t'],'score':pose['score'],'candidate_id':pose['candidate_id']})
        successful=len(ranked.selected)==target.instance_count
        resource_blocked=any(t['status']=='blocked' for t in traces)
        results.append({'object_id':target.object_id,'requested_instances':target.instance_count,'status':'blocked' if resource_blocked else ('completed' if successful else 'failed'),
                        'valid_predictions':len(ranked.selected),'candidate_count':len(pool),'candidates':traces,
                        'reason':'candidate_runtime_requires_retry' if resource_blocked else (None if successful else ('no_predicted_masks' if not pool else 'fewer_valid_distinct_poses_than_requested')),
                        'suppressed':ranked.suppressed,'rejected':ranked.rejected})
    elapsed=time.perf_counter()-started+reused_runtime_s
    for pose in predictions:pose['image_time_s']=elapsed
    value={'identity':expected_identity,'status':'complete' if all(t['status']=='completed' for t in results) else 'complete_with_failures',
           'targets':results,'predictions':predictions,'source':frame['source'],'runtime_s':elapsed,
           'reused_candidate_runtime_s':reused_runtime_s,
           'runtime_policy':'active current image wall time plus retained completed-candidate compute time; interruption downtime excluded',
           'GT_pose_access':False}
    atomic_json(complete if not any(t['status']=='blocked' for t in results) else folder/'partial.json',value)
    return value
