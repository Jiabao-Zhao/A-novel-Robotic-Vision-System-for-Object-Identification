"""Controlled finite-row GeDi pose ablation; no GT access during inference."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'2')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, csv, hashlib, json, sys, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/freezev2_bop_20261029'
FROZEN = BASE/'classic_core_base_v2'
sys.path.insert(0,str(FROZEN/'source_snapshot'))
from freezev2_repro.data import BOPDataset,sha256
from freezev2_repro.durable import atomic_json,atomic_npz,digest,ensure_identity,read_json,require_disk,worker_lock
from freezev2_repro.features import FrozenDINOv2,FrozenGeDi,PCATransform,sample_feature_map,fuse_features
from freezev2_repro.geometry import observed_clouds
from freezev2_repro.mask_index import MaskIndex
from freezev2_repro.registration import feature_ransac,refine_icp,joint_score,rank_candidates
from freezev2_repro.evaluation import PosePrediction,write_bop_csv,validate_bop_csv
from freezev2_repro.pipeline import _draw_pose
from gedi_valid_rows import extract_valid_rows
from pose_alignment import align_rows,paired_full_geometry_score
from visual_ablation import CADVisualAdapter,aggregate_cached_visuals


def local(value):
    value=str(value).replace('\\','/')
    if os.name!='nt' and len(value)>2 and value[1]==':': value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)


def array_sha(value):
    v=np.ascontiguousarray(value)
    return digest({'dtype':str(v.dtype),'shape':list(v.shape),'bytes':hashlib.sha256(v.tobytes()).hexdigest()})


def checked_npz(folder, identity, arrays=None):
    folder=Path(folder);ensure_identity(folder,identity)
    receipt=folder/'complete.json';path=folder/'arrays.npz'
    if receipt.exists():
        r=read_json(receipt)
        if r['sha256']!=sha256(path): raise ValueError('Changed array checkpoint: '+str(path))
        with np.load(path,allow_pickle=False) as d: return {k:d[k] for k in d.files}
    if arrays is None:return None
    atomic_npz(path,**arrays);atomic_json(receipt,{'sha256':sha256(path),'identity':identity})
    return arrays


def geometric(gedi,points,support,diameter,folder,identity):
    key={**identity,'points_sha256':array_sha(points),'support_sha256':array_sha(support),'diameter_m':diameter,'policy':'finite_prepared_rows_intersection_v1'}
    found=checked_npz(folder,key)
    if found is not None:return found
    import torch
    result=extract_valid_rows(gedi.backend,torch.from_numpy(points.astype(np.float32)),torch.from_numpy(support.astype(np.float32)),[.3*diameter,.4*diameter],seed=0)
    if any(s['zero_descriptor_indices'] for s in result['scales']):raise ValueError('Zero geometric descriptor; no fallback is authorized')
    diag={k:v for k,v in result.items() if k!='descriptors'}
    atomic_json(Path(folder)/'validity.json',diag)
    return checked_npz(folder,key,{'original_indices':result['original_indices'],'geometric':result['descriptors']})


def cad_bundle(obj,arm,dataset,recipe,dino,gedi,out,identity,heartbeat):
    t0=time.perf_counter();old=FROZEN/f'datasets/tless/onboarding/obj_{obj:06d}'
    model=dataset.model(obj);geometry=old/'geometry/geometry.npz'
    complete=read_json(old/'geometry/complete.json')
    if sha256(geometry)!=complete['files']['geometry.npz']:raise ValueError('Frozen geometry changed')
    with np.load(geometry,allow_pickle=False) as d:points=d['points_m'];support=d['support_points_m']
    geo=geometric(gedi,points,support,model['diameter_m'],out/f'shared/cad_{obj:06d}_geometric',{'source':identity['gedi'],'geometry_sha256':sha256(geometry)})
    target=out/arm/f'onboarding/obj_{obj:06d}'
    bundle_identity={'run':identity['digest'],'arm':arm,'object_id':obj,'geometry_sha256':sha256(geometry),'geometric':array_sha(geo['geometric'])}
    found=checked_npz(target,bundle_identity)
    if found is None:
        mode='baseline224' if arm=='repair224' else 'cad476'
        if mode=='baseline224' and (old/'features/complete.json').exists():
            old_receipt=read_json(old/'features/complete.json')
            for name in ('features.npz','pca.npz'):
                if sha256(old/'features'/name)!=old_receipt['files'][name]:raise ValueError('Frozen healthy CAD feature hash mismatch')
            with np.load(old/'features/features.npz',allow_pickle=False) as d:visual=d['visual_mean'];counts=d['view_counts']
            pca=PCATransform.load(old/'features/pca.npz')
            provenance={'reused_exact_baseline_visual_PCA':True}
        else:
            aggregated=aggregate_cached_visuals(old/'geometry',target/'visual',CADVisualAdapter(dino,mode=mode),min_views=18,baseline_map_folder=old/'features' if mode=='baseline224' else None,heartbeat=heartbeat)
            visual=aggregated['visual_mean'];counts=aggregated['view_counts']
            pca=PCATransform(64).fit(visual)
            provenance={'reused_exact_baseline_visual_PCA':False,'visual_identity':aggregated['identity']}
        # PCA uses ALL original visual rows before the GeDi validity subset.
        projected=pca.transform(visual)
        rows=align_rows(points,projected,geo['geometric'],geo['original_indices'])
        fused=fuse_features(rows['visual'],rows['geometric'])
        found=checked_npz(target,bundle_identity,{'points_m':rows['points'],'full_points_m':points,'original_indices':rows['original_indices'],'features':fused,'PCA_mean':pca.mean_,'PCA_components':pca.components_})
        comparison=None
        if arm=='repair224' and len(rows['points'])==len(points) and (old/'features/complete.json').exists():
            with np.load(old/'features/features.npz',allow_pickle=False) as d:original=d['fused']
            comparison={'max_absolute_difference':float(np.max(np.abs(original-fused))),'exactly_equal':bool(np.array_equal(original,fused))}
            if not np.array_equal(original,fused):raise ValueError('Healthy all-valid CAD changed despite unchanged DINO/GeDi')
        atomic_json(target/'onboarding.json',{'identity':bundle_identity,'original_points':len(points),'retained_feature_points':len(rows['points']),'removed_original_indices':np.setdiff1d(np.arange(len(points)),rows['original_indices']).tolist(),'PCA_fit_rows':len(visual),'PCA_scope':'all original CAD visual rows only, before finite-GeDi subset','ICP_geometry_rows':len(points),'geometry_score_denominator':len(points),'baseline_comparison':comparison,'provenance':provenance,'runtime_s':time.perf_counter()-t0})
    pca=PCATransform(64);pca.mean_=found['PCA_mean'];pca.components_=found['PCA_components']
    return {**found,'pca':pca,'model':model,'identity':digest(bundle_identity)}


def infer_candidate(candidate,frame,bundle,recipe,dino,gedi,folder,shared,identity,heartbeat):
    folder.mkdir(parents=True,exist_ok=True)
    key={'run':identity['digest'],'CAD':bundle['identity'],'candidate':candidate.provenance(),'source':frame['source']}
    ensure_identity(folder,key)
    if (folder/'diagnostic.json').exists():
        trace=read_json(folder/'diagnostic.json')
        for name,h in trace['artifacts'].items():
            if sha256(folder/name)!=h:raise ValueError('Changed saved candidate artifact')
        return trace
    start=time.perf_counter();trace={'identity':key,'candidate':candidate.provenance(),'GT_pose_access':False,'status':'started'}
    try:
        Image.fromarray(candidate.mask.astype(np.uint8)*255).save(folder/'mask.png')
        cloud=observed_clouds(frame['depth_m'],candidate.mask,frame['K'],grid=16,dense_count=3000,seed=0)
        crop=cloud['crop'].extract_rgb(frame['rgb']);Image.fromarray(crop).save(folder/'crop.png')
        sharedkey={'candidate':candidate.provenance(),'source':frame['source'],'dino':identity['dino'],'original_sparse_sha256':array_sha(cloud['sparse'])}
        cached=checked_npz(shared/'visual224',sharedkey)
        if cached is None:
            visual=sample_feature_map(dino.extract(crop),cloud['feature_uv'],image_size=crop.shape[:2])
            cached=checked_npz(shared/'visual224',sharedkey,{'visual':visual})
        visual=cached['visual']
        geo=geometric(gedi,cloud['sparse'],cloud['dense'],bundle['model']['diameter_m'],shared/'geometric',{'source':identity['gedi'],'candidate':candidate.provenance()})
        rows=align_rows(cloud['sparse'],bundle['pca'].transform(visual),geo['geometric'],geo['original_indices'])
        fused=fuse_features(rows['visual'],rows['geometric']);diameter=bundle['model']['diameter_m']
        heartbeat(stage='RANSAC',candidate=candidate.candidate_id)
        coarse=feature_ransac(bundle['points_m'],rows['points'],bundle['features'],fused,diameter,recipe['registration'])
        trace['registration']=coarse.diagnostics
        if coarse.status!='ok':raise ValueError('RANSAC:'+coarse.status)
        # Full original CAD/dense support remain intact for refinement and geometry.
        refined=refine_icp(bundle['full_points_m'],cloud['dense'],coarse.R,coarse.t,diameter,recipe['registration'])
        scores=paired_full_geometry_score(bundle['points_m'],rows['points'],refined.R,refined.t,diameter,coarse.correspondences,bundle['full_points_m'],cloud['dense'],recipe['registration'])
        np.testing.assert_allclose(scores['geometry_score'],refined.geometry_score,rtol=0,atol=1e-15)
        joint=joint_score(coarse.score,scores['feature_score'],scores['geometry_score'])
        pose={'candidate_id':candidate.candidate_id,'obj_id':candidate.object_id,'R':refined.R.tolist(),'t':refined.t.tolist(),'score':joint,'status':'ok'}
        trace.update(status='complete',pose=pose,coarse_pose={'R_m2c':coarse.R.tolist(),'t_m':coarse.t.tolist()},refined_pose={'R_m2c':refined.R.tolist(),'t_m':refined.t.tolist()},coarse_feature_score=coarse.score,refined_feature_score=scores['feature_score'],geometry_score=scores['geometry_score'],joint_score=joint,score_detail=scores,icp=refined.diagnostics,icp_status=refined.status,original_sparse_rows=len(cloud['sparse']),retained_sparse_rows=len(rows['points']),removed_sparse_indices=np.setdiff1d(np.arange(len(cloud['sparse'])),rows['original_indices']).tolist(),original_CAD_rows=len(bundle['full_points_m']),retained_CAD_rows=len(bundle['points_m']),dense_rows=len(cloud['dense']))
        atomic_npz(folder/'observed.npz',original_sparse_m=cloud['sparse'],original_indices=rows['original_indices'],sparse_m=rows['points'],dense_m=cloud['dense'],fused=fused,correspondence_indices=coarse.correspondences.cad_indices,correspondence_cosines=coarse.correspondences.similarities,CAD_original_indices=bundle['original_indices'])
        _draw_pose(frame['rgb'],bundle['full_points_m'],frame['K'],coarse,refined,folder/'pose_overlay.png')
    except (ValueError,RuntimeError,MemoryError) as exc:
        if any(s in str(exc).lower() for s in ('out of memory','device-side assert','illegal memory access')):raise
        trace.update(status='failed',failure_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    trace['runtime_s']=time.perf_counter()-start
    trace['artifacts']={p.name:sha256(p) for p in folder.iterdir() if p.name not in ('diagnostic.json','identity.json') and p.is_file()}
    atomic_json(folder/'diagnostic.json',trace)
    return trace


def run_arm(arm,targets,dataset,index,recipe,dino,gedi,out,identity,heartbeat):
    bundles={};errors={}
    for obj in sorted({t['obj_id'] for t in targets}):
        heartbeat(stage='onboarding',arm=arm,object_id=obj)
        try:bundles[obj]=cad_bundle(obj,arm,dataset,recipe,dino,gedi,out,identity,heartbeat)
        except (ValueError,RuntimeError,MemoryError) as exc:
            if any(s in str(exc).lower() for s in ('out of memory','device-side assert','illegal memory access')):raise
            errors[obj]={'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()}
    grouped=defaultdict(list)
    for t in targets:grouped[(t['scene_id'],t['im_id'])].append(t)
    predictions=[];ledgers=[]
    for (scene,image),rows in sorted(grouped.items()):
        imagefolder=out/arm/f'images/{scene:06d}_{image:06d}'
        imagekey={'run':identity['digest'],'arm':arm,'targets':rows}
        ensure_identity(imagefolder,imagekey)
        path=imagefolder/'complete.json'
        if path.exists():
            ledger=read_json(path)
            for item in ledger['targets']:
                for c in item.get('candidates',[]):
                    if sha256(imagefolder/c['directory']/'diagnostic.json')!=c['sha256']:raise ValueError('Candidate trace changed')
        else:
            start=time.perf_counter();frame=dataset.frame(scene,image);selected=[];records=[]
            for target in rows:
                obj=target['obj_id'];traces=[];poses=[]
                if obj not in bundles:
                    records.append({**target,'status':'blocked','predictions':0,'failure':errors[obj],'candidates':[]});continue
                for candidate in index.candidates(scene,image,obj,target['inst_count']):
                    relative=Path(f'obj_{obj:06d}/{candidate.segmentor}_{candidate.record_index:06d}')
                    heartbeat(stage='candidate',arm=arm,scene_id=scene,image_id=image,object_id=obj,record=candidate.record_index)
                    shared=out/f'shared/observed/{scene:06d}_{image:06d}_{obj:06d}_{candidate.record_index:06d}'
                    trace=infer_candidate(candidate,frame,bundles[obj],recipe,dino,gedi,imagefolder/relative,shared,identity,heartbeat)
                    traces.append({'directory':str(relative),'candidate_id':candidate.candidate_id,'status':trace['status'],'sha256':sha256(imagefolder/relative/'diagnostic.json')})
                    if trace['status']=='complete':poses.append(trace['pose'])
                ranked=rank_candidates(poses,translation_nms_threshold_m=.1*bundles[obj]['model']['diameter_m'],max_instances=target['inst_count'])
                for pose in ranked.selected:selected.append({'scene_id':scene,'image_id':image,'object_id':obj,'score':pose['score'],'R_m2c':pose['R'],'t_m':pose['t'],'candidate_id':pose['candidate_id']})
                records.append({**target,'status':'completed' if len(ranked.selected)==target['inst_count'] else 'failed','predictions':len(ranked.selected),'candidates':traces,'suppressed':ranked.suppressed,'rejected':ranked.rejected})
            elapsed=time.perf_counter()-start
            for p in selected:p['image_time_s']=elapsed
            ledger={'identity':imagekey,'targets':records,'predictions':selected,'runtime_s':elapsed,'source':frame['source'],'GT_pose_access':False}
            atomic_json(path,ledger)
        ledgers+=ledger['targets'];predictions+=ledger['predictions']
    csvpath=out/arm/'freezev2cohort_tless-test.csv'
    if not csvpath.exists():
        if predictions:write_bop_csv([PosePrediction(p['scene_id'],p['image_id'],p['object_id'],p['score'],np.asarray(p['R_m2c']),np.asarray(p['t_m']),p['image_time_s']) for p in predictions],csvpath)
        else:csvpath.write_text('scene_id,im_id,obj_id,score,R,t,time\n')
    result={'arm':arm,'expected_target_rows':len(targets),'requested_instances':sum(t['inst_count'] for t in targets),'completed_rows':sum(t['status']=='completed' for t in ledgers),'failed_rows':sum(t['status']=='failed' for t in ledgers),'blocked_rows':sum(t['status']=='blocked' for t in ledgers),'predictions':len(predictions),'targets':ledgers,'CSV':str(csvpath),'CSV_sha256':sha256(csvpath),'GT_pose_access':False,'metric_status':'not_evaluated_by_inference'}
    if predictions:result['CSV_validation']=validate_bop_csv(csvpath,ROOT/'outputs/cache/bop_toolkit')
    atomic_json(out/arm/'inference_summary.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));parser.add_argument('--arm',choices=['repair224','repair476','all'],default='all');args=parser.parse_args()
    config=read_json(args.config);cohortpath=Path(__file__).with_name('cohort.json');cohort=read_json(cohortpath)
    protocol=read_json(FROZEN/'protocol.json');recipe=protocol['base'];out=local(config['output']);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('Trained CUDA inference is required')
    sources={name:sha256(Path(__file__).with_name(name)) for name in ('run_pose_ablation.py','gedi_valid_rows.py','visual_ablation.py','pose_alignment.py')}
    # Verify every frozen source pin; never execute an altered baseline.
    manifest=read_json(FROZEN/'run_manifest.json')
    for name,h in manifest['method_identity']['code'].items():
        if sha256(FROZEN/'source_snapshot/freezev2_repro'/name)!=h:raise ValueError('Frozen source changed:'+name)
    f=recipe['features'];identity={'method':'finite-GeDi-rows-controlled-ablation-v1','sources':sources,'cohort_sha256':sha256(cohortpath),'config_sha256':sha256(args.config),'base_recipe':recipe,'frozen_source_manifest_sha256':sha256(FROZEN/'source_snapshot/source_manifest.json'),'dino':{k:v for k,v in f.items() if k.startswith('dino')},'gedi':{k:v for k,v in f.items() if k.startswith('gedi')},'baseline_final_results_sha256':sha256(FROZEN/'final_results.json')};identity['digest']=digest(identity)
    def heartbeat(**values):
        row={'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),**values};atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    with worker_lock(out/'worker.lock'):
        if (out/'run_manifest.json').exists():
            if read_json(out/'run_manifest.json')['identity']!=identity:raise ValueError('Changed run source/config: use a new output directory')
        else:atomic_json(out/'run_manifest.json',{'identity':identity,'protocol':'seven fixed original official rows; all failures in evaluation denominator','CAD_geometry_policy':'full 5000 original CAD points for ICP and geometry score; valid feature subset only for correspondences','PCA_policy':'fit all original CAD visual rows before GeDi mask; no test observations/GT','arms':config['arms'],'public_reference_revision':config['public_reference_revision']})
        require_disk(out,15)
        heartbeat(stage='loading_trained_models')
        p=protocol['paths']
        dino=FrozenDINOv2(local(p['dino_source']),local(p['dino_checkpoint']),layer_index=f['dino_layer_index'],checkpoint_sha256=f['dino_checkpoint_sha256'],expected_source_hashes=f['dino_expected_source_hashes'],source_revision_hint=f['dino_source_revision_hint'])
        gedi=FrozenGeDi(local(p['gedi_source']),local(p['gedi_checkpoint']),expected_source_sha256=f['gedi_source_sha256'],expected_revision=f['gedi_source_revision'],checkpoint_sha256=f['gedi_checkpoint_sha256'],forward_microbatch_size=32)
        dataset=BOPDataset(local(p['dataset_root']),'tless')
        asset=read_json(BASE/'assets_manifest.json')['datasets']['tless']['masks']
        source={'path':str(local(asset['path'])),'segmentor':'cnos-fastsam','source_id':'public_BOP23_default_CNOS_FastSAM:'+asset['sha256'],'sha256':asset['sha256'],'source_kind':'predicted'}
        index=MaskIndex(FROZEN/'datasets/tless/mask_index.sqlite',[source])
        try:
            for arm in config['arms'] if args.arm=='all' else [args.arm]:run_arm(arm,cohort['targets'],dataset,index,recipe,dino,gedi,out,identity,heartbeat)
        finally:index.close()
        heartbeat(stage='inference_finished',peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),metric_status='evaluation_is_separate')

if __name__=='__main__':main()