"""Auxiliary CPU-only projections of saved own-inference diagnostics.

No GT is read. Every projected CAD query point is drawn without hidden-surface
culling. This helper does not estimate, refine, rank, evaluate, or change poses.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import textwrap
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DATASETS=('lmo','tless','tudl','icbin','itodd','hb','ycbv')
RULE={'version':'receipt-order-v2','dataset_order':list(DATASETS),
      'eligible_statuses':['complete','failed','blocked'],
      'order':'ascending scene_id,im_id,obj_id,target receipt path; no scores or GT',
      'categories':['first_available','first_with_saved_pose','first_multi_instance','first_no_pose','first_failed_or_blocked'],
      'multi_instance':'official requested inst_count>1 OR saved selected_count>1',
      'no_pose':'saved selected_count=0; failures/blocks remain eligible',
      'deduplication':'one case may satisfy several categories; retain every category reason'}
COLORS=((255,76,55),(255,190,40),(205,85,255),(80,230,100),(255,110,185),(60,130,255))


def local(value,resolve=True):
    value=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    path=Path(value).expanduser()
    return path.resolve() if resolve else Path(os.path.abspath(path))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    os.replace(temporary,path)


def checked(path,root,label):
    path=local(path)
    if not path.is_relative_to(local(root)):raise ValueError(label+' escaped its GT-free allowed input root')
    if any(p.startswith('scene_gt') or p in ('mask_visib','mask_gt') for p in path.parts):
        raise ValueError('GT annotation access is prohibited')
    return path



def authorized_frame(config,receipt,dataset,scene,image,records,missing):
    paths=config['paths'];asset_path=local(paths['assets_output'])/'assets_manifest.json'
    if not input_record(asset_path,'inference_asset_manifest',records,missing):
        raise ValueError('Verified inference asset manifest is required for RGB/camera aliases')
    if records['inference_asset_manifest']['sha256']!=receipt.get('source_pins',{}).get('assets_manifest_sha256'):
        raise ValueError('Inference asset manifest differs from target completion receipt')
    assets=read(asset_path)
    if assets.get('GT_pose_access') is not False or assets.get('GT_annotation_links')!=0:
        raise ValueError('Only the recorded GT-free inference view is allowed')
    matches=[f for f in assets['datasets'][dataset]['frames'] if f['scene_id']==scene and f['im_id']==image]
    if len(matches)!=1:raise ValueError('Target image missing or duplicated in verified inference asset manifest')
    frame=matches[0];view=local(paths['bop_root'],resolve=False)
    logical_scene=view/dataset/'test'/f'{scene:06d}'
    rgb=local(logical_scene/'rgb'/f'{image:06d}.png',resolve=False)
    camera=local(logical_scene/'scene_camera.json',resolve=False)
    for path,source,expected in ((rgb,frame['rgb']['source_path'],frame['rgb']['source_sha256']),
                                 (camera,frame['camera_source_path'],frame['camera_sha256'])):
        if not path.is_relative_to(view) or path.resolve()!=local(source):
            raise ValueError('Inference image/camera alias no longer points to its authorized physical source')
        if any(part.startswith('scene_gt') or part in ('mask_visib','mask_gt') for part in path.resolve().parts):
            raise ValueError('GT annotation alias is prohibited')
        if sha(path)!=expected:raise ValueError('Authorized RGB/camera source checksum differs')
    if rgb!=local(frame['rgb']['alias_path'],resolve=False):raise ValueError('RGB alias differs from verified view path')
    return rgb,camera

def font(size):
    path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default()


def select_cases(own_root):
    """Read receipt metadata only; scores never enter representative selection."""
    own_root=local(own_root);selected=[]
    for dataset in DATASETS:
        eligible=[]
        for path in sorted((own_root/'datasets'/dataset/'targets').glob('*/receipt.json')):
            try:
                receipt=read(path);target=receipt['target']
                count=receipt.get('selected_count',len(receipt.get('poses',[])))
                if not isinstance(count,int) or isinstance(count,bool) or count<0:continue
                if receipt.get('status') not in RULE['eligible_statuses']:continue
                ids=tuple(int(target[k]) for k in ('scene_id','im_id','obj_id'))
                eligible.append((ids,str(path),receipt))
            except (OSError,ValueError,KeyError,TypeError):continue
        groups={}
        for ids,path,receipt in sorted(eligible):
            count=receipt.get('selected_count',len(receipt.get('poses',[])))
            categories=['first_available']
            if count>0:categories.append('first_with_saved_pose')
            if int(receipt['target'].get('inst_count',0))>1 or count>1:categories.append('first_multi_instance')
            if count==0:categories.append('first_no_pose')
            if receipt['status'] in ('failed','blocked'):categories.append('first_failed_or_blocked')
            for category in categories:
                if category not in groups:groups[category]=(ids,path,receipt)
        merged={}
        for category in RULE['categories']:
            if category not in groups:continue
            ids,path,receipt=groups[category]
            case=merged.setdefault(path,{'dataset':dataset,'receipt_path':path,'target':receipt['target'],
                'status':receipt['status'],'reasons':[],'receipt_sha256':sha(path)})
            case['reasons'].append(category)
        selected.extend(sorted(merged.values(),key=lambda c:(c['target']['scene_id'],c['target']['im_id'],c['target']['obj_id'])))
    return selected


def input_record(path,label,records,missing):
    if not Path(path).is_file():missing.append({'role':label,'path':str(path),'state':'missing'});return False
    records[label]={'path':str(path),'sha256':sha(path),'bytes':Path(path).stat().st_size}
    return True


def projection_api(source):
    spec=importlib.util.spec_from_file_location('_diagnostic_original_geometry_'+sha(source)[:12],source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.project_points


def project(points,K,pose,projector,size):
    R=np.asarray(pose['R'],dtype=np.float64).reshape(3,3);t=np.asarray(pose['t_mm'],dtype=np.float64).reshape(3)
    if not np.isfinite(R).all() or not np.isfinite(t).all():raise ValueError('Saved pose is nonfinite')
    uv,z=projector(points,K,R,t);uv=np.asarray(uv);z=np.asarray(z)
    width,height=size
    keep=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    return uv[keep],{'R':R.tolist(),'t_mm':t.tolist(),'query_count':len(points),
        'positive_finite_count':int((np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)).sum()),'in_frame_count':int(keep.sum())}


def outline(mask):
    image=Image.fromarray(np.uint8(mask)*255)
    eroded=np.asarray(image.filter(ImageFilter.MinFilter(3)))>0
    return Image.fromarray(np.uint8(mask&~eroded)*255).filter(ImageFilter.MaxFilter(3))


def overlay(rgb,masks,projections):
    result=rgb.convert('RGBA')
    for mask,color in masks:
        layer=Image.new('RGBA',rgb.size,(*color,0));layer.putalpha(outline(mask).point(lambda p:int(p*.85)))
        result=Image.alpha_composite(result,layer)
    for uv,color in projections:
        dots=Image.new('L',rgb.size,0)
        ImageDraw.Draw(dots).point([(int(round(x)),int(round(y))) for x,y in uv],fill=160)
        dots=dots.filter(ImageFilter.MaxFilter(3))
        layer=Image.new('RGBA',rgb.size,(*color,0));layer.putalpha(dots)
        result=Image.alpha_composite(result,layer)
    return result.convert('RGB')


def figure(left,right,titles,lines,path):
    width=min(640,left.width);height=round(left.height*width/left.width)
    panels=[p.resize((width,height)) for p in (left,right)]
    wrapped=[]
    for line in lines:wrapped.extend(textwrap.wrap(str(line),width=max(45,2*width//9)) or [''])
    top=58;bottom=26*len(wrapped)+16
    canvas=Image.new('RGB',(2*width+10,top+height+bottom),'#171c24');draw=ImageDraw.Draw(canvas)
    for x,panel,title in zip((0,width+10),panels,titles):
        canvas.paste(panel,(x,top));draw.text((x+8,8),title,font=font(17),fill='white')
    for i,line in enumerate(wrapped):draw.text((10,top+height+8+26*i),line,font=font(17),fill='#e6dbb0' if 'GT' in line or 'occlud' in line else '#e2e5ea')
    canvas.save(path)
    return {'path':str(path),'sha256':sha(path),'size_wh':list(canvas.size),'display_scale':width/left.width}


def recorded_scalar(value):
    value=np.asarray(value).item()
    if isinstance(value,float) and not np.isfinite(value):return {'nonfinite_saved_value':str(value)}
    return value


def pose_from_cache(path,key,records,missing,label):
    if not input_record(path,label,records,missing):return None
    try:
        with np.load(path,allow_pickle=False) as arrays:
            if key not in arrays:missing.append({'role':label,'state':'missing_pose_array','array':key});return None
            pose=np.asarray(arrays[key],dtype=np.float64)
            if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('invalid cached rigid pose')
            metadata={name:recorded_scalar(arrays[name]) for name in ('coarse_score','coarse_feature_score','fine_feature_score','icp_score','final_score','icp_rmse','icp_inlier_count','icp_iterations_run','icp_converged') if name in arrays and np.asarray(arrays[name]).ndim==0}
        return {'R':pose[:3,:3].tolist(),'t_mm':pose[:3,3].tolist(),'saved_scalars':metadata}
    except (OSError,ValueError,TypeError) as exc:
        missing.append({'role':label,'state':'unreadable_or_invalid','reason':str(exc)});return None


def visualize_case(config_path,receipt_path,output_folder):
    config_path=local(config_path);config=read(config_path);paths=config['paths'];own=local(paths['output'])
    receipt_path=checked(receipt_path,own/'datasets','Target receipt')
    output=checked(output_folder,own/'auxiliary_diagnostics','Auxiliary output');output.mkdir(parents=True,exist_ok=False)
    receipt=read(receipt_path)
    if receipt.get('status') not in RULE['eligible_statuses']:raise ValueError('Only terminal receipts may be visualized; running candidates are mutable')
    target=receipt['target'];ds=receipt_path.relative_to(own/'datasets').parts[0]
    if ds not in DATASETS:raise ValueError('Unsupported dataset')
    scene,image,obj=(int(target[k]) for k in ('scene_id','im_id','obj_id'))
    records={};missing=[];warnings=[];plots=[]
    input_record(config_path,'configuration',records,missing);input_record(receipt_path,'target_receipt',records,missing)
    if receipt.get('source_pins',{}).get('configuration_sha256')!=digest(config):warnings.append('Receipt configuration hash differs from supplied canonical configuration')
    rgb_path,camera_path=authorized_frame(config,receipt,ds,scene,image,records,missing)
    rgb=Image.open(rgb_path).convert('RGB') if input_record(rgb_path,'RGB',records,missing) else Image.new('RGB',(640,480),'#303030')
    K=None
    if input_record(camera_path,'camera_metadata',records,missing):
        cameras=read(camera_path);entry=cameras.get(str(image),cameras.get(f'{image:06d}'))
        if entry is None:missing.append({'role':'camera_metadata','state':'image_calibration_missing'})
        else:
            K=np.asarray(entry['cam_K'],dtype=np.float64).reshape(3,3)
            if not np.isfinite(K).all():raise ValueError('Nonfinite intrinsics')
    result={};result_path=receipt.get('original_result')
    if result_path:
        result_path=checked(result_path,receipt_path.parent,'Original result')
        if input_record(result_path,'original_result',records,missing):
            try:result=read(result_path)
            except (OSError,ValueError) as exc:missing.append({'role':'original_result','state':'unreadable','reason':str(exc)})
    else:missing.append({'role':'original_result','state':'not_saved','reason':receipt.get('reason')})
    query=checked(local(paths['features_dir'])/f'{ds}_obj_{obj:06d}_query.npz',own,'CAD query cache')
    points=None
    if input_record(query,'CAD_query_cache',records,missing):
        try:
            with np.load(query,allow_pickle=False) as arrays:points=np.asarray(arrays['query_points'],dtype=np.float64)
            if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Nonfinite or misaligned saved CAD points')
        except (OSError,KeyError,ValueError) as exc:missing.append({'role':'CAD_query_cache','state':'unreadable_or_invalid','reason':str(exc)});points=None
    source=checked(local(paths['public_source'])/'freezev2/geometry.py',paths['public_source'],'Original projection source')
    projector=None
    input_record(source,'original_projection_source',records,missing)
    manifest=own/'source_snapshot/source_manifest.json'
    if manifest.is_file():input_record(manifest,'sealed_orchestration_manifest',records,missing)
    run_manifest=own/'run_manifest.json'
    if run_manifest.is_file():
        input_record(run_manifest,'own_run_manifest',records,missing)
        expected=read(run_manifest).get('public_source_identity',{}).get('tracked_files',{}).get('freezev2/geometry.py')
        if expected and records.get('original_projection_source',{}).get('sha256')==expected:
            projector=projection_api(source)
        else:
            warnings.append('Original projection source hash differs from recorded public source identity; projection disabled')
    query_receipt=own/'stages'/ds/f'obj_{obj:06d}'/'fuse'/'receipt.json'
    if query_receipt.is_file():
        input_record(query_receipt,'query_completion_receipt',records,missing)
        qreceipt=read(query_receipt)
        expected={str(local(k)):v for k,v in qreceipt.get('outputs',{}).items()}.get(str(query))
        if qreceipt.get('status')!='complete' or expected!=records.get('CAD_query_cache',{}).get('sha256'):
            warnings.append('CAD query cache differs from its recorded completed feature-stage output')
        if qreceipt.get('source_pins',{}).get('configuration_sha256')!=digest(config):
            warnings.append('CAD query stage configuration differs from supplied configuration')
    else:
        missing.append({'role':'query_completion_receipt','path':str(query_receipt),'state':'not_saved'})

    candidates=result.get('candidates',[]);selected=result.get('selected',[])
    if len(selected)!=receipt.get('selected_count',len(selected)):warnings.append('Saved original selected count differs from target receipt')
    all_masks=[];selected_masks=[];all_selected=[];selected_records=[];candidate_records=[]
    index_to_candidate={c.get('candidate_index'):c for c in candidates}
    for j,chosen in enumerate(selected):
        color=COLORS[j%len(COLORS)];projection=None
        if points is not None and K is not None and projector is not None:
            uv,projection=project(points,K,chosen,projector,rgb.size);all_selected.append((uv,color))
        selected_records.append({'candidate_index':chosen.get('candidate_index'),'saved_final_score':recorded_scalar(chosen['final_score']) if chosen.get('final_score') is not None else None,
            'R':chosen.get('R'),'t_mm':chosen.get('t_mm'),'projection':projection,'color_RGB':list(color)})
    for number,candidate in enumerate(candidates):
        idx=candidate.get('candidate_index',number);mask=None;mask_path=candidate.get('mask_path')
        if mask_path:
            mask_path=checked(mask_path,receipt_path.parent,'Original candidate mask')
            if input_record(mask_path,f'candidate_{idx}_mask',records,missing):
                mask=np.asarray(Image.open(mask_path).convert('L'))>0
                if mask.shape!=(rgb.height,rgb.width):warnings.append(f'Candidate{idx} mask dimensions differ');mask=None
        if mask is not None:
            all_masks.append((mask,(0,220,240)))
            if any(p.get('candidate_index')==idx for p in selected):selected_masks.append((mask,(0,220,240)))
        coarse=None;fine=None
        if candidate.get('coarse_cache_path'):
            path=checked(candidate['coarse_cache_path'],receipt_path.parent,'Original coarse cache')
            coarse=pose_from_cache(path,'coarse_pose',records,missing,f'candidate_{idx}_coarse')
        if candidate.get('fine_cache_path'):
            path=checked(candidate['fine_cache_path'],receipt_path.parent,'Original refined cache')
            fine=pose_from_cache(path,'fine_pose',records,missing,f'candidate_{idx}_refined')
        if fine is None and candidate.get('R') is not None and candidate.get('t_mm') is not None:
            fine={'R':candidate['R'],'t_mm':candidate['t_mm'],'saved_scalars':{},'origin':'original JSON saved refined pose'}
        projections=[];stage_records={}
        for name,pose,color in (('coarse',coarse,(255,175,45)),('refined',fine,(255,70,125))):
            drawn=[]
            if pose is not None and points is not None and K is not None and projector is not None:
                uv,data=project(points,K,pose,projector,rgb.size);drawn=[(uv,color)];stage_records[name]={**pose,'projection':data}
            else:stage_records[name]={'state':'unavailable'}
            projections.append(overlay(rgb,[(mask,(0,220,240))] if mask is not None else [],drawn))
        scores={name:recorded_scalar(candidate[name]) if candidate.get(name) is not None else None for name in ('segmentation_confidence','coarse_score','fine_feature_score','icp_score','final_score')}
        notes=[f'{ds.upper()} scene{scene} image{image} object{obj}; candidate{idx}: {candidate.get("status","unknown")}; NMS={candidate.get("nms_status","unavailable")}',
            'Saved scores: '+', '.join(f'{k}={v:.6g}' if isinstance(v,(int,float)) else f'{k}=unavailable' for k,v in scores.items()),
            'Cyan: original predicted-mask outline. Orange: coarse CAD points. Pink: refined CAD points.',
            'NO GT. Unoccluded all-points projection: hidden/backside points are included. No pose accuracy is established.']
        if candidate.get('error'):notes.append('Original failure: '+str(candidate['error']))
        titles=(f'Candidate{idx}: saved coarse pose' if coarse is not None else f'Candidate{idx}: coarse pose unavailable',
                f'Candidate{idx}: saved refined pose' if fine is not None else f'Candidate{idx}: refined pose unavailable')
        plot=figure(*projections,titles,notes,output/f'candidate_{number:03d}.png');plots.append(plot)
        candidate_records.append({'candidate_index':idx,'status':candidate.get('status'),'nms_status':candidate.get('nms_status'),
            'nms_selected':candidate.get('nms_selected'),'error':candidate.get('error'),'saved_scores':scores,
            'mask_pixels':int(mask.sum()) if mask is not None else None,'stages':stage_records,'plot':plot})
    overview_masks=selected_masks if selected else all_masks
    overview=overlay(rgb,overview_masks,all_selected)
    lines=[f'{ds.upper()} scene{scene} image{image} object{obj}; receipt={receipt.get("status")}; requested instances={target.get("inst_count")}; saved selected poses={len(selected)}.',
        'Cyan: predicted-mask outlines. Colored CAD points: every original saved selected pose.' if selected else 'No selected pose. Available original candidate-mask outlines are shown; no pose was invented.',
        'NO GT. Unoccluded all-points projection includes hidden/backside CAD points. Visual diagnostics are not accuracy measurements.']
    if receipt.get('reason'):lines.append('Recorded failure/block: '+str(receipt['reason']))
    plots.insert(0,figure(rgb,overview,('Actual RGB input','All saved selected poses + predicted masks'),lines,output/'overview.png'))
    expected_outputs={str(local(k)):v for k,v in receipt.get('outputs',{}).items()}
    for label,record in records.items():
        if label=='original_result' or label.startswith('candidate_'):
            expected=expected_outputs.get(str(local(record['path'])))
            record['target_completion_receipt_sha256']=expected
            record['matches_target_completion_receipt']=expected==record['sha256'] if expected else None
            if expected is None:warnings.append('No completed target output hash recorded for '+label)
            elif expected!=record['sha256']:warnings.append('Completed target output checksum differs for '+label)
    current={label:sha(v['path']) for label,v in records.items()}
    if any(current[label]!=record['sha256'] for label,record in records.items()):warnings.append('An input changed while the diagnostic snapshot was generated')
    diagnostic={'schema':'no-GT-own-multiple-pose-auxiliary-diagnostic-v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'dataset':ds,'target':target,'target_status':receipt.get('status'),'selected_count':len(selected),'candidate_count':len(candidates),
        'state':'partial' if missing or warnings else 'complete_visual_diagnostic_only',
        'GT_access':False,'GPU_or_model_inference':False,'primary_outputs_modified':False,'pose_or_score_changes':False,
        'method_id':config.get('method_id'),'public_revision':config.get('public_revision'),
        'configuration_canonical_sha256':digest(config),'source_pins':receipt.get('source_pins'),
        'helper':{'path':str(output/'visualization_helper.py'),'original_source_path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'inputs':records,
        'K_native_pixels':K.tolist() if K is not None else None,'native_RGB_size_wh':list(rgb.size),
        'coordinate_frame':'Native BOP CAD millimetres -> OpenCV camera; camera_points=CAD_points@R.T+t_mm',
        'projection':'Original saved public geometry.project_points; positive finite camera depth and image bounds only',
        'visibility':'Unoccluded all-points projection; no hidden surface/depth visibility culling',
        'selected':selected_records,'candidates':candidate_records,'missing':missing,'warnings':warnings,'figures':plots,
        'visual_review':{'status':'pending image inspection'},'claim_boundary':'No GT comparison, pose accuracy, or benchmark improvement is established'}
    (output/'visualization_helper.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(output/'diagnostic_receipt.json',diagnostic)
    return diagnostic


def snapshot(config_path,output_folder=None):
    config_path=local(config_path);config=read(config_path);own=local(config['paths']['output'])
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=checked(output_folder or own/'auxiliary_diagnostics'/stamp,own/'auxiliary_diagnostics','Snapshot output')
    output.mkdir(parents=True,exist_ok=False)
    rule={'rule':RULE,'rule_sha256':digest(RULE),'helper_sha256':sha(__file__),'predeclared_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'selection_inputs':'target receipt metadata only, before inspecting candidate scores or any GT'}
    atomic_json(output/'selection_rule.json',rule)
    cases=select_cases(own);results=[]
    for case in cases:
        target=case['target'];name=f'{case["dataset"]}_s{target["scene_id"]:06d}_i{target["im_id"]:06d}_o{target["obj_id"]:06d}'
        try:
            diagnostic=visualize_case(config_path,case['receipt_path'],output/name)
            results.append({**case,'diagnostic_state':diagnostic['state'],'diagnostic_receipt':str(output/name/'diagnostic_receipt.json')})
        except Exception as exc:results.append({**case,'diagnostic_state':'failed_auxiliary_visualization','reason':type(exc).__name__+': '+str(exc)})
    manifest={'schema':'receipt-order-no-GT-visual-snapshot-v1','rule':rule,'cases':results,
        'datasets_without_available_terminal_receipts':[d for d in DATASETS if not any(c['dataset']==d for c in cases)],
        'configuration':{'path':str(config_path),'sha256':sha(config_path),'canonical_sha256':digest(config)},
        'GT_access':False,'primary_outputs_modified':False,'case_selection_by_test_scores':False,
        'suite_may_still_be_running':True,'claim_boundary':'Available immutable receipt snapshot only; absent datasets are not counted complete'}
    atomic_json(output/'snapshot_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--receipt');parser.add_argument('--output')
    args=parser.parse_args()
    if args.receipt:
        if not args.output:parser.error('--output is required for an explicit single receipt diagnostic')
        result=visualize_case(args.config,args.receipt,args.output)
        print(json.dumps({'state':result['state'],'selected_count':result['selected_count'],'figures':result['figures']},indent=2))
    else:
        result=snapshot(args.config,args.output);print(json.dumps({'cases':len(result['cases']),'missing_datasets':result['datasets_without_available_terminal_receipts']},indent=2))

if __name__=='__main__':main()