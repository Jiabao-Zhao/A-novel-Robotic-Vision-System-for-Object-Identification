"""Independent saved-result postprocessor; never generates/fits/validates grasps.

Recomputes budgets, diagnostic stages, CAD criteria, priorities, transforms and
complete-hand table distances. Collision/contact decisions remain saved engine
evidence. Frozen-angle summaries have no full per-angle diagnostics, so only
their budget/hash/count consistency is independently verifiable here.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_selection_20261029.task_inputs import build_request
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from .calibration import make_contact_calibration,hand_geometry_digest
from .contact_surface_audit import audit_saved_run,write_report

ROOT=Path(__file__).resolve().parents[2]
LOCAL=('candidate_input','opening_limits','contact_reconstruction','surface_normals',
       'pad_contact_geometry','target_closing_collision')
STAGES=dict(contact_valid=LOCAL,pickup_valid=LOCAL+('table_clearance',),
    final_goal_valid=LOCAL+('final_receiver_collision','protected_surfaces'),
    joint_valid=LOCAL+('table_clearance','final_receiver_collision','protected_surfaces'),
    contact_with_protected=LOCAL+('protected_surfaces',),final_endpoint_clear=('final_receiver_collision',),
    protected_task_compatible=('protected_surfaces',))
POST_FIELDS=('validity','valid','criteria','world_T_hand_pickup','world_T_hand_final',
             'world_T_grip_site_pickup','world_T_grip_site_final','approach_world')


def _read(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as handle:return json.load(handle)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
    default=lambda item:item.tolist() if isinstance(item,np.ndarray) else item.item()).encode()).hexdigest()


def saved_stages(validity):
    diag=validity['diagnostics'];answer={}
    for name,keys in STAGES.items():
        values=[diag.get(key,{}).get('status','unavailable') for key in keys]
        values=[value if value in ('pass','fail','unavailable','not_evaluated') else 'unavailable' for value in values]
        answer[name]=next((status for status in ('fail','unavailable','not_evaluated') if status in values),'pass')
    return answer


def independent_criteria(candidate,reference,pickup,plane):
    midpoint=np.asarray(candidate['contacts_cad_m'],float).mean(0)
    axis=np.asarray(reference['axis_direction_cad_m'] if 'axis_direction_cad_m' in reference else reference['axis_direction_cad'],float)
    axis=axis/np.linalg.norm(axis)
    tau=float(np.linalg.norm(np.cross(midpoint-np.asarray(reference['axis_point_cad_m']),axis)))
    surface=float(np.linalg.norm(midpoint-np.asarray(reference['surface_centroid_cad_m'])))
    approach=np.asarray(pickup)[:3,:3]@np.asarray(candidate['approach_cad'])
    approach/=np.linalg.norm(approach);up=np.asarray(plane)[:3];up=up/np.linalg.norm(up)
    return dict(q_tau_m=tau,q_surface_m=surface,r_tau=float(np.clip(tau/reference['D_tau_m'],0,1)) if reference['D_tau_m'] else 0.,
        r_surface=float(np.clip(surface/reference['D_surface_m'],0,1)) if reference['D_surface_m'] else 0.,
        theta_down_rad=float(np.arccos(np.clip(approach@(-up),-1,1))))


def independent_priority(records,tolerances):
    remaining=[row for row in records if row['validity']['valid']];stages=[]
    for key in ('r_tau','r_surface','theta_down_rad'):
        if not remaining:break
        value=min(row['criteria'][key] for row in remaining)
        remaining=[row for row in remaining if row['criteria'][key]-value<=tolerances[key]]
        stages.append(dict(criterion=key,minimum=value,numerical_tie_tolerance=tolerances[key],
            remaining_candidate_ids=sorted(row['candidate_id'] for row in remaining)))
    return (min(remaining,key=lambda row:row['candidate_id']) if remaining else None),stages


def independent_calibration_arrays(calibration,hand,require,close):
    """Check stored real-face, projected margin and common-anchor geometry."""
    components=hand['components'];intervals=defaultdict(list)
    fixed=np.deg2rad(1.);minimum_z=float(calibration['minimum_contact_z_grip_site_m'])
    margin=float(calibration['edge_margin_m']);patches=calibration['contact_patches']
    pad_tips=[np.asarray(c['vertices_zero_m'])[:,2].max() for c in components if c['is_pad']]
    close(minimum_z,float(np.mean(pad_tips)),'distal permission begins at original saved pad end',1e-12)
    close(margin,.00025,'fixed projected edge margin',1e-15)
    for patch in patches:
        component=components[patch['hand_component_index']]
        require(component['name']==patch['component'],'contact patch component identity')
        vertices=np.asarray(component['vertices_zero_m']);hull=ConvexHull(vertices)
        face_id=patch['hull_face_index'];equation=hull.equations[face_id]
        require(set(hull.simplices[face_id])==set(patch['source_vertex_indices']),'contact patch original source vertices')
        close(equation[:3],patch['normal_grip_site'],'actual source hull normal',1e-12)
        close(equation[3],patch['plane_offset_m'],'actual source hull plane',1e-12)
        inward=np.array([1.,0,0]) if patch['side']==0 else np.array([-1.,0,0])
        require(equation[:3]@inward>=np.cos(fixed)-1e-12,'fixed1degree hand surface permission')
        original=np.asarray(patch['polygon_grip_site_m']);eroded=np.asarray(patch['eroded_polygon_grip_site_m'])
        for polygon in (original,eroded):
            require(np.all(polygon@hull.equations[:,:3].T+hull.equations[:,3]<=1e-10),'contact patch inside complete source finger')
            close(polygon@equation[:3]+equation[3],np.zeros(len(polygon)),'contact patch on original face',1e-10)
            require(np.min(polygon[:,2])>=minimum_z-1e-10,'contact patch beyond saved pad end')
        yz=original[:,1:];test=eroded[:,1:]
        area=np.sum(yz[:,0]*np.roll(yz[:,1],-1)-yz[:,1]*np.roll(yz[:,0],-1))/2
        if area<0:yz=yz[::-1]
        for a,b in zip(yz,np.roll(yz,-1,axis=0)):
            edge=b-a;length=np.linalg.norm(edge)
            if length<1e-14:continue
            inward2=np.array([-edge[1],edge[0]])/length
            require(np.min((test-a)@inward2)>=margin-1e-10,'declared projected contact-patch erosion margin')
        crossings=[]
        for a,b in zip(test,np.roll(test,-1,axis=0)):
            if abs(a[0])<1e-12:crossings.append(float(a[1]))
            if a[0]*b[0]<0:crossings.append(float(a[1]-a[0]/(b[0]-a[0])*(b[1]-a[1])))
        if crossings:intervals[patch['side']].append((min(crossings),max(crossings)))
    heights=[min(left[1],right[1]) for left in intervals[0] for right in intervals[1]
        if max(left[0],right[0])<=min(left[1],right[1])+1e-12]
    require(bool(heights),'opposed source regions have a commonY0 interval')
    anchors=np.asarray(calibration['contact_anchors_zero_grip_site_m'])
    close(anchors[:,1],np.zeros(2),'common contact anchor Y',1e-12)
    if heights:close(anchors[:,2],np.full(2,max(heights)),'highest permitted common source contact Z',1e-10)
    for index,patch_id in enumerate(calibration['selected_contact_patch_indices']):
        patch=patches[patch_id];normal=np.asarray(patch['normal_grip_site'])
        close(anchors[index]@normal+patch['plane_offset_m'],0.,'actual anchor source plane',1e-12)
        component=components[patch['hand_component_index']]
        close(calibration['contact_anchors_per_opening'][index],component['translation_per_opening'],'contact anchor actual opening motion',1e-12)


def verify(run_directory,*,hand_path=None,observed_path=None,registry_path=None,
           visual_mesh_path=None,collision_mesh_path=None,xml_path=None,output_directory=None):
    run=Path(run_directory).resolve();output=Path(output_directory).resolve() if output_directory else run/'independent_audit'
    if not (run/'results.json').is_file() or not (run/'run_verification.json').is_file():
        raise ValueError('Full run is not complete; no partial matrix is certified')
    output.mkdir(parents=True,exist_ok=True)
    hand_path=Path(hand_path or ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json')
    observed_path=Path(observed_path or ROOT/'outputs/cad_grasp_geometry_20261026/inputs.json')
    registry_path=Path(registry_path or ROOT/'outputs/nist_scene_corrections_20261029/assembly_goals.json')
    hand,observed,registry=[_read(path) for path in (hand_path,observed_path,registry_path)]
    manifest=_read(run/'launch_manifest.json');results=_read(run/'results.json');summary=_read(run/'summary.json')
    errors=[];counts=Counter();hashes={};max_error=defaultdict(float)
    def require(condition,message):
        counts['assertions']+=1
        if not condition:errors.append(message)
    def close(a,b,label,tol=1e-9):
        a,b=np.asarray(a,float),np.asarray(b,float)
        good=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        difference=float(np.max(np.abs(a-b))) if good and a.size else 0.
        max_error[label]=max(max_error[label],difference)
        require(good and difference<=tol,label+' mismatch')
    for group in ('source_sha256','protected_sha256'):
        for name,expected in manifest[group].items():
            path=ROOT/Path(name.replace('\\','/'))
            require(path.is_file() and _sha(path)==expected,group+': '+name)
    for path in (hand_path,observed_path,registry_path):hashes[str(path)]=_sha(path)
    for flag in ('VLM','pickup','assembly','physics','simulator_object_poses_read','insertion_path_and_release'):
        require(manifest.get(flag) is False,'declared paused/execution flag: '+flag)
    calibration=make_contact_calibration(hand,normal_angle_rad=np.deg2rad(manifest['fixed_hand_surface_angle_deg']))
    require(calibration==_read(run/'contact_calibration.json'),'portable fixed contact calibration differs')
    independent_calibration_arrays(calibration,hand,require,close)
    require(manifest['fixed_hand_surface_angle_deg']==manifest['proposal_normal_angle_deg']==manifest['validation_normal_angle_deg']==1.,'primary angles are not separately fixed1degree')
    tip=fingertip_calibration(hand);observed_rows={(r['scene'],r['part']):r for r in observed['records']}
    expected=set(itertools.product(manifest['strategies'],range(len(manifest['seeds'])),manifest['modes'],
        [f'N{i:02d}' for i in range(8)],manifest['parts']))
    actual=[(r['sampler_strategy'],r['seed_index'],r['contact_mode'],r['scene'],r['part']) for r in results]
    require(len(actual)==len(set(actual)) and set(actual)==expected,'policy/seed/scene/part matrix incomplete or duplicated')
    grouped=defaultdict(list)
    for row in results:grouped[(row['sampler_strategy'],row['seed_index'],row['scene'],row['part'])].append(row)
    static={};frozen_hashes={};frozen_valid={};group_totals=defaultdict(Counter);primary_adapter=output/'task_informed_seed0_combined_finger_view'
    parity=defaultdict(Counter);part_budgets=defaultdict(Counter)
    upstream=[];primary_finger_valid=primary_finger_selected=0
    for index,(key,decisions) in enumerate(sorted(grouped.items())):
        strategy,seed_index,scene,part=key;by_mode={r['contact_mode']:r for r in decisions}
        observed_row=observed_rows[(scene,part)]
        if part not in static:
            built=build_request(part,observed_row['world_T_cad'],observed_row['table_plane_world'],registry=registry,hand=hand)
            require(built['status']=='available','static corrected inputs unavailable: '+part)
            request=built['request'];v=np.asarray(request['vertices_cad_m']);f=np.asarray(request['faces']);tri=v[f]
            area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            centroid=np.sum(area[:,None]*tri.mean(1),axis=0)/area.sum();point=np.asarray(request['axis_point_cad_m']);axis=np.asarray(request['axis_direction_cad']);axis=axis/np.linalg.norm(axis)
            reference=dict(surface_centroid_cad_m=centroid,axis_point_cad_m=point,axis_direction_cad=axis,
                D_tau_m=float(np.linalg.norm(np.cross(v-point,axis),axis=1).max()),D_surface_m=float(np.linalg.norm(v-centroid,axis=1).max()),surface_area_m2=float(area.sum()))
            receipt=_read(run/'task_inputs'/f'{part}.json')
            binding=_digest(dict(vertices=v,faces=f,hand=hand_geometry_digest(hand),final=np.asarray(request['world_T_cad_final']),
                receivers=request['receivers'],protected=request['protected_contact_region'],calibration=calibration['calibration_sha256'],model=receipt['contact_model']))
            require(binding==receipt['binding_sha256'],'static complete-CAD/task binding: '+part)
            require(receipt['calibration_sha256']==calibration['calibration_sha256'],'task calibration: '+part)
            static[part]=(request,reference,binding)
        request,reference,binding=static[part]
        path=run/by_mode['pad']['diagnostic_bank'];bank=_read(path);sampling_path=path.with_name('sampling.json.gz');sampling=_read(sampling_path)
        hashes[str(path)]=_sha(path);hashes[str(sampling_path)]=_sha(sampling_path)
        pads,fingers=bank['pad'],bank['finger'];n=len(pads)
        require(len(fingers)==n<=100 and n==sampling['proposed_count'],'shared candidate budget: '+str(key))
        part_budgets[part]['raw_main']+=n
        if strategy=='random' and seed_index==0 and manifest['sensitivity_angles_deg']:
            part_budgets[part]['sensitivity_frozen']+=n
        require(sampling['surface_sampling_budget']<=200 and sampling['maximum_angles_per_pair']<=10,'bounded proposal draws/angles: '+str(key))
        require(sampling.get('surface_origins_drawn',0)<=sampling['surface_sampling_budget'] and not sampling['rejection_resampling'],
            'actual surface draws/resampling: '+str(key))
        require(sampling['proposal_pool_size']<=sampling['normal_accepted_pair_count']*sampling['maximum_angles_per_pair'],
            'orientation pool cap: '+str(key))
        require(sampling['common_table_prefilter_rejected']+sampling['available_pool_after_table_prefilter']==sampling['proposal_pool_size'],
            'reported table-prefilter pool accounting: '+str(key))
        if sampling.get('table_prefilter_applied'):
            require(sampling['finger_prefilter_calibration_sha256']==calibration['calibration_sha256'],
                'fixed calibration in necessary table prefilter: '+str(key))
        require(sampling['proposal_normal_angle_rad']==np.deg2rad(1.) and not sampling['validation_angle_consumed'] and not sampling['hand_surface_angle_changed'],'proposal angle separation: '+str(key))
        pad_by_id={r['candidate_id']:r for r in pads};finger_by_id={r['candidate_id']:r for r in fingers}
        require(len(pad_by_id)==len(finger_by_id)==n and set(pad_by_id)==set(finger_by_id),'candidate IDs disagree: '+str(key))
        assignment=bank['combined_assignment'];assignment_counts=Counter(a['contact_mode'] for a in assignment)
        require(len(assignment)==n and set(a['candidate_id'] for a in assignment)==set(pad_by_id),'combined assignment completeness: '+str(key))
        require(all(a['contact_mode']==pad_by_id[a['candidate_id']]['combined_contact_mode'] for a in assignment),
            'combined differs from hashed preregistered raw assignment: '+str(key))
        require(set(assignment_counts)<=set(('pad','finger')) and abs(assignment_counts['pad']-assignment_counts['finger'])<=1,'preregistered mode balance: '+str(key))
        require(dict(assignment_counts)==sampling['combined_mode_counts'],'sampling assignment count: '+str(key))
        if n==100:require(assignment_counts==Counter(pad=50,finger=50),'combined not50+50: '+str(key))
        require(bank['benchmark_pool_evaluations']==2*n and bank['combined_is_preregistered_subset_not_union'],'benchmark vs policy budget: '+str(key))
        raw=[]
        for record in pads:
            r={k:v for k,v in record.items() if k not in POST_FIELDS and k not in ('contact_mode','actual_contacts_hand_m','actual_contacts_hand_frame')};raw.append(r)
        require(_digest(raw)==bank['raw_bank_sha256']==sampling['shared_raw_bank_sha256'],'shared raw bank hash: '+str(key))
        for candidate_id,pad in pad_by_id.items():
            finger=finger_by_id[candidate_id]
            close(finger['proposal_contacts_cad_m'],pad['contacts_cad_m'],'shared proposal pair',1e-12)
            close(finger['approach_cad'],pad['approach_cad'],'shared proposal orientation',1e-12)
            require(finger['pair_id']==pad['pair_id'] and finger['orientation_index']==pad['orientation_index'],
                'shared pair/angle IDs: '+str(key))
        combined=[(pad_by_id if a['contact_mode']=='pad' else finger_by_id)[a['candidate_id']] for a in assignment]
        pools=dict(pad=pads,finger=fingers,combined=combined)
        if strategy=='random' and seed_index==0:
            historical=dict(pad=ROOT/'outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected',
                finger=ROOT/'outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z')
            raw_fields=('candidate_id','pair_id','orientation_index','orientation_angle_rad','contacts_cad_m',
                'approach_cad','contact_face_ids','pre_opening_m','holding_opening_m','cad_T_hand','cad_T_grip_site')
            fit_fields=('contacts_cad_m','actual_contacts_hand_m','cad_T_hand','cad_T_grip_site','holding_opening_m','pre_opening_m')
            for mode,base in historical.items():
                old_path=base/'cases'/scene/part/'candidate_diagnostics.json';old_records=_read(old_path)
                hashes[str(old_path)]=_sha(old_path);current=pools[mode]
                require(len(old_records)==len(current),'historical bank size: '+str((key,mode)))
                for old,new in zip(old_records,current):
                    require(old['candidate_id']==new['candidate_id'],'historical candidate order')
                    for field in (raw_fields if mode=='pad' else fit_fields):
                        require((field in old)==(field in new),'historical field presence: '+field)
                        if field in old:close(old[field],new[field],'historical '+mode+' '+field,1e-12)
                    if mode=='finger':require(old['first_contact_fit']['status']==new['first_contact_fit']['status'],'historical first-contact-fit outcome')
                    old_status={k:v['status'] for k,v in old['validity']['diagnostics'].items()}
                    new_status={k:v['status'] for k,v in new['validity']['diagnostics'].items()}
                    require(old_status==new_status,'historical diagnostic parity: '+str((key,mode,new['candidate_id'])))
                old_valid=sum(r['validity']['valid'] for r in old_records);new_valid=sum(r['validity']['valid'] for r in current)
                require(old_valid==new_valid,'historical valid count parity: '+str((key,mode)))
                parity[mode]['cases']+=1;parity[mode]['old_valid_poses']+=old_valid;parity[mode]['new_valid_poses']+=new_valid
                parity[mode]['old_selected']+=int(old_valid>0);parity[mode]['new_selected']+=int(new_valid>0)
                del old_records
        for mode,records in pools.items():
            row=by_mode[mode];decision=row['selection'];label=str((key,mode))
            selection_path=run/'cases'/strategy/f'seed_{seed_index}'/mode/scene/part/'selection.json'
            require(_read(selection_path)==row,'case file differs from aggregate result: '+label)
            hashes[str(selection_path)]=_sha(selection_path)
            close(row['world_T_cad'],observed_row['world_T_cad'],'observed pickup transform',1e-12)
            close(row['table_plane_world'],observed_row['table_plane_world'],'observed table plane',1e-12)
            require(row['seed']==manifest['seeds'][seed_index]+sorted(manifest['parts']).index(part),'paired part seed: '+label)
            require(decision['proposed_count']==len(records)<=100,'policy budget: '+label)
            category=defaultdict(Counter);stage=defaultdict(Counter);valid=[];independent_records=[]
            for c in records:
                validity=c['validity'];statuses={k:d['status'] for k,d in validity['diagnostics'].items()}
                require(bool(validity['valid'])==all(s=='pass' for s in statuses.values()),'valid flag/category disagreement: '+label)
                for name,status in statuses.items():category[name][status]+=1
                for name,status in saved_stages(validity).items():
                    stage[name][status]+=1
                    require(validity.get('stages',{}).get(name,{}).get('status')==status,'saved stage mismatch: '+label)
                grip=np.asarray(c['cad_T_grip_site']);tip_pose=np.asarray(c['cad_T_hand'])
                close(tip_pose,grip@tip['grip_site_T_fingertip'],'fingertip reference transform')
                if c.get('actual_contacts_hand_m') is not None:
                    p=np.asarray(c['actual_contacts_hand_m']);close(p@grip[:3,:3].T+grip[:3,3],c['contacts_cad_m'],'actual contact frame reconstruction')
                if c['contact_mode']=='finger':require(c['contact_calibration_sha256']==calibration['calibration_sha256'],'fixed finger calibration hash: '+label)
                if validity['checks'].get('component_geometry') is not None:
                    minimum=np.inf;per={};pickup=np.asarray(row['world_T_cad']);plane=np.asarray(row['table_plane_world'])
                    for component in hand['components']:
                        points=np.concatenate([np.asarray(component['vertices_zero_m'])+q*np.asarray(component['translation_per_opening']) for q in (c['holding_opening_m'],c['pre_opening_m'])])
                        cad=points@grip[:3,:3].T+grip[:3,3];world=cad@pickup[:3,:3].T+pickup[:3,3]
                        per[component['name']]=float(np.min(world@plane[:3]+plane[3]));minimum=min(minimum,per[component['name']])
                    close(minimum,validity['checks']['table_minimum_clearance_m'],'whole-hand table minimum')
                    for component in validity['checks']['component_geometry']:close(per[component['component']],component['table_minimum_clearance_m'],'component table minimum')
                if validity['valid']:
                    criteria=independent_criteria(c,reference,row['world_T_cad'],row['table_plane_world'])
                    for name,value in criteria.items():close(value,c['criteria'][name],'independent '+name)
                    valid.append(c);independent_records.append(dict(c,criteria=criteria))
            for name,values in reference.items():close(values,decision['reference'][name],'fixed CAD reference '+name)
            require(dict(stage)==decision['stage_counts'],'stage totals: '+label)
            require(dict(category)==row['independent_check_counts'],'category totals: '+label)
            require(len(valid)==decision['valid_count'],'valid count: '+label)
            selected,priority=independent_priority(independent_records,decision['tie_tolerances'])
            require(len(priority)==len(decision['selection_stages']),'priority stage count: '+label)
            for computed,saved_stage in zip(priority,decision['selection_stages']):
                require(computed['criterion']==saved_stage['criterion'] and computed['remaining_candidate_ids']==saved_stage['remaining_candidate_ids'],
                    'priority survivors: '+label)
                close(computed['minimum'],saved_stage['minimum'],'priority minimum',1e-10)
                close(computed['numerical_tie_tolerance'],saved_stage['numerical_tie_tolerance'],'priority tie tolerance',1e-15)
            saved=decision['selected_grasp']
            require((selected is None)==(saved is None),'selected presence: '+label)
            if selected is not None:require(selected['candidate_id']==saved['candidate_id'],'deterministic selected ID: '+label)
            totals=group_totals[(strategy,mode)];totals['placements']+=1;totals['selected']+=int(selected is not None);totals['valid']+=len(valid);totals['proposed']+=len(records)
            for name,value in stage.items():totals['stage_candidate:'+name]+=value['pass'];totals['stage_placement:'+name]+=int(value['pass']>0)
        if strategy=='random' and seed_index==0:
            frozen=[{k:v for k,v in r.items() if k not in POST_FIELDS} for r in fingers]
            frozen_hashes[(scene,part)]=_digest(frozen);frozen_valid[(scene,part)]=by_mode['finger']['selection']['valid_count']
        if strategy=='task_informed' and seed_index==0:
            selected=by_mode['combined']['selection']['selected_grasp']
            finger_valid=[r for r in combined if r['contact_mode']=='finger' and r['validity']['valid']]
            selected_finger=selected if selected is not None and selected['contact_mode']=='finger' else None
            case=primary_adapter/'cases'/scene/part;case.mkdir(parents=True,exist_ok=True)
            (case/'candidate_diagnostics.json').write_text(json.dumps(finger_valid,separators=(',',':')))
            (case/'selection.json').write_text(json.dumps(dict(selection=dict(selected_grasp=selected_finger)),separators=(',',':')))
            upstream.append(dict(scene=scene,part=part,diagnostic_bank=str(path),diagnostic_bank_sha256=_sha(path),
                sampling_sha256=_sha(sampling_path),combined_assignment=assignment,valid_finger_candidate_ids=[r['candidate_id'] for r in finger_valid],
                selected_finger_candidate_id=selected_finger['candidate_id'] if selected_finger else None))
            primary_finger_valid+=len(finger_valid);primary_finger_selected+=int(selected_finger is not None)
        if (index+1)%48==0:print(json.dumps(dict(saved_banks_audited=index+1,total=len(grouped))),flush=True)
    for saved in summary['groups']:
        totals=group_totals[(saved['strategy'],saved['contact_mode'])]
        for name in ('placements','selected','valid','proposed'):require(saved[name]==totals[name],'group summary '+name)
        for name,value in saved['stage_candidates'].items():require(value==totals['stage_candidate:'+name],'group stage candidates')
        for name,value in saved['stage_placements'].items():require(value==totals['stage_placement:'+name],'group stage placements')
        for seed_row in saved['per_seed']:
            matching=[r for r in results if r['sampler_strategy']==saved['strategy'] and r['contact_mode']==saved['contact_mode'] and r['seed_index']==seed_row['seed_index']]
            require(seed_row['placements']==len(matching) and seed_row['selected']==sum(r['selection']['status']=='selected' for r in matching),'per-seed summary counts')
    require(summary['policy_placement_evaluations']==len(results),'aggregate policy-placement count')
    require(summary['logical_policy_candidate_evaluations']==sum(v['proposed'] for v in group_totals.values()),'aggregate logical policy budget')
    require(summary['underlying_mode_candidate_evaluations']==sum(v['proposed'] for (strategy,mode),v in group_totals.items() if mode!='combined'),'aggregate underlying comparison budget')
    primary=[row for row in results if row['sampler_strategy']=='task_informed' and row['seed_index']==0 and row['contact_mode']=='combined']
    require(_read(run/'primary/results.json')==primary,'preregistered primary subset differs')
    primary_summary=_read(run/'primary/summary.json')
    require(primary_summary['placements']==len(primary) and primary_summary['selected_placements']==sum(row['selection']['status']=='selected' for row in primary),'primary placement counts')
    require(primary_summary['valid_poses']==sum(row['selection']['valid_count'] for row in primary),'primary valid-pose count')
    require(parity['pad']['new_selected']==35 and parity['pad']['new_valid_poses']==495,'randomseed0 does not reproduce saved35/495 pad reference')
    require(parity['finger']['new_selected']==37 and parity['finger']['new_valid_poses']==75,'randomseed0 does not reproduce saved37/75 finger reference')
    for part,budget in part_budgets.items():
        receipt_path=run/'part_receipts'/f'{part}.json';receipt=_read(receipt_path);counters=receipt['counters']
        require(receipt['static_task_binding_sha256']==static[part][2],'part receipt static binding: '+part)
        require(counters.get('unique_finger_first_contact_fits',0)+counters.get('reused_finger_first_contact_fits',0)==
            budget['raw_main']+budget['sensitivity_frozen'],'actual fit-call accounting: '+part)
        extra_angles=sum(float(angle)!=1. for angle in manifest['sensitivity_angles_deg'])
        require(counters.get('unique_local_and_final_validations',0)+counters.get('reused_local_and_final_validations',0)==
            2*budget['raw_main']+extra_angles*budget['sensitivity_frozen'],'actual geometry-evaluation accounting: '+part)
        hashes[str(receipt_path)]=_sha(receipt_path)
    angles=_read(run/'frozen_angle_sensitivity.json');angle_groups=defaultdict(list)
    require(len(angles)==len(manifest['parts'])*8*len(manifest['sensitivity_angles_deg']),'sensitivity matrix completeness')
    for item in angles:
        key=item['scene'],item['part'];angle_groups[key].append(item)
        require(item['frozen_pose_sha256']==frozen_hashes[key],'frozen pose SHA: '+str(key))
        require(item['calibration_sha256']==calibration['calibration_sha256'],'sensitivity fixed calibration')
        require(item['extra_surface_draws']==item['extra_contact_fits']==0,'sensitivity extra sampling/fits')
        require(item['proposed']<=100 and len(set(item['rescued_candidate_ids']))==len(item['rescued_candidate_ids']),'sensitivity budget/IDs')
        if item['angle_deg']==1.:require(item['valid']==frozen_valid[key],'sensitivity1degree baseline')
    for key,items in angle_groups.items():
        require(sorted(i['angle_deg'] for i in items)==manifest['sensitivity_angles_deg'],'sensitivity angles missing: '+str(key))
        require(all(a['valid']<=b['valid'] for a,b in zip(sorted(items,key=lambda r:r['angle_deg']),sorted(items,key=lambda r:r['angle_deg'])[1:])),'sensitivity nonmonotone validity: '+str(key))
    visual_result=None
    if visual_mesh_path is not None:
        visual_result=audit_saved_run(primary_adapter,hand,visual_mesh_path,collision_mesh_path=collision_mesh_path,xml_path=xml_path)
        require(visual_result['all_saved_valid_pose_summary']['poses']==primary_finger_valid,'visual audit valid-finger subset')
        require(visual_result['selected_pose_summary']['poses']==primary_finger_selected,'visual audit selected-finger subset')
        write_report(visual_result,output/'primary_detailed_source_compatibility')
    provenance=dict(upstream_run=str(run),policy='task_informed / preregistered seed0 / combined50+50',
        source_is_derived_view_only=True,only_actual_finger_mode_contacts=True,
        no_new_candidates_fits_validation_or_selection=True,upstream_cases=upstream)
    (primary_adapter/'upstream_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    report=dict(status='passed' if not errors else 'failed',errors=errors,counts=dict(counts),
        complete_policy_matrix=len(results),underlying_banks=len(grouped),source_and_protected_hashes_checked=True,
        independent_methods=['explicit diagnostic stage conjunctions','source/input SHA256','fixed calibration digest plus original hull-plane/margin/common-anchor arithmetic',
            'complete-hand affine endpoint table distances','surface-area CAD centroid and fixed bounds','lexicographic minimum-with-ties and integerID',
            'combined subset/50+50 and shared raw-bank hash','frozen fit hashes and sensitivity metadata'],
        limits=['Saved target/contact/receiver/protected engine decisions were not rerun or independently certified.',
            'Full per-angle candidate diagnostics are not saved; angle acceptance/rescue counts are runner-reported. Only baseline agreement, frozen hashes/budgets, matrix and monotonicity were independently checked.',
            'Detailed visual mesh is another source asset, not physical material ground truth.'],
        no_new_grasps_fits_collision_queries_or_execution=True,maximum_numerical_differences=dict(max_error),
        fixed_calibration_sha256=calibration['calibration_sha256'],input_sha256=hashes,
        postprocessor_source_sha256=_sha(__file__),postprocessor_added_after_launch=True,
        random_seed0_historical_parity={mode:dict(value) for mode,value in parity.items()},
        summary_by_strategy_mode={str(k):dict(v) for k,v in group_totals.items()},
        primary_modeled_finger_valid_poses=primary_finger_valid,primary_modeled_finger_selected_placements=primary_finger_selected,
        detailed_source_compatibility=visual_result['selected_pose_summary'] if visual_result else dict(status='visual_asset_not_supplied'))
    (output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],errors=len(errors),output=str(output))))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    for name in ('hand','observed','registry','visual-mesh','collision-mesh','xml','output'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();report=verify(args.run,hand_path=args.hand,observed_path=args.observed,registry_path=args.registry,
        visual_mesh_path=args.visual_mesh,collision_mesh_path=args.collision_mesh,xml_path=args.xml,output_directory=args.output)
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
