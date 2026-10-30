"""Paired bounded CAD proposals with explicit task-informed construction.

The random reference retains the previous CAD pair/angle sampling. The task
arm balances clipped contact regions and axial thirds, includes projected
pickup-down/final-palm-up approaches, and keeps broad angular exploration.
No first-contact fit, collision validation or physical execution occurs here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import time

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import pad_calibration,rigid_transform
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from experiments.cad_grasp_selection_20261029.sampling_regions import _clip_polygon,_triangulate
from experiments.cad_grasp_refinement_20261029 import sampler as previous
from experiments.cad_grasp_refinement_20261029.contact_model import mesh_binding
from experiments.nist_mating_goals_20261028.cad_tools import mesh_component_labels
from .calibration import make_contact_calibration,hand_geometry_digest

_DOMAIN_CACHE={}


def _unit(value,name):
    value=np.asarray(value,float)
    if value.shape!=(3,) or not np.isfinite(value).all() or abs(np.linalg.norm(value)-1)>1e-8:
        raise ValueError(name+' requires a finite explicit unit vector')
    return value


def _basis(axis):
    reference=np.eye(3)[np.argmin(abs(axis))]
    first=np.cross(axis,reference);first/=np.linalg.norm(first)
    return first,np.cross(axis,first)


def _stratified_domain(vertices,faces,patches,source_ids,axis,point):
    """Exact clipped axial thirds, original component/face IDs retained."""
    digest=hashlib.sha256(np.ascontiguousarray(vertices,dtype='<f8').tobytes()
        +np.ascontiguousarray(faces,dtype='<i8').tobytes()
        +patches.tobytes()+source_ids.tobytes()+axis.tobytes()+point.tobytes()).hexdigest()
    if digest in _DOMAIN_CACHE:return _DOMAIN_CACHE[digest]
    labels=mesh_component_labels(faces)
    if not len(patches):return [],labels,[0.,0.]
    height=(patches.reshape(-1,3)-point)@axis
    lower,upper=float(height.min()),float(height.max())
    edges=np.linspace(lower,upper,4)
    groups=defaultdict(lambda:[[],[]])
    for triangle,source in zip(patches,source_ids):
        triangle_heights=(triangle-point)@axis
        if np.ptp(triangle_heights)<=1e-12 and upper-lower>1e-12:
            # A real face coplanar with a bin boundary has nonzero area;
            # assign it once, rather than duplicating it in two closed bins.
            assigned=int(np.clip((triangle_heights.mean()-lower)/(upper-lower)*3,0,2))
            axial_values=[assigned]
        else:axial_values=range(3 if upper-lower>1e-12 else 1)
        for axial in axial_values:
            polygon=triangle
            if upper-lower>1e-12:
                polygon=_clip_polygon(polygon,axis,-float(point@axis)-edges[axial],True)
                if not len(polygon):continue
                polygon=_clip_polygon(polygon,-axis,float(point@axis)+edges[axial+1],True)
            triangles,ids=_triangulate(polygon,int(source))
            if len(ids):
                group=groups[(int(labels[source]),axial)]
                group[0].extend(triangles);group[1].extend(ids)
    result=[]
    for key,(triangles,ids) in sorted(groups.items()):
        triangles=np.asarray(triangles,float);areas=np.linalg.norm(np.cross(
            triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        result.append((key,triangles,np.asarray(ids,int),areas))
    answer=result,labels,[lower,upper]
    _DOMAIN_CACHE[digest]=answer
    return answer


def _draw_origins(vertices,faces,patches,source_ids,areas,count,rng,strategy,axis,point):
    if strategy=='random':
        points,patch_ids=previous.legacy._surface_samples(patches,areas,count,rng)
        return points,source_ids[patch_ids],dict(method='area_uniform_reference',strata=[])
    groups,_,bounds=_stratified_domain(vertices,faces,patches,source_ids,axis,point)
    allocation=np.full(len(groups),count//len(groups),int)
    for i in rng.permutation(len(groups))[:count%len(groups)]:allocation[i]+=1
    points=[];ids=[];records=[]
    for (key,triangles,source,weights),quota in zip(groups,allocation):
        if quota:
            drawn,local=previous.legacy._surface_samples(triangles,weights,int(quota),rng)
            points.extend(drawn);ids.extend(source[local])
        records.append(dict(component_id=key[0],axial_third=key[1],draws=int(quota),area_m2=float(weights.sum())))
    return np.asarray(points,float).reshape(-1,3),np.asarray(ids,int),dict(
        method='balanced_component_and_equal_length_axial_thirds; area_weighted_within_each',
        axial_bounds_m=bounds,strata=records)


def _projected(vector,closing):
    value=vector-closing*(vector@closing)
    length=np.linalg.norm(value)
    return None if length<=1e-10 else value/length


def _orientation_bank(axes,phases,count,strategy,pickup_down,goal_down):
    result=[]
    for pair,(closing,phase) in enumerate(zip(axes,phases)):
        first,second=_basis(closing)
        explicit=[]
        if strategy=='task_informed':
            for name,target in (('pickup_down',pickup_down),('final_palm_up',goal_down)):
                value=_projected(target,closing)
                if value is None:continue
                found=next((record for record in explicit if record[1]@value>1-1e-12),None)
                if found is None:explicit.append(([name],value))
                else:found[0].append(name)
        broad=count-len(explicit)
        directions=[(names,value,float(np.arctan2(value@second,value@first))) for names,value in explicit]
        for index in range(broad):
            angle=float(phase+2*np.pi*index/broad)
            value=np.cos(angle)*first+np.sin(angle)*second
            if any(old[1]@value>1-1e-12 for old in directions):continue
            directions.append((['broad'],value,angle))
        for index,(sources,value,angle) in enumerate(directions):
            result.append(dict(pair_id=pair,orientation_index=index,orientation_angle_rad=angle,
                approach=value,orientation_sources=sources))
    return result


def table_bounds(vertices_by_component,motions,rotations,midpoints,spans,hand,
                 pickup,plane,contact_calibration,pre_opening_clearance_m=.001,
                 contact_inset_m=.00025):
    """Exact pad endpoint minimum and optimistic finger holding upper bound.

    For the saved symmetric jaws, q=(qL+qR)/2 and tx=(qR-qL)/4.
    Every vertex height is affine in qL,qR. Its maximum on the independent
    opening rectangle occurs at a corner. min_vertices(max_corners(height))
    is an upper bound on any possible complete-hand minimum. If this bound
    fails, no fitted holding configuration can clear the table; pre-opening
    cannot rescue a holding failure. This is a necessary test, not validation.
    """
    base=pad_calibration(hand);tip=fingertip_calibration(hand,contact_inset_m)
    rotations=np.asarray(rotations,float);midpoints=np.asarray(midpoints,float)
    up_cad=pickup[:3,:3].T@plane[:3]
    up_hand=np.einsum('nji,j->ni',rotations,up_cad)
    base_height=midpoints@up_cad+pickup[:3,3]@plane[:3]+plane[3]
    command=(np.asarray(spans)-base['gap_zero_m'])/base['gap_per_opening']
    opened=command+2*pre_opening_clearance_m/base['gap_per_opening']
    pad_anchor=tip['contact_anchors_zero_grip_site_m'].mean(axis=0)
    anchor_slope=tip['contact_anchors_per_opening'].mean(axis=0)
    pad_offset=base_height-np.einsum('ij,ij->i',up_hand,pad_anchor+command[:,None]*anchor_slope)
    finger_anchor=np.asarray(contact_calibration['contact_anchors_zero_grip_site_m']).mean(axis=0)
    finger_offset=base_height-up_hand@finger_anchor
    pad_min=np.full(len(rotations),np.inf);finger_upper=np.full(len(rotations),np.inf)
    low,high=base['opening_limits_m']
    jaw_slopes=np.asarray(base['centers_per_opening'])[:,0]
    corners=[]
    for left in (low,high):
        for right in (low,high):
            q=(left+right)/2
            tx=float(np.mean(jaw_slopes*(np.array([left,right])-q)))
            corners.append((q,tx))
    for points,motion in zip(vertices_by_component,motions):
        minimum=(up_hand@np.asarray(points).T).min(axis=1)
        slope=up_hand@np.asarray(motion)
        pad_min=np.minimum(pad_min,pad_offset+minimum+np.minimum(command*slope,opened*slope))
        best=np.max([q*slope+tx*up_hand[:,0] for q,tx in corners],axis=0)
        finger_upper=np.minimum(finger_upper,finger_offset+minimum+best)
    return pad_min,finger_upper


def _balanced_pick(indices,count,keys,rng,used):
    groups=defaultdict(list)
    for index in indices:
        if index not in used:groups[keys[index]].append(index)
    order=sorted(groups)
    rng.shuffle(order)
    for group in groups.values():rng.shuffle(group)
    chosen=[]
    while order and len(chosen)<count:
        remaining=[]
        for key in order:
            if len(chosen)==count:break
            index=groups[key].pop();chosen.append(index);used.add(index)
            if groups[key]:remaining.append(key)
        order=remaining
    return chosen


def _mode_assignment(candidates,seed):
    rng=np.random.RandomState((int(seed)^0x5A17A5A1)&0xffffffff)
    groups=defaultdict(list)
    for index,candidate in enumerate(candidates):
        groups[(candidate['retention_category'],candidate['sampling_axial_third'])].append(index)
    categories=sorted({key[0] for key in groups});rng.shuffle(categories);position=seed%2
    for category in categories:
        ordered=sorted(key for key in groups if key[0]==category);rng.shuffle(ordered)
        for key in ordered:
            rng.shuffle(groups[key])
            for index in groups[key]:
                candidates[index]['combined_contact_mode']='pad' if position%2==0 else 'finger'
                position+=1


def generate_candidates(vertices,faces,hand,*,world_T_cad_pickup,world_T_cad_final,
        table_plane_world,axis_point_cad_m,axis_direction_cad,sampling_contact_region,
        contact_model=None,contact_calibration=None,seed=0,strategy='random',
        contact_modes=('pad','finger'),max_poses=100,surface_budget=200,orientation_samples=10,
        proposal_normal_angle_rad=np.deg2rad(1.),contact_inset_m=.00025,
        pre_opening_clearance_m=.001,table_clearance_m=.0001,table_prefilter=True,
        final_up_world=None):
    started=time.perf_counter()
    seed=previous.legacy._integer(seed,'seed',0,2**32-1)
    surface_budget=previous.legacy._integer(surface_budget,'surface_budget',1,200)
    orientation_samples=previous.legacy._integer(orientation_samples,'orientation_samples',1,10)
    max_poses=previous.legacy._integer(max_poses,'max_poses',0,100)
    if strategy not in ('random','task_informed'):raise ValueError('Unknown proposal strategy')
    if strategy=='task_informed' and orientation_samples<3:
        raise ValueError('Task-informed construction needs two explicit slots plus broad exploration')
    modes=tuple(contact_modes)
    if modes not in (('pad',),('finger',),('pad','finger'),('finger','pad')):
        raise ValueError('Contact modes must be pad, finger, or their fixed balanced combination')
    if (not np.isfinite(proposal_normal_angle_rad) or not 0<=proposal_normal_angle_rad<np.pi/2
            or not np.isfinite(pre_opening_clearance_m) or pre_opening_clearance_m<0
            or not np.isfinite(table_clearance_m) or table_clearance_m<0):
        raise ValueError('Proposal angle and metric clearances must be finite')
    pickup=rigid_transform(world_T_cad_pickup,'observed pickup transform')
    final=rigid_transform(world_T_cad_final,'known final CAD transform')
    plane=np.asarray(table_plane_world,float)
    if plane.shape!=(4,) or not np.isfinite(plane).all():raise ValueError('Explicit finite table plane required')
    _unit(plane[:3],'table normal')
    goal_up=plane[:3] if final_up_world is None else _unit(final_up_world,'final board up')
    axis=_unit(axis_direction_cad,'CAD mating axis');point=np.asarray(axis_point_cad_m,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Finite CAD mating-axis point required')
    vertices=np.asarray(vertices,float);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min()<0 or faces.max()>=len(vertices)):raise ValueError('Complete valid CAD mesh required')
    base=pad_calibration(hand);fingertip_calibration(hand,contact_inset_m)
    triangles=vertices[faces];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(cross,axis=1)
    normals=np.divide(cross,lengths[:,None],out=np.zeros_like(cross),where=lengths[:,None]>0)
    patches,source,areas,contains,provenance=previous._origin_domain(
        vertices,faces,triangles,lengths/2,sampling_contact_region,'assembly_region_guided')
    if contact_model is not None and any(contact_model.provenance.get(k)!=v for k,v in mesh_binding(vertices,faces).items()):
        raise ValueError('Normal model belongs to a different complete CAD mesh')
    def normal_values(points,ids):
        value=np.asarray(normals[ids] if contact_model is None else contact_model.contact_normals(points,ids),float)
        if value.shape!=(len(points),3) or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value,axis=1),1,atol=1e-8,rtol=0):
            raise ValueError('One finite unit contact normal is required per source point')
        return value
    metadata=dict(strategy=strategy,seed=seed,surface_sampling_budget=surface_budget,
        max_complete_poses=max_poses,orientation_samples_per_pair=orientation_samples,
        proposal_normal_angle_rad=float(proposal_normal_angle_rad),
        hand_surface_angle_changed=False,validation_angle_consumed=False,
        sampling_region_provenance=provenance,contact_modes=list(modes),
        task_direction_scope='projected approach preferences; complete pickup and final clearance still require independent checks',
        final_up_source='explicit caller vector' if final_up_world is not None else 'declared parallel-to-pickup-table assumption',
        ray_intersection_geometry='complete original CAD triangles',pair_reduction=False,
        rejection_resampling=False,independent_depth_sampling=False,downward_gate=False,
        physical_execution=False,surface_origins_drawn=0,proposed_count=0,
        region_accepted_pair_count_before_normals=0,normal_accepted_pair_count=0,
        proposal_pool_size=0,common_table_prefilter_rejected=0,
        table_prefilter_applied=bool(strategy=='task_informed' and table_prefilter),
        retained_orientation_categories={},retained_contact_modes={},combined_mode_counts={},
        shared_raw_bank_sha256=hashlib.sha256(b'[]').hexdigest(),
        orientation_range_rad=[0.,float(2*np.pi)])
    if not len(patches):return [],dict(metadata,status='empty_allowed_contact_surface')
    rng=np.random.RandomState(seed)
    sampled,first_faces,draws=_draw_origins(vertices,faces,patches,source,areas,surface_budget,rng,strategy,axis,point)
    metadata.update(surface_origins_drawn=len(sampled),origin_allocation=draws)
    keep=previous._endpoint_mask(contains,sampled,first_faces,'origin')
    sampled,first_faces=sampled[keep],first_faces[keep]
    if contact_model is not None:
        sampled,snapped=contact_model.snap_origins_to_axial_edges(sampled,first_faces)
        keep=previous._endpoint_mask(contains,sampled,first_faces,'constructed origin')
        sampled,first_faces=sampled[keep],first_faces[keep]
    first_normals=normal_values(sampled,first_faces)
    hits,rays,second_faces,duplicates=previous.legacy._all_ray_hits(sampled,-first_normals,triangles)
    keep=previous._endpoint_mask(contains,hits,second_faces,'opposite endpoint')
    hits,rays,second_faces=hits[keep],rays[keep],second_faces[keep]
    metadata.update(region_accepted_pair_count_before_normals=len(hits),coincident_triangle_hits_removed=int(duplicates))
    pairs=np.stack((sampled[rays],hits),axis=1);spans=np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1)
    closing=np.divide(pairs[:,1]-pairs[:,0],spans[:,None],out=np.zeros_like(hits),where=spans[:,None]>1e-12)
    second_normals=normal_values(hits,second_faces);cosine=np.cos(proposal_normal_angle_rad)
    keep=(spans>previous.legacy.RAY_CONFIG['coincident_hit_tolerance_m'])&(
        np.einsum('ij,ij->i',first_normals[rays],-closing)>=cosine)&(
        np.einsum('ij,ij->i',second_normals,closing)>=cosine)
    pair_faces=np.column_stack((first_faces[rays],second_faces))[keep]
    pair_normals=np.stack((first_normals[rays],second_normals),axis=1)[keep]
    pairs,closing,spans=pairs[keep],closing[keep],spans[keep]
    phases=rng.uniform(0,2*np.pi,len(pairs))
    pool=_orientation_bank(closing,phases,orientation_samples,strategy,
        -pickup[:3,:3].T@plane[:3],-final[:3,:3].T@goal_up)
    metadata.update(normal_accepted_pair_count=len(pairs),proposal_pool_size=len(pool))
    if not pool:return [],dict(metadata,status='no_contact_pairs_within_fixed_budget')
    middle=pairs.mean(axis=1);heights=(middle-point)@axis
    domain_heights=(patches.reshape(-1,3)-point)@axis
    lower,upper=float(domain_heights.min()),float(domain_heights.max())
    bins=np.clip(((heights-lower)/max(upper-lower,1e-12)*3).astype(int),0,2)
    labels=mesh_component_labels(faces);one,two=_basis(axis);keys=[]
    for entry in pool:
        pair=entry['pair_id'];radial=closing[pair]-axis*(closing[pair]@axis)
        direction=4 if np.linalg.norm(radial)<=1e-10 else int((np.arctan2(radial@two,radial@one)%np.pi)/np.pi*4)%4
        components=sorted(labels[pair_faces[pair]].tolist())
        keys.append((int(components[0]),int(components[1]),int(bins[pair]),direction))
    available=np.arange(len(pool),dtype=int)
    filter_started=time.perf_counter()
    if strategy=='task_informed' and table_prefilter:
        if contact_calibration is None:contact_calibration=make_contact_calibration(hand)
        if contact_calibration.get('hand_geometry_sha256')!=hand_geometry_digest(hand):
            raise ValueError('Table prefilter contact calibration must bind to the complete supplied hand')
        definition=dict(contact_calibration);expected_hash=definition.pop('calibration_sha256',None)
        if expected_hash!=hashlib.sha256(json.dumps(definition,sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest():
            raise ValueError('Table prefilter contact calibration hash is invalid')
        rows=np.array([entry['pair_id'] for entry in pool]);approach=np.array([entry['approach'] for entry in pool])
        rotations=np.stack((closing[rows],np.cross(approach,closing[rows]),approach),axis=2)
        pad_bound,finger_bound=table_bounds(
            [np.asarray(c['vertices_zero_m']) for c in hand['components']],
            [np.asarray(c['translation_per_opening']) for c in hand['components']],
            rotations,middle[rows],spans[rows],hand,pickup,plane,contact_calibration,
            pre_opening_clearance_m,contact_inset_m)
        keep=(pad_bound>=table_clearance_m-1e-12)|(finger_bound>=table_clearance_m-1e-12)
        available=available[keep]
        metadata.update(common_table_prefilter_rejected=int((~keep).sum()),
            common_table_prefilter_proof='Reject only if exact pad endpoint minimum AND optimistic min_vertices(max_qLqR_corners height) for fingertip holding fail; unchanged complete hand',
            finger_prefilter_calibration_sha256=contact_calibration.get('calibration_sha256'))
    metadata['common_table_prefilter_seconds']=time.perf_counter()-filter_started
    metadata['available_pool_after_table_prefilter']=len(available)
    if strategy=='random':
        selected=rng.choice(len(pool),min(max_poses,len(pool)),replace=False).tolist()
        reasons=['uniform_random']*len(selected)
    else:
        selected=[];reasons=[];used=set()
        for category,quota in (('pickup_down',max_poses//5),('final_palm_up',max_poses//5),('broad',max_poses-2*(max_poses//5))):
            eligible=[int(i) for i in available if category in pool[i]['orientation_sources']]
            chosen=_balanced_pick(eligible,quota,keys,rng,used)
            selected.extend(chosen);reasons.extend([category]*len(chosen))
        fill=_balanced_pick(available.tolist(),max_poses-len(selected),keys,rng,used)
        selected.extend(fill);reasons.extend(['quota_shortfall_fill']*len(fill))
    candidates=[]
    for index,(pool_id,reason) in enumerate(zip(selected,reasons)):
        entry=pool[pool_id];pair=entry['pair_id']
        candidate=previous.pose_from_pair(pairs[pair],entry['approach'],hand,contact_inset_m)
        candidate.update(candidate_id=index,pair_id=pair,proposal_pool_index=int(pool_id),
            orientation_index=entry['orientation_index'],orientation_angle_rad=entry['orientation_angle_rad'],
            orientation_sources=entry['orientation_sources'],retention_category=reason,
            sampling_axial_third=int(bins[pair]),sampling_region_stratum=list(keys[pool_id]),
            contact_face_ids=pair_faces[pair].tolist(),contact_normals_cad=pair_normals[pair].tolist(),
            pre_opening_m=candidate['holding_opening_m']+2*pre_opening_clearance_m/base['gap_per_opening'])
        candidates.append(candidate)
    _mode_assignment(candidates,seed)
    raw_hash=hashlib.sha256(json.dumps(candidates,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for candidate in candidates:
        candidate['contact_mode']=modes[0] if len(modes)==1 else candidate['combined_contact_mode']
    metadata.update(status='bounded_task_proposals_completed',proposed_count=len(candidates),
        retained_orientation_categories=dict(Counter(reasons)),
        retained_contact_modes=dict(Counter(c['contact_mode'] for c in candidates)),
        combined_mode_counts=dict(Counter(c['combined_contact_mode'] for c in candidates)),
        shared_raw_bank_sha256=raw_hash,
        maximum_angles_per_pair=orientation_samples,broad_pool_fraction_at_least=float((orientation_samples-2)/orientation_samples) if strategy=='task_informed' else 1.,
        total_seconds=time.perf_counter()-started)
    return candidates,metadata
