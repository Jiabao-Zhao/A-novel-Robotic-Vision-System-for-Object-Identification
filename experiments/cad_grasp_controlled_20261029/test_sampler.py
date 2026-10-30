"""Synthetic fixed-budget/task-construction tests; no NIST trial execution."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.cad_grasp_selection_20261028.test_geometry import box,fixture
from experiments.cad_grasp_selection_20261028.geometry import component_vertices,transform_points,pad_calibration
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region
from experiments.cad_grasp_refinement_20261029 import sampler as random_reference
from . import sampler
from .calibration import make_contact_calibration


def setup():
    vertices,faces=box([-.006,-.008,-.025],[.006,.008,.025])
    hand=deepcopy(fixture()[0]['hand'])
    region=compile_sampling_region(vertices,faces,dict(type='face_ids',face_ids=[]))
    final=np.eye(4);final[:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()
    options=dict(world_T_cad_pickup=np.eye(4),world_T_cad_final=final,
        table_plane_world=[0,0,1,.1],axis_point_cad_m=[0,0,0],axis_direction_cad=[0,0,1],
        sampling_contact_region=region,seed=14,table_prefilter=False)
    return vertices,faces,hand,options


def moving_hand():
    hand=deepcopy(fixture()[0]['hand'])
    for side in (-1,1):
        low=[-.004,-.008,-.008] if side<0 else [0,-.008,-.008]
        high=[0,.008,.008] if side<0 else [.004,.008,.008]
        points,_=box(low,high)
        hand['components'].append(dict(name='left_finger' if side<0 else 'right_finger',is_pad=False,
            vertices_zero_m=points.tolist(),translation_per_opening=[side*.5,0,0]))
    return hand


def test_random_arm_reproduces_frozen_raw_pair_angle_bank():
    vertices,faces,hand,options=setup()
    old,_=random_reference.generate_candidates(vertices,faces,hand,
        sampling_contact_region=options['sampling_contact_region'],seed=options['seed'])
    new,meta=sampler.generate_candidates(vertices,faces,hand,**options,strategy='random')
    assert len(old)==len(new)==100
    for a,b in zip(old,new):
        for key,value in a.items():assert b[key]==value,(key,a,b)
    assert meta['common_table_prefilter_rejected']==0


def test_shared_100_pose_bank_and_fixed_50_50_allocation_across_contact_arms():
    vertices,faces,hand,options=setup()
    outputs=[];hashes=[]
    for modes in [('pad',),('finger',),('pad','finger')]:
        poses,meta=sampler.generate_candidates(vertices,faces,hand,**options,
            strategy='task_informed',contact_modes=modes)
        assert len(poses)==100 and meta['surface_origins_drawn']==200
        assert meta['maximum_angles_per_pair']==10
        assert meta['retained_orientation_categories']==dict(pickup_down=20,final_palm_up=20,broad=60)
        assert Counter(p['combined_contact_mode'] for p in poses)=={'pad':50,'finger':50}
        for category in ['pickup_down','final_palm_up','broad']:
            counts=Counter(p['combined_contact_mode'] for p in poses if p['retention_category']==category)
            assert counts['pad']==counts['finger']
        hashes.append(meta['shared_raw_bank_sha256'])
        outputs.append([{k:v for k,v in p.items() if k!='contact_mode'} for p in poses])
    assert hashes[0]==hashes[1]==hashes[2]
    assert outputs[0]==outputs[1]==outputs[2]


def test_explicit_task_directions_are_projected_to_valid_closing_perpendiculars():
    vertices,faces,hand,options=setup()
    poses,_=sampler.generate_candidates(vertices,faces,hand,**options,strategy='task_informed')
    targets=dict(pickup_down=np.array([0.,0,-1]),
                 final_palm_up=-options['world_T_cad_final'][:3,:3].T@np.array([0.,0,1]))
    for p in poses:
        closing=np.diff(np.asarray(p['contacts_cad_m']),axis=0)[0];closing/=np.linalg.norm(closing)
        approach=np.asarray(p['approach_cad'])
        assert abs(approach@closing)<1e-10
        for source in p['orientation_sources']:
            if source in targets:
                expected=targets[source]-closing*(targets[source]@closing)
                expected/=np.linalg.norm(expected)
                np.testing.assert_allclose(approach,expected,atol=1e-12,rtol=0)


def test_stratified_origin_clips_keep_total_area_and_exact_200_draw_budget():
    vertices,faces,hand,options=setup();region=options['sampling_contact_region']
    groups,labels,bounds=sampler._stratified_domain(vertices,faces,
        region['sampling_triangles_cad_m'],region['source_face_ids'],np.array([0.,0,1]),np.zeros(3))
    assert len(groups)==3
    assert sum(g[3].sum() for g in groups)==pytest.approx(region['eligible_area_m2'],abs=1e-14)
    before_v,before_f=vertices.tobytes(),faces.tobytes()
    _,meta=sampler.generate_candidates(vertices,faces,hand,**options,strategy='task_informed')
    quotas=[r['draws'] for r in meta['origin_allocation']['strata']]
    assert sum(quotas)==200 and max(quotas)-min(quotas)<=1
    assert (vertices.tobytes(),faces.tobytes())==(before_v,before_f)


@pytest.mark.parametrize('jaw_slope',[.5,.35])
def test_table_bound_matches_pad_exact_geometry_and_bounds_all_dense_finger_fits(jaw_slope):
    hand=moving_hand()
    for component in hand['components']:
        motion=component['translation_per_opening']
        if motion[0]:component['translation_per_opening']=[np.sign(motion[0])*jaw_slope,0,0]
    cal=make_contact_calibration(hand);rng=np.random.RandomState(6)
    rotations=Rotation.random(7,random_state=rng).as_matrix();middle=rng.uniform(-.015,.015,(7,3));spans=rng.uniform(.005,.025,7)
    pickup=np.eye(4);pickup[:3,:3]=Rotation.from_euler('xy',[10,6],degrees=True).as_matrix();pickup[:3,3]=[.01,.02,.03]
    plane=np.array([0.,0,1.,.01]);base=pad_calibration(hand);tip=fingertip_calibration(hand)
    pad,upper=sampler.table_bounds([np.asarray(c['vertices_zero_m']) for c in hand['components']],
        [np.asarray(c['translation_per_opening']) for c in hand['components']],rotations,middle,spans,hand,pickup,plane,cal)
    for i,rotation in enumerate(rotations):
        q=(spans[i]-base['gap_zero_m'])/base['gap_per_opening'];pose=np.eye(4);pose[:3,:3]=rotation
        pose[:3,3]=middle[i]-rotation@(tip['contact_anchors_zero_grip_site_m'].mean(0)+q*tip['contact_anchors_per_opening'].mean(0))
        heights=[]
        for c in hand['components']:
            for command in [q,q+.002/base['gap_per_opening']]:
                points=transform_points(transform_points(component_vertices(c,command),pose),pickup)
                heights.extend(points@plane[:3]+plane[3])
        assert pad[i]==pytest.approx(min(heights),abs=1e-12)
        for left in np.linspace(0,.08,9):
            for right in np.linspace(0,.08,9):
                command=(left+right)/2;shift=jaw_slope*(right-left)/2
                pose[:3,3]=middle[i]-rotation@np.asarray(cal['contact_anchors_zero_grip_site_m']).mean(0)+rotation@np.array([shift,0,0])
                heights=[]
                for c in hand['components']:
                    points=transform_points(transform_points(component_vertices(c,command),pose),pickup)
                    heights.extend(points@plane[:3]+plane[3])
                assert min(heights)<=upper[i]+1e-12


def test_common_prefilter_never_rejects_a_configuration_with_exact_pad_clearance():
    vertices,faces,_,options=setup();hand=moving_hand();cal=make_contact_calibration(hand)
    options['table_prefilter']=True;options['table_plane_world']=[0,0,1,0]
    poses,meta=sampler.generate_candidates(vertices,faces,hand,**options,strategy='task_informed',contact_calibration=cal)
    assert meta['common_table_prefilter_rejected']>0
    assert len(poses)<=100
    assert 'max_qLqR_corners' in meta['common_table_prefilter_proof']


def test_proposal_angle_changes_do_not_change_hand_contact_definition():
    vertices,faces,_,options=setup();hand=moving_hand();before=deepcopy(hand)
    calibration=make_contact_calibration(hand)
    hashes=[]
    for angle in [1.,3.]:
        _,meta=sampler.generate_candidates(vertices,faces,hand,**options,
            strategy='task_informed',contact_calibration=calibration,
            proposal_normal_angle_rad=np.deg2rad(angle))
        assert not meta['hand_surface_angle_changed'] and not meta['validation_angle_consumed']
        hashes.append(make_contact_calibration(hand)['calibration_sha256'])
    assert hashes[0]==hashes[1]==calibration['calibration_sha256']
    assert hand==before


def test_stratified_origins_still_cast_rays_against_complete_bore_mesh(monkeypatch):
    from experiments.cad_grasp_selection_20261029.test_guided_sampler import tube
    vertices,faces,inner=tube();hand=deepcopy(fixture()[0]['hand'])
    triangles=vertices[faces];ids=np.flatnonzero(abs(triangles.mean(1)[:,0])>.0199)
    region=compile_sampling_region(vertices,faces,dict(type='face_ids',face_ids=inner),
        sampling_region=dict(type='intersection',regions=[dict(type='face_ids',face_ids=ids.tolist()),
            dict(type='halfspace',normal_cad=[0,1,0],offset_m=.004),
            dict(type='halfspace',normal_cad=[0,-1,0],offset_m=.004)]))
    captured={};original=sampler.previous.legacy._all_ray_hits
    def observe(origins,directions,complete):
        np.testing.assert_array_equal(complete,triangles)
        result=original(origins,directions,complete);captured['ids']=result[2]
        return result
    monkeypatch.setattr(sampler.previous.legacy,'_all_ray_hits',observe)
    _,meta=sampler.generate_candidates(vertices,faces,hand,world_T_cad_pickup=np.eye(4),
        world_T_cad_final=np.eye(4),table_plane_world=[0,0,1,.1],
        axis_point_cad_m=[0,0,0],axis_direction_cad=[0,0,1],sampling_contact_region=region,
        strategy='task_informed',table_prefilter=False)
    assert np.isin(captured['ids'],inner).any()
    assert meta['normal_accepted_pair_count']==200
    assert meta['surface_origins_drawn']==200


@pytest.mark.parametrize('field,value',[('world_T_cad_pickup',None),('world_T_cad_final',None),
    ('table_plane_world',[0,0,0,0]),('axis_direction_cad',[0,0,2])])
def test_missing_or_invalid_task_inputs_do_not_silently_use_random_fallback(field,value):
    vertices,faces,hand,options=setup();options[field]=value
    with pytest.raises(ValueError):sampler.generate_candidates(vertices,faces,hand,**options,strategy='random')


@pytest.mark.parametrize('field,value',[('surface_budget',201),('orientation_samples',11),('max_poses',101)])
def test_no_hidden_budget_increase(field,value):
    vertices,faces,hand,options=setup();options[field]=value
    with pytest.raises(ValueError):sampler.generate_candidates(vertices,faces,hand,**options)
