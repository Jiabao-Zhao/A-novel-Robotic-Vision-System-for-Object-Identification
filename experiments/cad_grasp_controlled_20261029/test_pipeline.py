"""Synthetic policy-budget and cached-geometry tests; no saved NIST trial."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import box,fixture,clear_region
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration,FINGERTIP_FRAME
from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region
from experiments.cad_grasp_selection_20261029.test_fingertip import tip_candidate
from . import pipeline
from .calibration import make_contact_calibration
from .geometry import FingerGeometryValidator,PadGeometryValidator,STAGES,stage_classification


def request_and_candidates():
    kwargs,_=fixture();hand=kwargs['hand']
    # The saved-pad model protrudes 0.5 mm past these synthetic finger bodies.
    for component in hand['components']:
        if component['is_pad']:
            points=np.asarray(component['vertices_zero_m'])
            points[:,0]+=.0005 if component['name']=='left_pad' else -.0005
            component['vertices_zero_m']=points.tolist()
    for name,low,high,motion in [
        ('finger_left',[-.003,-.004,-.01],[0.,.004,.009],[-.5,0,0]),
        ('finger_right',[0.,-.004,-.01],[.003,.004,.009],[.5,0,0])]:
        hand['components'].append(dict(name=name,is_pad=False,
            vertices_zero_m=box(low,high)[0].tolist(),translation_per_opening=motion))
    vertices,faces=box([-.01,-.002,-.001],[.01,.002,.001])
    protected=dict(type='face_ids',face_ids=[])
    request=dict(part='synthetic box',vertices_cad_m=vertices,faces=faces,hand=hand,
        world_T_cad_pickup=np.eye(4),world_T_cad_final=np.eye(4),
        table_plane_world=np.array([0.,0,1.,.1]),axis_point_cad_m=[0,0,0],axis_direction_cad=[0,0,1],
        protected_contact_region=protected,protected_contact_check=clear_region,
        sampling_contact_region=compile_sampling_region(vertices,faces,protected),
        receivers=kwargs['receivers'],simulator_object_poses_read=False)
    calibration=make_contact_calibration(hand);tip=fingertip_calibration(hand)
    grip=np.eye(4);grip[2,3]=-.007
    finger=dict(candidate_id=0,pair_id=0,contact_mode='finger',approach_cad=[0,0,1],
        cad_T_hand=(grip@tip['grip_site_T_fingertip']).tolist(),cad_T_grip_site=grip.tolist(),
        hand_pose_reference=FINGERTIP_FRAME,holding_opening_m=.02,pre_opening_m=.03,
        contacts_cad_m=[[-.01,0,0],[.01,0,0]],actual_contacts_hand_m=[[-.01,0,.007],[.01,0,.007]],
        actual_contact_component_names=calibration['finger_components'],
        contact_calibration_sha256=calibration['calibration_sha256'],first_contact_fit=dict(status='fitted'))
    pad=dict(tip_candidate(hand),contact_mode='pad')
    return request,dict(pad=pad,finger=finger)


def direct(evaluator,candidate,pickup,table):
    kwargs=dict(vertices_cad=evaluator.vertices,faces=evaluator.faces,hand=evaluator.hand,
        pickup_T=pickup,final_T=evaluator.final,table_plane=table,
        protected_contact_check=evaluator.protected,receivers=evaluator.request['receivers'],
        contact_model=evaluator.model,validation_normal_angle_rad=evaluator.validation_angle)
    if candidate['contact_mode']=='finger':
        return FingerGeometryValidator(**kwargs,contact_calibration=evaluator.calibration).validate(candidate)
    return PadGeometryValidator(**kwargs).validate(candidate)


@pytest.mark.parametrize('mode',['pad','finger'])
@pytest.mark.parametrize('initial_clear',[False,True])
def test_cached_local_and_final_geometry_matches_direct_at_two_observed_pickups(mode,initial_clear):
    request,candidates=request_and_candidates()
    request['table_plane_world']=np.array([0.,0,1.,.1 if initial_clear else 0.])
    evaluator=pipeline.Evaluator(request);candidate=evaluator.fit(candidates[mode],mode)
    pickup=np.eye(4);pickup[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pickup[:3,3]=[.02,-.03,.001]
    table=np.array([0.,0,1.,0. if initial_clear else .1])
    first=evaluator.evaluate(candidate,evaluator.pickup0,evaluator.table0)
    record=evaluator.evaluate(candidate,pickup,table);cached=record['validity']
    fresh=direct(evaluator,candidate,pickup,table)
    assert first['validity']['valid']==initial_clear,first['validity']
    assert fresh['valid']==(not initial_clear),fresh
    assert cached['status']==fresh['status']
    assert {k:v['status'] for k,v in cached['diagnostics'].items()}=={k:v['status'] for k,v in fresh['diagnostics'].items()}
    assert cached['checks']['table_minimum_clearance_m']==pytest.approx(fresh['checks']['table_minimum_clearance_m'],abs=1e-12)
    for old,new in zip(cached['checks']['component_geometry'],fresh['checks']['component_geometry']):
        assert old['table_minimum_clearance_m']==pytest.approx(new['table_minimum_clearance_m'],abs=1e-12)
    for key in ['cad_T_hand','cad_T_grip_site','world_T_hand_pickup','world_T_grip_site_pickup',
                'world_T_hand_final','world_T_grip_site_final','approach_world']:
        assert (key in cached)==(key in fresh),(key,cached,fresh)
        if key in fresh:np.testing.assert_allclose(cached[key],fresh[key],atol=1e-12,rtol=0)
    assert cached['stages']==stage_classification(fresh)
    assert evaluator.counters['unique_local_and_final_validations']==1
    assert evaluator.counters['reused_local_and_final_validations']==1


@pytest.mark.parametrize('field,value',[('hand_pose_reference','foreign frame'),('contact_inset_from_pad_tip_m',.002)])
def test_cached_validity_does_not_hide_changed_contact_metadata(field,value):
    request,candidates=request_and_candidates();evaluator=pipeline.Evaluator(request)
    candidate=evaluator.fit(candidates['pad'],'pad')
    assert evaluator.evaluate(candidate,evaluator.pickup0,evaluator.table0)['validity']['valid']
    changed=deepcopy(candidate);changed[field]=value
    cached=evaluator.evaluate(changed,evaluator.pickup0,evaluator.table0)['validity']
    fresh=direct(evaluator,changed,evaluator.pickup0,evaluator.table0)
    assert cached['status']==fresh['status']!='valid'
    assert evaluator.counters['unique_local_and_final_validations']==2
    assert evaluator.counters['reused_local_and_final_validations']==0


def test_comparison_has_100_per_policy_and_predeclared_50_50_combined_not_300_evaluations(monkeypatch):
    request,_=request_and_candidates();evaluator=pipeline.Evaluator(request)
    raw=[dict(candidate_id=i,combined_contact_mode='pad' if i%2==0 else 'finger') for i in range(100)]
    original=deepcopy(raw);calls=[]
    def fake_fit(candidate,mode):return dict(candidate,contact_mode=mode)
    def fake_evaluate(candidate,pickup,table):
        mode=candidate['contact_mode'];calls.append((candidate['candidate_id'],mode))
        diagnostics={k:dict(status='pass') for k in set(k for values in STAGES.values() for k in values)}
        if mode=='finger':diagnostics['surface_normals']['status']='fail'
        validity=dict(status='valid' if mode=='pad' else 'invalid',diagnostics=diagnostics)
        return dict(candidate,valid=mode=='pad',validity=validity,
            criteria=dict(r_tau=0.,r_surface=0.,theta_down_rad=0.))
    monkeypatch.setattr(evaluator,'fit',fake_fit);monkeypatch.setattr(evaluator,'evaluate',fake_evaluate)
    decisions,pools=evaluator.compare(raw,np.eye(4),[0,0,1,.1],seed=7,strategy='random')
    assert len(calls)==len(set(calls))==200
    assert raw==original
    assert {mode:decision['proposed_count'] for mode,decision in decisions.items()}==dict(pad=100,finger=100,combined=100)
    assert {mode:decision['valid_count'] for mode,decision in decisions.items()}==dict(pad=100,finger=0,combined=50)
    assert Counter(record['contact_mode'] for record in pools['combined'])==dict(pad=50,finger=50)
    for record in pools['combined']:
        assert record is pools[record['contact_mode']][record['candidate_id']]


def test_small_actual_combined_request_is_portable_and_uses_six_total_checks(monkeypatch,tmp_path):
    request,_=request_and_candidates()
    from experiments.cad_fingertip_contacts_20261029 import contact_audit as historical
    from . import calibration as portable
    missing=tmp_path/'no_installed_hand_or_source_asset.json'
    monkeypatch.setattr(historical,'HAND',missing);monkeypatch.setattr(portable,'HAND',missing)
    result=pipeline.select_grasp(request,seed=3,mode='combined',strategy='random',
        max_poses=6,surface_budget=12,orientation_samples=3)
    assert result['proposed_count']==len(result['diagnostics'])==6
    assert Counter(r['contact_mode'] for r in result['diagnostics'])==dict(pad=3,finger=3)
    assert result['evaluation_counters']['unique_local_and_final_validations']<=6
    assert result['evaluation_counters']['unique_finger_first_contact_fits']<=3
    assert result['no_pickup_or_assembly_execution'] and result['no_VLM']
