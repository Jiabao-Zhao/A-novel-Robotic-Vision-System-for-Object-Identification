"""Offline independent postprocessor arithmetic; no trial reruns."""
import numpy as np
import pytest

from .verify_controlled import independent_criteria,independent_priority,saved_stages,STAGES,independent_calibration_arrays


def test_independent_moment_surface_and_downward_criteria():
    candidate=dict(contacts_cad_m=[[.1,0,0],[.3,0,0]],approach_cad=[0,0,-1])
    reference=dict(axis_direction_cad=[0,0,1],axis_point_cad_m=[0,0,0],
        surface_centroid_cad_m=[.1,0,0],D_tau_m=.4,D_surface_m=.2)
    score=independent_criteria(candidate,reference,np.eye(4),[0,0,1,0])
    assert score['q_tau_m']==pytest.approx(.2)
    assert score['q_surface_m']==pytest.approx(.1)
    assert score['r_tau']==pytest.approx(.5)
    assert score['r_surface']==pytest.approx(.5)
    assert score['theta_down_rad']==0.


def test_independent_priority_is_minimum_then_ties_then_integer_id():
    records=[dict(candidate_id=i,validity=dict(valid=True),criteria=dict(
        r_tau=tau,r_surface=surface,theta_down_rad=angle))
        for i,tau,surface,angle in [(7,.1,.2,.3),(2,.1,.2,.3),(1,.2,0.,0.),(8,.1,.4,0.)]]
    chosen,stages=independent_priority(records,dict(r_tau=1e-10,r_surface=1e-10,theta_down_rad=1e-10))
    assert chosen['candidate_id']==2
    assert stages[0]['remaining_candidate_ids']==[2,7,8]
    assert stages[-1]['remaining_candidate_ids']==[2,7]


def test_independent_stages_do_not_promote_missing_or_unperformed_checks():
    keys={key for values in STAGES.values() for key in values}
    diagnostics={key:dict(status='pass') for key in keys}
    diagnostics['table_clearance']['status']='not_evaluated'
    result=saved_stages(dict(diagnostics=diagnostics))
    assert result['contact_valid']=='pass'
    assert result['pickup_valid']==result['joint_valid']=='not_evaluated'
    diagnostics['table_clearance']['status']='unknown'
    assert saved_stages(dict(diagnostics=diagnostics))['joint_valid']=='unavailable'
    diagnostics['pad_contact_geometry']['status']='fail'
    assert saved_stages(dict(diagnostics=diagnostics))['joint_valid']=='fail'


def test_independent_source_planes_and_common_anchor_detect_a_tampered_anchor():
    from .test_geometry import actual_fixture
    kwargs,_,calibration=actual_fixture();errors=[]
    def require(ok,label):
        if not ok:errors.append(label)
    def close(a,b,label,tol=1e-9):require(np.allclose(a,b,atol=tol,rtol=0),label)
    independent_calibration_arrays(calibration,kwargs['hand'],require,close)
    assert not errors
    calibration['contact_anchors_zero_grip_site_m'][0][2]+=.001
    independent_calibration_arrays(calibration,kwargs['hand'],require,close)
    assert 'highest permitted common source contact Z' in errors
