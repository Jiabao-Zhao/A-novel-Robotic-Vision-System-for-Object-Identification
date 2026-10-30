"""Offline tests of decoupled angles and truthful saved-stage classification."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cad_fingertip_contacts_20261029.test_geometry import actual_fixture as historical_actual_fixture
from experiments.cad_grasp_selection_20261028.test_geometry import fixture
from experiments.cad_grasp_selection_20261029.test_fingertip import tip_candidate
from .geometry import FingerGeometryValidator,PadGeometryValidator,classify_stages,STAGES
from .calibration import make_contact_calibration

ROOT=Path(__file__).resolve().parents[2]


def actual_fixture():
    kwargs,candidate,_=historical_actual_fixture()
    calibration=make_contact_calibration(kwargs['hand'])
    candidate['contact_calibration_sha256']=calibration['calibration_sha256']
    return kwargs,candidate,calibration


def test_fixed_fit_acceptance_changes_without_changing_hand_labels_or_geometry():
    kwargs,candidate,calibration=actual_fixture()
    theta=np.deg2rad(2.)
    rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    kwargs['vertices_cad']=kwargs['vertices_cad']@rotation.T
    halfwidth=.01*np.cos(theta)+.002*np.sin(theta)
    y=.002*np.cos(theta)-.01*np.sin(theta)
    candidate.update(holding_opening_m=2*halfwidth,
        contacts_cad_m=[[-halfwidth,y,0],[halfwidth,-y,0]],
        actual_contacts_hand_m=[[-halfwidth,y,.007],[halfwidth,-y,.007]])
    original=deepcopy((kwargs['hand'],candidate,calibration))
    strict=FingerGeometryValidator(**kwargs,contact_calibration=calibration,validation_normal_angle_rad=np.deg2rad(1.))
    broad=FingerGeometryValidator(**kwargs,contact_calibration=calibration,validation_normal_angle_rad=np.deg2rad(3.))
    low,high=strict.validate(candidate),broad.validate(candidate)
    assert not low['valid'] and high['valid'],(low,high)
    assert low['diagnostics']['surface_normals']['status']=='fail'
    assert high['diagnostics']['surface_normals']['status']=='pass'
    for category in ['target_closing_collision','table_clearance','final_receiver_collision']:
        assert low['diagnostics'][category]['status']==high['diagnostics'][category]['status']=='pass'
    assert (kwargs['hand'],candidate,calibration)==original
    for name in strict.finger_regions:
        np.testing.assert_array_equal(strict.finger_regions[name]['allowed_equations'],broad.finger_regions[name]['allowed_equations'])
    assert low['angular_parameters']['calibration_sha256']==high['angular_parameters']['calibration_sha256']
    assert high['angular_parameters']['hand_surface_normal_angle_rad']==pytest.approx(np.deg2rad(1.))


def test_saved_hand_patch_membership_and_anchors_stay_identical_across_all_angles():
    kwargs,_=fixture()
    kwargs['hand']=json.loads((ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
    calibration=make_contact_calibration(kwargs['hand'])
    before=deepcopy(calibration)
    validators=[FingerGeometryValidator(**kwargs,contact_calibration=calibration,
        validation_normal_angle_rad=np.deg2rad(degrees)) for degrees in [.5,1,2,3,5]]
    assert all(not validator.errors for validator in validators)
    for validator in validators:
        assert validator.contact_calibration==before
        assert validator.contact_calibration['calibration_sha256']==before['calibration_sha256']
        for name in validators[0].finger_regions:
            np.testing.assert_array_equal(validator.finger_regions[name]['allowed_equations'],validators[0].finger_regions[name]['allowed_equations'])
        np.testing.assert_array_equal(validator.tip['physical_finger_tips_zero_grip_site_m'],validators[0].tip['physical_finger_tips_zero_grip_site_m'])


def test_calibration_from_a_different_hand_angle_is_explicitly_unavailable():
    kwargs,candidate,calibration=actual_fixture()
    changed=make_contact_calibration(kwargs['hand'],normal_angle_rad=np.deg2rad(3.))
    validator=FingerGeometryValidator(**kwargs,contact_calibration=changed,validation_normal_angle_rad=np.deg2rad(3.))
    assert validator.errors
    assert validator.validate(candidate)['status']=='unavailable_input'


def test_pad_validator_keeps_same_saved_hand_and_anchors_when_acceptance_changes():
    kwargs,_=fixture();candidate=tip_candidate(kwargs['hand']);before=deepcopy(kwargs['hand'])
    a=PadGeometryValidator(**kwargs,validation_normal_angle_rad=np.deg2rad(.5))
    b=PadGeometryValidator(**kwargs,validation_normal_angle_rad=np.deg2rad(5.))
    assert a.validate(candidate)['valid'] and b.validate(candidate)['valid']
    np.testing.assert_array_equal(a.calibration['centers_zero_m'],b.calibration['centers_zero_m'])
    assert kwargs['hand']==before


def test_stage_classification_reserves_protected_surfaces_for_explicit_task_stages():
    keys=set(key for values in STAGES.values() for key in values)
    diagnostics={key:dict(status='pass') for key in keys}
    diagnostics['protected_surfaces']['status']='fail'
    stages=classify_stages(dict(diagnostics=diagnostics))
    assert stages['contact_valid']['status']=='pass'
    assert stages['pickup_valid']['status']=='pass'
    assert stages['final_goal_valid']['status']=='fail'
    assert stages['joint_valid']['status']=='fail'
    assert stages['protected_task_compatible']['status']=='fail'


@pytest.mark.parametrize('status',['not_evaluated','unavailable'])
def test_unknown_or_unperformed_checks_never_pass(status):
    keys=set(key for values in STAGES.values() for key in values)
    diagnostics={key:dict(status='pass') for key in keys}
    diagnostics['table_clearance']['status']=status
    stages=classify_stages(diagnostics)
    assert stages['contact_valid']['status']=='pass'
    assert stages['pickup_valid']['status']==status
    assert stages['joint_valid']['status']==status
    assert stages['final_goal_valid']['status']=='pass'
    del diagnostics['table_clearance']
    assert classify_stages(diagnostics)['joint_valid']['status']=='unavailable'


def test_known_failure_still_fails_when_another_check_was_not_evaluated():
    diagnostics={key:dict(status='not_evaluated') for key in set(k for v in STAGES.values() for k in v)}
    diagnostics['pad_contact_geometry']['status']='fail'
    stages=classify_stages(dict(validity=dict(diagnostics=diagnostics)))
    assert stages['contact_valid']['status']=='fail'
    assert stages['joint_valid']['status']=='fail'
    assert stages['joint_valid']['category_statuses']['table_clearance']=='not_evaluated'


def test_complete_supplied_hand_needs_no_hardcoded_source_file(monkeypatch,tmp_path):
    kwargs,candidate,_=actual_fixture()
    from experiments.cad_fingertip_contacts_20261029 import contact_audit as historical
    from . import calibration as portable
    missing=tmp_path/'absent_saved_hand.json'
    monkeypatch.setattr(historical,'HAND',missing)
    monkeypatch.setattr(portable,'HAND',missing)
    calibration=portable.make_contact_calibration(kwargs['hand'])
    candidate['contact_calibration_sha256']=calibration['calibration_sha256']
    result=FingerGeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    assert result['valid'],result
    assert calibration['provenance']['source_hand_model_sha256'] is None
