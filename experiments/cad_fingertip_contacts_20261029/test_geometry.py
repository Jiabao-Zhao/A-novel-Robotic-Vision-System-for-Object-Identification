"""Offline synthetic actual-finger contact and complete-solid checks."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import fixture, box
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration, FINGERTIP_FRAME
from .contact_audit import make_contact_calibration
from .geometry import GeometryValidator
from experiments.cad_grasp_refinement_20261029.test_contact_model import cylinder
from experiments.cad_grasp_refinement_20261029.contact_model import build_contact_model
from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region


def actual_fixture():
    kwargs,_=fixture()
    for name,low,high,motion in [
        ('finger_left',[-.003,-.004,-.01],[0.,.004,.009],[-.5,0,0]),
        ('finger_right',[0.,-.004,-.01],[.003,.004,.009],[.5,0,0])]:
        kwargs['hand']['components'].append(dict(name=name,is_pad=False,
            vertices_zero_m=box(low,high)[0].tolist(),translation_per_opening=motion))
    kwargs['vertices_cad'],kwargs['faces']=box([-.01,-.002,-.001],[.01,.002,.001])
    calibration=make_contact_calibration(kwargs['hand'])
    tip=fingertip_calibration(kwargs['hand'])
    grip=np.eye(4);grip[2,3]=-.007
    pose=grip@tip['grip_site_T_fingertip']
    candidate=dict(cad_T_hand=pose.tolist(),cad_T_grip_site=grip.tolist(),
        hand_pose_reference=FINGERTIP_FRAME,holding_opening_m=.02,pre_opening_m=.03,
        contacts_cad_m=[[-.01,0,0],[.01,0,0]],
        actual_contacts_hand_m=[[-.01,0,.007],[.01,0,.007]],
        actual_contact_component_names=calibration['finger_components'],
        contact_calibration_sha256=calibration['calibration_sha256'],
        first_contact_fit=dict(status='fitted'))
    return kwargs,candidate,calibration


def test_real_finger_boundary_contact_passes_without_extending_pads_or_trimming_solids():
    kwargs,candidate,calibration=actual_fixture()
    original=deepcopy(kwargs['hand'])
    result=GeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    assert result['valid'],result
    assert kwargs['hand']==original
    assert len(result['checks']['component_geometry'])==5
    assert all(result['checks']['actual_finger_support'].values())
    assert result['actual_contact_distance_from_physical_tip_m']==pytest.approx([.002,.002])
    for report in result['checks']['component_geometry']:
        if report['is_pad']:assert report['contact_witness_count']==0


def test_saved_pad_contact_is_forbidden_even_when_intended_finger_points_are_valid():
    kwargs,candidate,calibration=actual_fixture()
    kwargs['vertices_cad'],kwargs['faces']=box([-.01,-.002,-.004],[.01,.002,.001])
    result=GeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    assert not result['valid']
    assert result['diagnostics']['pad_contact_geometry']['status']=='fail'
    assert any('left_pad' in reason for reason in result['reasons'])


def test_complete_finger_solid_penetration_and_other_failures_are_independent():
    kwargs,candidate,calibration=actual_fixture()
    protrusion,faces=box([-.012,-.001,-.0005],[-.009,.001,.0005])
    n=len(kwargs['vertices_cad'])
    kwargs['vertices_cad']=np.concatenate((kwargs['vertices_cad'],protrusion))
    kwargs['faces']=np.concatenate((kwargs['faces'],faces+n))
    kwargs['table_plane']=[0,0,1,0]
    receiver,faces=box([-.03]*3,[.03]*3)
    kwargs['receivers']=[dict(name='blocked',vertices=receiver,faces=faces,world_T_mesh=np.eye(4))]
    result=GeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    for key in ['pad_contact_geometry','target_closing_collision','table_clearance','final_receiver_collision']:
        assert result['diagnostics'][key]['status']=='fail',result


@pytest.mark.parametrize('kind',['position','frame','hash','components','missing'])
def test_candidate_contact_claims_are_verified(kind):
    kwargs,candidate,calibration=actual_fixture()
    if kind=='position':candidate['actual_contacts_hand_m'][0][0]-=.001
    elif kind=='frame':candidate['cad_T_grip_site'][0][3]+=.001
    elif kind=='hash':candidate['contact_calibration_sha256']='wrong'
    elif kind=='components':candidate['actual_contact_component_names'].reverse()
    else:candidate.pop('actual_contacts_hand_m')
    result=GeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    assert not result['valid']
    if kind=='position':assert result['diagnostics']['contact_reconstruction']['status']=='fail'
    else:assert result['status']=='unavailable_input'


def test_failed_first_contact_fit_is_invalid_not_a_pad_anchor_fallback():
    kwargs,candidate,calibration=actual_fixture()
    candidate['first_contact_fit']['status']='first_contact_not_on_distal_finger'
    candidate.pop('actual_contacts_hand_m')
    result=GeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    assert result['status']=='invalid'
    assert result['diagnostics']['pad_contact_geometry']['status']=='fail'
    assert result['diagnostics']['table_clearance']['status']=='not_evaluated'


def test_foreign_calibration_and_nonboolean_protected_callback_are_unavailable():
    kwargs,candidate,calibration=actual_fixture()
    bad=deepcopy(calibration);bad['hand_geometry_sha256']='foreign'
    assert GeometryValidator(**kwargs,contact_calibration=bad).validate(candidate)['status']=='unavailable_input'
    def numeric_mask(points,*,contact_polygons,contact_face_ids):
        if len(points)==0:return dict(status='available',eligible=np.ones(0,bool),polygon_eligible=np.ones(0,bool))
        return dict(status='available',eligible=np.ones(len(points)),polygon_eligible=np.ones(len(contact_polygons)))
    kwargs['protected_contact_check']=numeric_mask
    result=GeometryValidator(**kwargs,contact_calibration=calibration).validate(candidate)
    assert result['diagnostics']['protected_surfaces']['status']=='unavailable'


@pytest.mark.parametrize('finite_area',[False,True])
def test_verified_round_contact_uses_actual_finger_plane_and_support_dimension(finite_area):
    kwargs,candidate,calibration=actual_fixture()
    vertices,faces=cylinder(n=72,radius=.01,z=(-.002,.002))
    # Cylinder axis is hand Y. Half a facet rotation puts a real positive-area
    # side facet parallel to the real finger plane; no rotation gives an exact
    # source-edge tangent line. Both retain every original CAD triangle.
    angle=np.pi/72 if finite_area else 0.
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],
                       [0,0,1],[np.sin(angle),np.cos(angle),0.]])
    # This mapping has negative determinant: reverse triangle winding after it.
    vertices=vertices@rotation.T
    faces=faces[:,[0,2,1]]
    kwargs.update(vertices_cad=vertices,faces=faces)
    region=compile_sampling_region(vertices,faces,
        dict(type='halfspace',normal_cad=[0,1,0],offset_m=-1.))
    model=build_contact_model(vertices,faces,region,axis_point_cad_m=[0,0,0],axis_direction_cad=[0,1,0])
    assert model.verified_cylinder_bands
    radius=.01*np.cos(np.pi/72) if finite_area else .01
    candidate.update(holding_opening_m=2*radius,
        contacts_cad_m=[[-radius,0,0],[radius,0,0]],
        actual_contacts_hand_m=[[-radius,0,.007],[radius,0,.007]])
    result=GeometryValidator(**kwargs,contact_calibration=calibration,contact_model=model).validate(candidate)
    assert result['valid'],result
    reports=[r for r in result['checks']['component_geometry'] if r['actual_contact_component']]
    if finite_area:
        assert sum(r['source_facet_area_support_witnesses'] for r in reports)>0
    else:
        assert sum(r['exact_finger_plane_boundary_corrections'] for r in reports)>0
