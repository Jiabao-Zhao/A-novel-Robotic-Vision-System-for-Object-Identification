"""Synthetic supplemental witness correction; no NIST candidates generated."""
import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import fixture
from experiments.cad_grasp_selection_20261029.sampler import pose_from_pair
from experiments.cad_grasp_selection_20261029.test_fingertip import tip_candidate
from .geometry import GeometryValidator
from .support_geometry import SupportValidator
from .test_contact_model import cylinder, model_for
from .test_geometry import cylinder_case


def odd_cylinder_case():
    kwargs, _ = fixture()
    vertices, faces = cylinder(n=71, radius=.006)
    kwargs.update(vertices_cad=vertices, faces=faces)
    # At +X there is a source edge; at -X the opposite surface is a full facet.
    # Its plane normal is -X, but its corners' smooth normals differ by2.535deg.
    apothem = .006*np.cos(np.pi/71)
    contacts = [[-apothem,0,-.01],[.006,0,-.01]]
    candidate = pose_from_pair(contacts,[0,0,1],kwargs['hand'])
    candidate['pre_opening_m'] = .03
    return kwargs, candidate, model_for(vertices,faces)


def test_opposite_flat_facet_uses_actual_support_normal_without_changing_geometry():
    kwargs,candidate,model = odd_cylinder_case()
    # Keep this regression's contact area wholly inside the opposite facet, so
    # it isolates facet/radial normal consistency rather than the separate
    # tolerance-expanded adjacent-edge witness issue.
    for component in kwargs['hand']['components']:
        if component['is_pad']:
            points=np.asarray(component['vertices_zero_m'])
            points[:,1]=np.where(points[:,1]>0,.00025,-.00025)
            component['vertices_zero_m']=points.tolist()
    vertices_before,faces_before = kwargs['vertices_cad'].tobytes(),kwargs['faces'].tobytes()
    old = GeometryValidator(**kwargs,contact_model=model).validate(candidate)
    new = SupportValidator(**kwargs,contact_model=model).validate(candidate)
    assert old['diagnostics']['surface_normals']['status']=='pass'
    assert old['diagnostics']['pad_contact_geometry']['status']=='fail'
    assert all(value['status']=='pass' for key,value in old['diagnostics'].items()
               if key!='pad_contact_geometry'),old
    assert new['valid'],new
    assert all(value['status']=='pass' for value in new['diagnostics'].values())
    policy=new['support_normal_policy']
    assert policy['positive_area_facet_queries']>0
    assert policy['normal_angle_rad']==pytest.approx(np.deg2rad(1))
    assert policy['cylinder_to_inscribed_mesh_maximum_sag_m'][0]==pytest.approx(
        .006*(1-np.cos(np.pi/71)),rel=1e-7)
    assert not policy['physical_contact_accuracy_validated']
    assert (kwargs['vertices_cad'].tobytes(),kwargs['faces'].tobytes())==(vertices_before,faces_before)


def test_support_rule_cannot_hide_original_mesh_penetration():
    kwargs,candidate,model=cylinder_case(interior=True)
    result=SupportValidator(**kwargs,contact_model=model).validate(candidate)
    assert not result['valid']
    assert result['diagnostics']['target_closing_collision']['status']=='fail'


def test_sixteen_mm_wide_pad_preserves_facet_area_and_exact_adjacent_edge_lines():
    kwargs,candidate,model=odd_cylinder_case()
    for component in kwargs['hand']['components']:
        if component['is_pad']:
            points=np.asarray(component['vertices_zero_m'])
            points[:,1]=np.where(points[:,1]>0,.008,-.008)
            component['vertices_zero_m']=points.tolist()
    old=GeometryValidator(**kwargs,contact_model=model).validate(candidate)
    validator=SupportValidator(**kwargs,contact_model=model)
    before=validator.target.surface_contacts
    new=validator.validate(candidate)
    assert not old['valid'] and new['valid'],new
    assert new['support_normal_policy']['positive_area_facet_queries']>0
    assert new['support_normal_policy']['exact_inner_plane_boundary_corrections']>0
    assert new['support_normal_policy']['boundary_ribbon_guard_m']==8e-10
    assert validator.target.surface_contacts==before


def test_contacts_away_from_actual_inner_pad_plane_are_not_collapsed():
    from experiments.cad_grasp_selection_20261028.test_geometry import box
    kwargs,candidate,model=odd_cylinder_case()
    validator=SupportValidator(**kwargs,contact_model=model)
    validator.validate(candidate)
    triangle=kwargs['vertices_cad'][kwargs['faces'][0]]
    polygon=np.asarray([[.6,.2,.2],[.55,.25,.2],[.6,.15,.25]])@triangle
    assert model._band_mask(polygon,np.zeros(3,int),model._bands[0]).all()
    witness=dict(triangle_index=0,polygon_world_m=polygon.tolist(),
                 point_world_m=polygon.mean(axis=0).tolist())
    # This actual right-pad solid has an outer face near the target; its usable
    # inner plane is3mm inward. The witness cannot borrow the boundary correction.
    vertices,_=box([.003,-.008,-.008],[.006,.008,.008])
    result=validator._boundary_witnesses([witness],vertices)
    assert result[0] is witness
    assert result[0]['polygon_world_m']==polygon.tolist()


def test_axial_line_support_keeps_verified_radial_normals():
    kwargs,candidate,model=cylinder_case()
    result=SupportValidator(**kwargs,contact_model=model).validate(candidate)
    assert result['valid'],result
    assert result['diagnostics']['surface_normals']['status']=='pass'


@pytest.mark.parametrize('model_present',[False,True])
def test_non_circular_flat_checks_and_records_are_unchanged(model_present):
    kwargs,_=fixture()
    theta=np.deg2rad(2)
    rotation=np.array([[np.cos(theta),-np.sin(theta),0],
                       [np.sin(theta),np.cos(theta),0],[0,0,1]])
    kwargs['vertices_cad']=kwargs['vertices_cad']@rotation.T
    candidate=pose_from_pair([[-.01/np.cos(theta),0,0],[.01/np.cos(theta),0,0]],
                            [0,0,1],kwargs['hand'])
    candidate['pre_opening_m']=.03
    model=None
    if model_present:
        from .contact_model import build_contact_model
        model=build_contact_model(kwargs['vertices_cad'],kwargs['faces'],part='KET16_Square_16mm')
    old=GeometryValidator(**kwargs,contact_model=model).validate(candidate)
    new=SupportValidator(**kwargs,contact_model=model).validate(candidate)
    assert old==new
    assert not new['valid']
    assert new['diagnostics']['surface_normals']['status']=='fail'


def test_no_model_valid_case_remains_byte_equivalent():
    import json
    kwargs,_=fixture()
    candidate=tip_candidate(kwargs['hand'])
    old=GeometryValidator(**kwargs).validate(candidate)
    new=SupportValidator(**kwargs).validate(candidate)
    assert old['valid'] and new['valid']
    assert json.dumps(old,sort_keys=True)==json.dumps(new,sort_keys=True)
