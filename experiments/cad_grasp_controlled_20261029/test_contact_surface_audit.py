"""Analytic exact triangle distances and portable saved-contact reading."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import box
from experiments.cad_fingertip_contacts_20261029.test_geometry import actual_fixture
from .contact_surface_audit import (closest_triangle_surface,mesh_topology,
    material_location,audit_saved_run,write_report)


def test_triangle_interior_distance_is_not_nearest_vertex():
    vertices=np.array([[0.,0,0],[2,0,0],[0,2,0]])
    result=closest_triangle_surface([.5,.5,.1],vertices,np.array([[0,1,2]]))
    assert result['distance_m']==pytest.approx(.1)
    np.testing.assert_allclose(result['closest_point_m'],[.5,.5,0])
    np.testing.assert_allclose(result['barycentric'],[.5,.25,.25])
    assert result['closest_normal_unambiguous']
    assert min(np.linalg.norm(vertices-[.5,.5,.1],axis=1))>.7


@pytest.mark.parametrize('point,closest,distance',[
    ([2,2,0],[1,1,0],np.sqrt(2)),([-1,-1,0],[0,0,0],np.sqrt(2)),
    ([.5,-.2,.3],[.5,0,0],np.sqrt(.13))])
def test_exact_edge_and_vertex_minima(point,closest,distance):
    result=closest_triangle_surface(point,np.array([[0.,0,0],[2,0,0],[0,2,0]]),np.array([[0,1,2]]))
    assert result['distance_m']==pytest.approx(distance)
    np.testing.assert_allclose(result['closest_point_m'],closest)


def test_degenerate_triangle_falls_back_to_finite_segments():
    result=closest_triangle_surface([1,.2,0],np.array([[0.,0,0],[2,0,0],[1,0,0]]),np.array([[0,1,2]]))
    assert result['distance_m']==pytest.approx(.2)
    assert not result['closest_normal_unambiguous']


def test_closed_oriented_winding_and_ambiguous_edge_normals():
    vertices,faces=box([-1]*3,[1]*3);topology=mesh_topology(vertices,faces)
    assert topology['closed_manifold'] and topology['consistently_oriented']
    assert material_location([0,0,0],vertices,faces,topology,1.)['status']=='inside_detailed_mesh_solid'
    assert material_location([2,0,0],vertices,faces,topology,1.)['status']=='outside_detailed_mesh_solid'
    edge=closest_triangle_surface([2,2,0],vertices,faces)
    assert edge['distance_m']==pytest.approx(np.sqrt(2))
    assert not edge['closest_normal_unambiguous']
    open_mesh=mesh_topology(vertices,faces[:-1])
    assert material_location([0,0,0],vertices,faces[:-1],open_mesh,1.)['status'].startswith('unavailable')


def test_portable_supplied_hand_assets_and_right_mirror(tmp_path):
    kwargs,candidate,_=actual_fixture();hand=kwargs['hand'];before=deepcopy(hand)
    finger=next(c for c in hand['components'] if c['name']=='finger_left')
    vertices=np.asarray(finger['vertices_zero_m']);_,faces=box(vertices.min(0),vertices.max(0))
    mesh=tmp_path/'finger.stl'
    lines=['solid toy']
    for triangle in vertices[faces]:
        lines.extend(['facet normal 0 0 0','outer loop']+[f'vertex {p[0]} {p[1]} {p[2]}' for p in triangle]+['endloop','endfacet'])
    lines.append('endsolid');mesh.write_text('\n'.join(lines))
    candidate['candidate_id']=7;candidate['validity']=dict(valid=True)
    case=tmp_path/'run/cases/N00/toy';case.mkdir(parents=True)
    (case/'candidate_diagnostics.json').write_text(json.dumps([candidate]))
    (case/'selection.json').write_text(json.dumps(dict(selection=dict(selected_grasp=candidate))))
    result=audit_saved_run(tmp_path/'run',hand,mesh,collision_mesh_path=mesh,mesh_to_left_grip_site=np.eye(4))
    assert hand==before
    assert result['all_saved_valid_pose_summary']['poses']==1
    assert result['selected_pose_summary']['poses']==1
    assert result['selected_pose_summary']['thresholds'][0]['poses_with_both_contacts_within']==1
    assert max(result['selected_pose_summary']['distance_range_m'])<1e-12
    assert result['records'][0]['contacts'][1]['visual']['closest_point_m']==pytest.approx([0,0,.007])
    assert result['xml_contact_provenance']['status']=='not_supplied'
    write_report(result,tmp_path/'report')
    assert (tmp_path/'report/detailed_surface_audit.md').exists()
