from pathlib import Path

import numpy as np
import pytest

from experiments.nist_mating_goals_20261028.cad_tools import (
    conversion_receipt, geometry_from_segments, mesh_component_labels,
    plane_segments, read_native_mesh, section_geometry, section_segments,
    transform_points,
)


def box(lower, upper):
    lower, upper = np.asarray(lower), np.asarray(upper)
    vertices = np.array([[x,y,z] for z in (lower[2],upper[2])
                         for y in (lower[1],upper[1]) for x in (lower[0],upper[0])],float)
    faces = np.array([[0,2,3],[0,3,1],[4,5,7],[4,7,6],
                      [0,1,5],[0,5,4],[2,6,7],[2,7,3],
                      [0,4,6],[0,6,2],[1,3,7],[1,7,5]])
    return vertices,faces


def ring_prism():
    outer=np.array([[-2,-2],[2,-2],[2,2],[-2,2]])
    inner=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    vertices=np.array([[*p,z] for z in [-1,1] for p in np.r_[outer,inner]],float)
    faces=[]
    for i in range(4):
        j=(i+1)%4
        for a,b,c,d in [(i,j,j+8,i+8),(i+4,i+12,j+12,j+4),
                         (i,i+4,j+4,j),(i+8,j+8,j+12,i+12)]:
            faces.extend([[a,b,c],[a,c,d]])
    return vertices,np.array(faces)


def test_axis_section_keeps_native_dimensions_and_exact_plane():
    vertices,faces=box([-1,-2,-3],[1,2,3])
    original=vertices.copy()
    segments=section_segments(vertices,faces,2,0.)
    assert segments.shape[1:]==(2,2)
    geometry,report=section_geometry(vertices,faces,2,0.,return_report=True)
    assert report['complete']
    assert geometry.area==pytest.approx(8.)
    assert geometry.bounds==pytest.approx((-1,-2,1,2))
    assert np.array_equal(vertices,original)


def test_oblique_plane_and_empty_section():
    vertices,faces=box([-1,-1,-1],[1,1,1])
    segments=plane_segments(vertices,faces,[1,1,1],0.)
    assert len(segments)>0
    assert np.max(np.abs(segments.sum(axis=2)))<1e-12
    assert section_segments(vertices,faces,2,3.).shape==(0,2,2)
    assert section_geometry(vertices,faces,2,3.).is_empty


def test_coplanar_triangles_ignored_and_point_tangency_empty():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    assert len(plane_segments(vertices,[[0,1,2]],[0,0,1],0))==0
    assert len(plane_segments(vertices,[[0,1,3]],[0,0,1],-1))==0


def test_connected_hollow_solid_retains_hole():
    vertices,faces=ring_prism()
    assert len(np.unique(mesh_component_labels(faces)))==1
    geometry,report=section_geometry(vertices,faces,2,0.,return_report=True)
    assert report['complete']
    assert geometry.area==pytest.approx(12.)
    assert len(geometry.interiors)==1


def test_separate_overlapping_components_are_unioned_not_xored():
    a,fa=box([-2,-1,-1],[1,1,1])
    b,fb=box([0,-1,-1],[2,1,1])
    vertices=np.r_[a,b]
    faces=np.r_[fa,fb+len(a)]
    assert len(np.unique(mesh_component_labels(faces)))==2
    geometry=section_geometry(vertices,faces,2,0.)
    assert geometry.area==pytest.approx(8.)


def test_open_linework_reports_incompleteness():
    segments=np.array([[[0,0],[1,0]],[[1,0],[1,1]],[[1,1],[0,1]]],float)
    geometry,report=geometry_from_segments(segments,return_report=True)
    assert geometry.is_empty
    assert not report['complete']
    assert report['open_linework_length']==pytest.approx(3.)


def test_rigid_transform_rejects_scale_and_preserves_inputs():
    points=np.array([[1.,2.,3.]])
    transform=np.eye(4); transform[:3,3]=[2,3,4]
    assert np.array_equal(transform_points(points,transform),[[3,5,7]])
    assert np.array_equal(points,[[1,2,3]])
    transform[0,0]=2
    with pytest.raises(ValueError): transform_points(points,transform)


def test_existing_pin_conversion_is_explicit_and_unchanged():
    root=Path(__file__).resolve().parents[2]
    directory=root/'outputs/simulation/nist_task_board_1/assets/meshes'
    original=directory/'KET16_Square_16mm.STL'
    converted=directory/'KET16_Square_16mm_m.stl'
    if not original.is_file() or not converted.is_file():
        pytest.skip('local original CAD export is not included in every checkout')
    before=(original.read_bytes(),converted.read_bytes())
    record=conversion_receipt('KET16_Square_16mm',original,converted)
    assert record['maximum_source_to_converted_nearest_vertex_distance_m']<1e-8
    assert not record['shared_original_assembly_frame_established']
    assert np.diag(np.array(record['source_mm_to_native_m_affine'])[:3,:3])==pytest.approx([.001]*3)
    vertices,_=read_native_mesh(converted)
    assert np.ptp(vertices,axis=0)==pytest.approx([.016,.010,.05],abs=1e-8)
    assert before==(original.read_bytes(),converted.read_bytes())
