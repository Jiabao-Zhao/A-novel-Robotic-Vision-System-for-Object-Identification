"""Analytic surface-area and region tests; no grasp bank or dynamics run."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import box
from .sampling_regions import compile_sampling_region, mesh_binding


def triangle():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]])


def halfspace(normal, offset):
    return dict(type='halfspace', normal_cad=normal, offset_m=offset)


def no_protection():
    return dict(type='face_ids', protected_face_ids=[])


def test_partial_triangle_is_clipped_before_area_sampling_without_changing_source():
    vertices, faces = triangle()
    original_v, original_f = vertices.copy(), faces.copy()
    declaration = halfspace([1, 0, 0], -.25)
    original_declaration = deepcopy(declaration)
    result = compile_sampling_region(vertices, faces, declaration, boundary_margin_m=0.)
    assert result['eligible_area_m2'] == pytest.approx(.21875, abs=1e-14)
    assert len(result['sampling_triangles_cad_m']) == 2
    assert np.max(result['sampling_triangles_cad_m'][:, :, 0]) == .25
    assert np.all(result['source_face_ids'] == 0)
    assert np.all(np.cross(result['sampling_triangles_cad_m'][:, 1]-result['sampling_triangles_cad_m'][:, 0],
                           result['sampling_triangles_cad_m'][:, 2]-result['sampling_triangles_cad_m'][:, 0])[:, 2] > 0)
    assert np.array_equal(vertices, original_v) and np.array_equal(faces, original_f)
    assert declaration == original_declaration
    points = np.array([[.2, .2, 0], [.25, .2, 0], [.3, .2, 0], [.2, .2, .001], [.2, .9, 0]])
    assert result['endpoint_contains'](points, np.zeros(len(points), int)).tolist() == [True, False, False, False, False]


def test_exposed_pin_side_area_uses_20mm_inserted_end_threshold():
    vertices, faces = box([-.004, -.004, -.025], [.004, .004, .025])
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    side_ids = np.flatnonzero(np.abs(normals[:, 2]) < 1e-15)
    # Fifty-mm pin, inserted end z>=+5mm: exactly30mm of the sides remains.
    result = compile_sampling_region(vertices, faces, halfspace([0, 0, 1], -.005),
        dict(type='face_ids', face_ids=side_ids.tolist()), boundary_margin_m=0.)
    assert result['eligible_area_m2'] == pytest.approx(4*.008*.030, abs=1e-15)
    assert np.max(result['sampling_triangles_cad_m'][:, :, 2]) == pytest.approx(.005)
    assert not np.any(np.isin(result['source_face_ids'], np.setdiff1d(np.arange(len(faces)), side_ids)))


def test_positive_closed_box_keeps_coplanar_cad_walls():
    vertices, faces = box([-.02, -.008, -.0218], [.02, .008, .0152])
    clauses = []
    for axis in range(3):
        direction = np.eye(3)[axis]
        clauses.extend([halfspace(direction.tolist(), -vertices[:, axis].min()),
                        halfspace((-direction).tolist(), vertices[:, axis].max())])
    result = compile_sampling_region(vertices, faces, no_protection(),
        dict(type='intersection', regions=clauses))
    assert result['provenance']['eligible_area_fraction'] == pytest.approx(1.)
    np.testing.assert_allclose(result['sampling_triangles_cad_m'], vertices[faces])
    assert result['endpoint_contains'](vertices[faces].mean(axis=1), np.arange(len(faces))).all()


def test_overlapping_positive_union_has_no_double_counted_sampling_area():
    vertices, faces = triangle()
    positive = dict(type='union', regions=[halfspace([-1, 0, 0], .75), halfspace([1, 0, 0], -.25)])
    result = compile_sampling_region(vertices, faces, no_protection(), positive, boundary_margin_m=0.)
    assert result['eligible_area_m2'] == pytest.approx(.5, abs=1e-14)
    assert result['triangle_areas_m2'].sum() == pytest.approx(.5, abs=1e-14)


def test_protected_union_and_intersection_have_correct_disjoint_areas():
    vertices, faces = triangle()
    clauses = [halfspace([1, 0, 0], -.25), halfspace([0, 1, 0], -.25)]
    union = compile_sampling_region(vertices, faces, dict(type='union', regions=clauses), boundary_margin_m=0.)
    intersection = compile_sampling_region(vertices, faces,
        dict(type='intersection', regions=clauses), boundary_margin_m=0.)
    assert union['eligible_area_m2'] == pytest.approx(.0625, abs=1e-14)
    assert intersection['eligible_area_m2'] == pytest.approx(.375, abs=1e-14)


def test_face_id_exclusions_and_empty_allowed_region_have_no_fallback():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    result = compile_sampling_region(vertices, faces, dict(type='face_ids', protected_face_ids=[1]))
    assert result['eligible_area_m2'] == .5
    assert result['source_face_ids'].tolist() == [0]
    empty = compile_sampling_region(vertices, faces, dict(type='face_ids', protected_face_ids=[0, 1]))
    assert empty['eligible_area_m2'] == 0
    assert empty['sampling_triangles_cad_m'].shape == (0, 3, 3)
    assert not empty['endpoint_contains'](vertices[faces].mean(axis=1), np.arange(2)).any()


def test_region_binding_includes_exact_vertices_faces_and_face_order():
    vertices, faces = triangle()
    result = compile_sampling_region(vertices, faces, no_protection())
    assert {key: result[key] for key in mesh_binding(vertices, faces)} == mesh_binding(vertices, faces)
    changed = vertices.copy()
    changed[0, 0] += 1e-9
    assert mesh_binding(changed, faces)['mesh_vertices_sha256'] != result['mesh_vertices_sha256']
    assert mesh_binding(vertices, faces[:, ::-1])['mesh_faces_sha256'] != result['mesh_faces_sha256']


def test_invalid_or_missing_region_does_not_become_unrestricted():
    vertices, faces = triangle()
    with pytest.raises(ValueError, match='explicit'):
        compile_sampling_region(vertices, faces, None)
    with pytest.raises(ValueError, match='unit'):
        compile_sampling_region(vertices, faces, halfspace([0, 0, 2], 0))
    with pytest.raises(ValueError, match='face IDs'):
        compile_sampling_region(vertices, faces, dict(type='face_ids', face_ids=[100]))
