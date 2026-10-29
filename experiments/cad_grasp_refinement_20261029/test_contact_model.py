"""Synthetic CAD contact-model verification; no NIST grasp experiments."""
import numpy as np
import pytest

from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region
from .contact_model import build_contact_model


def cylinder(n=72, radius=.006, z=(-.025, .025)):
    angle = np.arange(n)*2*np.pi/n
    ring = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
    vertices = np.vstack((np.column_stack((ring, np.full(n, z[0]))),
                          np.column_stack((ring, np.full(n, z[1]))),
                          [[0, 0, z[0]], [0, 0, z[1]]]))
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces.extend([[i, j, j+n], [i, j+n, i+n]])
    for i in range(n):
        j = (i+1) % n
        faces.extend([[2*n, j, i], [2*n+1, i+n, j+n]])
    return vertices, np.asarray(faces, int)


def region_for(vertices, faces):
    return compile_sampling_region(vertices, faces,
        dict(type='halfspace', normal_cad=[0, 0, 1], offset_m=-.005),
        sampling_region=dict(type='face_ids', face_ids=list(range(len(faces)//2))))


def model_for(vertices, faces, *, part=None):
    return build_contact_model(vertices, faces, region_for(vertices, faces), part=part,
        axis_point_cad_m=[0,0,0], axis_direction_cad=[0,0,1])


def test_verified_cylinder_uses_radial_normals_on_original_surface_only():
    vertices, faces = cylinder()
    before_v, before_f = vertices.tobytes(), faces.tobytes()
    model = model_for(vertices, faces)
    assert model.provenance['status'] == 'verified_cylindrical_side_normals'
    point = vertices[faces[0]].mean(0)
    expected = point.copy(); expected[2] = 0; expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.contact_normals([point], np.array([0]))[0], expected)
    # Neither protected axial continuation nor an unrelated face receives a
    # radial override, even when its XY happens to lie on the same circle.
    outside = point.copy(); outside[2] = .015
    np.testing.assert_array_equal(model.contact_normals([outside], np.array([0]))[0], model.normals[0])
    cap_id = len(faces)-1
    np.testing.assert_array_equal(model.contact_normals([vertices[faces[cap_id]].mean(0)], np.array([cap_id]))[0], model.normals[cap_id])
    assert vertices.tobytes() == before_v and faces.tobytes() == before_f


def test_axial_edge_snap_preserves_source_triangle_and_axial_height():
    vertices, faces = cylinder()
    model = model_for(vertices, faces)
    points = np.einsum('j,ijk->ik', [.6,.2,.2], vertices[faces[:8]])
    snapped, mask = model.snap_origins_to_axial_edges(points, np.arange(8))
    assert mask.all()
    np.testing.assert_allclose(snapped[:,2], points[:,2], atol=1e-15)
    assert region_for(vertices, faces)['endpoint_contains'](snapped, np.arange(8)).all()
    # These are original polygon vertices in XY, so radial tangent planes
    # support the complete convex cylinder rather than cut adjacent facets.
    normal = model.contact_normals(snapped, np.arange(8))
    support = np.einsum('nvi,ni->nv', vertices[None]-snapped[:,None], normal)
    assert np.max(support) < 1e-14


@pytest.mark.parametrize('kind', ['ellipse', 'box', 'unknown_part'])
def test_unverified_geometry_keeps_original_face_normals(kind):
    vertices, faces = cylinder(n=16)
    if kind == 'ellipse': vertices[:,0] *= 1.2
    if kind == 'box': vertices, faces = cylinder(n=4)
    model = model_for(vertices, faces, part='Gear_Large' if kind == 'unknown_part' else None)
    assert not model.verified_cylinder_bands
    points = vertices[faces[:2]].mean(1)
    np.testing.assert_array_equal(model.contact_normals(points, np.arange(2)), model.normals[:2])


def test_region_binding_and_query_validation():
    vertices, faces = cylinder()
    region = region_for(vertices, faces)
    with pytest.raises(ValueError, match='different'):
        build_contact_model(vertices*2, faces, region,
            axis_point_cad_m=[0,0,0], axis_direction_cad=[0,0,1])
    model = model_for(vertices, faces)
    with pytest.raises(ValueError, match='integer face'):
        model.contact_normals([[0,0,0]], [0.0])


def test_inward_bore_normals_are_not_relabelled_as_outward_cylinder():
    vertices, faces = cylinder()
    faces = faces[:, ::-1].copy()
    model = model_for(vertices, faces)
    assert not model.verified_cylinder_bands
    assert 'not all outward' in model.provenance['verification_reason']
