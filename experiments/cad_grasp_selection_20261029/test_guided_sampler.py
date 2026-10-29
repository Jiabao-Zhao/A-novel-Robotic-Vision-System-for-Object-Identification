"""Synthetic-only origin guidance and complete-mesh ray checks."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import box, fixture
from experiments.cad_grasp_selection_20261028.test_pipeline import request_fixture
from . import pipeline, sampler
from .sampling_regions import compile_sampling_region


def hand():
    return fixture()[0]['hand']


def pin():
    return box([-.005, -.005, -.025], [.005, .005, .025])


def pin_region(vertices, faces):
    return compile_sampling_region(vertices, faces,
        dict(type='halfspace', normal_cad=[0, 0, 1], offset_m=-.005))


def tube():
    """A closed rectangular tube with a genuine through-bore along Z."""
    outer = np.array([[-.02, -.015], [.02, -.015], [.02, .015], [-.02, .015]])
    inner = np.array([[-.01, -.005], [.01, -.005], [.01, .005], [-.01, .005]])
    points, faces = [], []
    def quad(p):
        start = len(points)
        points.extend(p)
        faces.extend([[start, start+1, start+2], [start, start+2, start+3]])
    for ring in [outer, inner[::-1]]:
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            quad([[*a, -.01], [*b, -.01], [*b, .01], [*a, .01]])
    inner_faces = list(range(8, 16))
    for i in range(4):
        j = (i+1) % 4
        p = [[*outer[i], .01], [*outer[j], .01], [*inner[j], .01], [*inner[i], .01]]
        quad(p)
        quad([[x, y, -.01] for x, y, _ in p[::-1]])
    vertices, inverse = np.unique(points, axis=0, return_inverse=True)
    return vertices, inverse[np.asarray(faces)], inner_faces


def test_long_pin_origins_and_both_contacts_avoid_inserted_twenty_mm():
    vertices, faces = pin()
    region = pin_region(vertices, faces)
    before_v, before_f, before_hand = vertices.tobytes(), faces.tobytes(), deepcopy(hand())
    candidates, metadata = sampler.generate_candidates(vertices, faces, before_hand,
        sampling_contact_region=region, seed=39, surface_budget=80, orientation_samples=4)
    assert candidates
    origins = np.asarray(metadata['surface_origins_cad_m'])
    assert len(origins) == metadata['surface_origins_sampled'] == 80
    assert np.all(origins[:, 2] < .005)
    for candidate in candidates:
        assert region['endpoint_contains'](candidate['contacts_cad_m'], candidate['contact_face_ids']).all()
        assert np.all(np.asarray(candidate['contacts_cad_m'])[:, 2] < .005)
    # Allowed 30 mm of four 10 mm-wide sides plus one 10x10 mm end,
    # with only the existing 10 nm protected boundary guard removed.
    assert metadata['eligible_surface_area_m2'] == pytest.approx(.0013-4*.01*1e-8, abs=1e-14)
    assert metadata['full_CAD_surface_area_m2'] == pytest.approx(.0022)
    assert metadata['eligible_surface_fraction'] == pytest.approx((.0013-4*.01*1e-8)/.0022)
    assert vertices.tobytes() == before_v and faces.tobytes() == before_f
    assert before_hand == hand()


def test_partial_triangle_origins_keep_original_face_ids_and_area_weights(monkeypatch):
    vertices, faces = pin()
    region = pin_region(vertices, faces)
    # Cutting a long side face produces more patches than retained source
    # faces; sampling complete original triangles would include protected tips.
    ids = region['source_face_ids']
    assert len(ids) > len(np.unique(ids))
    captured = {}
    original = sampler.legacy._surface_samples
    def observe(triangles, areas, count, rng):
        captured.update(triangles=triangles.copy(), areas=areas.copy(), count=count)
        return original(triangles, areas, count, rng)
    monkeypatch.setattr(sampler.legacy, '_surface_samples', observe)
    _, metadata = sampler.generate_candidates(vertices, faces, hand(),
        sampling_contact_region=region, surface_budget=37, max_poses=0)
    np.testing.assert_array_equal(captured['triangles'], region['sampling_triangles_cad_m'])
    np.testing.assert_array_equal(captured['areas'], region['triangle_areas_m2'])
    assert captured['count'] == metadata['surface_origins_drawn'] == 37
    assert metadata['clipped_patch_source_face_ids'] == ids.tolist()
    for patch, face in zip(metadata['surface_origin_patch_indices'], metadata['surface_origin_source_face_ids']):
        assert face == ids[patch]


def test_rays_see_full_bore_and_protected_surfaces_before_pair_bank(monkeypatch):
    vertices, faces, inner_faces = tube()
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals = cross/np.linalg.norm(cross, axis=1)[:, None]
    outer_x_faces = np.flatnonzero(np.abs(triangles.mean(axis=1)[:, 0]) > .0199)
    outer_x_faces = [int(i) for i in outer_x_faces if abs(normals[i, 0]) > .99]
    region = compile_sampling_region(vertices, faces,
        dict(type='face_ids', protected_face_ids=inner_faces),
        sampling_region=dict(type='intersection', regions=[
            dict(type='face_ids', face_ids=outer_x_faces),
            dict(type='halfspace', normal_cad=[0, 1, 0], offset_m=.004),
            dict(type='halfspace', normal_cad=[0, -1, 0], offset_m=.004)]))
    captured = {}
    original = sampler.legacy._all_ray_hits
    def observe(origins, directions, ray_triangles):
        np.testing.assert_array_equal(ray_triangles, triangles)
        result = original(origins, directions, ray_triangles)
        captured['hit_face_ids'] = result[2]
        return result
    monkeypatch.setattr(sampler.legacy, '_all_ray_hits', observe)
    candidates, metadata = sampler.generate_candidates(vertices, faces, hand(),
        sampling_contact_region=region, surface_budget=24, orientation_samples=7, max_poses=100, seed=7)
    assert metadata['surface_origins_sampled'] == 24
    assert metadata['unique_ray_hits'] == 3*24
    assert metadata['second_endpoint_region_rejections'] == 2*24
    assert np.isin(captured['hit_face_ids'], inner_faces).any()
    assert metadata['normal_accepted_pair_count'] == 24
    assert metadata['proposal_pool_size'] == 24*7
    assert len(candidates) == 100
    for candidate in candidates:
        assert region['endpoint_contains'](candidate['contacts_cad_m'], candidate['contact_face_ids']).all()
        assert candidate['contact_span_m'] == pytest.approx(.04)
        assert set(candidate['contact_face_ids']).issubset(set(outer_x_faces))
    assert metadata['region_filter_before_pair_orientation_bank']
    assert metadata['ray_intersection_geometry'].startswith('complete original CAD')


def test_empty_region_generates_zero_origins_pairs_and_poses_without_rays(monkeypatch):
    vertices, faces = pin()
    region = compile_sampling_region(vertices, faces,
        dict(type='face_ids', protected_face_ids=list(range(len(faces)))))
    def forbidden(*args, **kwargs):
        raise AssertionError('Empty allowed region must not sample or raycast')
    monkeypatch.setattr(sampler.legacy, '_surface_samples', forbidden)
    monkeypatch.setattr(sampler.legacy, '_all_ray_hits', forbidden)
    candidates, metadata = sampler.generate_candidates(vertices, faces, hand(), sampling_contact_region=region)
    assert candidates == []
    assert metadata['status'] == 'empty_allowed_contact_surface'
    assert metadata['eligible_surface_area_m2'] == metadata['eligible_surface_fraction'] == 0
    assert metadata['surface_origins_drawn'] == metadata['surface_origins_sampled'] == 0
    assert metadata['normal_accepted_pair_count'] == metadata['proposal_pool_size'] == 0


def test_guided_seed_budget_and_complete_pose_cap_are_reproducible():
    vertices, faces = pin()
    region = pin_region(vertices, faces)
    options = dict(sampling_contact_region=region, seed=23, surface_budget=40,
                   orientation_samples=8, max_poses=100)
    a, ma = sampler.generate_candidates(vertices, faces, hand(), **options)
    b, mb = sampler.generate_candidates(vertices, faces, hand(), **options)
    assert a == b and ma == mb
    assert len(a) == 100
    assert ma['surface_origins_drawn'] == 40
    assert not ma['rejection_resampling']
    assert not ma['centroid_sort'] and not ma['pair_reduction']
    assert ma['hand_pose_reference'] == a[0]['hand_pose_reference']
    with pytest.raises(ValueError, match='max_poses'):
        sampler.generate_candidates(vertices, faces, hand(), sampling_contact_region=region, max_poses=101)


def test_mesh_binding_rejects_region_reuse_after_geometry_or_face_order_changes():
    vertices, faces = pin()
    region = pin_region(vertices, faces)
    moved = vertices.copy()
    moved[0, 0] += .001
    for v, f in [(moved, faces), (vertices, faces[::-1])]:
        with pytest.raises(ValueError, match='complete CAD mesh'):
            sampler.generate_candidates(v, f, hand(), sampling_contact_region=region)


@pytest.mark.parametrize('site', ['origin', 'second_endpoint'])
@pytest.mark.parametrize('bad_result', ['nan', 'numeric', 'wrong_shape'])
def test_unknown_nonboolean_or_wrong_shape_endpoint_flags_are_rejected(site, bad_result):
    vertices, faces = pin()
    region = pin_region(vertices, faces)
    valid_query = region['endpoint_contains']
    calls = 0
    def query(points, face_ids):
        nonlocal calls
        calls += 1
        if site == 'second_endpoint' and calls == 1:
            return valid_query(points, face_ids)
        if bad_result == 'nan':
            return np.full(len(points), np.nan)
        if bad_result == 'numeric':
            return np.ones(len(points), dtype=float)
        return np.ones(len(points)+1, dtype=bool)
    region['endpoint_contains'] = query
    with pytest.raises(ValueError, match='explicit boolean'):
        sampler.generate_candidates(vertices, faces, hand(), sampling_contact_region=region,
                                     surface_budget=8, max_poses=2)
    assert calls == (1 if site == 'origin' else 2)


def test_pipeline_reports_unknown_endpoint_flags_as_unavailable():
    request = request_fixture()
    region = compile_sampling_region(request['vertices_cad_m'], request['faces'],
        dict(type='face_ids', protected_face_ids=[]))
    region['endpoint_contains'] = lambda points, faces: np.full(len(points), np.nan)
    request['sampling_contact_region'] = region
    result = pipeline.select_grasp(request, surface_budget=8, max_poses=2)
    assert result['status'] == 'unavailable_inputs'
    assert result['proposed_count'] == 0 and result['selected_grasp'] is None
    assert 'explicit boolean' in str(result['missing_or_invalid_inputs'])


def test_pipeline_requires_task_labels_before_sampling_and_explicit_legacy_mode(monkeypatch):
    request = request_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing task labels must fail before sampling')
    monkeypatch.setattr(pipeline, 'generate_candidates', forbidden)
    result = pipeline.select_grasp(request)
    assert result['status'] == 'unavailable_inputs'
    assert result['proposed_count'] == 0 and result['selected_grasp'] is None
    assert 'sampling' in str(result['missing_or_invalid_inputs'])
    monkeypatch.undo()
    request['sampling_mode'] = 'legacy_unrestricted'
    result = pipeline.select_grasp(request, max_poses=0, surface_budget=3)
    assert result['status'] == 'no_valid_grasp_within_budget', result
    assert result['sampling']['sampling_mode'] == 'legacy_unrestricted'


def test_json_cli_compiles_declarative_region_without_serializing_callables(tmp_path, monkeypatch):
    request = request_fixture()
    request.pop('protected_contact_check')
    request['protected_contact_region'] = dict(type='halfspace', normal_cad=[0, 0, 1], offset_m=1.)
    path, out = tmp_path/'request.json', tmp_path/'result.json'
    path.write_text(json.dumps(request, default=lambda value: value.tolist()), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['pipeline', str(path), '--output', str(out)])
    pipeline.main()
    result = json.loads(out.read_text(encoding='utf-8'))
    assert result['status'] == 'no_valid_grasp_within_budget', result
    assert result['sampling']['sampling_mode'] == 'assembly_region_guided'
    assert result['sampling']['eligible_surface_area_m2'] == 0
    assert result['sampling']['full_CAD_face_count'] == len(request['faces'])
