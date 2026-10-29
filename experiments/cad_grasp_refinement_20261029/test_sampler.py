"""Bounded synthetic generation checks for the refined contact construction."""
from copy import deepcopy
import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import box, fixture
from experiments.cad_grasp_selection_20261029 import sampler as previous
from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region
from . import sampler
from .test_contact_model import cylinder, model_for, region_for


def hand(): return deepcopy(fixture()[0]['hand'])


def test_on_surface_rays_recover_thin_slab_skipped_by_fixed_half_mm_offset():
    vertices, faces = box([-.0001,-.01,-.01],[.0001,.01,.01])
    triangles = vertices[faces]
    ids = np.flatnonzero(abs(triangles.mean(1)[:,0]) > .000099)
    region = compile_sampling_region(vertices, faces, dict(type='face_ids', face_ids=[]),
        sampling_region=dict(type='face_ids', face_ids=ids.tolist()))
    old, _ = previous.generate_candidates(vertices, faces, hand(),
        sampling_contact_region=region, surface_budget=20)
    new, metadata = sampler.generate_candidates(vertices, faces, hand(),
        sampling_contact_region=region, surface_budget=20)
    assert not old and new
    assert metadata['normal_accepted_pair_count'] == 20
    assert metadata['inward_origin_offset_m'] == 0
    for candidate in new:
        assert np.linalg.norm(np.diff(candidate['contacts_cad_m'],axis=0)) == pytest.approx(.0002)


def test_verified_round_contacts_share_strict_policy_and_keep_full_circle_seed_and_cap():
    vertices, faces = cylinder(n=72)
    model, region = model_for(vertices, faces), region_for(vertices, faces)
    kwargs = dict(sampling_contact_region=region, contact_model=model, seed=19,
                  surface_budget=60, orientation_samples=10, max_poses=100)
    before_v, before_f = vertices.tobytes(), faces.tobytes()
    candidates, metadata = sampler.generate_candidates(vertices, faces, hand(), **kwargs)
    again, _ = sampler.generate_candidates(vertices, faces, hand(), **kwargs)
    assert candidates == again and len(candidates) == 100
    assert metadata['cylindrical_origins_snapped_to_source_edges'] == 60
    assert metadata['opposing_normal_cosine'] == pytest.approx(np.cos(np.deg2rad(1)))
    assert metadata['orientation_range_rad'] == [0., 2*np.pi]
    assert not metadata['pair_reduction'] and not metadata['independent_depth_sampling']
    assert not metadata['downward_gate']
    for c in candidates:
        pair = np.asarray(c['contacts_cad_m']); axis = np.diff(pair,axis=0)[0]; axis /= np.linalg.norm(axis)
        norms = model.contact_normals(pair, np.asarray(c['contact_face_ids']))
        assert norms[0]@(-axis) >= np.cos(np.deg2rad(1))
        assert norms[1]@axis >= np.cos(np.deg2rad(1))
        assert region['endpoint_contains'](pair, np.asarray(c['contact_face_ids'])).all()
        assert c['hand_pose_reference'] == previous.FINGERTIP_FRAME
    assert vertices.tobytes() == before_v and faces.tobytes() == before_f


def test_flats_are_not_relaxed_and_contact_model_must_match_full_mesh():
    vertices, faces = box([-.01,-.01,-.01],[.01,.01,.01])
    _, metadata = sampler.generate_candidates(vertices, faces, hand(),
        sampling_mode='legacy_unrestricted', opposing_normal_cosine=.1)
    assert metadata['opposing_normal_cosine'] == pytest.approx(np.cos(np.deg2rad(1)))
    cv, cf = cylinder()
    with pytest.raises(ValueError, match='different'):
        sampler.generate_candidates(vertices, faces, hand(), sampling_mode='legacy_unrestricted',
            contact_model=model_for(cv,cf))


def test_three_degree_skew_flats_no_longer_enter_one_degree_validation_bank():
    vertices, faces = box([-.01,-.01,-.01],[.01,.01,.01])
    original = vertices.copy()
    ids = np.flatnonzero(abs(original[faces].mean(1)[:,0]) > .0099)
    positive = vertices[:,0] > 0
    vertices[positive,0] += np.tan(np.deg2rad(3))*vertices[positive,1]
    region = compile_sampling_region(vertices, faces, dict(type='face_ids', face_ids=[]),
        sampling_region=dict(type='face_ids', face_ids=ids.tolist()))
    old, _ = previous.generate_candidates(vertices,faces,hand(),
        sampling_contact_region=region,surface_budget=25)
    new, metadata = sampler.generate_candidates(vertices,faces,hand(),
        sampling_contact_region=region,surface_budget=25)
    assert old and not new
    assert metadata['normal_rejected_pair_count'] == 25


def test_full_mesh_rays_still_include_bore_intersections(monkeypatch):
    from experiments.cad_grasp_selection_20261029.test_guided_sampler import tube
    vertices, faces, inner = tube()
    triangles = vertices[faces]
    ids = np.flatnonzero(abs(triangles.mean(1)[:,0]) > .0199)
    region = compile_sampling_region(vertices, faces,
        dict(type='face_ids', face_ids=inner), sampling_region=dict(type='intersection', regions=[
            dict(type='face_ids', face_ids=ids.tolist()),
            dict(type='halfspace',normal_cad=[0,1,0],offset_m=.004),
            dict(type='halfspace',normal_cad=[0,-1,0],offset_m=.004)]))
    observed = {}; original = sampler.legacy._all_ray_hits
    def track(origins, directions, complete):
        np.testing.assert_array_equal(complete, triangles)
        result = original(origins,directions,complete); observed['ids'] = result[2]
        return result
    monkeypatch.setattr(sampler.legacy,'_all_ray_hits',track)
    candidates, metadata = sampler.generate_candidates(vertices, faces, hand(),
        sampling_contact_region=region,surface_budget=20)
    assert np.isin(observed['ids'],inner).any()
    assert metadata['second_endpoint_region_rejections'] == 40
    assert candidates
