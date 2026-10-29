"""Offline frame/contact/whole-hand checks; no NIST sampling experiment."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.geometry import (
    component_vertices, pad_calibration, transform_points,
)
from experiments.cad_grasp_selection_20261028.sampler import pose_from_pair as center_pose
from experiments.cad_grasp_selection_20261028.test_geometry import fixture, box
from .geometry import GeometryValidator
from .hand_frame import (
    FINGERTIP_FRAME, fingertip_calibration, hand_in_fingertip_frame,
    grip_site_pose_from_fingertip, fingertip_pose_from_grip_site,
)
from .sampler import pose_from_pair
from . import sampler
from . import pipeline

ROOT = Path(__file__).resolve().parents[2]


def tip_candidate(hand, *, approach=(0., 0., 1.), inset=.00025):
    candidate = pose_from_pair([[-.01, 0, 0], [.01, 0, 0]], approach,
                               hand, contact_inset_m=inset)
    return dict(candidate, pre_opening_m=.03, candidate_id=0)


def test_saved_hand_calibration_uses_distal_edges_and_preserves_frozen_inputs():
    path = ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'
    hand = json.loads(path.read_text())
    before = deepcopy(hand)
    c = fingertip_calibration(hand)
    np.testing.assert_allclose(c['usable_pad_tip_origin_grip_site_m'], [0, 0, .0044], atol=1e-9)
    assert c['fingertip_origin_grip_site_m'][2] == pytest.approx(.009249032, abs=1e-8)
    assert c['fingertip_origin_grip_site_m'][2] > c['usable_pad_tip_origin_grip_site_m'][2]
    np.testing.assert_allclose(c['contact_anchors_zero_grip_site_m'][:, 2], [.00415]*2, atol=1e-9)
    assert c['full_hand_distal_extent_beyond_pad_tip_m'] == pytest.approx(.004849032, abs=1e-8)
    assert hand == before
    launch = json.loads((ROOT/'outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/launch_manifest.json').read_text())
    for group in ['source_sha256', 'protected_sha256']:
        for name, digest in launch[group].items():
            if name.replace('\\', '/').startswith('experiments/cad_grasp_selection_20261028/') or name.endswith('hand_model.json'):
                assert hashlib.sha256((ROOT/name.replace('\\', '/')).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('saved_hand', [False, True])
def test_full_hand_world_vertices_are_invariant_under_frame_conversion(saved_hand):
    kwargs, _ = fixture()
    hand = (json.loads((ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
            if saved_hand else kwargs['hand'])
    c = fingertip_calibration(hand)
    exported = hand_in_fingertip_frame(hand)
    assert exported['frame'] == FINGERTIP_FRAME
    assert len(exported['components']) == len(hand['components'])
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    pose[:3, 3] = [.12, -.08, .2]
    tip_pose = fingertip_pose_from_grip_site(pose, hand)
    np.testing.assert_allclose(grip_site_pose_from_fingertip(tip_pose, hand), pose)
    for q in [0, .02, .08]:
        for old, new in zip(hand['components'], exported['components']):
            assert old['name'] == new['name']
            assert len(old['vertices_zero_m']) == len(new['vertices_zero_m'])
            np.testing.assert_allclose(
                transform_points(component_vertices(old, q), pose),
                transform_points(component_vertices(new, q), tip_pose), atol=1e-12)
    np.testing.assert_allclose(c['grip_site_T_fingertip']@c['fingertip_T_grip_site'], np.eye(4))


def test_explicit_unrestricted_mode_keeps_original_pair_orientation_and_preopening_bank():
    kwargs, _ = fixture()
    hand = kwargs['hand']
    options = dict(seed=3, max_poses=8, surface_budget=12, orientation_samples=3)
    old_bank, old_metadata = sampler.legacy.generate_candidates(
        kwargs['vertices_cad'], kwargs['faces'], hand, **options)
    bank, metadata = sampler.generate_candidates(kwargs['vertices_cad'], kwargs['faces'], hand,
        sampling_mode='legacy_unrestricted', **options)
    assert len(bank) == len(old_bank) > 0
    for current, old in zip(bank, old_bank):
        for key in ['candidate_id', 'pair_id', 'orientation_index', 'pre_opening_m',
                    'holding_opening_m', 'contacts_cad_m', 'approach_cad']:
            np.testing.assert_allclose(current[key], old[key], atol=1e-15, rtol=0.)
        assert 'reconstructed_pad_centers_cad_m' not in current
    assert metadata['normal_accepted_pair_count'] == old_metadata['normal_accepted_pair_count']
    assert metadata['hand_pose_reference'] == FINGERTIP_FRAME


def test_contact_placement_changes_from_centers_to_near_tip_not_just_pose_name():
    kwargs, _ = fixture()
    hand = kwargs['hand']
    old = center_pose([[-.01, 0, 0], [.01, 0, 0]], [0, 0, 1], hand)
    candidate = tip_candidate(hand)
    assert candidate['hand_pose_reference'] == FINGERTIP_FRAME
    np.testing.assert_allclose(candidate['reconstructed_contact_anchors_cad_m'], candidate['contacts_cad_m'])
    old_grip, new_grip = np.asarray(old['cad_T_hand']), np.asarray(candidate['cad_T_grip_site'])
    np.testing.assert_allclose(new_grip[:3, 3]-old_grip[:3, 3], [0, 0, -.00375])
    base = pad_calibration(hand)
    centers = base['centers_zero_m']+.02*base['centers_per_opening']
    assert not np.allclose(transform_points(centers, new_grip), candidate['contacts_cad_m'])
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['valid'], result
    assert 'contact_anchor_reconstruction_error_m' in result['checks']
    assert 'pad_center_reconstruction_error_m' not in result['checks']


def test_validator_rejects_mixed_reference_fields_and_unknown_frames():
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    candidate['cad_T_hand'][2][3] += .001
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'unavailable_input'
    assert 'disagree' in result['reasons'][0]
    candidate = tip_candidate(kwargs['hand'])
    candidate.pop('hand_pose_reference')
    assert GeometryValidator(**kwargs).validate(candidate)['status'] == 'unavailable_input'
    hand = deepcopy(kwargs['hand'])
    hand['frame'] = 'unverified hand offset'
    with pytest.raises(ValueError, match='calibration'):
        fingertip_calibration(hand)


def test_contact_inset_and_validator_margin_must_agree():
    kwargs, _ = fixture()
    validator = GeometryValidator(**kwargs, config={'pad_edge_margin_m': .001})
    assert validator.validate(tip_candidate(kwargs['hand']))['status'] == 'unavailable_input'
    result = validator.validate(tip_candidate(kwargs['hand'], inset=.001))
    assert result['valid'], result


def test_near_tip_pose_retains_whole_palm_table_and_receiver_rejection():
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    kwargs['table_plane'] = [0, 0, 1, .02]
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and any('table' in x for x in result['reasons'])
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    kwargs['hand']['components'][-1]['vertices_zero_m'] = box([-.003]*3, [.003]*3)[0].tolist()
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and any('palm' in x for x in result['reasons'])
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    receiver, faces = box([-.026, -.007, -.030], [.026, .007, -.021])
    kwargs['receivers'] = [dict(name='receiver', vertices=receiver, faces=faces, world_T_mesh=np.eye(4))]
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and any('receiver' in x for x in result['reasons'])


def test_current_pipeline_uses_tip_generator_and_outputs_both_command_frames(monkeypatch):
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    observed = {}
    def fixed_candidates(vertices, faces, hand, **options):
        observed.update(options)
        return [deepcopy(candidate)], {'test_fixture': 'one explicit candidate, no random sampling'}
    monkeypatch.setattr(pipeline, 'generate_candidates', fixed_candidates)
    request = dict(vertices_cad_m=kwargs['vertices_cad'], faces=kwargs['faces'],
        world_T_cad_pickup=kwargs['pickup_T'], world_T_cad_final=kwargs['final_T'],
        axis_point_cad_m=[0, 0, 0], axis_direction_cad=[0, 0, 1],
        table_plane_world=kwargs['table_plane'], hand=kwargs['hand'],
        sampling_mode='legacy_unrestricted',
        protected_contact_check=kwargs['protected_contact_check'], receivers=kwargs['receivers'])
    result = pipeline.select_grasp(request)
    assert result['status'] == 'selected', result
    assert observed['contact_inset_m'] == .00025
    assert result['hand_pose_reference'] == FINGERTIP_FRAME
    selected = result['selected_grasp']
    np.testing.assert_allclose(selected['world_T_hand_pickup'], candidate['cad_T_hand'])
    np.testing.assert_allclose(selected['world_T_grip_site_pickup'], candidate['cad_T_grip_site'])
    request['simulator_object_poses_read'] = True
    assert pipeline.select_grasp(request)['status'] == 'unavailable_inputs'
