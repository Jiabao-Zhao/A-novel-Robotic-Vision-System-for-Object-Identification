"""Corrected task input wiring only; never sample or select a NIST grasp."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import fixture, box
from .task_inputs import build_request, protected_checker


def registry_fixture(tmp_path):
    vertices, faces = box([-.004, -.004, -.025], [.004, .004, .025])
    part_path = tmp_path/'pin.npz'
    np.savez(part_path, vertices=vertices, faces=faces)
    receiver_path = tmp_path/'receiver.npz'
    rv, rf = box([-.1, -.1, -.0044958], [.1, .1, .0044958])
    np.savez(receiver_path, vertices=rv, faces=rf)
    final = np.diag([1., -1., -1., 1.])
    final[:3, 3] = [.03, -.02, .0094958]
    receiver_pose = np.eye(4)
    receiver_pose[:3, 3] = [.01, .02, 0]
    row = dict(part='RGOCG8-50_8mm', part_mesh=str(part_path),
        part_mesh_sha256=hashlib.sha256(part_path.read_bytes()).hexdigest(),
        board_T_part=final.tolist(), world_T_cad_final=final.tolist(),
        axis_point_cad_m=[0, 0, 0], axis_direction_cad=[0, 0, 1],
        axis_direction_board=[0, 0, -1], pin_tip_depth_below_board_top_m=.020,
        protected_contact_region=dict(type='halfspace', normal_cad=[0, 0, 1], offset_m=-.005,
                                      inequality='normal@point+offset>=0'),
        sampling_region=dict(type='face_ids', face_ids=list(range(len(faces))),
                             provenance='explicit synthetic surface annotation'),
        receivers=[dict(name='finished_board', mesh_path=str(receiver_path),
            mesh_sha256=hashlib.sha256(receiver_path.read_bytes()).hexdigest(),
            board_T_mesh=receiver_pose.tolist())])
    registry = dict(units='metres', world_T_board=np.eye(4).tolist(),
        board_top_board_m=.0044958, pin_tip_depth_below_board_top_m=.020, parts=[row])
    hand = fixture()[0]['hand']
    return registry, hand


def test_20mm_pin_goal_and_protected_threshold_are_wired(tmp_path):
    registry, hand = registry_fixture(tmp_path)
    result = build_request('RGOCG8-50_8mm', np.eye(4), [0, 0, 1, 0],
                           registry=registry, hand=hand)
    assert result['status'] == 'available', result
    request = result['request']
    tip = np.asarray(request['world_T_cad_final'])@np.array([0, 0, .025, 1])
    assert registry['board_top_board_m']-tip[2] == pytest.approx(.020)
    points = np.array([[0, 0, .0049], [0, 0, .005], [0, 0, .025]])
    crossing = np.array([[0, 0, .0049], [.001, 0, .0051], [0, .001, .0049]])
    check = request['protected_contact_check'](points, contact_polygons=[crossing], contact_face_ids=[0])
    assert check['eligible'].tolist() == [True, False, False]
    assert check['polygon_eligible'].tolist() == [False]
    assert result['source']['pin_tip_depth_below_board_top_m'] == .020
    assert result['source']['task_registry'] == 'explicit caller-supplied corrected registry'
    assert result['source']['no_candidate_generation']
    compiled = request['sampling_contact_region']
    assert request['sampling_mode'] == 'assembly_region_guided'
    assert compiled['provenance']['positive_region_provided']
    assert np.max(compiled['sampling_triangles_cad_m'][:, :, 2]) < .005
    assert compiled['eligible_area_m2'] < compiled['provenance']['original_surface_area_m2']


def test_board_override_composes_goal_and_receivers_without_mutating_inputs(tmp_path):
    registry, hand = registry_fixture(tmp_path)
    before = deepcopy(registry)
    board = np.array([[0, -1, 0, .3], [1, 0, 0, -.1], [0, 0, 1, .2], [0, 0, 0, 1.]])
    pickup = np.eye(4)
    pickup[:3, 3] = [-.1, .2, .03]
    result = build_request('RGOCG8-50_8mm', pickup, [0, 0, 1, 0],
                           world_T_board=board, registry=registry, hand=hand)
    assert result['status'] == 'available', result
    request = result['request']
    np.testing.assert_allclose(request['world_T_cad_pickup'], pickup)
    np.testing.assert_allclose(request['world_T_cad_final'], board@registry['parts'][0]['board_T_part'])
    np.testing.assert_allclose(request['receivers'][0]['world_T_mesh'], board@registry['parts'][0]['receivers'][0]['board_T_mesh'])
    assert registry == before
    assert request['simulator_object_poses_read'] is False


@pytest.mark.parametrize('fault', ['old_depth', 'old_region', 'axis', 'world_goal', 'hash', 'missing_region', 'missing_receiver', 'missing_part'])
def test_stale_mismatched_or_missing_task_inputs_are_unavailable(tmp_path, fault):
    registry, hand = registry_fixture(tmp_path)
    row = registry['parts'][0]
    if fault == 'old_depth':
        row['pin_tip_depth_below_board_top_m'] = .012
    elif fault == 'old_region':
        row['protected_contact_region']['offset_m'] = -.013
    elif fault == 'axis':
        row['axis_direction_board'] = [0, 0, 1]
    elif fault == 'world_goal':
        row['world_T_cad_final'][0][3] += .001
    elif fault == 'hash':
        row['part_mesh_sha256'] = 'incorrect_source_hash'
    elif fault == 'missing_region':
        row.pop('protected_contact_region')
    elif fault == 'missing_receiver':
        row['receivers'] = []
    elif fault == 'missing_part':
        registry['parts'] = []
    result = build_request('RGOCG8-50_8mm', np.eye(4), [0, 0, 1, 0], registry=registry, hand=hand)
    assert result['status'] == 'unavailable_inputs', result
    assert result['request'] is None
    assert result['message']


def test_missing_observation_is_not_replaced_with_identity(tmp_path):
    registry, hand = registry_fixture(tmp_path)
    result = build_request('RGOCG8-50_8mm', None, [0, 0, 1, 0], registry=registry, hand=hand)
    assert result['status'] == 'unavailable_inputs'
    assert result['request'] is None


def test_missing_positive_annotation_needs_explicit_protected_only_opt_in(tmp_path, monkeypatch):
    from . import sampling_annotations
    registry, hand = registry_fixture(tmp_path)
    registry['parts'][0].pop('sampling_region')
    monkeypatch.setattr(sampling_annotations, 'annotation_for', lambda *args: None)
    blocked = build_request('RGOCG8-50_8mm', np.eye(4), [0, 0, 1, 0], registry=registry, hand=hand)
    assert blocked['status'] == 'unavailable_inputs'
    assert 'positive' in blocked['message']
    explicit = build_request('RGOCG8-50_8mm', np.eye(4), [0, 0, 1, 0], registry=registry, hand=hand,
                             allow_protected_only_sampling=True)
    assert explicit['status'] == 'available'
    assert not explicit['request']['sampling_contact_region']['provenance']['positive_region_provided']


def test_protected_union_checks_complete_polygons(tmp_path):
    registry, _ = registry_fixture(tmp_path)
    region = dict(type='union', regions=[registry['parts'][0]['protected_contact_region'],
        dict(type='halfspace', normal_cad=[1, 0, 0], offset_m=-.003)])
    vertices, faces = box([-.004, -.004, -.025], [.004, .004, .025])
    check = protected_checker(region, vertices, faces)
    polygon = np.array([[.002, 0, 0], [.0035, 0, 0], [.002, .001, 0]])
    result = check(np.array([[0, 0, 0], [.0031, 0, 0], [0, 0, .006]]),
                   contact_polygons=[polygon], contact_face_ids=[0])
    assert result['eligible'].tolist() == [True, False, False]
    assert result['polygon_eligible'].tolist() == [False]
