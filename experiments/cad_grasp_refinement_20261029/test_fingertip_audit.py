"""Static vertex-height tests; no candidate bank or execution."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import fixture
from .fingertip_audit import audit_case, audit_saved_hand, INPUTS, HAND, REGISTRY


def test_static_exposed_axis_bound_is_exact_affine_translation():
    kwargs, _ = fixture()
    hand = deepcopy(kwargs['hand'])
    # Tilted pin axis with a vertical component, while the hand still approaches
    # straight down. The whole unchanged hand translates rigidly along the axis.
    theta = np.deg2rad(30.)
    pickup = np.eye(4)
    pickup[:3, :3] = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    pickup[2, 3] = .03
    row = dict(world_T_cad=pickup.tolist(), table_plane_world=[0, 0, 1, 0])
    result = audit_case(row, hand, exposed_axial_interval_m=(-.025, .005))
    np.testing.assert_allclose(result['approach_world'], [0, 0, -1])
    assert result['optimistic_best_axial_location_m'] == .005
    assert result['optimistic_best_complete_hand_clearance_m'] == pytest.approx(
        result['declared_pose_complete_hand_minimum_clearance_m']+.005*np.cos(theta))
    assert hand == kwargs['hand']
    assert sum(r['vertices_evaluated'] for r in result['components']) == 2*sum(
        len(component['vertices_zero_m']) for component in hand['components'])


def test_saved_8mm_audit_measures_complete_fingers_and_preserves_sources():
    paths = [INPUTS, HAND, REGISTRY]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    result = audit_saved_hand()
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
    assert not result['candidate_generation_run']
    assert not result['geometry_modified']
    assert not result['simulator_object_poses_read']
    assert result['independent_depth_offsets_m'] == []
    assert len(result['records']) == 8
    assert result['pin_tip_depth_target_m'] == .02
    assert result['full_hand_distal_extent_beyond_pad_tip_m'] == pytest.approx(.004849034365359544)
    for row in result['records']:
        assert len(row['components']) == 5
        assert row['limiting_component'] in ('gripper0_finger1_collision', 'gripper0_finger2_collision')
        if row['scene'] == 'N05':
            assert row['declared_pose_table_clear']
            assert row['declared_pose_complete_hand_minimum_clearance_m'] > .019
        else:
            assert not row['declared_pose_table_clear']
            assert not row['optimistic_bound_can_clear_table']
            assert -.00111 < row['optimistic_best_complete_hand_clearance_m'] < -.00109
