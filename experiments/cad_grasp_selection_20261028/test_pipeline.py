"""Offline integration of rays, saved calibration, geometry and one selection."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.cad_grasp_selection_20261028 import pipeline
from experiments.cad_grasp_selection_20261028.geometry import pad_calibration
from experiments.cad_grasp_selection_20261028.sampler import generate_candidates, pose_from_pair
from experiments.cad_grasp_selection_20261028.test_geometry import box, fixture


ROOT = Path(__file__).resolve().parents[2]


def request_fixture():
    old, _ = fixture()
    return dict(vertices_cad_m=old['vertices_cad'], faces=old['faces'],
                hand=old['hand'], world_T_cad_pickup=old['pickup_T'],
                world_T_cad_final=old['final_T'], table_plane_world=old['table_plane'],
                protected_contact_check=old['protected_contact_check'],
                receivers=old['receivers'], axis_point_cad_m=[0., 0., 0.],
                axis_direction_cad=[0., 0., 1.])


def test_saved_pad_center_calibration_reconstructs_arbitrary_pose_without_depth():
    hand = json.loads((ROOT / 'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
    calibration = pad_calibration(hand)
    assert calibration['gap_zero_m'] == pytest.approx(-.001)
    assert calibration['gap_per_opening'] == pytest.approx(1.)
    rotation = Rotation.from_euler('xyz', [.7, -.5, 1.2]).as_matrix()
    midpoint = np.array([.03, -.02, .05])
    pair = midpoint + np.array([-.008, .008])[:, None] * rotation[:, 0]
    proposal = pose_from_pair(pair, rotation[:, 2], hand)
    assert proposal['holding_opening_m'] == pytest.approx(.017)
    np.testing.assert_allclose(proposal['reconstructed_pad_centers_cad_m'], pair, atol=1e-12)
    np.testing.assert_allclose(np.asarray(proposal['cad_T_hand'])[:3, :3], rotation, atol=1e-12)
    assert not np.allclose(np.asarray(proposal['cad_T_hand'])[:3, 3], midpoint)
    assert not any('depth' in name for name in proposal)


def test_cap_is_complete_poses_before_validation_and_full_circle_is_allowed():
    request = request_fixture()
    proposals, metadata = generate_candidates(request['vertices_cad_m'], request['faces'],
        request['hand'], seed=71, surface_budget=80, orientation_samples=12, max_poses=100)
    assert len(proposals) == metadata['proposed_count'] == 100
    assert metadata['proposal_pool_size'] > 100
    assert not metadata['pair_reduction'] and not metadata['centroid_sort']
    assert not metadata['independent_depth_sampling'] and not metadata['downward_gate']
    approaches = np.array([record['approach_cad'] for record in proposals])
    assert approaches[:, 2].min() < -.3 and approaches[:, 2].max() > .3
    again, _ = generate_candidates(request['vertices_cad_m'], request['faces'],
        request['hand'], seed=71, surface_budget=80, orientation_samples=12, max_poses=100)
    assert proposals == again
    with pytest.raises(ValueError, match='max_poses'):
        generate_candidates(request['vertices_cad_m'], request['faces'], request['hand'], max_poses=101)


def test_preopening_clearance_uses_actual_pad_gap_calibration():
    request = request_fixture()
    for component in request['hand']['components']:
        component['translation_per_opening'] = (
            2 * np.asarray(component['translation_per_opening'])).tolist()
    proposals, _ = generate_candidates(request['vertices_cad_m'], request['faces'],
        request['hand'], max_poses=3, pre_opening_clearance_m=.001)
    for proposal in proposals:
        assert proposal['pre_opening_m'] - proposal['holding_opening_m'] == pytest.approx(.001)


def test_one_selected_output_with_correct_pickup_goal_frames_and_separate_diagnostics():
    request = request_fixture()
    pickup, final = np.eye(4), np.eye(4)
    pickup[:3, :3] = Rotation.from_euler('z', .37).as_matrix()
    pickup[:3, 3] = [.2, .1, .4]
    final[:3, :3] = Rotation.from_euler('y', .2).as_matrix()
    final[:3, 3] = [.6, .2, .5]
    request.update(world_T_cad_pickup=pickup, world_T_cad_final=final)
    debug = []
    result = pipeline.select_grasp(request, seed=17, max_poses=37,
                                   surface_budget=40, orientation_samples=8, diagnostics=debug)
    assert result['status'] == 'selected', result
    assert result['proposed_count'] == len(debug) == 37
    assert result['valid_count'] > 0
    assert 'diagnostics' not in result and 'candidates' not in result
    selected = result['selected_grasp']
    np.testing.assert_allclose(selected['world_T_hand_pickup'], pickup @ selected['cad_T_hand'])
    np.testing.assert_allclose(selected['world_T_hand_final'], final @ selected['cad_T_hand'])
    np.testing.assert_allclose(selected['approach_world'],
                               pickup[:3, :3] @ selected['approach_cad'])
    assert selected['validity']['valid']
    assert [stage['criterion'] for stage in result['selection_stages']] == [
        'r_tau', 'r_surface', 'theta_down_rad']
    assert not result['robot_or_physics_execution']


def test_missing_required_axis_or_geometry_never_samples_or_falls_back(monkeypatch):
    request = request_fixture()
    request['axis_point_cad_m'] = None
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing task input must be detected before sampling')
    monkeypatch.setattr(pipeline, 'generate_candidates', forbidden)
    result = pipeline.select_grasp(request)
    assert result['status'] == 'unavailable_inputs'
    assert result['selected_grasp'] is None and result['proposed_count'] == 0
    assert 'axis_point_cad_m' in result['missing_or_invalid_inputs']
    request = request_fixture()
    request['faces'] = request['faces'][:-1]
    result = pipeline.select_grasp(request)
    assert result['status'] == 'unavailable_inputs' and result['proposed_count'] == 0


def test_empty_budget_has_explicit_failure_without_feasibility_claim():
    result = pipeline.select_grasp(request_fixture(), max_poses=0)
    assert result['status'] == 'no_valid_grasp_within_budget'
    assert result['message'] == 'No valid grasp found within the sampling budget.'
    assert result['selected_grasp'] is None and result['valid_count'] == 0


def test_unavailable_protected_geometry_is_detected_even_with_empty_bank():
    request = request_fixture()
    request['protected_contact_check'] = lambda points, **kwargs: dict(
        status='unavailable_zero_engagement', eligible=np.ones(len(points), bool))
    result = pipeline.select_grasp(request, max_poses=0)
    assert result['status'] == 'unavailable_inputs'
    assert result['selected_grasp'] is None


def test_all_invalid_openings_do_not_produce_fallback():
    result = pipeline.select_grasp(request_fixture(), max_poses=12,
                                   pre_opening_clearance_m=.1)
    assert result['status'] == 'no_valid_grasp_within_budget'
    assert result['proposed_count'] == 12 and result['valid_count'] == 0
    assert result['selected_grasp'] is None


def test_explicit_protected_triangle_mask_classifies_whole_contact_polygons():
    vertices, faces = box([-.01]*3, [.01]*3)
    triangle = vertices[faces[0]]
    check = pipeline.protected_triangle_check(vertices, faces, [0])
    output = check([triangle.mean(axis=0), [1., 1., 1.]],
                   contact_polygons=[triangle], contact_face_ids=[0])
    assert output['eligible'].tolist() == [False, True]
    assert output['polygon_eligible'].tolist() == [False]
    empty = check(np.empty((0, 3)), contact_polygons=[], contact_face_ids=[])
    assert empty['eligible'].shape == empty['polygon_eligible'].shape == (0,)


def test_malformed_json_mask_returns_unavailable_file_not_traceback(tmp_path):
    request = request_fixture()
    request.pop('protected_contact_check')
    request['protected_face_indices'] = [99999]
    def serial(value):
        return value.tolist() if isinstance(value, np.ndarray) else value
    source, output = tmp_path/'request.json', tmp_path/'selection.json'
    source.write_text(json.dumps(request, default=serial), encoding='utf-8')
    completed = subprocess.run([sys.executable, '-B', '-m',
        'experiments.cad_grasp_selection_20261028.pipeline', str(source), '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result['status'] == 'unavailable_inputs' and result['selected_grasp'] is None


def test_json_entry_point_selects_one_and_keeps_debug_separate(tmp_path):
    request = request_fixture()
    hand = request.pop('hand')
    request.pop('protected_contact_check')
    request.update(hand_model_file='hand.json', protected_face_indices=[])
    def serial(value):
        return value.tolist() if isinstance(value, np.ndarray) else value
    source = tmp_path/'request.json'
    output, debug = tmp_path/'selection.json', tmp_path/'debug.json'
    (tmp_path/'hand.json').write_text(json.dumps(hand), encoding='utf-8')
    source.write_text(json.dumps(request, default=serial), encoding='utf-8')
    completed = subprocess.run([sys.executable, '-B', '-m',
        'experiments.cad_grasp_selection_20261028.pipeline', str(source),
        '--output', str(output), '--diagnostics', str(debug)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result, records = json.loads(output.read_text()), json.loads(debug.read_text())
    assert result['status'] == 'selected', result
    assert result['selected_grasp'] is not None
    assert len(records) == result['proposed_count'] == 100
    assert 'diagnostics' not in result
