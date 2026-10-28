"""Saved-data gallery checks; no CAD selection or simulated motion runs."""
import copy

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.gallery import (
    render_sheets, selected_pose, validate_records,
)


def records():
    selected = dict(candidate_id=7, valid=True, validity=dict(status='valid'),
        contacts_cad_m=[[-.01, 0., 0.], [.01, 0., 0.]], cad_T_hand=np.eye(4).tolist(),
        world_T_hand_pickup=np.eye(4).tolist(), approach_world=[0., 0., -1.],
        holding_opening_m=.021, pre_opening_m=.023,
        criteria=dict(q_tau_m=0., q_surface_m=0., r_tau=0., r_surface=0.,
                      theta_down_rad=0.))
    return [dict(scene=f'N{scene:02d}', part=f'Part_{part:02d}', mesh_path='unused.stl',
                 world_T_cad=np.eye(4).tolist(), selection=dict(status='selected',
                 selected_grasp=copy.deepcopy(selected), valid_count=2, proposed_count=100))
            for scene in range(8) for part in range(12)]


def test_all_placements_validate_and_failure_never_draws_fallback():
    data = records()
    scenes, parts = validate_records(data)
    assert len(scenes) == 8 and len(parts) == 12
    failed = copy.deepcopy(data[0])
    failed['selection']['status'] = 'no_valid_grasp_within_budget'
    with pytest.raises(ValueError, match='fallback'):
        selected_pose(failed)
    failed['selection']['selected_grasp'] = None
    assert selected_pose(failed) is None


def test_invalid_selected_pose_or_mismatched_frame_is_rejected():
    data = records()
    data[0]['selection']['selected_grasp']['validity']['status'] = 'invalid'
    with pytest.raises(ValueError, match='geometry'):
        validate_records(data)
    data = records()
    data[0]['world_T_cad'][0][3] = .1
    with pytest.raises(ValueError, match='rendered CAD frame'):
        validate_records(data)


def test_missing_observed_pose_is_only_allowed_without_selected_grasp():
    data = records()
    data[0]['world_T_cad'] = None
    with pytest.raises(ValueError, match='without its estimated CAD pose'):
        validate_records(data)
    data[0]['selection'].update(status='unavailable_inputs', selected_grasp=None)
    validate_records(data)


def test_one_scene_atlas_draws_one_pair_per_selected_part(tmp_path):
    vertices = np.array([[-.01, -.02, -.003], [.01, -.02, -.003],
                         [.01, .02, -.003], [-.01, .02, -.003]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    data = records()
    data[0]['selection'].update(status='no_valid_grasp_within_budget', selected_grasp=None)
    parts = sorted({record['part'] for record in data})
    meshes = {part: (vertices, faces) for part in parts}
    output = render_sheets(data, meshes, ['N00'], parts, tmp_path)
    assert output[0]['selected_count'] == 11
    assert (tmp_path / 'N00.png').is_file()
    assert (tmp_path / 'N00_preview.png').is_file()
    assert sum(tile['selected_contact_overlay_px'] is not None
               for tile in output[0]['tiles']) == 11
    assert output[0]['tiles'][0]['selected_contact_overlay_px'] is None
