"""Saved-data visualization checks, without grasp generation or execution."""
import copy
import json

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_gallery import records
from experiments.cad_grasp_selection_20261028.test_geometry import fixture
from experiments.cad_grasp_refinement_20261029.gallery import (
    CAPTION, HTML, failure_reason, pickup_geometry, render_sheets,
    render_side_sheets, transform_points, validate_records,
)


def pickup_record():
    row = records()[0]
    world = np.eye(4)
    world[:3, 3] = [.2, -.1, .3]
    row['world_T_cad'] = world.tolist()
    row['table_plane_world'] = [0., 0., 1., -.2]
    pose = row['selection']['selected_grasp']
    grip_site = np.eye(4)
    grip_site[2, 3] = .02
    fingertip = grip_site.copy()
    fingertip[2, 3] += .035
    pose['cad_T_grip_site'] = grip_site.tolist()
    pose['cad_T_hand'] = fingertip.tolist()
    pose['world_T_hand_pickup'] = (world@fingertip).tolist()
    return row


def test_complete_hand_uses_original_grip_site_not_fingertip_reference():
    kwargs, _ = fixture()
    row = pickup_record()
    geom = pickup_geometry(row, kwargs['vertices_cad'], kwargs['hand'])
    pose = row['selection']['selected_grasp']
    expected_transform = np.asarray(row['world_T_cad'])@np.asarray(pose['cad_T_grip_site'])
    assert np.allclose(geom['world_T_grip_site'], expected_transform)
    assert len(geom['components']) == len(kwargs['hand']['components'])
    for actual, component in zip(geom['components'], kwargs['hand']['components']):
        opened = (np.asarray(component['vertices_zero_m'])
                  + pose['holding_opening_m']*np.asarray(component['translation_per_opening']))
        assert np.allclose(actual['vertices_world_m'], transform_points(opened, expected_transform))
    expected_tip_height = (.3+.02+.035-.2)*1000
    assert geom['fingertip_projected_mm'][1] == pytest.approx(expected_tip_height)
    wrong = copy.deepcopy(row)
    del wrong['selection']['selected_grasp']['cad_T_grip_site']
    with pytest.raises(ValueError, match='original cad_T_grip_site'):
        pickup_geometry(wrong, kwargs['vertices_cad'], kwargs['hand'])


def test_side_vertical_is_distance_to_saved_table_not_world_z():
    kwargs, _ = fixture()
    row = pickup_record()
    row['table_plane_world'] = [0., 2., 2., -.2]
    geom = pickup_geometry(row, kwargs['vertices_cad'], kwargs['hand'])
    normal = np.array([0., 1., 1.])/np.sqrt(2)
    assert np.allclose(geom['vertical_world'], normal)
    assert geom['horizontal_world']@normal == pytest.approx(0., abs=1e-12)
    expected = geom['part_world_m']@normal - .2/np.sqrt(8)
    assert np.allclose(geom['part_projected_mm'][:, 1], expected*1000)


def test_saved_selected_fingertip_world_transform_is_still_checked():
    data = records()
    data[0] = pickup_record()
    validate_records(data)
    data[0]['selection']['selected_grasp']['world_T_hand_pickup'][2][3] += .035
    with pytest.raises(ValueError, match='rendered CAD frame'):
        validate_records(data)


def test_gallery_has_one_pair_and_complete_side_hand_per_selected_part(tmp_path):
    kwargs, _ = fixture()
    data = [pickup_record()]
    for part in range(1, 12):
        row = copy.deepcopy(data[0])
        row['part'] = f'Part_{part:02d}'
        data.append(row)
    data[0]['selection'].update(status='no_valid_grasp_within_budget', selected_grasp=None)
    data[0]['rejection_counts'] = {'Complete gripper violates table clearance during closing': 7}
    parts = [row['part'] for row in data]
    meshes = {part: (kwargs['vertices_cad'], kwargs['faces']) for part in parts}
    saved = copy.deepcopy(data)
    top = render_sheets(data, meshes, ['N00'], parts, tmp_path)
    hand = tmp_path/'hand.json'
    hand.write_text(json.dumps(kwargs['hand']), encoding='utf-8')
    side, _ = render_side_sheets(data, meshes, ['N00'], parts, tmp_path, hand)
    assert data == saved
    assert top[0]['selected_count'] == 11
    assert top[0]['tiles'][0]['selected_contact_overlay_px'] is None
    assert side[0]['rendered_count'] == 11
    assert all(tile['components_rendered'] == 3 for tile in side[0]['tiles'][1:])
    assert (tmp_path/'N00.png').is_file() and (tmp_path/'N00_pickup_side.png').is_file()
    assert failure_reason(data[0]) == 'A finger or the palm would touch the table'
    assert 'VLM selection is excluded' in CAPTION
    assert 'VLM selection is excluded' in HTML
    assert 'VLM selection pending' not in HTML
