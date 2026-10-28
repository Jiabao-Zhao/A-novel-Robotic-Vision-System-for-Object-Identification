"""Offline CAD-feature and complete protected-polygon regressions."""
from pathlib import Path
import json

import numpy as np
import pytest

from experiments.fabrica_original_20261027.gallery import read_stl, local_path
from experiments.fabrica_goal_geometry_20261027.task_inputs import geometry_spec
from experiments.cad_grasp_selection_20261028.task_geometry import (
    derive_task, halfspace_checker, _bore_lip_centers,
)


ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT/'outputs/simulation/nist_task_board_1/assets/meshes'


def cad(part):
    if part == 'USB_Male':
        # The eight-scene study uses its preserved cable model, not the rotated
        # catalog-only connector. Verify the actual supplied native CAD frame.
        rows = json.loads((ROOT/'outputs/cad_grasp_geometry_20261026/inputs.json').read_text())['records']
        row = next(row for row in rows if row['part'] == part)
        return read_stl(local_path(row['mesh_path']))
    return read_stl(MESHES/(part+'_m.stl'))


@pytest.mark.parametrize('part', ['KET8_Square_8mm', 'KET12_Square_12mm',
    'KET16_Square_16mm', 'RGOCG8-50_8mm', 'RGOCG12-50_12mm', 'RGOCG16-50_16mm'])
def test_pin_axis_and_partial_face_region_use_cad_and_unchanged_nominal_goal(part):
    vertices, faces = cad(part)
    before = vertices.copy()
    goal_before = geometry_spec(part)['goal_T_cad_board'].copy()
    result = derive_task(part, vertices, faces)
    assert result['status'].startswith('available'), result
    np.testing.assert_allclose(result['axis_direction_cad'], [0, 0, -1])
    np.testing.assert_allclose(result['axis_point_cad_m'], [0, 0, 0], atol=1e-8)
    check = result['protected_contact_check']
    x = vertices[:, 0].max()
    points = np.array([[x, 0., -.024], [x, 0., .020]])
    crossing = np.array([[x, -.001, -.024], [x, .001, .020], [x, 0, .020]])
    clear = crossing.copy()
    clear[:, 2] = .02
    output = check(points, contact_polygons=[crossing, clear], contact_face_ids=[0, 0])
    assert output['eligible'].tolist() == [False, True]
    assert output['polygon_eligible'].tolist() == [False, True]
    np.testing.assert_array_equal(vertices, before)
    np.testing.assert_array_equal(result['task_spec']['goal_T_cad_board'], goal_before)


@pytest.mark.parametrize('part, direction, plane', [
    ('BNC_Male', [0, 0, -1], -.002495),
    ('DSUB_Male', [0, 0, 1], .0152),
    ('Waterproof_Male', [0, 0, 1], .0018),
    ('USB_Male', [-1, 0, 0], -.012785),
])
def test_connector_mating_feature_is_not_inferred_from_lower_bounding_plane(part, direction, plane):
    vertices, faces = cad(part)
    result = derive_task(part, vertices, faces)
    assert result['status'].startswith('available'), result
    np.testing.assert_allclose(result['axis_direction_cad'], direction)
    axis = int(np.argmax(np.abs(direction)))
    assert result['axis_point_cad_m'][axis] == pytest.approx(plane, abs=2e-8)
    assert not result['evidence']['protected_region']['actual_engagement_depth_inferred']
    assert 'zero assigned engagement' in result['evidence']['goal_warning']
    checker = result['protected_contact_check']
    center = np.asarray(result['axis_point_cad_m'])
    toward, away = center+.001*np.asarray(direction), center-.001*np.asarray(direction)
    output = checker([toward, away], contact_polygons=[np.tile(toward, (3, 1)),
                       np.tile(away, (3, 1))], contact_face_ids=[0, 0])
    assert output['eligible'].tolist() == [False, True]
    assert output['polygon_eligible'].tolist() == [False, True]
    empty = checker(np.empty((0, 3)), contact_polygons=[], contact_face_ids=[])
    assert empty['eligible'].shape == empty['polygon_eligible'].shape == (0,)


def test_threaded_nut_axis_comes_from_bore_mouths_not_a_helical_midsection():
    vertices, faces = cad('M16_Hex_Nut')
    records = _bore_lip_centers(vertices, faces, vertices[faces], 2, np.zeros(3))
    assert len(records) == 2
    for record in records:
        np.testing.assert_allclose(record['center_transverse_m'], [0, 0], atol=1e-7)
        assert record['radius_m'] == pytest.approx(.008, abs=1e-7)
        assert record['rms_m'] < 1e-7


def test_halfspace_checks_whole_polygon_and_reports_explicit_available_empty_query():
    checker = halfspace_checker([0, 0, 1], 0., label='test_region')
    polygon = np.array([[0, 0, -.001], [1, 0, -.001], [0, 1, .001]])
    output = checker([[0, 0, -.001]], contact_polygons=[polygon], contact_face_ids=[0])
    assert output['eligible'].all() and not output['polygon_eligible'].any()
    empty = checker(np.empty((0, 3)), contact_polygons=[], contact_face_ids=[])
    assert empty['status'].startswith('available')


def test_unknown_part_is_unavailable_and_has_no_permissive_checker():
    vertices, faces = cad('KET8_Square_8mm')
    result = derive_task('not_a_NIST_part', vertices, faces)
    assert result['status'].startswith('unavailable')
    assert result['protected_contact_check'] is None
