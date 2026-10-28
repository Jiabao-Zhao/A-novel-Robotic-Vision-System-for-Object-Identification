"""Focused offline checks for declared frames and mating-contact restrictions."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from experiments.fabrica_goal_geometry_20261027.task_inputs import (
    geometry_spec, evaluate_contact_pairs, MESHES,
)


def test_nominal_goals_and_receiver_scope_use_same_board_frame():
    pin = geometry_spec('RGOCG8-50_8mm')
    np.testing.assert_allclose(pin['goal_T_cad_board'][:2, 3],
                               [-.07555038592618402, .15517576492409133])
    assert pin['official_final_assembly_pose'] is False
    assert [r['name'] for r in pin['receivers']] == ['board']
    np.testing.assert_allclose(pin['receivers'][0]['board_T_mesh'][:2, 3], 0.)
    gear = geometry_spec('Gear_Large')
    assert [r['name'] for r in gear['receivers']] == [
        'board', 'fixture_0_Gear_Plate', 'fixture_1_Gear_Shaft']
    np.testing.assert_allclose(gear['goal_T_cad_board'][:2, 3],
                               gear['receivers'][2]['board_T_mesh'][:2, 3])
    nut = geometry_spec('M16_Hex_Nut')
    assert [r['name'] for r in nut['receivers']] == ['board', 'fixture_7_M16_Screw']
    dsub = geometry_spec('DSUB_Male')
    assert [r['name'] for r in dsub['receivers']] == [
        'board', 'fixture_12_DB_Housing', 'fixture_13_DSUB_Female']
    assert all(Path(r['mesh_path']).is_file() for r in dsub['receivers'])


@pytest.mark.parametrize('part', ['RGOCG8-50_8mm', 'KET16_Square_16mm'])
def test_only_inserted_pin_segment_is_reserved(part):
    spec = geometry_spec(part)
    contacts = np.array([
        [[-.004, 0., -.020], [.004, 0., -.020]],
        [[-.004, 0., -.010], [.004, 0., -.010]],
        [[-.004, 0., .020], [.004, 0., .020]],
        [[-.004, 0., -.013], [.004, 0., -.013]],
    ])
    result = evaluate_contact_pairs(None, contacts, spec)
    assert result['eligible'].tolist() == [False, True, True, False]
    assert result['reasons'][0] == 'reserved_inserted_pin_segment'


def test_connector_region_is_unavailable_not_fabricated():
    for part in ('BNC_Male', 'USB_Male', 'DSUB_Male', 'Waterproof_Male'):
        result = evaluate_contact_pairs(None, np.zeros((1, 2, 3)), geometry_spec(part))
        assert result['eligible'].tolist() == [True]
        assert result['status'] == 'unavailable_zero_engagement'
        assert result['reasons'] == ['contact_region_unavailable_zero_engagement']


@pytest.mark.parametrize('part', ['Gear_Large', 'M16_Hex_Nut'])
def test_exact_native_bore_and_bottom_are_reserved_but_top_is_clear(part):
    mesh = trimesh.load_mesh(MESHES / f'{part}_m.stl', process=False, maintain_order=True)
    centers = mesh.triangles_center
    normals = mesh.face_normals
    z = centers[:, 2]
    bottom = np.flatnonzero((abs(z - mesh.bounds[0, 2]) < 1e-8) & (normals[:, 2] < -.99))[0]
    top = np.flatnonzero((abs(z - mesh.bounds[1, 2]) < 1e-8) & (normals[:, 2] > .99))[0]
    radial = np.linalg.norm(centers[:, :2], axis=1)
    inward = np.flatnonzero(np.einsum('ij,ij->i', centers[:, :2], normals[:, :2]) < -1e-8)
    bore = inward[np.argmin(radial[inward])]
    points = centers[[bottom, top, bore]]
    contacts = np.repeat(points[:, None, :], 2, axis=1)
    result = evaluate_contact_pairs(mesh, contacts, geometry_spec(part))
    assert result['eligible'].tolist() == [False, True, False]
    assert result['reasons'][0] == 'reserved_bottom_seating_face'
    assert result['reasons'][2] == 'reserved_central_bore_surface'
    assert result['status'] == 'available_nominal_geometry'


def test_off_surface_gear_contact_is_unclassified_and_not_accepted():
    mesh = trimesh.load_mesh(MESHES / 'Gear_Large_m.stl', process=False, maintain_order=True)
    result = evaluate_contact_pairs(mesh, np.full((1, 2, 3), .5), geometry_spec('Gear_Large'))
    assert result['eligible'].tolist() == [False]
    assert result['reasons'] == ['unclassified_contact_geometry']
    assert result['status'] == 'available_nominal_geometry_with_unclassified_contacts'


def test_inward_facing_outer_gear_recess_is_not_the_central_bore():
    mesh = trimesh.load_mesh(MESHES / 'Gear_Large_m.stl', process=False, maintain_order=True)
    centers, normals = mesh.triangles_center, mesh.face_normals
    inward = np.flatnonzero(np.einsum('ij,ij->i', centers[:, :2], normals[:, :2]) < -1e-8)
    face = inward[np.argmax(np.linalg.norm(centers[inward, :2], axis=1))]
    contacts = np.repeat(centers[face][None, None, :], 2, axis=1)
    result = evaluate_contact_pairs(mesh, contacts, geometry_spec('Gear_Large'))
    assert result['eligible'].tolist() == [True]
    assert result['reasons'] == ['contact_region_clear']
