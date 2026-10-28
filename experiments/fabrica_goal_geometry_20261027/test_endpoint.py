"""Offline checks of the static final-pose geometry kernel; no motion executes."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = ROOT / 'outputs/cache/fabrica_native_20261027/python'
sys.path.insert(0, str(DEPENDENCIES))

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.fabrica_original_20261027 import native
from experiments.fabrica_goal_geometry_20261027.endpoint import EndpointChecker


def receiver_description(tmp_path, mesh, name):
    path = tmp_path / f'{name}.stl'
    mesh.export(str(path))
    return dict(name=name, mesh_path=str(path), board_T_mesh=np.eye(4))


def interior_box(mesh, trimesh):
    """Construct a small solid strictly inside the actual non-convex palm."""
    candidates = np.vstack([mesh.center_mass, mesh.centroid,
        mesh.triangles_center[:128] - .0005 * mesh.face_normals[:128]])
    candidates = candidates[mesh.contains(candidates)]
    assert len(candidates), 'Test fixture could not find palm interior'
    _, distances, _ = mesh.nearest.on_surface(candidates)
    index = int(np.argmax(distances))
    clearance = float(distances[index])
    assert clearance > 1e-5
    half_width = min(clearance / (4. * np.sqrt(3.)), .0002)
    box = trimesh.creation.box(extents=[2. * half_width] * 3)
    box.apply_translation(candidates[index])
    assert mesh.contains(box.vertices).all()
    return box


def test_real_fcl_distinguishes_clear_and_intersecting_receiver(tmp_path):
    up = native.load()
    empty = EndpointChecker([])
    surface_point = empty.hand_meshes['panda_hand'].triangles[0].mean(axis=0)
    receiver = up.trimesh.creation.box(extents=[.012] * 3)
    receiver.apply_translation(surface_point)
    checker = EndpointChecker([receiver_description(tmp_path, receiver, 'receiver')])
    identity = np.eye(4)
    distant = np.eye(4)
    distant[:3, 3] = [1., .5, .3]

    accepted, reasons = checker.check([distant, identity], [.5, .5])
    assert accepted.tolist() == [True, False]
    assert reasons[0] == 'clear_at_nominal_goal'
    assert reasons[1] == 'hand_surface_intersects:receiver'
    assert checker.hand_manager.in_collision_other(checker.receivers[0]['collision'])
    checker.check([distant], [.5])
    assert not checker.hand_manager.in_collision_other(checker.receivers[0]['collision'])


def test_containment_rejects_hand_entirely_inside_large_receiver(tmp_path, monkeypatch):
    up = native.load()
    receiver = up.trimesh.creation.box(extents=[2.] * 3)
    checker = EndpointChecker([receiver_description(tmp_path, receiver, 'large_receiver')])

    accepted, reasons = checker.check([np.eye(4)], [.5])
    assert accepted.tolist() == [False]
    assert reasons[0].startswith(('hand_surface_intersects:large_receiver',
                                  'hand_inside_receiver:large_receiver:'))
    transforms = checker.component_transforms(np.eye(4), .5)
    for name, mesh in checker.hand_meshes.items():
        world = mesh.vertices @ transforms[name][:3, :3].T + transforms[name][:3, 3]
        assert np.all(np.abs(world) < 1.), 'Every hand vertex must be inside the box'

    # Installed FCL versions can already flag this nested configuration. Force
    # only its boundary-query result to false to exercise the containment guard.
    monkeypatch.setattr(checker.hand_manager, 'in_collision_other', lambda other: False)
    accepted, reasons = checker.check([np.eye(4)], [.5])
    assert accepted.tolist() == [False]
    assert reasons[0].startswith('hand_inside_receiver:large_receiver:')


def test_containment_rejects_receiver_entirely_inside_palm(tmp_path):
    up = native.load()
    empty = EndpointChecker([])
    receiver = interior_box(empty.hand_meshes['panda_hand'], up.trimesh)
    checker = EndpointChecker([receiver_description(tmp_path, receiver, 'tiny_receiver')])

    accepted, reasons = checker.check([np.eye(4)], [.5])
    assert not checker.hand_manager.in_collision_other(checker.receivers[0]['collision'])
    assert accepted.tolist() == [False]
    assert reasons[0] == 'receiver_inside_hand:tiny_receiver:panda_hand'


@pytest.mark.parametrize('opening', [0., .375, .95])
def test_metre_hand_vertices_match_original_native_transforms(opening):
    up = native.load()
    raw_cm = up.geometry['load_gripper_meshes'](
        'panda', str(native.VENDOR / 'assets'), has_ft_sensor=False, visual=False)
    checker = EndpointChecker([])
    rotation = Rotation.from_euler('xyz', [37., -21., 18.], degrees=True)
    board_T_hand = np.eye(4)
    board_T_hand[:3, :3] = rotation.as_matrix()
    board_T_hand[:3, 3] = [.17, -.08, .23]
    quat_wxyz = rotation.as_quat()[[3, 0, 1, 2]]

    # The independent upstream calculation retains its right-finger mirror
    # within the original transform and applies it to unmodified cm vertices.
    original_cm = up.geometry['get_gripper_meshes_transforms'](
        'panda', raw_cm, board_T_hand[:3, 3] * 100., quat_wxyz,
        np.eye(4), opening)
    actual_m = checker.component_transforms(board_T_hand, opening)
    for name, mesh in raw_cm.items():
        raw_homogeneous = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
        expected = (raw_homogeneous @ original_cm[name].T)[:, :3] / 100.
        baked_mesh = checker.hand_meshes[name]
        baked_homogeneous = np.column_stack([
            baked_mesh.vertices, np.ones(len(baked_mesh.vertices))])
        actual = (baked_homogeneous @ actual_m[name].T)[:, :3]
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual_m[name][:3, :3].T @ actual_m[name][:3, :3],
                                   np.eye(3), rtol=0., atol=1e-12)
        assert np.isclose(np.linalg.det(actual_m[name][:3, :3]), 1.)
