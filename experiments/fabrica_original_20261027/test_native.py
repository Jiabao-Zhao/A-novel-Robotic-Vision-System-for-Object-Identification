"""Offline integration checks for the pinned Fabrica geometry implementation.

Run in a fresh process with the isolated dependency directory on PYTHONPATH so
pytest plugins cannot import the shared environment's NumPy first.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEPENDENCIES = ROOT / 'outputs/cache/fabrica_native_20261027/python'
VENDOR = HERE / 'vendor/fabrica'
sys.path.insert(0, str(DEPENDENCIES.resolve()))
sys.path.insert(0, str(VENDOR.resolve()))

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from planning.robot import geometry
from planning.robot import util_grasp


@contextmanager
def native_loop_deadline(seconds=10):
    """Keep a failing native unbounded sampler from hanging the test process."""
    if not hasattr(signal, 'SIGALRM'):
        pytest.skip('The native integration checks run in the existing Linux WSL environment')
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError('Original Fabrica sampler exceeded the synthetic-test deadline')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def test_vendored_files_are_exact_pinned_git_blobs():
    manifest = json.loads((HERE / 'vendor/source_manifest.json').read_text())
    assert manifest['commit'] == '215a30fe51e59299588a5b5e417a9cb934fb393e'
    assert len(manifest['files']) == 13
    for entry in manifest['files']:
        data = (VENDOR / entry['path']).read_bytes()
        assert len(data) == entry['bytes'], entry['path']
        header = f'blob {len(data)}\0'.encode('ascii')
        assert hashlib.sha1(header + data).hexdigest() == entry['git_blob_sha'], entry['path']


def test_pinned_local_dependencies_execute_real_fcl_collision_queries():
    import fcl
    import rtree

    for module in (np, trimesh, fcl, rtree):
        assert Path(module.__file__).resolve().is_relative_to(DEPENDENCIES.resolve())
    assert metadata.version('numpy') == '1.26.4'
    assert metadata.version('scipy') == '1.12.0'
    assert metadata.version('trimesh') == '4.4.1'
    assert metadata.version('python-fcl') == '0.7.0.6'
    assert metadata.version('rtree') == '1.3.0'
    assert Path(util_grasp.__file__).resolve() == VENDOR / 'planning/robot/util_grasp.py'
    assert Path(geometry.__file__).resolve() == VENDOR / 'planning/robot/geometry.py'

    manager = trimesh.collision.CollisionManager()
    manager.add_object('fixed_box', trimesh.creation.box(extents=[2., 2., 2.]))
    moving = trimesh.creation.box(extents=[1., 1., 1.])
    assert manager.in_collision_single(moving)
    moving.apply_translation([4., 0., 0.])
    assert not manager.in_collision_single(moving)


def test_original_sampler_returns_analytic_box_surface_pairs_without_ground_gate():
    # Native units are centimetres. No box point exceeds the native default
    # ground threshold z > 1 cm; the public ground_z override is essential here.
    mesh = trimesh.creation.box(extents=[2., 2., 2.])
    assert mesh.bounds[1, 2] == 1.
    state = np.random.get_state()
    try:
        np.random.seed(19)
        with native_loop_deadline():
            pairs = util_grasp.compute_antipodal_pairs(
                mesh, sample_budget=8, ground_z=-np.inf)
    finally:
        np.random.set_state(state)
    assert pairs.ndim == 3 and pairs.shape[1:] == (2, 3)
    assert len(pairs) > 0
    assert np.isfinite(pairs).all()
    assert np.all(np.abs(pairs) <= 1. + 1e-9)
    for first, second in pairs:
        difference = second - first
        axis = int(np.argmax(np.abs(difference)))
        # An inward normal ray through a unit half-width axis-aligned box must
        # exit the opposite face, with the other two coordinates unchanged.
        assert np.isclose(abs(difference[axis]), 2., atol=1e-9)
        assert np.allclose(np.delete(difference, axis), 0., atol=1e-9)
        assert np.isclose(abs(first[axis]), 1., atol=1e-9)
        assert np.isclose(abs(second[axis]), 1., atol=1e-9)
        assert first[axis] * second[axis] < 0.


def test_original_panda_pose_expansion_preserves_hand_frame_and_native_offset():
    center = np.array([1.2, -0.3, -2.5])
    pair = np.array([center + [-0.5, 0., 0.], center + [0.5, 0., 0.]])
    open_ratio = geometry.get_gripper_open_ratio('panda', pair)
    assert np.isclose(open_ratio, 1. / 8.)
    state = np.random.get_state()
    try:
        np.random.seed(29)
        positions, quaternions = util_grasp.generate_gripper_states(
            'panda', pair, open_ratio, sample_budget=10)
    finally:
        np.random.set_state(state)
    assert len(positions) == len(quaternions) == 5
    bases = []
    for position, quaternion in zip(positions, quaternions):
        assert np.isclose(np.linalg.norm(quaternion), 1., atol=1e-12)
        rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])
        closing = rotation.apply([0., 1., 0.])
        base = rotation.apply([0., 0., -1.])
        assert np.allclose(closing, [1., 0., 0.], atol=1e-10)
        assert base[2] >= -1e-12
        assert np.isclose(np.dot(base, closing), 0., atol=1e-12)
        assert np.allclose(np.asarray(position) - center, 11.24 * base, atol=1e-10)
        bases.append(base)
    assert len(np.unique(np.round(bases, 9), axis=0)) == len(bases)
    assert any(base[2] < 0.99 for base in bases), 'Native poses must retain tilted approaches'


def test_original_target_check_accepts_clear_pose_and_rejects_palm_overlap(capsys):
    from experiments.fabrica_original_20261027 import native

    up = native.load()
    cls = up.generator_class
    context = cls.__new__(cls)
    part_id = 'target'
    target = trimesh.creation.box(extents=[1., 1., 1.])
    context.gripper_type = 'panda'
    context.part_ids = [part_id]
    context.part_final_transforms = {part_id: np.eye(4)}
    context.G_preced = SimpleNamespace(nodes={part_id: {'path': None, 'parts_after': []}})
    context.gripper_meshes = up.geometry['load_gripper_meshes'](
        'panda', str(VENDOR / 'assets'), has_ft_sensor=False, visual=False)
    buffered = up.geometry['get_buffered_gripper_meshes']('panda', context.gripper_meshes)
    manager = trimesh.collision.CollisionManager
    context.ground_col_manager = manager()
    context.part_col_manager = manager()
    context.part_col_manager.add_object(part_id, target)
    context.gripper_col_manager = manager()
    context.gripper_col_manager_buffered = manager()
    for name, mesh in context.gripper_meshes.items():
        context.gripper_col_manager.add_object(name, mesh)
    for name, mesh in buffered.items():
        context.gripper_col_manager_buffered.add_object(name, mesh)

    pair = np.array([[-0.5, 0., 0.], [0.5, 0., 0.]])
    ratio = up.geometry['get_gripper_open_ratio']('panda', pair)
    position, quaternion = up.utility['get_gripper_pos_quat'](
        'panda', pair.mean(axis=0), np.array([0., 0., 1.]),
        np.array([1., 0., 0.]), ratio)
    grasp_type = up.utility['Grasp']
    clear = grasp_type(part_id, 0, position, quaternion, ratio)
    assert context.check_grasp_feasible(clear, part_id, verbose=True) is not None

    # Move a known point on a real palm triangle to the center of the target.
    # This deliberately overlapping input tests the original target FCL gate,
    # rather than depending on an assumed palm center or a workspace obstacle.
    rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])
    palm_surface = context.gripper_meshes['panda_hand'].triangles[0].mean(axis=0)
    bad_position = -rotation.apply(palm_surface)
    blocked = grasp_type(part_id, 1, bad_position, quaternion, ratio)
    assert context.check_grasp_feasible(blocked, part_id, verbose=True) is None
    assert 'gripper-grasping part collision' in capsys.readouterr().out
