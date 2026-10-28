"""Execute pinned Fabrica functions and class methods in an object-only context.

Original bodies are compiled unchanged. The context supplies one observed CAD,
an empty workspace collision manager, and the public ground_z=-inf override.
No finite-pad checker, robot framework, learned model or VLM is imported.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(__file__).with_name('vendor') / 'fabrica'
DEPENDENCIES = ROOT / 'outputs/cache/fabrica_native_20261027/python'
if str(DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(DEPENDENCIES))

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


_UPSTREAM = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(path, namespace, classes=False):
    """Retain original AST bodies; exclude unrelated top-level imports/drivers."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    kinds = (ast.FunctionDef, ast.ClassDef) if classes else (ast.FunctionDef,)
    nodes = [node for node in tree.body if isinstance(node, kinds)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load():
    global _UPSTREAM
    if _UPSTREAM is not None:
        return _UPSTREAM
    import sys
    if str(DEPENDENCIES) not in sys.path:
        sys.path.insert(0, str(DEPENDENCIES))
    import trimesh
    from tqdm import tqdm
    transform = definitions(VENDOR/'assets/transform.py', {'np':np, 'Rotation':R})
    geometry = definitions(VENDOR/'planning/robot/geometry.py',
        dict(transform, np=np, R=R, trimesh=trimesh, os=os))
    utility = definitions(VENDOR/'planning/robot/util_grasp.py',
        dict(geometry, np=np, R=R, trimesh=trimesh, copy=copy, cKDTree=cKDTree), classes=True)
    parallel = definitions(VENDOR/'utils/parallel.py',
        dict(mp=mp, tqdm=tqdm, traceback=traceback))
    config = {}
    config_path = VENDOR/'planning/config.py'
    exec(compile(config_path.read_text(), str(config_path), 'exec'), config)
    base = dict(utility, **{k:v for k,v in parallel.items() if k != '__builtins__'})
    base.update({k:v for k,v in config.items() if k != '__builtins__'})
    runner = definitions(VENDOR/'planning/run_grasp_gen.py', base, classes=True)
    _UPSTREAM = SimpleNamespace(trimesh=trimesh, transform=transform, geometry=geometry,
        utility=utility, runner=runner, generator_class=runner['GraspGenerator'])
    return _UPSTREAM


def generate(mesh_cm, seed=0, surface_budget=200, angle_budget=10, ground_z=-np.inf):
    """Return every upstream pair/pose and decision; all native lengths are cm.

    The original constructor assumes an assembly graph and fixture workcell.
    Its methods execute on an explicit one-part, identity-path context instead.
    Trace wrappers change logging only, never a feasibility predicate.
    """
    up = load()
    import random
    random.seed(seed)
    np.random.seed(seed)
    cls = up.generator_class
    instance = cls.__new__(cls)
    part = 'target'
    instance.gripper_type = 'panda'
    instance.part_ids = [part]
    instance.part_meshes = {part:mesh_cm.copy()}
    instance.part_final_transforms = {part:np.eye(4)}
    instance.G_preced = SimpleNamespace(nodes={part:dict(parts_after=[], path=None)})
    instance.n_surface_pt = surface_budget
    instance.n_angle = angle_budget
    instance.antipodal_thres = 0.95
    instance.offset_delta = 0.0
    instance.gripper_meshes = up.geometry['load_gripper_meshes'](
        'panda', str(VENDOR/'assets'), has_ft_sensor=False, visual=False)
    instance.gripper_meshes_buffered = up.geometry['get_buffered_gripper_meshes'](
        'panda', instance.gripper_meshes)
    manager = up.trimesh.collision.CollisionManager
    instance.part_col_manager = manager()
    instance.part_col_manager.add_object(part, mesh_cm)
    instance.gripper_col_manager = manager()
    instance.gripper_col_manager_buffered = manager()
    for name, mesh in instance.gripper_meshes.items():
        instance.gripper_col_manager.add_object(name, mesh)
    for name, mesh in instance.gripper_meshes_buffered.items():
        instance.gripper_col_manager_buffered.add_object(name, mesh)
    instance.ground_col_manager = manager()

    trace = dict(pairs_cm=np.empty((0,2,3)), generated=[], decisions=[], native_check_messages=[])
    pair_lookup = {}
    original_pairs = up.utility['compute_antipodal_pairs']
    original_states = up.utility['generate_gripper_states']
    original_check = cls.check_grasp_feasible.__get__(instance, cls)

    def traced_pairs(mesh, *args, **kwargs):
        kwargs['ground_z'] = ground_z
        pairs = original_pairs(mesh, *args, **kwargs)
        trace['pairs_cm'] = pairs.copy()
        for index, pair in enumerate(pairs):
            pair_lookup.setdefault(np.asarray(pair, dtype=float).tobytes(), index)
        return pairs

    def traced_states(gripper_type, points, open_ratio, *args, **kwargs):
        positions, quaternions = original_states(gripper_type, points, open_ratio, *args, **kwargs)
        pair_index = pair_lookup[np.asarray(points, dtype=float).tobytes()]
        for position, quaternion in zip(positions, quaternions):
            trace['generated'].append(dict(grasp_id=len(trace['generated']), pair_index=pair_index,
                pos_cm=np.asarray(position).copy(), quat_wxyz=np.asarray(quaternion).copy(),
                open_ratio=float(open_ratio)))
        return positions, quaternions

    def traced_check(grasp, part_id, verbose=False):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            result = original_check(grasp, part_id, verbose=True)
        message = capture.getvalue().strip()
        trace['decisions'].append(dict(grasp_id=int(grasp.grasp_id), retained=result is not None,
            native_message=message or 'passed_native_check'))
        return result

    old_pairs = up.runner['compute_antipodal_pairs']
    old_states = up.runner['generate_gripper_states']
    up.runner['compute_antipodal_pairs'] = traced_pairs
    up.runner['generate_gripper_states'] = traced_states
    instance.check_grasp_feasible = traced_check
    try:
        retained = instance.generate_grasps(part, max_n_grasp=None, n_proc=1, verbose=False)
    finally:
        up.runner['compute_antipodal_pairs'] = old_pairs
        up.runner['generate_gripper_states'] = old_states
    if len(trace['generated']) != len(trace['decisions']):
        raise AssertionError('Native candidate/decision trace mismatch')
    accepted = {int(grasp.grasp_id) for grasp in retained}
    for pose, decision in zip(trace['generated'], trace['decisions']):
        if pose['grasp_id'] != decision['grasp_id'] or decision['retained'] != (pose['grasp_id'] in accepted):
            raise AssertionError('Native trace changed grasp identity/order')
        pose.update(decision)
    trace['retained_ids'] = sorted(accepted)
    trace['source_functions'] = ['compute_antipodal_pairs', 'filter_sparse_pairs',
        'sort_antipodal_pairs', 'generate_gripper_states', 'get_gripper_pos_quat',
        'get_gripper_open_ratio', 'get_gripper_meshes_transforms',
        'GraspGenerator.generate_grasps', 'GraspGenerator.check_grasp_feasible']
    return trace


def pose_metres(pose):
    transform = np.eye(4)
    transform[:3,:3] = R.from_quat(np.asarray(pose['quat_wxyz'])[[1,2,3,0]]).as_matrix()
    transform[:3,3] = np.asarray(pose['pos_cm'])/100.
    return transform
