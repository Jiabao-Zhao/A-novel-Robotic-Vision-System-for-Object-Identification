"""Small reviewed CAD labels for the current twelve NIST study parts.

These annotations define proposal locations, not physical graspability. They
never remove CAD triangles from ray casting or collision checking. Component
IDs are meaningful only for the recorded mesh hash and triangle ordering.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.nist_mating_goals_20261028.cad_tools import mesh_component_labels

CONFIG = Path(__file__).with_name('config.json')
RADIAL_NORMAL_TOLERANCE_M = 1e-8
_CACHE = {}


@lru_cache(maxsize=1)
def _annotations():
    return json.loads(CONFIG.read_text(encoding='utf-8'))['sampling_annotations']


def _box(lower, upper):
    regions = []
    for axis in range(3):
        if lower[axis] is not None:
            n = np.eye(3)[axis]
            regions.append(dict(type='halfspace', normal_cad=n.tolist(), offset_m=-float(lower[axis])))
        if upper[axis] is not None:
            n = -np.eye(3)[axis]
            regions.append(dict(type='halfspace', normal_cad=n.tolist(), offset_m=float(upper[axis])))
    return regions


def annotation_for(part, vertices, faces, row):
    """Return explicit positive face/halfspace labels, or None if unannotated."""
    entry = _annotations().get(part)
    if entry is None:
        return None
    if row.get('part_mesh_sha256') != entry['part_mesh_sha256']:
        raise ValueError('Sampling annotation CAD hash does not match the corrected task mesh')
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (faces.shape != (entry['triangle_count'], 3) or faces.dtype.kind not in 'iu'
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()):
        raise ValueError('Sampling annotation requires its recorded CAD triangle ordering')
    point = np.asarray(row['axis_point_cad_m'], float)
    axis = np.asarray(row['axis_direction_cad'], float)
    if point.shape != (3,) or axis.shape != (3,) or abs(np.linalg.norm(axis)-1) > 1e-8:
        raise ValueError('Sampling annotation requires the explicit CAD mating axis')
    key = (part, hashlib.sha256(vertices.tobytes()+faces.tobytes()+point.tobytes()+axis.tobytes()).hexdigest())
    if key in _CACHE:
        return deepcopy(_CACHE[key])
    triangles = vertices[faces]
    cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    length = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, length[:,None], out=np.zeros_like(cross), where=length[:,None]>0)
    radial = triangles.mean(axis=1)-point
    radial -= np.outer(radial@axis, axis)
    # Outward radial faces include cylinder sides, hex flats and gear flanks.
    # Axial caps have zero radial dot product; inward bore walls are negative.
    # This is a declared geometric label rule, not a learned affordance or a
    # guarantee that a finite pad can fit. Final geometry checks remain required.
    eligible = (np.einsum('ij,ij->i', radial, normals) > RADIAL_NORMAL_TOLERANCE_M) & (length > 0)
    if 'component_ids' in entry:
        labels = mesh_component_labels(faces)
        ids = np.asarray(entry['component_ids'], int)
        if not np.isin(ids, np.unique(labels)).all():
            raise ValueError('Reviewed sampling component IDs are unavailable in the current CAD')
        eligible &= np.isin(labels, ids)
    regions = [dict(type='face_ids', face_ids=np.flatnonzero(eligible).tolist())]
    if 'bounds_cad_m' in entry:
        bounds = entry['bounds_cad_m']
        regions.extend(_box(bounds[0], bounds[1]))
    result = dict(type='intersection', regions=regions, label=entry['label'],
        provenance=dict(source='explicit initial CAD-region annotation',
            rule='outward radial CAD faces, optional reviewed component IDs and housing limits',
            radial_normal_tolerance_m=RADIAL_NORMAL_TOLERANCE_M,
            part_mesh_sha256=entry['part_mesh_sha256'],
            reviewed_geometry_basis=entry['reviewed_geometry_basis'],
            functional_graspability_validated=False,
            full_CAD_must_remain_in_ray_and_collision_checks=True))
    _CACHE[key] = result
    return deepcopy(result)
