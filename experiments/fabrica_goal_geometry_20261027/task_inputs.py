"""Declared nominal NIST goals and contact regions; no simulator-state inputs.

Lengths are metres. Board coordinates retain world Z and put the plate centre
at XY=(0, 0). Native centred CAD coordinates are never re-centred or scaled.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'experiments/runtime_antipodal_assembly_20261023/goal_audit.json'
MESHES = ROOT / 'outputs/simulation/nist_task_board_1/assets/meshes'
PROVENANCE = MESHES.parent / 'provenance.json'
SOURCE_SCENE = (ROOT / 'outputs/datasets/nist_task_board/legacy_workspace_inputs'
                '/nist-taskboard-workspace/scene.json')
# Numerical comparisons only; these are not manufacturing/uncertainty margins.
PLANE_EPS_M = 1e-8
SURFACE_EPS_M = 1e-6
RAY_EPS_M = 1e-7
QUERY_CHUNK = 512


@lru_cache(maxsize=1)
def _sources():
    audit = json.loads(AUDIT.read_text())
    dimensions = json.loads(PROVENANCE.read_text())['dimensions_m']
    scene = json.loads(SOURCE_SCENE.read_text())
    tree = ast.parse((ROOT / 'simulation/nist_task_board_1.py').read_text())
    fixtures = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'FIXTURES')
    return audit, dimensions, np.asarray(scene['board']['center_xy_m']), fixtures


def _pose(position, yaw_deg=0.):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:2, :2] = [[c, -s], [s, c]]
    result[:3, 3] = position
    return result


def geometry_spec(part):
    """Return CAD goals and only the local assembly receiver for one NIST part.

    Connector contact regions remain unavailable: their existing nominal goal
    specifies bounding-plane contact, with no mating engagement depth.
    """
    audit, dimensions, legacy_center, fixtures = _sources()
    if part not in audit['current_nist12_parts']:
        raise ValueError(f'Part is not in the twelve-part Fabrica study: {part}')
    nominal = deepcopy(audit['parts'][part]['nominal_simulation_goal'])
    if nominal.get('world_T_cad_goal') is None:
        raise ValueError(f'No declared nominal goal for {part}')
    goal = np.asarray(nominal['world_T_cad_goal'], dtype=float).copy()
    goal[:2, 3] -= legacy_center
    board_top = float(audit['plate_top_world_z_m'])
    board_height = float(audit['plate_thickness_m'])
    receivers = [dict(name='board', mesh_path=MESHES / 'GMC_Laser_Plate_Virtual_m.stl',
                      board_T_mesh=_pose([0., 0., board_top - board_height / 2]))]

    def add_fixture(index):
        name, x_mm, y_mm, base_mm, yaw = fixtures[index]
        xy = (np.array([x_mm, y_mm], dtype=float) - 192.) * .001
        z = board_top + base_mm * .001 + dimensions[name][2] / 2
        receivers.append(dict(name=f'fixture_{index}_{name}',
            mesh_path=MESHES / f'{name}_m.stl',
            board_T_mesh=_pose([*xy, z], yaw)))

    if part.startswith(('RGOCG', 'KET')):
        kind = 'inserted_pin_segment'
        status = 'available_nominal_geometry'
        note = ('Contacts at or below the plate top in the declared seated goal '
                'are reserved. Only the inserted 12 mm segment is excluded; '
                'whole mesh faces and the remaining round side are not excluded.')
    elif part == 'Gear_Large':
        add_fixture(0)
        add_fixture(nominal['fixture_index'])
        kind = 'central_bore_and_bottom_seating_face'
        status = 'available_nominal_geometry'
        note = ('Reserve the centre-bore surface visible radially from the CAD Z '
                'axis, and the actual lowest downward-facing seating triangles. '
                'Shaft assignment, tooth phase and seating pose remain nominal.')
    elif part == 'M16_Hex_Nut':
        add_fixture(nominal['fixture_index'])
        kind = 'central_bore_and_bottom_seating_face'
        status = 'available_nominal_geometry'
        note = ('Reserve the centre-bore/thread surface visible radially from '
                'the CAD Z axis and lowest downward-facing seating triangles. '
                'The nominal target represents no thread engagement or tightening.')
    else:
        if part == 'DSUB_Male':
            add_fixture(12)
        add_fixture(nominal['fixture_index'])
        kind = 'unavailable_zero_engagement'
        status = 'unavailable_zero_engagement'
        note = ('The saved connector target touches bounding planes with zero '
                'assigned engagement. No mating contact region is invented. '
                'These candidates may receive the separate nominal endpoint '
                'hand-versus-receiver check only.')
    return dict(part=part, goal_T_cad_board=goal, receivers=receivers,
        region_kind=kind, region_status=status, region_note=note,
        nominal_goal_record=nominal, board_top_m=board_top,
        legacy_board_center_world_xy_m=legacy_center.copy(),
        frame='board-centred XY; original scene world Z; metres',
        goal_status='nominal_simulation_assumption',
        official_final_assembly_pose=False,
        source_paths=[str(AUDIT), str(PROVENANCE), str(SOURCE_SCENE),
                      str(ROOT / 'simulation/nist_task_board_1.py')])


def _nearest_faces(mesh, points):
    """Query unchanged triangles in native Fabrica centimetres, bounding memory.

    Trimesh's absolute tolerances can misclassify tiny metric STL triangles as
    degenerate. A uniform numerical unit conversion preserves face IDs and the
    physical surface; the original mesh and returned metric distances are kept.
    """
    import trimesh
    query_mesh = mesh.copy()
    query_mesh.apply_scale(100.)
    distances, faces = [], []
    for start in range(0, len(points), QUERY_CHUNK):
        _, distance, face = trimesh.proximity.closest_point(
            query_mesh, points[start:start + QUERY_CHUNK] * 100.)
        distances.append(distance / 100.)
        faces.append(face)
    return np.concatenate(distances), np.concatenate(faces)


def _radially_visible(mesh, points):
    """A bore faces the axis and is its first surface along a horizontal ray.

    The first-hit condition prevents inward-facing gear-tooth or hub recesses
    from being mislabeled as the central bore. No cylinder radius is invented.
    """
    visible = np.zeros(len(points), dtype=bool)
    known = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), QUERY_CHUNK):
        batch = points[start:start + QUERY_CHUNK]
        origins = np.zeros_like(batch)
        origins[:, 2] = batch[:, 2]
        radii = np.linalg.norm(batch[:, :2], axis=1)
        directions = batch - origins
        valid = radii > RAY_EPS_M
        directions[valid] /= radii[valid, None]
        locations, ray_ids, _ = mesh.ray.intersects_location(
            origins[valid], directions[valid], multiple_hits=True)
        minimum = np.full(np.count_nonzero(valid), np.inf)
        if len(ray_ids):
            lengths = np.linalg.norm(locations - origins[valid][ray_ids], axis=1)
            np.minimum.at(minimum, ray_ids, lengths)
        valid_ids = np.flatnonzero(valid)
        known[start + valid_ids] = np.isfinite(minimum)
        visible[start + valid_ids] = np.abs(minimum - radii[valid]) <= RAY_EPS_M
    return visible, known


def evaluate_contact_pairs(mesh_cad, contacts_cad, spec):
    """Classify supplied contact pairs without moving/resampling any candidate.

    Unavailable connector regions return eligibility for the *separate* endpoint
    test, with explicit unavailable status. Unclassifiable geometry on a part
    with an available region is rejected rather than counted as a clear region.
    """
    pairs = np.asarray(contacts_cad, dtype=float)
    if pairs.ndim != 3 or pairs.shape[1:] != (2, 3):
        raise ValueError('contacts_cad must have shape (P, 2, 3)')
    if not np.isfinite(pairs).all():
        raise ValueError('Contacts must be finite native metric CAD coordinates')
    count = len(pairs)
    kind = spec['region_kind']
    result = dict(eligible=np.ones(count, dtype=bool),
        reasons=['contact_region_clear'] * count,
        status=spec['region_status'], note=spec['region_note'])
    if kind == 'unavailable_zero_engagement':
        result['reasons'] = ['contact_region_unavailable_zero_engagement'] * count
        return result
    if count == 0:
        return result
    if kind == 'inserted_pin_segment':
        goal = np.asarray(spec['goal_T_cad_board'])
        heights = pairs @ goal[2, :3] + goal[2, 3]
        blocked = np.any(heights <= spec['board_top_m'] + PLANE_EPS_M, axis=1)
        result['eligible'][blocked] = False
        for index in np.flatnonzero(blocked):
            result['reasons'][index] = 'reserved_inserted_pin_segment'
        return result
    if kind != 'central_bore_and_bottom_seating_face':
        raise ValueError(f'Unsupported declared region: {kind}')

    points = pairs.reshape(-1, 3)
    distance, face_ids = _nearest_faces(mesh_cad, points)
    normals = np.asarray(mesh_cad.face_normals)[face_ids]
    triangles = np.asarray(mesh_cad.triangles)[face_ids]
    on_surface = distance <= SURFACE_EPS_M
    minimum_z = float(mesh_cad.bounds[0, 2])
    seat = (np.all(np.abs(triangles[:, :, 2] - minimum_z) <= PLANE_EPS_M, axis=1)
            & (normals[:, 2] < 0.))
    radial_dot = np.einsum('ij,ij->i', points[:, :2], normals[:, :2])
    inward = (radial_dot < -PLANE_EPS_M) & on_surface & ~seat
    bore = np.zeros(len(points), dtype=bool)
    known = on_surface.copy()
    if np.any(inward):
        visible, ray_known = _radially_visible(mesh_cad, points[inward])
        bore[inward] = visible
        known[inward] &= ray_known
    labels = np.full(len(points), 'clear', dtype=object)
    labels[seat] = 'reserved_bottom_seating_face'
    labels[bore] = 'reserved_central_bore_surface'
    labels[~known] = 'unclassified_contact_geometry'
    labels = labels.reshape(-1, 2)
    for index, pair_labels in enumerate(labels):
        reasons = sorted(set(pair_labels) - {'clear'})
        if reasons:
            result['eligible'][index] = False
            result['reasons'][index] = '+'.join(reasons)
    if not np.all(known):
        result['status'] = 'available_nominal_geometry_with_unclassified_contacts'
    return result
