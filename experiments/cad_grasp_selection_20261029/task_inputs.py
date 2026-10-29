"""Build explicit current CAD/task requests without generating any grasps.

Pickup alignment and table plane are supplied by the caller. Final poses,
receivers, axes and protected regions come only from the corrected task registry.
No old nominal-goal derivation or simulator object pose is used.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.cad_grasp_geometry_20261026.gallery import read_stl
from experiments.cad_grasp_geometry_20261026.inputs import local
from experiments.cad_grasp_selection_20261028.pipeline import _transform, protected_triangle_check
from .hand_frame import fingertip_calibration
from .sampling_regions import compile_sampling_region

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT/'outputs/nist_scene_corrections_20261029/assembly_goals.json'
HAND = ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'
REGION_TOLERANCE_M = 1e-8


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=48)
def _read_mesh_cached(path_text, digest):
    path = Path(path_text)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            vertices, faces = data['vertices'].copy(), data['faces'].copy()
    else:
        vertices, faces = read_stl(path)
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Task geometry needs finite metric vertices and integer triangle faces')
    return vertices, faces


def _load_mesh(path_text, expected_hash=None):
    path = local(path_text).resolve()
    digest = _sha(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f'Task mesh hash mismatch: {path}')
    vertices, faces = _read_mesh_cached(str(path), digest)
    return vertices, faces, dict(path=str(path), sha256=digest)


def _halfspace_checker(region):
    normal = np.asarray(region['normal_cad'], float)
    offset = float(region['offset_m'])
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(np.linalg.norm(normal)-1) > 1e-8 or not np.isfinite(offset)):
        raise ValueError('Protected halfspace needs an explicit finite unit CAD normal and metre offset')
    if region.get('inequality', 'normal@point+offset>=0') != 'normal@point+offset>=0':
        raise ValueError('Unsupported protected-halfspace inequality')

    def check(points, *, contact_polygons, contact_face_ids):
        values = np.asarray(points, float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError('Protected query points need finite CAD coordinates')
        if len(contact_polygons) != len(contact_face_ids):
            raise ValueError('Every contact polygon needs its CAD face ID')
        polygon_eligible = []
        for polygon in contact_polygons:
            polygon = np.asarray(polygon, float)
            if (polygon.ndim != 2 or polygon.shape[1] != 3 or not len(polygon)
                    or not np.isfinite(polygon).all()):
                raise ValueError('Protected checker needs finite complete contact polygons')
            # A linear halfspace reaches its extrema at polygon vertices, so
            # this certifies the whole polygon, including its interior.
            polygon_eligible.append(bool(np.all(polygon@normal+offset < -REGION_TOLERANCE_M)))
        return dict(status='available_corrected_task_halfspace',
            eligible=values@normal+offset < -REGION_TOLERANCE_M,
            polygon_eligible=np.asarray(polygon_eligible, dtype=bool))
    return check


def protected_checker(region, vertices, faces):
    """Compile only explicit corrected region declarations, never infer one."""
    if not isinstance(region, dict):
        raise ValueError('Corrected protected-contact region is unavailable')
    kind = region.get('type')
    if kind == 'halfspace':
        return _halfspace_checker(region)
    if kind == 'face_ids':
        ids = region.get('protected_face_ids', region.get('face_ids'))
        if ids is None:
            raise ValueError('Protected face IDs are unavailable')
        return protected_triangle_check(vertices, faces, ids)
    if kind == 'union':
        regions = region.get('regions')
        if not isinstance(regions, list) or not regions:
            raise ValueError('Protected union needs explicit nonempty component regions')
        checkers = [protected_checker(item, vertices, faces) for item in regions]
        def check(points, *, contact_polygons, contact_face_ids):
            checked = [c(points, contact_polygons=contact_polygons,
                         contact_face_ids=contact_face_ids) for c in checkers]
            return dict(status='available_corrected_task_region_union',
                eligible=np.logical_and.reduce([r['eligible'] for r in checked]),
                polygon_eligible=np.logical_and.reduce([r['polygon_eligible'] for r in checked]))
        return check
    raise ValueError(f'Unsupported or missing corrected protected-region type: {kind}')


@lru_cache(maxsize=24)
def _prepared_target(part, path_text, digest):
    vertices, faces = _read_mesh_cached(path_text, digest)
    # Retain the same declared 50 nm Gear_Large seam weld. Other targets stay
    # unchanged, and triangle ordering (protected face IDs) is preserved.
    from experiments.cad_grasp_selection_20261028.run_nist import prepare_target
    prepared, prepared_faces, _, receipt = prepare_target(part, vertices, faces)
    return prepared, prepared_faces, receipt


def build_request(part, world_T_cad_pickup, table_plane_world, world_T_board=None,
                  registry=None, hand=None, *, allow_protected_only_sampling=False):
    """Return ``{status, request, source}``, or explicit unavailable input.

    This constructs data and region callbacks only. It never calls a candidate
    sampler, selector, simulator, CAD pose fitter or learned model. Board pose
    defaults to the nominal task placement explicitly stored in the registry.
    """
    unavailable = dict(status='unavailable_inputs', request=None, part=part)
    try:
        registry_source = str(REGISTRY) if registry is None else 'explicit caller-supplied corrected registry'
        if registry is None:
            registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
        if not isinstance(registry, dict) or registry.get('units') not in ('metres', 'm'):
            raise ValueError('Corrected task registry with explicit metre units is required')
        rows = registry.get('parts')
        if not isinstance(rows, list):
            raise ValueError('Corrected task part records are unavailable')
        matches = [row for row in rows if row.get('part') == part]
        if len(matches) != 1:
            raise ValueError('Exactly one corrected task record is required for the requested part')
        row = matches[0]
        pickup = _transform(world_T_cad_pickup, 'caller-supplied observed pickup alignment')
        plane = np.asarray(table_plane_world, float)
        if (plane.shape != (4,) or not np.isfinite(plane).all()
                or abs(np.linalg.norm(plane[:3])-1) > 1e-8):
            raise ValueError('Observed table plane needs a finite unit normal and metre offset')
        board = _transform(registry.get('world_T_board') if world_T_board is None else world_T_board,
                           'explicit world_T_board')
        final_board = _transform(row.get('board_T_part'), 'corrected board_T_part')
        if world_T_board is None and row.get('world_T_cad_final') is not None:
            if not np.allclose(board@final_board, _transform(row['world_T_cad_final'], 'registry final pose'), atol=1e-9, rtol=0):
                raise ValueError('Registry board-relative and world final poses disagree')
        axis_point = np.asarray(row.get('axis_point_cad_m'), float)
        axis = np.asarray(row.get('axis_direction_cad'), float)
        if (axis_point.shape != (3,) or not np.isfinite(axis_point).all()
                or axis.shape != (3,) or not np.isfinite(axis).all()
                or abs(np.linalg.norm(axis)-1) > 1e-8):
            raise ValueError('Corrected mating axis needs a finite CAD point and unit direction')
        expected_board_axis = np.asarray(row.get('axis_direction_board'), float)
        if expected_board_axis.shape != (3,) or not np.allclose(final_board[:3, :3]@axis, expected_board_axis, atol=1e-8, rtol=0):
            raise ValueError('Corrected CAD and board mating-axis frames disagree')
        _, _, mesh_source = _load_mesh(row['part_mesh'], row.get('part_mesh_sha256'))
        vertices, faces, weld = _prepared_target(part, mesh_source['path'], mesh_source['sha256'])
        check = protected_checker(row.get('protected_contact_region'), vertices, faces)
        check(np.empty((0, 3)), contact_polygons=[], contact_face_ids=[])
        if not isinstance(allow_protected_only_sampling, bool):
            raise ValueError('Protected-only sampling opt-in must be an explicit boolean')
        positive_region = row.get('sampling_region')
        if positive_region is None and not allow_protected_only_sampling:
            from .sampling_annotations import annotation_for
            positive_region = annotation_for(part, vertices, faces, row)
        if positive_region is None and not allow_protected_only_sampling:
            raise ValueError('Reviewed positive contact-sampling annotation is unavailable')
        sampling = compile_sampling_region(vertices, faces,
            row['protected_contact_region'], positive_region)
        if part.startswith(('KET', 'RGOCG')):
            depth = float(registry.get('pin_tip_depth_below_board_top_m',
                                       row.get('pin_tip_depth_below_board_top_m', np.nan)))
            board_top = float(registry.get('board_top_board_m', np.nan))
            if not np.isfinite([depth, board_top]).all() or depth <= 0:
                raise ValueError('Pin task needs an explicit positive depth below the board top')
            if ('pin_tip_depth_below_board_top_m' in row
                    and abs(float(row['pin_tip_depth_below_board_top_m'])-depth) > 1e-10):
                raise ValueError('Pin goal and current depth requirement disagree')
            tip_axis_coordinate = float(np.max((vertices-axis_point)@axis))
            tip = final_board[:3, :3]@(axis_point+tip_axis_coordinate*axis)+final_board[:3, 3]
            if abs(board_top-tip[2]-depth) > 1e-8:
                raise ValueError('Pin final transform does not implement the declared depth below board top')
            boundary = axis_point+(tip_axis_coordinate-depth)*axis
            probe = np.asarray([boundary-1e-5*axis, boundary, boundary+1e-5*axis])
            classified = check(probe, contact_polygons=[], contact_face_ids=[])['eligible']
            if not np.array_equal(classified, [True, False, False]):
                raise ValueError('Pin protected region does not match the current inserted end/depth')
        if hand is None:
            hand = json.loads(HAND.read_text(encoding='utf-8'))
        tip = fingertip_calibration(hand)
        receiver_rows = row.get('receivers')
        if not isinstance(receiver_rows, list) or not receiver_rows:
            raise ValueError('Corrected receiver geometry is unavailable')
        receivers, receiver_sources = [], []
        for receiver in receiver_rows:
            v, f, source = _load_mesh(receiver['mesh_path'], receiver.get('mesh_sha256'))
            transform = _transform(receiver.get('board_T_mesh'), 'corrected receiver board_T_mesh')
            receivers.append(dict(name=receiver['name'], vertices=v, faces=f,
                                   world_T_mesh=board@transform))
            receiver_sources.append(dict(source, name=receiver['name'], board_T_mesh=transform.tolist()))
        request = dict(vertices_cad_m=vertices, faces=faces, hand=hand,
            world_T_cad_pickup=pickup, world_T_cad_final=board@final_board,
            axis_point_cad_m=axis_point, axis_direction_cad=axis,
            table_plane_world=plane, protected_contact_check=check,
            protected_contact_region=row['protected_contact_region'],
            sampling_region=positive_region, sampling_contact_region=sampling,
            sampling_mode='assembly_region_guided',
            receivers=receivers, simulator_object_poses_read=False)
        return dict(status='available', part=part, request=request,
            source=dict(task_registry=registry_source,
                pickup_pose_source='explicit caller-supplied observed CAD alignment',
                table_plane_source='explicit caller-supplied observation/calibration',
                world_T_board=board.tolist(), mesh=mesh_source, target_preprocessing=weld,
                receivers=receiver_sources, hand_pose_reference=tip['frame'],
                final_goal_status=row.get('status', 'explicit corrected CAD reference'),
                native_registration_status=row.get('native_registration_status'),
                physical_assembly_validated=False,
                sampling_region=sampling['provenance'],
                pin_tip_depth_below_board_top_m=registry.get('pin_tip_depth_below_board_top_m',
                    row.get('pin_tip_depth_below_board_top_m')),
                no_candidate_generation=True, no_pose_estimation=True,
                no_simulator_state_read=True, no_pickup_or_assembly_execution=True))
    except (OSError, ValueError, TypeError, KeyError, IndexError, ImportError) as exc:
        return dict(unavailable, message=str(exc))
