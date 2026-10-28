"""Select one CAD-guided grasp without simulation, perception or execution.

Programmatic callers supply the aligned CAD and explicit task geometry. The
small JSON entry point also accepts an explicit protected-triangle mask. It
never derives a mating axis from the final translation or loads simulator state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .criteria import build_reference, evaluate_criteria, select_priority
from .geometry import GeometryValidator
from .sampler import generate_candidates


REQUIRED = ('vertices_cad_m', 'faces', 'world_T_cad_pickup', 'world_T_cad_final',
            'axis_point_cad_m', 'axis_direction_cad', 'table_plane_world',
            'hand', 'protected_contact_check', 'receivers')
UNCHECKED = ['robot_arm_reachability', 'full_pregrasp_approach_path',
             'insertion_path', 'release', 'friction_and_physical_grasp_stability',
             'closing_induced_part_motion', 'physical_assembly_accuracy']


def _transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1e-9)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3),
                               rtol=0., atol=1e-8)
            or abs(np.linalg.det(matrix[:3, :3]) - 1.) > 1e-8):
        raise ValueError(f'{name} must be a finite proper rigid transform')
    return matrix


def protected_triangle_check(vertices, faces, protected_face_indices,
                             surface_tolerance_m=1e-6):
    """Build a checker from an EXPLICIT caller-supplied CAD surface mask.

    This does not discover mating surfaces. An explicitly empty mask means the
    caller declared no protected triangles; a missing mask is unavailable. For
    partial-face regions (e.g. part of a long pin), pass a region callback to
    select_grasp instead of over-labeling complete mesh faces.
    """
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    indices = np.asarray(protected_face_indices)
    if (indices.ndim != 1 or (len(indices) and not np.issubdtype(indices.dtype, np.integer))
            or (len(indices) and (indices.min() < 0 or indices.max() >= len(faces)))
            or not np.isfinite(surface_tolerance_m) or surface_tolerance_m <= 0):
        raise ValueError('Protected face IDs and surface tolerance must be explicit and valid')
    triangles = vertices[faces[indices.astype(int)]]
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge1, edge2)
    norm = np.linalg.norm(cross, axis=1)
    if np.any(norm <= 0):
        raise ValueError('Protected region contains degenerate triangles')
    normals = cross / norm[:, None]
    aa, ab, bb = (np.einsum('ij,ij->i', edge1, edge1),
                  np.einsum('ij,ij->i', edge1, edge2),
                  np.einsum('ij,ij->i', edge2, edge2))
    denominator = aa * bb - ab * ab

    def check(points_cad, *, contact_polygons=None, contact_face_ids=None):
        points = np.asarray(points_cad, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            return dict(status='unavailable_invalid_region_query', eligible=None)
        eligible = np.ones(len(points), dtype=bool)
        for row, point in enumerate(points):
            delta = point - triangles[:, 0]
            distance = np.einsum('ij,ij->i', delta, normals)
            ad = np.einsum('ij,ij->i', delta, edge1)
            bd = np.einsum('ij,ij->i', delta, edge2)
            u = (bb * ad - ab * bd) / denominator
            v = (aa * bd - ab * ad) / denominator
            # Barycentric roundoff is numerical, separate from the metre-scale
            # distance tolerance. Edge points belong to both adjacent surfaces.
            blocked = ((np.abs(distance) <= surface_tolerance_m)
                       & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1. + 1e-9))
            eligible[row] = not np.any(blocked)
        polygon_eligible = None
        if contact_face_ids is not None and contact_polygons is not None:
            # Every clipped contact polygon belongs to the corresponding CAD
            # triangle. Face labels therefore certify its entire interior,
            # including protected areas not detected by vertex-only queries.
            polygon_eligible = ~np.isin(contact_face_ids, indices)
        return dict(status='available_explicit_protected_triangle_mask',
                    eligible=eligible, polygon_eligible=polygon_eligible,
                    protected_face_count=len(indices))

    return check


def select_grasp(request, *, seed=0, max_poses=100, surface_budget=200,
                 orientation_samples=10, pre_opening_clearance_m=.001,
                 geometry_config=None, tie_tolerances=None, diagnostics=None):
    """Return one selected grasp or an explicit failure/unavailable status.

    ``request`` contains REQUIRED fields. Receivers use vertices/faces and a
    world_T_mesh transform. protected_contact_check takes (N,3) CAD points and
    returns {status, eligible:(N,), polygon_eligible:(P,)} for points plus the
    supplied contact_polygons/contact_face_ids keyword arguments. An unavailable
    status is never a pass. Partial-face region callbacks must certify polygon
    interiors, not just vertices.
    Candidate diagnostics append to the optional separate list, not the primary
    output. No caller-supplied array or previous experiment result is modified.
    """
    result = dict(status='unavailable_inputs', selected_grasp=None,
                  proposed_count=0, valid_count=0, seed=seed,
                  unchecked=UNCHECKED.copy(), robot_or_physics_execution=False,
                  pose_input='caller-supplied estimated CAD alignment; no environment state read',
                  contacts_frame='native_CAD', units='metres/radians')
    if not isinstance(request, dict):
        result.update(message='Required inputs must be supplied as a dictionary.',
                      missing_or_invalid_inputs=['request'])
        return result
    missing = [name for name in REQUIRED if request.get(name) is None]
    if missing:
        result.update(message='Required grasp-selection inputs are unavailable.',
                      missing_or_invalid_inputs=missing)
        return result
    try:
        if request.get('simulator_object_poses_read', False):
            raise ValueError('Simulator object ground-truth poses are not permitted inputs')
        if not callable(request['protected_contact_check']):
            raise ValueError('protected_contact_check must be an explicit region checker')
        if not len(request['receivers']):
            raise ValueError('Nominal receiver geometry is required')
        pickup = _transform(request['world_T_cad_pickup'], 'world_T_cad_pickup')
        final = _transform(request['world_T_cad_final'], 'world_T_cad_final')
        plane = np.asarray(request['table_plane_world'], dtype=float)
        if (plane.shape != (4,) or not np.isfinite(plane).all()
                or abs(np.linalg.norm(plane[:3]) - 1.) > 1e-8):
            raise ValueError('table_plane_world must have a finite upward UNIT normal')
        reference = build_reference(request['vertices_cad_m'], request['faces'],
                                    request['axis_point_cad_m'], request['axis_direction_cad'])
        validator = GeometryValidator(
            request['vertices_cad_m'], request['faces'], request['hand'],
            pickup, final, plane, request['protected_contact_check'],
            request['receivers'], config=geometry_config)
        # Constant task geometry/calibration errors should not be mislabeled as
        # failure of a finite random search.
        errors = validator.errors
        if errors:
            raise ValueError('; '.join(errors))
        candidates, sampling = generate_candidates(
            request['vertices_cad_m'], request['faces'], request['hand'], seed=seed,
            max_poses=max_poses, surface_budget=surface_budget,
            orientation_samples=orientation_samples,
            pre_opening_clearance_m=pre_opening_clearance_m)
        # Validate tie settings even if no ray pair was generated.
        select_priority([], tie_tolerances)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        result.update(message='Required geometry, calibration or configuration is unavailable or invalid.',
                      missing_or_invalid_inputs=[str(error)])
        return result
    result.update(reference=reference, sampling=sampling, proposed_count=len(candidates))
    valid = []
    unavailable = []
    for candidate in candidates:
        validation = validator.validate(candidate)
        record = dict(candidate, validity=validation)
        if validation['status'] == 'valid':
            cad_hand = np.asarray(candidate['cad_T_hand'])
            world_hand_pickup = pickup @ cad_hand
            world_hand_final = final @ cad_hand
            approach_world = pickup[:3, :3] @ np.asarray(candidate['approach_cad'])
            try:
                criteria = evaluate_criteria(candidate['contacts_cad_m'], reference,
                                              approach_world, plane[:3])
            except ValueError as error:
                record['validity'] = dict(status='unavailable_input', valid=False,
                                          reasons=['criteria_input_error:' + str(error)])
                unavailable.append(record['validity'])
            else:
                record.update(criteria=criteria, valid=True,
                              world_T_hand_pickup=world_hand_pickup.tolist(),
                              world_T_hand_final=world_hand_final.tolist(),
                              approach_world=approach_world.tolist())
                valid.append(record)
        elif validation['status'].startswith('unavailable'):
            unavailable.append(validation)
        if diagnostics is not None:
            diagnostics.append(record)
    decision = select_priority(valid, tie_tolerances)
    result.update(valid_count=len(valid), unavailable_candidate_count=len(unavailable),
                  selection_stages=decision['stages'],
                  tie_tolerances=decision['tie_tolerances'],
                  final_tied_candidate_ids=decision['final_tied_candidate_ids'])
    if valid:
        result.update(status='selected', selected_grasp=decision['selected_candidate'],
                      message='Selected one geometrically valid grasp by explicit criterion priority.')
    elif unavailable:
        result.update(status='unavailable_inputs',
                      message='Required candidate geometry or protected-surface information is unavailable.',
                      missing_or_invalid_inputs=[item.get('reasons', []) for item in unavailable])
    else:
        result.update(status='no_valid_grasp_within_budget',
                      message='No valid grasp found within the sampling budget.')
    return result


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path, help='Explicit metric CAD/task/gripper JSON inputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    diagnostics = [] if args.diagnostics else None
    try:
        request = json.loads(args.request.read_text(encoding='utf-8-sig'))
        if not isinstance(request, dict):
            raise ValueError('Request JSON must contain an object')
        if 'hand_model_file' in request and 'hand' not in request:
            hand_path = Path(request['hand_model_file'])
            if not hand_path.is_absolute():
                hand_path = args.request.parent / hand_path
            request['hand'] = json.loads(hand_path.read_text(encoding='utf-8-sig'))
        if ('protected_face_indices' in request and 'protected_contact_check' not in request
                and 'vertices_cad_m' in request and 'faces' in request):
            request['protected_contact_check'] = protected_triangle_check(
                request['vertices_cad_m'], request['faces'], request['protected_face_indices'])
        result = select_grasp(request, seed=args.seed, diagnostics=diagnostics,
                              geometry_config=request.get('geometry_config'),
                              tie_tolerances=request.get('tie_tolerances'))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        result = dict(status='unavailable_inputs', selected_grasp=None,
                      proposed_count=0, valid_count=0, seed=args.seed,
                      message='Required request geometry or calibration is unavailable or invalid.',
                      missing_or_invalid_inputs=[str(error)], unchecked=UNCHECKED.copy(),
                      robot_or_physics_execution=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=_json, allow_nan=False) + '\n',
                           encoding='utf-8')
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2, default=_json,
                                               allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], proposed_count=result['proposed_count'],
                         valid_count=result['valid_count'], output=str(args.output))))


if __name__ == '__main__':
    main()
