"""Finite saved-hand checks; no robot state, physics or approach-angle gate.

Reuse the existing unchanged-target/convex-component collision kernel. Closing
is a linear translation for each calibrated component, so its complete swept
volume is the convex hull of endpoint vertices, not discrete collision samples.
CAD solids are the union of closed edge-connected shells, as in that kernel.
"""
from __future__ import annotations

import numpy as np

from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider, NUMERICAL_TOL_M


DEFAULT_CONFIG = dict(normal_angle_rad=float(np.deg2rad(1.)),
                      contact_tolerance_m=1e-6, pad_edge_margin_m=.00025,
                      table_clearance_m=.0001, collision_tolerance_m=1e-8)
KNOWN_HAND_FRAME = 'Panda gripper0_grip_site; X jaw closing, +Z approach'


def _available_status(value):
    status = str(value).lower()
    return status.startswith('available') and not any(
        term in status for term in ('unavailable', 'missing', 'unknown'))


def _polygon_area(points):
    """Area of a clipped planar polygon; stable for translated small CAD."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.
    centered = points-points.mean(axis=0)
    return .5*float(np.linalg.norm(np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)))


def transform_points(points, pose):
    pose = np.asarray(pose, dtype=float)
    return np.asarray(points, dtype=float) @ pose[:3, :3].T + pose[:3, 3]


def rigid_transform(value, label='transform'):
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1., atol=1e-8, rtol=0)):
        raise ValueError(f'{label} must be a finite proper rigid 4x4 transform')
    return result.copy()


def component_vertices(component, opening):
    return (np.asarray(component['vertices_zero_m'], dtype=float)
            + opening * np.asarray(component['translation_per_opening'], dtype=float))


def pad_calibration(hand):
    """Read the verified grip-site export; never borrow Fabrica's hand offset.

    ``opening`` is the exported joint command, not necessarily physical pad gap.
    Unknown frame strings are unavailable rather than guessed. Rectangular
    usable inner faces are obtained from the two saved convex pad components.
    """
    if not isinstance(hand, dict) or hand.get('frame') != KNOWN_HAND_FRAME:
        raise ValueError('Missing verified hand-frame calibration (X closing, +Z approach)')
    limits = np.asarray(hand.get('opening_limits_m'), dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]:
        raise ValueError('Missing finite gripper opening limits')
    components = hand.get('components')
    if not isinstance(components, list) or not components:
        raise ValueError('Missing complete convex hand components')
    names = set()
    for component in components:
        points = np.asarray(component.get('vertices_zero_m'), dtype=float)
        motion = np.asarray(component.get('translation_per_opening'), dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
                or not np.isfinite(points).all() or np.linalg.matrix_rank(points-points[0]) != 3
                or motion.shape != (3,) or not np.isfinite(motion).all()
                or not isinstance(component.get('is_pad'), bool)
                or not component.get('name') or component['name'] in names):
            raise ValueError('Each hand component needs named finite 3D convex vertices and linear opening calibration')
        names.add(component['name'])
    pads = sorted((item for item in components if item['is_pad']),
                  key=lambda item: np.mean(item['vertices_zero_m'], axis=0)[0])
    if len(pads) != 2:
        raise ValueError('Exactly two calibrated usable inner pad components are required')
    centers, slopes, bounds = [], [], []
    for index, pad in enumerate(pads):
        points = np.asarray(pad['vertices_zero_m'], dtype=float)
        motion = np.asarray(pad['translation_per_opening'], dtype=float)
        if np.linalg.norm(motion[1:]) > 1e-9 or (motion[0] >= 0 if index == 0 else motion[0] <= 0):
            raise ValueError('Pad opening motions do not match the calibrated X-closing frame')
        inner = points[:, 0].max() if index == 0 else points[:, 0].min()
        face = points[np.abs(points[:, 0]-inner) <= 1e-9]
        if len(face) < 4 or np.ptp(face[:, 1]) <= 0 or np.ptp(face[:, 2]) <= 0:
            raise ValueError('Missing rectangular usable inner pad face')
        lower, upper = face.min(axis=0), face.max(axis=0)
        centers.append((lower+upper)/2)
        slopes.append(motion)
        bounds.append([lower, upper])
    centers, slopes = np.asarray(centers), np.asarray(slopes)
    if np.max(np.abs((centers[1]-centers[0])[1:])) > 1e-8:
        raise ValueError('Pad centers are not aligned on the calibrated closing axis')
    gap_slope = float(slopes[1, 0]-slopes[0, 0])
    if gap_slope <= 0:
        raise ValueError('Pad gap must increase with opening')
    return dict(centers_zero_m=centers, centers_per_opening=slopes,
                midpoint_zero_m=centers.mean(axis=0), midpoint_per_opening=slopes.mean(axis=0),
                gap_zero_m=float(centers[1, 0]-centers[0, 0]), gap_per_opening=gap_slope,
                closing_hand=np.array([1., 0., 0.]), approach_hand=np.array([0., 0., 1.]),
                pads=pads, usable_bounds_hand_zero_m=np.asarray(bounds), opening_limits_m=limits)


def _solid_collider(vertices, faces, label):
    vertices, raw_faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if raw_faces.dtype.kind not in 'iu':
        raise ValueError(f'{label} triangle indices must be integers')
    faces = raw_faces.astype(int, copy=False)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError(f'{label} needs finite vertices and triangle indices')
    # Exact welding only: STL duplicates must not make closed CAD look open.
    welded, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    cross = np.cross(welded[faces[:, 1]]-welded[faces[:, 0]],
                     welded[faces[:, 2]]-welded[faces[:, 0]])
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-20):
        raise ValueError(f'{label} contains degenerate triangles')
    collider = MeshCollider(welded, faces)
    if not collider.closed_components.all():
        raise ValueError(f'{label} has open or nonmanifold components; solid containment unavailable')
    # Normals are used for contact, so opposite traversal of shared edges matters.
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    keys, group = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(np.bincount(group, weights=signs, minlength=len(keys)) != 0):
        raise ValueError(f'{label} has inconsistent triangle winding')
    for index in range(collider.component_count):
        triangles = collider.faces[collider.component_ids == index]
        shifted = triangles-triangles.reshape(-1, 3).mean(axis=0)
        volume = np.sum(np.einsum('ij,ij->i', shifted[:, 0],
                        np.cross(shifted[:, 1], shifted[:, 2])))/6
        if volume <= 0:
            raise ValueError(f'{label} needs outward-oriented positive-volume solid shells')
    return collider


def _faces_at_point(collider, point, tolerance):
    """Point-on-triangle test in double precision, including shared edges."""
    ids = collider._query_box(point-tolerance, point+tolerance)
    if not len(ids):
        return ids
    triangles = collider.faces[ids]
    ab, ac, ap = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0], point-triangles[:, 0]
    normals = np.cross(ab, ac)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= lengths[:, None]
    plane_distance = np.abs(np.einsum('ij,ij->i', ap, normals))
    d00, d01, d11 = np.einsum('ij,ij->i', ab, ab), np.einsum('ij,ij->i', ab, ac), np.einsum('ij,ij->i', ac, ac)
    d20, d21 = np.einsum('ij,ij->i', ap, ab), np.einsum('ij,ij->i', ap, ac)
    denominator = d00*d11-d01*d01
    b = (d11*d20-d01*d21)/denominator
    c = (d00*d21-d01*d20)/denominator
    return ids[(plane_distance <= tolerance) & (b >= -1e-8) & (c >= -1e-8) & (b+c <= 1+1e-8)]


class GeometryValidator:
    """One-target validator. Missing mandatory inputs never become a valid pose.

    ``table_plane`` is [up_x, up_y, up_z, offset] with up.x + offset = 0.
    ``protected_contact_check(points_cad, contact_polygons=...,
    contact_face_ids=...)`` returns ``eligible`` (one bool per point),
    ``polygon_eligible`` (one bool per complete contact polygon) and an available
    ``status``. Checking polygon vertices alone cannot exclude a protected
    island inside the polygon. An unavailable status is rejected even when a
    legacy helper returns eligible=True. Face IDs retain the supplied CAD face
    order; exact vertex welding does not remove or reorder triangles.
    Receivers are dicts with vertices, faces and world_T_mesh. All complete hand
    components are convex collision solids from the verified export.

    ``pad_edge_margin_m`` guards intended pad centers only. Physical contact may
    cover the complete usable inner face, including its boundary. Surface
    contact queries use only the shared collision kernel's numerical tolerance;
    the larger contact_tolerance_m is for intended point/reconstruction checks,
    never a band of nearby noncontact surfaces whose normals must agree.
    """
    def __init__(self, vertices_cad, faces, hand, pickup_T, final_T, table_plane,
                 protected_contact_check, receivers, config=None):
        self.errors = []
        self.input_errors = self.errors
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        self.hand = hand
        self.protected_contact_check = protected_contact_check
        self.receivers = []
        try:
            if set(self.config) != set(DEFAULT_CONFIG):
                raise ValueError('Unknown geometry configuration key')
            if any(not np.isfinite(v) or v < 0 for v in self.config.values()):
                raise ValueError('Geometry tolerances must be finite and nonnegative')
            if self.config['normal_angle_rad'] >= np.pi/2:
                raise ValueError('Opposing-normal tolerance must be less than pi/2')
            self.calibration = pad_calibration(hand)
            self.target = _solid_collider(vertices_cad, faces, 'target CAD')
            self.pickup_T = rigid_transform(pickup_T, 'estimated pickup transform')
            self.final_T = rigid_transform(final_T, 'final assembly transform')
            self.table_plane = np.asarray(table_plane, dtype=float)
            if (self.table_plane.shape != (4,) or not np.isfinite(self.table_plane).all()
                    or not np.isclose(np.linalg.norm(self.table_plane[:3]), 1., atol=1e-8)):
                raise ValueError('Table plane needs a finite upward unit normal and offset')
            if not callable(protected_contact_check):
                raise ValueError('Protected mating-surface geometry/classifier is required')
            # Probe availability without proposing a contact. A missing task
            # region is an input problem even when the random bank is empty.
            probe = protected_contact_check(np.empty((0, 3)),
                                           contact_polygons=[], contact_face_ids=[])
            if not isinstance(probe, dict) or not _available_status(probe.get('status', '')):
                raise ValueError('Protected mating-surface information unavailable')
            if (np.asarray(probe.get('eligible')).shape != (0,)
                    or np.asarray(probe.get('eligible')).dtype.kind != 'b'
                    or np.asarray(probe.get('polygon_eligible')).shape != (0,)
                    or np.asarray(probe.get('polygon_eligible')).dtype.kind != 'b'):
                raise ValueError('Protected checker must classify complete contact polygons and support empty availability queries')
            if not isinstance(receivers, (list, tuple)) or not receivers:
                raise ValueError('Receiver geometry is required')
            for index, receiver in enumerate(receivers):
                world = rigid_transform(receiver['world_T_mesh'], 'receiver transform')
                vertices = transform_points(receiver['vertices'], world)
                self.receivers.append((receiver.get('name', str(index)),
                                       _solid_collider(vertices, receiver['faces'], 'receiver')))
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            self.errors.append(str(exc))

    def validate(self, candidate):
        checks = {}
        result = dict(status='invalid', valid=False, reasons=[], checks=checks,
                      closing_clearance_method='continuous_linear_convex_component_sweep',
                      closing_discrete_samples=0, tolerances=dict(self.config),
                      unchecked=['robot_arm_reachability', 'full_approach_path', 'insertion_path',
                                 'release', 'physical_grasp_stability'])
        def fail(reason, unavailable=False):
            result['reasons'].append(reason)
            result['status'] = 'unavailable_input' if unavailable else 'invalid'
            return result
        if self.errors:
            return fail('; '.join(self.errors), True)
        try:
            pose = rigid_transform(candidate['cad_T_hand'], 'candidate hand pose')
            contacts = np.asarray(candidate['contacts_cad_m'], dtype=float)
            held, opened = float(candidate['holding_opening_m']), float(candidate['pre_opening_m'])
            if contacts.shape != (2, 3) or not np.isfinite(contacts).all() or not np.isfinite([held, opened]).all():
                return fail('Candidate needs two finite CAD-frame contacts and finite openings')
            low, high = self.calibration['opening_limits_m']
            checks['opening_feasible'] = bool(low <= held <= opened <= high)
            if not checks['opening_feasible']:
                return fail('Holding or pre-opening outside calibrated limits')
            centers = self.calibration['centers_zero_m'] + held*self.calibration['centers_per_opening']
            reconstructed = transform_points(centers, pose)
            errors = np.linalg.norm(reconstructed-contacts, axis=1)
            checks['pad_center_reconstruction_error_m'] = errors.tolist()
            if np.any(errors > self.config['contact_tolerance_m']):
                return fail('Transformed pad centers do not coincide with intended contacts')
            cosine = np.cos(self.config['normal_angle_rad'])
            axis = pose[:3, :3] @ self.calibration['closing_hand']
            for index, point in enumerate(contacts):
                ids = _faces_at_point(self.target, point, self.config['contact_tolerance_m'])
                if not len(ids):
                    return fail('Intended pad center does not contact CAD surface')
                tri = self.target.faces[ids]
                normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                expected = -axis if index == 0 else axis
                if not np.any(normals @ expected >= cosine):
                    return fail('Intended pad-center surface normal is incompatible with pad')
            checks['pad_centers_contact_with_opposing_normals'] = True

            protected_points = [contacts]
            protected_polygons = []
            protected_face_ids = []
            margin = self.config['pad_edge_margin_m']
            tol = self.config['contact_tolerance_m']
            pad_lookup = {pad['name']: index for index, pad in enumerate(self.calibration['pads'])}
            table_min = np.inf
            for component in self.hand['components']:
                local_held = component_vertices(component, held)
                local_open = component_vertices(component, opened)
                world_held = transform_points(transform_points(local_held, pose), self.pickup_T)
                world_open = transform_points(transform_points(local_open, pose), self.pickup_T)
                table_min = min(table_min, float(np.min(np.concatenate((world_held, world_open))
                                                @ self.table_plane[:3] + self.table_plane[3])))
                cad_held, cad_open = transform_points(local_held, pose), transform_points(local_open, pose)
                if component['is_pad']:
                    index = pad_lookup[component['name']]
                    bounds = self.calibration['usable_bounds_hand_zero_m'][index] + held*np.asarray(component['translation_per_opening'])
                    center_edge_distance = np.minimum(centers[index, 1:]-bounds[0, 1:],
                                                       bounds[1, 1:]-centers[index, 1:])
                    if np.any(center_edge_distance < margin):
                        return fail('Intended pad center violates its pad-edge margin')
                    contact_polygons = self.target.surface_contacts(cad_held)
                    if not contact_polygons:
                        return fail('No modeled inner-pad contact region')
                    # Numerical clipping can turn a zero-area edge witness into
                    # a ribbon ~NUMERICAL_TOL_M wide. It is not positive-area
                    # support from the adjacent perpendicular CAD face.
                    area_tolerance = 8*NUMERICAL_TOL_M*float(np.max(bounds[1, 1:]-bounds[0, 1:]))
                    checks.setdefault('pad_boundary_support_area_tolerance_m2', {})[
                        component['name']] = area_tolerance
                    checks['pad_contact_region_query_clearance_m'] = 0.
                    aligned_witness = False
                    for witness in contact_polygons:
                        polygon = np.asarray(witness['polygon_world_m'])
                        local = (polygon-pose[:3, 3]) @ pose[:3, :3]
                        inner_x = centers[index, 0]
                        if (np.any(np.abs(local[:, 0]-inner_x) > 2*tol)
                                or np.any(local[:, 1:] < bounds[0, 1:]-2*tol)
                                or np.any(local[:, 1:] > bounds[1, 1:]+2*tol)):
                            return fail('Part contact occurs outside usable inner pad face')
                        expected = -axis if index == 0 else axis
                        aligned = bool(np.asarray(witness['normal_world']) @ expected >= cosine)
                        aligned_witness |= aligned
                        if not aligned and _polygon_area(polygon) > area_tolerance:
                            return fail('Surrounding contact geometry is not aligned with inner pad')
                        protected_points.append(polygon)
                        protected_polygons.append(polygon)
                        protected_face_ids.append(int(witness['triangle_index']))
                    if not aligned_witness:
                        return fail('No aligned inner-pad support witness')
                    # Retract toward pad body by the declared numerical collision
                    # tolerance so intended boundary contact is not penetration.
                    away = (-axis if index == 0 else axis)*self.config['collision_tolerance_m']
                    cad_held, cad_open = cad_held+away, cad_open+away
                if self.target.intersects(np.concatenate((cad_open, cad_held))):
                    return fail('Gripper solid intersects target during modeled closing: '+component['name'])
                final_vertices = transform_points(transform_points(local_held, pose), self.final_T)
                for name, receiver in self.receivers:
                    if receiver.intersects(final_vertices):
                        return fail('Complete gripper intersects receiver at nominal final pose: '+name)
            checks['whole_gripper_and_closing_clear'] = True
            checks['table_minimum_clearance_m'] = float(table_min)
            if table_min < self.config['table_clearance_m']:
                return fail('Complete gripper violates table clearance during closing')
            checks['table_clear'] = True
            points = np.unique(np.concatenate(protected_points), axis=0)
            protected = self.protected_contact_check(
                points, contact_polygons=protected_polygons,
                contact_face_ids=protected_face_ids)
            if not isinstance(protected, dict) or 'eligible' not in protected or 'status' not in protected:
                return fail('Protected-surface checker did not return explicit eligibility and status', True)
            status = str(protected['status'])
            checks['protected_surface_status'] = status
            if not _available_status(status):
                return fail('Protected mating-surface information unavailable: '+status, True)
            eligible = np.asarray(protected['eligible'])
            if eligible.shape != (len(points),) or eligible.dtype.kind != 'b':
                return fail('Protected-surface eligibility must contain one boolean per checked point', True)
            polygon_eligible = np.asarray(protected.get('polygon_eligible'))
            if (polygon_eligible.shape != (len(protected_polygons),)
                    or polygon_eligible.dtype.kind != 'b'):
                return fail('Protected-surface checker must explicitly classify every complete contact polygon', True)
            checks['protected_contact_polygons_checked'] = len(protected_polygons)
            if not eligible.all() or not polygon_eligible.all():
                return fail('Inner-pad contact occupies a protected mating surface')
            checks['protected_surfaces_clear'] = True
            checks['nominal_final_pose_clear'] = True
            result.update(status='valid', valid=True, cad_T_hand=pose.tolist(),
                          world_T_hand_pickup=(self.pickup_T@pose).tolist(),
                          world_T_hand_final=(self.final_T@pose).tolist(),
                          approach_world=(self.pickup_T[:3, :3]@pose[:3, :3]@self.calibration['approach_hand']).tolist())
            return result
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            return fail(str(exc), True)
