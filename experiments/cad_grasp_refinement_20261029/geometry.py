"""Complete-hand validator with verified round-side normals and full diagnostics.

The collision mesh and calibrated hand are unchanged. Only a geometry-verified
cylinder contact model may replace a side-face normal; flat-face tolerances stay
at the configured one degree. Independent checks share one traversal rather
than repeating full validation to discover failures hidden by early returns.
"""
from __future__ import annotations

import hashlib
import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import (
    _available_status, _faces_at_point, _polygon_area, component_vertices,
    rigid_transform, transform_points,
)
from experiments.cad_grasp_geometry_20261026.geometry_kernel import NUMERICAL_TOL_M
from experiments.cad_grasp_selection_20261029.geometry import GeometryValidator as CurrentValidator
from experiments.cad_grasp_selection_20261029.hand_frame import FINGERTIP_FRAME

CATEGORIES = (
    'candidate_input', 'opening_limits', 'contact_reconstruction',
    'surface_normals', 'pad_contact_geometry', 'target_closing_collision',
    'table_clearance', 'final_receiver_collision', 'protected_surfaces',
)


def _digest(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


class GeometryValidator(CurrentValidator):
    def __init__(self, vertices_cad, faces, hand, pickup_T, final_T, table_plane,
                 protected_contact_check, receivers, config=None, contact_model=None):
        super().__init__(vertices_cad, faces, hand, pickup_T, final_T, table_plane,
                         protected_contact_check, receivers, config=config)
        self.contact_model = contact_model
        if self.errors or contact_model is None:
            return
        try:
            provenance = contact_model.provenance
            if (not callable(contact_model.contact_normals) or not isinstance(provenance, dict)
                    or provenance.get('mesh_vertices_sha256') != _digest(vertices_cad, '<f8')
                    or provenance.get('mesh_faces_sha256') != _digest(faces, '<i8')):
                raise ValueError('Contact model must be verified and bound to the complete current CAD arrays')
        except (AttributeError, TypeError, ValueError) as exc:
            self.errors.append(str(exc))

    def _normal_values(self, points, face_ids):
        ids = np.asarray(face_ids, dtype=int)
        if self.contact_model is None:
            triangles = self.target.faces[ids]
            values = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            values /= np.linalg.norm(values, axis=1)[:, None]
        else:
            values = np.asarray(self.contact_model.contact_normals(points, ids), dtype=float)
        if (values.shape != (len(ids), 3) or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError('Contact model must return a finite unit normal for every queried point/face')
        return values

    def validate(self, candidate):
        diagnostics = {key: dict(status='not_evaluated', reasons=[], failed=False, unavailable=False)
                       for key in CATEGORIES}
        checks = {}
        result = dict(status='invalid', valid=False, reasons=[], checks=checks,
            diagnostics=diagnostics, diagnostic_method='one_pass_independent_geometry_checks',
            closing_clearance_method='continuous_linear_convex_component_sweep',
            closing_discrete_samples=0, tolerances=dict(self.config),
            hand_pose_reference=FINGERTIP_FRAME,
            surface_normal_model=('original_mesh_facets' if self.contact_model is None
                                  else 'verified_round_side_model_with_original_facet_fallback'),
            unchecked=['robot_arm_reachability', 'full_approach_path', 'insertion_path',
                       'release', 'physical_grasp_stability'])

        def passed(category):
            if diagnostics[category]['status'] == 'not_evaluated':
                diagnostics[category]['status'] = 'pass'

        def fail(category, reason, unavailable=False):
            entry = diagnostics[category]
            entry['reasons'].append(str(reason))
            result['reasons'].append(str(reason))
            if unavailable:
                entry['unavailable'] = True
            else:
                entry['failed'] = True
            entry['status'] = 'unavailable' if entry['unavailable'] else 'fail'

        def finish():
            statuses = [entry['status'] for entry in diagnostics.values()]
            valid = all(status == 'pass' for status in statuses)
            result['valid'] = valid
            result['status'] = ('valid' if valid else 'unavailable_input'
                                if 'unavailable' in statuses else 'invalid')
            result['diagnostic_failure_categories'] = [key for key, entry in diagnostics.items() if entry['failed']]
            result['diagnostic_unavailable_categories'] = [key for key, entry in diagnostics.items() if entry['unavailable']]
            return result

        if self.errors:
            fail('candidate_input', '; '.join(self.errors), True)
            return finish()
        try:
            if candidate.get('hand_pose_reference') != FINGERTIP_FRAME:
                raise ValueError('Candidate must explicitly name the calibrated fingertip frame')
            tip_pose = rigid_transform(candidate['cad_T_hand'], 'fingertip pose')
            pose = tip_pose@self.tip['fingertip_T_grip_site']
            if ('cad_T_grip_site' in candidate and not np.allclose(
                    rigid_transform(candidate['cad_T_grip_site']), pose, atol=1e-9, rtol=0)):
                raise ValueError('Fingertip and grip-site pose fields disagree')
            inset = float(candidate.get('contact_inset_from_pad_tip_m', np.nan))
            if not np.isfinite(inset) or abs(inset-self.tip['contact_inset_m']) > 1e-12:
                raise ValueError('Candidate contact inset disagrees with the validator edge margin')
            contacts = np.asarray(candidate['contacts_cad_m'], dtype=float)
            held, opened = float(candidate['holding_opening_m']), float(candidate['pre_opening_m'])
            if contacts.shape != (2, 3) or not np.isfinite(contacts).all() or not np.isfinite([held, opened]).all():
                raise ValueError('Candidate needs two finite CAD contacts and finite opening commands')
            passed('candidate_input')
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            fail('candidate_input', str(exc), True)
            return finish()

        low, high = self.calibration['opening_limits_m']
        checks['opening_feasible'] = bool(low <= held <= opened <= high)
        if not checks['opening_feasible']:
            fail('opening_limits', 'Holding or pre-opening outside calibrated limits')
        else:
            passed('opening_limits')
        # All finite proposed configurations are inspected independently, even
        # when opening limits fail; such geometry is explicitly extrapolated and
        # cannot turn an opening-infeasible candidate into a valid robot pose.
        checks['geometry_extrapolated_outside_opening_limits'] = not checks['opening_feasible']
        centers = self.calibration['centers_zero_m']+held*self.calibration['centers_per_opening']
        reconstructed = transform_points(centers, pose)
        errors = np.linalg.norm(reconstructed-contacts, axis=1)
        checks['contact_anchor_reconstruction_error_m'] = errors.tolist()
        if np.any(errors > self.config['contact_tolerance_m']):
            fail('contact_reconstruction', 'Transformed near-tip contact anchors do not coincide with intended contacts')
        else:
            passed('contact_reconstruction')
        cosine = np.cos(self.config['normal_angle_rad'])
        axis = pose[:3, :3]@self.calibration['closing_hand']
        intended_errors = []
        try:
            for index, point in enumerate(contacts):
                ids = _faces_at_point(self.target, point, self.config['contact_tolerance_m'])
                if not len(ids):
                    fail('surface_normals', f'Intended contact {index} is not on the complete CAD surface')
                    intended_errors.append(None)
                    continue
                normals = self._normal_values(np.repeat(point[None], len(ids), axis=0), ids)
                expected = -axis if index == 0 else axis
                dots = normals@expected
                intended_errors.append(float(np.rad2deg(np.arccos(np.clip(dots.max(), -1, 1)))))
                if not np.any(dots >= cosine):
                    fail('surface_normals', f'Intended contact {index} surface normal is incompatible with pad')
            passed('surface_normals')
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            fail('surface_normals', str(exc), True)
        checks['intended_contact_normal_minimum_error_deg'] = intended_errors
        checks['contact_anchors_contact_with_opposing_normals'] = diagnostics['surface_normals']['status'] == 'pass'

        protected_points, protected_polygons, protected_face_ids = [contacts], [], []
        margin, tol = self.config['pad_edge_margin_m'], self.config['contact_tolerance_m']
        pad_lookup = {pad['name']: index for index, pad in enumerate(self.calibration['pads'])}
        table_min = np.inf
        complete_contact_regions = True
        component_reports = []
        for component in self.hand['components']:
            name = component['name']
            local_held, local_open = component_vertices(component, held), component_vertices(component, opened)
            cad_held, cad_open = transform_points(local_held, pose), transform_points(local_open, pose)
            world_held, world_open = transform_points(cad_held, self.pickup_T), transform_points(cad_open, self.pickup_T)
            component_min = float(np.min(np.concatenate((world_held, world_open))@self.table_plane[:3]+self.table_plane[3]))
            table_min = min(table_min, component_min)
            detail = dict(component=name, is_pad=component['is_pad'], table_minimum_clearance_m=component_min)
            if component['is_pad']:
                index = pad_lookup[name]
                bounds = self.calibration['usable_bounds_hand_zero_m'][index]+held*np.asarray(component['translation_per_opening'])
                edge_distance = np.minimum(centers[index, 1:]-bounds[0, 1:], bounds[1, 1:]-centers[index, 1:])
                if np.any(edge_distance < margin):
                    fail('pad_contact_geometry', f'Intended contact anchor violates pad-edge margin: {name}')
                try:
                    witnesses = self.target.surface_contacts(cad_held)
                    detail['contact_witness_count'] = len(witnesses)
                    if not witnesses:
                        complete_contact_regions = False
                        fail('pad_contact_geometry', 'No modeled inner-pad contact region: '+name)
                    area_tolerance = 8*NUMERICAL_TOL_M*float(np.max(bounds[1, 1:]-bounds[0, 1:]))
                    checks.setdefault('pad_boundary_support_area_tolerance_m2', {})[name] = area_tolerance
                    checks['pad_contact_region_query_clearance_m'] = 0.
                    aligned_witness = False
                    angular_errors = []
                    for witness in witnesses:
                        polygon = np.asarray(witness['polygon_world_m'])
                        local = (polygon-pose[:3, 3])@pose[:3, :3]
                        inner_x = centers[index, 0]
                        if (np.any(np.abs(local[:, 0]-inner_x) > 2*tol)
                                or np.any(local[:, 1:] < bounds[0, 1:]-2*tol)
                                or np.any(local[:, 1:] > bounds[1, 1:]+2*tol)):
                            fail('pad_contact_geometry', 'Part contact occurs outside usable inner pad face: '+name)
                        expected = -axis if index == 0 else axis
                        ids = np.full(len(polygon), int(witness['triangle_index']), dtype=int)
                        normals = self._normal_values(polygon, ids)
                        dots = normals@expected
                        aligned = bool(np.all(dots >= cosine))
                        angular_errors.append(float(np.rad2deg(np.arccos(np.clip(dots.min(), -1, 1)))))
                        aligned_witness |= aligned
                        if not aligned and _polygon_area(polygon) > area_tolerance:
                            fail('pad_contact_geometry', 'Surrounding contact geometry is not aligned with inner pad: '+name)
                        protected_points.append(polygon)
                        protected_polygons.append(polygon)
                        protected_face_ids.append(int(witness['triangle_index']))
                    detail['witness_maximum_normal_error_deg'] = max(angular_errors) if angular_errors else None
                    if not aligned_witness:
                        fail('pad_contact_geometry', 'No aligned inner-pad support witness: '+name)
                except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
                    complete_contact_regions = False
                    fail('pad_contact_geometry', str(exc), True)
                away = (-axis if index == 0 else axis)*self.config['collision_tolerance_m']
                cad_held, cad_open = cad_held+away, cad_open+away
            try:
                intersects = self.target.intersects(np.concatenate((cad_open, cad_held)))
                detail['target_closing_intersection'] = bool(intersects)
                if intersects:
                    fail('target_closing_collision', 'Gripper solid intersects target during modeled closing: '+name)
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                fail('target_closing_collision', str(exc), True)
            final_vertices = transform_points(transform_points(local_held, pose), self.final_T)
            receiver_reports = []
            for receiver_name, receiver in self.receivers:
                try:
                    intersects = receiver.intersects(final_vertices)
                    receiver_reports.append(dict(receiver=receiver_name, intersection=bool(intersects)))
                    if intersects:
                        fail('final_receiver_collision', 'Complete gripper intersects receiver at nominal final pose: '+receiver_name+'; component '+name)
                except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                    fail('final_receiver_collision', str(exc), True)
            detail['final_receiver_checks'] = receiver_reports
            component_reports.append(detail)
        for key in ['pad_contact_geometry', 'target_closing_collision', 'final_receiver_collision']:
            passed(key)
        checks['component_geometry'] = component_reports
        checks['whole_gripper_and_closing_clear'] = diagnostics['target_closing_collision']['status'] == 'pass'
        checks['nominal_final_pose_clear'] = diagnostics['final_receiver_collision']['status'] == 'pass'
        checks['table_minimum_clearance_m'] = float(table_min)
        if table_min < self.config['table_clearance_m']:
            fail('table_clearance', 'Complete gripper violates table clearance during closing')
        else:
            passed('table_clearance')
        checks['table_clear'] = diagnostics['table_clearance']['status'] == 'pass'
        try:
            points = np.unique(np.concatenate(protected_points), axis=0)
            protected = self.protected_contact_check(points, contact_polygons=protected_polygons,
                                                       contact_face_ids=protected_face_ids)
            if not isinstance(protected, dict) or 'eligible' not in protected or 'status' not in protected:
                raise ValueError('Protected-surface checker must return explicit eligibility and status')
            status = str(protected['status'])
            checks['protected_surface_status'] = status
            if not _available_status(status):
                raise ValueError('Protected mating-surface information unavailable: '+status)
            eligible, polygons = np.asarray(protected['eligible']), np.asarray(protected.get('polygon_eligible'))
            if eligible.shape != (len(points),) or eligible.dtype.kind != 'b':
                raise ValueError('Protected-surface eligibility needs one explicit boolean per checked point')
            if polygons.shape != (len(protected_polygons),) or polygons.dtype.kind != 'b':
                raise ValueError('Protected checker must explicitly classify each complete contact polygon')
            checks['protected_contact_polygons_checked'] = len(protected_polygons)
            if not eligible.all() or not polygons.all():
                fail('protected_surfaces', 'Inner-pad contact occupies a protected mating surface')
            elif complete_contact_regions:
                passed('protected_surfaces')
            else:
                diagnostics['protected_surfaces']['reasons'].append('Complete pad contact regions unavailable; known contacts alone cannot certify the full protected-surface check')
            checks['protected_surfaces_clear'] = diagnostics['protected_surfaces']['status'] == 'pass'
        except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            fail('protected_surfaces', str(exc), True)
        result.update(contact_inset_from_pad_tip_m=self.tip['contact_inset_m'],
            full_hand_distal_extent_beyond_pad_tip_m=self.tip['full_hand_distal_extent_beyond_pad_tip_m'])
        finish()
        if result['valid']:
            result.update(cad_T_hand=tip_pose.tolist(), cad_T_grip_site=pose.tolist(),
                world_T_hand_pickup=(self.pickup_T@tip_pose).tolist(),
                world_T_hand_final=(self.final_T@tip_pose).tolist(),
                world_T_grip_site_pickup=(self.pickup_T@pose).tolist(),
                world_T_grip_site_final=(self.final_T@pose).tolist(),
                approach_world=(self.pickup_T[:3, :3]@pose[:3, :3]@self.calibration['approach_hand']).tolist())
        return result
