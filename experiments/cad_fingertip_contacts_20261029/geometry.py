"""Validate actual distal finger contacts against the unchanged full solids.

The saved pad boxes are collision-only in this mode. Intended contact is allowed
only on verified inward-facing distal surfaces of the original finger convex
hulls. A surface witness is not a license for solid penetration elsewhere.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_selection_20261028.geometry import (
    _available_status, _faces_at_point, _polygon_area, component_vertices,
    rigid_transform, transform_points,
)
from experiments.cad_grasp_geometry_20261026.geometry_kernel import NUMERICAL_TOL_M
from experiments.cad_grasp_selection_20261029.hand_frame import FINGERTIP_FRAME
from experiments.cad_grasp_refinement_20261029.geometry import (
    GeometryValidator as SupportValidator, CATEGORIES,
)
from experiments.cad_grasp_refinement_20261029.support_geometry import SupportValidator as BoundaryTools


class GeometryValidator(SupportValidator):
    """One-pass independent checks for real finger-end contact candidates."""
    def __init__(self, *args, contact_calibration=None, tip_contact_calibration=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_calibration = (contact_calibration if contact_calibration is not None
                                    else tip_contact_calibration)
        self.finger_regions, self.finger_names = {}, []
        if self.errors:
            return
        try:
            from .contact_audit import make_contact_calibration
            expected = make_contact_calibration(self.hand,
                normal_angle_rad=self.config['normal_angle_rad'],
                edge_margin_m=self.config['pad_edge_margin_m'])
            if (not isinstance(self.contact_calibration, dict)
                    or self.contact_calibration.get('hand_geometry_sha256') != expected['hand_geometry_sha256']
                    or self.contact_calibration.get('calibration_sha256') != expected.get('calibration_sha256')
                    or not expected.get('calibration_sha256')):
                raise ValueError('Actual finger-contact calibration must be bound to this complete saved hand')
            # Derive the permission planes independently from the immutable
            # component vertices, never trust candidate-supplied face labels.
            fingers = sorted([component for component in self.hand['components']
                if not component['is_pad'] and
                abs(np.asarray(component['translation_per_opening'])[0]) > 1e-9],
                key=lambda item:np.asarray(item['translation_per_opening'])[0])
            if len(fingers) != 2:
                raise ValueError('Actual-contact mode needs exactly two original moving finger solids')
            self.distal_minimum_z = float(self.tip['usable_pad_tip_origin_grip_site_m'][2])
            cosine = np.cos(self.config['normal_angle_rad'])
            for index, component in enumerate(fingers):
                vertices = np.asarray(component['vertices_zero_m'], float)
                hull = ConvexHull(vertices)
                expected_normal = np.array([1., 0., 0.]) if index == 0 else np.array([-1., 0., 0.])
                allowed = hull.equations[:,:3]@expected_normal >= cosine
                allowed &= vertices[hull.simplices][...,2].max(axis=1) >= self.distal_minimum_z
                if not allowed.any():
                    raise ValueError('No real inward distal surface within the normal tolerance: '+component['name'])
                self.finger_names.append(component['name'])
                self.finger_regions[component['name']] = dict(index=index, component=component,
                    equations=hull.equations.copy(), allowed_equations=hull.equations[allowed].copy())
        except (ImportError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            self.errors.append(str(exc))

    def _contact_planes(self, name, points_grip, opening):
        """Find real allowed support planes containing the whole polygon.

        All points must be inside the original convex finger and share one
        allowed face plane. Since the witness is convex, this certifies its
        interior too; unrelated allowed faces cannot hide an interior crossing.
        """
        region = self.finger_regions[name]
        points = np.asarray(points_grip, float)-opening*np.asarray(
            region['component']['translation_per_opening'], float)
        tolerance = self.config['collision_tolerance_m']+2*NUMERICAL_TOL_M
        if (not len(points) or np.any(points[:,2] < self.distal_minimum_z-tolerance)
                or np.any(points@region['equations'][:,:3].T+region['equations'][:,3] > tolerance)):
            return np.empty((0,3))
        planes = region['allowed_equations']
        on = np.all(np.abs(points@planes[:,:3].T+planes[:,3]) <= tolerance, axis=0)
        return np.unique(planes[on,:3],axis=0)

    def _witness_contact(self,polygon,face_id,name,normal_grip,pose,opening):
        """Match normal policy to real area versus exact boundary contact.

        Only a verified round-side witness can receive this dimensionality
        correction. Its exact plane is the actual original finger hull plane,
        never a fictitious pad plane or the global jaw-closing axis.
        """
        polygon=np.asarray(polygon,float)
        ids=np.full(len(polygon),int(face_id),int)
        model=self.contact_model
        eligible=any(np.all(model._band_mask(polygon,ids,band))
                     for band in getattr(model,'_bands',[])) if model is not None else False
        normal_cad=pose[:3,:3]@normal_grip
        triangle=self.target.faces[int(face_id)]
        source=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);source/=np.linalg.norm(source)
        parallel=abs(float(source@normal_cad))>=1-1e-12
        correction=False
        if eligible and len(polygon)>=3 and not parallel:
            region=self.finger_regions[name]
            equations=region['allowed_equations']
            same=np.linalg.norm(equations[:,:3]-normal_grip,axis=1)<1e-12
            equation=equations[np.flatnonzero(same)[0]]
            motion=np.asarray(region['component']['translation_per_opening'])
            offset=float(normal_cad@pose[:3,3]+opening*(normal_grip@motion)-equation[3])
            if np.max(np.abs(polygon@normal_cad-offset))<=8*NUMERICAL_TOL_M:
                boundary=BoundaryTools._plane_intersection(polygon,normal_cad,offset)
                if len(boundary) and _polygon_area(boundary)<=NUMERICAL_TOL_M**2:
                    polygon=boundary;ids=np.full(len(polygon),int(face_id),int);correction=True
        span=float(np.max(np.ptp(np.asarray(self.finger_regions[name]['component']['vertices_zero_m']),axis=0)))
        area_threshold=8*NUMERICAL_TOL_M*span
        finite_area_support=eligible and parallel and _polygon_area(polygon)>area_threshold
        normals=(np.repeat(source[None],len(polygon),axis=0) if finite_area_support
                 else self._normal_values(polygon,ids))
        return polygon,normals,dict(exact_boundary_correction=correction,
            original_facet_area_support=finite_area_support,area_threshold_m2=area_threshold)

    def validate(self, candidate):
        diagnostics = {key:dict(status='not_evaluated', reasons=[], failed=False, unavailable=False)
                       for key in CATEGORIES}
        checks = {}
        result = dict(status='invalid', valid=False, reasons=[], checks=checks,
            diagnostics=diagnostics, diagnostic_method='one_pass_independent_geometry_checks',
            closing_clearance_method='continuous_linear_convex_component_sweep',
            closing_discrete_samples=0, tolerances=dict(self.config),
            hand_pose_reference=FINGERTIP_FRAME,
            actual_contact_mode='original_distal_inward_finger_hull_surfaces',
            surface_normal_model=('original_mesh_facets' if self.contact_model is None
                                  else 'verified_round_side_model_with_original_facet_fallback'),
            unchecked=['robot_arm_reachability','full_approach_path','insertion_path',
                       'release','physical_grasp_stability'])
        def passed(category):
            if diagnostics[category]['status'] == 'not_evaluated': diagnostics[category]['status'] = 'pass'
        def fail(category, reason, unavailable=False):
            entry = diagnostics[category]
            # Keep candidate reasons compact; separate witness counts live in
            # component reports, rather than duplicating one sentence per face.
            if reason not in entry['reasons']:
                entry['reasons'].append(str(reason)); result['reasons'].append(str(reason))
            entry['unavailable' if unavailable else 'failed'] = True
            entry['status'] = 'unavailable' if entry['unavailable'] else 'fail'
        def finish():
            statuses = [entry['status'] for entry in diagnostics.values()]
            result['valid'] = all(status == 'pass' for status in statuses)
            result['status'] = 'valid' if result['valid'] else 'unavailable_input' if 'unavailable' in statuses else 'invalid'
            result['diagnostic_failure_categories'] = [key for key,e in diagnostics.items() if e['failed']]
            result['diagnostic_unavailable_categories'] = [key for key,e in diagnostics.items() if e['unavailable']]
            return result
        if self.errors:
            fail('candidate_input','; '.join(self.errors),True); return finish()
        try:
            if candidate.get('hand_pose_reference') != FINGERTIP_FRAME:
                raise ValueError('Candidate must explicitly name the calibrated physical fingertip frame')
            tip_pose = rigid_transform(candidate['cad_T_hand'],'fingertip pose')
            pose = tip_pose@self.tip['fingertip_T_grip_site']
            if ('cad_T_grip_site' not in candidate or not np.allclose(
                    rigid_transform(candidate['cad_T_grip_site']),pose,atol=1e-9,rtol=0)):
                raise ValueError('Actual fingertip/grip-site transforms must be explicit and consistent')
            held, opened = float(candidate['holding_opening_m']), float(candidate['pre_opening_m'])
            if candidate.get('contact_calibration_sha256') != self.contact_calibration['calibration_sha256']:
                raise ValueError('Candidate calibration hash disagrees with the complete-hand contact calibration')
            if not np.isfinite([held,opened]).all():
                raise ValueError('Actual-contact opening commands must be finite')
            fitted = candidate.get('first_contact_fit',{})
            if not isinstance(fitted,dict) or fitted.get('status') != 'fitted':
                passed('candidate_input')
                fail('pad_contact_geometry','Actual first-contact geometry fit did not produce both real finger contacts: '+str(fitted.get('status','missing')))
                result['first_contact_fit']=fitted
                return finish()
            contacts = np.asarray(candidate['contacts_cad_m'],float)
            local_contacts = np.asarray(candidate['actual_contacts_hand_m'],float)
            if (contacts.shape != (2,3) or local_contacts.shape != (2,3)
                    or not np.isfinite(contacts).all() or not np.isfinite(local_contacts).all()
                    or not np.isfinite([held,opened]).all()):
                raise ValueError('Actual-contact candidate needs two finite CAD and grip-site contact points')
            if candidate.get('actual_contact_component_names') != self.finger_names:
                raise ValueError('Actual contact points must identify the left and right original finger solids in order')
            passed('candidate_input')
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
            fail('candidate_input',str(exc),True); return finish()
        low,high = self.calibration['opening_limits_m']
        checks['opening_feasible'] = bool(low <= held <= opened <= high)
        if checks['opening_feasible']: passed('opening_limits')
        else: fail('opening_limits','Holding or pre-opening outside calibrated limits')
        checks['geometry_extrapolated_outside_opening_limits'] = not checks['opening_feasible']
        reconstructed = transform_points(local_contacts,pose)
        error = np.linalg.norm(reconstructed-contacts,axis=1)
        checks['actual_contact_reconstruction_error_m'] = error.tolist()
        if np.any(error > self.config['contact_tolerance_m']):
            fail('contact_reconstruction','Actual grip-site contact points do not reconstruct intended CAD contacts')
        contact_planes=[]
        for index,name in enumerate(self.finger_names):
            normals = self._contact_planes(name,local_contacts[index:index+1],held)
            contact_planes.append(normals)
            if not len(normals):
                fail('contact_reconstruction','Declared contact is not on an actual permitted distal finger surface: '+name)
        passed('contact_reconstruction')
        cosine=np.cos(self.config['normal_angle_rad'])
        normal_errors=[]
        try:
            for index,point in enumerate(contacts):
                ids=_faces_at_point(self.target,point,self.config['contact_tolerance_m'])
                if not len(ids) or not len(contact_planes[index]):
                    fail('surface_normals','Actual intended contact lacks original target/finger surface support')
                    normal_errors.append(None);continue
                normals=self._normal_values(np.repeat(point[None],len(ids),axis=0),ids)
                opposite=-contact_planes[index]@pose[:3,:3].T
                dots=normals@opposite.T
                normal_errors.append(float(np.rad2deg(np.arccos(np.clip(dots.max(),-1,1)))))
                if not np.any(dots>=cosine):
                    fail('surface_normals',f'Actual contact {index} target normal is not opposed to its real finger surface')
            passed('surface_normals')
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            fail('surface_normals',str(exc),True)
        checks['intended_contact_normal_minimum_error_deg']=normal_errors
        checks['actual_contacts_hand_m']=local_contacts.tolist()
        checks['actual_contact_distance_from_physical_tip_m']=(
            self.tip['fingertip_origin_grip_site_m'][2]-local_contacts[:,2]).tolist()
        points,polygons,face_ids=[contacts],[],[]
        full_contacts=True
        support_seen={name:False for name in self.finger_names}
        table_min=np.inf
        reports=[]
        for component in self.hand['components']:
            name=component['name'];finger=name in self.finger_regions
            local_held,local_open=component_vertices(component,held),component_vertices(component,opened)
            cad_held,cad_open=transform_points(local_held,pose),transform_points(local_open,pose)
            world=np.concatenate((transform_points(cad_held,self.pickup_T),transform_points(cad_open,self.pickup_T)))
            component_min=float(np.min(world@self.table_plane[:3]+self.table_plane[3]));table_min=min(table_min,component_min)
            detail=dict(component=name,is_pad=component['is_pad'],actual_contact_component=finger,
                        table_minimum_clearance_m=component_min)
            # Every component is queried, including saved pads/palm. Boundary
            # contact outside the authorized real finger-end regions is invalid.
            try:
                witnesses=self.target.surface_contacts(cad_held)
                detail['contact_witness_count']=len(witnesses)
                bad_geometry=bad_normals=boundary_corrections=facet_area_supports=0
                for witness in witnesses:
                    polygon=np.asarray(witness['polygon_world_m'],float)
                    grip=(polygon-pose[:3,3])@pose[:3,:3]
                    if not finger:
                        bad_geometry+=1
                    else:
                        planes=self._contact_planes(name,grip,held)
                        if not len(planes):bad_geometry+=1
                        else:
                            aligned=False
                            for plane in planes:
                                corrected,normals,policy=self._witness_contact(polygon,witness['triangle_index'],name,plane,pose,held)
                                opposite=-pose[:3,:3]@plane
                                if np.all(normals@opposite>=cosine):
                                    polygon=corrected;aligned=True
                                    boundary_corrections+=int(policy['exact_boundary_correction'])
                                    facet_area_supports+=int(policy['original_facet_area_support'])
                                    break
                            if aligned:support_seen[name]=True
                            elif _polygon_area(polygon)>8*NUMERICAL_TOL_M*float(np.max(np.ptp(local_held,axis=0))):bad_normals+=1
                    points.append(polygon);polygons.append(polygon);face_ids.append(int(witness['triangle_index']))
                detail.update(unauthorized_surface_witnesses=bad_geometry,
                              misaligned_surface_witnesses=bad_normals,
                              exact_finger_plane_boundary_corrections=boundary_corrections,
                              source_facet_area_support_witnesses=facet_area_supports)
                if bad_geometry:
                    fail('pad_contact_geometry','Contact occurs outside the actual permitted finger-end surface: '+name)
                if bad_normals:
                    fail('pad_contact_geometry','Target contact is not aligned with the actual finger surface: '+name)
            except (ValueError,TypeError,KeyError,IndexError,np.linalg.LinAlgError) as exc:
                full_contacts=False;fail('pad_contact_geometry',str(exc),True)
            if finger:
                index=self.finger_regions[name]['index']
                outward=pose[:3,:3]@np.array([-1.,0,0] if index==0 else [1.,0,0])
                # A declared numerical retreat permits only boundary contact.
                # All original solid vertices, including the whole finger body,
                # remain in the continuous closing-volume collision query.
                cad_held=cad_held+outward*self.config['collision_tolerance_m']
                cad_open=cad_open+outward*self.config['collision_tolerance_m']
            try:
                hit=bool(self.target.intersects(np.concatenate((cad_open,cad_held))))
                detail['target_closing_intersection']=hit
                if hit: fail('target_closing_collision','Complete solid intersects target during closing: '+name)
            except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
                fail('target_closing_collision',str(exc),True)
            final=transform_points(transform_points(local_held,pose),self.final_T)
            receiver_reports=[]
            for receiver_name,receiver in self.receivers:
                try:
                    hit=bool(receiver.intersects(final))
                    receiver_reports.append(dict(receiver=receiver_name,intersection=hit))
                    if hit: fail('final_receiver_collision','Complete gripper intersects receiver at nominal final pose: '+receiver_name+'; component '+name)
                except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
                    fail('final_receiver_collision',str(exc),True)
            detail['final_receiver_checks']=receiver_reports;reports.append(detail)
        for name,seen in support_seen.items():
            if not seen:
                full_contacts=False;fail('pad_contact_geometry','No aligned actual finger-end support witness: '+name)
        for key in ['pad_contact_geometry','target_closing_collision','final_receiver_collision']:passed(key)
        checks.update(component_geometry=reports,actual_finger_support=support_seen,
            table_minimum_clearance_m=float(table_min),
            whole_gripper_and_closing_clear=diagnostics['target_closing_collision']['status']=='pass',
            nominal_final_pose_clear=diagnostics['final_receiver_collision']['status']=='pass')
        if table_min<self.config['table_clearance_m']:fail('table_clearance','Complete gripper violates table clearance during closing')
        else:passed('table_clearance')
        checks['table_clear']=diagnostics['table_clearance']['status']=='pass'
        try:
            known=np.unique(np.concatenate(points),axis=0)
            protected=self.protected_contact_check(known,contact_polygons=polygons,contact_face_ids=face_ids)
            if not isinstance(protected,dict) or not _available_status(protected.get('status','')):
                raise ValueError('Protected mating-surface information unavailable')
            eligible,polygon_clear=np.asarray(protected.get('eligible')),np.asarray(protected.get('polygon_eligible'))
            if (eligible.shape!=(len(known),) or eligible.dtype.kind!='b' or
                    polygon_clear.shape!=(len(polygons),) or polygon_clear.dtype.kind!='b'):
                raise ValueError('Protected checker must explicitly classify points and complete actual-contact polygons')
            checks['protected_surface_status']=protected['status']
            checks['protected_contact_polygons_checked']=len(polygons)
            if not eligible.all() or not polygon_clear.all():fail('protected_surfaces','Actual contact occupies a protected mating surface')
            elif full_contacts:passed('protected_surfaces')
            else:diagnostics['protected_surfaces']['reasons'].append('Complete actual finger support unavailable; known points cannot certify it')
        except (ValueError,TypeError,KeyError,IndexError,np.linalg.LinAlgError) as exc:
            fail('protected_surfaces',str(exc),True)
        checks['protected_surfaces_clear']=diagnostics['protected_surfaces']['status']=='pass'
        result.update(actual_contact_distance_from_physical_tip_m=checks['actual_contact_distance_from_physical_tip_m'],
            tip_contact_calibration_hand_sha256=self.contact_calibration['hand_geometry_sha256'])
        finish()
        if result['valid']:
            result.update(cad_T_hand=tip_pose.tolist(),cad_T_grip_site=pose.tolist(),
                world_T_hand_pickup=(self.pickup_T@tip_pose).tolist(),
                world_T_hand_final=(self.final_T@tip_pose).tolist(),
                world_T_grip_site_pickup=(self.pickup_T@pose).tolist(),
                world_T_grip_site_final=(self.final_T@pose).tolist(),
                approach_world=(self.pickup_T[:3,:3]@pose[:3,2]).tolist())
        return result
