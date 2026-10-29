"""Supplemental contact-witness consistency correction; frozen run unchanged.

The complete collision/contact surface is still the original triangle mesh.
An actual positive-area contact on one verified cylinder facet therefore uses
that facet's support normal. Point contacts and numerical boundary/line contacts
use the verified smooth cylinder normal, as in the original refined validator.
The one-degree tolerance, all geometry and every other check remain unchanged.
"""
from __future__ import annotations

import numpy as np

from experiments.cad_grasp_geometry_20261026.geometry_kernel import NUMERICAL_TOL_M
from experiments.cad_grasp_selection_20261028.geometry import _polygon_area
from .geometry import GeometryValidator


class SupportValidator(GeometryValidator):
    """Match the normal to the actual dimensionality of a CAD support witness."""

    @staticmethod
    def _plane_intersection(polygon, axis, offset):
        """Intersect a convex witness with the exact pad plane, without a slab."""
        signed = polygon@axis-offset
        points = []
        epsilon = NUMERICAL_TOL_M*1e-3
        for index, a in enumerate(polygon):
            j = (index+1) % len(polygon)
            if abs(signed[index]) <= epsilon:
                points.append(a-signed[index]*axis)
            if signed[index]*signed[j] < 0:
                points.append(a+signed[index]/(signed[index]-signed[j])*(polygon[j]-a))
        distinct = []
        for point in points:
            if not any(np.linalg.norm(point-previous) <= epsilon for previous in distinct):
                distinct.append(point)
        return np.asarray(distinct, float).reshape(-1,3)

    def _boundary_witnesses(self, witnesses, convex_vertices):
        """Undo only numerical ribbons around a verified true line contact."""
        axis = self._support_closing_axis
        model = self.contact_model
        if axis is None or not getattr(model, '_bands', []):
            return witnesses
        projection = np.asarray(convex_vertices)@axis
        # These queries are made only for the two actual usable pad solids.
        # Their inward side is the extreme facing the gap midpoint.
        center = float(projection.mean())
        offset = float(projection.max() if center < self._support_gap_midpoint else projection.min())
        guard = 8*NUMERICAL_TOL_M
        corrected = []
        for witness in witnesses:
            polygon = np.asarray(witness['polygon_world_m'],float)
            ids = np.full(len(polygon),int(witness['triangle_index']),int)
            eligible = any(np.all(model._band_mask(polygon,ids,band)) for band in model._bands)
            triangle = self.target.faces[ids[0]]
            normal = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
            normal /= np.linalg.norm(normal)
            # Actual finite-area penetration, contacts at outer pad faces and
            # coplanar real area support are never collapsed to a boundary.
            if (eligible and len(polygon)>=3 and abs(normal@axis)<1-1e-12
                    and np.max(np.abs(polygon@axis-offset)) <= guard):
                boundary = self._plane_intersection(polygon,axis,offset)
                if len(boundary) and _polygon_area(boundary) <= NUMERICAL_TOL_M**2:
                    witness = dict(witness,polygon_world_m=boundary.tolist(),
                        point_world_m=boundary.mean(axis=0).tolist())
                    self._boundary_dimension_queries += 1
            corrected.append(witness)
        return corrected

    def _normal_values(self, points, face_ids):
        values = super()._normal_values(points, face_ids)
        model = self.contact_model
        points, ids = np.asarray(points, float), np.asarray(face_ids, int)
        if (model is None or not getattr(model, 'verified_cylinder_bands', [])
                or len(points) < 3 or not np.all(ids == ids[0])):
            return values
        # This is the same numerical area guard used by the inherited pad check
        # to distinguish a boundary witness from finite-area surface contact.
        bounds = self.calibration['usable_bounds_hand_zero_m']
        span = float(np.max(bounds[:, 1, 1:]-bounds[:, 0, 1:]))
        if _polygon_area(points) <= 8*NUMERICAL_TOL_M*span:
            return values
        # No analytic override beyond the geometry-verified source side band.
        # In particular, a polygon crossing a cap, chamfer or protected band end
        # cannot borrow this support rule from one nearby circular contact.
        for band in getattr(model, '_bands', []):
            if np.all(model._band_mask(points, ids, band)):
                triangle = self.target.faces[ids[0]]
                normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                normal /= np.linalg.norm(normal)
                # The clipping kernel thickens an exact line contact by its
                # numerical tolerance. Such a narrow strip is not a physical
                # flat-area contact. Genuine positive-area contact requires the
                # source facet and the usable pad face to be parallel; the
                # actual clipped witness already supplies their shared points.
                axis = self._support_closing_axis
                if axis is None or abs(normal@axis) < 1-1e-12:
                    return values
                self._positive_area_support_queries += 1
                return np.repeat(normal[None], len(points), axis=0)
        return values

    def validate(self, candidate):
        self._positive_area_support_queries = 0
        self._boundary_dimension_queries = 0
        self._support_closing_axis = None
        self._support_gap_midpoint = 0.
        if not self.errors:
            try:
                pose = np.asarray(candidate['cad_T_hand'], float)@self.tip['fingertip_T_grip_site']
                if pose.shape == (4,4) and np.isfinite(pose).all():
                    self._support_closing_axis = pose[:3,:3]@self.calibration['closing_hand']
                    held = float(candidate['holding_opening_m'])
                    local = self.calibration['centers_zero_m']+held*self.calibration['centers_per_opening']
                    middle = pose[:3,:3]@local.mean(axis=0)+pose[:3,3]
                    self._support_gap_midpoint = float(middle@self._support_closing_axis)
            except (ValueError, TypeError, KeyError):
                pass  # The inherited candidate-input check reports the error.
        model = self.contact_model
        bands = getattr(model, 'verified_cylinder_bands', []) if model is not None else []
        original_query = getattr(self.target,'surface_contacts',None) if not self.errors else None
        if bands and original_query is not None:
            def query(vertices,*args,**kwargs):
                return self._boundary_witnesses(original_query(vertices,*args,**kwargs),vertices)
            self.target.surface_contacts = query
        try:
            result = super().validate(candidate)
        finally:
            if bands and original_query is not None:
                self.target.surface_contacts = original_query
        # Preserve baseline and all unverified-part records exactly.
        if not bands:
            return result
        sag = [float(band['radius_m']*(1-np.cos(band['maximum_circumferential_gap_rad']/2)))
               for band in bands]
        result['support_normal_policy'] = dict(
            method='actual source-facet normal for positive-area coplanar verified cylinder witnesses; smooth radial normal for intended points and boundary/line witnesses',
            positive_area_facet_queries=self._positive_area_support_queries,
            exact_inner_plane_boundary_corrections=self._boundary_dimension_queries,
            boundary_ribbon_guard_m=float(8*NUMERICAL_TOL_M),
            positive_area_threshold='existing8*NUMERICAL_TOL_M*maximum_usable_pad_side_length',
            facet_pad_parallelism_numerical_dot_tolerance=1e-12,
            normal_angle_rad=float(self.config['normal_angle_rad']),
            complete_original_mesh_unchanged=True,
            cylinder_to_inscribed_mesh_maximum_sag_m=sag,
            physical_contact_accuracy_validated=False,
            qualification='mesh-consistent geometric validity does not establish smooth-CAD or physical contact accuracy')
        return result
