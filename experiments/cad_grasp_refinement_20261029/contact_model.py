"""Verified smooth side normals, without replacing the complete CAD mesh.

Only a complete, constant-radius set of straight axial faces can establish a
cylinder. Knurls, chamfers, caps, partial arcs and unverified parts retain their
original triangle normals. These normals are a contact model, never a collision
model: ray intersections and finite-hand checks still use every CAD triangle.
"""
from __future__ import annotations

import hashlib
import numpy as np


def mesh_binding(vertices, faces):
    return dict(mesh_vertices_sha256=hashlib.sha256(
        np.ascontiguousarray(vertices, dtype='<f8').tobytes()).hexdigest(),
        mesh_faces_sha256=hashlib.sha256(
        np.ascontiguousarray(faces, dtype='<i8').tobytes()).hexdigest())


class ContactModel:
    def __init__(self, vertices, faces, *, part=None):
        self.vertices = np.array(vertices, dtype=float, copy=True)
        if np.asarray(faces).dtype.kind not in 'iu':
            raise ValueError('Contact model requires original integer CAD face indices')
        self.faces = np.array(faces, dtype=np.int64, copy=True)
        if (self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,)
                or not np.isfinite(self.vertices).all() or self.faces.ndim != 2
                or self.faces.shape[1:] != (3,) or not len(self.faces)
                or self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError('Contact model requires finite complete CAD triangles')
        self.triangles = self.vertices[self.faces]
        cross = np.cross(self.triangles[:, 1]-self.triangles[:, 0],
                         self.triangles[:, 2]-self.triangles[:, 0])
        length = np.linalg.norm(cross, axis=1)
        self.normals = np.divide(cross, length[:, None], out=np.zeros_like(cross),
                                 where=length[:, None] > 0)
        self.verified_cylinder_bands = []
        self._bands = []
        self.provenance = dict(mesh_binding(self.vertices, self.faces), part=part,
            method='constant-radius straight CAD-side verification; radial normals only inside verified bands',
            collision_geometry='unchanged complete original CAD triangles',
            status='original_triangle_normals', verified_cylinder_bands=[])

    def _query(self, points, face_ids):
        points, ids = np.asarray(points, float), np.asarray(face_ids)
        if (points.ndim != 2 or points.shape[1:] != (3,)
                or not np.isfinite(points).all() or ids.shape != (len(points),)
                or ids.dtype.kind not in 'iu' or (len(ids) and
                (ids.min() < 0 or ids.max() >= len(self.faces)))):
            raise ValueError('Contact normals need finite(N,3) points and original integer face IDs')
        return points, ids.astype(int, copy=False)

    def _band_mask(self, points, ids, band):
        t = self.triangles[ids]
        a, b = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
        q = points-t[:, 0]
        aa, ab, bb = np.sum(a*a, 1), np.sum(a*b, 1), np.sum(b*b, 1)
        aq, bq = np.sum(a*q, 1), np.sum(b*q, 1)
        det = aa*bb-ab*ab
        u = np.divide(bb*aq-ab*bq, det, out=np.full(len(ids), np.inf), where=det>0)
        v = np.divide(aa*bq-ab*aq, det, out=np.full(len(ids), np.inf), where=det>0)
        plane = np.abs(np.sum(q*self.normals[ids], 1))
        z = (points-band['origin'])@band['axis']
        radial = points-band['origin']-z[:, None]*band['axis']
        radius = np.linalg.norm(radial, axis=1)
        return (band['face_mask'][ids] & (plane <= 1e-8)
            & (u >= -1e-6) & (v >= -1e-6) & (u+v <= 1+1e-6)
            & (z >= band['z_min']-1e-8) & (z <= band['z_max']+1e-8)
            & (radius >= band['minimum_mesh_radius']-1e-8)
            & (radius <= band['radius']+1e-8))

    def contact_normals(self, points, face_ids):
        points, ids = self._query(points, face_ids)
        result = self.normals[ids].copy()
        for band in self._bands:
            mask = self._band_mask(points, ids, band)
            q = points[mask]-band['origin']
            radial = q-np.outer(q@band['axis'], band['axis'])
            result[mask] = radial/np.linalg.norm(radial, axis=1)[:, None]
        if len(result) and (not np.isfinite(result).all() or
                not np.allclose(np.linalg.norm(result, axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError('Requested contact lies on a degenerate CAD face')
        return result

    def snap_origins_to_axial_edges(self, points, face_ids):
        """Use an actual source edge where a smooth tangent is a support plane.

        This is a deterministic local contact construction after the bounded
        area draw, not a new rejection/resampling loop or gripper-depth offset.
        The endpoint remains on its original triangle at the sampled axial level.
        """
        points, ids = self._query(points, face_ids)
        result, snapped = points.copy(), np.zeros(len(points), dtype=bool)
        for band in self._bands:
            eligible = self._band_mask(points, ids, band)
            for k in np.flatnonzero(eligible):
                triangle = self.triangles[ids[k]]
                z = float((points[k]-band['origin'])@band['axis'])
                choices = []
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    delta = triangle[j]-triangle[i]
                    height = float(delta@band['axis'])
                    if abs(height) < 1e-9 or np.linalg.norm(delta-height*band['axis']) > 1e-8:
                        continue
                    fraction = (z-(triangle[i]-band['origin'])@band['axis'])/height
                    if -1e-8 <= fraction <= 1+1e-8:
                        value = triangle[i]+np.clip(fraction, 0, 1)*delta
                        choices.append((np.linalg.norm(value-points[k]), value))
                if choices:
                    result[k] = min(choices, key=lambda x:x[0])[1]
                    snapped[k] = True
        return result, snapped


def build_contact_model(vertices, faces, sampling_contact_region=None, *, part=None,
                        axis_point_cad_m=None, axis_direction_cad=None):
    """Verify a labelled round side before replacing its facet normals.

    NIST opt-in names are the three RGOCG round pins and BNC. Synthetic callers
    can use part=None with an explicit axis. A failed fit is recorded and leaves
    the original normal policy intact; no approximate cylinder is imposed.
    """
    model = ContactModel(vertices, faces, part=part)
    if part is not None and not (part.startswith('RGOCG') or part == 'BNC_Male'):
        model.provenance['verification_reason'] = 'part is outside the declared circular-side family'
        return model
    if axis_point_cad_m is None or axis_direction_cad is None:
        model.provenance['verification_reason'] = 'explicit CAD axis unavailable'
        return model
    point, axis = np.asarray(axis_point_cad_m, float), np.asarray(axis_direction_cad, float)
    if (point.shape != (3,) or axis.shape != (3,) or not np.isfinite(point).all()
            or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-8):
        raise ValueError('Cylinder verification needs a finite explicit unit CAD axis')
    if sampling_contact_region is None:
        model.provenance['verification_reason'] = 'explicit allowed CAD side band unavailable'
        return model
    if any(sampling_contact_region.get(k) != value for k, value in mesh_binding(vertices, faces).items()):
        raise ValueError('Contact region belongs to a different complete CAD mesh')
    source_ids = np.asarray(sampling_contact_region['source_face_ids'], int)
    patches = np.asarray(sampling_contact_region['sampling_triangles_cad_m'], float)
    ids = np.unique(source_ids)
    ids = ids[np.abs(model.normals[ids]@axis) < 1e-5]
    # A circle fitted to the triangle vertices, not triangle centroids, recovers
    # the underlying CAD radius without replacing its inscribed polygon surface.
    points = np.unique(model.triangles[ids].reshape(-1, 3), axis=0)
    if len(points) < 12:
        model.provenance['verification_reason'] = 'insufficient straight side geometry'
        return model
    reference = np.eye(3)[np.argmin(abs(axis))]
    e1 = np.cross(axis, reference); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    xy = np.column_stack(((points-point)@e1, (points-point)@e2))
    design = np.column_stack((2*xy, np.ones(len(xy))))
    fit, _, rank, _ = np.linalg.lstsq(design, np.sum(xy*xy, axis=1), rcond=None)
    if rank != 3:
        model.provenance['verification_reason'] = 'circle fit is rank deficient'
        return model
    center = fit[:2]
    radius = np.sqrt(max(0., fit[2]+center@center))
    residual = np.abs(np.linalg.norm(xy-center, axis=1)-radius)
    angles = np.sort(np.unique(np.round(np.arctan2(xy[:,1]-center[1], xy[:,0]-center[0]), 9)))
    gap = float(np.diff(np.r_[angles, angles[0]+2*np.pi]).max())
    model.provenance['circle_fit_max_vertex_residual_m'] = float(residual.max())
    model.provenance['circle_fit_max_angular_gap_rad'] = gap
    if radius <= 1e-6 or residual.max() > 1e-8 or len(angles) < 12 or gap > np.pi/4:
        model.provenance['verification_reason'] = 'allowed straight faces do not verify one complete constant-radius cylinder'
        return model
    origin = point+center[0]*e1+center[1]*e2
    centers = model.triangles[ids].mean(axis=1)-origin
    radial_centers = centers-np.outer(centers@axis, axis)
    outward = np.einsum('ij,ij->i', model.normals[ids], radial_centers)
    if np.any(outward <= 0):
        model.provenance['verification_reason'] = 'straight side normals are not all outward from the fitted cylinder'
        return model
    selected_patches = patches[np.isin(source_ids, ids)]
    z = (selected_patches.reshape(-1, 3)-origin)@axis
    if np.ptp(z) <= 1e-6:
        model.provenance['verification_reason'] = 'allowed cylindrical side has no finite axial band'
        return model
    face_mask = np.zeros(len(model.faces), bool); face_mask[ids] = True
    band = dict(origin=origin, axis=axis.copy(), radius=float(radius), z_min=float(z.min()),
        z_max=float(z.max()), face_mask=face_mask,
        minimum_mesh_radius=float(radius*np.cos(gap/2)))
    record = dict(axis_point_cad_m=origin.tolist(), axis_direction_cad=axis.tolist(),
        radius_m=float(radius), axial_bounds_m=[band['z_min'], band['z_max']],
        source_face_ids=ids.tolist(), maximum_vertex_fit_error_m=float(residual.max()),
        maximum_circumferential_gap_rad=gap, verification_tolerance_m=1e-8,
        contact_surface='original source triangles; smooth radial normal only',
        construction='sampled axial level on an actual source axial edge')
    model._bands.append(band)
    model.verified_cylinder_bands.append(record)
    model.provenance.update(status='verified_cylindrical_side_normals',
        verification_reason='complete constant-radius axial side verified against original CAD vertices',
        verified_cylinder_bands=model.verified_cylinder_bands)
    return model
