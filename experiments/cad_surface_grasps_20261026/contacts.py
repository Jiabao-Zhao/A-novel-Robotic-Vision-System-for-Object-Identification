"""Finite pad near-contact regions, with paired rather than arbitrary witnesses.

The 25 micrometre band is geometric tolerance, not rubber deformation, pressure,
or a claim that this whole region carries load. Native triangles stay unchanged.
"""
import numpy as np

from experiments.cad_grasp_geometry_20261026.geometry_kernel import _clip_triangle
from experiments.cad_grasp_geometry_20261026.hand_model import vertices as hand_vertices


def signed_area(points):
    p = np.asarray(points)
    return float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))/2)


def intersect_polygons(first, second):
    """Intersection of two convex Y/Z polygons; empty for an edge-only touch."""
    polygon = np.asarray(first,dtype=float)
    clip = np.asarray(second,dtype=float)
    if len(polygon)<3 or len(clip)<3:
        return np.empty((0,2))
    if signed_area(clip)<0:
        clip = clip[::-1]
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        if not len(polygon):
            return np.empty((0,2))
        edge = b-a
        # Left of every CCW edge: the same boundary defines classification and
        # interpolation. The units here are square metres.
        d = edge[0]*(polygon[:,1]-a[1])-edge[1]*(polygon[:,0]-a[0])+1e-16
        inside = d>=0
        out = []
        for i in range(len(polygon)):
            j = i-1
            if inside[i]!=inside[j]:
                t = np.clip(d[j]/(d[j]-d[i]),0.,1.)
                out.append(polygon[j]+t*(polygon[i]-polygon[j]))
            if inside[i]:
                out.append(polygon[i])
        polygon = np.asarray(out).reshape(-1,2)
    return polygon


def closest_point(polygon, point):
    p = np.asarray(point)
    poly = np.asarray(polygon)
    sign = 1. if signed_area(poly)>=0 else -1.
    edge = np.roll(poly,-1,axis=0)-poly
    d = sign*(edge[:,0]*(p[1]-poly[:,1])-edge[:,1]*(p[0]-poly[:,0]))
    if np.all(d>=-1e-14):
        return p
    scale = np.einsum('ij,ij->i',edge,edge)
    t = np.clip(np.einsum('ij,ij->i',p-poly,edge)/np.maximum(scale,1e-30),0.,1.)
    candidates = poly+t[:,None]*edge
    return candidates[np.argmin(np.sum((candidates-p)**2,axis=1))]


def pad_regions(candidate, collider, pads, normal_tolerance_deg, band_m=.000025, edge_margin_m=.00025,triangle_ids=None):
    pose = np.asarray(candidate['world_T_grip'])
    local_triangles = (collider.faces-pose[:3,3])@pose[:3,:3]
    cross = np.cross(local_triangles[:,1]-local_triangles[:,0],local_triangles[:,2]-local_triangles[:,0])
    normals = cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
    cosine = np.cos(np.deg2rad(normal_tolerance_deg))
    near = np.abs(normals[:,0])>=cosine
    if triangle_ids is not None:
        near &= np.isin(np.arange(len(normals)),triangle_ids)
    records = []
    for pad,opening in zip(pads,candidate['first_contact_openings_m']):
        pv = hand_vertices(pad,opening)
        negative = float(np.asarray(pad['translation_per_opening'])[0])<0
        inner = float(pv[:,0].max() if negative else pv[:,0].min())
        low,high = pv.min(axis=0),pv.max(axis=0)
        # The band is separate from the finite pad Y/Z boundary guard.
        low[0],high[0] = inner-band_m,inner+band_m
        low[1:] += edge_margin_m
        high[1:] -= edge_margin_m
        planes = np.vstack((np.c_[np.eye(3),-high],np.c_[-np.eye(3),low]))
        active = near & np.all(local_triangles.max(axis=1)>=low,axis=1) & np.all(local_triangles.min(axis=1)<=high,axis=1)
        polygons = []
        for index in np.flatnonzero(active):
            polygon = _clip_triangle(local_triangles[index],planes)
            if len(polygon)>=3 and abs(signed_area(polygon[:,1:]))>1e-14:
                polygons.append(dict(triangle_index=int(index),polygon_local_m=polygon.tolist(),
                         normal_local=normals[index].tolist(),
                         pad_normal_error_deg=float(np.degrees(np.arccos(np.clip(abs(normals[index,0]),0,1))))))
        records.append(dict(pad=pad['name'],inner_plane_x_m=inner,polygons=polygons))
    return records,local_triangles


def inspect_contacts(candidate,collider,pads,normal_tolerance_deg=1.,cylinder=None):
    """Add stricter surface checks after the unchanged interference checker."""
    actual_errors = [float(np.degrees(np.arccos(np.clip(c['normal_alignment'],0,1)))) for c in candidate['actual_contacts']]
    evidence = dict(actual_pad_normal_errors_deg=actual_errors,
                    surface_normal_tolerance_deg=normal_tolerance_deg,near_contact_band_m=.000025)
    if max(actual_errors)>normal_tolerance_deg+1e-8:
        return 'actual_pad_normal_not_aligned_with_surface',evidence
    regions,triangles = pad_regions(candidate,collider,pads,normal_tolerance_deg,
                                   triangle_ids=cylinder['supporting_triangle_ids'] if cylinder else None)
    evidence['pad_surface_regions'] = regions
    intersections = []
    desired = np.array([0.,candidate['pad_depth_m']])
    for a in regions[0]['polygons']:
        p = np.asarray(a['polygon_local_m'])[:,1:]
        for b in regions[1]['polygons']:
            if float(np.asarray(a['normal_local'])@b['normal_local'])>-np.cos(np.deg2rad(2*normal_tolerance_deg)):
                continue
            q = np.asarray(b['polygon_local_m'])[:,1:]
            if np.any(p.max(axis=0)<q.min(axis=0)) or np.any(q.max(axis=0)<p.min(axis=0)):
                continue
            overlap = intersect_polygons(p,q)
            if len(overlap)>=3 and abs(signed_area(overlap))>1e-14:
                point = closest_point(overlap,desired)
                intersections.append((float(np.linalg.norm(point-desired)),point,a,b,overlap))
    if not intersections:
        return 'no_common_opposed_surface_region_inside_pads',evidence
    _,yz,a,b,overlap = min(intersections,key=lambda row:row[0])
    local_points = []
    for record in (a,b):
        triangle = triangles[record['triangle_index']]
        n = np.asarray(record['normal_local'])
        x = (n@triangle[0]-n[1:]@yz)/n[0]
        local_points.append([x,*yz])
    pose = np.asarray(candidate['world_T_grip'])
    world_points = np.asarray(local_points)@pose[:3,:3].T+pose[:3,3]
    evidence.update(paired_surface_points_world_m=world_points.tolist(),
                    paired_surface_triangle_indices=[a['triangle_index'],b['triangle_index']],
                    common_region_polygon_yz_m=overlap.tolist(),
                    paired_surface_center_offset_m=float(np.linalg.norm(yz-desired)))
    if cylinder:
        shaft = np.asarray(cylinder['shaft_axis_world'])
        origin = np.asarray(cylinder['point_on_axis_world'])
        perpendicular_error=float(np.degrees(np.arcsin(np.clip(abs(shaft@pose[:3,0]),0,1))))
        evidence['cylinder_closing_perpendicular_error_deg']=perpendicular_error
        if perpendicular_error>1.+1e-8:
            return 'closing_not_transverse_to_fitted_shaft',evidence
        vectors = world_points-origin
        radial = vectors-(vectors@shaft)[:,None]*shaft
        radial /= np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-30)
        errors = np.degrees(np.arccos(np.clip(np.abs(radial@pose[:3,0]),0,1)))
        evidence['fitted_cylinder_tangent_errors_deg'] = errors.tolist()
        # A triangulated circular section touches a flat pad on a facet/vertex,
        # rather than the exact tangent point of its fitted smooth cylinder.
        # Bound that discretization by each actual contact triangle's radial
        # angular span; do not confuse it with flat-face alignment tolerance.
        sectors=[]
        for index in evidence['paired_surface_triangle_indices']:
            vertices=collider.faces[index]-origin
            v=vertices-(vertices@shaft)[:,None]*shaft
            v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-30)
            sectors.append(float(np.degrees(np.arccos(np.clip((v@v.T).min(),-1,1)))))
        evidence['mesh_contact_sector_widths_deg']=sectors
        allowance=1.+np.asarray(sectors)
        evidence['fitted_cylinder_tangent_allowances_deg']=allowance.tolist()
        if np.any(errors>allowance+1e-8):
            return 'round_contact_outside_native_facet_angular_bound',evidence
    return None,evidence
