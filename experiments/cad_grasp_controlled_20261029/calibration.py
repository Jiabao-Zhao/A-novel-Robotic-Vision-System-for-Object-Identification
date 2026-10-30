"""Portable distal contact calibration from the supplied complete hand arrays.

This is a collision-model contact calibration. The installed detailed visual
mesh has a central groove which the coarse convex collision model bridges.
Nothing here establishes physical finger-pad geometry or changes any mesh.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/cad_fingertip_contacts_20261029'
HAND=ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hand_geometry_digest(hand):
    values={key:hand[key] for key in ('frame','opening_limits_m','components')}
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def _clip(polygon,normal,offset):
    points=np.asarray(polygon,float)
    if not len(points):return points
    signed=points@normal+offset
    result=[]
    for i,a in enumerate(points):
        j=(i+1)%len(points);b=points[j]
        inside=signed[i]>=-1e-14;other=signed[j]>=-1e-14
        if inside:result.append(a)
        if inside!=other:
            result.append(a+signed[i]/(signed[i]-signed[j])*(b-a))
    return np.asarray(result,float).reshape(-1,points.shape[1])


def _inward_planes(polygon):
    p=np.asarray(polygon,float)
    area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))/2
    if area<0:p=p[::-1]
    result=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        edge=b-a;length=np.linalg.norm(edge)
        if length<=1e-14:continue
        n=np.array([-edge[1],edge[0]])/length
        result.append((n,-n@a))
    return result


def _erode(polygon,margin):
    result=np.asarray(polygon,float)
    for normal,offset in _inward_planes(polygon):
        result=_clip(result,normal,offset-margin)
        if len(result)<3:return np.empty((0,2))
    return result


def _intersect(first,second):
    result=np.asarray(first,float)
    for normal,offset in _inward_planes(second):
        result=_clip(result,normal,offset)
        if len(result)<3:return np.empty((0,2))
    return result


def _on_y0(polygon):
    p=np.asarray(polygon,float);hits=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        if abs(a[0])<=1e-14:hits.append(a[1])
        if a[0]*b[0]<0:hits.append(float(a[1]-a[0]/(b[0]-a[0])*(b[1]-a[1])))
    return None if not hits else [float(min(hits)),float(max(hits))]


def _lift(yz,normal,offset):
    yz=np.asarray(yz,float)
    x=-(yz@normal[1:]+offset)/normal[0]
    return np.column_stack((x,yz))


def make_contact_calibration(hand,normal_angle_rad=np.deg2rad(1.),edge_margin_m=.00025):
    """Return exact hull patches, common-region anchors and explicit limits.

    Patches are existing original finger hull triangles clipped at the old
    pad's distal plane, then eroded by the declared margin in their YZ
    projection. Contact is on those real collision-model surfaces, not on
    extended pad boxes. Left/right ordering follows opening motion along X.
    """
    if (not np.isfinite(normal_angle_rad) or not 0<normal_angle_rad<np.pi/2
            or not np.isfinite(edge_margin_m) or edge_margin_m<0):
        raise ValueError('Contact angle and projected edge margin must be finite')
    tip=fingertip_calibration(hand)
    minimum_z=float(tip['usable_pad_tip_origin_grip_site_m'][2])
    fingers=sorted([(i,c) for i,c in enumerate(hand['components'])
        if not c['is_pad'] and abs(c['translation_per_opening'][0])>1e-9],
        key=lambda item:item[1]['translation_per_opening'][0])
    if len(fingers)!=2:raise ValueError('Exactly two saved moving finger hulls are required')
    patches=[];by_side=[[],[]]
    for side,(component_id,component) in enumerate(fingers):
        points=np.asarray(component['vertices_zero_m'],float);hull=ConvexHull(points)
        inward=np.array([1.,0,0]) if side==0 else np.array([-1.,0,0])
        for face_id,(face,plane) in enumerate(zip(hull.simplices,hull.equations)):
            normal,offset=plane[:3],float(plane[3])
            error=float(np.arccos(np.clip(normal@inward,-1,1)))
            if error>normal_angle_rad:continue
            triangle=points[face].copy()
            if np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])@normal<0:
                triangle=triangle[[0,2,1]]
            polygon=_clip(triangle,np.array([0.,0,1.]),-minimum_z)
            if len(polygon)<3:continue
            yz=polygon[:,1:];eroded=_erode(yz,edge_margin_m)
            if len(eroded)<3:continue
            patch=dict(component=component['name'],hand_component_index=component_id,
                side=side,hull_face_index=face_id,source_vertex_indices=face.tolist(),
                normal_grip_site=normal.tolist(),plane_offset_m=offset,
                normal_error_to_closing_deg=float(np.rad2deg(error)),
                polygon_grip_site_m=polygon.tolist(),
                eroded_polygon_grip_site_m=_lift(eroded,normal,offset).tolist(),
                polygon_yz_m=yz.tolist(),eroded_polygon_yz_m=eroded.tolist(),
                projected_margin_m=float(edge_margin_m))
            by_side[side].append(len(patches));patches.append(patch)
    common=[]
    for left in by_side[0]:
        for right in by_side[1]:
            overlap=_intersect(patches[left]['eroded_polygon_yz_m'],patches[right]['eroded_polygon_yz_m'])
            if len(overlap)<3:continue
            interval=_on_y0(overlap)
            common.append(dict(left_patch_index=left,right_patch_index=right,
                polygon_yz_m=overlap.tolist(),y0_z_interval_m=interval))
    eligible=[r for r in common if r['y0_z_interval_m'] is not None]
    if not eligible:raise ValueError('No common opposed distal contact region at sharedY=0')
    chosen=max(eligible,key=lambda r:r['y0_z_interval_m'][1])
    z=chosen['y0_z_interval_m'][1]
    selected=[patches[chosen[key]] for key in ('left_patch_index','right_patch_index')]
    anchors=np.asarray([_lift([[0.,z]],np.asarray(p['normal_grip_site']),p['plane_offset_m'])[0]
                        for p in selected])
    slopes=np.asarray([component['translation_per_opening'] for _,component in fingers])
    normal_pair=np.asarray([p['normal_grip_site'] for p in selected])
    geometry_hash=hand_geometry_digest(hand)
    result=dict(status='available_saved_collision_fingertip_contact_calibration',
        frame=hand['frame'],units='metres/radians',hand_geometry_sha256=geometry_hash,
        finger_components=[component['name'] for _,component in fingers],
        contact_anchors_zero_grip_site_m=anchors.tolist(),
        contact_anchors_per_opening=slopes.tolist(),
        contact_anchors_zero_fingertip_m=(anchors-tip['fingertip_origin_grip_site_m']).tolist(),
        selected_contact_patch_indices=[chosen['left_patch_index'],chosen['right_patch_index']],
        contact_patches=patches,common_opposed_regions_yz_m=common,
        normal_angle_rad=float(normal_angle_rad),edge_margin_m=float(edge_margin_m),
        minimum_contact_z_grip_site_m=minimum_z,shared_anchor_y_m=0.,
        opposed_surface_normal_disagreement_deg=float(np.rad2deg(np.arccos(np.clip(-normal_pair[0]@normal_pair[1],-1,1)))),
        gap_zero_m=float(anchors[1,0]-anchors[0,0]),
        gap_per_opening=float(slopes[1,0]-slopes[0,0]),
        physical_fingertip_origin_grip_site_m=tip['fingertip_origin_grip_site_m'].tolist(),
        full_finger_extent_beyond_anchor_m=float(tip['fingertip_origin_grip_site_m'][2]-z),
        provenance=dict(contact_surface='original saved convex finger collision hull facets; no boxes extended and no vertices removed',
            source_hand_model=None,source_hand_model_sha256=None,
            source_hand_geometry_sha256=geometry_hash,
            source_kind='caller-supplied complete hand; optional file provenance belongs to the caller manifest',
            installed_asset_provenance='robosuite/models/assets/grippers/panda_gripper.xml and meshes/panda_gripper/finger.stl',
            contact_definition='inward hull faces within1degree, clippedZ>=oldpadtip, projectedYZboundarymargin0.25mm',
            visual_geometry_caveat='detailed finger_vis.stl has a central V-groove bridged by the coarse convex collision hull atY=0; contact here is a collision-model approximation',
            physical_contact_accuracy_validated=False,full_target_contact_fit_required=True,
            physical_pickup_or_assembly_success_established=False))
    result['calibration_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return result

