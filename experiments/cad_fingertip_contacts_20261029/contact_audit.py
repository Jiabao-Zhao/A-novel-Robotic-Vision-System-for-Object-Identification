"""Auditable distal inner contact surfaces on the unchanged saved hand hulls.

This is a collision-model contact calibration. The installed detailed visual
mesh has a central groove which the coarse convex collision model bridges.
Nothing here establishes physical finger-pad geometry or changes any mesh.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

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
            source_hand_model=str(HAND.relative_to(ROOT)),source_hand_model_sha256=sha(HAND),
            installed_asset_provenance='robosuite/models/assets/grippers/panda_gripper.xml and meshes/panda_gripper/finger.stl',
            contact_definition='inward hull faces within1degree, clippedZ>=oldpadtip, projectedYZboundarymargin0.25mm',
            visual_geometry_caveat='detailed finger_vis.stl has a central V-groove bridged by the coarse convex collision hull atY=0; contact here is a collision-model approximation',
            physical_contact_accuracy_validated=False,full_target_contact_fit_required=True,
            physical_pickup_or_assembly_success_established=False))
    result['calibration_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return result


def audit():
    """Write source-bound calibration and a side comparison without simulation."""
    from experiments.cad_grasp_geometry_20261026.gallery import read_stl
    hand=json.loads(HAND.read_text(encoding='utf-8'))
    result=make_contact_calibration(hand)
    runtime_calibration=json.loads(json.dumps(result))
    assets=OUTPUT/'source_assets'
    records=[];mesh={}
    for name in ('finger','finger_vis','finger_longer'):
        path=assets/(name+'.stl');vertices,faces=read_stl(path)
        vertices=np.asarray(vertices);faces=np.asarray(faces)
        grip=np.column_stack((-vertices[:,1],vertices[:,0],vertices[:,2]-.0446))
        mesh[name]=(grip,faces)
        normals=np.cross(grip[faces[:,1]]-grip[faces[:,0]],grip[faces[:,2]]-grip[faces[:,0]])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        ids=np.flatnonzero(normals[:,0]>=np.cos(np.deg2rad(1)))
        records.append(dict(asset=name,path=str(path.relative_to(ROOT)),sha256=sha(path),
            installed_source_path='/home/jiabao/.venvs/lerobot-libero/lib/python3.12/site-packages/robosuite/models/assets/grippers/meshes/panda_gripper/'+name+'.stl',
            vertices=len(vertices),triangles=len(faces),bounds_grip_site_m=[grip.min(0).tolist(),grip.max(0).tolist()],
            near_parallel_inward_source_face_ids=ids.tolist(),
            used_in_saved_scene_collision=name=='finger',
            used_in_saved_scene_visual=name=='finger_vis'))
    left=next(c for c in hand['components'] if c['name']==result['finger_components'][0])
    points=np.asarray(left['vertices_zero_m'])
    raw,source_faces=mesh['finger'];distance,mapping=cKDTree(points).query(raw)
    source_mapping=[]
    for index,patch in enumerate(result['contact_patches']):
        source_set=set(patch['source_vertex_indices'])
        source_mapping.append(dict(contact_patch_index=index,component=patch['component'],
            hull_face_index=patch['hull_face_index'],installed_collision_source_face_ids=[
                int(i) for i,face in enumerate(source_faces) if set(mapping[face])==source_set]))
    result['installed_collision_source_face_mapping']=source_mapping
    result['runtime_calibration']=runtime_calibration
    result['calibration_sha256_scope']='canonical make_contact_calibration dictionary in runtime_calibration, excluding only its own calibration_sha256 field; all other top-level additions are audit-only'
    result['installed_assets']=records
    result['installed_collision_to_saved_export_max_vertex_error_m']=float(distance.max())
    result['installed_left_mesh_to_grip_site_transform']=[[0,-1,0,0],[1,0,0,0],[0,0,1,-.0446],[0,0,0,1]]
    result['source_xml_sha256']=sha(assets/'panda_gripper.xml')
    result['source_helper_sha256']=sha(__file__)
    result['source_scene_sha256']=sha(ROOT/'outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml')
    result['unused_longer_finger_note']='finger_longer is declared as a mesh asset but no saved scene geom uses it; not adopted'
    sections={}
    for name,(vertices,faces) in mesh.items():
        segments=[]
        for triangle in vertices[faces]:
            hits=[]
            for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
                if abs(a[1])<=1e-12:hits.append(a)
                if a[1]*b[1]<0:hits.append(a-a[1]/(b[1]-a[1])*(b-a))
            if len(hits)>=2:segments.append(np.asarray(hits)[:,[0,2]])
        sections[name]=segments
    anchor_z=result['contact_anchors_zero_grip_site_m'][0][2]
    fronts={}
    for name in ('finger','finger_vis'):
        hits=[]
        for segment in sections[name]:
            a,b=segment[0],segment[-1]
            if min(a[1],b[1])<=anchor_z<=max(a[1],b[1]) and abs(b[1]-a[1])>1e-14:
                hits.append(float(a[0]+(anchor_z-a[1])/(b[1]-a[1])*(b[0]-a[0])))
        fronts[name]=None if not hits else max(hits)
    result['y0_anchor_cross_section_front_x_m']=fronts
    result['collision_hull_minus_visual_front_at_anchor_m']=fronts['finger']-fronts['finger_vis']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'contact_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,7),constrained_layout=True)
    for ax,name in zip(axes,('finger','finger_vis')):
        vertices,faces=mesh[name]
        for triangle in vertices[faces]:
            q=np.vstack((triangle,triangle[0]))
            ax.plot(q[:,0]*1000,q[:,2]*1000,color='#b7c2cf',linewidth=.6,alpha=.55)
        for index,segment in enumerate(sections[name]):
            ax.plot(segment[:,0]*1000,segment[:,1]*1000,color='#3b5869',linewidth=1.8,
                    label='Actual Y=0 cross-section' if index==0 else None)
        anchor=np.asarray(result['contact_anchors_zero_grip_site_m'][0])
        ax.plot(anchor[0]*1000,anchor[2]*1000,'o',color='#007a73',markersize=7)
        ax.axhline(4.4,color='#d7971e',linestyle='--',label='Old pad distal edge: 4.4 mm')
        ax.axhline(result['contact_anchors_zero_grip_site_m'][0][2]*1000,color='#007a73',label='New hull contact anchor')
        ax.axhline(result['physical_fingertip_origin_grip_site_m'][2]*1000,color='#9c3158',linestyle=':',label='Most-distal hull vertex')
        ax.set_xlim(-17,2);ax.set_ylim(-3,11);ax.set_aspect('equal')
        ax.set_xlabel('Grip-site X (mm); inward is right');ax.set_ylabel('Grip-site Z (mm); fingertip is up')
        ax.set_title('Saved collision source' if name=='finger' else 'Detailed visual source (audit only)')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Contact moves onto existing collision-model finger surfaces\nThe collision hull bridges the visual groove; physical contact calibration remains unverified',fontsize=12)
    fig.savefig(OUTPUT/'finger_contact_side_audit.png',dpi=170);plt.close(fig)
    return result


if __name__=='__main__':
    result=audit()
    print(json.dumps({key:result[key] for key in ('contact_anchors_zero_grip_site_m',
        'finger_components','selected_contact_patch_indices','full_finger_extent_beyond_anchor_m',
        'installed_collision_to_saved_export_max_vertex_error_m')},indent=2))
