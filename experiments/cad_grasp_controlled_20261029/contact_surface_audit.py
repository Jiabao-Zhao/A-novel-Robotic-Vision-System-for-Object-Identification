"""Read-only exact triangle-surface audit of saved modeled finger contacts.

Detailed visual geometry is an independently supplied model, not physical
ground truth. Distances are diagnostic: this helper never changes acceptance,
fits contacts, generates candidates, or operates a simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from experiments.cad_grasp_geometry_20261026.gallery import read_stl

LEFT_MESH_TO_GRIP_SITE=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,-.0446],[0.,0.,0.,1.]])
THRESHOLDS_M=(1e-6,.0001,.00025)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closest_triangle_surface(point,vertices,faces):
    """Exact Euclidean closest point: triangle projection or its three edges.

    This is not a nearest-vertex approximation. Degenerate triangles are safely
    handled by their edge segments. Returned normals are flagged ambiguous when
    equally near noncoplanar faces meet at an edge or vertex.
    """
    point=np.asarray(point,float);vertices=np.asarray(vertices,float);faces=np.asarray(faces)
    if (point.shape!=(3,) or not np.isfinite(point).all() or vertices.ndim!=2
            or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Exact surface query needs finite point/vertices and original integer triangles')
    tri=vertices[faces];a,b,c=tri[:,0],tri[:,1],tri[:,2]
    ab,ac=b-a,c-a
    normal=np.cross(ab,ac);norm2=np.sum(normal*normal,axis=1)
    unit=np.divide(normal,np.sqrt(norm2)[:,None],out=np.zeros_like(normal),where=norm2[:,None]>0)
    d00=np.sum(ab*ab,axis=1);d01=np.sum(ab*ac,axis=1);d11=np.sum(ac*ac,axis=1)
    ap=point-a;d20=np.sum(ap*ab,axis=1);d21=np.sum(ap*ac,axis=1)
    denominator=d00*d11-d01*d01
    v=np.divide(d11*d20-d01*d21,denominator,out=np.full(len(tri),np.inf),where=denominator>0)
    w=np.divide(d00*d21-d01*d20,denominator,out=np.full(len(tri),np.inf),where=denominator>0)
    inside=(v>=0)&(w>=0)&(v+w<=1)&(norm2>0)
    projection=a+np.where(inside,v,0.)[:,None]*ab+np.where(inside,w,0.)[:,None]*ac
    candidates=[np.where(inside[:,None],projection,np.inf)]
    barycentric=[np.column_stack((1-v-w,v,w))]
    for start,end,i,j in [(a,b,0,1),(b,c,1,2),(c,a,2,0)]:
        edge=end-start;length2=np.sum(edge*edge,axis=1)
        t=np.clip(np.divide(np.sum((point-start)*edge,axis=1),length2,
            out=np.zeros(len(tri)),where=length2>0),0,1)
        candidates.append(start+t[:,None]*edge)
        weights=np.zeros((len(tri),3));weights[:,i]=1-t;weights[:,j]=t;barycentric.append(weights)
    choices=np.stack(candidates);squared=np.sum((choices-point)**2,axis=2)
    kind=np.argmin(squared,axis=0);index=np.arange(len(tri))
    best_squared=squared[kind,index];face_id=int(np.argmin(best_squared))
    distance=float(np.sqrt(max(0.,best_squared[face_id])))
    closest=choices[kind[face_id],face_id]
    weights=np.stack(barycentric)[kind[face_id],face_id]
    distances=np.sqrt(np.maximum(best_squared,0))
    ties=np.flatnonzero(np.abs(distances-distance)<=1e-12)
    tied_normals=unit[ties];tied_valid=np.linalg.norm(tied_normals,axis=1)>0
    meaningful=bool(norm2[face_id]>0 and tied_valid.all() and
        np.all(tied_normals@unit[face_id]>=1-1e-10))
    return dict(distance_m=distance,closest_point_m=closest.tolist(),triangle_index=face_id,
        barycentric=weights.tolist(),closest_face_normal=unit[face_id].tolist(),
        closest_normal_unambiguous=meaningful,equally_near_triangle_count=len(ties),
        exact_method='minimum of orthogonal triangle-interior projection and three finite edge segments')


def mesh_topology(vertices,faces):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    welded,inverse=np.unique(vertices,axis=0,return_inverse=True);f=inverse[faces]
    edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    _,groups=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True)
    counts=np.bincount(groups);sign=np.where(edges[:,0]<edges[:,1],1,-1)
    oriented=bool(np.all(np.bincount(groups,weights=sign)==0))
    return dict(vertices=len(vertices),triangles=len(faces),closed_manifold=bool(np.all(counts==2)),
        consistently_oriented=oriented,boundary_or_nonmanifold_edges=int(np.sum(counts!=2)))


def material_location(point,vertices,faces,topology,distance):
    """Generalized winding only when the supplied mesh is closed/oriented."""
    if distance<=1e-10:return dict(status='on_surface_numerically',winding_number=None)
    if not topology['closed_manifold'] or not topology['consistently_oriented']:
        return dict(status='unavailable_nonclosed_or_inconsistent_visual_mesh',winding_number=None)
    tri=np.asarray(vertices,float)[np.asarray(faces,int)]-np.asarray(point,float)
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    la,lb,lc=[np.linalg.norm(t,axis=1) for t in (a,b,c)]
    numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=la*lb*lc+np.sum(a*b,axis=1)*lc+np.sum(b*c,axis=1)*la+np.sum(c*a,axis=1)*lb
    winding=float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi))
    return dict(status='inside_detailed_mesh_solid' if abs(winding)>.5 else 'outside_detailed_mesh_solid',
        winding_number=winding)


def xml_contact_provenance(path):
    if path is None:return dict(status='not_supplied')
    path=Path(path);root=ET.parse(path).getroot()
    names=['finger1_visual','finger2_visual','finger1_collision','finger2_collision',
           'finger1_pad_collision','finger2_pad_collision']
    return dict(status='available',path=str(path),sha256=sha(path),
        relevant_geoms=[dict(node.attrib) for node in root.iter('geom') if node.get('name') in names],
        interpretation='Detailed finger_vis is a noncolliding display mesh in this XML; actual collision uses finger plus separate high-friction pad boxes. Visual-surface agreement is a diagnostic model-consistency check, not verified physical material calibration.')


def _summary(rows):
    contacts=[contact for row in rows for contact in row['contacts']]
    distances=[contact['visual']['distance_m'] for contact in contacts]
    return dict(poses=len(rows),contact_points=len(contacts),
        distance_range_m=[min(distances),max(distances)] if distances else None,
        distance_median_m=float(np.median(distances)) if distances else None,
        thresholds=[dict(threshold_m=threshold,contacts_within=sum(d<=threshold for d in distances),
            poses_with_both_contacts_within=sum(all(c['visual']['distance_m']<=threshold for c in row['contacts']) for row in rows))
            for threshold in THRESHOLDS_M],
        material_location_counts={status:sum(c['material_location']['status']==status for c in contacts)
            for status in sorted({c['material_location']['status'] for c in contacts})},
        closest_normals_unambiguous=sum(c['visual']['closest_normal_unambiguous'] for c in contacts),
        collision_source_distance_range_m=[min(c['collision']['distance_m'] for c in contacts),max(c['collision']['distance_m'] for c in contacts)]
            if contacts and contacts[0]['collision'] is not None else None)


def audit_saved_run(run_directory,hand,visual_mesh_path,*,collision_mesh_path=None,
                    xml_path=None,mesh_to_left_grip_site=LEFT_MESH_TO_GRIP_SITE):
    """Audit saved valid contacts using only explicitly supplied hand/assets."""
    run=Path(run_directory);visual_path=Path(visual_mesh_path)
    vertices,faces=read_stl(visual_path);transform=np.asarray(mesh_to_left_grip_site,float)
    if (transform.shape!=(4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1])
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-10)
            or not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-10)):
        raise ValueError('Detailed-mesh mapping must be an explicit metric rigid transform')
    visual_left=np.asarray(vertices)@transform[:3,:3].T+transform[:3,3]
    topology=mesh_topology(visual_left,faces)
    collision_left=collision_faces=None
    if collision_mesh_path is not None:
        cv,collision_faces=read_stl(Path(collision_mesh_path))
        collision_left=np.asarray(cv)@transform[:3,:3].T+transform[:3,3]
    fingers=sorted([component for component in hand['components'] if not component['is_pad']
        and abs(component['translation_per_opening'][0])>1e-9],key=lambda c:c['translation_per_opening'][0])
    if len(fingers)!=2:raise ValueError('Exactly two supplied moving finger solids are required')
    names=[component['name'] for component in fingers];rows=[];sources={};cache={}
    for path in sorted(run.glob('cases/*/*/candidate_diagnostics.json')):
        selected_path=path.parent/'selection.json';selection=json.loads(selected_path.read_text())['selection']
        selected=selection.get('selected_grasp');selected_id=selected.get('candidate_id') if selected else None
        sources[str(path)]=sha(path);sources[str(selected_path)]=sha(selected_path)
        for candidate in json.loads(path.read_text()):
            if not candidate['validity'].get('valid'):continue
            if candidate.get('actual_contact_component_names')!=names:raise ValueError('Saved contact component order disagrees with supplied hand')
            q=float(candidate['holding_opening_m']);contacts=np.asarray(candidate['actual_contacts_hand_m'],float)
            grip=np.asarray(candidate['cad_T_grip_site']);target_normals=np.asarray(candidate.get('contact_normals_cad',[]),float)
            values=[]
            for side,(component,point) in enumerate(zip(fingers,contacts)):
                zero=point-q*np.asarray(component['translation_per_opening'])
                # Right is the same installed mesh rotated180degrees aboutZ.
                mirror=np.array([1.,1.,1.]) if side==0 else np.array([-1.,-1.,1.])
                key=tuple(zero*mirror)
                if key not in cache:
                    visual=closest_triangle_surface(zero*mirror,visual_left,faces)
                    collision=(closest_triangle_surface(zero*mirror,collision_left,collision_faces)
                               if collision_left is not None else None)
                    location=material_location(zero*mirror,visual_left,faces,topology,visual['distance_m'])
                    cache[key]=dict(visual=visual,collision=collision,material_location=location)
                value=json.loads(json.dumps(cache[key]))
                value.update(component=component['name'],side=side,point_zero_grip_site_m=zero.tolist())
                value['visual']['closest_point_m']=(np.asarray(value['visual']['closest_point_m'])*mirror).tolist()
                value['visual']['closest_face_normal']=(np.asarray(value['visual']['closest_face_normal'])*mirror).tolist()
                if value['collision'] is not None:
                    value['collision']['closest_point_m']=(np.asarray(value['collision']['closest_point_m'])*mirror).tolist()
                    value['collision']['closest_face_normal']=(np.asarray(value['collision']['closest_face_normal'])*mirror).tolist()
                if target_normals.shape==(2,3):
                    expected=-target_normals[side]@grip[:3,:3]
                    normal=np.asarray(value['visual']['closest_face_normal'])
                    value['nearest_visual_normal_to_expected_contact_deg']=float(np.rad2deg(np.arccos(np.clip(normal@expected,-1,1))))
                value['normal_defines_existing_contact']=bool(value['visual']['closest_normal_unambiguous'] and value['visual']['distance_m']<=1e-6)
                values.append(value)
            rows.append(dict(scene=path.parent.parent.name,part=path.parent.name,candidate_id=candidate['candidate_id'],
                selected=candidate['candidate_id']==selected_id,holding_opening_m=q,contacts=values))
    unique={}
    for row in rows:
        key=(row['part'],row['candidate_id'],row['holding_opening_m'],tuple(tuple(c['point_zero_grip_site_m']) for c in row['contacts']))
        unique.setdefault(key,row)
    return dict(study='Saved finger-end contacts versus detailed visual triangle surfaces',
        method='Exact point-to-triangle Euclidean distance; optional generalized winding on closed oriented detailed mesh',
        new_candidates_generated=False,new_fits=False,new_validation=False,execution=False,
        run_directory=str(run),visual_mesh_path=str(visual_path),visual_mesh_sha256=sha(visual_path),
        collision_mesh_path=str(collision_mesh_path) if collision_mesh_path else None,
        collision_mesh_sha256=sha(collision_mesh_path) if collision_mesh_path else None,
        supplied_hand_geometry_sha256=hashlib.sha256(json.dumps(hand,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        mesh_to_left_grip_site=transform.tolist(),right_zero_frame_transform=np.diag([-1.,-1.,1.,1.]).tolist(),
        visual_topology=topology,xml_contact_provenance=xml_contact_provenance(xml_path),
        thresholds_scope='1um is the existing intended-contact tolerance;0.1mm and0.25mm are diagnostic reporting thresholds, never tuned acceptance limits',
        validity_scope='Distances do not modify modeled validity and do not establish physical contact material or pickup/assembly success',
        all_saved_valid_pose_summary=_summary(rows),selected_pose_summary=_summary([r for r in rows if r['selected']]),
        unique_CAD_valid_pose_summary=_summary(list(unique.values())),
        per_part_summary={part:dict(all_valid=_summary([r for r in rows if r['part']==part]),
            selected=_summary([r for r in rows if r['part']==part and r['selected']]))
            for part in sorted({r['part'] for r in rows})},case_file_sha256=sources,records=rows)


def write_report(result,output_directory):
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
    path=output/'detailed_surface_audit.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Detailed finger-surface audit','',result['method'],
        '',result['validity_scope'],'','| Saved subset | Poses | Contact points | Distance range(mm) | Within1um: points / both-contact poses |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('all_saved_valid_pose_summary','All modeled-valid'),('selected_pose_summary','Selected'),('unique_CAD_valid_pose_summary','Unique CAD valid fits')]:
        s=result[key];span=s['distance_range_m'];t=s['thresholds'][0]
        span_text=f'{span[0]*1000:.6f}–{span[1]*1000:.6f}' if span else 'unavailable (no contacts)'
        lines.append(f"|{label}|{s['poses']}|{s['contact_points']}|{span_text}|{t['contacts_within']} / {t['poses_with_both_contacts_within']}|")
    lines+=['','| Diagnostic threshold | All valid: points / both-contact poses | Selected: points / both-contact poses |',
        '|---|---:|---:|']
    for first,second in zip(result['all_saved_valid_pose_summary']['thresholds'],result['selected_pose_summary']['thresholds']):
        lines.append(f"|{first['threshold_m']*1000:g}mm|{first['contacts_within']} / {first['poses_with_both_contacts_within']}|{second['contacts_within']} / {second['poses_with_both_contacts_within']}|")
    lines+=['','Detailed finger_vis is visual-only in the supplied XML. Collision uses the coarser finger mesh and separate pad boxes. Disagreement exposes a model-consistency issue; the detailed visual asset is not physical calibration.','',result['thresholds_scope']]
    (output/'detailed_surface_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','hand','visual-mesh','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--collision-mesh',type=Path);parser.add_argument('--xml',type=Path)
    args=parser.parse_args();hand=json.loads(args.hand.read_text())
    result=audit_saved_run(args.run,hand,args.visual_mesh,collision_mesh_path=args.collision_mesh,xml_path=args.xml)
    print(write_report(result,args.output));print(json.dumps(result['all_saved_valid_pose_summary'],indent=2))


if __name__=='__main__':main()
