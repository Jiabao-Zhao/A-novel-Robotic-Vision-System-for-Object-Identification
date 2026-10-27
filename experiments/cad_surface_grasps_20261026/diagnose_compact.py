"""Read-only inspection of frozen compact grasps and native CAD contact normals."""
import json
from pathlib import Path
import numpy as np

from experiments.cad_grasp_geometry_20261026 import run_trial as old
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_regions_20261026 import run as compact
from experiments.cad_grasp_regions_20261026.regions import contact_regions
from experiments.cad_grasp_regions_20261026.regions import recast_pair
from experiments.cad_grasp_geometry_20261026.hand_model import vertices as hand_vertices

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'outputs/cad_grasp_regions_20261026/runs/20261026T184853Z'
OUT = ROOT/'outputs/cad_surface_grasps_20261026/compact_diagnostics.json'


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)) / (np.linalg.norm(a)*np.linalg.norm(b)), 0, 1))))


def clip_rectangle(points, low, high):
    """Clip a 3D polygon to a rectangle in hand Y/Z, preserving local X."""
    polygon = np.asarray(points)
    for axis in (1,2):
        for value,sign in ((low[axis-1],1.),(high[axis-1],-1.)):
            result=[]
            if not len(polygon):
                return polygon
            for p,q in zip(polygon,np.roll(polygon,-1,axis=0)):
                dp,dq=sign*(p[axis]-value),sign*(q[axis]-value)
                ip,iq=dp>=-1e-12,dq>=-1e-12
                if ip:
                    result.append(p)
                if ip!=iq:
                    result.append(p+np.clip(dp/(dp-dq),0.,1.)*(q-p))
            polygon=np.asarray(result)
    return polygon


def flat_face_overlap(triangles,normals,index,pose,pad,opening):
    """Projected finite-pad/facet overlap, NOT physical loaded contact area."""
    normal=normals[index]
    anchor=triangles[index,0]
    coplanar=(np.abs(normals@normal)>np.cos(np.deg2rad(.01)))
    coplanar &= np.max(np.abs((triangles-anchor)@normal),axis=1)<=1e-7
    pv=hand_vertices(pad,opening)
    low,high=pv[:,1:].min(axis=0)+.00025,pv[:,1:].max(axis=0)-.00025
    negative=float(np.asarray(pad['translation_per_opening'])[0])<0
    inner_x=float(pv[:,0].max() if negative else pv[:,0].min())
    all_polygons=[]
    area=0.
    for t in triangles[coplanar]:
        local=(t-pose[:3,3])@pose[:3,:3]
        poly=clip_rectangle(local,low,high)
        if len(poly)<3:
            continue
        yz=poly[:,1:]
        a=abs(float(yz[:,0]@np.roll(yz[:,1],-1)-yz[:,1]@np.roll(yz[:,0],-1)))/2
        if a>1e-15:
            area+=a
            all_polygons.append(poly)
    verts=np.concatenate(all_polygons) if all_polygons else np.empty((0,3))
    return dict(projected_guarded_overlap_area_m2=area, coplanar_triangles=int(coplanar.sum()),
          pad_footprint_area_m2=float(np.prod(high-low)),
          x_variation_over_overlap_m=float(np.ptp(verts[:,0])) if len(verts) else None,
          maximum_inner_plane_distance_m=float(np.max(np.abs(verts[:,0]-inner_x))) if len(verts) else None,
          overlap_polygons_hand_m=[p.tolist() for p in all_polygons],
          note='Geometric projection of native coplanar facet onto finite guarded pad; not physical loaded contact area')


def transverse_probe(row,v,f,shaft):
    components,pads,gap=old.hand_geometry()
    x=np.cross([0.,0.,1.],shaft)
    if np.linalg.norm(x)<1e-8:
        return dict(status='vertical_shaft_direction_not_unique')
    x/=np.linalg.norm(x)
    result=contact_regions(v,f,x,.067,grid_spacing_m=.001,normal_angle_deg=15.,
                          neighbor_width_jump_m=.002,neighbor_midpoint_jump_m=.001,neighbor_normal_angle_deg=30.)
    regions=compact.merge_regions(result,pads)
    collider=MeshCollider(v,f)
    plane=np.asarray(row['table_plane_world'])
    records=[]
    for region in regions:
        placement,reason=compact.prepare_family(region,result,x,components,pads,gap,plane,collider)
        entry=dict(region_id=region['region_id'],z_min_m=region['z_min'],z_max_m=region['z_max'],
                   center_height_m=region['center_z'],median_width_m=region['median_width_m'],
                   preparation_rejection=reason)
        center_pair=recast_pair(result,region['center_y'],region['center_z'])
        if center_pair is not None:
            center=np.mean(center_pair['contacts_world_m'],axis=0)
            table_h=compact.minimum_table_height(center,x,center_pair['width_m']+gap+.012,components,plane)
            pad_high=min(np.max(p['vertices_zero_m'],axis=0)[2] for p in pads)
            entry.update(minimum_table_hand_height_m=table_h,
                         minimum_guarded_pad_contact_height_m=table_h-pad_high+.00025,
                         center_pair_width_m=center_pair['width_m'])
        if not reason:
            aliases=[]
            for sign in (1.,-1.):
                pose=np.eye(4)
                pose[:3,:3]=compact.hand_rotation(sign*x)
                pair=np.asarray(placement['pair']['contacts_world_m'])
                pose[:3,3]=pair.mean(axis=0)
                pose[2,3]=placement['height_m']
                candidate=dict(world_T_grip=pose.tolist(),contacts_world_m=pair.tolist(),
                  opening_m=placement['pair']['width_m']+gap+.012,pad_depth_m=placement['pad_depth_m'])
                rejection,evidence=old.validate(candidate,collider,components,pads,plane)
                aliases.append(dict(jaw_sign=sign,valid=rejection is None,rejection=rejection,evidence=evidence,
                                    world_T_grip=pose.tolist(),height_m=placement['height_m'],pad_depth_m=placement['pad_depth_m']))
            entry['aliases']=aliases
        records.append(entry)
    return dict(direction_world=x.tolist(),axis_to_shaft_angle_deg=angle(x,shaft),
                qualified_cells=result['metadata']['qualified_cells'],contact_regions=records,
                valid_aliases=sum(sum(a['valid'] for a in r.get('aliases',[])) for r in records),
                metadata=result['metadata'])


def inspect(row):
    v, f, _ = old.load_target(row)
    cad = np.asarray(row['world_T_cad'])
    tri = v[f]
    normal = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    normal /= np.linalg.norm(normal, axis=1)[:,None]
    centered = (v-cad[:3,3])@cad[:3,:3]
    cov = np.cov(centered.T)
    _, vectors = np.linalg.eigh(cov)
    shaft_cad = vectors[:,-1]
    shaft_world = cad[:3,:3]@shaft_cad
    record = dict(scene=row['scene'], part=row['part'], native_cad_extent_m=np.ptp(centered,axis=0).tolist(),
                  principal_long_axis_cad=shaft_cad.tolist(), principal_long_axis_world=shaft_world.tolist(),
                  allowed_axes=row['axes'], families=[])
    _,pads,_=old.hand_geometry()
    for family in row['grasps']:
        aliases = []
        for grasp in family['jaw_aliases']:
            pose = np.asarray(grasp['world_T_grip'])
            x = pose[:3,0]
            contacts = np.asarray(grasp['contact_points_world_m'])
            native_angles = [angle(x,cad[:3,i]) for i in range(3)]
            exact = []
            for pad,contact in zip(pads,grasp['actual_contacts']):
                index = contact['first_contact']['triangle_index']
                n = normal[index]
                exact.append(dict(triangle_index=index, native_triangle_normal_world=n.tolist(),
                                  normal_to_closing_angle_deg=angle(n,x),
                                  triangle_world_m=tri[index].tolist(),
                                  witness_world_m=contact['point_world_m'],
                                  flat_face_overlap=flat_face_overlap(tri,normal,index,pose,pad,contact['opening_m'])
                                  if row['part']=='KET16_Square_16mm' else None))
            aliases.append(dict(id=grasp['id'],axis_label=grasp['axis_label'],
                native_axis_angles_deg=native_angles,shaft_to_closing_angle_deg=angle(x,shaft_world),
                closing_axis_world=x.tolist(),normal_angles_deg=[c['normal_to_closing_angle_deg'] for c in exact],
                actual_contact_witnesses_hand_m=((contacts-pose[:3,3])@pose[:3,:3]).tolist(),
                actual_contact_witnesses_cad_m=((contacts-cad[:3,3])@cad[:3,:3]).tolist(),
                sampled_pair_hand_m=((np.asarray(grasp['contacts_world_m'])-pose[:3,3])@pose[:3,:3]).tolist(),
                actual_witness_line_to_closing_angle_deg=angle(contacts[1]-contacts[0],x),
                contact_opening_m=grasp['contact_opening_m'],table_clearance_m=grasp['table_clearance_m'],
                exact_native_contacts=exact))
        record['families'].append(dict(family_id=family['family_id'],aliases=aliases))
    record['preparation_rejection_counts'] = row['preparation_rejection_counts']
    if row['part']!='KET16_Square_16mm':
        # Use the model's geometric long direction, not either arbitrary roll axis.
        record['transverse_probe']=transverse_probe(row,v,f,shaft_world)
    return record


def main():
    old.initialize()
    rows = json.loads((RUN/'results.json').read_text())
    picked = [r for r in rows if (r['part']=='KET16_Square_16mm' and r['scene'] in ('N01','N02'))
              or (r['part'] in ('RGOCG8-50_8mm','RGOCG16-50_16mm') and r['scene'] in ('N00','N02'))
              or (r['part']=='BNC_Male' and r['scene'] in ('N00','N01'))]
    data = [inspect(r) for r in picked]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(dict(source_run=str(RUN.relative_to(ROOT)), cases=data),indent=2)+'\n')
    for r in data:
        print(r['scene'],r['part'],'axes',[(a['label'],round(a['projection_error_deg'],5),a['usable']) for a in r['allowed_axes']],
              'families',[(g['family_id'],[(a['id'],a['normal_angles_deg'],a['shaft_to_closing_angle_deg'],a['actual_witness_line_to_closing_angle_deg']) for a in g['aliases']]) for g in r['families']])
        if r.get('transverse_probe'):
            p=r['transverse_probe']
            print('TRANSVERSE',p.get('valid_aliases'),[(q['region_id'],q['preparation_rejection'],q['center_height_m'],q['z_max_m'],
                  [(a['valid'],a['rejection']) for a in q.get('aliases',[])]) for q in p.get('contact_regions',[])])
        for family in r['families']:
            if r['part']=='KET16_Square_16mm':
                a=family['aliases'][0]
                print('FLAT_OVERLAP',a['id'],[(c['flat_face_overlap']['projected_guarded_overlap_area_m2']*1e6,
                        c['flat_face_overlap']['x_variation_over_overlap_m']*1e6,
                        c['flat_face_overlap']['maximum_inner_plane_distance_m']*1e6) for c in a['exact_native_contacts']])


if __name__=='__main__':
    main()
