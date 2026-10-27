"""Independent saved surface-contact and candidate-set audit; no execution."""
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import open3d as o3d

from experiments.cad_grasp_geometry_20261026.inputs import OUTPUT, REPO, local, read, sha256
from experiments.cad_grasp_geometry_20261026.verify_results import Checks, triangle_distances, moved_vertices
from experiments.cad_grasp_regions_20261026.verify_results import audit_geometry, same_physical_grip


def native_triangles(row):
    mesh=o3d.io.read_triangle_mesh(str(local(row['mesh_path'])))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if row['part']=='Gear_Large':
        mesh.merge_close_vertices(5e-8)
    pose=np.asarray(row['world_T_cad'])
    vertices=np.asarray(mesh.vertices)@pose[:3,:3].T+pose[:3,3]
    return vertices[np.asarray(mesh.triangles)]


def polygon_area(polygon):
    p=np.asarray(polygon)
    return abs(float(p[:,0]@np.roll(p[:,1],-1)-p[:,1]@np.roll(p[:,0],-1)))/2


def inside_convex(points, polygon, tolerance=1e-12):
    points=np.asarray(points).reshape(-1,2)
    polygon=np.asarray(polygon)
    if len(polygon)<3:
        return False
    edge=np.roll(polygon,-1,axis=0)-polygon
    delta=points[:,None]-polygon
    cross=edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]
    return bool(np.all(cross>=-tolerance) or np.all(cross<=tolerance))


def audit_contract(row,candidates,config,checks):
    name=row['scene']+'/'+row['part']
    start=checks.failed_count
    directions={d['id']:d for d in row['directions']}
    checks.check(len(directions)==len(row['directions']),name+': unique surface direction IDs')
    for d in directions.values():
        x=np.asarray(d['closing_axis_world'])
        checks.check(x.shape==(3,) and np.isfinite(x).all() and abs(np.linalg.norm(x)-1)<=1e-10,
                     name+': finite unit closing direction')
        checks.check(abs(x[2])<=1e-12,name+': closing direction horizontal')
        checks.check(isinstance(d['source'],str) and bool(d['source']),name+': direction provenance')
    families={f['family_id']:f for f in row['contact_families']}
    checks.check(len(families)==len(row['contact_families'])==row['merged_contact_regions'],name+': contact-region count')
    prepared=[f for f in families.values() if f['prepared']]
    checks.check(len(prepared)==row['prepared_contact_families'],name+': prepared region count')
    queries=sum(f.get('center_projection',{}).get('geometry_queries',0) for f in families.values())
    checks.check(queries==row['center_projection_geometry_queries'],name+': extra centering geometry queries declared')
    checks.check(len(candidates)==2*len(prepared)==row['generated_count'],name+': one placement and both jaws per prepared region')
    by_id={c['id']:c for c in candidates}
    checks.check(len(by_id)==len(candidates),name+': unique candidate IDs')
    valid=[c for c in candidates if c['valid']]
    checks.check(len(valid)==row['valid_count'],name+': passing checked alternatives count')
    checks.check(Counter(c['rejection_reason'] for c in candidates if not c['valid'])==row['rejection_counts'],name+': rejection counts')
    checks.check(all(c['rejection_reason'] is None for c in valid),name+': passing candidate has no rejection')
    checks.check(all(isinstance(c['rejection_reason'],str) for c in candidates if not c['valid']),name+': rejected candidate has reason')
    checks.check(Counter(f['preparation_rejection'] for f in families.values() if not f['prepared'])
                 ==row['preparation_rejection_counts'],name+': preparation failures explicit')
    members=[]
    for family in families.values():
        label=name+'/'+family['family_id']
        d=directions[family['direction_id']]
        aliases=[c for c in candidates if c['family_id']==family['family_id']]
        if not family['prepared']:
            checks.check(not aliases,label+': unavailable region has no fabricated pose')
            continue
        checks.check(len(aliases)==2 and {c['jaw_sign'] for c in aliases}=={-1.,1.},label+': both physical jaws checked')
        checks.check(family['candidate_ids']==[c['id'] for c in aliases],label+': exact region candidates')
        checks.check(family['valid_alias_ids']==[c['id'] for c in aliases if c['valid']],label+': exact region passing candidates')
        members.extend(c['id'] for c in aliases)
        for c in aliases:
            pose=np.asarray(c['world_T_grip'])
            checks.check(c['direction_id']==d['id'] and c['direction_source']==d['source'],label+': source-to-candidate mapping')
            checks.check(np.allclose(pose[:3,0],c['jaw_sign']*np.asarray(d['closing_axis_world']),atol=1e-12),label+': signed physical closing direction')
            checks.check(np.allclose(pose[:3,2],[0,0,-1],atol=1e-12),label+': straight-down approach')
            checks.check(abs(pose[2,3]-family['height_m'])<=1e-12 and abs(c['pad_depth_m']-family['pad_depth_m'])<=1e-12,
                         label+': one derived height per prepared region')
            checks.check(c['center_projection']==family['center_projection'],label+': same family centering evidence for both jaw assignments')
            audit_center_projection(c,checks,label+'/'+c['id'])
        if len(aliases)==2:
            checks.check(np.allclose(np.asarray(aliases[0]['world_T_grip'])[:3,3],np.asarray(aliases[1]['world_T_grip'])[:3,3],atol=1e-12),
                         label+': jaw exchange keeps translation')
    checks.check(len(members)==len(set(members))==len(candidates) and set(members)==set(by_id),name+': regions partition all proposals')
    groups=row['grasps']
    checks.check(len(groups)==row['grouped_count'],name+': displayed family count')
    kept=[]
    for group in groups:
        ids=group['grouped_candidate_ids']
        aliases=[by_id[i] for i in ids]
        kept.extend(ids)
        checks.check(bool(aliases) and all(c['valid'] for c in aliases),name+': only valid checked alternatives retained')
        checks.check(group['jaw_aliases']==aliases,name+': saved checked alternatives preserved')
        checks.check(group['source_family_ids']==sorted({c['family_id'] for c in aliases}),name+': merged source regions preserved')
        chosen=min(aliases,key=lambda c:(0 if c['jaw_sign']>0 else 1,c['opening_m'],c['id']))
        checks.check(group['id']==chosen['id'] and all(group[k]==v for k,v in chosen.items()),name+': canonical display is a checked alternative')
        physical=[]
        for c in aliases:
            if not any(same_physical_grip(c,p) for p in physical):
                physical.append(c)
        checks.check(len(physical)<=2,name+': grouped alternatives share at most two final jaw poses')
    checks.check(len(kept)==len(set(kept))==len(valid) and set(kept)=={c['id'] for c in valid},name+': every valid alternative retained once')
    checks.check(not any(same_physical_grip(a,b) for i,a in enumerate(groups) for b in groups[:i]),name+': no duplicate final display grip')
    return dict(contract_failures=checks.failed_count-start,prepared_regions=len(prepared),surface_directions=len(directions))


def audit_center_projection(candidate,checks,label='candidate'):
    """Panda +/- opening/2 kinematics: replay one pre-validation shift."""
    evidence=candidate['center_projection']
    before=np.asarray(evidence['world_T_grip_before'])
    after=np.asarray(candidate['world_T_grip'])
    openings=evidence['first_contact_openings_before_m']
    shift=evidence['translation_along_closing_m']
    checks.check(np.isfinite(before).all() and before.shape==(4,4),label+': finite pre-centering hand pose')
    checks.check(np.allclose(after[:3,3],before[:3,3]+shift*before[:3,0],atol=1e-12),label+': centered translation follows saved positive-jaw closing axis')
    checks.check(abs(after[2,3]-before[2,3])<=1e-12,label+': centering does not change contact height')
    expected_rotation=before[:3,:3]@np.diag([candidate['jaw_sign'],candidate['jaw_sign'],1.])
    checks.check(np.allclose(after[:3,:3],expected_rotation,atol=1e-12),label+': centering preserves each jaw orientation')
    if evidence['applied']:
        checks.check(len(openings)==2 and evidence['geometry_queries']==2,label+': both finite-pad contact queries counted')
        checks.check(abs(shift-(openings[1]-openings[0])/4)<=1e-12,label+': Panda first-contact midpoint equation')
        checks.check(abs(evidence['expected_equal_contact_opening_m']-np.mean(openings))<=1e-12,label+': expected equal opening is arithmetic mean')
        checks.check(all(0<=o<=candidate['opening_m'] for o in openings),label+': pre-centering contact openings within proposed motion')
        if candidate['valid']:
            # Each old first-contact search brackets a fraction to 1e-4.
            bracket=2*candidate['opening_m']*1e-4+1e-9
            checks.check(max(abs(o-evidence['expected_equal_contact_opening_m']) for o in candidate['first_contact_openings_m'])<=bracket,
                         label+': revalidated centered contacts agree within numerical search brackets')
    else:
        checks.check(shift==0. and evidence['geometry_queries']==len(openings)+1 and len(openings)<2,
                     label+': unsuccessful centering query explicit, without invented shift')
        checks.check(isinstance(evidence.get('reason'),str),label+': unsuccessful projection has reason')
        checks.check(not candidate['valid'],label+': valid grasp requires both pre-centering pad contacts')


def audit_surface_regions(candidate,triangles,pads,checks,label='case'):
    """Replay normals and polygon/triangle membership without contacts.py."""
    start=checks.failed_count
    pose=np.asarray(candidate['world_T_grip'])
    local_tri=(np.asarray(triangles)-pose[:3,3])@pose[:3,:3]
    cross=np.cross(local_tri[:,1]-local_tri[:,0],local_tri[:,2]-local_tri[:,0])
    normals=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
    actual_indices=[w['first_contact']['triangle_index'] for w in candidate['actual_contacts']]
    errors=np.degrees(np.arccos(np.clip(np.abs(normals[actual_indices,0]),0,1)))
    tolerance=candidate['surface_normal_tolerance_deg']
    cylinder=candidate.get('cylinder_model')
    checks.check(tolerance==(15. if cylinder else 1.),label+': declared flat/round normal tolerance')
    checks.check(np.allclose(errors,candidate['actual_pad_normal_errors_deg'],atol=1e-6),label+': independently reconstructed native surface-normal errors')
    checks.check(float(errors.max())<=tolerance+1e-7,label+': actual native surface alignment')
    band=candidate['near_contact_band_m']
    checks.check(band==.000025,label+': geometric near-contact band stays 25 micrometers')
    records=candidate['pad_surface_regions']
    checks.check(len(records)==2 and {r['pad'] for r in records}==set(pads),label+': both pad regions recorded')
    polygon_count=0
    lookup={}
    for record in records:
        pad=pads[record['pad']]
        witness=next(w for w in candidate['actual_contacts'] if w['pad']==record['pad'])
        pv=moved_vertices(pad,witness['opening_m'])
        inner=float(pv[:,0].max() if pad['translation_per_opening'][0]<0 else pv[:,0].min())
        checks.check(abs(inner-record['inner_plane_x_m'])<=1e-12,label+': pad inner plane replay')
        lookup[record['pad']]={}
        for saved in record['polygons']:
            polygon_count+=1
            polygon=np.asarray(saved['polygon_local_m'])
            index=saved['triangle_index']
            lookup[record['pad']].setdefault(index,[]).append(polygon)
            checks.check(polygon.ndim==2 and polygon.shape[1]==3 and len(polygon)>=3 and np.isfinite(polygon).all(),label+': finite surface polygon')
            checks.check(polygon_area(polygon[:,1:])>1e-14,label+': positive projected surface region')
            checks.check(np.max(np.abs(polygon[:,0]-inner))<=band+1e-9,label+': polygon remains in stated gap band')
            checks.check(np.all(polygon[:,1:]>=pv.min(axis=0)[1:]+.00025-1e-9)
                         and np.all(polygon[:,1:]<=pv.max(axis=0)[1:]-.00025+1e-9),label+': region inside guarded finite pad')
            native=np.repeat(local_tri[index][None],len(polygon),axis=0)
            checks.check(float(triangle_distances(polygon,native).max())<=5e-9,label+': polygon vertices lie on recorded native triangle')
            normal=normals[index]
            checks.check(np.allclose(normal,saved['normal_local'],atol=1e-9),label+': polygon normal replay')
            normal_error=float(np.degrees(np.arccos(np.clip(abs(normal[0]),0,1))))
            checks.check(abs(normal_error-saved['pad_normal_error_deg'])<=1e-6 and normal_error<=tolerance+1e-7,label+': polygon surface-normal alignment')
    paired=np.asarray(candidate['paired_surface_points_world_m'])
    local_points=(paired-pose[:3,3])@pose[:3,:3]
    indices=candidate['paired_surface_triangle_indices']
    checks.check(paired.shape==(2,3) and np.isfinite(paired).all(),label+': finite paired near-contact surface points')
    checks.check(np.max(np.abs(local_points[0,1:]-local_points[1,1:]))<=1e-8,label+': paired surface points share hand Y/Z')
    checks.check(float(triangle_distances(paired,np.asarray(triangles)[indices]).max())<=5e-9,label+': paired points on native recorded triangles')
    common=np.asarray(candidate['common_region_polygon_yz_m'])
    checks.check(common.ndim==2 and common.shape[1]==2 and len(common)>=3 and polygon_area(common)>1e-14,label+': positive common region polygon')
    checks.check(inside_convex(local_points[:,1:],common),label+': paired point inside common region')
    selected_normals=[]
    for record,index,point in zip(records,indices,local_points):
        polygons=lookup[record['pad']].get(index,[])
        checks.check(bool(polygons) and any(inside_convex(common,p[:,1:]) for p in polygons),label+': common polygon contained in both selected pad regions')
        checks.check(any(inside_convex(point[None,1:],p[:,1:]) for p in polygons),label+': paired point in guarded pad region')
        checks.check(abs(point[0]-record['inner_plane_x_m'])<=band+1e-9,label+': paired point within pad gap band')
        selected_normals.append(normals[index])
    checks.check(float(selected_normals[0]@selected_normals[1])<=-np.cos(np.deg2rad(2*tolerance))+1e-9,label+': paired native surface normals oppose')
    center_offset=float(np.linalg.norm(local_points[0,1:]-[0.,candidate['pad_depth_m']]))
    checks.check(abs(center_offset-candidate['paired_surface_center_offset_m'])<=1e-9,label+': paired center-offset arithmetic')
    tangent_errors=None
    sector_widths=None
    perpendicular_error=None
    if cylinder:
        shaft=np.asarray(cylinder['shaft_axis_world'])
        origin=np.asarray(cylinder['point_on_axis_world'])
        checks.check(abs(np.linalg.norm(shaft)-1)<=1e-9,label+': fitted shaft axis normalized')
        perpendicular_error=float(np.degrees(np.arcsin(np.clip(abs(shaft@pose[:3,0]),0,1))))
        checks.check(abs(perpendicular_error-candidate['cylinder_closing_perpendicular_error_deg'])<=1e-6,
                     label+': shaft/closing perpendicular-error replay')
        checks.check(perpendicular_error<=1.+1e-7,label+': closing transverse to fitted shaft within 1 degree')
        vectors=paired-origin
        radial=vectors-(vectors@shaft)[:,None]*shaft
        radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-30)
        tangent_errors=np.degrees(np.arccos(np.clip(np.abs(radial@pose[:3,0]),0,1)))
        checks.check(np.allclose(tangent_errors,candidate['fitted_cylinder_tangent_errors_deg'],atol=1e-6),label+': analytic cylinder tangency replay')
        sector_widths=[]
        for index in indices:
            vertices=np.asarray(triangles)[index]-origin
            directions=vertices-(vertices@shaft)[:,None]*shaft
            directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-30)
            # Independently enumerate the three pairwise vertex angles.
            pairs=[float(np.degrees(np.arccos(np.clip(directions[a]@directions[b],-1,1))))
                   for a,b in ((0,1),(0,2),(1,2))]
            sector_widths.append(max(pairs))
        allowance=1.+np.asarray(sector_widths)
        checks.check(np.allclose(sector_widths,candidate['mesh_contact_sector_widths_deg'],atol=1e-6),
                     label+': native contact-triangle radial sector-width replay')
        checks.check(np.allclose(allowance,candidate['fitted_cylinder_tangent_allowances_deg'],atol=1e-6),
                     label+': declared fitted-tangency allowance includes only native sector plus 1 degree')
        checks.check(np.all(tangent_errors<=allowance+1e-7),label+': fitted radial error bounded by native triangle sector plus 1 degree')
    return dict(surface_failures=checks.failed_count-start,near_contact_polygons=polygon_count,
                maximum_native_normal_error_deg=float(errors.max()),
                maximum_fitted_tangent_error_deg=float(max(tangent_errors)) if tangent_errors is not None else None,
                maximum_native_contact_sector_width_deg=max(sector_widths) if sector_widths is not None else None,
                cylinder_closing_perpendicular_error_deg=perpendicular_error)


def audit_used_cylinders(row,candidates,triangles,config,checks):
    """Replay used native-vertex circle fits and small-band shaft support."""
    label=row['scene']+'/'+row['part']
    hypotheses={h['id']:h for h in row['direction_analysis']['cylinder_hypotheses']}
    used={s['cylinder_id'] for d in row['directions'] for s in d['sources'] if s['type']=='cylinder_radial'}
    used.update(c['cylinder_model']['id'] for c in candidates if c.get('cylinder_model'))
    used.update(m['cylinder_id'] for c in candidates for m in c.get('checked_cylinder_models',[]))
    used.update(hypotheses[i]['direction_support_cylinder_id'] for i in list(used)
                if hypotheses[i].get('classification_only'))
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas=np.linalg.norm(cross,axis=1)/2
    minimum_area=max(config['cylinder_min_band_area_m2'],areas.sum()*config['cylinder_min_surface_fraction'])
    reports=[]
    for identifier in sorted(used):
        model=hypotheses[identifier]
        name=label+'/'+identifier
        shaft=np.asarray(model['shaft_axis_world'])
        origin=np.asarray(model['point_on_axis_world'])
        indices=np.asarray(model['supporting_triangle_ids'],dtype=int)
        radius=model['fit_radius_m']
        checks.check(len(indices)==len(np.unique(indices)) and len(indices)>0,name+': unique nonempty native cylinder support')
        checks.check(np.isfinite(shaft).all() and abs(np.linalg.norm(shaft)-1)<=1e-9,name+': normalized fitted shaft')
        checks.check(np.isfinite(origin).all() and radius>0,name+': finite axis point and positive fitted radius')
        points=triangles[indices].reshape(-1,3)
        delta=points-origin
        radial=delta-(delta@shaft)[:,None]*shaft
        residual=np.abs(np.linalg.norm(radial,axis=1)-radius)
        limit=max(config['cylinder_max_circle_residual_m'],config['cylinder_max_circle_residual_fraction']*radius)
        r95=float(np.quantile(residual,.95))
        rmax=float(residual.max())
        checks.check(abs(r95-model['circle_vertex_residual95_m'])<=5e-10,name+': independent fitted radius residual95')
        checks.check(abs(rmax-model['circle_vertex_residual_max_m'])<=5e-10,name+': independent fitted radius residual maximum')
        checks.check(rmax<=limit+5e-10,name+': every native support vertex within declared circle residual limit')
        axial=points@shaft
        span=float(np.ptp(axial))
        support_area=float(areas[indices].sum())
        checks.check(abs(span-model['axial_span_m'])<=1e-9 and span>=config['cylinder_min_axial_span_m']-1e-10,
                     name+': native cylinder support axial span')
        checks.check(abs(axial.min()-model['axial_min_m'])<=1e-9 and abs(axial.max()-model['axial_max_m'])<=1e-9,
                     name+': recorded native axial bounds')
        checks.check(abs(support_area-model['area_m2'])<=1e-12,name+': native triangle support area replay')
        checks.check(abs(minimum_area-model['direction_area_gate_m2'])<=1e-12,name+': strong direction-support area threshold replay')
        normals=cross[indices]/(2*areas[indices,None])
        checks.check(np.max(np.abs(normals@shaft))<=np.sin(np.deg2rad(config['cylinder_normal_perpendicular_deg']))+1e-9,
                     name+': supported triangle normals perpendicular to fitted shaft')
        if model['supports_direction_generation']:
            checks.check(support_area>=minimum_area-1e-12 and model['classification_only'] is False,
                         name+': direction-generating cylinder has strong area support')
            checks.check(model['direction_support_cylinder_id'] is None,name+': primary shaft has no fabricated support link')
        else:
            support=hypotheses[model['direction_support_cylinder_id']]
            supporting_axis=np.asarray(support['shaft_axis_world'])
            supporting_origin=np.asarray(support['point_on_axis_world'])
            checks.check(model['classification_only'] is True and model['area_below_direction_gate'] is True
                         and support_area<minimum_area,name+': small band is classification-only')
            checks.check(support['supports_direction_generation'] is True,name+': classification-only band links to strong shaft')
            checks.check(abs(shaft@supporting_axis)>=np.cos(np.deg2rad(.25))-1e-10,name+': supporting shaft has same geometric direction')
            endpoints=origin+np.array([axial.min()-origin@shaft,axial.max()-origin@shaft])[:,None]*shaft
            error=float(np.linalg.norm(np.cross(endpoints-supporting_origin,supporting_axis),axis=1).max())
            checks.check(error<=limit+5e-10,name+': both small-band endpoints lie on supported shaft within residual bound')
            checks.check(abs(error-model['coaxial_max_axis_distance_m'])<=1e-9,name+': coaxial support-distance replay')
        reports.append(dict(id=identifier,classification_only=model['classification_only'],
                            radius_m=radius,native_vertex_residual95_m=r95,native_vertex_residual_max_m=rmax,
                            residual_limit_m=limit,support_area_m2=support_area,axial_span_m=span))
    for candidate in candidates:
        if not candidate.get('cylinder_model'):
            continue
        saved=candidate['cylinder_model']
        checks.check(saved==hypotheses[saved['id']],label+'/'+candidate['id']+': candidate uses unchanged declared cylinder model')
        support=set(saved['supporting_triangle_ids'])
        actual={w['first_contact']['triangle_index'] for w in candidate['actual_contacts']}
        checks.check(actual<=support,label+'/'+candidate['id']+': both actual first-contact triangles belong to classified cylinder')
        checks.check(set(candidate['paired_surface_triangle_indices'])<=support,
                     label+'/'+candidate['id']+': both paired surface triangles belong to classified cylinder')
    return reports


def verify(run_path):
    start=perf_counter()
    run=local(run_path)
    manifest=read(run/'manifest.json')
    rows=read(run/'results.json')
    inputs=read(local(manifest['input_manifest']))
    inventory={(r['scene'],r['part']):r for r in inputs['records']}
    hand=read(OUTPUT/'hand_model.json')
    pads={p['name']:p for p in hand['components'] if p['is_pad']}
    checks=Checks()
    keys=[(r['scene'],r['part']) for r in rows]
    full=manifest['cases']==96
    checks.check(len(inventory)==96,'frozen inventory has 96 observed placements')
    checks.check(len(keys)==len(set(keys))==manifest['cases'],'unique case count')
    checks.check(set(keys)<=set(inventory),'known observed placements only')
    if full:
        checks.check(set(keys)==set(inventory),'all eight-by-twelve placements audited')
    for flag in ('no_vlm','no_robot_execution','no_physics_steps','no_candidate_cap','no_score_ranking'):
        checks.check(manifest.get(flag) is True,'manifest: '+flag)
    validation=manifest['validation_config']
    baseline_manifest=read(OUTPUT/'runs/20261026T172654Z_surface_coverage/manifest.json')
    checks.check(validation==baseline_manifest['config'],'unchanged original geometry checker thresholds')
    protected={p:sha256(local(p))==digest for p,digest in manifest['protected_sha256'].items()}
    for path,same in protected.items():
        checks.check(same,'protected source/evidence unchanged: '+path)
    preservation=read(run/'preservation.json')
    checks.check(preservation['unchanged'] is True and preservation['sha256_after']==manifest['protected_sha256'],'run preservation arithmetic')
    critical={}
    for filename in ('run.py','directions.py','contacts.py','centering.py'):
        path=Path(__file__).with_name(filename)
        key=path.relative_to(REPO).as_posix()
        critical[key]=sha256(path)==manifest['source_sha256'].get(key)
        checks.check(critical[key],'generation source hash: '+key)
    for path,digest in inputs['source_sha256'].items():
        checks.check(sha256(local(path))==digest,'saved observed source unchanged: '+path)
    cases=[]
    for row in rows:
        name=row['scene']+'/'+row['part']
        expected=inventory[row['scene'],row['part']]
        for field in ('mesh_path','mesh_sha256','world_T_cad','table_plane_world','pose_source','candidate_id'):
            checks.check(row[field]==expected[field],name+': preserved observed input '+field)
        checks.check(row.get('simulator_object_poses_read') is False and row.get('associations_assumed') is True,name+': observed poses with supplied identity')
        checks.check(row['status'] in ('feasible_grasps_available','no_feasible_surface_grasp'),name+': explicit available/unavailable result')
        candidates=read(local(row['all_candidate_checks']))
        before=checks.failed_count
        contract=audit_contract(row,candidates,manifest['config'],checks)
        geometry=audit_geometry(row,candidates,hand,validation,checks)
        triangles=native_triangles(row)
        cylinder_fits=audit_used_cylinders(row,candidates,triangles,manifest['direction_config'],checks)
        surfaces=[audit_surface_regions(c,triangles,pads,checks,name+'/'+c['id']) for c in candidates if c['valid']]
        cases.append(dict(scene=row['scene'],part=row['part'],**contract,**geometry,
                          near_contact_polygons=sum(s['near_contact_polygons'] for s in surfaces),
                          maximum_native_normal_error_deg=max((s['maximum_native_normal_error_deg'] for s in surfaces),default=None),
                          maximum_fitted_tangent_error_deg=max((s['maximum_fitted_tangent_error_deg'] for s in surfaces if s['maximum_fitted_tangent_error_deg'] is not None),default=None),
                          maximum_native_contact_sector_width_deg=max((s['maximum_native_contact_sector_width_deg'] for s in surfaces if s['maximum_native_contact_sector_width_deg'] is not None),default=None),
                          maximum_cylinder_perpendicular_error_deg=max((s['cylinder_closing_perpendicular_error_deg'] for s in surfaces if s['cylinder_closing_perpendicular_error_deg'] is not None),default=None),
                          used_cylinder_models=cylinder_fits,
                          failed_checks=checks.failed_count-before))
        print(json.dumps(dict(audited=len(cases),total=len(rows),case=name,failed_checks=checks.failed_count)),flush=True)
    result=dict(status='failed' if checks.failed_count else 'passed_full_96' if full else 'passed_development_subset',
                checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,
                audited_cases=len(rows),full_96_case_inventory_verified=full and set(keys)==set(inventory),
                totals=dict(proposals=sum(r['generated_count'] for r in rows),valid_aliases=sum(r['valid_count'] for r in rows),
                            display_families=sum(r['grouped_count'] for r in rows),placements_with_options=sum(bool(r['grasps']) for r in rows)),
                protected_sources_unchanged=protected,critical_generation_sources_unchanged=critical,
                cases=cases,no_force_or_execution_claim=True,
                region_definition='Native triangle polygons within a 25 micrometer gap band and finite guarded pad; not loaded pressure/contact area.',
                audit_source_sha256=sha256(Path(__file__)),seconds=perf_counter()-start)
    (run/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','failed_checks','totals','seconds')}),flush=True)
    return result


if __name__=='__main__':
    result=verify(sys.argv[1])
    raise SystemExit(bool(result['failed_checks']))
