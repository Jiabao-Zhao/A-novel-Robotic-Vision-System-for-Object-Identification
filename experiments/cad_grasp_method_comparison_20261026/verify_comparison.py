"""Independent saved upstream-cache, adapter, finite-pad and table arithmetic audit.

This does not call the generator or repeat its full swept triangle kernel. The
original contact endpoints may differ in Z: only the hand closing axis is
projected horizontally. No fitted-cylinder or shared-region rule is required.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from collections import Counter
from pathlib import Path
import json
import sys
import time
import numpy as np
import open3d as o3d

from experiments.cad_grasp_geometry_20261026.inputs import local, read, sha256, OUTPUT, REPO
from experiments.cad_grasp_geometry_20261026.verify_results import Checks, moved_vertices, triangle_distances

PROTOCOLS=('midpoint_proj1','midpoint_proj15','panda_table_proj1','panda_table_proj15')
METHODS={'one_default','one_budget512','graspkit_512'}


def sparse_replay(pairs, distance):
    """Replay Fabrica's sequential two-endpoint neighborhood suppression."""
    pairs=np.asarray(pairs,float).reshape(-1,2,3)
    keep=np.ones(len(pairs),bool)
    for index in range(len(pairs)):
        if not keep[index]:
            continue
        near=np.linalg.norm(pairs-pairs[index],axis=2)<=distance
        suppressed=np.flatnonzero(near.all(axis=1))
        keep[suppressed[suppressed!=index]]=False
    return pairs[keep]


def replay_pairs(raw,pose,max_width,config):
    pairs=np.asarray(raw,float)@pose[:3,:3].T+pose[:3,3]
    pairs=pairs.copy()
    for pair in pairs:
        if tuple(pair[1])<tuple(pair[0]):
            pair[:]=pair[::-1].copy()
    widths=np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1)
    good=np.isfinite(pairs).all(axis=(1,2))&(widths>1e-6)&(widths<=max_width)
    width_count=int(good.sum())
    tilt=np.degrees(np.arcsin(np.clip(abs(pairs[:,1,2]-pairs[:,0,2])/np.maximum(widths,1e-30),0,1)))
    good&=tilt<=config['maximum_closing_projection_deg']
    eligible=pairs[good]
    return sparse_replay(eligible,config['pair_sparsity_m']),width_count,len(eligible)


def hand_rotation(x):
    down=np.array([0.,0.,-1.])
    return np.column_stack((x,np.cross(down,x),down))


def summarize(candidates,protocol):
    trials=[c for c in candidates if protocol in c['protocols']]
    passing=[c for c in trials if c['valid']]
    return dict(checked=len(trials),valid=len(passing),available=bool(passing),
                passing_ids=[c['id'] for c in passing],families=len({c['family_id'] for c in passing}),
                rejection_counts=dict(Counter(c['rejection_reason'] for c in trials if not c['valid'])))


def audit_counts(row,candidates,sparse,config,checks):
    label=f"{row['method']}/{row['scene']}/{row['part']}"
    primary=[c for c in candidates if config['primary_protocol'] in c['protocols']]
    passing=[c for c in primary if c['valid']]
    by_id={c['id']:c for c in candidates}
    checks.check(len(by_id)==len(candidates),f'{label}: repeated IDs')
    checks.check(row['generated_count']==len(candidates)==4*len(sparse),f'{label}: all-adapter generated count')
    checks.check(row['primary_generated_count']==len(primary)==2*len(sparse),f'{label}: primary generated count')
    checks.check(row['valid_count']==len(passing),f'{label}: primary valid count')
    checks.check(row['all_adapter_valid_count']==sum(c['valid'] for c in candidates),f'{label}: all-adapter valid count')
    checks.check(row['grouped_count']==len(row['grasps']),f'{label}: retained groups count')
    checks.check(row['status']==('feasible_grasps_available' if passing else 'no_feasible_upstream_pair_in_common_adapter'),f'{label}: status')
    for protocol in PROTOCOLS:
        checks.check(row['protocol_summary'][protocol]==summarize(candidates,protocol),f'{label}: {protocol} summary')
    checks.check(row['rejection_counts']==summarize(candidates,config['primary_protocol'])['rejection_counts'],f'{label}: primary rejections')
    checks.check(all((c['rejection_reason'] is None)==bool(c['valid']) for c in candidates),f'{label}: pass/reject flags')
    expected={(i,a,s) for i in range(1,len(sparse)+1) for a in ('midpoint','panda_table') for s in (-1.,1.)}
    actual=[(c['pair_index'],c['adapter'],c['jaw_sign']) for c in candidates]
    checks.check(len(actual)==len(set(actual)) and set(actual)==expected,f'{label}: both adapters and jaw assignments')
    members=[identifier for g in row['grasps'] for identifier in g['grouped_candidate_ids']]
    checks.check(len(members)==len(set(members))==len(passing) and set(members)=={c['id'] for c in passing},f'{label}: groups partition every primary pass once')
    for group in row['grasps']:
        ids=group['grouped_candidate_ids']
        good=[by_id[i] for i in ids if i in by_id]
        checks.check(bool(good) and group['id'] in ids,f'{label}: group representative membership')
        if not good:
            continue
        chosen=min(good,key=lambda c:(0 if c['jaw_sign']>0 else 1,c['opening_m'],c['id']))
        added={'grouped_candidate_ids','jaw_aliases','source_family_ids','representative_rule'}
        checks.check({k:v for k,v in group.items() if k not in added}==chosen,f'{label}: retained pose differs from checked pose')
        checks.check(group['jaw_aliases']==good,f'{label}: retained alias records changed')
        checks.check(group['source_family_ids']==sorted({c['family_id'] for c in good}),f'{label}: source families')
    return passing


class NativeMeshes:
    def __init__(self):
        self.meshes={}

    def get(self,row):
        if row['part'] not in self.meshes:
            mesh=o3d.io.read_triangle_mesh(str(local(row['mesh_path'])))
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            if row['part']=='Gear_Large':
                mesh.merge_close_vertices(5e-8)
            triangles=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
            scene=o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            self.meshes[row['part']]=(triangles,scene)
        return self.meshes[row['part']]

    def raw_distance(self,row,pairs):
        points=np.asarray(pairs,float).reshape(-1,3)
        if not len(points):
            return 0.
        triangles,scene=self.get(row)
        answer=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
        distances=triangle_distances(points,triangles[answer['primitive_ids'].numpy()])
        # Open3D can return an adjacent facet near seams/edges. An upper-bound
        # distance to that facet must not be mistaken for nearest CAD distance.
        # Refine suspect points against every original triangle independently.
        for index in np.flatnonzero(distances>5e-7):
            repeated=np.broadcast_to(points[index],(len(triangles),3))
            distances[index]=triangle_distances(repeated,triangles).min()
        return float(distances.max())


def audit_geometry(row,candidates,sparse,hand,config,validation,meshes,checks):
    label=f"{row['method']}/{row['scene']}/{row['part']}"
    components=hand['components']
    pads=sorted((p for p in components if p['is_pad']),key=lambda p:np.mean(p['vertices_zero_m'],axis=0)[0])
    pad_by_name={p['name']:p for p in pads}
    gap=max(np.asarray(pads[0]['vertices_zero_m'])[:,0])-min(np.asarray(pads[1]['vertices_zero_m'])[:,0])
    pad_center=(max(np.asarray(p['vertices_zero_m'])[:,2].min() for p in pads)+min(np.asarray(p['vertices_zero_m'])[:,2].max() for p in pads))/2
    plane=np.asarray(row['table_plane_world'])
    cad_pose=np.asarray(row['world_T_cad'])
    triangles,_=meshes.get(row)
    before=checks.failed_count
    max_witness_distance=0.
    min_table=None
    center_queries=0
    for candidate in candidates:
        name=f"{label}/{candidate['id']}"
        pair=sparse[candidate['pair_index']-1]
        difference=pair[1]-pair[0]
        width=np.linalg.norm(difference)
        x=difference.copy(); x[2]=0.; x/=np.linalg.norm(x)
        tilt=float(np.degrees(np.arcsin(abs(difference[2])/width)))
        protocols={candidate['adapter']+'_proj15'}
        if tilt<=config['strict_projection_subset_deg']:
            protocols.add(candidate['adapter']+'_proj1')
        checks.check(set(candidate['protocols'])==protocols,f'{name}: projection protocol subset')
        checks.check(abs(candidate['native_pair_projection_deg']-tilt)<1e-10,f'{name}: projection angle')
        checks.check(candidate['family_id']==f"PAIR{candidate['pair_index']:04d}",f'{name}: family ID')
        checks.check(np.allclose(candidate['contacts_world_m'],pair,atol=1e-14,rtol=0),f'{name}: original pair changed')
        pose=np.asarray(candidate['world_T_grip']); rotation=pose[:3,:3]
        checks.check(pose.shape==(4,4) and np.isfinite(pose).all(),f'{name}: finite pose')
        checks.check(np.allclose(pose[3],[0,0,0,1],atol=1e-12),f'{name}: homogeneous row')
        checks.check(np.allclose(rotation,hand_rotation(candidate['jaw_sign']*x),atol=1e-12,rtol=0),f'{name}: exact-down projected hand frame')
        checks.check(np.allclose(rotation.T@rotation,np.eye(3),atol=1e-12) and abs(np.linalg.det(rotation)-1)<1e-12,f'{name}: proper rotation')
        opening=width+gap+config['extra_opening_m']
        checks.check(abs(candidate['width_m']-width)<1e-12 and abs(candidate['opening_m']-opening)<1e-12,f'{name}: width/opening arithmetic')
        checks.check(0<=opening<=hand['opening_limits_m'][1]+1e-12,f'{name}: physical opening range')
        midpoint=pair.mean(axis=0); expected=midpoint.copy(); expected[2]+=pad_center
        projection=candidate['center_projection']
        if candidate['adapter']=='midpoint':
            checks.check(projection is None,f'{name}: midpoint control unexpectedly centered')
        else:
            required=-np.inf
            xy=np.r_[midpoint[:2],0.]
            for sign in (-1.,1.):
                for component in components:
                    for at_opening in (0.,opening):
                        points=moved_vertices(component,at_opening)@hand_rotation(sign*x).T+xy
                        required=max(required,(validation['table_clearance_m']-(points@plane[:3]+plane[3]).min())/plane[2])
            expected[2]=max(expected[2],required+.00002)
            before_pose=np.eye(4);before_pose[:3,:3]=hand_rotation(x);before_pose[:3,3]=expected
            checks.check(projection is not None and np.allclose(projection['world_T_grip_before'],before_pose,atol=1e-12,rtol=0),f'{name}: table-clamped initial frame')
            if projection:
                if candidate['jaw_sign']>0:
                    center_queries+=projection['geometry_queries']
                values=projection['first_contact_openings_before_m']
                shift=(values[1]-values[0])/4 if projection['applied'] else 0.
                checks.check(abs(shift-projection['translation_along_closing_m'])<1e-12,f'{name}: first-contact centering equation')
                checks.check(projection['geometry_queries']==(2 if projection['applied'] else len(values)+1),f'{name}: projection query count')
                if projection['applied']:
                    checks.check(abs(projection['expected_equal_contact_opening_m']-np.mean(values))<1e-12,f'{name}: projected equal opening')
                expected+=shift*x
        checks.check(np.allclose(pose[:3,3],expected,atol=1e-12,rtol=0),f'{name}: adapter translation')
        checks.check(abs(candidate['pad_depth_m']-(expected[2]-midpoint[2]))<1e-12,f'{name}: pad-depth arithmetic')
        if not candidate['valid']:
            continue
        checks.check(all(np.asarray(p['vertices_zero_m'])[:,2].min()+validation['pad_edge_margin_m']<=candidate['pad_depth_m']<=np.asarray(p['vertices_zero_m'])[:,2].max()-validation['pad_edge_margin_m'] for p in pads),f'{name}: original finite-pad depth margin')
        table=min(float(((moved_vertices(c,o)@rotation.T+pose[:3,3])@plane[:3]+plane[3]).min()) for c in components for o in (0.,opening))
        min_table=table if min_table is None else min(min_table,table)
        checks.check(table>=validation['table_clearance_m']-1e-10 and abs(table-candidate['table_clearance_m'])<=1e-9,f'{name}: finite original-hand table clearance')
        openings=candidate['first_contact_openings_m']; mismatch=abs(openings[0]-openings[1])
        checks.check(mismatch<=validation['contact_opening_difference_m']+1e-12 and abs(mismatch-candidate['contact_opening_difference_m'])<1e-12,f'{name}: simultaneous first contact')
        checks.check(abs(max(openings)-candidate['contact_opening_m'])<1e-12,f'{name}: earlier stop opening')
        witnesses=candidate['actual_contacts']
        checks.check(len(witnesses)==2 and {w['pad'] for w in witnesses}==set(pad_by_name),f'{name}: both finite-pad contacts')
        checks.check(np.allclose(candidate['contact_points_world_m'],[w['point_world_m'] for w in witnesses],atol=1e-12,rtol=0),f'{name}: witness point summary')
        for witness in witnesses:
            pad=pad_by_name[witness['pad']]
            v=moved_vertices(pad,witness['opening_m'])
            point=(np.asarray(witness['point_world_m'])-pose[:3,3])@rotation
            tolerance=validation['contact_inner_face_tolerance_m']+2e-8
            inner=v[:,0].max() if pad['translation_per_opening'][0]<0 else v[:,0].min()
            checks.check(np.isfinite(point).all() and abs(point[0]-inner)<=tolerance and np.all(point>=v.min(0)-tolerance) and np.all(point<=v.max(0)+tolerance),f'{name}: finite inner-pad witness')
            checks.check(witness['normal_alignment']>=np.cos(np.deg2rad(validation['normal_angle_deg']))-1e-12,f'{name}: original normal angle')
            first=witness['first_contact']
            checks.check(first['lower_fraction']<=first['fraction']<=first['upper_fraction'] and first['upper_fraction']-first['lower_fraction']<=validation['contact_search_fraction_tolerance']+1e-12,f'{name}: first-contact bracket')
            checks.check(abs(witness['opening_m']-opening*(1-first['fraction']))<1e-12,f'{name}: witness opening fraction')
            cad_point=(np.asarray(witness['point_world_m'])-cad_pose[:3,3])@cad_pose[:3,:3]
            triangle=triangles[first['triangle_index']]
            normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
            alignment=abs(float((normal@cad_pose[:3,:3].T)@rotation[:,0]))
            checks.check(abs(alignment-witness['normal_alignment'])<=1e-10,f'{name}: independently reconstructed native triangle normal')
            distance=float(triangle_distances(cad_point[None],triangle[None])[0])
            max_witness_distance=max(max_witness_distance,distance)
            checks.check(distance<=5e-9,f'{name}: witness lies on its recorded native CAD triangle')
    checks.check(row['center_projection_geometry_queries']==center_queries,f'{label}: geometry-query arithmetic')
    return dict(audit_failures=checks.failed_count-before,maximum_witness_triangle_distance_m=max_witness_distance,minimum_table_clearance_m=min_table)


def verify(run_path):
    started=time.perf_counter();run=local(run_path)
    manifest=read(run/'manifest.json');results=read(run/'results.json')
    checks=Checks();config=manifest['config'];validation=manifest['validation_config']
    inputs=read(local(manifest['input_manifest']));input_rows={(r['scene'],r['part']):r for r in inputs['records']}
    cache=local(manifest['intrinsic_cache']);cache_manifest=read(cache/'manifest.json');cached=read(cache/'records.json')
    cache_lookup={(r['method'],r['part']):r for r in cached}
    hand=read(OUTPUT/'hand_model.json')
    pads=sorted((p for p in hand['components'] if p['is_pad']),key=lambda p:np.mean(p['vertices_zero_m'],axis=0)[0])
    gap=max(np.asarray(pads[0]['vertices_zero_m'])[:,0])-min(np.asarray(pads[1]['vertices_zero_m'])[:,0])
    max_width=hand['opening_limits_m'][1]-gap-config['extra_opening_m']
    full=manifest['cases_per_method']==96
    checks.check(len(input_rows)==96 and len(cached)==36 and set(manifest['methods'])==METHODS,'96 placements / 36 intrinsic records / 3 methods')
    keys=[(r['method'],r['scene'],r['part']) for r in results]
    checks.check(len(keys)==len(set(keys))==3*manifest['cases_per_method'],'comparison case inventory')
    if full:
        checks.check(set(keys)=={(m,s,p) for m in METHODS for s,p in input_rows},'full 288-case cross product')
    checks.check(tuple(manifest['protocols'])==PROTOCOLS and config['primary_protocol']=='panda_table_proj15','four protocols and primary')
    reference=read(local(manifest['reference_run'])/'manifest.json')
    checks.check(validation==reference['validation_config'],'original unchanged core validation configuration')
    for flag in ('no_vlm','no_robot_execution','no_physics_steps','no_our_shared_region_guard','no_our_cylinder_classification','no_our_flat_one_degree_filter'):
        checks.check(manifest.get(flag) is True,f'manifest declaration: {flag}')
    preserved={}
    for group in ('source_sha256','protected_sha256'):
        for path,digest in manifest[group].items():
            same=sha256(local(path))==digest;preserved[path]=same
            checks.check(same,f'{group} changed: {path}')
    preservation=read(run/'preservation.json')
    checks.check(preservation['unchanged'] is True and preservation['sha256_after']==manifest['protected_sha256'],'protected preservation record')
    for path,digest in cache_manifest['source_sha256'].items():
        checks.check(sha256(local(path))==digest,f'intrinsic generation source changed: {path}')
    for path,digest in inputs['source_sha256'].items():
        checks.check(sha256(local(path))==digest,f'observed input source changed: {path}')
    meshes=NativeMeshes();raw_cache={};cache_audits=[]
    for record in cached:
        key=(record['method'],record['part']);data=np.load(local(record['npz_path']));raw=data['pairs_m'];normals=data['normals'] if 'normals' in data.files else data['normals_m']
        raw_cache[key]=raw
        checks.check(record['status']=='sampled' and record['frame']=='native_CAD' and record['units']=='meters',f'{key}: cache coordinate contract')
        checks.check(raw.shape==(record['raw_pairs'],2,3) and normals.shape==raw.shape and np.isfinite(raw).all() and np.isfinite(normals).all(),f'{key}: cached pair/normal shape and finiteness')
        checks.check(not len(normals) or np.allclose(np.linalg.norm(normals,axis=2),1.,atol=2e-6),f'{key}: original cached unit normals')
        checks.check(sha256(local(record['mesh_path']))==record['mesh_sha256'],f'{key}: native mesh hash')
        matching=next(r for r in inputs['records'] if r['part']==record['part'])
        distance=meshes.raw_distance(matching,raw)
        checks.check(distance<=2e-6,f'{key}: cached native endpoints on CAD (2 micron numerical diagnostic)')
        protected_cache=all(record[field] in manifest['protected_sha256'] for field in ('npz_path','metadata_path'))
        if full:
            checks.check(protected_cache,f'{key}: full run must protect original pair and normal cache bytes')
        # Pilot predates the protection expansion; report this provenance gap.
        cache_audits.append(dict(method=record['method'],part=record['part'],pairs=len(raw),maximum_native_surface_distance_m=distance,raw_data_protected_by_run_manifest=protected_cache))
    audits=[]
    for number,row in enumerate(results,1):
        label=f"{row['method']}/{row['scene']}/{row['part']}";before=checks.failed_count
        original=input_rows[row['scene'],row['part']]
        for field in ('mesh_path','mesh_sha256','world_T_cad','table_plane_world','pose_source','candidate_id'):
            checks.check(row[field]==original[field],f'{label}: observed input changed: {field}')
        checks.check(row.get('simulator_object_poses_read') is False and row.get('associations_assumed') is True,f'{label}: observed pose source/association flags')
        if row['status']=='experiment_error':
            checks.check(False,f'{label}: experiment error {row.get("error")}');continue
        intrinsic=cache_lookup[row['method'],row['part']]
        checks.check(row['intrinsic_record']==intrinsic,f'{label}: raw intrinsic record changed')
        checks.check(row['input_frame_of_cached_pairs']=='native_CAD_meters',f'{label}: cached coordinate frame')
        raw=raw_cache[row['method'],row['part']]
        sparse,width_count,projection_count=replay_pairs(raw,np.asarray(row['world_T_cad']),max_width,config)
        saved=np.asarray(read(local(row['all_candidate_checks']).with_name('sparse_world_pairs.json')),float).reshape(-1,2,3)
        checks.check(saved.shape==sparse.shape and np.allclose(saved,sparse,atol=1e-14,rtol=0),f'{label}: native-to-world and original 5mm sparsity replay')
        checks.check(row['native_pairs']==len(raw) and row['width_qualified_pairs']==width_count and row['horizontal_projection_qualified_pairs']==projection_count and row['sparse_pairs']==len(sparse),f'{label}: proposal reduction stage counts')
        candidates=read(local(row['all_candidate_checks']))
        audit_counts(row,candidates,sparse,config,checks)
        geometry=audit_geometry(row,candidates,sparse,hand,config,validation,meshes,checks)
        audits.append(dict(method=row['method'],scene=row['scene'],part=row['part'],audit_failures=checks.failed_count-before,**{k:v for k,v in geometry.items() if k!='audit_failures'}))
        print(json.dumps(dict(audited=number,total=len(results),failed=checks.failed_count,case=label)),flush=True)
    totals={method:{p:dict(checked=sum(r['protocol_summary'][p]['checked'] for r in results if r['method']==method),valid=sum(r['protocol_summary'][p]['valid'] for r in results if r['method']==method),available=sum(r['protocol_summary'][p]['available'] for r in results if r['method']==method)) for p in PROTOCOLS} for method in METHODS}
    report=dict(status='failed' if checks.failed_count else 'passed_full_288' if full else 'passed_development_subset',audited_cases=len(audits),checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,full_288_inventory=full,cache_audits=cache_audits,cases=audits,protocol_totals=totals,preserved_hashes=preserved,audit_source_sha256=sha256(Path(__file__)),seconds=time.perf_counter()-started,
        limitations=['Does not repeat the complete swept triangle kernel or prove pickup, friction, assembly or robot motion.','Raw CAD samplers and common adapters are compared; complete upstream frameworks are not executed.','Raw endpoint Z can differ after horizontal projection; actual first-contact witnesses are checked on finite native pads and CAD triangles.','Original normals are retained in the intrinsic cache; they are not substituted for actual first-contact normal checks.','Runtime includes both conversion adapters and their two jaw assignments. Intrinsic sampling happens once per CAD; no pose-estimation, VLM or gallery time.'])
    (run/'comparison_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','audited_cases','checks','failed_checks','seconds')}),flush=True)
    return report


if __name__=='__main__':
    outcome=verify(sys.argv[1]);raise SystemExit(bool(outcome['failed_checks']))
