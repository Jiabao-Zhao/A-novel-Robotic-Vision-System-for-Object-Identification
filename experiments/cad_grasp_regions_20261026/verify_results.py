"""Independent compact-set contract and saved native-contact geometry audit.

This replays saved coordinates and counts. It does not rerun the collision
kernel, simulate contact forces, or establish pickup/assembly success.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from experiments.cad_grasp_geometry_20261026.inputs import OUTPUT, REPO, local, read, sha256
from experiments.cad_grasp_geometry_20261026.verify_results import (
    Checks, moved_vertices, surface_distances, transform,
    WITNESS_TRIANGLE_NUMERICAL_TOLERANCE_M,
)


def same_physical_grip(first, second, tolerance_m=1e-6, tolerance_rotation=1e-6):
    """Physical equality ignores seed depth/triangle labels, but retains yaw."""
    a,b = np.asarray(first['world_T_grip']),np.asarray(second['world_T_grip'])
    return (np.linalg.norm(a[:3,3]-b[:3,3])<=tolerance_m
            and np.max(np.abs(a[:3,:3]-b[:3,:3]))<=tolerance_rotation
            and abs(first['contact_opening_m']-second['contact_opening_m'])<=tolerance_m)


def audit_contract(row, candidates, config, checks):
    """No point/depth Cartesian product or hidden highest-score selection."""
    label = row['scene']+'/'+row['part']
    before = checks.failed_count
    axes = {axis['label']:axis for axis in row['axes']}
    rotation = np.asarray(row['world_T_cad'])[:3,:3]
    checks.check(set(axes)=={'CAD_X','CAD_Y','CAD_Z'}, label+': exactly three declared CAD axes')
    for axis in axes.values():
        index = axis['cad_index']
        original = rotation[:,index]
        error = float(np.degrees(np.arcsin(np.clip(abs(original[2]),0,1))))
        usable = error<=config['axis_horizontal_tolerance_deg']
        checks.check(np.allclose(original,axis['cad_axis_world'],atol=1e-12), label+': declared native CAD axis')
        checks.check(abs(error-axis['projection_error_deg'])<=1e-9, label+': CAD projection-error arithmetic')
        checks.check(axis['usable']==usable, label+': horizontal-axis availability')
        if usable:
            projected = original.copy()
            projected[2]=0.
            projected/=np.linalg.norm(projected)
            checks.check(np.allclose(projected,axis['closing_axis_world'],atol=1e-12),
                         label+': horizontal projection is explicit and not arbitrary yaw')
    families = row['contact_families']
    family_by_id = {family['family_id']:family for family in families}
    checks.check(len(families)==len(family_by_id), label+': unique contact family IDs')
    checks.check(row['merged_contact_regions']==len(families), label+': merged region count')
    prepared = [family for family in families if family['prepared']]
    checks.check(row['prepared_contact_families']==len(prepared), label+': prepared family count')
    checks.check(len(candidates)==2*len(prepared), label+': two jaw aliases, one height per prepared region')
    checks.check(row['generated_count']==len(candidates), label+': generated count')
    by_id = {candidate['id']:candidate for candidate in candidates}
    checks.check(len(by_id)==len(candidates), label+': unique candidate IDs')
    valid = [candidate for candidate in candidates if candidate['valid']]
    rejected = [candidate for candidate in candidates if not candidate['valid']]
    checks.check(row['valid_count']==len(valid), label+': valid alias count')
    checks.check(Counter(c['rejection_reason'] for c in rejected)==row['rejection_counts'], label+': rejection counts')
    checks.check(all(c['rejection_reason'] is None for c in valid), label+': valid alias has no rejection')
    checks.check(all(isinstance(c['rejection_reason'],str) for c in rejected), label+': rejected alias has a reason')
    expected_skips=Counter(f['preparation_rejection'] for f in families if not f['prepared'])
    checks.check(expected_skips==row['preparation_rejection_counts'],label+': unavailable-family reasons')
    covered=[]
    for family in families:
        name=label+'/'+family['family_id']
        axis=axes[family['axis_label']]
        checks.check(axis['usable'],name+': family belongs to available axis')
        aliases=[c for c in candidates if c['family_id']==family['family_id']]
        if not family['prepared']:
            checks.check(not aliases and isinstance(family.get('preparation_rejection'),str),name+': skipped family has no invented pose')
            if family.get('preparation_rejection')=='duplicate_physical_hand_placement':
                source=family_by_id.get(family.get('duplicate_of'))
                checks.check(source is not None and source['prepared'],name+': duplicate points to a prepared placement')
            continue
        checks.check(len(aliases)==2 and {c['jaw_sign'] for c in aliases}=={-1.,1.},name+': both jaw assignments checked')
        checks.check(family['candidate_ids']==[c['id'] for c in aliases],name+': candidate membership')
        checks.check(family['valid_alias_ids']==[c['id'] for c in aliases if c['valid']],name+': valid alias membership')
        covered.extend(c['id'] for c in aliases)
        for candidate in aliases:
            pose=np.asarray(candidate['world_T_grip'])
            x=candidate['jaw_sign']*np.asarray(axis['closing_axis_world'])
            checks.check(candidate['axis_label']==family['axis_label'] and candidate['cad_axis_index']==axis['cad_index'],
                         name+': candidate axis mapping')
            checks.check(np.allclose(pose[:3,0],x,atol=1e-12),name+': candidate closes along signed declared axis')
            checks.check(abs(candidate['cad_axis_projection_error_deg']-axis['projection_error_deg'])<=1e-9,
                         name+': candidate declares actual CAD-axis error')
            checks.check(abs(pose[2,3]-family['height_m'])<=1e-12,name+': one physical hand height')
            checks.check(abs(candidate['pad_depth_m']-family['pad_depth_m'])<=1e-12,name+': no depth-label product')
        if len(aliases)==2:
            checks.check(np.allclose(np.asarray(aliases[0]['world_T_grip'])[:3,3],
                                     np.asarray(aliases[1]['world_T_grip'])[:3,3],atol=1e-12),name+': jaw exchange preserves position')
    checks.check(len(covered)==len(set(covered))==len(candidates) and set(covered)==set(by_id),label+': families partition every generated alias')
    groups=row['grasps']
    checks.check(row['grouped_count']==len(groups),label+': display family count')
    expected_families={c['family_id'] for c in valid}
    members=[]
    source_members=[]
    for group in groups:
        name=label+'/'+group['id']
        ids=group['grouped_candidate_ids']
        aliases=[by_id[i] for i in ids]
        members.extend(ids)
        checks.check(bool(aliases) and all(c['valid'] for c in aliases),name+': saved alternatives are valid checked aliases')
        sources=sorted({c['family_id'] for c in aliases})
        source_members.extend(sources)
        checks.check(group['source_family_ids']==sources,name+': all contributing contact regions retained')
        for source in sources:
            expected={c['id'] for c in valid if c['family_id']==source}
            checks.check(expected<=set(ids),name+': source region keeps every valid jaw assignment')
        physical=[]
        for alias in aliases:
            if not any(same_physical_grip(alias,other) for other in physical):
                physical.append(alias)
        checks.check(len(physical)<=2,name+': at most two distinct physical final jaw orientations')
        if len(physical)==2:
            a,b=(np.asarray(c['world_T_grip']) for c in physical)
            checks.check(np.linalg.norm(a[:3,3]-b[:3,3])<=config.get('physical_duplicate_position_m',1e-6)
                         and np.allclose(a[:3,0],-b[:3,0],atol=1e-6),
                         name+': alternate physical hand pose is only the exchanged jaw assignment')
        chosen=min(aliases,key=lambda c:(0 if c['jaw_sign']>0 else 1,c['opening_m'],c['id']))
        expected=dict(chosen,grouped_candidate_ids=ids,jaw_aliases=aliases,
                      source_family_ids=sources,
                      representative_rule='canonical_jaw_then_smallest_checked_approach_opening')
        checks.check(group==expected,name+': canonical display record is a checked alias, not score selection')
        checks.check(group['jaw_aliases']==aliases,name+': actual alternate hand poses retained')
    checks.check(len(members)==len(set(members))==len(valid) and set(members)=={c['id'] for c in valid},
                 label+': display families partition every valid alias')
    checks.check(len(source_members)==len(set(source_members)) and set(source_members)==expected_families,
                 label+': every valid contact region belongs to one display group')
    duplicate_pairs=[]
    for i,first in enumerate(groups):
        for second in groups[:i]:
            if same_physical_grip(first,second):
                duplicate_pairs.append([first['id'],second['id']])
    checks.check(not duplicate_pairs,label+': repeated physical grip despite different metadata')
    return dict(families=len(families),prepared_families=len(prepared),proposals=len(candidates),
                valid_aliases=len(valid),display_families=len(groups),exact_physical_duplicate_pairs=duplicate_pairs,
                contract_failures=checks.failed_count-before)


def audit_geometry(row, candidates, hand, config, checks):
    """Reconstruct finite pad contacts/table clearance without the generator."""
    label = f"{row['scene']}/{row['part']}"
    components = hand['components']
    pads = {pad['name']:pad for pad in components if pad['is_pad']}
    ordered = sorted(pads.values(), key=lambda p:np.mean(p['vertices_zero_m'], axis=0)[0])
    gap_offset = float(np.asarray(ordered[0]['vertices_zero_m'])[:,0].max()
                       - np.asarray(ordered[1]['vertices_zero_m'])[:,0].min())
    plane = np.asarray(row['table_plane_world'])
    nominal, actual, triangle_ids = [], [], []
    rotation_errors, table_clearances, mismatch_values, table_errors = [], [], [], []
    for candidate in candidates:
        name = label+'/'+candidate['id']
        pose = np.asarray(candidate['world_T_grip'], dtype=float)
        contacts = np.asarray(candidate['contacts_world_m'], dtype=float)
        checks.check(pose.shape==(4,4) and np.isfinite(pose).all(), name+': finite 4x4 pose')
        if pose.shape!=(4,4) or not np.isfinite(pose).all():
            continue
        rotation = pose[:3,:3]
        error = float(np.abs(rotation.T@rotation-np.eye(3)).max())
        rotation_errors.append(error)
        checks.check(error<=1e-10 and abs(np.linalg.det(rotation)-1)<=1e-10, name+': proper rigid rotation')
        checks.check(np.allclose(pose[3],[0,0,0,1],atol=1e-12), name+': homogeneous final row')
        checks.check(np.allclose(rotation[:,2],[0,0,-1],atol=1e-12), name+': world-down approach')
        checks.check(contacts.shape==(2,3) and np.isfinite(contacts).all(), name+': finite contact pair')
        if contacts.shape!=(2,3) or not np.isfinite(contacts).all():
            continue
        nominal.extend(contacts)
        checks.check(abs(contacts[0,2]-contacts[1,2])<=1e-9, name+': horizontal nominal contacts')
        width = float(np.linalg.norm(contacts[1]-contacts[0]))
        checks.check(abs(width-candidate['width_m'])<=5e-8, name+': contact width arithmetic')
        opening = candidate['opening_m']
        checks.check(abs(opening-width-gap_offset-config['extra_opening_m'])<=5e-8,
                     name+': commanded opening includes actual pad offset and full extra opening')
        checks.check(hand['opening_limits_m'][0]-1e-12<=opening<=hand['opening_limits_m'][1]+1e-12,
                     name+': physical commanded opening limits')
        # Hand approach X is exactly the horizontal line through the samples.
        direction = (contacts[1]-contacts[0])/width
        checks.check(abs(float(direction@rotation[:,0]))>=1-1e-10, name+': fingers close along sampled contact line')
        if not candidate['valid']:
            continue
        for pad in pads.values():
            pv = np.asarray(pad['vertices_zero_m'])
            checks.check(pv[:,2].min()+config['pad_edge_margin_m']-1e-10
                         <=candidate['pad_depth_m']<=pv[:,2].max()-config['pad_edge_margin_m']+1e-10,
                         name+': nominal contact inside finite pad depth')
        distances = [float((transform(moved_vertices(component,o),pose)@plane[:3]+plane[3]).min())
                     for component in components for o in (0.,opening)]
        smallest = min(distances)
        table_clearances.append(smallest)
        table_errors.append(abs(smallest-candidate['table_clearance_m']))
        checks.check(smallest>=config['table_clearance_m']-1e-10, name+': independent full-vertex table clearance')
        checks.check(abs(smallest-candidate['table_clearance_m'])<=1e-9, name+': stored table clearance replay')
        openings = candidate['first_contact_openings_m']
        mismatch = abs(openings[0]-openings[1])
        mismatch_values.append(mismatch)
        checks.check(mismatch<=config['contact_opening_difference_m']+1e-12, name+': compatible first-contact openings')
        checks.check(abs(mismatch-candidate['contact_opening_difference_m'])<=1e-12, name+': opening mismatch arithmetic')
        checks.check(abs(max(openings)-candidate['contact_opening_m'])<=1e-12, name+': stop at earlier contact')
        witnesses = candidate['actual_contacts']
        checks.check(len(witnesses)==2 and {w['pad'] for w in witnesses}==set(pads), name+': both pad witnesses')
        checks.check(candidate['contact_points_world_m']==[w['point_world_m'] for w in witnesses],
                     name+': displayed actual contacts equal native witnesses')
        for witness in witnesses:
            pad = pads[witness['pad']]
            pv = moved_vertices(pad,witness['opening_m'])
            point = (np.asarray(witness['point_world_m'])-pose[:3,3])@rotation
            inner_x = pv[:,0].max() if pad['translation_per_opening'][0]<0 else pv[:,0].min()
            tolerance = config['contact_inner_face_tolerance_m']+2e-8
            checks.check(abs(point[0]-inner_x)<=tolerance, name+': actual witness on inner pad face')
            checks.check(np.all(point>=pv.min(0)-tolerance) and np.all(point<=pv.max(0)+tolerance),
                         name+': actual witness inside finite pad')
            checks.check(witness['normal_alignment']>=np.cos(np.deg2rad(config['normal_angle_deg']))-1e-12,
                         name+': actual surface faces finger closing')
            first = witness['first_contact']
            checks.check(first['lower_fraction']<=first['fraction']<=first['upper_fraction'], name+': first-contact bracket')
            checks.check(first['upper_fraction']-first['lower_fraction']<=config['contact_search_fraction_tolerance']+1e-12,
                         name+': bounded first-contact bracket')
            checks.check(abs(witness['opening_m']-opening*(1-first['fraction']))<=1e-12,
                         name+': fraction/opening arithmetic')
            actual.append(witness['point_world_m'])
            triangle_ids.append(first['triangle_index'])
    surfaces = surface_distances(row,nominal,actual,triangle_ids)
    if surfaces['actual_first_contact']['unique_points']:
        checks.check(surfaces['actual_first_contact']['recorded_native_triangle_maximum_distance_m']
                     <=WITNESS_TRIANGLE_NUMERICAL_TOLERANCE_M,
                     label+': actual contacts lie on recorded native CAD triangles')
    return dict(maximum_rotation_error=max(rotation_errors,default=0.),
                minimum_table_clearance_m=min(table_clearances,default=None),
                maximum_table_replay_error_m=max(table_errors,default=0.),
                maximum_contact_opening_mismatch_m=max(mismatch_values,default=0.),
                surface_distance_audit=surfaces)


def verify(run_path):
    start=perf_counter()
    run=local(run_path)
    manifest=read(run/'manifest.json')
    rows=read(run/'results.json')
    inputs=read(local(manifest['input_manifest']))
    input_rows={(r['scene'],r['part']):r for r in inputs['records']}
    hand=read(OUTPUT/'hand_model.json')
    config=manifest['config']
    validation=manifest['validation_config']
    baseline=local(manifest['baseline_run'])
    baseline_manifest=read(baseline/'manifest.json')
    checks=Checks()
    keys=[(r['scene'],r['part']) for r in rows]
    checks.check(len(input_rows)==96,'full observed-input inventory has 96 cases')
    checks.check(len(keys)==len(set(keys))==manifest['cases'],'case count and uniqueness')
    checks.check(set(keys)<=set(input_rows),'known input scene/part pairs only')
    full=manifest['cases']==96
    if full:
        checks.check(set(keys)==set(input_rows),'all eight-by-twelve cases checked')
    checks.check(validation==baseline_manifest['config'],'old geometric checker thresholds unchanged')
    checks.check(config['contact_depths_per_region']==1 and config['jaw_aliases_per_region']==2,
                 'one region height and two jaw assignments declared')
    for flag in ('no_vlm','no_robot_execution','no_physics_steps','no_candidate_cap','no_score_ranking'):
        checks.check(manifest.get(flag) is True,'manifest declaration: '+flag)
    protected={}
    for path,digest in manifest['protected_sha256'].items():
        protected[path]=sha256(local(path))==digest
        checks.check(protected[path],'protected source/evidence unchanged: '+path)
    preservation=read(run/'preservation.json')
    checks.check(preservation['unchanged'] is True,'run preservation succeeded')
    checks.check(preservation['sha256_after']==manifest['protected_sha256'],'preservation arithmetic')
    critical={}
    for filename in ('run.py','regions.py'):
        source=Path(__file__).with_name(filename)
        key=source.relative_to(REPO).as_posix()
        critical[key]=sha256(source)==manifest['source_sha256'].get(key)
        checks.check(critical[key],'critical compact-generation source: '+key)
    frozen_input=True
    for path,digest in inputs['source_sha256'].items():
        same=sha256(local(path))==digest
        frozen_input &=same
        checks.check(same,'frozen observed input: '+path)
    cases=[]
    for index,row in enumerate(rows,1):
        label=row['scene']+'/'+row['part']
        expected=input_rows[row['scene'],row['part']]
        for field in ('mesh_path','mesh_sha256','world_T_cad','table_plane_world','pose_source','candidate_id'):
            checks.check(row[field]==expected[field],label+': original observed input field '+field)
        checks.check(row.get('simulator_object_poses_read') is False,label+': no simulator object pose')
        checks.check(row.get('associations_assumed') is True,label+': supplied CAD/crop association')
        checks.check(row['status'] in ('feasible_grasps_available','no_feasible_centered_CAD_axis_grasp'),
                     label+': explicit available/unavailable status')
        candidates=read(local(row['all_candidate_checks']))
        before=checks.failed_count
        contract=audit_contract(row,candidates,config,checks)
        geometry=audit_geometry(row,candidates,hand,validation,checks)
        cases.append(dict(scene=row['scene'],part=row['part'],**contract,**geometry,
                          failed_checks=checks.failed_count-before))
        print(json.dumps(dict(audited=index,total=len(rows),case=label,failed_checks=checks.failed_count)),flush=True)
    report=dict(status='failed' if checks.failed_count else 'passed_full_96' if full else 'passed_development_subset',
         full_96_case_inventory_verified=full and set(keys)==set(input_rows),audited_cases=len(rows),
         checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,
         totals=dict(proposals=sum(r['generated_count'] for r in rows),valid_aliases=sum(r['valid_count'] for r in rows),
                     display_families=sum(r['grouped_count'] for r in rows),placements_with_options=sum(r['grouped_count']>0 for r in rows)),
         cases=cases,protected_sources_unchanged=protected,critical_generation_sources_unchanged=critical,
         observed_input_sources_unchanged=bool(frozen_input),checker_thresholds_unchanged=validation==baseline_manifest['config'],
         grouping_definition='One geometry-derived height per prepared contact region; both jaw aliases checked. Physically equivalent final grips merge after validation, retaining all source regions, checked approach openings and jaw aliases. Physical equality ignores sampling labels.',
         no_force_or_execution_claim=True,audit_source_sha256=sha256(Path(__file__)),seconds=perf_counter()-start)
    (run/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','audited_cases','failed_checks','totals','seconds')}),flush=True)
    return report


if __name__=='__main__':
    result=verify(sys.argv[1])
    raise SystemExit(1 if result['failed_checks'] else 0)
