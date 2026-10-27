"""Filter frozen passing grasps by both actual CAD contact-normal directions.

The default one-degree trial is a conservative geometric parameter, not a
calibrated assembly tolerance. On curved CAD it tests local facet/tangent
alignment, not a finite flat patch. No CAD classification or polygon overlap is
used; the existing down-facing Panda poses and all source contact data stay put.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import time
import numpy as np

from .verify_comparison import NativeMeshes,audit_geometry
from . import run_compare as compare
from experiments.cad_grasp_geometry_20261026.inputs import REPO,local,read,sha256
from experiments.cad_grasp_geometry_20261026.verify_results import Checks,triangle_distances

SOURCE=REPO/'outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z'
BASE=REPO/'outputs/cad_grasp_method_comparison_20261026/aligned_contacts'
METHOD='one_budget512'
PROTOCOL='panda_table_proj15'
TOLERANCES=(1.,3.,5.,15.)
DEFAULT=1.


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def path_label(path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def normal_angles(normals,closing_axis):
    normals=np.asarray(normals,float)
    axis=np.asarray(closing_axis,float)
    if normals.shape!=(2,3) or axis.shape!=(3,) or not np.isfinite(normals).all() or not np.isfinite(axis).all():
        raise ValueError('Exactly two finite contact normals and a closing axis are required')
    lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths<=1e-15) or np.linalg.norm(axis)<=1e-15:
        raise ValueError('Contact normals and closing axis must be nonzero')
    signed=(normals/lengths[:,None])@(axis/np.linalg.norm(axis))
    return np.degrees(np.arccos(np.clip(np.abs(signed),0.,1.))),signed


def passes_alignment(evaluation,tolerance):
    return bool(np.all(np.asarray(evaluation['angles_deg'])<=tolerance+1e-10))


def retain_groups(source_groups,by_id,evaluations,tolerance):
    """Keep every passing original alias, including a later alias if needed."""
    groups=[]
    for parent in source_groups:
        aliases=[]
        for identifier in parent['grouped_candidate_ids']:
            if passes_alignment(evaluations[identifier],tolerance):
                alias=deepcopy(by_id[identifier])
                alias.update(parent_id=identifier,parent_group_id=parent['id'],
                    alignment=deepcopy(evaluations[identifier]))
                aliases.append(alias)
        if not aliases:
            continue
        chosen=min(aliases,key=lambda c:(0 if c['jaw_sign']>0 else 1,c['opening_m'],c['id']))
        group=deepcopy(chosen)
        group.update(grouped_candidate_ids=[c['id'] for c in aliases],jaw_aliases=aliases,
            source_family_ids=sorted({c['family_id'] for c in aliases}),
            parent_family_ids=parent['source_family_ids'],
            parent_grouped_candidate_ids=parent['grouped_candidate_ids'],
            representative_rule='same canonical jaw/opening/id display representative among all passing parent aliases; no quality ranking')
        groups.append(group)
    return groups


def evaluate(candidate,row,triangles,checks):
    pose=np.asarray(candidate['world_T_grip'],float)
    cad_pose=np.asarray(row['world_T_cad'],float)
    normals=[];contacts=[]
    for witness in candidate['actual_contacts']:
        index=int(witness['first_contact']['triangle_index'])
        triangle=triangles[index]
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
        normal/=np.linalg.norm(normal)
        world=normal@cad_pose[:3,:3].T
        world/=np.linalg.norm(world)
        cad_point=(np.asarray(witness['point_world_m'])-cad_pose[:3,3])@cad_pose[:3,:3]
        distance=float(triangle_distances(cad_point[None],triangle[None])[0])
        checks.check(distance<=5e-9,f"{row['scene']}/{row['part']}/{candidate['id']}: actual contact on recorded native triangle")
        normals.append(world)
        contacts.append(dict(pad=witness['pad'],native_triangle_index=index,
            normal_cad=normal.tolist(),normal_world=world.tolist(),
            actual_point_world_m=witness['point_world_m'],native_triangle_distance_m=distance))
    angles,signed=normal_angles(normals,pose[:3,0])
    for index,witness in enumerate(candidate['actual_contacts']):
        checks.check(abs(abs(signed[index])-witness['normal_alignment'])<=1e-10,
            f"{row['scene']}/{row['part']}/{candidate['id']}: independently computed normal alignment matches core evidence")
    return dict(parent_id=candidate['id'],parent_family_id=candidate['family_id'],
        source_core_valid=True,closing_axis_world=pose[:3,0].tolist(),contacts=contacts,
        signed_normal_dot_closing=signed.tolist(),angles_deg=angles.tolist(),
        maximum_angle_deg=float(angles.max()),
        meaning='Both selected first-contact CAD facet normals relative to the pad closing direction; local tangent interpretation on curved parts.',
        passes_by_tolerance_deg={f'{t:g}':bool(np.all(angles<=t+1e-10)) for t in TOLERANCES},
        default_tolerance_deg=DEFAULT,
        default_decision_reason='both_selected_contact_normals_within_1_deg' if np.all(angles<=DEFAULT+1e-10) else 'one_or_both_selected_contact_normals_exceed_1_deg')


def saved_witness_diagnostics(output):
    started=time.perf_counter()
    source_manifest=read(SOURCE/'manifest.json')
    raw=read(SOURCE/'results.json')
    rows=[r for r in raw if r['method']==METHOD]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    print('SAVED_WITNESS_DIAGNOSTIC_OUTPUT='+str(output),flush=True)
    protected={**source_manifest['source_sha256'],**source_manifest['protected_sha256']}
    for path in (SOURCE/'manifest.json',SOURCE/'results.json',SOURCE/'comparison_audit.json'):
        protected[str(path.relative_to(REPO))]=sha256(path)
    for row in rows:
        for path in (local(row['all_candidate_checks']),local(row['mesh_path'])):
            protected[path_label(path)]=sha256(path)
    sources=[Path(__file__),Path(__file__).with_name('verify_comparison.py'),
             REPO/'experiments/cad_grasp_geometry_20261026/verify_results.py']
    manifest=dict(created_utc=stamp,source_run=str(SOURCE.relative_to(REPO)),method=METHOD,protocol=PROTOCOL,
        definition='saved_selected_witness_only_diagnostic_not_full_one_degree_revalidation',
        default_tolerance_deg=DEFAULT,diagnostic_tolerances_deg=list(TOLERANCES),
        selection_rule='For both actual selected first-contact CAD triangles, arccos(abs(normal_world dot gripper_closing_axis_world)) <= tolerance.',
        source_sha256={str(p.relative_to(REPO)):sha256(p) for p in sources},protected_sha256=protected,
        source_primary_passing_poses=sum(r['valid_count'] for r in rows),cases=96,
        no_pose_generation=True,no_pose_modification=True,no_physics_steps=True,no_robot_execution=True,no_vlm=True,
        no_fitted_surface_classification=True,no_shared_contact_region_check=True,no_quality_ranking=True,no_fixed_count_cap=True,
        scope='Only the frozen one_budget512 primary passing pool after its original 5 mm pair suppression. Not comprehensive unreduced generation.',
        contact_interpretation='Local selected CAD facet/tangent alignment on every shape; no requirement for whole planar CAD faces and no finite flat-contact-patch guarantee.',
        threshold_interpretation='One degree is a conservative diagnostic trial parameter, not a physically calibrated threshold, assembly tolerance or guarantee of zero object movement.',
        source_table_panda_core_checks='Previously audited and unchanged; this read-only filter does not move any source hand transform, pair or contact witness.')
    dump(output/'manifest.json',manifest)
    checks=Checks();meshes=NativeMeshes();retained=[];diagnostics=[]
    checks.check(len(rows)==96 and len({(r['scene'],r['part']) for r in rows})==96,'all 96 one_budget512 rows present')
    source_audit=read(SOURCE/'comparison_audit.json')
    checks.check(source_audit['status']=='passed_full_288' and source_audit['failed_checks']==0,'frozen source geometry audit passed')
    for path,digest in protected.items():
        checks.check(sha256(local(path))==digest,f'frozen protected source mismatch before filter: {path}')
    for row in rows:
        all_candidates=read(local(row['all_candidate_checks']))
        primary=[c for c in all_candidates if c['valid'] and PROTOCOL in c['protocols']]
        by_id={c['id']:c for c in primary}
        checks.check(len(primary)==row['valid_count'],f"{row['scene']}/{row['part']}: source primary count")
        member_ids=[i for g in row['grasps'] for i in g['grouped_candidate_ids']]
        checks.check(len(member_ids)==len(set(member_ids))==len(primary) and set(member_ids)==set(by_id),f"{row['scene']}/{row['part']}: source groups partition every passing alias")
        triangles,_=meshes.get(row)
        evaluations={c['id']:evaluate(c,row,triangles,checks) for c in primary}
        per_tolerance={}
        for tolerance in TOLERANCES:
            groups=retain_groups(row['grasps'],by_id,evaluations,tolerance)
            passing_ids=[c['id'] for c in primary if passes_alignment(evaluations[c['id']],tolerance)]
            kept_ids=[i for g in groups for i in g['grouped_candidate_ids']]
            checks.check(set(kept_ids)==set(passing_ids) and len(kept_ids)==len(passing_ids),f"{row['scene']}/{row['part']}/{tolerance:g}: all passing aliases retained once")
            per_tolerance[f'{tolerance:g}']=dict(passing_poses=len(passing_ids),retained_families=len(groups),available=bool(groups),passing_parent_ids=passing_ids)
            for group in groups:
                for alias in group['jaw_aliases']:
                    original=by_id[alias['parent_id']]
                    checks.check(all(alias[k]==value for k,value in original.items()),f"{row['scene']}/{row['part']}/{alias['id']}: original pose/pair/contact evidence untouched")
                checks.check(all(group[k]==value for k,value in by_id[group['parent_id']].items()),f"{row['scene']}/{row['part']}/{group['id']}: retained representative original data untouched")
        checks.check(per_tolerance['15']['passing_poses']==len(primary),f"{row['scene']}/{row['part']}: 15-degree diagnostic reproduces source passing pool")
        groups=retain_groups(row['grasps'],by_id,evaluations,DEFAULT)
        copy_fields=('scene','part','mesh_path','mesh_sha256','world_T_cad','table_plane_world','table_z_m','input_status','pose_source','candidate_id','associations_assumed','simulator_object_poses_read','pose_geometry_score','pose_depth_mae_m','observed_points')
        result={k:deepcopy(row[k]) for k in copy_fields if k in row}
        result.update(method=METHOD,protocol=PROTOCOL,source_run=str(SOURCE.relative_to(REPO)),
            source_status=row['status'],source_primary_checked_count=row['primary_generated_count'],
            source_primary_passing_count=len(primary),source_retained_family_count=row['grouped_count'],
            alignment_checked_count=len(primary),alignment_passing_pose_count=per_tolerance['1']['passing_poses'],
            grouped_count=len(groups),grasps=groups,alignment_tolerance_deg=DEFAULT,
            status='aligned_contact_grasps_available' if groups else 'no_source_primary_passing_pose' if not primary else 'no_source_pose_with_both_contact_normals_within_1_deg',
            decision_reason='both_selected_contact_normals_within_1_deg' if groups else 'source_pool_empty_before_alignment_filter' if not primary else 'every_source_primary_passing_pose_has_one_or_both_selected_contact_normals_over_1_deg',
            diagnostic_counts_by_tolerance_deg=per_tolerance,
            all_alignment_checks=str((output/'cases'/row['scene']/row['part']/'alignment_checks.json').relative_to(REPO)))
        retained.append(result)
        diagnostics.append(dict(scene=row['scene'],part=row['part'],source_primary_passing_count=len(primary),by_tolerance_deg=per_tolerance))
        dump(local(result['all_alignment_checks']),list(evaluations.values()))
        print(json.dumps({k:result[k] for k in ('scene','part','alignment_checked_count','alignment_passing_pose_count','grouped_count','status')}),flush=True)
    retained.sort(key=lambda r:(r['scene'],r['part']))
    dump(output/'results.json',retained)
    dump(output/'diagnostics.json',diagnostics)
    totals={f'{t:g}':dict(passing_poses=sum(r['by_tolerance_deg'][f'{t:g}']['passing_poses'] for r in diagnostics),retained_families=sum(r['by_tolerance_deg'][f'{t:g}']['retained_families'] for r in diagnostics),available_placements=sum(r['by_tolerance_deg'][f'{t:g}']['available'] for r in diagnostics)) for t in TOLERANCES}
    parts=sorted({r['part'] for r in rows})
    per_part={part:{f'{t:g}':dict(passing_poses=sum(r['by_tolerance_deg'][f'{t:g}']['passing_poses'] for r in diagnostics if r['part']==part),retained_families=sum(r['by_tolerance_deg'][f'{t:g}']['retained_families'] for r in diagnostics if r['part']==part),available_placements=sum(r['by_tolerance_deg'][f'{t:g}']['available'] for r in diagnostics if r['part']==part)) for t in TOLERANCES} for part in parts}
    after={path:sha256(local(path)) for path in protected}
    checks.check(after==protected,'all frozen source/evidence hashes preserved')
    checks.check(all(sha256(local(path))==digest for path,digest in manifest['source_sha256'].items()),'filter/audit source unchanged while executing')
    dump(output/'preservation.json',dict(unchanged=after==protected,sha256_after=after))
    summary=dict(status='passed' if not checks.failed_count else 'failed',cases=96,source_primary_passing_poses=manifest['source_primary_passing_poses'],default_tolerance_deg=DEFAULT,totals_by_tolerance_deg=totals,per_part=per_part,
        default_zero_rows=[dict(scene=r['scene'],part=r['part'],decision_reason=r['decision_reason']) for r in retained if not r['grasps']],checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,
        preserved_frozen_sources=after==protected,seconds=time.perf_counter()-started,
        limitations=[manifest['scope'],manifest['contact_interpretation'],manifest['threshold_interpretation'],'No physics, pickup, force, slip or assembly precision was measured.'])
    dump(output/'summary.json',summary)
    dump(output/'audit.json',dict(status=summary['status'],checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,source_sha256=manifest['source_sha256'],output_sha256={name:sha256(output/name) for name in ('results.json','diagnostics.json','summary.json')},all_original_source_poses_pairs_and_witnesses_untouched=True))
    print(json.dumps({k:summary[k] for k in ('status','cases','source_primary_passing_poses','totals_by_tolerance_deg','checks','failed_checks','seconds')}),flush=True)
    return output,summary


def initialize_alignment_workers():
    compare.core.initialize()
    compare.core.CONFIG=deepcopy(compare.core.CONFIG)
    compare.core.CONFIG['normal_angle_deg']=DEFAULT


PROPOSAL_FIELDS=('id','family_id','adapter','protocols','pair_index','native_pair_projection_deg',
    'jaw_sign','world_T_grip','contacts_world_m','width_m','opening_m','pad_depth_m','center_projection')
INPUT_FIELDS=('scene','part','mesh_path','mesh_sha256','world_T_cad','table_plane_world','table_z_m',
    'input_status','pose_source','candidate_id','associations_assumed','simulator_object_poses_read',
    'pose_geometry_score','pose_depth_mae_m','observed_points')


def group_revalidated(candidates):
    families=[]
    for index in sorted({c['pair_index'] for c in candidates}):
        aliases=[c for c in candidates if c['pair_index']==index and c['valid']]
        representative=compare.placement.canonical_family(aliases)
        if representative:
            families.append(representative)
    return compare.placement.collapse_equivalent_families(families)


def revalidate_case(source_row,output):
    started=time.perf_counter()
    row={k:deepcopy(source_row[k]) for k in INPUT_FIELDS if k in source_row}
    row.update(method=METHOD,protocol=PROTOCOL,
        definition='revalidated_actual_contact_alignment_1deg',
        source_all_candidate_checks=source_row['all_candidate_checks'],
        source_primary_checked_count=source_row['primary_generated_count'],
        source_primary_passing_count=source_row['valid_count'],
        source_retained_family_count=source_row['grouped_count'],
        alignment_tolerance_deg=DEFAULT,center_projection_was_rerun=False)
    folder=Path(output)/'cases'/row['scene']/row['part']
    try:
        components,pads,_=compare.core.hand_geometry()
        vertices,faces,mesh_info=compare.core.load_target(source_row)
        collider=compare.MeshCollider(vertices,faces)
        plane=np.asarray(source_row['table_plane_world'])
        source_candidates=read(local(source_row['all_candidate_checks']))
        primary=[c for c in source_candidates if PROTOCOL in c['protocols']]
        checked=[]
        for original in primary:
            candidate={k:deepcopy(original[k]) for k in PROPOSAL_FIELDS}
            candidate.update(parent_id=original['id'],source_core_valid=original['valid'],
                source_rejection_reason=original['rejection_reason'])
            reason,evidence=compare.core.validate(candidate,collider,components,pads,plane)
            candidate.update(valid=reason is None,rejection_reason=reason,**evidence)
            checked.append(candidate)
        row.update(mesh_info=mesh_info,target_solid_topology=collider.topology,
            revalidated_pose_count=len(checked),valid_count=sum(c['valid'] for c in checked),
            grasps=group_revalidated(checked),
            rejection_counts=dict(Counter(c['rejection_reason'] for c in checked if not c['valid'])))
        row['grouped_count']=len(row['grasps'])
        row['status']='aligned_contact_grasps_available' if row['grasps'] else 'no_feasible_actual_contact_alignment_1deg_in_frozen_pose_pool'
        row['decision_reason']='all_unchanged_core_checks_pass_with_both_selected_contact_normals_within_1_deg' if row['grasps'] else 'no_frozen_primary_hand_pose_passes_the_unchanged_core_checker_with_one_degree_contact_normals'
        row['all_candidate_checks']=str((folder/'candidate_checks.json').relative_to(REPO))
        dump(folder/'candidate_checks.json',checked)
    except Exception as error:
        import traceback
        row.update(status='experiment_error',error=str(error),traceback=traceback.format_exc(),grasps=[],valid_count=0,grouped_count=0)
    row['revalidation_seconds']=time.perf_counter()-started
    dump(folder/'result.json',row)
    return row


def main():
    started=time.perf_counter()
    source_manifest=read(SOURCE/'manifest.json')
    source_rows=[r for r in read(SOURCE/'results.json') if r['method']==METHOD]
    source_lookup={(r['scene'],r['part']):r for r in source_rows}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=BASE/stamp;output.mkdir(parents=True,exist_ok=False)
    print('MAIN_OUTPUT='+str(output),flush=True)
    effective=deepcopy(source_manifest['validation_config']);effective['normal_angle_deg']=DEFAULT
    protected={**source_manifest['source_sha256'],**source_manifest['protected_sha256']}
    for path in (SOURCE/'manifest.json',SOURCE/'results.json',SOURCE/'comparison_audit.json'):
        protected[str(path.relative_to(REPO))]=sha256(path)
    for row in source_rows:
        for path in (local(row['all_candidate_checks']),local(row['mesh_path'])):
            protected[path_label(path)]=sha256(path)
    sources=[Path(__file__),Path(__file__).with_name('verify_comparison.py'),REPO/'experiments/cad_grasp_geometry_20261026/verify_results.py']
    manifest=dict(created_utc=stamp,source_run=str(SOURCE.relative_to(REPO)),method=METHOD,protocol=PROTOCOL,
        definition='revalidated_actual_contact_alignment_1deg',cases=96,
        source_primary_hand_poses=sum(r['primary_generated_count'] for r in source_rows),
        source_primary_passing_poses=sum(r['valid_count'] for r in source_rows),
        effective_validation_config=effective,source_validation_config=source_manifest['validation_config'],
        changed_parameter='Only core.CONFIG normal_angle_deg: 15 to 1 degree, in isolated worker memory; original source file unchanged.',
        default_tolerance_deg=DEFAULT,workers=4,
        source_sha256={str(p.relative_to(REPO)):sha256(p) for p in sources},protected_sha256=protected,
        no_pose_generation=True,no_pose_modification=True,no_centering_rerun=True,no_physics_steps=True,no_robot_execution=True,no_vlm=True,
        no_fitted_surface_classification=True,no_shared_contact_region_check=True,no_quality_ranking=True,no_fixed_count_cap=True,
        scope='All frozen one_budget512 primary hand poses after original 5 mm pair suppression. Not comprehensive unreduced generation.',
        contact_interpretation='Recompute all simultaneous first-contact witnesses under the one-degree normal rule. On curved parts this is selected local CAD facet/tangent alignment, not a finite flat contact patch.',
        threshold_interpretation='One degree is a conservative trial parameter, not a physically calibrated threshold or guarantee of zero part movement.',
        diagnostic_subdirectory='saved_witness_diagnostics',
        timing_note='Per-row time repeats original modeled checks only at the existing hand poses; it excludes prior CAD sampling/pose estimation/centering, VLM and gallery.')
    dump(output/'manifest.json',manifest)
    checks=Checks()
    checks.check(len(source_rows)==96 and manifest['source_primary_hand_poses']==7826,'all 96 rows and 7826 frozen primary hand poses')
    for path,digest in protected.items():
        checks.check(sha256(local(path))==digest,f'frozen file hash before revalidation: {path}')
    diagnostic_output,diagnostic_summary=saved_witness_diagnostics(output/'saved_witness_diagnostics')
    checks.check(diagnostic_summary['failed_checks']==0,'saved-witness diagnostic audit')
    print('STARTING_CORE_REVALIDATION_1DEG',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize_alignment_workers) as pool:
        jobs=[pool.submit(revalidate_case,row,str(output)) for row in source_rows]
        for job in as_completed(jobs):
            row=job.result();rows.append(row);rows.sort(key=lambda r:(r['scene'],r['part']))
            dump(output/'results.json',rows)
            print(json.dumps({k:row.get(k) for k in ('scene','part','revalidated_pose_count','valid_count','grouped_count','revalidation_seconds','status','error')}),flush=True)
    meshes=NativeMeshes();hand=read(REPO/'outputs/cad_grasp_geometry_20261026/hand_model.json')
    audits=[]
    for row in rows:
        before=checks.failed_count;key=(row['scene'],row['part']);source=source_lookup[key]
        checks.check(row['status']!='experiment_error',f'{key}: experiment error {row.get("error")}')
        if row['status']=='experiment_error':
            continue
        source_primary=[c for c in read(local(source['all_candidate_checks'])) if PROTOCOL in c['protocols']]
        old={c['id']:c for c in source_primary}
        candidates=read(local(row['all_candidate_checks']))
        checks.check(len(candidates)==len(source_primary)==row['revalidated_pose_count'],f'{key}: all frozen primary poses rechecked')
        checks.check({c['id'] for c in candidates}==set(old),f'{key}: primary candidate identity preserved')
        for candidate in candidates:
            original=old[candidate['parent_id']]
            checks.check(all(candidate[k]==original[k] for k in PROPOSAL_FIELDS),f"{key}/{candidate['id']}: source pose/pair/opening untouched")
        valid=[c for c in candidates if c['valid']]
        grouped_ids=[i for g in row['grasps'] for i in g['grouped_candidate_ids']]
        checks.check(len(grouped_ids)==len(set(grouped_ids))==len(valid) and set(grouped_ids)=={c['id'] for c in valid},f'{key}: all revalidated aliases retained once')
        checks.check(row['valid_count']==len(valid) and row['grouped_count']==len(row['grasps']),f'{key}: valid/group count arithmetic')
        checks.check(row['rejection_counts']==dict(Counter(c['rejection_reason'] for c in candidates if not c['valid'])),f'{key}: exact rejection reasons')
        triangles,_=meshes.get(row)
        actual_checks=[evaluate(c,row,triangles,checks) for c in valid]
        checks.check(all(passes_alignment(e,DEFAULT) for e in actual_checks),f'{key}: both actual selected normals satisfy one degree')
        source_sparse=np.asarray(read(local(source['all_candidate_checks']).with_name('sparse_world_pairs.json'))).reshape(-1,2,3)
        helper_row=dict(row,center_projection_geometry_queries=source['center_projection_geometry_queries'])
        geometry=audit_geometry(helper_row,candidates,source_sparse,hand,source_manifest['config'],effective,meshes,checks)
        dump(output/'cases'/row['scene']/row['part']/'actual_alignment_checks.json',actual_checks)
        audits.append(dict(scene=row['scene'],part=row['part'],revalidated_pose_count=len(candidates),valid_count=len(valid),grouped_count=len(row['grasps']),audit_failures=checks.failed_count-before,
            maximum_actual_normal_error_deg=max((e['maximum_angle_deg'] for e in actual_checks),default=None),
            maximum_witness_triangle_distance_m=geometry['maximum_witness_triangle_distance_m'],
            minimum_table_clearance_m=geometry['minimum_table_clearance_m']))
    checks.check(len(rows)==96 and sum(r.get('revalidated_pose_count',0) for r in rows)==7826,'complete final revalidation inventory')
    after={path:sha256(local(path)) for path in protected}
    checks.check(after==protected,'frozen source/evidence unchanged after revalidation')
    checks.check(all(sha256(local(path))==digest for path,digest in manifest['source_sha256'].items()),'alignment/audit source unchanged during run')
    dump(output/'preservation.json',dict(unchanged=after==protected,sha256_after=after))
    per_part={part:dict(checked=sum(r['revalidated_pose_count'] for r in rows if r['part']==part),valid=sum(r['valid_count'] for r in rows if r['part']==part),retained_families=sum(r['grouped_count'] for r in rows if r['part']==part),available_placements=sum(bool(r['grasps']) for r in rows if r['part']==part)) for part in sorted({r['part'] for r in rows})}
    summary=dict(status='passed' if not checks.failed_count else 'failed',definition=manifest['definition'],cases=96,
        revalidated_pose_count=sum(r.get('revalidated_pose_count',0) for r in rows),valid_count=sum(r['valid_count'] for r in rows),
        retained_families=sum(r['grouped_count'] for r in rows),available_placements=sum(bool(r['grasps']) for r in rows),
        per_part=per_part,zero_rows=[dict(scene=r['scene'],part=r['part'],decision_reason=r.get('decision_reason'),rejection_counts=r.get('rejection_counts')) for r in rows if not r['grasps']],
        checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,
        saved_witness_only_diagnostics=diagnostic_summary['totals_by_tolerance_deg'],
        preserved_frozen_sources=after==protected,seconds=time.perf_counter()-started,
        limitations=[manifest['scope'],manifest['contact_interpretation'],manifest['threshold_interpretation'],'No physics, pickup, force, slip or assembly precision measured.'])
    dump(output/'summary.json',summary)
    dump(output/'audit.json',dict(status=summary['status'],checks=checks.count,failed_checks=checks.failed_count,failures=checks.failures,cases=audits,
        source_sha256=manifest['source_sha256'],output_sha256={name:sha256(output/name) for name in ('results.json','summary.json')},all_original_source_poses_pairs_and_openings_untouched=True))
    print(json.dumps({k:summary[k] for k in ('status','cases','revalidated_pose_count','valid_count','retained_families','available_placements','checks','failed_checks','seconds')}),flush=True)
    return output,summary


if __name__=='__main__':
    _,outcome=main();raise SystemExit(bool(outcome['failed_checks']))
