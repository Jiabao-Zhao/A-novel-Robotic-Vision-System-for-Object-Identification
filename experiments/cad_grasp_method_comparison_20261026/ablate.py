"""Fixed-proposal contact-screen ablation; no generation, physics or VLM.

The strict paired-point replay is copied structurally from the frozen experiment
with only its proximity-band width made explicit. The matched witness variants
use identical first-touch evaluation points so guard removal has a defined
continuation even when no shared point exists. These are separate comparisons.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
import traceback
import numpy as np

from experiments.cad_grasp_geometry_20261026 import run_trial as core
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_surface_grasps_20261026 import contacts

ROOT=core.ROOT
SOURCE=ROOT/'outputs/cad_surface_grasps_20261026/runs/20261026T200202Z'
OUTPUT=ROOT/'outputs/cad_grasp_method_comparison_20261026/ablation'
BANDS_M=(.000025,.00005,.0001,.00025)
POLICIES=('saved_strict','strict_25','band_50','band_100','band_250',
          'normal_screen_only','witness_shared_25','witness_no_shared','core_only')
WORKERS=2


def decision(reason=None,**evidence):
    return dict(valid=reason is None,reason=reason,**evidence)


def native_normals(collider,indices):
    t=collider.faces[np.asarray(indices,dtype=int)]
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    return n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)


def radial_check(candidate,collider,fit,points,indices):
    pose=np.asarray(candidate['world_T_grip'])
    shaft=np.asarray(fit['shaft_axis_world'])
    origin=np.asarray(fit['point_on_axis_world'])
    perpendicular=float(np.degrees(np.arcsin(np.clip(abs(shaft@pose[:3,0]),0,1))))
    evidence=dict(cylinder_id=fit['id'],shaft_perpendicular_error_deg=perpendicular)
    if perpendicular>1.+1e-8:
        return decision('closing_not_transverse_to_fitted_shaft',**evidence)
    p=np.asarray(points)-origin
    radial=p-(p@shaft)[:,None]*shaft
    radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-30)
    errors=np.degrees(np.arccos(np.clip(np.abs(radial@pose[:3,0]),0,1)))
    sectors=[]
    for index in indices:
        vertices=collider.faces[index]-origin
        v=vertices-(vertices@shaft)[:,None]*shaft
        v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-30)
        sectors.append(float(np.degrees(np.arccos(np.clip((v@v.T).min(),-1,1)))))
    allowance=1.+np.asarray(sectors)
    evidence.update(radial_errors_deg=errors.tolist(),sector_widths_deg=sectors,
                    radial_allowances_deg=allowance.tolist())
    if np.any(errors>allowance+1e-8):
        return decision('round_contact_outside_native_facet_angular_bound',**evidence)
    return decision(**evidence)


def normal_check(candidate,tolerance):
    errors=[float(np.degrees(np.arccos(np.clip(c['normal_alignment'],0,1))))
            for c in candidate['actual_contacts']]
    return decision('actual_pad_normal_not_aligned_with_surface' if max(errors)>tolerance+1e-8 else None,
                    actual_normal_errors_deg=errors,normal_tolerance_deg=tolerance)


def shared_probe(candidate,collider,pads,tolerance,fit,band):
    regions,triangles=contacts.pad_regions(candidate,collider,pads,tolerance,band_m=band,
                              triangle_ids=fit['supporting_triangle_ids'] if fit else None)
    intersections=[]
    desired=np.array([0.,candidate['pad_depth_m']])
    for a in regions[0]['polygons']:
        p=np.asarray(a['polygon_local_m'])[:,1:]
        for b in regions[1]['polygons']:
            if float(np.asarray(a['normal_local'])@b['normal_local'])>-np.cos(np.deg2rad(2*tolerance)):
                continue
            q=np.asarray(b['polygon_local_m'])[:,1:]
            if np.any(p.max(axis=0)<q.min(axis=0)) or np.any(q.max(axis=0)<p.min(axis=0)):
                continue
            overlap=contacts.intersect_polygons(p,q)
            if len(overlap)>=3 and abs(contacts.signed_area(overlap))>1e-14:
                yz=contacts.closest_point(overlap,desired)
                intersections.append((float(np.linalg.norm(yz-desired)),yz,a,b))
    evidence=dict(band_m=band,individual_pad_region_polygons=[len(r['polygons']) for r in regions],
                  shared_region_exists=bool(intersections))
    if not intersections:
        return None,evidence
    _,yz,a,b=min(intersections,key=lambda row:row[0])
    local=[]
    for record in (a,b):
        triangle=triangles[record['triangle_index']]
        n=np.asarray(record['normal_local'])
        x=(n@triangle[0]-n[1:]@yz)/n[0]
        local.append([x,*yz])
    pose=np.asarray(candidate['world_T_grip'])
    points=np.asarray(local)@pose[:3,:3].T+pose[:3,3]
    return (points,[a['triangle_index'],b['triangle_index']]),evidence


def paired_variant(candidate,collider,pads,tolerance,fit,band,cache):
    normal=normal_check(candidate,tolerance)
    if not normal['valid']:
        return normal
    key=(candidate['id'],fit['id'] if fit else None,band)
    if key not in cache:
        cache[key]=shared_probe(candidate,collider,pads,tolerance,fit,band)
    pair,evidence=cache[key]
    if pair is None:
        return decision('no_common_opposed_surface_region_inside_pads',**evidence)
    if fit:
        result=radial_check(candidate,collider,fit,*pair)
        result.update(evidence)
        return result
    return decision(**evidence)


def witness_variant(candidate,collider,pads,tolerance,fit,require_shared,cache):
    normal=normal_check(candidate,tolerance)
    if not normal['valid']:
        return normal
    key=(candidate['id'],fit['id'] if fit else None,.000025)
    if key not in cache:
        cache[key]=shared_probe(candidate,collider,pads,tolerance,fit,.000025)
    _,evidence=cache[key]
    if not all(evidence['individual_pad_region_polygons']):
        return decision('one_pad_has_no_guarded_facing_region',**evidence)
    indices=[c['first_contact']['triangle_index'] for c in candidate['actual_contacts']]
    normals=native_normals(collider,indices)
    if normals[0]@normals[1]>-np.cos(np.deg2rad(2*tolerance))+1e-12:
        return decision('actual_witness_normals_not_opposed',**evidence)
    # Crucially, continue this check even when the shared-region test failed.
    # The same native first-touch witnesses are used in both matched variants.
    if fit:
        result=radial_check(candidate,collider,fit,
                            [c['point_world_m'] for c in candidate['actual_contacts']],indices)
        if not result['valid']:
            result.update(evidence)
            return result
    else:
        result=decision()
    if require_shared and not evidence['shared_region_exists']:
        return decision('no_common_opposed_surface_region_inside_pads',**evidence)
    result.update(evidence)
    return result


def candidate_variants(candidate,collider,pads,analysis):
    result=dict(saved_strict=decision(candidate['rejection_reason'] if not candidate['valid'] else None))
    if not candidate['interference_check_passed']:
        for name in POLICIES[1:]:
            result[name]=decision(candidate['rejection_reason'])
        return result
    result['core_only']=decision()
    ids=[p['first_contact']['triangle_index'] for p in candidate['actual_contacts']]
    fits=[h for h in analysis['cylinder_hypotheses'] if all(i in h['supporting_triangle_ids'] for i in ids)]
    fits=sorted(fits,key=lambda h:h['circle_vertex_residual95_m']) or [None]
    cache={}
    for name in POLICIES[1:-1]:
        checked=[]
        for fit in fits:
            tolerance=15. if fit else 1.
            if name=='normal_screen_only':
                current=normal_check(candidate,tolerance)
            elif name in ('witness_shared_25','witness_no_shared'):
                current=witness_variant(candidate,collider,pads,tolerance,fit,name=='witness_shared_25',cache)
            else:
                band=BANDS_M[('strict_25','band_50','band_100','band_250').index(name)]
                current=paired_variant(candidate,collider,pads,tolerance,fit,band,cache)
            current['classified_contact']='fitted_round_side' if fit else 'flat_screen_no_matching_cylinder'
            checked.append(dict(model_id=fit['id'] if fit else None,reason=current['reason']))
            if current['valid']:
                break
        result[name]=dict(current,models_checked=checked)
    return result


def run_case(case_path,output):
    start=time.perf_counter()
    case_path=Path(case_path)
    row=json.loads((case_path/'result.json').read_text())
    candidates=json.loads((case_path/'candidate_checks.json').read_text())
    record=dict(scene=row['scene'],part=row['part'],source_case=str(case_path.relative_to(ROOT)),
                proposed=len(candidates),preparation_rejections=row['preparation_rejection_counts'])
    try:
        components,pads,_=core.hand_geometry()
        vertices,faces,_=core.load_target(row)
        collider=MeshCollider(vertices,faces)
        plane=np.asarray(row['table_plane_world'])
        decisions=[]
        mismatches=[]
        for candidate in candidates:
            before=time.perf_counter()
            reason,evidence=core.validate(candidate,collider,components,pads,plane)
            if (reason is None)!=candidate['interference_check_passed']:
                raise AssertionError('Frozen core decision changed: '+candidate['id'])
            if reason and reason!=candidate['rejection_reason']:
                raise AssertionError('Frozen core rejection changed: '+candidate['id'])
            evaluation=candidate_variants(candidate,collider,pads,row['direction_analysis'])
            if evaluation['strict_25']['valid']!=candidate['valid'] or evaluation['strict_25']['reason']!=candidate['rejection_reason']:
                mismatches.append(dict(id=candidate['id'],saved=candidate['rejection_reason'],replay=evaluation['strict_25']))
            decisions.append(dict(id=candidate['id'],family_id=candidate['family_id'],jaw_sign=candidate['jaw_sign'],
                                  variants=evaluation,core_replay_seconds=time.perf_counter()-before))
        if mismatches:
            raise AssertionError('Strict25 replay differs: '+str(mismatches[:3]))
        record['variants']={name:dict(valid=sum(d['variants'][name]['valid'] for d in decisions),
            ids=[d['id'] for d in decisions if d['variants'][name]['valid']],
            rejections=dict(Counter(d['variants'][name]['reason'] for d in decisions if not d['variants'][name]['valid'])))
            for name in POLICIES}
        record.update(status='completed',strict_replay_matches=True,core_replay_matches=True)
        folder=Path(output)/'cases'/row['scene']/row['part']
        core.dump(folder/'evaluations.json',decisions)
        record['evaluations']=str((folder/'evaluations.json').relative_to(ROOT))
    except Exception as error:
        record.update(status='error',error=str(error),traceback=traceback.format_exc())
    record['replay_seconds']=time.perf_counter()-start
    core.dump(Path(output)/'cases'/row['scene']/row['part']/'result.json',record)
    return record


def summarize(records,output):
    totals={name:dict(valid=sum(r['variants'][name]['valid'] for r in records),
        placements=sum(r['variants'][name]['valid']>0 for r in records),
        rejections=dict(sum((Counter(r['variants'][name]['rejections']) for r in records),Counter()))) for name in POLICIES}
    unavailable=[r for r in records if not r['variants']['saved_strict']['valid']]
    summary=dict(cases=len(records),proposals=sum(r['proposed'] for r in records),totals=totals,
        original_unavailable=[dict(scene=r['scene'],part=r['part'],proposals=r['proposed'],
             counts={name:r['variants'][name]['valid'] for name in POLICIES},
             preparation_rejections=r['preparation_rejections'],strict_rejections=r['variants']['strict_25']['rejections']) for r in unavailable],
        witness_guard_recovered=sum(r['variants']['witness_no_shared']['valid']>0 and r['variants']['witness_shared_25']['valid']==0 for r in records),
        no_generation=True,no_physics=True,no_vlm=True,
        runtime_note='Replay cost with two CPU workers, not fresh generation time. Native geometry/table/hand and all518proposalposes remain fixed. Largerbands are sensitivity experiments, not calibrated deformation or acceptableassemblyoffset.')
    core.dump(output/'summary.json',summary)
    lines=['# Fixed-proposal shared-contact-region ablation','',
        'All518 frozen hand proposals from200202Z are held fixed. The real modeled Panda hand, target triangles, observed poses/table and core thresholds are unchanged. No new grasp is generated and no pickup is attempted.','',
        '| Condition | Passing checked poses | Placements with at least one |','|---|---:|---:|']
    lines += [f"| {name} | {totals[name]['valid']} | {totals[name]['placements']}/96 |" for name in POLICIES]
    lines += ['', 'Strict25 exactly reproduces every frozen decision. Core-only independently replays all518 original full-hand checks.','',
        'The paired-point25/50/100/250µm tests change only the geometric proximity band used to find common regions; all later paired-point normal/shaft/radial checks still run. Largerbands do not modify robot solids, object triangles, actual first contacts or the core interference check.','',
        'The matched witness_shared25 and witness_no_shared tests use identical actual first-touch witnesses, individual guarded pad-region existence, normal opposition, cylinder membership and radial/mesh checks. Only requiring sharedY/Z differs between those two columns. This is a clean guard-effect comparison under that witness formulation. Comparing either witness condition directly with the original strict paired-point test also changes where the later radial checks are evaluated; it is not described as pure guard removal.','',
        'normal_screen_only retains the added1°flat/15°classified-cylinder native-normal test without the common-region or fitted-radial screens. It isolates how much rejection already occurs before shared-region computation.','',
        '| Original missing placement | Proposals | Strict25 | Normal only | Witness+shared | Witness without shared | Core only |','|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['scene']}/{r['part']} | {r['proposed']} | {r['variants']['strict_25']['valid']} | {r['variants']['normal_screen_only']['valid']} | {r['variants']['witness_shared_25']['valid']} | {r['variants']['witness_no_shared']['valid']} | {r['variants']['core_only']['valid']} |" for r in unavailable]
    lines += ['', 'The two levels of failure matter: preparation had excluded some contact regions before a hand pose existed; this fixed-proposal test cannot recover those regions. Every listed proposal did exist in the frozen run. Removed screens increase modeled candidate availability, not demonstrated pickup success or assembly precision.','',summary['runtime_note'],'']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=OUTPUT/stamp
    output.mkdir(parents=True,exist_ok=False)
    records=json.loads((SOURCE/'results.json').read_text())
    protected=[SOURCE/'manifest.json',SOURCE/'results.json',SOURCE/'audit.json',ROOT/'docs/cad_view_prompt.txt',
               Path(core.__file__),Path(core.__file__).with_name('geometry_kernel.py'),Path(contacts.__file__)]
    before={str(p.relative_to(ROOT)):core.sha(p) for p in protected}
    core.dump(output/'manifest.json',dict(created_utc=stamp,source_run=str(SOURCE.relative_to(ROOT)),
        policies=POLICIES,bands_m=BANDS_M,workers=WORKERS,source_sha256=core.sha(Path(__file__)),protected_sha256=before,
        no_generation=True,no_physics_steps=True,no_vlm=True,poses_fixed=True))
    print('OUTPUT='+str(output),flush=True)
    result=[]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=core.initialize) as pool:
        jobs=[pool.submit(run_case,str(SOURCE/'cases'/r['scene']/r['part']),str(output)) for r in records]
        for job in as_completed(jobs):
            case=job.result()
            result.append(case)
            result.sort(key=lambda r:(r['scene'],r['part']))
            core.dump(output/'results.json',result)
            print(json.dumps({k:case.get(k) for k in ('scene','part','status','error','replay_seconds')}),flush=True)
    errors=[r for r in result if r['status']!='completed']
    if errors:
        raise RuntimeError('Ablation errors: '+str([(r['scene'],r['part'],r['error']) for r in errors]))
    summary=summarize(result,output)
    after={str(p.relative_to(ROOT)):core.sha(p) for p in protected}
    core.dump(output/'preservation.json',dict(unchanged=before==after,sha256_after=after))
    if before!=after:
        raise RuntimeError('Protected evidence changed')
    print(json.dumps(summary['totals'],indent=2),flush=True)
    print('COMPLETE='+str(output),flush=True)


def run_extra_case(case_path,output):
    start=time.perf_counter()
    case_path=Path(case_path)
    row=json.loads((case_path/'result.json').read_text())
    candidates=json.loads((case_path/'candidate_checks.json').read_text())
    _,pads,_=core.hand_geometry()
    vertices,faces,_=core.load_target(row)
    collider=MeshCollider(vertices,faces)
    evaluations=[]
    for candidate in candidates:
        if not candidate['interference_check_passed']:
            current=decision(candidate['rejection_reason'])
        else:
            ids=[p['first_contact']['triangle_index'] for p in candidate['actual_contacts']]
            fits=sorted([h for h in row['direction_analysis']['cylinder_hypotheses']
                        if all(i in h['supporting_triangle_ids'] for i in ids)],
                        key=lambda h:h['circle_vertex_residual95_m']) or [None]
            cache={}
            for fit in fits:
                current=paired_variant(candidate,collider,pads,15. if fit else 1.,fit,.003,cache)
                if current['valid']:
                    break
        evaluations.append(dict(id=candidate['id'],family_id=candidate['family_id'],
            direction_source=candidate['direction_source'],decision=current))
    folder=Path(output)/'cases'/row['scene']/row['part']
    core.dump(folder/'evaluations.json',evaluations)
    record=dict(scene=row['scene'],part=row['part'],proposals=len(candidates),
        valid=sum(e['decision']['valid'] for e in evaluations),
        valid_ids=[e['id'] for e in evaluations if e['decision']['valid']],
        rejections=dict(Counter(e['decision']['reason'] for e in evaluations if not e['decision']['valid'])),
        seconds=time.perf_counter()-start)
    core.dump(folder/'result.json',record)
    return record


def extra_band_main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=OUTPUT/(stamp+'_band3000')
    output.mkdir(parents=True,exist_ok=False)
    rows=[r for r in json.loads((SOURCE/'results.json').read_text()) if not r['grasps']]
    protected=[SOURCE/'manifest.json',SOURCE/'results.json',SOURCE/'audit.json',
               ROOT/'docs/cad_view_prompt.txt',Path(core.__file__),Path(contacts.__file__)]
    before={str(p.relative_to(ROOT)):core.sha(p) for p in protected}
    core.dump(output/'manifest.json',dict(source_run=str(SOURCE.relative_to(ROOT)),
        subset='nine originally unavailable placements only',band_m=.003,created_utc=stamp,
        no_generation=True,no_physics=True,no_vlm=True,core_evidence_reused=True,
        source_sha256=core.sha(Path(__file__)),protected_sha256=before,
        interpretation='Exploratory3mm version of our polygon proximity screen; not execution or reproduction of GPG pointcloud extremal threshold'))
    (output/'ablate_source.py').write_bytes(Path(__file__).read_bytes())
    print('OUTPUT='+str(output),flush=True)
    records=[]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=core.initialize) as pool:
        for job in as_completed([pool.submit(run_extra_case,str(SOURCE/'cases'/r['scene']/r['part']),str(output)) for r in rows]):
            record=job.result()
            records.append(record)
            records.sort(key=lambda r:(r['scene'],r['part']))
            core.dump(output/'results.json',records)
            print(json.dumps(record),flush=True)
    summary=dict(cases=len(records),proposals=sum(r['proposals'] for r in records),
        passing=sum(r['valid'] for r in records),placements=sum(r['valid']>0 for r in records),
        band_m=.003,results=records,
        caveat='Only the nine previously missing cases were retested. Do not extrapolate a full96case3mm score; polygon screen is not GPG pointcloud criterion or calibrated pad compression.')
    core.dump(output/'summary.json',summary)
    after={str(p.relative_to(ROOT)):core.sha(p) for p in protected}
    core.dump(output/'preservation.json',dict(unchanged=before==after,sha256_after=after))
    if before!=after:
        raise RuntimeError('Protected evidence changed')
    lines=['# 3 mm polygon-band sensitivity on the nine missing placements','',
        'Same saved hand poses, actual first-contact and native target/table geometry. This is our polygon rule at a different parameter, not an execution of the GPG algorithm. A3mm distance band does not imply permissible physical compression or assembly error.','',
        '| Placement | Existing proposals | Passing3mm paired-point checks |','|---|---:|---:|']
    lines += [f"| {r['scene']}/{r['part']} | {r['proposals']} | {r['valid']} |" for r in records]
    lines += ['',f"Recovered options in {summary['placements']}/9 original missing placements. {summary['passing']}/{summary['proposals']} proposals pass this exploratory condition.",
              '',summary['caveat'],'']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE='+str(output),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='band3000':
        extra_band_main()
    else:
        main()
