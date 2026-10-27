"""Three-case spacing sensitivity using the frozen upstream comparison harness.

Only Fabrica's pair-neighborhood radius changes. Raw CAD samples, downward
projection, Panda adapters and old finite geometry checks stay unchanged.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import time
import numpy as np

from . import run_compare as compare
from . import verify_comparison as audit
from .pair_reduction import original_pair_sparsity
from experiments.cad_grasp_geometry_20261026.inputs import local,read,sha256,load_inputs,OUTPUT,REPO

REFERENCE=REPO/'outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z'
CASES=(('one_budget512','N06','RGOCG8-50_8mm'),('graspkit_512','N00','USB_Male'),
       ('graspkit_512','N01','RGOCG8-50_8mm'))
SPACINGS=(('spacing5000um',.005),('spacing2500um',.0025),('spacing0000um',0.))


def zero_spacing_check():
    pairs=np.array([[[0.,0.,0.],[.02,0.,0.]],[[0.,0.,0.],[.02,0.,0.]],
                    [[1e-14,0.,0.],[.02,0.,0.]]])
    actual=original_pair_sparsity(pairs,0.)
    expected=pairs[[0,2]]
    if not np.array_equal(actual,expected):
        raise ValueError('Original Fabrica zero-radius behavior is not exact-duplicate-only')
    return dict(passed=True,input_pairs=3,retained_pairs=2,
                explanation='Identical pair removed, 1e-14 m differing endpoint retained by original cKDTree function.')


def main():
    started=time.perf_counter()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=compare.BASE/'sparsity_followup'/stamp;output.mkdir(parents=True,exist_ok=False)
    reference_manifest=read(REFERENCE/'manifest.json')
    reference_rows={(r['method'],r['scene'],r['part']):r for r in read(REFERENCE/'results.json')}
    inputs={(r['scene'],r['part']):r for r in load_inputs()}
    cache=local(reference_manifest['intrinsic_cache'])
    intrinsic={(r['method'],r['part']):r for r in read(cache/'records.json')}
    hand=read(OUTPUT/'hand_model.json')
    pads=sorted((p for p in hand['components'] if p['is_pad']),key=lambda p:np.mean(p['vertices_zero_m'],axis=0)[0])
    gap=max(np.asarray(pads[0]['vertices_zero_m'])[:,0])-min(np.asarray(pads[1]['vertices_zero_m'])[:,0])
    sources=[Path(__file__),Path(audit.__file__)]
    protected={**reference_manifest['source_sha256'],**reference_manifest['protected_sha256']}
    for path in [REFERENCE/'manifest.json',REFERENCE/'results.json',REFERENCE/'comparison_audit.json']:
        protected[str(path.relative_to(REPO))]=sha256(path)
    for key in CASES:
        for name in ('candidate_checks.json','sparse_world_pairs.json','result.json'):
            path=local(reference_rows[key]['all_candidate_checks']).with_name(name)
            protected[str(path.relative_to(REPO))]=sha256(path)
    original_config=deepcopy(compare.CONFIG)
    checks=audit.Checks();meshes=audit.NativeMeshes()
    original_sha={str(p.relative_to(REPO)):sha256(p) for p in sources}
    zero=zero_spacing_check()
    compare.core.initialize()
    records=[]
    print('OUTPUT='+str(output),flush=True)
    try:
        for tag,spacing in SPACINGS:
            folder=output/tag;folder.mkdir()
            config=deepcopy(original_config);config['pair_sparsity_m']=spacing
            compare.CONFIG=config
            manifest=dict(created_utc=stamp,reference_run=str(REFERENCE.relative_to(REPO)),
                intrinsic_cache=str(cache.relative_to(REPO)),cases=[dict(method=m,scene=s,part=p) for m,s,p in CASES],
                case_count=3,config=config,validation_config=reference_manifest['validation_config'],
                protocols=list(audit.PROTOCOLS),source_sha256=original_sha,protected_sha256=protected,
                changed_parameter='Fabrica original filter_sparse_pairs min_distance in meters',
                zero_radius_semantics=zero if spacing==0 else None,
                no_new_sampler=True,no_vlm=True,no_robot_execution=True,no_physics_steps=True,
                no_shared_region_check=True,no_fitted_cylinder_or_flat_one_degree_check=True,
                timing_note='Sequential three-case parameter exploration. Both common adapters and both jaw assignments run. Raw intrinsic CAD generation, input pose estimation, VLM and gallery are excluded. Warm-cache/order effects are possible; this is not a calibrated deployment-time benchmark.')
            compare.core.dump(folder/'manifest.json',manifest)
            rows=[];audits=[]
            for key in CASES:
                method,scene,part=key
                row=compare.run_case(inputs[scene,part],intrinsic[method,part],str(folder))
                rows.append(row)
                compare.core.dump(folder/'results.json',rows)
                checks.check(row['status']!='experiment_error',f'{tag}/{key}: experiment error {row.get("error")}')
                if row['status']=='experiment_error':
                    continue
                before=checks.failed_count
                raw=np.load(local(intrinsic[method,part]['npz_path']))['pairs_m']
                sparse,width_count,projected_count=audit.replay_pairs(raw,np.asarray(row['world_T_cad']),hand['opening_limits_m'][1]-gap-config['extra_opening_m'],config)
                saved=np.asarray(read(local(row['all_candidate_checks']).with_name('sparse_world_pairs.json'))).reshape(-1,2,3)
                checks.check(saved.shape==sparse.shape and np.allclose(saved,sparse,atol=1e-14,rtol=0),f'{tag}/{key}: independent source sparsity replay')
                checks.check(row['width_qualified_pairs']==width_count and row['horizontal_projection_qualified_pairs']==projected_count and row['sparse_pairs']==len(sparse),f'{tag}/{key}: reduction counts')
                candidates=read(local(row['all_candidate_checks']))
                passing=audit.audit_counts(row,candidates,sparse,config,checks)
                geometry=audit.audit_geometry(row,candidates,sparse,hand,config,reference_manifest['validation_config'],meshes,checks)
                original=reference_rows[key]
                old_candidates=read(local(original['all_candidate_checks']))
                if spacing==.005:
                    checks.check(candidates==old_candidates,f'{tag}/{key}: 5mm full saved-candidate evidence changed from frozen comparison')
                    checks.check(row['grasps']==original['grasps'] and row['protocol_summary']==original['protocol_summary'],f'{tag}/{key}: 5mm baseline groups or protocol results changed')
                if spacing==0:
                    unique=len(np.unique(saved.reshape(-1,6),axis=0))
                    checks.check(unique==len(saved)<=projected_count,f'{tag}/{key}: zero spacing duplicate-only count')
                result=dict(spacing_m=spacing,method=method,scene=scene,part=part,
                    raw_pairs=row['native_pairs'],width_qualified_pairs=width_count,
                    projection_qualified_pairs=projected_count,sparse_pairs=len(sparse),
                    all_adapter_checked=row['generated_count'],primary_checked=row['primary_generated_count'],
                    primary_valid=row['valid_count'],retained_families=row['grouped_count'],
                    primary_available=bool(passing),protocol_summary=row['protocol_summary'],
                    primary_rejection_counts=row['rejection_counts'],
                    runtime_seconds=row['timing_seconds'],sampling_and_reduction_seconds=row['runtime_sampling_seconds'],
                    adapter_validation_seconds=row['adapter_validation_seconds'],
                    frozen_5mm_primary_valid=original['valid_count'],
                    frozen_5mm_retained_families=original['grouped_count'],
                    audit_failures=checks.failed_count-before,
                    maximum_witness_triangle_distance_m=geometry['maximum_witness_triangle_distance_m'],
                    minimum_table_clearance_m=geometry['minimum_table_clearance_m'],
                    all_candidate_checks=row['all_candidate_checks'])
                audits.append(result);records.append(result)
                print(json.dumps(result),flush=True)
            after={path:sha256(local(path)) for path in protected}
            checks.check(after==protected,f'{tag}: frozen source/evidence preservation')
            compare.core.dump(folder/'preservation.json',dict(unchanged=after==protected,sha256_after=after))
            compare.core.dump(folder/'audit.json',dict(cases=audits,failed_checks=sum(r['audit_failures'] for r in audits),
                source_sha256=original_sha,scope='Three selected method/placement cases only; no full-run coverage inference.'))
    finally:
        compare.CONFIG=original_config
    after={path:sha256(local(path)) for path in protected}
    checks.check(after==protected,'final frozen evidence preservation')
    checks.check(all(sha256(local(path))==digest for path,digest in original_sha.items()),'follow-up/audit source unchanged during execution')
    report=dict(status='passed' if not checks.failed_count else 'failed',
        case_count=3,spacing_conditions=3,records=records,checks=checks.count,
        failed_checks=checks.failed_count,failures=checks.failures,
        reference_run=str(REFERENCE.relative_to(REPO)),protected_sources_unchanged=after==protected,
        zero_radius_semantics=zero,source_sha256=original_sha,seconds=time.perf_counter()-started,
        limitations=['Three deliberately selected prior failures; no new full-96 result is claimed.',
                     'Original upstream contact caches and Fabrica pair suppression plus common adapters are executed; complete upstream systems are not.',
                     'Geometric feasibility only: no pickup dynamics, force, friction or assembly precision measured.',
                     'Each runtime includes both adapters. Intrinsic generation is reused and warm-cache/order effects are possible.'])
    compare.core.dump(output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('status','checks','failed_checks','seconds')}),flush=True)
    return report


if __name__=='__main__':
    outcome=main();raise SystemExit(bool(outcome['failed_checks']))
