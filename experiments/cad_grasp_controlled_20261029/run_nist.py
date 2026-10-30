"""Three paired seeds, two bounded samplers and three contact policies."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from experiments.cad_grasp_selection_20261029.task_inputs import build_request,ROOT
from experiments.cad_grasp_refinement_20261029.run_nist import validate_observed_rows
from experiments.cad_grasp_selection_20261028.run_nist import sha,protected_paths
from .pipeline import Evaluator,digest,_json
from .sampler import generate_candidates
from .geometry import stage_classification
from .calibration import make_contact_calibration

PUBLISHED=ROOT/'results/grasp/controlled_20261029/shared_inputs'
def input_path(original,published):
    path=ROOT/original
    return path if path.is_file() else PUBLISHED/published
INPUTS=input_path('outputs/cad_grasp_geometry_20261026/inputs.json','observed_alignment_inputs.json')
HAND=input_path('outputs/cad_grasp_geometry_20261026/hand_model.json','hand_model.json')
REGISTRY=input_path('outputs/nist_scene_corrections_20261029/assembly_goals.json','assembly_goals.json')
BASE=ROOT/'outputs/cad_grasp_controlled_20261029/runs'
SEEDS=(20261028,20262028,20263028)
STRATEGIES=('random','task_informed')
ANGLES=(.5,1.,2.,3.,5.)
SETTINGS=dict(max_poses=100,surface_budget=200,orientation_samples=10)


def dump(path,value,compressed=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,default=_json)
    if compressed:
        with path.open('wb') as handle:
            with gzip.GzipFile(fileobj=handle,mode='wb',mtime=0) as stream:stream.write(text.encode())
    else:path.write_text(json.dumps(value,indent=2,allow_nan=False,default=_json)+'\n',encoding='utf-8')


def _counts(records):
    categories={}
    for row in records:
        for key,entry in row['validity']['diagnostics'].items():categories.setdefault(key,Counter())[entry['status']]+=1
    return {name:dict(values) for name,values in categories.items()}


def run_part(part,rows,part_index,output,seeds,strategies,sensitivity):
    output=Path(output);started=time.perf_counter()
    hand=json.loads(HAND.read_text());registry=json.loads(REGISTRY.read_text())
    built=build_request(part,rows[0]['world_T_cad'],rows[0]['table_plane_world'],registry=registry,hand=hand)
    if built['status']!='available':raise ValueError(part+': '+str(built.get('message')))
    request=dict(built['request'],part=part,simulator_object_poses_read=False)
    evaluator=Evaluator(request)
    dump(output/'task_inputs'/f'{part}.json',dict(source=built['source'],binding_sha256=evaluator.binding,
        contact_model=evaluator.model.provenance,calibration_sha256=evaluator.calibration['calibration_sha256']))
    results=[];angle_results=[]
    for strategy in strategies:
        for seed_index,base_seed in enumerate(seeds):
            seed=int(base_seed)+part_index;branch_started=time.perf_counter()
            for row in sorted(rows,key=lambda item:item['scene']):
                case_started=time.perf_counter()
                if row.get('input_status')!='available':raise ValueError('Observed CAD alignment unavailable')
                raw,sampling=generate_candidates(evaluator.vertices,evaluator.faces,hand,
                    world_T_cad_pickup=row['world_T_cad'],world_T_cad_final=evaluator.final,
                    table_plane_world=row['table_plane_world'],axis_point_cad_m=request['axis_point_cad_m'],
                    axis_direction_cad=request['axis_direction_cad'],sampling_contact_region=evaluator.region,
                    contact_model=evaluator.model,contact_calibration=evaluator.calibration,
                    seed=seed,strategy=strategy,contact_modes=('pad','finger'),final_up_world=[0,0,1],
                    proposal_normal_angle_rad=evaluator.proposal_angle,**SETTINGS)
                bank=Path('banks')/strategy/f'seed_{seed_index}'/row['scene']/part
                dump(output/bank/'sampling.json.gz',sampling,True)
                decisions,pools=evaluator.compare(raw,row['world_T_cad'],row['table_plane_world'],seed=seed,strategy=strategy,
                    sampling_reference=str(bank/'sampling.json.gz'))
                dump(output/bank/'candidate_diagnostics.json.gz',dict(pad=pools['pad'],finger=pools['finger'],
                    combined_assignment=[dict(candidate_id=c['candidate_id'],contact_mode=c['combined_contact_mode']) for c in raw],
                    raw_bank_sha256=sampling['shared_raw_bank_sha256'],
                    benchmark_pool_evaluations=len(pools['pad'])+len(pools['finger']),
                    per_policy_budget=100,combined_is_preregistered_subset_not_union=True),True)
                case_seconds=time.perf_counter()-case_started
                for mode,selection in decisions.items():
                    records=pools[mode]
                    reasons=Counter(reason for candidate in records for reason in candidate['validity'].get('reasons',[]))
                    record=dict(scene=row['scene'],part=part,sampler_strategy=strategy,contact_mode=mode,
                        seed_index=seed_index,base_seed=base_seed,seed=seed,mesh_path=row['mesh_path'],
                        world_T_cad=row['world_T_cad'],table_plane_world=row['table_plane_world'],
                        pose_source=row.get('pose_source'),hand_path=str(HAND),selection=selection,
                        rejection_counts=dict(reasons),independent_check_counts=_counts(records),
                        diagnostic_bank=str(bank/'candidate_diagnostics.json.gz'),
                        diagnostic_mode=mode,shared_benchmark_case_wall_seconds=case_seconds,
                        independent_policy_wall_seconds=None,
                        no_VLM=True,no_pickup_or_assembly_execution=True,simulator_object_poses_read=False,
                        no_insertion_or_release_checks=True,
                        proposal_bank_shared_across_contact_arms=True,physical_grasp_success_measured=False)
                    dump(output/'cases'/strategy/f'seed_{seed_index}'/mode/row['scene']/part/'selection.json',record)
                    results.append(record)
                if sensitivity and strategy=='random' and seed_index==0:
                    baseline={c['candidate_id']:c for c in pools['finger']}
                    frozen=[evaluator.fit(c,'finger') for c in raw]
                    frozen_sha=digest(frozen)
                    for angle in ANGLES:
                        records=(pools['finger'] if angle==1. else [evaluator.evaluate(c,row['world_T_cad'],row['table_plane_world'],angle=np.deg2rad(angle)) for c in frozen])
                        if digest(frozen)!=frozen_sha:raise ValueError('Sensitivity changed a fitted pose')
                        rescued=[];already_clear=[];newly_certified=[]
                        nonangular=('candidate_input','opening_limits','contact_reconstruction',
                            'target_closing_collision','table_clearance','final_receiver_collision','protected_surfaces')
                        for candidate in records:
                            old=baseline[candidate['candidate_id']]
                            if candidate['validity']['status']=='valid' and old['validity']['status']!='valid':
                                rescued.append(candidate['candidate_id'])
                                if all(old['validity']['diagnostics'][name]['status']=='pass' for name in nonangular):already_clear.append(candidate['candidate_id'])
                                else:newly_certified.append(candidate['candidate_id'])
                        selection=evaluator.decide(records,mode='finger',seed=seed,strategy='frozen_random_pose_sensitivity')
                        angle_results.append(dict(scene=row['scene'],part=part,angle_deg=angle,
                            proposed=len(records),fitted=sum(c.get('first_contact_fit',{}).get('status')=='fitted' for c in records),
                            valid=selection['valid_count'],selected=selection['status']=='selected',
                            stage_counts=selection['stage_counts'],rescued_candidate_ids=rescued,
                            rescued_with_all_other_checks_already_passed=already_clear,
                            rescued_with_other_checks_previously_unavailable_or_not_evaluated=newly_certified,
                            frozen_pose_sha256=frozen_sha,calibration_sha256=evaluator.calibration['calibration_sha256'],
                            extra_surface_draws=0,extra_contact_fits=0,source_bank=str(bank)))
            print(json.dumps(dict(part=part,strategy=strategy,seed_index=seed_index,
                eight_scenes_complete=True,seconds=round(time.perf_counter()-branch_started,2),
                cumulative_evaluation_counts=dict(evaluator.counters))),flush=True)
    dump(output/'part_receipts'/f'{part}.json',dict(part=part,seconds=time.perf_counter()-started,
        counters=dict(evaluator.counters),static_task_binding_sha256=evaluator.binding))
    return results,angle_results


def summarize(rows):
    groups=[]
    for strategy in sorted({r['sampler_strategy'] for r in rows}):
        for mode in ('pad','finger','combined'):
            selected=[r for r in rows if r['sampler_strategy']==strategy and r['contact_mode']==mode]
            seeds=sorted({r['seed_index'] for r in selected})
            stage_names=list(selected[0]['selection']['stage_counts']) if selected else []
            groups.append(dict(strategy=strategy,contact_mode=mode,placements=len(selected),
                selected=sum(r['selection']['status']=='selected' for r in selected),
                valid=sum(r['selection']['valid_count'] for r in selected),
                proposed=sum(r['selection']['proposed_count'] for r in selected),
                stage_placements={name:sum(r['selection']['stage_counts'][name].get('pass',0)>0 for r in selected) for name in stage_names},
                stage_candidates={name:sum(r['selection']['stage_counts'][name].get('pass',0) for r in selected) for name in stage_names},
                per_seed=[dict(seed_index=s,selected=sum(r['selection']['status']=='selected' for r in selected if r['seed_index']==s),
                    placements=sum(r['seed_index']==s for r in selected)) for s in seeds]))
    return groups


def main(output=None,workers=4,seeds=SEEDS,strategies=STRATEGIES,sensitivity=True,parts=None):
    rows=json.loads(INPUTS.read_text())['records'];validate_observed_rows(rows)
    all_parts=sorted({r['part'] for r in rows})
    if parts:
        if not set(parts)<=set(all_parts):raise ValueError('Unknown requested NIST part')
        rows=[r for r in rows if r['part'] in parts]
    selected_parts=sorted({r['part'] for r in rows})
    output=Path(output).resolve() if output else BASE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=False)
    sources=set(Path(__file__).parent.glob('*.py'))
    for folder in ('cad_fingertip_contacts_20261029','cad_grasp_refinement_20261029',
            'cad_grasp_selection_20261029','cad_grasp_selection_20261028','cad_grasp_geometry_20261026',
            'fabrica_candidates_20261027','fabrica_original_20261027','nist_mating_goals_20261028','nist_scene_corrections_20261029'):
        sources.update((ROOT/'experiments'/folder).glob('*.py'))
    protected={str(p.relative_to(ROOT)):sha(p) for p in [*protected_paths(),INPUTS,HAND,REGISTRY]}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),settings=SETTINGS,seeds=list(seeds),
        strategies=list(strategies),modes=['pad','finger','combined'],parts=selected_parts,workers=workers,
        full_eight_scene_trial=not bool(parts),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)},
        protected_sha256=protected,
        fixed_hand_surface_angle_deg=1.,proposal_normal_angle_deg=1.,validation_normal_angle_deg=1.,
        sensitivity_angles_deg=list(ANGLES) if sensitivity else [],
        sensitivity_scope='Primary random seed, finger mode, exactly frozen calibration and fitted candidate poses; no new fits or proposals.',
        seed_policy='Base seed plus fixed sorted 12-part index; the same seed is paired across samplers and contact policies.',
        per_policy_candidate_cap=100,combined_mode_assignment='50/50 before validation; same shared raw proposal bank; no union of two 100-pose result sets',
        underlying_comparison_evaluations='Two mode-specific candidates per raw configuration. Combined replays a preregistered subset. Benchmark work is distinct from a policy budget.',
        geometry_cache='Only part-local and fixed-final-goal checks reused. Full observed-pickup table distances and approach preference recomputed for every placement.',
        physical_contact_status='Both are explicit simulation contact models; detailed-source mismatch audited, hardware material/contact unverified.',
        VLM=False,pickup=False,assembly=False,physics=False,simulator_object_poses_read=False,
        insertion_path_and_release=False)
    dump(output/'launch_manifest.json',manifest)
    calibration=make_contact_calibration(json.loads(HAND.read_text()));dump(output/'contact_calibration.json',calibration)
    started=time.perf_counter();results=[];angles=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(run_part,part,[r for r in rows if r['part']==part],all_parts.index(part),output,list(seeds),list(strategies),sensitivity) for part in selected_parts]
        for job in as_completed(jobs):
            found,sweep=job.result();results.extend(found);angles.extend(sweep)
    results.sort(key=lambda r:(r['sampler_strategy'],r['seed_index'],r['contact_mode'],r['scene'],r['part']))
    expected=len(selected_parts)*8*len(seeds)*len(strategies)*3
    if len(results)!=expected:raise ValueError('Incomplete policy/scene results')
    dump(output/'results.json',results);dump(output/'summary.json',dict(groups=summarize(results),
        policy_placement_evaluations=len(results),underlying_mode_candidate_evaluations=sum(
            r['selection']['proposed_count'] for r in results if r['contact_mode']!='combined'),
        logical_policy_candidate_evaluations=sum(r['selection']['proposed_count'] for r in results),
        elapsed_wall_seconds=time.perf_counter()-started,
        timing_scope='Shared benchmark, all policies/seeds plus separately recorded validation-only sweep; excludes rendering/publication.',
        physical_success_measured=False))
    dump(output/'frozen_angle_sensitivity.json',angles)
    primary=[r for r in results if r['sampler_strategy']=='task_informed' and r['contact_mode']=='combined' and r['seed_index']==0]
    if primary:
        dump(output/'primary/results.json',primary);dump(output/'primary/summary.json',dict(
            selected_placements=sum(r['selection']['status']=='selected' for r in primary),placements=len(primary),
            valid_poses=sum(r['selection']['valid_count'] for r in primary),
            proposed_poses=sum(r['selection']['proposed_count'] for r in primary),
            primary_seed_preregistered=True,not_best_seed_selection=True))
    if any(sha(ROOT/name)!=expected for name,expected in manifest['source_sha256'].items()):raise ValueError('Launched source changed')
    if any(sha(ROOT/name)!=expected for name,expected in protected.items()):raise ValueError('Protected reference changed')
    dump(output/'run_verification.json',dict(launched_sources_unchanged=True,protected_inputs_unchanged=True,
        complete_policy_matrix=True,per_policy_cap=100,physical_success_measured=False))
    print(json.dumps(dict(output=str(output),policy_placements=len(results),elapsed_seconds=time.perf_counter()-started)),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    parser.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=list(STRATEGIES))
    parser.add_argument('--parts',nargs='+');parser.add_argument('--no-sensitivity',action='store_true')
    args=parser.parse_args()
    main(args.output,args.workers,args.seeds,args.strategies,not args.no_sensitivity,args.parts)
