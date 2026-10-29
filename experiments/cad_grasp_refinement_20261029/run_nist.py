"""Compare previous and refined contact construction on 96 saved CAD alignments."""
from __future__ import annotations

import os
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from experiments.cad_grasp_selection_20261028.run_nist import dump, sha, protected_paths
from experiments.cad_grasp_selection_20261029.task_inputs import build_request, REGISTRY, HAND, ROOT
from experiments.nist_scene_corrections_20261029.history import verify_frozen_run
from .contact_model import build_contact_model
from .pipeline import select_grasp

INPUTS = ROOT/'outputs/cad_grasp_geometry_20261026/inputs.json'
BASE = ROOT/'outputs/cad_grasp_refinement_20261029/runs'
SETTINGS = dict(max_poses=100, surface_budget=200, orientation_samples=10,
                pre_opening_clearance_m=.001)
BASE_SEED = 20261028


def validate_observed_rows(rows):
    if (len(rows)!=96 or len({row['part'] for row in rows})!=12
            or len({(row['scene'],row['part']) for row in rows})!=96
            or sorted({row['scene'] for row in rows}) != [f'N{i:02d}' for i in range(8)]):
        raise ValueError('Expected twelve CAD products across all eight saved scenes')
    if any(row.get('simulator_object_poses_read') is not False for row in rows):
        raise ValueError('Every saved alignment must explicitly exclude simulator object poses')


def _diagnostic_counts(diagnostics):
    counts = {}
    for candidate in diagnostics:
        for category, checked in candidate['validity'].get('diagnostics', {}).items():
            status = checked.get('status', 'unavailable') if isinstance(checked, dict) else str(checked)
            counts.setdefault(category, Counter())[status] += 1
    return {key:dict(value) for key,value in sorted(counts.items())}


def run_part(part, rows, seed, output, variant):
    started = time.perf_counter()
    hand = json.loads(HAND.read_text(encoding='utf-8'))
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    base = build_request(part, rows[0]['world_T_cad'], rows[0]['table_plane_world'],
                         registry=registry, hand=hand)
    if base['status'] != 'available':
        raise ValueError(f"{part}: {base.get('message')}")
    request_base = dict(base['request'], part=part)
    model = None
    model_started = time.perf_counter()
    if variant == 'refined':
        model = build_contact_model(request_base['vertices_cad_m'], request_base['faces'],
            request_base['sampling_contact_region'], part=part,
            axis_point_cad_m=request_base['axis_point_cad_m'],
            axis_direction_cad=request_base['axis_direction_cad'])
        request_base['contact_model'] = model
    model_seconds = time.perf_counter()-model_started
    preparation_seconds = time.perf_counter()-started
    dump(Path(output)/'task_inputs'/f'{part}.json', dict(
        source=base['source'], contact_model=model.provenance if model is not None else None,
        complete_CAD_vertices=len(request_base['vertices_cad_m']),
        complete_CAD_triangles=len(request_base['faces']),
        static_input_preparation_seconds=preparation_seconds,
        contact_model_preparation_seconds=model_seconds))
    records = []
    for row in sorted(rows, key=lambda item:item['scene']):
        case_started = time.perf_counter()
        diagnostics = []
        if row.get('input_status') != 'available' or row.get('world_T_cad') is None:
            selection = dict(status='unavailable_inputs', selected_grasp=None,
                proposed_count=0, valid_count=0, message='Saved observed CAD alignment is unavailable.')
        else:
            request = dict(request_base, world_T_cad_pickup=np.asarray(row['world_T_cad']),
                           table_plane_world=np.asarray(row['table_plane_world']),
                           simulator_object_poses_read=row.get('simulator_object_poses_read'))
            selection = select_grasp(request, seed=seed, sampler_variant=variant,
                                     diagnostics=diagnostics, **SETTINGS)
        reasons = Counter(reason for candidate in diagnostics
                          for reason in candidate['validity'].get('reasons', []))
        record = dict(scene=row['scene'], part=part, mesh_path=row['mesh_path'],
            world_T_cad=row['world_T_cad'], table_plane_world=row['table_plane_world'],
            pose_source=row.get('pose_source'), observed_alignment_status=row['input_status'],
            selection=selection, seed=seed, sampler_variant=variant,
            wall_seconds=time.perf_counter()-case_started,
            rejection_counts=dict(reasons), independent_check_counts=_diagnostic_counts(diagnostics),
            no_VLM=True, no_pickup_or_assembly_execution=True, no_physics_steps=True,
            no_new_pose_estimation=True, no_insertion_or_release_checks=True,
            simulator_object_poses_read=row.get('simulator_object_poses_read'),
            final_pose_status='corrected CAD reference; physical mating not validated')
        folder = Path(output)/'cases'/row['scene']/part
        dump(folder/'candidate_diagnostics.json', diagnostics)
        dump(folder/'selection.json', record)
        records.append(record)
        print(json.dumps(dict(variant=variant, scene=row['scene'], part=part,
            status=selection['status'], proposed=selection['proposed_count'],
            valid=selection['valid_count'], seconds=round(record['wall_seconds'],3))), flush=True)
    return records, dict(part=part, total_seconds=time.perf_counter()-started,
                         input_preparation_seconds=preparation_seconds,
                         contact_model_preparation_seconds=model_seconds)


def run_variant(rows, output, variant, workers):
    output.mkdir(parents=True, exist_ok=False)
    parts = sorted({row['part'] for row in rows})
    started = time.perf_counter()
    records, timings = [], []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        jobs = [pool.submit(run_part, part, [row for row in rows if row['part']==part],
                            BASE_SEED+index, output, variant)
                for index,part in enumerate(parts)]
        for job in as_completed(jobs):
            found, timing = job.result()
            records.extend(found)
            timings.append(timing)
    elapsed = time.perf_counter()-started
    records.sort(key=lambda row:(row['scene'], row['part']))
    if len(records)!=96:
        raise ValueError('Expected 96 completed case records')
    summary = dict(sampler_variant=variant, placements=len(records),
        selected_placements=sum(row['selection']['status']=='selected' for row in records),
        valid_poses=sum(row['selection']['valid_count'] for row in records),
        proposed_poses=sum(row['selection']['proposed_count'] for row in records),
        outcome_counts=dict(Counter(row['selection']['status'] for row in records)),
        elapsed_wall_seconds=elapsed,
        summed_case_seconds=sum(row['wall_seconds'] for row in records),
        summed_sampling_seconds=sum(row['selection'].get('timings_seconds',{}).get('sampling',0) for row in records),
        per_part={part:dict(selected=sum(r['selection']['status']=='selected' for r in records if r['part']==part),
                           valid_poses=sum(r['selection']['valid_count'] for r in records if r['part']==part),
                           proposed_poses=sum(r['selection']['proposed_count'] for r in records if r['part']==part))
                  for part in parts},
        workers=workers, part_timings=sorted(timings,key=lambda x:x['part']),
        timing_scope='Wall time includes input/region/model preparation, sampling, exhaustive geometric checks, selection, worker startup and case JSON writes; excludes galleries and post-run verification.',
        checks='Independent diagnostic categories share the same exhaustive checker in both variants.',
        pickup_or_assembly_success_measured=False, VLM=False, physics=False)
    dump(output/'results.json', records)
    dump(output/'summary.json', summary)
    return summary


def main(output=None, workers=4):
    rows = json.loads(INPUTS.read_text(encoding='utf-8'))['records']
    validate_observed_rows(rows)
    output = Path(output) if output else BASE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=False)
    preserved = {str(path.relative_to(ROOT)):sha(path) for path in protected_paths()}
    source_paths = [*Path(__file__).parent.glob('*.py'),
        *(ROOT/'experiments/cad_grasp_selection_20261029').glob('*.py'),
        ROOT/'experiments/cad_grasp_selection_20261029/config.json', REGISTRY,
        *(ROOT/'experiments/cad_grasp_selection_20261028').glob('*.py'),
        *(ROOT/'experiments/cad_grasp_geometry_20261026').glob('*.py'),
        *(ROOT/'experiments/fabrica_candidates_20261027').glob('*.py'),
        *(ROOT/'experiments/fabrica_original_20261027').glob('*.py'),
        *(ROOT/'experiments/nist_mating_goals_20261028').glob('*.py'),
        *(ROOT/'experiments/nist_scene_corrections_20261029').glob('*.py')]
    sources = {str(path.relative_to(ROOT)):sha(path) for path in sorted(source_paths)}
    manifest = dict(settings=SETTINGS, workers=workers, base_seed=BASE_SEED,
        seed_policy='One fixed seed per sorted CAD product, reused across scenes and variants; no retry search.',
        variants=['previous','refined'], source_sha256=sources, protected_sha256=preserved,
        observed_manifest=str(INPUTS.relative_to(ROOT)), observed_manifest_sha256=sha(INPUTS),
        hand_model_sha256=sha(HAND), corrected_registry_sha256=sha(REGISTRY),
        scope='Two 96-case CAD-only candidate/selection trials. No robot or simulation execution.',
        previous_variant='Current region-guided sampler, fixed 0.5 mm ray offset, facet normals and broad proposal threshold; shared exhaustive validator with strict facet contact normals.',
        refined_variant='Surface-aware contact construction, safe ray starts and shared strict proposal/validator contact policy; same regions, goals, hand and criterion order.',
        old_Oct28_results='Historical reference only: different goals, sampling domain and pad-center convention; not a controlled ablation.',
        independent_diagnostics=True, VLM=False, pickup=False, assembly=False,
        physics=False, simulator_object_poses_read=False, insertion_and_release_checks=False)
    dump(output/'launch_manifest.json',manifest)
    summaries = {}
    for variant in ('previous','refined'):
        summaries[variant] = run_variant(rows, output/variant, variant, workers)
    after = {str(path.relative_to(ROOT)):sha(path) for path in protected_paths()}
    if preserved!=after:
        raise ValueError('A protected input or historical result changed during the study')
    source_changes = [path for path,digest in sources.items() if sha(ROOT/path)!=digest]
    if source_changes:
        raise ValueError(f'Launched source changed: {source_changes}')
    dump(output/'comparison.json',dict(variants=summaries,
        protected_inputs_unchanged=True, launched_sources_unchanged=True,
        historical_evidence_verification=verify_frozen_run(),
        pickup_or_assembly_success_measured=False))
    print(json.dumps(dict(output=str(output), previous=summaries['previous']['selected_placements'],
                         refined=summaries['refined']['selected_placements'])), flush=True)
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    main(args.output,args.workers)
