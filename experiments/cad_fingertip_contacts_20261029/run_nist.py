"""Actual modeled fingertip contacts across the eight saved NIST CAD scenes."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

from experiments.cad_grasp_refinement_20261029 import run_nist as previous
from experiments.nist_scene_corrections_20261029.history import verify_frozen_run
from .contact_audit import make_contact_calibration

ROOT=previous.ROOT
BASE=ROOT/'outputs/cad_fingertip_contacts_20261029/runs'
REFERENCE=ROOT/'outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected'


def run_part(part,rows,seed,output):
    from .pipeline import select_grasp
    previous.select_grasp=select_grasp
    return previous.run_part(part,rows,seed,output,'refined')


def main(output=None,workers=4):
    rows=json.loads(previous.INPUTS.read_text())['records']
    previous.validate_observed_rows(rows)
    output=Path(output).resolve() if output else BASE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=False)
    hand=json.loads(previous.HAND.read_text())
    calibration=make_contact_calibration(hand)
    previous.dump(output/'contact_calibration.json',calibration)
    source_paths=set(Path(__file__).parent.glob('*.py'))
    for folder in ('cad_grasp_refinement_20261029','cad_grasp_selection_20261029',
                   'cad_grasp_selection_20261028','cad_grasp_geometry_20261026',
                   'fabrica_candidates_20261027','fabrica_original_20261027',
                   'nist_mating_goals_20261028','nist_scene_corrections_20261029'):
        source_paths.update((ROOT/'experiments'/folder).glob('*.py'))
    sources={str(p.relative_to(ROOT)):previous.sha(p) for p in sorted(source_paths)}
    protected={str(p.relative_to(ROOT)):previous.sha(p) for p in previous.protected_paths()}
    for p in (previous.INPUTS,previous.HAND,previous.REGISTRY,REFERENCE/'results.json',REFERENCE/'summary.json'):
        protected[str(p.relative_to(ROOT))]=previous.sha(p)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,protected_sha256=protected,
        settings=previous.SETTINGS,workers=workers,base_seed=previous.BASE_SEED,
        seed_policy='Same fixed per-part seeds and CAD ray/angle bank as the saved refined reference; no retries.',
        contact_calibration_sha256=calibration['calibration_sha256'],
        hand_geometry_changed=False,contact_labels_changed=True,
        actual_contacts='First-contact surface witnesses on the original complete finger hulls; old pads remain collision solids.',
        physical_contact_accuracy_validated=False,
        VLM=False,pickup=False,assembly=False,physics=False,simulator_object_poses_read=False,
        insertion_path_and_release_checks=False)
    previous.dump(output/'launch_manifest.json',manifest)
    started=time.perf_counter()
    parts=sorted({row['part'] for row in rows})
    records,timings=[],[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(run_part,part,[r for r in rows if r['part']==part],previous.BASE_SEED+i,output)
              for i,part in enumerate(parts)]
        for job in as_completed(jobs):
            found,timing=job.result();records.extend(found);timings.append(timing)
    elapsed=time.perf_counter()-started
    records.sort(key=lambda row:(row['scene'],row['part']))
    if len(records)!=96:raise ValueError('Missing saved scene/part outcomes')
    summary=dict(placements=96,selected_placements=sum(r['selection']['status']=='selected' for r in records),
        proposed_poses=sum(r['selection']['proposed_count'] for r in records),
        valid_poses=sum(r['selection']['valid_count'] for r in records),elapsed_wall_seconds=elapsed,
        per_part={part:dict(selected=sum(r['selection']['status']=='selected' for r in records if r['part']==part),
            valid_poses=sum(r['selection']['valid_count'] for r in records if r['part']==part),
            proposed_poses=sum(r['selection']['proposed_count'] for r in records if r['part']==part)) for part in parts},
        part_timings=sorted(timings,key=lambda row:row['part']),workers=workers,
        original_hand_geometry_unchanged=True,physical_or_simulated_execution=False,
        timing_scope='Complete new 96-case trial including first-contact fits cached by identical per-part CAD banks, checking and JSON writes; excludes gallery/report.',
        comparison_reference=str(REFERENCE))
    previous.dump(output/'results.json',records)
    previous.dump(output/'summary.json',summary)
    if any(previous.sha(ROOT/p)!=digest for p,digest in sources.items()):
        raise ValueError('Launched source changed during trial')
    if any(previous.sha(ROOT/p)!=digest for p,digest in protected.items()):
        raise ValueError('Protected input or reference changed during trial')
    identities=[]
    for row in records:
        rel=Path('cases')/row['scene']/row['part']/'candidate_diagnostics.json'
        old=json.loads((REFERENCE/rel).read_text())
        new=json.loads((output/rel).read_text())
        if len(old)!=len(new):raise ValueError('Raw CAD bank count changed')
        for a,b in zip(old,new):
            if (a['candidate_id']!=b['candidate_id'] or a['contacts_cad_m']!=b['proposal_contacts_cad_m']
                or a['approach_cad']!=b['approach_cad']):
                raise ValueError('Raw CAD proposal bank changed')
        identities.append(dict(scene=row['scene'],part=row['part'],raw_pairs_and_approaches_identical=True,
            count=len(new)))
    previous.dump(output/'bank_identity.json',identities)
    previous.dump(output/'run_verification.json',dict(launched_sources_unchanged=True,
        protected_inputs_unchanged=True,raw_pairs_and_approaches_identical_in_all_96_cases=True,
        historical_evidence=verify_frozen_run(),physical_success_measured=False))
    print(json.dumps(dict(output=str(output),selected=summary['selected_placements'],valid=summary['valid_poses'],seconds=elapsed)),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    main(args.output,args.workers)
