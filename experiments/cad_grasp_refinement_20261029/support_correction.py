"""Correct finite-pad round-side witnesses using the same 24 generated banks.

The original two complete trials remain unchanged. Only round-side contact
witness interpretation changes. All 72 unaffected refined cases are copied
exactly, and the 24 regenerated banks must match their original coordinates.
This is an incremental validation correction, not a new timed 96-case run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

from . import run_nist as initial
from .report_streaming import audit_variant

PARTS = ('RGOCG12-50_12mm','RGOCG16-50_16mm','RGOCG8-50_8mm')
BANK_FIELDS = ('candidate_id','pair_id','contacts_cad_m','cad_T_hand','cad_T_grip_site',
    'contact_face_ids','contact_normals_cad','approach_cad','holding_opening_m',
    'pre_opening_m','orientation_index','orientation_angle_rad',
    'contact_inset_from_pad_tip_m')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def corrected_part(part, rows, seed, output):
    # Each worker has its own module state. The frozen runner supplies identical
    # data preparation and serialization, with the explicit revised selector.
    from .support_pipeline import select_grasp
    initial.select_grasp=select_grasp
    return initial.run_part(part,rows,seed,output,'refined')


def run(original_root):
    original_root=Path(original_root).resolve()
    if not (original_root/'comparison.json').is_file():
        raise ValueError('The original two full trials must finish before the correction')
    original_manifest=read(original_root/'launch_manifest.json')
    for name,digest in original_manifest['source_sha256'].items():
        if initial.sha(initial.ROOT/name)!=digest:
            raise ValueError('An original launched source changed: '+name)
    rows=read(initial.INPUTS)['records']
    initial.validate_observed_rows(rows)
    output=original_root/'support_correction'
    output.mkdir(exist_ok=False)
    source_paths={initial.ROOT/name for name in original_manifest['source_sha256']}
    source_paths.update(Path(__file__).parent.glob('*.py'))
    source_hashes={str(path.relative_to(initial.ROOT)):initial.sha(path) for path in sorted(source_paths)}
    input_hashes={str(path.relative_to(initial.ROOT)):initial.sha(path)
                  for path in (initial.INPUTS,initial.HAND,initial.REGISTRY)}
    initial.dump(output/'launch_manifest.json',dict(
        phase='Post-launch finite-pad witness correction; 24 round-pin cases only',
        original_comparison=str(original_root),
        original_launch_manifest_sha256=initial.sha(original_root/'launch_manifest.json'),
        settings=initial.SETTINGS, source_sha256=source_hashes, input_sha256=input_hashes,
        changed_logic='Interpret coplanar facet support and numerical edge witnesses consistently; same 1 degree flat alignment and unchanged collision geometry.',
        seed_policy='Exact original per-part seed and full candidate generation settings; verify every bank field after rerun.',
        unaffected_cases=72, corrected_cases=24,
        VLM=False,pickup=False,assembly=False,physics=False,
        note='This receipt is created for the correction, after the original full trial launch.'))
    parts=sorted({row['part'] for row in rows})
    started=time.perf_counter()
    records=[]
    part_timings=[]
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3,mp_context=context) as pool:
        jobs=[pool.submit(corrected_part,part,[row for row in rows if row['part']==part],
                          initial.BASE_SEED+parts.index(part),output) for part in PARTS]
        for job in as_completed(jobs):
            cases,timing=job.result()
            records.extend(cases)
            part_timings.append(timing)
    elapsed=time.perf_counter()-started
    records.sort(key=lambda row:(row['scene'],row['part']))
    bank_checks=[]
    for row in records:
        relative=Path('cases')/row['scene']/row['part']/'candidate_diagnostics.json'
        before=read(original_root/'refined'/relative)
        after=read(output/relative)
        if len(before)!=len(after):
            raise ValueError('A corrected candidate bank changed length')
        for old,new in zip(before,after):
            for key in BANK_FIELDS:
                if old.get(key)!=new.get(key):
                    raise ValueError(f"A generated pose changed: {row['scene']}/{row['part']}/{key}")
        old_valid={c['candidate_id'] for c in before if c['validity']['status']=='valid'}
        new_valid={c['candidate_id'] for c in after if c['validity']['status']=='valid'}
        if not old_valid.issubset(new_valid):
            raise ValueError('The contact-witness correction rejected an originally valid round pose')
        bank_checks.append(dict(scene=row['scene'],part=row['part'],poses=len(after),
            generated_fields_identical=True,previous_valid_ids=sorted(old_valid),
            corrected_valid_ids=sorted(new_valid)))
    initial.dump(output/'results.json',records)
    initial.dump(output/'bank_identity.json',bank_checks)
    initial.dump(output/'summary.json',dict(cases=24,elapsed_wall_seconds=elapsed,
        selected_placements=sum(row['selection']['status']=='selected' for row in records),
        valid_poses=sum(row['selection']['valid_count'] for row in records),
        generated_banks_match_original=True,part_timings=part_timings,
        timing_scope='Only the 24 round-pin reruns; not a new complete 96-case wall time.'))
    final=original_root/'support_corrected'
    shutil.copytree(original_root/'refined',final)
    original_rows=read(original_root/'refined/results.json')
    revised={(row['scene'],row['part']):row for row in records}
    merged=[]
    derivation=[]
    for old in original_rows:
        key=old['scene'],old['part']
        corrected=key in revised
        row=revised.get(key,old)
        source=output if corrected else original_root/'refined'
        relative=Path('cases')/row['scene']/row['part']
        if corrected:
            shutil.copytree(source/relative,final/relative,dirs_exist_ok=True)
        files={name:initial.sha(source/relative/name)
               for name in ('selection.json','candidate_diagnostics.json')}
        if any(initial.sha(final/relative/name)!=digest for name,digest in files.items()):
            raise ValueError('Derived case was not copied exactly')
        merged.append(row)
        derivation.append(dict(scene=row['scene'],part=row['part'],
            source_directory=str(source/relative),source_file_sha256=files,
            finite_pad_witness_correction=corrected,case_files_copied_exactly=True))
    summary=read(original_root/'refined/summary.json')
    initial_refined_seconds=summary['elapsed_wall_seconds']
    summary.update(selected_placements=sum(r['selection']['status']=='selected' for r in merged),
        valid_poses=sum(r['selection']['valid_count'] for r in merged),
        proposed_poses=sum(r['selection']['proposed_count'] for r in merged),
        outcome_counts=dict(Counter(r['selection']['status'] for r in merged)),
        elapsed_wall_seconds=None,assembled_from_multiple_trials=True,
        initial_full_trial_wall_seconds=initial_refined_seconds,
        additional_24_case_correction_wall_seconds=elapsed,
        timing_scope='No new complete 96-case wall time measured. Initial full trial and additional 24-case correction times are reported separately.',
        per_part={part:dict(selected=sum(r['selection']['status']=='selected' for r in merged if r['part']==part),
                           valid_poses=sum(r['selection']['valid_count'] for r in merged if r['part']==part),
                           proposed_poses=sum(r['selection']['proposed_count'] for r in merged if r['part']==part))
                  for part in parts})
    initial.dump(final/'results.json',merged)
    initial.dump(final/'summary.json',summary)
    initial.dump(final/'derivation.json',dict(
        original_trials_preserved=True,unchanged_cases=72,corrected_cases=24,
        no_unrecorded_new_search=True,generated_round_banks_match_original=True,cases=derivation))
    hand,registry=read(initial.HAND),read(initial.REGISTRY)
    audit,_,_,watched=audit_variant(final,hand,registry,{})
    if any(initial.sha(path)!=digest for path,digest in watched.items()):
        raise ValueError('Derived result changed during independent audit')
    if any(initial.sha(initial.ROOT/name)!=digest for name,digest in source_hashes.items()):
        raise ValueError('Correction source changed after its launch')
    if any(initial.sha(initial.ROOT/name)!=digest for name,digest in input_hashes.items()):
        raise ValueError('Correction inputs changed after its launch')
    initial.dump(final/'verification.json',dict(
        independent_saved_result_audit=audit,source_and_input_hashes_unchanged=True,
        original_trials_preserved=True,exactly_copied_cases=96,
        identical_regenerated_round_banks=24,
        robot_or_physical_success_measured=False))
    previous=read(original_root/'previous/summary.json')
    lines=['# Corrected finite-pad contact results','',
        'The original two full trials are preserved. Their round-pin pad checker mixed curved-surface directions with whole flat-facet contact patches. This correction handles actual coplanar facet support and numerical edge witnesses separately, keeping the 1° contact requirement and complete collision meshes.', '',
        'All 24 round-pin cases were rerun with exactly the same generated pose banks. The other 72 case files are exact copies from the first refined trial. This is an incrementally corrected 96-case result, not a newly timed full run.', '',
        f"Baseline: **{previous['selected_placements']}/96** selected placements. Corrected refinement: **{summary['selected_placements']}/96**. These counts measure candidate availability, not pickup or assembly success.", '',
        '| Part | Baseline selected /8 | Corrected selected /8 | Corrected valid poses |',
        '|---|---:|---:|---:|']
    for part in parts:
        a,b=previous['per_part'][part],summary['per_part'][part]
        lines.append(f"| {part.replace('_',' ')} | {a['selected']} | {b['selected']} | {b['valid_poses']} |")
    labels=dict(surface_normals='Intended contact alignment',pad_contact_geometry='Complete pad contact',
        target_closing_collision='Finger or palm enters the part during closing',
        table_clearance='Finger or palm reaches the table',
        final_receiver_collision='Gripper intersects the final receiver',
        protected_surfaces='Pad covers a surface needed for assembly')
    lines += ['', '## Remaining failures', '',
        'Counts below are by candidate and overlap: the same pose can fail several checks. They describe the bounded sampled banks, not every possible grasp.', '',
        '| Part | Intended alignment fails | Pad contact fails | Closing collision | Table | Final receiver | Protected surfaces |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for part in parts:
        failures=Counter()
        for case in audit['case_audits']:
            if case['part']==part:
                for key,counts in case['independent_check_counts'].items():
                    failures[key]+=counts.get('fail',0)
        keys=('surface_normals','pad_contact_geometry','target_closing_collision',
              'table_clearance','final_receiver_collision','protected_surfaces')
        lines.append('| '+part.replace('_',' ')+' | '+' | '.join(str(failures[key]) for key in keys)+' |')
    baseline_rows=read(original_root/'previous/results.json')
    before_selected={(r['scene'],r['part']) for r in baseline_rows if r['selection']['status']=='selected'}
    after_selected={(r['scene'],r['part']) for r in merged if r['selection']['status']=='selected'}
    describe=lambda values:', '.join(scene+' '+part.replace('_',' ') for scene,part in sorted(values)) or 'none'
    lines += ['', 'Newly available placements: '+describe(after_selected-before_selected)+'.', '',
        'Placements selected by the baseline but missing from the final fixed-budget bank: '+describe(before_selected-after_selected)+'.', '',
        'A stricter proposal rule changes which pairs enter the seeded 100-pose sample and can lose a previously sampled valid orientation. This combined revision is not guaranteed to improve every placement. No retries were used to repair the reported outcomes.']
    lines += ['',f"Original complete trials: baseline {previous['elapsed_wall_seconds']:.2f} s; initial refinement {initial_refined_seconds:.2f} s. Additional 24-case correction: {elapsed:.2f} s. The final merged result has no separately measured full-run wall time.",'',
        'The complete saved finger geometry is unchanged. The separate 8 mm pin height audit limits its declared straight-down side grip in seven lying scenes; it does not prove that every possible orientation fails.', '',
        'The 1° flat alignment, original CAD ray/collision geometry, protected regions, 20 mm pin depth and selection priorities are retained. Cylinder-to-mesh approximation limits are disclosed in the contact checker; physical contact precision is unverified.', '',
        '[Selected grasps and pickup side views](gallery/index.html) · [Independent audit](verification.json) · [Exact case provenance](derivation.json) · [Unchanged generated banks](../support_correction/bank_identity.json) · [Original full-trial report](../report.md)', '']
    (final/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(final=str(final),selected=summary['selected_placements'],
                         round_correction_seconds=elapsed)),flush=True)
    return final


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_refinement_20261029.support_correction ORIGINAL_ROOT')
    run(sys.argv[1])
