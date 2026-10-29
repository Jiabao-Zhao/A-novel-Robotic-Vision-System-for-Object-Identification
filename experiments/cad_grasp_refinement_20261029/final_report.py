"""Resume independent reporting after the finite-pad and audit corrections.

No grasp is generated or revalidated here. Original trial files stay intact.
The reporter repair is resolved to its exact saved pre-repair source bytes.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys

from . import report_streaming as reporting
from .run_nist import ROOT, HAND, REGISTRY, dump, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def report(root):
    root=Path(root).resolve()
    final=root/'support_corrected'
    correction=root/'support_correction'
    source_resolution=[]
    for phase in (root,correction):
        manifest=read(phase/'launch_manifest.json')
        for name,expected in manifest['source_sha256'].items():
            path=ROOT/name
            if sha(path)==expected:
                continue
            snapshot=correction/'frozen_sources/report_streaming.py'
            if path.name=='report_streaming.py' and snapshot.is_file() and sha(snapshot)==expected:
                source_resolution.append(dict(source=name,expected_sha256=expected,
                    exact_saved_original=str(snapshot),current_reporter_sha256=sha(path),
                    reason='Post-processing repair: compare each hand component with its own table minimum, rather than the whole-hand minimum. Grasp results are unchanged.'))
            else:
                raise ValueError('Unexpected source change: '+name)
        for name,expected in manifest.get('protected_sha256',manifest.get('input_sha256',{})).items():
            if sha(ROOT/name)!=expected:
                raise ValueError('Protected trial input changed: '+name)
    identities=read(correction/'bank_identity.json')
    if len(identities)!=24 or not all(row['generated_fields_identical'] for row in identities):
        raise ValueError('The 24 corrected banks are not identical to their original poses')
    derivation=read(final/'derivation.json')
    if len(derivation['cases'])!=96:
        raise ValueError('Missing derived case provenance')
    for row in derivation['cases']:
        for name,expected in row['source_file_sha256'].items():
            source=Path(row['source_directory'])/name
            dest=final/'cases'/row['scene']/row['part']/name
            if sha(source)!=expected or sha(dest)!=expected:
                raise ValueError('A derived case is not an exact copy of its recorded source')
    rows=read(final/'results.json')
    summary=read(final/'summary.json')
    if 'part_timings' in summary:
        summary['initial_full_trial_part_timings']=summary.pop('part_timings')
    summary['initial_full_trial_summed_case_seconds']=read(root/'refined/summary.json')['summed_case_seconds']
    summary['summed_case_seconds']=sum(row['wall_seconds'] for row in rows)
    summary['summed_sampling_seconds']=sum(row['selection'].get('timings_seconds',{}).get('sampling',0) for row in rows)
    summary['retained_case_timing_scope']='Sum of the 72 original and 24 corrected retained case executions, not elapsed time for a new 96-case run.'
    dump(final/'summary.json',summary)
    audit,rows,summary,watched=reporting.audit_variant(final,read(HAND),read(REGISTRY),{})
    if any(sha(path)!=expected for path,expected in watched.items()):
        raise ValueError('A saved result changed during independent reporting')
    dump(final/'verification.json',dict(
        independent_saved_result_audit=audit,
        source_resolution=source_resolution,
        original_launched_grasp_sources_unchanged=True,
        protected_trial_inputs_unchanged=True,
        postprocessing_helper_sha256={str(path.relative_to(ROOT)):sha(path)
            for path in (Path(__file__),Path(reporting.__file__))},
        no_new_candidate_generation_or_collision_checks=True,
        identical_regenerated_round_banks=24,
        unchanged_nonround_cases_copied_exactly=72,
        case_files_copied_exactly=96,
        robot_or_physical_success_measured=False))
    with (final/'placement_counts.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['scene','part','status','proposed','valid','selected_candidate',
                         'moment_arm_mm','surface_center_distance_mm','angle_from_down_deg'])
        for row in rows:
            selection=row['selection']
            pose=selection.get('selected_grasp')
            values=[row['scene'],row['part'],selection['status'],selection['proposed_count'],selection['valid_count']]
            if pose is None:
                values+=['','','','']
            else:
                criteria=pose['criteria']
                values += [pose['candidate_id'],criteria['q_tau_m']*1000,
                           criteria['q_surface_m']*1000,criteria['theta_down_rad']*180/3.141592653589793]
            writer.writerow(values)
    previous=read(root/'previous/summary.json')
    original_refined=read(root/'refined/summary.json')
    corrected=read(correction/'summary.json')
    labels={'surface_normals':'Intended contact alignment','pad_contact_geometry':'Complete pad contact',
        'target_closing_collision':'Finger or palm enters the part during closing',
        'table_clearance':'Finger or palm reaches the table',
        'final_receiver_collision':'Gripper intersects the final receiver',
        'protected_surfaces':'Pad covers a surface needed for assembly'}
    keys=list(labels)
    lines=['# Final CAD grasp refinement results','',
        f"The current twelve CAD products were checked in all eight saved scenes: 96 placements. The region-guided baseline selected grasps for **{previous['selected_placements']}/96** placements; the corrected refinement selected **{summary['selected_placements']}/96**. Valid sampled poses increased from **{previous['valid_poses']}** to **{summary['valid_poses']}**, out of 9,600 retained proposals per method. These are geometric candidate counts, not pickup or assembly success.",'',
        '## What was checked','',
        'Both methods use the same corrected assembly goals, protected regions, positive sampling labels, full hand, observed CAD alignments, table planes, seeds and 100-pose cap. Priorities remain mating-axis moment arm, surface-centroid distance and downward preference. There is no VLM selection, robot movement, physics, insertion-path or release execution. All poses and task input frames remain explicit.','',
        'The first full refinement exposed a contact-representation mismatch. Intended contacts used verified cylinder directions, while a finite flat CAD facet was incorrectly checked using curved-surface directions at all its corners. The correction distinguishes actual coplanar facet support from numerical edge ribbons. It retains the 1° requirement and complete collision geometry. Numerical ribbons within 0.8 nm of the actual inner-pad plane are reduced to their exact line/point intersection; genuine penetration and other pad surfaces are not removed.','',
        'All 24 round-pin cases were rerun, and every generated pose field was verified identical to the first refined bank. The other 72 case files are exact copies. The original two full trials remain unchanged. The final 96-case result is an incrementally corrected result, with explicit per-case provenance, rather than a newly timed full run.','',
        '| Part | Baseline selected /8 | Corrected selected /8 | Corrected valid poses /800 |',
        '|---|---:|---:|---:|']
    for part in sorted(summary['per_part']):
        a,b=previous['per_part'][part],summary['per_part'][part]
        lines.append(f"| {part.replace('_',' ')} | {a['selected']} | {b['selected']} | {b['valid_poses']} |")
    lines += ['', '## Remaining failed checks','',
        'Counts below overlap: one pose may fail several checks. They describe the sampled banks and do not establish that no feasible grasp exists.','',
        '| Part | Contact alignment | Pad contact | Closing collision | Table | Final receiver | Protected surfaces |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for part in sorted(summary['per_part']):
        counts=Counter()
        for case in audit['case_audits']:
            if case['part']==part:
                for key,statuses in case['independent_check_counts'].items():
                    counts[key]+=statuses.get('fail',0)
        lines.append('| '+part.replace('_',' ')+' | '+' | '.join(str(counts[key]) for key in keys)+' |')
    before={(row['scene'],row['part']) for row in read(root/'previous/results.json') if row['selection']['status']=='selected'}
    after={(row['scene'],row['part']) for row in rows if row['selection']['status']=='selected'}
    describe=lambda values:', '.join(scene+' '+part.replace('_',' ') for scene,part in sorted(values)) or 'none'
    lines += ['', 'Newly available placements: '+describe(after-before)+'.','',
        'Baseline placements absent from the final fixed-budget bank: '+describe(before-after)+'.','',
        'The changed proposal rule changes which pairs and angles enter the seeded 100-pose sample. It can lose a previously sampled valid pose; this revision does not improve every placement. No retry search was used to repair the reported outcomes.','',
        '## Finger height and timing','',
        'The saved complete fingers extend 5.099 mm beyond the declared near-tip contact anchors. The separate straight-down 8 mm pin audit finds about 1.1 mm table penetration in seven lying scenes, while N05 is upright. That audit concerns its declared side-grip construction; it does not rule out every other orientation. The final search retains only the upright N05 8 mm round-pin placement. The hand geometry was not shortened or enlarged.','',
        f"Original complete trial wall times: baseline **{previous['elapsed_wall_seconds']:.2f} s**; initial refinement **{original_refined['elapsed_wall_seconds']:.2f} s**. Additional 24-case correction: **{corrected['elapsed_wall_seconds']:.2f} s**. No separate full-run wall time was measured for the final merged result. Trial times include exhaustive checking and case output; rendering and independent reporting are separate. This evidence does not establish an overall runtime improvement.",'',
        '## Verification','',
        'The audit independently recomputes all retained hand transforms, near-tip contact anchors, normalized criteria, deterministic selection priorities and complete-hand table distances. Each component is compared with its own endpoint minimum, and the whole hand with the overall minimum. It does not rerun the target/receiver collision engine or measure physical holding/assembly accuracy.','',
        'Only the three round-pin sides verify the declared cylindrical model. BNC retains its original surface directions because its textured barrel is not one straight cylinder. Nominal fitted-cylinder chord-sag estimates are recorded in selected validity metadata, with vertex-fit residuals separately recorded in the contact model; physical precision remains unverified.','',
        'The first post-processing audit incorrectly compared per-component table heights against the global minimum. Its exact pre-repair source is preserved and hash-checked. The reporter was corrected without changing generated grasps, geometry decisions or original trial results. The source resolution is explicit in verification.json.','',
        '[Selected grasps and pickup side views](gallery/index.html) · [Independent verification](verification.json) · [All placement counts](placement_counts.csv) · [Exact case provenance](derivation.json) · [Identical regenerated banks](../support_correction/bank_identity.json) · [Finger-height audit](../fingertip_height_audit.json) · [Original full-trial report](../report.md)','']
    (final/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(final_report=str(final/'report.md'),selected=summary['selected_placements'],
                         valid_poses=summary['valid_poses'],verification='passed')),flush=True)
    return final


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_refinement_20261029.final_report ORIGINAL_ROOT')
    report(sys.argv[1])
