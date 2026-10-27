"""Summarize saved geometry decisions without changing candidate selection."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np


def main(run):
    run = Path(run)
    rows = json.loads((run/'results.json').read_text())
    if len(rows) != 96:
        raise ValueError('Expected all 96 placements before reporting the full experiment')
    manifest = json.loads((run/'manifest.json').read_text())
    parts = sorted({r['part'] for r in rows})
    summary = []
    rejections = Counter()
    for part in parts:
        cases = [r for r in rows if r['part']==part]
        groups = [r['grouped_count'] for r in cases]
        summary.append(dict(part=part, cases=len(cases), with_grasps=sum(r['valid_count']>0 for r in cases),
              generated=sum(r.get('total_proposals_checked_including_base',r['generated_count']) for r in cases), valid=sum(r['valid_count'] for r in cases),
              grouped=sum(groups), min_groups=min(groups), max_groups=max(groups),
              median_seconds=float(np.median([r['timing_seconds'] for r in cases])),
              mean_seconds=float(np.mean([r['timing_seconds'] for r in cases]))))
        for r in cases:
            rejections.update(r['rejection_counts'])
            rejections.update(r.get('base_rejection_counts',{}))
    timings = [r['timing_seconds'] for r in rows]
    passing = [g for r in rows for g in r['grasps']]
    result = dict(cases=len(rows), placements_with_grasps=sum(r['valid_count']>0 for r in rows),
           generated=sum(r.get('total_proposals_checked_including_base',r['generated_count']) for r in rows), valid=sum(r['valid_count'] for r in rows),
           grouped=sum(r['grouped_count'] for r in rows),
           median_seconds=float(np.median(timings)), mean_seconds=float(np.mean(timings)),
           minimum_seconds=float(min(timings)), maximum_seconds=float(max(timings)),
           summed_placement_seconds=float(sum(timings)),
           minimum_retained_table_gap_m=float(min(g['table_clearance_m'] for g in passing)),
           maximum_contact_opening_mismatch_m=float(max(g['contact_opening_difference_m'] for g in passing)),
           no_grasp_cases=[dict(scene=r['scene'],part=r['part'], reasons=r['rejection_counts']) for r in rows if not r['valid_count']],
           by_part=summary, primary_rejection_counts=dict(rejections),
           timing_note='Four concurrent CPU workers. Per-placement times include loading, setup, sampling, validation, grouping and candidate evidence writing; imports and input pose fitting excluded. Sum is not elapsed wall time.')
    (run/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (run/'per_part.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ['# Pre-VLM CAD grasp geometry results', '',
      f"**{result['placements_with_grasps']}/96 placements have at least one passing geometric grasp.**",
      'Eight existing N00--N07 scenes, twelve parts each. These are model-only checks, not pickup or assembly success.', '',
      f"Generated and checked {result['generated']:,} hand poses. {result['valid']:,} passed, represented by {result['grouped']:,} grouped options. No VLM was called and no winning grasp was chosen.", '',
      'Correct CAD/crop association was supplied. Positions/orientations came from saved observed-depth CAD fits and the saved camera transform; simulator object poses were not read. The table plane came from observed depth.', '',
      '| Part | Placements with grasps | Grouped options per scene | Median time per placement |',
      '|---|---:|---:|---:|']
    lines += [f"| {s['part']} | {s['with_grasps']}/8 | {s['min_groups']}--{s['max_groups']} | {s['median_seconds']:.2f} s |" for s in summary]
    lines += ['', f"Overall median {result['median_seconds']:.2f} s and mean {result['mean_seconds']:.2f} s per placement, using four concurrent CPU workers. The range was {min(timings):.2f}--{max(timings):.2f} s. Input pose fitting and Python imports are excluded; these are not serial timings comparable with the old single-scene sampler.", '',
      '## What passing means', '',
      'The approach is downward. The open hand and its modeled downward movement clear the native target mesh. Proposed contacts fit the finite inner pads. Both pads reach inner-face contact at compatible openings; closing stops at the first contact within the declared numerical bracket. Finger bodies/palm avoid the part, and the hand stays above the observed table plane. All distinct proposals are checked before grouping.', '',
      f"The smallest retained table clearance is {1000*result['minimum_retained_table_gap_m']:.3f} mm (required 0.100 mm). The largest retained first-contact opening mismatch is {1000*result['maximum_contact_opening_mismatch_m']:.4f} mm (allowed 0.050 mm). These are model calculations, not measured physical tolerances.", '',
      'The actual Panda pads have a maximum 79 mm gap at the 80 mm command. A full 12 mm extra approach opening limits proposed contact separation to 67 mm. Historical +6/+8 mm depths are tested but rejected because their proposed contacts lie beyond the inner pads. No successful depth is silently relabeled as pad contact.', '',
      '## Rejections', '',
      'Counts below record the first failed condition per candidate; later conditions need not have been evaluated.', '',
      '| Primary reason | Candidates |', '|---|---:|']
    lines += [f"| {reason.replace('_',' ')} | {count:,} |" for reason,count in rejections.most_common()]
    if result['no_grasp_cases']:
        lines += ['', 'No passing grasp was found for: '+', '.join(r['scene']+'/'+r['part'] for r in result['no_grasp_cases'])+'. This is a finite search failure, not proof that the part cannot be grasped.']
    lines += ['', '## Limits and evidence', '',
      'Geometric feasibility does not establish pad support area, friction, resistance to gravity, slipping, pose uncertainty, pickup success or assembly precision. Other parts, arm reach and robot self-contact are outside this trial. Native target holes remain; the large gear has an explicitly recorded 50 nm numerical seam weld, with source CAD unchanged. Closed CAD components use union containment; internal component surfaces can conservatively reject a placement.', '',
      'A concrete task-quality limitation is N00 / 8 mm round pin: its two retained poses both hold the two ends, roughly 50 mm apart; they exchange the jaws rather than offering a different contact region. This is geometrically feasible, but the VLM would not have an alternative side-contact region to choose in that placement. Do not interpret 96/96 candidate availability as 96/96 assembly-suitable grasps.', '',
      '[Interactive grasp gallery](gallery/index.html) · [Per-part CSV](per_part.csv) · [Settings and provenance](manifest.json) · [Independent audit](audit.json).', '']
    if manifest.get('base_run'):
        label = 'CAD surface-coverage' if manifest.get('surface_config') else 'height-coverage'
        lines += [f'This result includes a separately declared exploratory {label} follow-up. Initial zero-grasp cases were revisited with the declared sampling variant; passing initial cases were reused. Read the follow-up manifest and base report for original counts and additional computation. It is not a fresh held-out validation.', '']
    (run/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('cases','placements_with_grasps','generated','valid','grouped','median_seconds','mean_seconds')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
