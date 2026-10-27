"""Compare saved candidate sets; do not change or select grasps."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[2]


def totals(rows, previous=False):
    times = [r['timing_seconds'] for r in rows]
    return dict(placements=len(rows), with_grasps=sum(bool(r['grasps']) for r in rows),
                full_hand_proposals=sum(r.get('total_proposals_checked_including_base', r['generated_count'])
                                        if previous else r['generated_count'] for r in rows),
                valid_hand_proposals=sum(r['valid_count'] for r in rows),
                displayed_groups=sum(r['grouped_count'] for r in rows),
                min_groups=min(r['grouped_count'] for r in rows),
                max_groups=max(r['grouped_count'] for r in rows),
                median_seconds=median(times), mean_seconds=mean(times),
                min_seconds=min(times), max_seconds=max(times),
                summed_placement_seconds=sum(times))


def main(run):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    baseline = ROOT/manifest['baseline_run']
    rows = json.loads((run/'results.json').read_text())
    old = json.loads((baseline/'results.json').read_text())
    if len(rows) != 96 or {(r['scene'], r['part']) for r in rows} != {(r['scene'], r['part']) for r in old}:
        raise ValueError('Comparison requires the same 96 placements')
    if any(r['status']=='experiment_error' for r in rows):
        raise ValueError('Resolve experiment errors before reporting')
    before, after = totals(old, True), totals(rows)
    summary = dict(baseline_run=str(baseline.relative_to(ROOT)), previous=before, compact=after)
    summary['horizontal_ray_lines'] = sum(r['horizontal_grid_lines'] for r in rows)
    summary['raw_contact_regions'] = sum(r['raw_contact_regions'] for r in rows)
    summary['merged_contact_regions'] = sum(r['merged_contact_regions'] for r in rows)
    summary['prepared_contact_families'] = sum(r['prepared_contact_families'] for r in rows)
    summary['preparation_rejections'] = dict(sum((Counter(r['preparation_rejection_counts']) for r in rows), Counter()))
    summary['candidate_rejections'] = dict(sum((Counter(r['rejection_counts']) for r in rows), Counter()))
    summary['unavailable'] = [dict(scene=r['scene'], part=r['part'], reasons=r['preparation_rejection_counts'])
                              for r in rows if not r['grasps']]
    by_part = []
    for part in sorted({r['part'] for r in rows}):
        a = totals([r for r in old if r['part']==part], True)
        b = totals([r for r in rows if r['part']==part])
        by_part.append(dict(part=part, previous_with_grasps=a['with_grasps'], compact_with_grasps=b['with_grasps'],
                            previous_min_groups=a['min_groups'], previous_max_groups=a['max_groups'],
                            compact_min_families=b['min_groups'], compact_max_families=b['max_groups'],
                            previous_median_seconds=a['median_seconds'], compact_median_seconds=b['median_seconds']))
    summary['by_part'] = by_part
    summary['timing_note'] = ('Both use four concurrent CPU workers and the same geometry checker/input poses. '
        'Recorded times include CAD loading, sampling, preparation, geometric checks and candidate-check JSON '
        'writing; omit process/import startup, pose estimation, gallery/VLM, and some final result/report writing. '
        'Sums are not elapsed wall time. Eight previous USB times accumulate the earlier search and exploratory '
        'follow-up. The compact method also has lower coverage; this is not a controlled '
        'same-output speed benchmark.')
    (run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    with (run/'per_part.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_part[0]))
        writer.writeheader()
        writer.writerows(by_part)
    lines = ['# Compact CAD contact-region trial', '',
        f"**{after['with_grasps']} of the same 96 saved placements have a modeled feasible grasp. No pickup, assembly or VLM call was run.**", '',
        'The inputs are the existing N00--N07 scenes, twelve parts each, with saved observed-depth CAD fits. '
        'Correct CAD/crop association is supplied. Simulator object positions/orientations are not used. '
        'The table plane is fitted from saved depth. Other parts, arm reach and robot self-contact are outside this trial.', '',
        '| Measure | Previous surface-coverage run | Compact contact-region run |',
        '|---|---:|---:|',
        f"| Full hand proposals checked | {before['full_hand_proposals']:,} | {after['full_hand_proposals']:,} |",
        f"| Passing hand proposals, before grouping | {before['valid_hand_proposals']:,} | {after['valid_hand_proposals']:,} |",
        f"| Displayed groups/families | {before['displayed_groups']:,} | {after['displayed_groups']:,} |",
        f"| Placements with candidates | {before['with_grasps']}/96 | {after['with_grasps']}/96 |",
        f"| Median generation/check time per placement | {before['median_seconds']:.3f} s | {after['median_seconds']:.3f} s |",
        f"| Mean time per placement | {before['mean_seconds']:.3f} s | {after['mean_seconds']:.3f} s |", '',
        summary['timing_note'], '',
        'The old display counted separate hand placements and depth/contact labels. The new display combines '
        'the checked 180-degree jaw alternatives under a contact family and merges physically identical final grips. '
        'Every passing alternative remains selectable in the gallery; the totals count different things intentionally.', '',
        '| Part | Previous groups per scene | Compact families per scene | Compact placements with candidates | Previous median | Compact median |',
        '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['part']} | {r['previous_min_groups']}--{r['previous_max_groups']} | "
              f"{r['compact_min_families']}--{r['compact_max_families']} | {r['compact_with_grasps']}/8 | "
              f"{r['previous_median_seconds']:.3f} s | {r['compact_median_seconds']:.3f} s |" for r in by_part]
    lines += ['', '## What was reduced', '',
        f"The generator casts {summary['horizontal_ray_lines']:,} cheap horizontal CAD ray lines across all 96 placements. "
        f"It groups their qualified intersections into {summary['raw_contact_regions']:,} connected regions, then "
        f"{summary['merged_contact_regions']:,} pad-size contact regions. Only {summary['prepared_contact_families']:,} "
        f"regions produce a calculated hand placement for full checking. Thus “{after['full_hand_proposals']} hand proposals” does not mean "
        f"only {after['full_hand_proposals']} geometric queries. Failed height projections are recorded as preparation failures, before a full pose is proposed.", '',
        'Closing uses a projected native CAD X/Y/Z axis only when that axis is within 1 degree of horizontal. '
        'The approach stays exactly down. One centered height is calculated per contact region and adjusted for '
        'the target and table; there is no six-depth expansion, arbitrary yaw sweep, top-12 cap or first-success stop. '
        'Both jaw alternatives are checked because the complete hand is not perfectly symmetric.', '',
        '## Eight placements without a candidate', '',
        '| Part | Scenes | Recorded reason |', '|---|---|---|',
        '| Large gear | N01, N03, N05, N07 | No clear open-hand approach was found within the selected region’s contact-compatible height range. |',
        '| BNC connector | N01, N03, N04, N06 | No clear open-hand approach was found within the selected region’s contact-compatible height range. |', '',
        'All eight stop during preparation; they are not attempted pickups. Detailed N01 examples: for the gear, both finger solids and pads '
        'intersect the target during the modeled approach even at the highest contact-compatible height. For N01 BNC, the first sampled width is 19.873 mm. '
        'After raising the hand for the table, recasting the contact ray gives 8.471 mm; the corresponding smaller '
        'approach opening cannot clear the surrounding connector. This is a limitation of the axis and centered-region '
        'policy together, not proof that every axis-aligned grasp is impossible. The closest saved passing old grasps '
        'are 1.36--1.37 degrees away from the native axes for these gear cases, and 28.52--42.88 degrees for these BNC cases. '
        'Those saved old grasps do not follow the projected CAD-axis directions retained by this compact generator. '
        'The 1-degree threshold decides whether a native axis is sufficiently horizontal to use; the generated '
        'closing direction then matches its horizontal projection.', '',
        '## What passing establishes', '',
        'The unchanged geometric checker checks the actual finite inner finger pads, open approach, first contact '
        'while closing, finger/palm interference and observed table clearance. It preserves native holes and the earlier '
        'declared gear seam treatment. A passing result does not establish holding strength, resistance to sliding, '
        'pickup success, or assembly precision. End grasps remain possible for round pins and may be unsuitable '
        'for insertion; CAD-axis alignment alone does not solve that task decision. The VLM has not yet selected a grasp.', '',
        '[Review every contact family and checked jaw alternative](gallery/index.html). '
        '[Independent audit](audit.json), [viewer checks](gallery/viewer_verification.json), '
        '[per-part CSV](per_part.csv), [settings and source hashes](manifest.json), '
        '[frozen-evidence preservation](preservation.json).', '']
    (run/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
