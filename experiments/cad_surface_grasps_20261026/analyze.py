"""Summarize surface-derived candidates without choosing a VLM winner."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median,mean
import sys

ROOT=Path(__file__).resolve().parents[2]


def counts(rows):
    return dict(placements=len(rows),with_options=sum(bool(r['grasps']) for r in rows),
        checked=sum(r['generated_count'] for r in rows),valid=sum(r['valid_count'] for r in rows),
        families=sum(r['grouped_count'] for r in rows),min_families=min(r['grouped_count'] for r in rows),
        max_families=max(r['grouped_count'] for r in rows),
        median_seconds=median(r['timing_seconds'] for r in rows),mean_seconds=mean(r['timing_seconds'] for r in rows))


def main(run):
    run=Path(run)
    rows=json.loads((run/'results.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    previous=json.loads((ROOT/manifest['baseline_run']/'results.json').read_text())
    if len(rows)!=96 or {(r['scene'],r['part']) for r in rows}!={(r['scene'],r['part']) for r in previous}:
        raise ValueError('Full comparison needs the same 96 placements')
    if any(r['status']=='experiment_error' for r in rows):
        raise ValueError('Resolve experiment errors before reporting')
    summary=dict(previous=counts(previous),surface=counts(rows),by_part=[],unavailable=[],
                 primary_rejections=dict(sum((Counter(r['rejection_counts']) for r in rows),Counter())))
    summary['center_projection_geometry_queries']=sum(r.get('center_projection_geometry_queries',0) for r in rows)
    for part in sorted({r['part'] for r in rows}):
        old=[r for r in previous if r['part']==part]
        new=[r for r in rows if r['part']==part]
        a,b=counts(old),counts(new)
        summary['by_part'].append(dict(part=part,previous_available=a['with_options'],surface_available=b['with_options'],
            previous_min=a['min_families'],previous_max=a['max_families'],surface_min=b['min_families'],surface_max=b['max_families'],
            curved_side_placements=sum(any(g.get('contact_geometry')=='fitted_round_side' for g in r['grasps']) for r in new),
            old_median_seconds=a['median_seconds'],new_median_seconds=b['median_seconds']))
    for r in rows:
        if not r['grasps']:
            summary['unavailable'].append(dict(scene=r['scene'],part=r['part'],preparation=r['preparation_rejection_counts'],validation=r['rejection_counts']))
    aliases=[a for r in rows for g in r['grasps'] for a in g['jaw_aliases']]
    flat=[a for a in aliases if a['contact_geometry']=='opposing_flat_surfaces']
    curved=[a for a in aliases if a['contact_geometry']=='fitted_round_side']
    summary.update(max_flat_normal_error_deg=max((max(a['actual_pad_normal_errors_deg']) for a in flat),default=None),
                   max_curved_native_normal_error_deg=max((max(a['actual_pad_normal_errors_deg']) for a in curved),default=None),
                   max_curved_closing_shaft_error_deg=max((a['cylinder_closing_perpendicular_error_deg'] for a in curved),default=None),
                   flat_aliases=len(flat),curved_aliases=len(curved),
                   timing_note='Four CPU workers; per-placement loading, geometry and candidate-check writing. Omits startup, input CAD pose fitting, VLM, rendering and some final writing. Coverage/checks differ; not a controlled equal-output timing benchmark.')
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (run/'per_part.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary['by_part'][0]))
        writer.writeheader()
        writer.writerows(summary['by_part'])
    a,b=summary['previous'],summary['surface']
    lines=['# Surface-derived CAD grasp revision', '',
        f"**{b['with_options']}/96 saved placements have options, with {b['families']} displayed families and {b['valid']} checked alternatives.**", '',
        'N00--N07 contain twelve parts each. Correct CAD/crop identity is supplied; positions/orientations and the table plane come from the same saved observed-depth inputs. Simulator object poses are not read. No robot move, pickup, assembly or VLM selection is executed.', '',
        '| Measure | Native CAD-axis revision | Surface-derived revision |', '|---|---:|---:|',
        f"| Placements with options | {a['with_options']}/96 | {b['with_options']}/96 |",
        f"| Full hand poses checked | {a['checked']} | {b['checked']} |",
        f"| Passing checked alternatives | {a['valid']} | {b['valid']} |",
        f"| Displayed contact families | {a['families']} | {b['families']} |",
        f"| Families per placement | {a['min_families']}--{a['max_families']} | {b['min_families']}--{b['max_families']} |",
        f"| Median recorded generation/check time | {a['median_seconds']:.3f} s | {b['median_seconds']:.3f} s |", '',
        summary['timing_note'], '',
        f"The final hand checks are additional to {summary['center_projection_geometry_queries']} finite-pad first-contact queries used to center the hand geometrically. "
        'These are geometry calculations, not pickup attempts or a new position/angle sweep.', '',
        '| Part | Old availability | New availability | New families per scene | Placements with a verified round-side family |',
        '|---|---:|---:|---:|---:|']
    lines += [f"| {s['part']} | {s['previous_available']}/8 | {s['surface_available']}/8 | {s['surface_min']}--{s['surface_max']} | {s['curved_side_placements']}/8 |" for s in summary['by_part']]
    lines += ['', 'The last column counts local cylinder-side contact, not a claim that the whole connector or gear is rotationally symmetric. Other adjoining features remain in the full target interference check.', '',
        '## Surface alignment and drawing correction', '',
        f"Flat contacts pass an added 1-degree normal check; the largest retained flat-face mismatch is {summary['max_flat_normal_error_deg']:.6f} degrees. "
        'The pad planes are parallel to flat object faces; their normals align with finger closing. Round contacts use a fitted shaft and horizontal transverse closing. '
        'Native triangles approximate the smooth round surface: the raw triangle-normal error, fitted radial error, and mesh-derived angular allowance are saved separately. '
        'A curved contact is not mislabeled as a flat face with zero error.', '',
        'Both finite pads must have a common opposing near-contact region within the unchanged 25-micrometer gap band. '
        'The green pair now uses native surface locations with matching hand Y/Z, chosen nearest the centered placement; it is not a line between arbitrary independent triangle witnesses. '
        'Translucent regions show geometric proximity within the band, not measured loaded area, rubber deformation or pressure. The original first-touch witnesses remain reviewable.', '',
        'Before checking both jaw assignments, the hand is centered once using the actual modeled pad first contacts. '
        'For Panda finger motion of plus/minus half the opening, shift along closing by (right contact opening minus left contact opening)/4, then fully recheck. '
        'This is an analytic contact-midpoint correction, not an assembly-error score. Round-band contact polygons and paired native points are restricted to the same verified fitted band.', '',
        'The earlier scene-2 KET16 drawing had a 44.853-degree witness line while the actual hand-to-face error was only 0.001561 degrees. '
        'That drawing did not establish an oblique hand. Neither the old drawing nor the corrected normal check establishes zero shift or reliable assembly.', '',
        '## Why the 8 mm side can still be missing', '',
        'For the lying N00 8 mm pin, the stock guarded pad cannot contact below 5.279086 mm while clearing the observed table. '
        'The sampled, near-horizontal side region ends at 5.057108 mm: a 0.221978 mm shortfall. '
        'N02 has a 0.222096 mm shortfall. Removing arbitrary CAD roll fixes direction generation but not this separate pad/table restriction. '
        'The retained end grip in such a case must not be described as a round-side alternative. '
        '[Full numerical diagnosis](../../compact_diagnostics.json).', '',
        '## Missing cases and rejected proposals', '']
    if summary['unavailable']:
        for row in summary['unavailable']:
            lines.append(f"- {row['scene']} / {row['part']}: preparation {row['preparation']}; full checks {row['validation']}.")
    else:
        lines.append('Every placement has at least one passing option. This is candidate availability, not pickup success.')
    lines += ['', '| First failed full check | Count |', '|---|---:|']
    lines += [f"| {k.replace('_',' ')} | {v} |" for k,v in Counter(summary['primary_rejections']).most_common()]
    lines += ['', '## Scope and references', '',
        'The method adapts the actual-surface-normal/opposing-ray principles reviewed in the pinned open-source one planner through existing local helpers. '
        'It does not install or execute that framework. Cylinder fitting, compact connected regions and stricter finite-pad checks are local additions. '
        'There is no automatic assembly-error score, arbitrary top-twelve cap, or first-success stop. All proposed hand alternatives are checked before grouping; VLM final selection remains later.', '',
        '[Review all families and alternatives](gallery/index.html) · [Independent audit](audit.json) · '
        '[Viewer audit](gallery/viewer_verification.json) · [Per-part CSV](per_part.csv) · '
        '[Source and threshold manifest](manifest.json) · [Preservation](preservation.json).', '']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('previous','surface','max_flat_normal_error_deg','flat_aliases','curved_aliases','unavailable')},indent=2))


if __name__=='__main__':
    main(sys.argv[1])
