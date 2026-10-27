"""Summarize the frozen comparison without changing its proposals or decisions."""
from collections import Counter
from pathlib import Path
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'outputs/cad_surface_grasps_20261026/runs/20261026T200202Z/results.json'


def median(values):
    return float(statistics.median(values)) if values else None


def summarize(folder):
    folder=Path(folder)
    records=json.loads((folder/'results.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    reference=json.loads(REFERENCE.read_text())
    missing={(r['scene'],r['part']) for r in reference if not r['valid_count']}
    methods={}
    case_rows=[]
    for method in manifest['methods']:
        rows=[r for r in records if r['method']==method]
        errors=[r for r in rows if r['status']=='experiment_error']
        good=[r for r in rows if r['status']!='experiment_error']
        protocols={}
        for protocol in manifest['protocols']:
            summaries=[r['protocol_summary'][protocol] for r in good]
            protocols[protocol]=dict(
                placements_with_candidates=sum(s['available'] for s in summaries),
                checked_poses=sum(s['checked'] for s in summaries),
                passing_poses=sum(s['valid'] for s in summaries),
                recovered_original_missing=[f"{r['scene']}/{r['part']}" for r in good
                    if (r['scene'],r['part']) in missing and r['protocol_summary'][protocol]['available']],
                missing=[f"{r['scene']}/{r['part']}" for r in good if not r['protocol_summary'][protocol]['available']],
                rejection_counts=dict(sum((Counter(s['rejection_counts']) for s in summaries),Counter())))
        unique={r['part']:r['intrinsic_record'] for r in good}
        methods[method]=dict(placements=len(rows),errors=errors,protocols=protocols,
            cached_native_pairs_per_part={p:r['raw_pairs'] for p,r in unique.items()},
            intrinsic_generation_once_per_cad_s={p:r['sourcegen_seconds'] for p,r in unique.items()},
            intrinsic_generation_sum_s=sum(r['sourcegen_seconds'] for r in unique.values()),
            runtime_both_adapters_sum_s=sum(r['timing_seconds'] for r in good),
            runtime_both_adapters_median_s=median([r['timing_seconds'] for r in good]),
            runtime_preparation_median_s=median([r['runtime_sampling_seconds'] for r in good]),
            panda_validation_median_s=median([r['adapter_validation_seconds'].get('panda_table',0.) for r in good]),
            primary_runtime_component_median_s=median([r['runtime_sampling_seconds']+
                r['adapter_validation_seconds'].get('panda_table',0.) for r in good]),
            sparse_pair_range=[min(r['sparse_pairs'] for r in good),max(r['sparse_pairs'] for r in good)] if good else [],
            primary_family_range=[min(r['grouped_count'] for r in good),max(r['grouped_count'] for r in good)] if good else [],
            primary_family_total=sum(r['grouped_count'] for r in good))
        for r in good:
            case_rows.append(dict(method=method,scene=r['scene'],part=r['part'],native_pairs=r['native_pairs'],
                sparse_pairs=r['sparse_pairs'],families=r['grouped_count'],passing_poses=r['valid_count'],
                primary_runtime_component_s=r['runtime_sampling_seconds']+r['adapter_validation_seconds'].get('panda_table',0.),
                protocol_available={k:v['available'] for k,v in r['protocol_summary'].items()},
                rejection_counts=r['rejection_counts']))
    result=dict(methods=methods,cases=case_rows,original_missing=[f'{s}/{p}' for s,p in sorted(missing)],
        timing_caution='CPU times under four concurrent workers. Primary component adds shared preparation and Panda-adapter checking; excludes other control checks but is instrumented inside a two-control run, not an isolated deployment timing. CAD-intrinsic sampling once per part is reported separately. No pose estimation, VLM, physics or pickup time.')
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Original contact samplers with common Panda adapters','',
        'This executes pinned upstream contact-generation functions, not their complete robot frameworks. '
        'All 96 saved CAD placements and the target/table/hand checker are held fixed. '
        'The shared-region, fitted-cylinder and 1-degree flat-face hard filters are absent. '
        'No learned ranking, pickup, assembly or physical holding test is run.','',
        '| Sampler | Conversion | Poses checked | Poses passing | Placements with candidates | Original nine recovered |',
        '|---|---|---:|---:|---:|---:|']
    for method,m in methods.items():
        for protocol,s in m['protocols'].items():
            lines.append(f"| {method} | {protocol} | {s['checked_poses']} | {s['passing_poses']} | {s['placements_with_candidates']}/96 | {len(s['recovered_original_missing'])}/9 |")
    lines+=['','`midpoint` places the finite finger-pad center at the sampled pair midpoint. '
        '`panda_table` also raises the hand to clear the table and centers it using modeled first finger contacts. '
        '`proj1` permits at most 1 degree adjustment of the native pair to horizontal; `proj15` permits 15 degrees. '
        'Every resulting hand approaches straight down and is independently rechecked against native target triangles and the table.','',
        '| Sampler | Cached CAD generation, sum for 12 parts (s) | Median runtime preparation + Panda checking (s) | Retained primary families per placement |',
        '|---|---:|---:|---:|']
    for method,m in methods.items():
        lines.append(f"| {method} | {m['intrinsic_generation_sum_s']:.3f} | {m['primary_runtime_component_median_s']:.3f} | {m['primary_family_range']} |")
    lines+=['',result['timing_caution'],'','## Original nine missing placements','',
        '| Placement | one default | one budget 512 | GraspKit 512 |', '|---|---:|---:|---:|']
    lookup={(r['method'],f"{r['scene']}/{r['part']}"):r for r in case_rows}
    for case in result['original_missing']:
        vals=[str(lookup[(m,case)]['families']) if (m,case) in lookup else 'unavailable' for m in ('one_default','one_budget512','graspkit_512')]
        lines.append('| '+case+' | '+' | '.join(vals)+' |')
    lines+=['','Counts in the last table are distinct pair families under `panda_table_proj15`, '
        'with both passing jaw assignments preserved in the detailed records. '
        'Availability does not establish that the grasp leaves the insertion end free.','',
        '## Per-part primary coverage','', '| Part | one default | one budget 512 | GraspKit 512 |', '|---|---:|---:|---:|']
    for part in sorted({r['part'] for r in case_rows}):
        vals=[]
        for method in ('one_default','one_budget512','graspkit_512'):
            selected=[r for r in case_rows if r['part']==part and r['method']==method]
            vals.append(f"{sum(r['passing_poses']>0 for r in selected)}/{len(selected)}")
        lines.append('| '+part+' | '+' | '.join(vals)+' |')
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({m:{p:s['placements_with_candidates'] for p,s in v['protocols'].items()} for m,v in methods.items()},indent=2))


if __name__=='__main__':
    summarize(sys.argv[1])
