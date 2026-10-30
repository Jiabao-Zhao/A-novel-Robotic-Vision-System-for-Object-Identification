"""Post-launch descriptive reporting; does not generate, fit or select grasps."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path


STAGES = ('contact_valid', 'pickup_valid', 'final_goal_valid', 'joint_valid')
STRATEGIES = ('random', 'task_informed')
MODES = ('pad', 'finger', 'combined')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def selected(row):
    return row['selection']['status'] == 'selected'


def pair(left, right):
    """Descriptive pairing on exactly the same seed, scene and part."""
    key = lambda r: (r['seed_index'], r['scene'], r['part'])
    if len({key(r) for r in left}) != len(left) or len({key(r) for r in right}) != len(right):
        raise ValueError('Duplicate placement in a paired comparison')
    left, right = {key(r): r for r in left}, {key(r): r for r in right}
    if left.keys() != right.keys():
        raise ValueError('Paired comparison has mismatched placements')
    counts = Counter((selected(left[k]), selected(right[k])) for k in left)
    return dict(placements=len(left), both=counts[True, True],
                left_only=counts[True, False], right_only=counts[False, True],
                neither=counts[False, False],
                net_change=counts[False, True]-counts[True, False])


def make_report(run):
    run = Path(run).resolve()
    rows, summary = read(run/'results.json'), read(run/'summary.json')
    manifest, angles = read(run/'launch_manifest.json'), read(run/'frozen_angle_sensitivity.json')
    if len(rows) != 1728:
        raise ValueError('This report expects the preregistered full three-seed NIST matrix')
    if any(r['selection']['proposed_count'] > 100 for r in rows):
        raise ValueError('Candidate cap exceeded')
    grouped = {(s, m): [r for r in rows if (r['sampler_strategy'], r['contact_mode']) == (s, m)]
               for s in STRATEGIES for m in MODES}
    comparisons = {}
    for mode in MODES:
        comparisons['task_informed_vs_random_'+mode] = pair(grouped['random', mode], grouped['task_informed', mode])
    for strategy in STRATEGIES:
        for mode in ('finger', 'combined'):
            comparisons[strategy+'_'+mode+'_vs_pad'] = pair(grouped[strategy, 'pad'], grouped[strategy, mode])
    per_part = []
    for part in sorted({r['part'] for r in rows}):
        for (strategy, mode), subset in grouped.items():
            subset = [r for r in subset if r['part'] == part]
            per_part.append(dict(part=part, strategy=strategy, contact_mode=mode, placements=len(subset),
                selected=sum(map(selected, subset)),
                per_seed=[sum(selected(r) for r in subset if r['seed_index'] == seed) for seed in range(3)],
                **{stage: sum(r['selection']['stage_counts'][stage].get('pass', 0) > 0 for r in subset)
                   for stage in STAGES}))
    angle_summary = []
    for angle in manifest['sensitivity_angles_deg']:
        subset = [r for r in angles if r['angle_deg'] == angle]
        if len(subset) != 96:
            raise ValueError('Sensitivity is incomplete')
        angle_summary.append(dict(angle_deg=angle, placements=len(subset),
            selected=sum(r['selected'] for r in subset), valid_candidates=sum(r['valid'] for r in subset),
            evaluated_candidates=sum(r['proposed'] for r in subset),
            rescued_candidates_with_other_checks_already_passed=sum(len(r['rescued_with_all_other_checks_already_passed']) for r in subset),
            rescued_candidates_with_other_checks_newly_available=sum(len(r['rescued_with_other_checks_previously_unavailable_or_not_evaluated']) for r in subset),
            rescued_placements_with_other_checks_already_passed=sum(bool(r['rescued_with_all_other_checks_already_passed']) for r in subset),
            rescued_placements_with_other_checks_newly_available=sum(bool(r['rescued_with_other_checks_previously_unavailable_or_not_evaluated']) for r in subset)))
    by_placement = {}
    for row in angles:
        by_placement.setdefault((row['scene'], row['part']), []).append(row)
    if any(len({r['frozen_pose_sha256'] for r in v}) != 1 or len({r['calibration_sha256'] for r in v}) != 1
           for v in by_placement.values()):
        raise ValueError('Sensitivity altered poses or calibration')
    primary = grouped['task_informed', 'combined']
    primary = [r for r in primary if r['seed_index'] == 0]
    selected_modes = Counter(r['selection']['selected_grasp']['contact_mode'] for r in primary if selected(r))
    # Hashes describe actual raw proposals; repeated placements are not new CAD samples.
    banks = list(run.glob('banks/*/seed_*/*/*/sampling.json.gz'))
    unique_banks = {strategy: set() for strategy in STRATEGIES}
    for path in banks:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            sampling = json.load(stream)
        unique_banks[path.relative_to(run).parts[1]].add(sampling['shared_raw_bank_sha256'])
    result = dict(groups=summary['groups'], paired_comparisons=comparisons, per_part=per_part,
        sensitivity=angle_summary, primary_selected_contact_modes=dict(selected_modes),
        unique_raw_banks_by_strategy={s: len(v) for s, v in unique_banks.items()},
        timing_scope=summary['timing_scope'], shared_benchmark_wall_seconds=summary['elapsed_wall_seconds'],
        per_policy_wall_seconds=None, physical_success_measured=False,
        source_results_sha256=hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
        postprocessing_only=True)
    (run/'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (run/'per_part_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['part', 'strategy', 'contact_mode', 'placements', 'selected', *STAGES, 'seed_0', 'seed_1', 'seed_2']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in per_part:
            writer.writerow({k: v for k, v in row.items() if k != 'per_seed'} |
                            {f'seed_{i}': n for i, n in enumerate(row['per_seed'])})
    lines = ['# Controlled NIST grasp geometry trial', '',
        'Eight saved scenes, 12 products, three paired seeds, using the saved Panda hand geometry. Correct CAD association and saved CAD alignment are assumed. '
        'These counts measure availability of a geometrically valid modeled grasp, not pickup or assembly success.', '',
        '## Candidate budget and comparison', '',
        'Each policy receives at most 100 proposals. Combined mode declares 50 pad and 50 finger contacts before validation when 100 poses are retained; '
        'shorter sets keep the two allocations balanced within one. Both samplers use 200 surface-origin draws and at most ten orientations per pair. '
        'The task-informed table prefilter can leave fewer than 100 eligible poses; no additional surface draws refill that set. '
        'The benchmark checks pad and finger alternatives on the shared raw bank (up to 200 underlying candidates); combined replays its declared subset. '
        'It does not select from a union of two 100-candidate searches.', '',
        '| Sampler | Contact policy | Seed 0 /96 | Seed 1 /96 | Seed 2 /96 | Total /288 | Valid / evaluated candidates |',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        counts = {r['seed_index']: r['selected'] for r in g['per_seed']}
        lines.append(f"| {g['strategy']} | {g['contact_mode']} | {counts[0]} | {counts[1]} | {counts[2]} | {g['selected']} | {g['valid']} / {g['proposed']} |")
    lines += ['', 'The 288 evaluations per arm repeat the same 96 placements under three seeds. '
        'They are not 288 independent objects or scenes. Descriptive paired changes follow; three seeds do not establish a universal best sampler. '
        'The task-informed arm changes contact-position allocation, explicit orientations and the safe table prefilter together; '
        'this comparison cannot assign a gain to one of those components alone. The 50/50 contact allocation is fixed, not an optimized mixture.', '',
        '| Task-informed versus random | Both valid | Random only | Task-informed only | Neither | Net change |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        p = comparisons['task_informed_vs_random_'+mode]
        lines.append(f"| {mode} | {p['both']} | {p['left_only']} | {p['right_only']} | {p['neither']} | {p['net_change']:+d} |")
    lines += ['', '## Separate geometry stages', '',
        'Contact-valid requires permitted contact, opening limits, normal alignment and complete-hand/part clearance. '
        'Pickup-valid adds table clearance to contact-valid. Final-goal-valid separately adds final receiver clearance and protected mating surfaces to contact-valid, without the pickup-table requirement. '
        'Joint-valid requires all of these for one unchanged grasp. Counts below are placements with at least one candidate passing each stage; '
        'a different candidate may supply each stage. Unavailable or unevaluated checks are never passes.', '',
        '| Sampler | Policy | Contact /288 | Pickup /288 | Final goal /288 | Joint /288 |',
        '|---|---|---:|---:|---:|---:|']
    for g in summary['groups']:
        p = g['stage_placements']
        lines.append('| '+g['strategy']+' | '+g['contact_mode']+' | '+' | '.join(str(p[s]) for s in STAGES)+' |')
    lines += ['', '## Per-part joint availability', '',
        'Each cell is seed 0 / seed 1 / seed 2, each out of eight placements.', '',
        '| Part | Random pad | Random finger | Random combined | Task pad | Task finger | Task combined |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for part in sorted({r['part'] for r in rows}):
        values = []
        for strategy in STRATEGIES:
            for mode in MODES:
                r = next(r for r in per_part if (r['part'], r['strategy'], r['contact_mode']) == (part, strategy, mode))
                values.append(' / '.join(map(str, r['per_seed'])))
        lines.append('| '+part.replace('_', ' ')+' | '+' | '.join(values)+' |')
    lines += ['', '## Frozen-pose contact-angle sensitivity', '',
        'Only the final normal-acceptance tolerance changes. Hand surface labels and calibration, random seed-0 proposals, fitted poses and all other tolerances remain fixed. '
        'This does not test changing the proposal-angle tolerance, modifying the hand, frictional holding or physical compliance.', '',
        '| Acceptance angle | Placements /96 | Valid / evaluated candidates | Newly valid candidates: other checks already passed | Newly valid candidates: some other checks previously unevaluated |',
        '|---|---:|---:|---:|---:|']
    for a in angle_summary:
        lines.append(f"| {a['angle_deg']:g}° | {a['selected']} | {a['valid_candidates']} / {a['evaluated_candidates']} | {a['rescued_candidates_with_other_checks_already_passed']} | {a['rescued_candidates_with_other_checks_newly_available']} |")
    lines += ['', 'Newly valid counts are relative to the fixed 1° baseline. The final column includes checks that become evaluable only after a valid contact witness exists; '
        'it must not be described as candidates known to pass every nonangular check at 1°.', '',
        '## Gallery, contact surfaces and time', '',
        '[Primary gallery](primary/gallery/index.html) uses task-informed combined mode, seed 0 fixed before results, not the best seed. '
        f"Selected contact modes: {dict(selected_modes)}. It shows saved selected contacts on CAD; failed placements have no replacement grasp.", '',
        'Pad and finger-end contacts are explicit alternatives in the existing simulation hand model. '
        'The convex finger collision shape bridges a groove in the detailed display mesh. Neither is a verified physical hand surface. '
        'The source-surface audit reports that discrepancy separately; it does not silently turn display geometry into hardware truth. '
        'UR5e/RG2 geometry and hardware contact calibration are not supplied by this trial.', '',
        f"Shared benchmark wall time: {summary['elapsed_wall_seconds']:.3f} s with {manifest['workers']} worker processes, "
        'including all policies/seeds and the separately recorded frozen-angle sweep; excluding rendering/publication. '
        'The benchmark caches CAD-local and fixed-final-goal checks, recomputes complete pickup table checks and approach preference in every scene, '
        'and shares work between policies. Independent policy runtimes were not measured, so this is not a policy speed comparison.', '',
        f"Underlying mode-candidate evaluations: {summary['underlying_mode_candidate_evaluations']:,}; "
        f"logical evaluations across the three policies: {summary['logical_policy_candidate_evaluations']:,}. "
        f"Distinct retained raw-bank hashes: {result['unique_raw_banks_by_strategy']}. Raw proposals reused across scenes are not new independent draws.", '',
        '## Scope and evidence', '',
        'The geometric priority selector is unchanged: minimum normalized assembly-axis moment arm, then surface-centroid distance, '
        'then downward approach angle, then candidate ID, with explicit numerical ties. '
        'This is a geometric proxy; insertion force, actual torque, frictional holding, compliance and execution accuracy are not measured. '
        'Recognition policies were not changed. VLM selection, robot motion, pickup, assembly, insertion-path and release checks were not run.', '',
        '[Launch manifest](launch_manifest.json), [run verification](run_verification.json), '
        '[full results](results.json), [summary](summary.json), [paired comparisons](comparison.json), '
        '[per-part CSV](per_part_comparison.csv), [frozen-angle records](frozen_angle_sensitivity.json).', '',
        'This report is postprocessing added after launch. It reads saved outcomes without modifying numerical results or selecting another seed.']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    args = parser.parse_args(); make_report(args.run)
