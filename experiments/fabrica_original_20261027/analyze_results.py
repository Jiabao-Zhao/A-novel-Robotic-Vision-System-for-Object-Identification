"""Describe saved native Fabrica candidates; never generate or execute grasps."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIDE_MAX_AXIAL_COMPONENT = 0.15
END_MIN_AXIAL_COMPONENT = 0.985
PAIR_TYPES = ('side_spanning', 'end_to_end', 'mixed')
PREVIOUS_RUN = ROOT / 'outputs/fabrica_candidates_20261027/runs/20261027T185418Z'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, sources):
    path = Path(path)
    before = sha(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if sha(path) != before:
        raise RuntimeError(f'Source changed while being read: {path}')
    sources[str(path)] = before
    return data


def pair_classes(row):
    """Classify proposal directions against the shaft's native CAD Z axis."""
    pairs = np.asarray(row.get('all_sampled_pairs_world_m', []), dtype=float).reshape(-1, 2, 3)
    if not np.isfinite(pairs).all():
        raise ValueError('Nonfinite saved contact-pair coordinates')
    rotation = np.asarray(row['world_T_cad'], dtype=float)[:3, :3]
    if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError('Invalid saved CAD rotation')
    # For row vectors: world = CAD @ R.T, hence CAD = world @ R.
    directions = (pairs[:, 1] - pairs[:, 0]) @ rotation
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= 0):
        raise ValueError('A sampled proposal has coincident endpoints')
    axial = np.abs(directions[:, 2] / lengths)
    labels = np.full(len(pairs), 'mixed', dtype=object)
    labels[axial < SIDE_MAX_AXIAL_COMPONENT] = 'side_spanning'
    labels[axial > END_MIN_AXIAL_COMPONENT] = 'end_to_end'
    retained_indices = [int(index) for index in row.get('retained_pair_indices', [])]
    if len(set(retained_indices)) != len(retained_indices):
        raise ValueError('Repeated retained native pair index')
    if any(index < 0 or index >= len(pairs) for index in retained_indices):
        raise ValueError('Retained pair index is outside the saved native pair array')
    poses = row.get('retained_poses', [])
    pose_indices = [int(pose['pair_index']) for pose in poses]
    if set(pose_indices) != set(retained_indices):
        raise ValueError('Retained pair and retained pose mappings disagree')
    displayed = np.asarray(row.get('contact_pairs_world_m', []), dtype=float).reshape(-1, 2, 3)
    if displayed.shape != pairs[retained_indices].shape or not np.allclose(
            displayed, pairs[retained_indices], atol=1e-12, rtol=0):
        raise ValueError('Displayed pairs differ from their saved native indices')
    count = lambda values: {name: int(sum(value == name for value in values)) for name in PAIR_TYPES}
    return dict(
        sampled_pair_counts=count(labels),
        retained_pair_counts=count(labels[retained_indices]),
        retained_pose_counts=count(labels[pose_indices]),
        retained_pair_indices=retained_indices,
        retained_pair_labels=[str(labels[index]) for index in retained_indices],
    )


def rejection_counts(row, sources):
    if 'native_rejection_counts' in row:
        return Counter({key: int(value) for key, value in row['native_rejection_counts'].items()})
    saved = row.get('candidate_trace_path')
    if not saved:
        return Counter()
    path = ROOT / saved
    if not path.exists():
        return Counter()
    trace = read(path, sources)
    return Counter(pose.get('native_message', 'unlabeled_native_rejection')
                   for pose in trace if not pose['retained'])


def bounds(rows, key):
    values = [int(row.get(key, 0)) for row in rows]
    return dict(min=min(values), max=max(values))


def combine_counts(rows, key):
    counter = Counter()
    for row in rows:
        counter.update(row[key])
    return {name: int(counter[name]) for name in PAIR_TYPES}


def plain_reason(message):
    if 'gripper-grasping part collision' in message:
        return 'Gripper overlaps the target CAD in the native opened-gripper collision check'
    if 'gripper-ground collision' in message:
        return 'Native buffered gripper overlaps an external ground object'
    if 'gripper-after parts collision' in message:
        return 'Native buffered gripper overlaps another assembly part'
    return message


def version_rows(manifest, sources):
    pinned_path = Path(__file__).with_name('vendor') / 'fabrica/requirements.txt'
    pins = {}
    if pinned_path.exists():
        sources[str(pinned_path)] = sha(pinned_path)
        for line in pinned_path.read_text().splitlines():
            if '==' in line:
                name, version = line.strip().split('==', 1)
                pins[name.lower()] = version
    return [dict(package=name, installed=version, upstream_pin=pins.get(name.lower()))
            for name, version in manifest.get('dependency_versions', {}).items()]


def render_report(analysis, manifest):
    total = analysis['totals']
    lines = [
        '# Original Fabrica grasp generation: object-only context', '',
        f"The saved trial contains **{total['placements']} placements: twelve NIST parts in eight scenes**. "
        'This is not the historical fourteen-part dataset. The gallery shows the CAD silhouettes '
        'with colored contact-pair lines, matching the requested format.', '',
        '[Open the CAD contact gallery](gallery/index.html). One line denotes one retained sampled '
        'pair; different native gripper orientations can share that line. Height and orientation '
        'are saved in the pose records and are not represented by the top-down line.', '',
        f"**{total['available_placements']}/{total['placements']} placements** have at least one retained "
        f"native pose. The original sampler returned **{total['sampled_pairs']:,} contact pairs**, "
        f"expanded to **{total['generated_poses']:,} gripper poses**. Native checks retained "
        f"**{total['retained_poses']:,} poses** supported by **{total['retained_pairs']:,} sampled-pair indices**.", '',
        'These are generated geometric candidates. No pickup, holding, insertion or assembly success '
        'was measured. VLM selection and its prompt/model are untouched.', '',
        '## Results for each part', '',
        '| Part | Placements with retained poses | Contact pairs per placement | Retained poses per placement |',
        '|---|---:|---:|---:|',
    ]
    for part, item in analysis['by_part'].items():
        p, g = item['retained_pair_count_range'], item['retained_pose_count_range']
        lines.append(f"| {part.replace('_', ' ')} | {item['available_placements']}/{item['placements']} "
                     f"| {p['min']}–{p['max']} | {g['min']}–{g['max']} |")
    lines.extend(['', 'The pair counts refer to unique native sample indices, not a second geometric '
                  'deduplication or a quality ranking. No local cap or display grouping is applied.', '',
                  '## Round-pin direction check', '',
                  'The following labels describe the **sampled proposal direction**, transformed back '
                  'into each pin’s CAD frame. They do not certify actual finger contact or force direction. '
                  'The pin shaft is taken as native CAD Z. A direction with `abs(unit_direction.z) < 0.15` '
                  'is side-spanning; a value `> 0.985` is end-to-end; other directions are mixed.', '',
                  '| Pin | Placements with retained side proposals | Retained side pairs | Retained end pairs | Retained mixed pairs | Poses from side pairs |',
                  '|---|---:|---:|---:|---:|---:|'])
    for part, item in analysis['round_pins']['by_part'].items():
        pairs, poses = item['retained_pair_counts'], item['retained_pose_counts']
        lines.append(f"| {part.replace('_', ' ')} | {item['placements_with_side_proposals']}/{item['placements']} "
                     f"| {pairs['side_spanning']:,} | {pairs['end_to_end']:,} | {pairs['mixed']:,} "
                     f"| {poses['side_spanning']:,} |")
    lines.extend(['', 'Per-scene pair and pose counts, including the exact retained native indices, '
                  'are recorded in `analysis.json`.', '', '## What was executed', '',
                  f"The pinned source is [Fabrica commit {manifest['commit'][:7]}]"
                  f"(https://github.com/yunshengtian/Fabrica/tree/{manifest['commit']}). "
                  'The original contact sampling, pair thinning/ordering, Panda pose construction, '
                  '`GraspGenerator.generate_grasps`, and `check_grasp_feasible` bodies executed unchanged. '
                  'The adapter loads original definitions while avoiding imports for unused assembly and '
                  'robot-execution stages.', '',
                  'The supplied context contains one aligned CAD in world coordinates and an identity '
                  'part path. The native assembly/fixture constructor is bypassed. Native centimetres '
                  'are used inside Fabrica; exported coordinates and hand translations are in metres. '
                  'The pose frame is `world_T_fabrica_panda_hand`, with local Y closing the fingers '
                  'and quaternion order `wxyz`; it is not our earlier gripper-site frame.', '',
                  'To remove workspace constraints as requested, the public `ground_z=-infinity` '
                  'parameter disables the contact-height gate, and the external ground collision '
                  'manager is empty. No table, fixture, other loose part or robot arm is supplied.', '',
                  'Fabrica’s own target-clearance spheres, `0.95` opposite-normal threshold, 0.5 mm '
                  'pair spacing, 200-point budget, ten sampled roll angles, Panda width rule and '
                  'target/gripper FCL checks remain. Native roll generation can discard some sampled '
                  'angles, so ten requested angles need not produce ten poses per pair. Native target '
                  'collision checking uses its opened-gripper margin; it does not simulate fingers '
                  'closing or predict part displacement.', '',
                  'Our downward-only restriction, one-degree contact-normal rule, three-height search, '
                  'contact-centering adjustment and local grouping/score were not used. No final grasp '
                  'was selected. This is the original **grasp-generation stage in an explicitly changed '
                  'context**, not the complete Fabrica assembly pipeline or reproduction of its paper’s '
                  'assembly success rates.', '', '## Timing, dependencies and checks', ''])
    elapsed = analysis['timing']['elapsed_seconds']
    if elapsed is not None:
        lines.append(f"- Candidate run elapsed time: **{elapsed:.2f} seconds**, including four-worker "
                     'generation, native FCL checks, serialization and audit. Rendering is separate.')
    lines.extend(['- Per-case times are concurrent task durations; summing them does not give elapsed time.',
                  '- Five focused integration checks passed before the full trial: exact pinned source '
                  'bytes, real FCL queries, analytic box contacts, native Panda frame/offset, and clear '
                  'versus deliberately overlapping target checks.',
                  '- Before/after hashes in the run audit protect the frozen CAD inputs, earlier '
                  'results and VLM files. Candidate evidence is not physical validation.', '',
                  '| Package | Installed | Upstream requirement |', '|---|---|---|'])
    for item in analysis['dependencies']:
        lines.append(f"| {item['package']} | {item['installed']} | {item['upstream_pin'] or 'unavailable'} |")
    lines.extend(['', 'Dependencies are installed in a project-local cache. The geometry libraries '
                  '`trimesh`, `python-fcl` and `Rtree` use the released pins. NumPy and SciPy use '
                  'Python 3.12-compatible versions rather than the older upstream requirements; the '
                  'shared Python environment was not modified.', '', '## Native default-ground limitation', '',
                  'The object-only run above disables the native ground gate. Separately, the saved '
                  'preflight checks whether every mesh vertex is at or below the literal native '
                  'ground threshold of 10 mm at `ground_z=0`. If so, no surface contact can satisfy '
                  'the strict height test. The original sampler’s unbounded loop is not run in that '
                  'analytically empty case. Other default-ground cases are marked not evaluated, '
                  'rather than reported as successful native-default results.', '',
                  f"Preflight status counts: `{json.dumps(analysis['native_default_ground_preflight_counts'], sort_keys=True)}`.", '',
                  '## Native rejections and context comparison', ''])
    if analysis['native_rejection_counts']:
        lines.extend(['| Recorded native failure | Rejected pose count |', '|---|---:|'])
        for message, count in sorted(analysis['native_rejection_counts'].items(), key=lambda x: -x[1]):
            lines.append(f'| {plain_reason(message)} | {count:,} |')
    else:
        lines.append('No native rejection was recorded in the completed saved candidate traces.')
    previous = analysis.get('previous_context')
    if previous:
        lines.extend(['', f"The earlier adapted run had {previous['available_placements']}/"
                      f"{previous['placements']} placements with passing poses; this run has "
                      f"{total['available_placements']}/{total['placements']}. **This is not a controlled "
                      'method comparison.** Sampling budgets, orientation generation, workspace '
                      'constraints and contact checks differ. Availability, timing and proposal counts '
                      'cannot establish which method has better pickup or assembly performance.'])
    unavailable = analysis['unavailable_cases']
    if unavailable:
        lines.extend(['', '## Unavailable cases', '', '| Scene | Part | Status | Detail |', '|---|---|---|---|'])
        for item in unavailable:
            detail = str(item.get('error') or 'No native pose retained').replace('|', '/')
            lines.append(f"| {item['scene']} | {item['part']} | {item['status']} | {detail} |")
    return '\n'.join(lines) + '\n'


def main(run_directory):
    run_directory = Path(run_directory).expanduser().resolve()
    sources = {}
    records = read(run_directory / 'results.json', sources)
    manifest = read(run_directory / 'manifest.json', sources)
    summary = read(run_directory / 'summary.json', sources)
    if len(records) != 96 or len({(row['scene'], row['part']) for row in records}) != 96:
        raise ValueError('Expected all 96 unique placements in the completed twelve-part trial')
    if len({row['part'] for row in records}) != 12 or len({row['scene'] for row in records}) != 8:
        raise ValueError('Expected twelve parts and eight scenes')
    by_part = {}
    aggregate_rejections = Counter()
    preflight = Counter()
    for row in records:
        aggregate_rejections.update(rejection_counts(row, sources))
        preflight[row.get('native_default_ground_preflight', {}).get('status', 'unavailable')] += 1
    for part in sorted({row['part'] for row in records}):
        rows = [row for row in records if row['part'] == part]
        by_part[part] = dict(
            placements=len(rows), available_placements=sum(row['retained_pose_count'] > 0 for row in rows),
            sampled_pairs=sum(row.get('native_pair_count', 0) for row in rows),
            generated_poses=sum(row.get('generated_pose_count', 0) for row in rows),
            retained_pairs=sum(row.get('retained_pair_count', 0) for row in rows),
            retained_poses=sum(row.get('retained_pose_count', 0) for row in rows),
            sampled_pair_count_range=bounds(rows, 'native_pair_count'),
            retained_pair_count_range=bounds(rows, 'retained_pair_count'),
            retained_pose_count_range=bounds(rows, 'retained_pose_count'),
            status_counts=dict(Counter(row['status'] for row in rows)),
        )
    pin_cases = []
    for row in sorted(records, key=lambda r: (r['part'], r['scene'])):
        if row['part'].startswith('RGOCG'):
            pin_cases.append(dict(scene=row['scene'], part=row['part'], status=row['status'], **pair_classes(row)))
    pins_by_part = {}
    for part in sorted({row['part'] for row in pin_cases}):
        rows = [row for row in pin_cases if row['part'] == part]
        pins_by_part[part] = dict(
            placements=len(rows),
            placements_with_side_proposals=sum(row['retained_pair_counts']['side_spanning'] > 0 for row in rows),
            sampled_pair_counts=combine_counts(rows, 'sampled_pair_counts'),
            retained_pair_counts=combine_counts(rows, 'retained_pair_counts'),
            retained_pose_counts=combine_counts(rows, 'retained_pose_counts'),
        )
    analysis = dict(
        scope='Read-only analysis of saved original Fabrica candidates; no generation, selection or execution.',
        totals=dict(placements=len(records), parts=12, scenes=8,
            available_placements=sum(row['retained_pose_count'] > 0 for row in records),
            sampled_pairs=sum(row.get('native_pair_count', 0) for row in records),
            generated_poses=sum(row.get('generated_pose_count', 0) for row in records),
            retained_pairs=sum(row.get('retained_pair_count', 0) for row in records),
            retained_poses=sum(row.get('retained_pose_count', 0) for row in records)),
        status_counts=dict(Counter(row['status'] for row in records)), by_part=by_part,
        round_pins=dict(definition=dict(
            assumed_shaft_axis='native_CAD_Z', units='direction cosines; translations and displayed pairs in meters',
            vector_transform='row_direction_CAD = row_direction_world @ world_R_CAD',
            side_spanning=f'abs(unit_direction_CAD.z) < {SIDE_MAX_AXIAL_COMPONENT}',
            end_to_end=f'abs(unit_direction_CAD.z) > {END_MIN_AXIAL_COMPONENT}', mixed='all remaining directions',
            pair_indexing='Zero-based native sampled-pair index; no additional geometric deduplication.',
            limitation='Proposal-direction classification only; not actual finger contact or force validation.'),
            by_part=pins_by_part, cases=pin_cases),
        native_rejection_counts=dict(aggregate_rejections),
        native_default_ground_preflight_counts=dict(preflight),
        timing=dict(elapsed_seconds=summary.get('elapsed_seconds'),
            sum_concurrent_case_seconds=sum(row.get('timing_seconds', 0) for row in records),
            note=summary.get('timing_note', 'Rendering excluded; sum of concurrent cases is not elapsed time.')),
        dependencies=version_rows(manifest, sources),
        unavailable_cases=[{key: row.get(key) for key in ('scene', 'part', 'status', 'error')}
            for row in records if not row['retained_pose_count']],
        no_VLM=True, no_candidate_generation=True, no_physics_or_execution=True,
        source_sha256=sources, analysis_source_sha256=sha(__file__),
    )
    previous_path = PREVIOUS_RUN / 'summary.json'
    if previous_path.exists():
        previous = read(previous_path, sources)
        analysis['previous_context'] = dict(
            source=str(previous_path), available_placements=previous['available'], placements=previous['placements'],
            controlled_comparison=False,
            reason='Different source budgets, pose orientations, workspace constraints and contact checks; no winner or success-rate claim.')
    for path, digest in sources.items():
        if sha(path) != digest:
            raise RuntimeError(f'Input changed during saved-result analysis: {path}')
    (run_directory / 'analysis.json').write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (run_directory / 'report.md').write_text(render_report(analysis, manifest), encoding='utf-8')
    print(json.dumps(analysis['totals']), flush=True)
    print('REPORT=' + str(run_directory / 'report.md'), flush=True)
    return analysis


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.fabrica_original_20261027.analyze_results RUN_DIR')
    main(sys.argv[1])
