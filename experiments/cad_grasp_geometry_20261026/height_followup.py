"""Declared exploratory height coverage for zero-grasp cases of a frozen run.

Usage: python -m experiments.cad_grasp_geometry_20261026.height_followup BASE_RUN

Requires all 96 baseline results. The initial sampler and saved baseline results
are unchanged. Every zero-valid baseline case is rerun at a fixed height grid
along its existing horizontal contact lines, without stopping after a success.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

from . import run_trial as trial
from .inputs import load_inputs, local, MANIFEST


HEIGHT_CONFIG = dict(lower_surface_margin_m=.001, upper_surface_margin_m=.001,
                     vertical_step_m=.002, include_upper_endpoint=True,
                     line_position_rounding_m=.00001, line_angle_rounding_rad=.000001,
                     contact_coordinate_rounding_m=.00001, workers=4)


def height_grid(minimum_z, maximum_z):
    low = float(minimum_z) + HEIGHT_CONFIG['lower_surface_margin_m']
    high = float(maximum_z) - HEIGHT_CONFIG['upper_surface_margin_m']
    if high < low:
        return np.empty(0)
    grid = np.arange(low, high + 1e-12, HEIGHT_CONFIG['vertical_step_m'])
    if not len(grid) or abs(grid[-1]-high) > 1e-10:
        grid = np.r_[grid, high]
    return grid


def existing_xy_lines(candidates):
    """Remove repeated infinite XY lines, including repeated jaw/depth variants."""
    lines, seen = [], set()
    for candidate in candidates:
        pair = np.asarray(candidate['contacts_world_m'], dtype=float)
        center = pair.mean(axis=0)
        axis = pair[1]-pair[0]
        axis[2] = 0.
        length = np.linalg.norm(axis)
        if length <= 1e-10:
            continue
        axis /= length
        if axis[0] < 0 or (abs(axis[0]) < 1e-12 and axis[1] < 0):
            axis *= -1
        anchor = center[:2] - axis[:2]*(axis[:2]@center[:2])
        angle = np.arctan2(axis[1], axis[0])
        key = tuple(np.r_[np.round(anchor/HEIGHT_CONFIG['line_position_rounding_m']),
                           np.round(angle/HEIGHT_CONFIG['line_angle_rounding_rad'])].astype(np.int64))
        if key not in seen:
            lines.append((center.copy(), axis.copy()))
            seen.add(key)
    return lines


def recast_height_pairs(vertices, triangles, gap_offset, lines):
    grid = height_grid(vertices[:, 2].min(), vertices[:, 2].max())
    max_width = trial.HAND['opening_limits_m'][1] - gap_offset - trial.CONFIG['extra_opening_m']
    metadata = dict(sampling_method='existing_xy_lines_fixed_height_grid',
                    upstream_seed_pairs=0, source_xy_line_count=len(lines),
                    height_grid_world_m=grid.tolist(), height_grid_count=len(grid),
                    horizontal_lines_cast=len(lines)*len(grid),
                    maximum_contact_width_m=float(max_width), pair_cap=None,
                    qualified_horizontal_pairs=0, unique_horizontal_pairs=0)
    if not len(lines) or not len(grid):
        return [], metadata
    centers = np.asarray([center for center, axis in lines])
    axes = np.asarray([axis for center, axis in lines])
    centers = np.tile(centers, (len(grid), 1))
    centers[:, 2] = np.repeat(grid, len(lines))
    axes = np.tile(axes, (len(grid), 1))
    distance = float(np.linalg.norm(np.ptp(vertices, axis=0)) + .1)
    first_origins, second_origins = centers-distance*axes, centers+distance*axes
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.int32))))
    first = scene.cast_rays(o3d.core.Tensor(np.c_[first_origins, axes].astype(np.float32)))
    second = scene.cast_rays(o3d.core.Tensor(np.c_[second_origins, -axes].astype(np.float32)))
    first_distance, second_distance = first['t_hit'].numpy(), second['t_hit'].numpy()
    finite = np.isfinite(first_distance) & np.isfinite(second_distance)
    if not finite.any():
        return [], metadata
    axes = axes[finite]
    first_points = first_origins[finite] + axes*first_distance[finite, None]
    second_points = second_origins[finite] - axes*second_distance[finite, None]
    widths = np.linalg.norm(second_points-first_points, axis=1)
    first_normals = first['primitive_normals'].numpy()[finite]
    second_normals = second['primitive_normals'].numpy()[finite]
    cosine = np.cos(np.deg2rad(trial.CONFIG['normal_angle_deg']))
    keep = (widths > 1e-6) & (widths <= max_width)
    keep &= np.abs(np.sum(first_normals*axes, axis=1)) >= cosine
    keep &= np.abs(np.sum(second_normals*axes, axis=1)) >= cosine
    keep &= np.sum(first_normals*second_normals, axis=1) <= -np.cos(np.deg2rad(30.))
    pairs = np.stack((first_points[keep], second_points[keep]), axis=1)
    metadata['qualified_horizontal_pairs'] = len(pairs)
    representatives, seen = [], set()
    for pair in pairs:
        if tuple(pair[1]) < tuple(pair[0]):
            pair = pair[::-1].copy()
        key = tuple(np.round(pair.reshape(-1)/HEIGHT_CONFIG['contact_coordinate_rounding_m']).astype(np.int64))
        if key not in seen:
            representatives.append(pair)
            seen.add(key)
    metadata['unique_horizontal_pairs'] = len(representatives)
    return representatives, metadata


def _run_followup_case(base_record, input_row, source_run, output):
    started = time.perf_counter()
    source_checks = base_record.get('all_candidate_checks')
    candidates = json.loads(local(source_checks).read_text()) if source_checks else []
    lines = existing_xy_lines(candidates)
    original_sampler = trial.contact_pairs
    trial.contact_pairs = lambda v, f, offset: recast_height_pairs(v, f, offset, lines)
    try:
        result = trial.run_case(input_row, output)
    finally:
        trial.contact_pairs = original_sampler
    case = Path(output)/'cases'/input_row['scene']/input_row['part']
    if result.get('all_candidate_checks'):
        path = local(result['all_candidate_checks'])
        checks = json.loads(path.read_text())
        for candidate in checks:
            candidate['id'] = 'H'+candidate['id'][1:]
        trial.dump(path, checks)
        for grasp in result['grasps']:
            grasp['id'] = 'H'+grasp['id'][1:]
            grasp['grouped_candidate_ids'] = ['H'+value[1:] for value in grasp['grouped_candidate_ids']]
    extra_seconds = time.perf_counter()-started
    result.update(search_stage='height_followup', source_run=source_run,
                  base_case_result=str((Path(source_run)/'cases'/input_row['scene']/input_row['part']/'result.json')),
                  base_generated_count=base_record['generated_count'],
                  base_valid_count=base_record['valid_count'], base_grouped_count=base_record['grouped_count'],
                  base_rejection_counts=base_record.get('rejection_counts', {}),
                  base_timing_seconds=base_record['timing_seconds'],
                  base_sampling_seconds=base_record.get('sampling_seconds'),
                  base_validation_seconds=base_record.get('validation_seconds'),
                  height_followup_case_seconds=result['timing_seconds'],
                  height_followup_seconds=extra_seconds,
                  timing_seconds=base_record['timing_seconds']+extra_seconds,
                  total_proposals_checked_including_base=base_record['generated_count']+result['generated_count'])
    trial.dump(case/'result.json', result)
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_geometry_20261026.height_followup BASE_RUN')
    source = local(sys.argv[1])
    if not source.is_dir():
        raise SystemExit(f'Baseline run directory does not exist: {source}')
    baseline = json.loads((source/'results.json').read_text())
    baseline_manifest = json.loads((source/'manifest.json').read_text())
    if not (source/'preservation.json').exists() or not json.loads((source/'preservation.json').read_text()).get('unchanged'):
        raise SystemExit('Baseline preservation check is missing or failed; wait for baseline completion.')
    inputs = {(row['scene'], row['part']):row for row in load_inputs()}
    if len(baseline) != 96 or {(row['scene'], row['part']) for row in baseline} != set(inputs):
        raise SystemExit('Height follow-up requires all 96 distinct baseline results; baseline is incomplete.')
    if baseline_manifest['config'] != trial.CONFIG:
        raise SystemExit('Current validation configuration differs from the frozen baseline.')
    critical_sources = [Path(trial.__file__), Path(__file__).with_name('geometry_kernel.py'),
                        Path(__file__).with_name('hand_model.py')]
    for path in critical_sources:
        key = str(path.relative_to(trial.ROOT))
        expected = baseline_manifest['source_sha256'].get(key)
        if expected is None or trial.sha(path) != expected:
            raise SystemExit(f'Current checker source differs from frozen baseline: {key}')
    source_relative = str(source.relative_to(trial.ROOT))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = trial.BASE/'runs'/(stamp+'_height_followup')
    output.mkdir(parents=True, exist_ok=False)
    sources = list(Path(__file__).parent.glob('*.py'))
    protected = list(source.rglob('*.json'))+critical_sources+[MANIFEST, trial.BASE/'hand_model.json']
    manifest = dict(created_utc=stamp, cases=96, config=trial.CONFIG, height_config=HEIGHT_CONFIG,
                    base_run=source_relative, base_manifest_sha256=trial.sha(source/'manifest.json'),
                    base_results_sha256=trial.sha(source/'results.json'),
                    input_manifest=str(MANIFEST.relative_to(trial.ROOT)),
                    pose_assumption=baseline_manifest['pose_assumption'],
                    quality_definition=baseline_manifest['quality_definition'],
                    followup_cases=[dict(scene=row['scene'],part=row['part']) for row in baseline if row['valid_count']==0],
                    protocol='Exploratory fixed height-coverage follow-up designed after the N00 USB diagnostic; not a preregistered baseline.',
                    no_vlm=True, no_robot_execution=True, no_physics_steps=True,
                    no_other_part_or_robot_self_checks=True,
                    no_candidate_cap=True, no_success_stopping=True,
                    successful_base_records_copied_verbatim=True,
                    timing='Follow-up case timing_seconds includes base_timing_seconds plus height_followup_seconds; successful baseline records retain their original time.',
                    source_sha256={str(path.relative_to(trial.ROOT)):trial.sha(path) for path in sources},
                    protected_sha256={str(path.relative_to(trial.ROOT)):trial.sha(path) for path in protected})
    trial.dump(output/'manifest.json', manifest)
    print(f'OUTPUT={output}', flush=True)
    combined = {(row['scene'],row['part']):row for row in baseline}
    targets = [row for row in baseline if row['valid_count']==0]
    trial.dump(output/'results.json', sorted(combined.values(), key=lambda row:(row['scene'],row['part'])))
    completed = []
    trial.dump(output/'fallback_results.json', [])
    with ProcessPoolExecutor(max_workers=HEIGHT_CONFIG['workers'], initializer=trial.initialize) as pool:
        jobs = [pool.submit(_run_followup_case, row, inputs[(row['scene'],row['part'])], source_relative, str(output)) for row in targets]
        for job in as_completed(jobs):
            result = job.result()
            combined[(result['scene'],result['part'])] = result
            completed.append(result)
            trial.dump(output/'fallback_results.json', sorted(completed, key=lambda row:(row['scene'],row['part'])))
            trial.dump(output/'results.json', sorted(combined.values(), key=lambda row:(row['scene'],row['part'])))
            print(json.dumps({key:result.get(key) for key in ('scene','part','status','unique_horizontal_pairs','generated_count','valid_count','grouped_count','height_followup_seconds','timing_seconds','error')}), flush=True)
    after = {str(path.relative_to(trial.ROOT)):trial.sha(path) for path in protected}
    trial.dump(output/'preservation.json', dict(unchanged=after==manifest['protected_sha256'],sha256_after=after))
    trial.dump(output/'completion.json',dict(completed_followup_cases=len(completed), combined_cases=len(combined), complete=len(completed)==len(targets)))
    print(f'COMPLETE={output}', flush=True)


if __name__ == '__main__':
    main()
