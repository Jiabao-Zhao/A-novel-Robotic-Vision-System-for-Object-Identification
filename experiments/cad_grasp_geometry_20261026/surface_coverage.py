"""Exploratory native-CAD area sampling; unchanged hand and validity checks.

Usage: python -m experiments.cad_grasp_geometry_20261026.surface_coverage usb
   or: python -m experiments.cad_grasp_geometry_20261026.surface_coverage BASE_RUN

The first mode diagnoses all eight USB placements. The second requires a frozen
96-case baseline and reruns every zero-valid case, keeping successes verbatim.
This coverage change was designed after the failed USB ray/height searches.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from scipy.stats import qmc

from . import run_trial as trial
from .inputs import load_inputs, local, MANIFEST


SURFACE_CONFIG = dict(surface_samples=512, area_cdf='midpoint_strata',
                      barycentric_sequence='unscrambled_Halton_bases_2_3_skip_first',
                      contact_coordinate_rounding_m=.00001, workers=4,
                      contact_height='sampled_native_surface_height',
                      additional_height_grid=False)


def surface_pairs(vertices, triangles, gap_offset):
    """Area-distributed native triangle points seed exact horizontal recasting."""
    faces = vertices[triangles]
    raw_normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    twice_area = np.linalg.norm(raw_normals, axis=1)
    active = twice_area > 1e-18
    faces, raw_normals, twice_area = faces[active], raw_normals[active], twice_area[active]
    if not len(faces):
        raise ValueError('CAD has no nondegenerate surface triangles')
    number = SURFACE_CONFIG['surface_samples']
    cdf = np.cumsum(twice_area)
    indices = np.searchsorted(cdf, (np.arange(number)+.5)/number*cdf[-1])
    sequence = qmc.Halton(d=2, scramble=False)
    sequence.fast_forward(1)
    samples = sequence.random(number)
    root = np.sqrt(samples[:, 0])
    barycentric = np.c_[1-root, root*(1-samples[:, 1]), root*samples[:, 1]]
    centers = np.einsum('ni,nij->nj', barycentric, faces[indices])
    normals = raw_normals[indices]/twice_area[indices, None]
    axes = normals.copy()
    axes[:, 2] = 0.
    lengths = np.linalg.norm(axes, axis=1)
    usable = lengths > 1e-8
    centers, axes = centers[usable], axes[usable]/lengths[usable, None]
    max_width = trial.HAND['opening_limits_m'][1]-gap_offset-trial.CONFIG['extra_opening_m']
    metadata = dict(sampling_method='native_triangle_area_cdf_Halton_horizontal_recast',
                    upstream_seed_pairs=0, surface_samples=number, horizontal_seed_lines=len(axes),
                    maximum_contact_width_m=float(max_width), pair_cap=None,
                    qualified_horizontal_pairs=0, unique_horizontal_pairs=0,
                    native_surface_area_m2=float(cdf[-1]/2))
    if not len(axes):
        return [], metadata
    distance = float(np.linalg.norm(np.ptp(vertices, axis=0))+.1)
    first_origins, second_origins = centers-distance*axes, centers+distance*axes
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(triangles.astype(np.int32))))
    first = scene.cast_rays(o3d.core.Tensor(np.c_[first_origins, axes].astype(np.float32)))
    second = scene.cast_rays(o3d.core.Tensor(np.c_[second_origins, -axes].astype(np.float32)))
    first_distance, second_distance = first['t_hit'].numpy(), second['t_hit'].numpy()
    finite = np.isfinite(first_distance) & np.isfinite(second_distance)
    axes = axes[finite]
    first_points = first_origins[finite]+axes*first_distance[finite, None]
    second_points = second_origins[finite]-axes*second_distance[finite, None]
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
        key = tuple(np.round(pair.reshape(-1)/SURFACE_CONFIG['contact_coordinate_rounding_m']).astype(np.int64))
        if key not in seen:
            representatives.append(pair)
            seen.add(key)
    metadata['unique_horizontal_pairs'] = len(representatives)
    return representatives, metadata


def run_surface_case(input_row, output, base_record=None, source_run=None):
    started = time.perf_counter()
    original_sampler = trial.contact_pairs
    trial.contact_pairs = surface_pairs
    try:
        result = trial.run_case(input_row, output)
    finally:
        trial.contact_pairs = original_sampler
    if result.get('all_candidate_checks'):
        path = local(result['all_candidate_checks'])
        checks = json.loads(path.read_text())
        for candidate in checks:
            candidate['id'] = 'S'+candidate['id'][1:]
        trial.dump(path, checks)
        for grasp in result['grasps']:
            grasp['id'] = 'S'+grasp['id'][1:]
            grasp['grouped_candidate_ids'] = ['S'+value[1:] for value in grasp['grouped_candidate_ids']]
    extra_seconds = time.perf_counter()-started
    result.update(search_stage='surface_coverage', surface_coverage_seconds=extra_seconds)
    if base_record is not None:
        result.update(source_run=source_run,
            base_case_result=str(Path(source_run)/'cases'/input_row['scene']/input_row['part']/'result.json'),
            base_generated_count=base_record['generated_count'], base_valid_count=base_record['valid_count'],
            base_grouped_count=base_record['grouped_count'], base_rejection_counts=base_record.get('rejection_counts', {}),
            base_timing_seconds=base_record['timing_seconds'], base_sampling_seconds=base_record.get('sampling_seconds'),
            base_validation_seconds=base_record.get('validation_seconds'), surface_coverage_case_seconds=result['timing_seconds'],
            timing_seconds=base_record['timing_seconds']+extra_seconds,
            total_proposals_checked_including_base=base_record['generated_count']+result['generated_count'])
    trial.dump(Path(output)/'cases'/input_row['scene']/input_row['part']/'result.json', result)
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_geometry_20261026.surface_coverage usb|BASE_RUN')
    inputs = {(row['scene'],row['part']):row for row in load_inputs()}
    diagnostic = sys.argv[1] == 'usb'
    source, source_relative, baseline_manifest = None, None, None
    baseline = []
    critical_sources = [Path(trial.__file__), Path(__file__).with_name('geometry_kernel.py'), Path(__file__).with_name('hand_model.py')]
    protected = critical_sources+[MANIFEST,trial.BASE/'hand_model.json']
    if diagnostic:
        targets = [row for row in inputs.values() if row['part']=='USB_Male']
        if len(targets) != 8:
            raise SystemExit('USB diagnostic expects all eight scene inputs')
    else:
        source = local(sys.argv[1])
        baseline = json.loads((source/'results.json').read_text())
        baseline_manifest = json.loads((source/'manifest.json').read_text())
        if len(baseline)!=96 or {(row['scene'],row['part']) for row in baseline}!=set(inputs):
            raise SystemExit('Surface follow-up requires all 96 distinct baseline results')
        if not (source/'preservation.json').exists() or not json.loads((source/'preservation.json').read_text()).get('unchanged'):
            raise SystemExit('Baseline preservation check is missing or failed')
        if baseline_manifest['config'] != trial.CONFIG:
            raise SystemExit('Current validation configuration differs from baseline')
        for path in critical_sources:
            if trial.sha(path) != baseline_manifest['source_sha256'].get(str(path.relative_to(trial.ROOT))):
                raise SystemExit(f'Current checker differs from baseline: {path.name}')
        source_relative = str(source.relative_to(trial.ROOT))
        targets = [inputs[(row['scene'],row['part'])] for row in baseline if row['valid_count']==0]
        protected += list(source.rglob('*.json'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = trial.BASE/('diagnostics' if diagnostic else 'runs')/(stamp+'_surface_coverage')
    output.mkdir(parents=True,exist_ok=False)
    manifest = dict(created_utc=stamp,cases=8 if diagnostic else 96,config=trial.CONFIG,surface_config=SURFACE_CONFIG,
        diagnostic_only=diagnostic, protocol='Exploratory area-coverage change after USB bounds-ray and height-grid limitations; not a preregistered baseline.',
        input_manifest=str(MANIFEST.relative_to(trial.ROOT)),no_vlm=True,no_robot_execution=True,no_physics_steps=True,
        no_other_part_or_robot_self_checks=True,no_candidate_cap=True,no_success_stopping=True,
        quality_definition='Modeled contact and interference feasibility only; no force/slip or assembly-precision guarantee',
        followup_cases=[dict(scene=row['scene'],part=row['part']) for row in targets],
        source_sha256={str(path.relative_to(trial.ROOT)):trial.sha(path) for path in Path(__file__).parent.glob('*.py')},
        protected_sha256={str(path.relative_to(trial.ROOT)):trial.sha(path) for path in protected})
    if source is not None:
        manifest.update(base_run=source_relative,base_manifest_sha256=trial.sha(source/'manifest.json'),
            base_results_sha256=trial.sha(source/'results.json'),successful_base_records_copied_verbatim=True,
            pose_assumption=baseline_manifest['pose_assumption'])
    trial.dump(output/'manifest.json',manifest)
    print(f'OUTPUT={output}',flush=True)
    base_by_key = {(row['scene'],row['part']):row for row in baseline}
    combined = dict(base_by_key)
    completed = []
    with ProcessPoolExecutor(max_workers=SURFACE_CONFIG['workers'],initializer=trial.initialize) as pool:
        jobs = [pool.submit(run_surface_case,row,str(output),base_by_key.get((row['scene'],row['part'])),source_relative) for row in targets]
        for job in as_completed(jobs):
            result = job.result()
            combined[(result['scene'],result['part'])] = result
            completed.append(result)
            trial.dump(output/'fallback_results.json',sorted(completed,key=lambda row:(row['scene'],row['part'])))
            trial.dump(output/'results.json',sorted(combined.values(),key=lambda row:(row['scene'],row['part'])))
            print(json.dumps({key:result.get(key) for key in ('scene','part','status','unique_horizontal_pairs','generated_count','valid_count','grouped_count','surface_coverage_seconds','timing_seconds','error')}),flush=True)
    after = {str(path.relative_to(trial.ROOT)):trial.sha(path) for path in protected}
    trial.dump(output/'preservation.json',dict(unchanged=after==manifest['protected_sha256'],sha256_after=after))
    trial.dump(output/'completion.json',dict(completed_followup_cases=len(completed),combined_cases=len(combined),complete=len(completed)==len(targets)))
    print(f'COMPLETE={output}',flush=True)


if __name__ == '__main__':
    main()
