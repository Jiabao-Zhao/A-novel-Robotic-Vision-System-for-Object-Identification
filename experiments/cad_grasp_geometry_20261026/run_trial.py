"""Finite downward CAD candidates and modeled validation; no robot execution/VLM."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import ConvexHull

from experiments.runtime_antipodal_assembly_20261023.vendor import AntipodalSampler, AntipodalSamplerCfg
from .inputs import load_inputs, local, MANIFEST
from .hand_model import vertices as hand_vertices, load_hand
from .geometry_kernel import MeshCollider

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/cad_grasp_geometry_20261026'
CONFIG = dict(surface_rays=512, normal_angle_deg=15., extra_opening_m=.012,
              pair_position_spacing_m=.004, pair_width_spacing_m=.002, pair_angle_spacing_deg=10.,
              finger_depths_m=[-.008, -.004, 0., .004, .006, .008],
              open_clearance_m=.0001, table_clearance_m=.0001,
              pad_edge_margin_m=.00025, contact_opening_difference_m=.00005,
              contact_inner_face_tolerance_m=.000025, contact_search_fraction_tolerance=.0001,
              straight_approach_m=.10, workers=4)
HAND = None


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    global HAND
    HAND = json.loads((BASE / 'hand_model.json').read_text())
    torch.set_num_threads(1)


def transform(v, pose):
    return np.asarray(v) @ pose[:3, :3].T + pose[:3, 3]


def load_target(row):
    mesh = o3d.io.read_triangle_mesh(str(local(row['mesh_path'])))
    raw_vertices, raw_triangles = len(mesh.vertices), len(mesh.triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    weld_displacement = 0.
    if row['part'] == 'Gear_Large':
        # Explicit STL numerical seam weld: no filling, hull or removed faces.
        from scipy.spatial import cKDTree
        before = np.asarray(mesh.vertices).copy()
        mesh.merge_close_vertices(5e-8)
        weld_displacement = float(cKDTree(np.asarray(mesh.vertices)).query(before)[0].max())
    v = transform(np.asarray(mesh.vertices), np.asarray(row['world_T_cad']))
    f = np.asarray(mesh.triangles).copy()
    return v, f, dict(raw_vertices=raw_vertices, raw_triangles=raw_triangles,
                     merged_vertices=len(v), validation_triangles=len(f),
                     numerical_seam_weld_m=5e-8 if row['part']=='Gear_Large' else 0.,
                     maximum_weld_displacement_m=weld_displacement,
                     boundary_or_nonmanifold_edges=len(mesh.get_non_manifold_edges(False)))


def contact_pairs(v, f, gap_offset):
    """Reuse GraspKit seeds, then recompute contacts on exactly horizontal rays."""
    max_width = HAND['opening_limits_m'][1] - gap_offset - CONFIG['extra_opening_m']
    sampler = AntipodalSampler(AntipodalSamplerCfg(n_sample=CONFIG['surface_rays'],
                min_length=1e-6, max_length=max_width, max_angle=np.deg2rad(CONFIG['normal_angle_deg'])))
    seeds = sampler.sample(torch.tensor(v, dtype=torch.float32), torch.tensor(f, dtype=torch.int64)).numpy().astype(np.float64)
    seeds = seeds[np.all(np.isfinite(seeds), axis=(1, 2))]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.int32))))
    center = seeds.mean(axis=1)
    axis = seeds[:, 1] - seeds[:, 0]
    axis[:, 2] = 0.
    norm = np.linalg.norm(axis, axis=1)
    center, axis = center[norm > 1e-8], axis[norm > 1e-8] / norm[norm > 1e-8, None]
    length = float(np.linalg.norm(np.ptp(v, axis=0)) + .1)
    left_origin, right_origin = center - length * axis, center + length * axis
    first = scene.cast_rays(o3d.core.Tensor(np.c_[left_origin, axis].astype(np.float32)))
    second = scene.cast_rays(o3d.core.Tensor(np.c_[right_origin, -axis].astype(np.float32)))
    a, b = first['t_hit'].numpy(), second['t_hit'].numpy()
    p, q = left_origin + axis * a[:, None], right_origin - axis * b[:, None]
    width = np.linalg.norm(q - p, axis=1)
    n1, n2 = first['primitive_normals'].numpy(), second['primitive_normals'].numpy()
    cosine = np.cos(np.deg2rad(CONFIG['normal_angle_deg']))
    keep = np.isfinite(a) & np.isfinite(b) & (width > 1e-6) & (width <= max_width)
    keep &= (np.abs(np.sum(n1 * axis, axis=1)) >= cosine)
    keep &= (np.abs(np.sum(n2 * axis, axis=1)) >= cosine)
    keep &= (np.sum(n1 * n2, axis=1) <= -np.cos(np.deg2rad(30.)))
    pairs = np.stack((p[keep], q[keep]), axis=1)
    representatives = []
    seen = set()
    for pair in pairs:
        if tuple(pair[1]) < tuple(pair[0]):
            pair = pair[::-1].copy()
        key = tuple(np.round(pair.reshape(-1)/1e-5).astype(np.int64))
        if key not in seen:
            representatives.append(pair)
            seen.add(key)
    return representatives, dict(upstream_seed_pairs=len(seeds),
                qualified_horizontal_pairs=len(pairs), unique_horizontal_pairs=len(representatives),
                maximum_contact_width_m=max_width, pair_cap=None)


def group_valid(grasps):
    """Group only after every proposal was checked; preserve depth and jaw yaw."""
    groups = []
    for grasp in grasps:
        pose = np.asarray(grasp['world_T_grip'])
        contacts = np.asarray(grasp['contact_points_world_m'])
        matched = None
        for old in groups:
            if old['pad_depth_m'] != grasp['pad_depth_m']:
                continue
            old_pose = np.asarray(old['world_T_grip'])
            if float(old_pose[:3, 0]@pose[:3, 0]) < np.cos(np.deg2rad(CONFIG['pair_angle_spacing_deg'])):
                continue
            if np.linalg.norm(old_pose[:3, 3]-pose[:3, 3]) > CONFIG['pair_position_spacing_m']:
                continue
            if np.max(np.linalg.norm(np.asarray(old['contact_points_world_m'])-contacts, axis=1)) > CONFIG['pair_position_spacing_m']:
                continue
            if abs(old['contact_opening_m']-grasp['contact_opening_m']) > CONFIG['pair_width_spacing_m']:
                continue
            matched = old
            break
        if matched is None:
            groups.append(dict(grasp, grouped_candidate_ids=[grasp['id']]))
        else:
            matched['grouped_candidate_ids'].append(grasp['id'])
    return groups


def hand_geometry():
    components = []
    for record in HAND['components']:
        component = dict(record)
        v = np.asarray(component['vertices_zero_m'])
        # MuJoCo uses convex mesh collision components. Preserve each component,
        # but never convexify the target object.
        component['vertices_zero_m'] = v[ConvexHull(v).vertices]
        components.append(component)
    pads = sorted((c for c in components if c['is_pad']), key=lambda c: np.mean(c['vertices_zero_m'], axis=0)[0])
    gap_offset = float(np.max(pads[0]['vertices_zero_m'][:, 0]) - np.min(pads[1]['vertices_zero_m'][:, 0]))
    return components, pads, gap_offset


def validate(candidate, collider, components, pads, plane):
    pose = np.asarray(candidate['world_T_grip'])
    opening = candidate['opening_m']
    depth = candidate['pad_depth_m']
    if not HAND['opening_limits_m'][0] <= opening <= HAND['opening_limits_m'][1]:
        return 'commanded_opening_outside_hand_limits', {}
    for pad in pads:
        v = np.asarray(pad['vertices_zero_m'])
        if not v[:, 2].min()+CONFIG['pad_edge_margin_m'] <= depth <= v[:, 2].max()-CONFIG['pad_edge_margin_m']:
            return 'proposed_contact_outside_inner_pad', {}
    world_open = [(c, transform(hand_vertices(c, opening), pose)) for c in components]
    world_closed = [(c, transform(hand_vertices(c, 0.), pose)) for c in components]
    clearance = min(float((v @ plane[:3] + plane[3]).min()) for _, v in world_open + world_closed)
    if clearance < CONFIG['table_clearance_m']:
        return 'finger_or_palm_too_low_for_table', {'table_clearance_m': clearance}
    approach = np.array([0., 0., CONFIG['straight_approach_m']])
    for c, v in world_open:
        if collider.swept_intersects(v, approach, CONFIG['open_clearance_m']):
            return 'open_hand_or_downward_approach_hits_target', {'blocking_component': c['name']}
    pad_results = []
    for pad in pads:
        v = transform(hand_vertices(pad, opening), pose)
        displacement = pose[:3, :3] @ (-opening * np.asarray(pad['translation_per_opening']))
        contact = collider.first_contact(v, displacement, CONFIG['contact_search_fraction_tolerance'])
        if contact is None:
            return 'inner_pad_cannot_reach_target', {'pad': pad['name']}
        if contact.get('triangle_index') is None or contact.get('point_world_m') is None:
            return 'first_pad_contact_has_no_surface_witness', {'pad': pad['name']}
        contact_opening = opening * (1-contact['fraction'])
        pad_v = hand_vertices(pad, contact_opening)
        negative = float(np.asarray(pad['translation_per_opening'])[0]) < 0
        inner_x = pad_v[:, 0].max() if negative else pad_v[:, 0].min()
        simultaneous = collider.surface_contacts(transform(pad_v, pose))
        inner_contacts = []
        for witness in simultaneous:
            point_local = (np.asarray(witness['point_world_m'])-pose[:3, 3])@pose[:3, :3]
            if abs(point_local[0]-inner_x) <= CONFIG['contact_inner_face_tolerance_m']:
                inner_contacts.append((witness, point_local))
        if not inner_contacts:
            return 'pad_side_or_back_touches_before_inner_face', {'pad':pad['name']}
        facing = [(w, p, abs(float(np.asarray(w['normal_world'])@pose[:3, 0]))) for w, p in inner_contacts]
        facing = [(w,p,n) for w,p,n in facing if n >= np.cos(np.deg2rad(CONFIG['normal_angle_deg']))]
        if not facing:
            return 'first_contact_surface_not_facing_finger_closing', {'pad': pad['name']}
        witness, point_local, alignment = min(facing, key=lambda item:np.linalg.norm(item[1][1:]-[0., depth]))
        contact.update(point_world_m=witness['point_world_m'], triangle_index=witness['triangle_index'])
        pad_results.append(dict(pad=pad['name'], opening_m=float(contact_opening),
                  point_world_m=contact['point_world_m'], first_contact=contact,
                  normal_alignment=alignment))
    mismatch = abs(pad_results[0]['opening_m']-pad_results[1]['opening_m'])
    if mismatch > CONFIG['contact_opening_difference_m']:
        return 'fingers_reach_target_at_different_openings', {'first_contact_openings_m':[p['opening_m'] for p in pad_results], 'opening_difference_m':mismatch}
    # Stop at the EARLIER finger contact. The other contact is only bracketed
    # within the declared 50 micrometre opening tolerance, never forced through CAD.
    stop_opening = max(p['opening_m'] for p in pad_results)
    for c, v in world_open:
        if c['is_pad']:
            continue
        displacement = pose[:3, :3] @ ((stop_opening-opening)*np.asarray(c['translation_per_opening']))
        if collider.swept_intersects(v, displacement):
            return 'finger_body_or_palm_touches_before_pad_stop', {'blocking_component':c['name']}
    return None, dict(table_clearance_m=clearance, contact_opening_m=float(stop_opening),
                     first_contact_openings_m=[p['opening_m'] for p in pad_results],
                     contact_opening_difference_m=float(mismatch), actual_contacts=pad_results,
                     contact_points_world_m=[p['point_world_m'] for p in pad_results])


def run_case(row, output):
    started = time.perf_counter()
    record = dict(row, status='started', generated_count=0, valid_count=0, grouped_count=0, grasps=[])
    case = Path(output) / 'cases' / row['scene'] / row['part']
    try:
        components, pads, gap_offset = hand_geometry()
        v, f, mesh_info = load_target(row)
        record['mesh_info'] = mesh_info
        collider = MeshCollider(v, f)
        record['target_solid_topology'] = collider.topology
        if collider.topology['closed_component_count'] != collider.topology['component_count']:
            raise ValueError('CAD has an open component; full solid containment is unavailable')
        start = time.perf_counter()
        pairs, sampling = contact_pairs(v, f, gap_offset)
        record.update(sampling)
        record['sampling_seconds'] = time.perf_counter()-start
        rejects, all_candidates = Counter(), []
        plane = np.asarray(row['table_plane_world'])
        validation_start = time.perf_counter()
        for pair in pairs:
            width = float(np.linalg.norm(pair[1]-pair[0]))
            for jaw_sign in (1., -1.):
                x = jaw_sign*(pair[1]-pair[0])/width
                a = np.array([0., 0., -1.])
                rotation = np.column_stack((x, np.cross(a, x), a))
                for depth in CONFIG['finger_depths_m']:
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = pair.mean(axis=0)-rotation@np.array([0., 0., depth])
                    candidate = dict(id=f'G{len(all_candidates)+1:04d}', world_T_grip=pose.tolist(),
                              contacts_world_m=pair.tolist(), width_m=width,
                              opening_m=width+gap_offset+CONFIG['extra_opening_m'], pad_depth_m=depth)
                    reason, evidence = validate(candidate, collider, components, pads, plane)
                    candidate.update(valid=reason is None, rejection_reason=reason, **evidence)
                    all_candidates.append(candidate)
                    if reason:
                        rejects[reason] += 1
                    else:
                        record['grasps'].append(candidate)
        valid_count = len(record['grasps'])
        record['grasps'] = group_valid(record['grasps'])
        record.update(generated_count=len(all_candidates), valid_count=valid_count,
                      grouped_count=len(record['grasps']), rejection_counts=dict(rejects),
                      validation_seconds=time.perf_counter()-validation_start,
                      status='feasible_grasps_available' if record['grasps'] else 'no_feasible_grasp_in_finite_search')
        dump(case / 'candidate_checks.json', all_candidates)
        record['all_candidate_checks'] = str((case/'candidate_checks.json').relative_to(ROOT))
    except Exception as error:
        record.update(status='experiment_error', error=str(error), traceback=traceback.format_exc())
    record['timing_seconds'] = time.perf_counter()-started
    dump(case / 'result.json', record)
    return record


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    if not (BASE/'hand_model.json').exists():
        dump(BASE/'hand_model.json', load_hand())
    initialize()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/'runs'/stamp
    output.mkdir(parents=True, exist_ok=False)
    rows = load_inputs()
    if len(sys.argv)>1:
        # Explicit development subset only. Full run has no subset argument.
        rows = rows[:int(sys.argv[1])]
    sources = list(Path(__file__).parent.glob('*.py'))
    protected = [ROOT/'docs/cad_view_prompt.txt', ROOT/'experiments/runtime_antipodal_assembly_20261023/grasp_planning.py', MANIFEST]
    manifest = dict(created_utc=stamp, config=CONFIG, cases=len(rows),
             input_manifest=str(MANIFEST.relative_to(ROOT)), hand_source=str(SCENE) if 'SCENE' in globals() else 'saved Panda scene',
             pose_assumption='Correct CAD/crop association supplied; saved observed-depth CAD fit, no simulator object pose',
             no_vlm=True, no_robot_execution=True, no_physics_steps=True,
             no_other_part_or_robot_self_checks=True, no_candidate_cap=True,
             quality_definition='Modeled contact and interference feasibility only; no force/slip or assembly-precision guarantee',
             source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
             protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in protected})
    dump(output/'manifest.json', manifest)
    print(f'OUTPUT={output}', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG['workers'], initializer=initialize) as pool:
        jobs = {pool.submit(run_case, row, str(output)):row for row in rows}
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            results.sort(key=lambda r:(r['scene'], r['part']))
            dump(output/'results.json', results)
            print(json.dumps({k:result.get(k) for k in ('scene','part','status','unique_horizontal_pairs','generated_count','valid_count','grouped_count','timing_seconds','error')}), flush=True)
    after = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    dump(output/'preservation.json', dict(unchanged=after==manifest['protected_sha256'], sha256_after=after))
    print(f'COMPLETE={output}', flush=True)


if __name__ == '__main__':
    main()
