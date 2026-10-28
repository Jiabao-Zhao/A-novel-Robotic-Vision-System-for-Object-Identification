"""Audit saved selections independently of sampler ordering or selector calls."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from experiments.cad_grasp_geometry_20261026.gallery import read_stl
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_geometry_20261026.inputs import local
from .geometry import _faces_at_point, pad_calibration
from .run_nist import ROOT, HAND, INPUTS, dump, sha


def verify(run_dir):
    run_dir = Path(run_dir)
    records = json.loads((run_dir/'results.json').read_text())
    launch = json.loads((run_dir/'launch_manifest.json').read_text())
    inputs = {(row['scene'], row['part']): row
              for row in json.loads(INPUTS.read_text())['records']}
    if len(records) != 96 or len({(r['scene'], r['part']) for r in records}) != 96:
        raise AssertionError('Trial omitted or repeated a saved placement')
    hand = json.loads(HAND.read_text())
    calibration = pad_calibration(hand)
    parts = sorted({row['part'] for row in records})
    source_colliders = {}
    selected_count = 0
    checked_contact_points = 0
    max_reconstruction_error = 0.
    max_source_surface_distance_tolerance = 1e-6
    for part in parts:
        row = next(item for item in records if item['part']==part)
        vertices, faces = read_stl(local(row['mesh_path']))
        source_colliders[part] = MeshCollider(vertices, faces)
    for row in records:
        key = row['scene'], row['part']
        original = inputs[key]
        if not np.array_equal(row['world_T_cad'], original['world_T_cad']):
            raise AssertionError('Saved observed CAD alignment was changed')
        selection = row['selection']
        diagnostics = json.loads((run_dir/'cases'/row['scene']/row['part']/'candidate_diagnostics.json').read_text())
        if len(diagnostics) != selection['proposed_count'] or len(diagnostics)>100:
            raise AssertionError('Complete-pose count mismatch or cap violation')
        valid = [item for item in diagnostics if item.get('valid') and item['validity']['status']=='valid']
        if len(valid) != selection['valid_count']:
            raise AssertionError('Valid candidate count does not match diagnostics')
        if selection['status'] != 'selected':
            if selection.get('selected_grasp') is not None or valid:
                raise AssertionError('Failed case contains an invalid fallback or ignores a valid grasp')
            continue
        pose = selection['selected_grasp']
        selected_count += 1
        remaining = valid
        for name in ('r_tau','r_surface','theta_down_rad'):
            minimum = min(item['criteria'][name] for item in remaining)
            epsilon = selection['tie_tolerances'][name]
            remaining = [item for item in remaining if item['criteria'][name]-minimum<=epsilon]
        expected = min(item['candidate_id'] for item in remaining)
        if pose['candidate_id'] != expected:
            raise AssertionError('Saved selection violates sequential priority or deterministic tie')
        transform = np.asarray(pose['cad_T_hand'])
        pickup = np.asarray(row['world_T_cad'])
        task = json.loads((run_dir/row['derived_task_path']).read_text())
        final = np.asarray(task['task_spec']['goal_T_cad_board'])
        if not np.allclose(pose['world_T_hand_pickup'], pickup@transform, rtol=0, atol=1e-12):
            raise AssertionError('Pickup hand transform mismatch')
        if not np.allclose(pose['world_T_hand_final'], final@transform, rtol=0, atol=1e-12):
            raise AssertionError('Nominal final hand transform mismatch')
        held = pose['holding_opening_m']
        centers = calibration['centers_zero_m']+held*calibration['centers_per_opening']
        actual = centers@transform[:3,:3].T+transform[:3,3]
        pair = np.asarray(pose['contacts_cad_m'])
        error = float(np.linalg.norm(actual-pair,axis=1).max())
        max_reconstruction_error = max(max_reconstruction_error,error)
        if error>1e-9:
            raise AssertionError('Intended pad centers were not reconstructed')
        for point in pair:
            if not len(_faces_at_point(source_colliders[row['part']], point,
                                      max_source_surface_distance_tolerance)):
                raise AssertionError('Contact does not lie within tolerance of original unwelded CAD')
            checked_contact_points += 1
        midpoint = pair.mean(axis=0)
        a, d = np.asarray(task['axis_point_cad_m']), np.asarray(task['axis_direction_cad'])
        mesh = np.load(run_dir/'derived_tasks'/f"{row['part']}_validation_mesh.npz")
        vertices, faces = mesh['vertices'], mesh['faces']
        triangles = vertices[faces]
        areas = .5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                          triangles[:,2]-triangles[:,0]),axis=1)
        centroid = np.sum(areas[:,None]*triangles.mean(axis=1),axis=0)/areas.sum()
        q_tau = float(np.linalg.norm(np.cross(midpoint-a,d)))
        q_surface = float(np.linalg.norm(midpoint-centroid))
        D_tau = float(np.linalg.norm(np.cross(vertices-a,d),axis=1).max())
        D_surface = float(np.linalg.norm(vertices-centroid,axis=1).max())
        approach = pickup[:3,:3]@transform[:3,:3]@calibration['approach_hand']
        plane = np.asarray(row['table_plane_world'])
        theta = float(np.arccos(np.clip(approach@(-plane[:3]),-1,1)))
        recomputed = dict(q_tau_m=q_tau,q_surface_m=q_surface,
                          r_tau=q_tau/D_tau if D_tau else 0.,
                          r_surface=q_surface/D_surface if D_surface else 0.,theta_down_rad=theta)
        if any(not np.isclose(pose['criteria'][name], value, rtol=0, atol=1e-10)
               for name,value in recomputed.items()):
            raise AssertionError('Saved raw or normalized criteria differ from direct recomputation')
        table_min = np.inf
        for component in hand['components']:
            for opening in (pose['holding_opening_m'],pose['pre_opening_m']):
                vertices = np.asarray(component['vertices_zero_m'])+opening*np.asarray(component['translation_per_opening'])
                world = vertices@(pickup@transform)[:3,:3].T+(pickup@transform)[:3,3]
                table_min = min(table_min,float(np.min(world@plane[:3]+plane[3])))
        if table_min<pose['validity']['tolerances']['table_clearance_m']:
            raise AssertionError('Saved selected full hand violates pickup table clearance')
    protected_ok = all(sha(ROOT/path)==digest for path,digest in launch['protected_sha256'].items())
    source_ok = all(sha(ROOT/path)==digest for path,digest in launch['source_sha256'].items())
    if not protected_ok or not source_ok:
        raise AssertionError('A protected input or launched source changed')
    result = dict(placements=96, selected_count=selected_count,
        outcome_counts=dict(Counter(row['selection']['status'] for row in records)),
        complete_pose_cap_verified=True, all_observed_alignments_unchanged=True,
        sequential_priority_independently_recomputed=True,
        selected_transforms_and_pad_centers_reconstructed=True,
        selected_criteria_directly_recomputed=True,
        selected_full_hand_table_clearance_recomputed=True,
        contact_points_checked_against_original_CAD=checked_contact_points,
        original_CAD_surface_tolerance_m=max_source_surface_distance_tolerance,
        maximum_pad_center_reconstruction_error_m=max_reconstruction_error,
        protected_inputs_unchanged=protected_ok, launched_source_unchanged=source_ok,
        collision_check_scope='production validator; no independent physics or second collision engine',
        no_VLM=True,no_pickup_or_assembly_execution=True,no_physics_steps=True)
    dump(run_dir/'verification.json', result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    verify(sys.argv[1])
