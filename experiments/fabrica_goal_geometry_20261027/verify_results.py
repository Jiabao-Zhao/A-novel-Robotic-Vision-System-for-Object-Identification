"""Audit frozen-bank filtering and independent native-mesh FCL spot checks."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys

from experiments.fabrica_original_20261027 import native
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.cad_grasp_geometry_20261026.inputs import local, read
from .endpoint import welded_mesh, contains_batched
from .task_inputs import geometry_spec
from .run_trial import dump, ROOT


def independent_endpoint(pose, pick, spec, receivers, up, raw_cm):
    """Apply original native transforms to vertices before FCL registration.

    Production instead bakes one mirror and moves persistent BVHs. This audit
    independently checks that those different constructions occupy equal space.
    """
    goal = spec['goal_T_cad_board'] @ np.linalg.inv(pick) @ np.asarray(pose['world_T_fabrica_panda_hand'])
    quaternion = Rotation.from_matrix(goal[:3, :3]).as_quat()[[3, 0, 1, 2]]
    transforms_cm = up.geometry['get_gripper_meshes_transforms'](
        'panda', raw_cm, goal[:3, 3] * 100., quaternion, np.eye(4), pose['open_ratio'])
    hand = up.trimesh.collision.CollisionManager()
    hand_meshes = []
    for name, original in raw_cm.items():
        mesh = original.copy()
        mesh.apply_transform(transforms_cm[name])
        mesh.vertices *= .01
        hand.add_object(name, mesh)
        hand_meshes.append(mesh)
    if hand.in_collision_other(receivers['manager']):
        return False
    for mesh in hand_meshes:
        for receiver in receivers['meshes']:
            if contains_batched(receiver, mesh.vertices).any():
                return False
            if contains_batched(mesh, receiver.vertices).any():
                return False
    return True


def main(run_dir):
    run_dir = local(run_dir)
    records = read(run_dir/'results.json')
    up = native.load()
    raw_cm = up.geometry['load_gripper_meshes']('panda', str(native.VENDOR/'assets'),
        has_ft_sensor=False, visual=False)
    receiver_cache = {}
    rows, samples, mismatches = [], [], []
    for record in records:
        scene, part = record['scene'], record['part']
        spec = geometry_spec(part)
        if part not in receiver_cache:
            meshes = []
            manager = up.trimesh.collision.CollisionManager()
            for item in spec['receivers']:
                mesh = welded_mesh(item['mesh_path'])
                mesh.apply_transform(item['board_T_mesh'])
                meshes.append(mesh)
                manager.add_object(item['name'], mesh)
            receiver_cache[part] = dict(meshes=meshes, manager=manager)
        source = read(ROOT/record['source_case_path'])
        decisions = read(run_dir/'cases'/scene/part/'decisions.json')
        by_id = {p['id']: p for p in source['retained_poses']}
        keep = [d for d in decisions if d['retained']]
        reject = [d for d in decisions if not d['endpoint_pass']]
        region_only = [d for d in decisions if d['endpoint_pass'] and not d['mating_region_eligible']]
        selected = keep[:1] + keep[-1:] + reject[:1] + region_only[:1]
        seen = set()
        for decision in selected:
            if decision['id'] in seen:
                continue
            seen.add(decision['id'])
            observed = independent_endpoint(by_id[decision['id']],
                np.asarray(record['world_T_cad']), spec, receiver_cache[part], up, raw_cm)
            item = dict(scene=scene, part=part, id=decision['id'],
                expected=decision['endpoint_pass'], independently_clear=observed,
                recorded_reason=decision['endpoint_reason'])
            samples.append(item)
            if observed != decision['endpoint_pass']:
                mismatches.append(item)
        pose_ids = {p['id'] for p in record['retained_poses']}
        if pose_ids != {d['id'] for d in decisions if d['retained']}:
            raise AssertionError('Trace and final bank differ')
        if len(decisions) != record['source_retained_pose_count']:
            raise AssertionError('Not every saved candidate was assessed')
        counts = Counter(d['endpoint_reason'] for d in decisions if not d['endpoint_pass'])
        rows.append(dict(scene=scene, part=part, source_poses=len(decisions),
            region_eligible=record['region_eligible_poses'], endpoint_clear=record['endpoint_pass_poses'],
            remaining_poses=len(keep), remaining_pairs=record['retained_pair_count'],
            mating_region_status=record['mating_region_check_status'],
            endpoint_rejection_counts=json.dumps(dict(counts), sort_keys=True)))
    with (run_dir/'placement_counts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [r for r in rows if not r['remaining_poses']]
    round_sides = {}
    for part in sorted({r['part'] for r in records if r['part'].startswith('RGOCG')}):
        per_scene = {}
        mesh = welded_mesh(local(next(r['mesh_path'] for r in records if r['part'] == part)))
        for record in [r for r in records if r['part'] == part]:
            inverse = np.linalg.inv(record['world_T_cad'])
            side_count = 0
            for pose in record['retained_poses']:
                points = np.asarray(pose['contacts_world_m']) @ inverse[:3, :3].T + inverse[:3, 3]
                if np.all(points[:, 2] > mesh.bounds[0, 2] + 1e-6) and np.all(points[:, 2] < mesh.bounds[1, 2] - 1e-6):
                    side_count += 1
            per_scene[record['scene']] = side_count
        round_sides[part] = per_scene
    manifest = read(run_dir/'manifest.json')
    protected = all(native.sha(ROOT/path) == sha for path, sha in manifest['protected_sha256'].items())
    receiver_unchanged = all(native.sha(ROOT/path) == sha for path, sha in manifest['receiver_sha256'].items())
    verification = dict(placements=96, saved_candidates_assessed=sum(r['source_poses'] for r in rows),
        every_trace_matches_survivors=True, native_vertex_FCL_spot_checks=len(samples),
        spot_check_mismatches=mismatches, samples=samples, zero_survivor_placements=failed,
        retained_round_side_poses=round_sides, protected_files_unchanged=protected,
        receiver_files_unchanged=receiver_unchanged,
        no_insertion_check=True, no_release_check=True, no_VLM=True, no_execution=True)
    dump(run_dir/'independent_verification.json', verification)
    if mismatches or not protected or not receiver_unchanged:
        raise AssertionError('Independent verification failed; inspect saved details')
    print(json.dumps({k: v for k, v in verification.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
