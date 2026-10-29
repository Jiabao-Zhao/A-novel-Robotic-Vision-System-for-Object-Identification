"""Inspect corrected CAD task inputs without generating or executing grasps."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

import numpy as np

from experiments.nist_mating_goals_20261028.cad_tools import read_native_mesh, transform_points
from experiments.cad_grasp_geometry_20261026.inputs import local
from .history import verify_frozen_run

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/nist_scene_corrections_20261029'


def verify():
    from experiments.cad_grasp_selection_20261029.task_inputs import build_request
    from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration

    registry = json.loads((OUTPUT / 'assembly_goals.json').read_text(encoding='utf-8'))
    rows = {r['part']: r for r in registry['parts']}
    top = float(registry['board_top_board_m'])
    pins = []
    for name, row in rows.items():
        if row.get('board_T_part') is None:
            raise ValueError(f'Corrected reference unavailable: {name}')
        T = np.asarray(row['board_T_part'], float)
        assert np.allclose(T[3], [0,0,0,1], atol=1e-10)
        assert np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-9)
        assert np.linalg.det(T[:3,:3]) > .99999999
        if name.startswith(('KET','RGOCG')):
            v, _ = read_native_mesh(ROOT / row['part_mesh'])
            placed = transform_points(v, T)
            depth = top - float(placed[:,2].min())
            assert abs(depth-.020) < 1e-8, (name, depth)
            direction = T[:3,:3] @ np.asarray(row['axis_direction_cad'])
            assert np.allclose(direction, [0,0,-1], atol=1e-8), (name, direction)
            pins.append(dict(part=name, measured_tip_depth_below_board_top_m=depth))
    assert len(pins) == 6
    old_inputs = ROOT / 'outputs/cad_grasp_geometry_20261026/inputs.json'
    observed = json.loads(old_inputs.read_text(encoding='utf-8'))['records']
    hand = json.loads((ROOT / 'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text(encoding='utf-8'))
    calibration = fingertip_calibration(hand)
    inspected = []
    for record in observed:
        part_record = rows[record['part']]
        observed_mesh_hash = hashlib.sha256(local(record['mesh_path']).read_bytes()).hexdigest()
        assert observed_mesh_hash == part_record['part_mesh_sha256'], record['part']
        result = build_request(record['part'], record['world_T_cad'], record['table_plane_world'],
                               registry=registry, hand=hand)
        if record.get('input_status') == 'available':
            assert result['status'] == 'available', (record['scene'], record['part'], result.get('message'))
            request = result['request']
            assert np.allclose(request['world_T_cad_pickup'], record['world_T_cad'])
            assert callable(request['protected_contact_check'])
            assert request['receivers']
            region = request['sampling_contact_region']
            assert request['sampling_mode'] == 'assembly_region_guided'
            assert region['eligible_area_m2'] > 0
            assert region['provenance']['positive_region_provided']
            assert region['mesh_face_count'] == len(request['faces'])
        inspected.append(dict(scene=record['scene'], part=record['part'],
                              status=result['status'], grasp_generation_called=False))
    assert len(inspected) == 96 and len({r['scene'] for r in inspected}) == 8
    history = verify_frozen_run()
    gallery_links = []
    page = OUTPUT / 'gallery/index.html'
    if page.exists():
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding='utf-8')):
            if link.startswith(('https:', '#')):
                continue
            assert (page.parent / link).resolve().is_file(), link
            gallery_links.append(link)
    images = list((OUTPUT / 'gallery').glob('*.png'))
    for path in images:
        with Image.open(path) as img:
            img.verify()
    for path in [*Path(__file__).parent.glob('*.py'),
                 *(ROOT / 'experiments/cad_grasp_selection_20261029').glob('*.py'),
                 ROOT / 'simulation/nist_task_board_1.py', ROOT / 'simulation/nist_board_geometry.py']:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    receipt = dict(
        scope='Source correction, static CAD construction and input inspection only',
        pin_tip_depth_below_board_top_m=.020, pin_depth_checks=pins,
        corrected_goal_count=len(rows), observed_input_cases_inspected=len(inspected),
        target_mesh_frames_match_observed_CAD_hashes=True,
        observed_input_scene_count=8,
        offline_tests=dict(executed_by_this_inspector=False,
            note='Run the separate pytest command; this input inspector does not infer its outcome.',
            command='python -B -m pytest -q -p no:cacheprovider tests/test_nist_scene_corrections.py experiments/nist_scene_corrections_20261029 experiments/cad_grasp_selection_20261029'),
        local_gallery_links_verified=len(gallery_links),
        gallery_PNG_files_verified=len(images),
        observed_source_sha256=hashlib.sha256(old_inputs.read_bytes()).hexdigest(),
        cases=inspected, history=history,
        fingertip_origin_grip_site_m=calibration['fingertip_origin_grip_site_m'].tolist(),
        contact_inset_from_usable_pad_tip_m=calibration['contact_inset_m'],
        full_hand_distal_extent_beyond_pad_tip_m=calibration['full_hand_distal_extent_beyond_pad_tip_m'],
        complete_finger_pad_palm_geometry_retained=True,
        assembly_guided_sampling_regions_nonempty_for_all_observed_inputs=True,
        grasp_generation=False,VLM_selection=False,pickup=False,assembly=False,physics_steps=False,
        insertion_and_release_checks=False,
        note='Input construction is not candidate generation, pickup success or verified physical assembly. Current scene source is updated; exact old source is preserved at the recorded snapshot.')
    (OUTPUT / 'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    source_paths = [*Path(__file__).parent.glob('*.py'),
                    *Path(__file__).parent.glob('*.json'),
                    *(ROOT / 'experiments/cad_grasp_selection_20261029').glob('*.py'),
                    ROOT / 'experiments/cad_grasp_selection_20261029/config.json',
                    ROOT / 'simulation/nist_task_board_1.py', ROOT / 'simulation/nist_board_geometry.py',
                    ROOT / 'experiments/nist_official_reference_20261029/task_requirements.json']
    inventory = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(source_paths)}
    (OUTPUT / 'source_inventory.json').write_text(json.dumps(dict(
        note='Recorded after implementing and checking the corrections; not a historical pre-run freeze',
        sha256=inventory), indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['corrected_goal_count','observed_input_cases_inspected','observed_input_scene_count','grasp_generation','physics_steps']}))
    return receipt


if __name__ == '__main__':
    verify()
