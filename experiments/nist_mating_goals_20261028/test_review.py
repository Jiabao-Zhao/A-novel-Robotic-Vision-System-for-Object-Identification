"""Offline checks on the corrected frame mapping and unresolved-goal contract."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/nist_mating_goals_20261028'


def records():
    return {r['part']:r for r in json.loads((OUTPUT/'assembly_goals.json').read_text())['parts']}


def test_unresolved_mates_never_fall_back_to_old_nominal_goals():
    rows=records()
    assert len(rows)==12
    for part in ['BNC_Male','DSUB_Male','Waterproof_Male','M16_Hex_Nut']:
        assert rows[part]['board_T_part'] is None
        assert np.asarray(rows[part]['review_only_board_T_part']).shape==(4,4)
    assert all(not r['grasp_generation_enabled'] for r in rows.values())


def test_pin_inserted_region_follows_corrected_positive_end():
    for part,row in records().items():
        if not part.startswith(('KET','RGOCG')): continue
        h=row['protected_inserted_halfspace']
        n=np.asarray(h['normal_cad']); b=h['offset_m']
        assert np.dot(n,[0,0,.025])+b>0
        assert np.dot(n,[0,0,-.025])+b<0
        assert abs(np.dot(n,[0,0,.013])+b)<1e-8
        T=np.asarray(row['board_T_part'])
        assert np.allclose(T[:3,:3]@[0,0,1],[0,0,-1],atol=1e-10)


def test_usb_goal_composes_frozen_model_mapping_not_identity():
    reviews=json.loads((OUTPUT/'connectors_review.json').read_text())['parts']
    row=next(x for x in reviews if x['part']=='USB_Male')
    mapping=np.asarray(row['measurements']['catalog_to_frozen_mapping']['catalog_to_frozen'])
    source=np.asarray(row['measurements']['source_female_T_catalog_male'])
    female=next(r for r in row['receivers'] if 'USB_Female' in r['name'])
    expected=np.asarray(female['board_T_mesh'])@source@np.linalg.inv(mapping)
    assert np.allclose(row['proposed_board_T_part'],expected,atol=1e-12)
    assert not np.allclose(expected[:3,:3],np.eye(3))


def test_m16_mount_correction_places_shoulder_below_board():
    reviews=json.loads((OUTPUT/'gears_nut_review.json').read_text())['parts']
    row=next(x for x in reviews if x['part']=='M16_Hex_Nut')
    audit=row['measurements']['thread_and_mount_audit']
    old=np.asarray(audit['current_mount']['board_T_screw'])
    new=np.asarray(audit['corrected_mount_candidate']['board_T_screw'])
    thickness=audit['current_mount']['board_top_m']-audit['current_mount']['board_bottom_m']
    assert abs(old[2,3]-new[2,3]-thickness)<1e-12
    assert abs(new[2,3]-.0045-audit['current_mount']['board_bottom_m'])<1e-12
    assert row['proposed_board_T_part'] is None


def test_mating_frame_chain_reconstructs_all_review_poses():
    for row in records().values():
        receiver=np.asarray(row['board_T_receiver_mating'])
        local=np.asarray(row['receiver_mating_T_part_mating_review'])
        part=np.asarray(row['part_T_part_mating'])
        assert np.allclose(receiver@local@np.linalg.inv(part),row['review_only_board_T_part'],atol=1e-10)
