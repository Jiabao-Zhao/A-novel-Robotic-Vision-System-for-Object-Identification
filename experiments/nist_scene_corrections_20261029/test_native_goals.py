"""Focused offline checks for CAD registration, frames and corrected datums."""
from pathlib import Path
import json

import numpy as np

from experiments.nist_scene_corrections_20261029 import native_goals as n


def _registry():
    return json.loads((n.OUTPUT/'assembly_goals.json').read_text())


def test_native_registration_witnesses_are_independent_geometric_correspondence():
    data=json.loads((n.OUTPUT/'native_registration.json').read_text())
    tested=0
    for name,record in data['registrations'].items():
        if not record.get('witness_native_points_m'):continue
        T=np.array(record['native_to_mesh_m'])
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-10)
        assert np.linalg.det(T[:3,:3])>.999999
        mapped=n._transform(record['witness_native_points_m'],T)
        error=np.linalg.norm(mapped-np.array(record['witness_mesh_points_m']),axis=1)
        assert error.max()<=record['match_tolerance_m']
        assert record['matching_STL_vertices']>=4 and record['matched_point_rank']>=2
        tested+=1
    assert tested>=20


def test_compact_source_config_is_sufficient_without_raw_audit(monkeypatch):
    monkeypatch.setattr(n,'AUDIT',Path('unavailable-audit.json'))
    assert len(n.fixture_transforms())==15
    assert (n.HERE/'config.json').stat().st_size<100000


def test_board_face_flip_is_a_proper_rotation_and_applies_to_xy():
    R=n.BOARD_NATIVE_TO_WORKING[:3,:3]
    np.testing.assert_allclose(R@[0,0,-1],[0,0,1])
    np.testing.assert_allclose(R@[0,1,0],[0,-1,0])
    assert abs(np.linalg.det(R)-1)<1e-12
    reg=_registry()
    np.testing.assert_allclose(reg['world_T_board'],n.WORLD_T_BOARD)
    assert reg['board_top_board_m']==n.BOARD_SIZE_M[2]/2


def test_all_bolt_shoulders_meet_underside_in_the_correct_designated_holes():
    fixtures={x['name']:x for x in n.fixture_transforms()}
    expected={'M4_Screw':[191.4496135243,122.1757772952],
              'M8_Screw':[41.4496135243,47.1757772952],
              'M12_Screw':[266.4496135243,347.1757772952],
              'M16_Screw':[341.4496135243,47.1757772952]}
    for name,xy in expected.items():
        r=fixtures[name];T=np.array(r['board_T_mesh'])
        v,_=n.read_native_mesh(r['mesh_path'])
        radius=int(name.split('_')[0][1:])*.0005
        shoulder=v[np.linalg.norm(v[:,:2],axis=1)>1.05*radius,2].max()
        assert abs(T[2,3]+shoulder+n.BOARD_SIZE_M[2]/2)<1e-10
        point=(np.array(xy)-192)*.001
        np.testing.assert_allclose(T[:2,3],[point[0],-point[1]],atol=1e-9)


def test_receiver_mount_shift_preserves_source_record():
    fixtures={x['name']:x for x in n.fixture_transforms()}
    for name in ('Waterproof_Female','DSUB_Female','DB_Housing'):
        r=fixtures[name];source=np.array(r['source_board_T_mesh']);current=np.array(r['board_T_mesh'])
        expected=source.copy();expected[1,3]+=.00025
        np.testing.assert_allclose(current,expected,atol=1e-12)
        assert r['applied_mount_correction']['native_board_Y_shift_m']==-.00025


def test_all_six_chamfered_pin_tips_reach_exactly_20mm_and_protection_uses_same_plane():
    reg=_registry();count=0
    for row in reg['parts']:
        if row['part'] not in n.PIN_HOLES_MM:continue
        v,_=n.read_native_mesh(n.ROOT/row['part_mesh']);T=np.array(row['board_T_part'])
        world=n._transform(v,T)
        np.testing.assert_allclose(reg['board_top_board_m']-world[:,2].min(),.020,atol=1e-10)
        np.testing.assert_allclose(T[:3,:3]@row['axis_direction_cad'],[0,0,-1],atol=1e-10)
        p=row['protected_contact_region'];normal=np.array(p['normal_cad']);offset=p['offset_m']
        for z,blocked in [(reg['board_top_board_m']+.0001,False),
                          (reg['board_top_board_m'],True),
                          (reg['board_top_board_m']-.005,True)]:
            boardpoint=T[:3,3].copy();boardpoint[2]=z
            cadpoint=n._transform(boardpoint[None],np.linalg.inv(T))[0]
            assert (normal@cadpoint+offset>=-1e-12)==blocked
        count+=1
    assert count==6


def test_all_part_frames_protected_geometry_and_usb_mapping_are_explicit():
    reg=_registry();assert len(reg['parts'])==12
    for row in reg['parts']:
        T=np.array(row['board_T_part'])
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-9)
        assert np.linalg.det(T[:3,:3])>.999999
        np.testing.assert_allclose(row['axis_direction_board'],[0,0,-1],atol=1e-5)
        assert row['part_mesh_sha256']==n._hash(n.ROOT/row['part_mesh'])
        assert row['protected_contact_region']['type'] in ('halfspace','face_ids')
        for receiver in row['receivers']:
            assert receiver['mesh_sha256']==n._hash(n.ROOT/receiver['mesh_path'])
    usb=next(x for x in reg['parts'] if x['part']=='USB_Male')
    assert 'usb_connector_with_cable.stl' in usb['part_mesh']
    mapping=usb['evidence']['catalog_to_current_CAD_mapping']
    assert mapping['maximum_catalog_vertex_to_frozen_distance_m']<1e-8
    nut=next(x for x in reg['parts'] if x['part']=='M16_Hex_Nut')
    assert 'unverified' in nut['status'] and 'thread_warning' in nut['evidence']


def test_unavailable_native_geometry_is_not_silently_assumed_registered():
    records=n.load_native_registration()['registrations']
    assert records['KET12_Square_12mm']['native_to_mesh_m'] is None
    assert '12x12x50mm' in records['KET12_Square_12mm']['reason']
    assert records['Gear_Large']['native_to_mesh_m'] is None
    goal=next(r for r in _registry()['parts'] if r['part']=='Gear_Large')
    assert 'paired export' in goal['evidence']['goal_source']
