"""Offline source-surface calibration checks; no candidate generation."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from .contact_audit import HAND,make_contact_calibration,hand_geometry_digest,_inward_planes


def hand():return json.loads(HAND.read_text(encoding='utf-8'))


def test_actual_contact_anchors_are_on_existing_distal_finger_hull_faces():
    saved=HAND.read_bytes();model=hand();before=deepcopy(model)
    calibration=make_contact_calibration(model)
    anchors=np.asarray(calibration['contact_anchors_zero_grip_site_m'])
    np.testing.assert_allclose(anchors[:,2],.007490821792003614,atol=1e-12,rtol=0)
    np.testing.assert_allclose(anchors[:,0],[.00010636337542361626,-.00010636337542361626],atol=1e-12,rtol=0)
    assert np.all(anchors[:,2]>.0044)
    for side,index in enumerate(calibration['selected_contact_patch_indices']):
        patch=calibration['contact_patches'][index]
        assert patch['component']==calibration['finger_components'][side]
        normal=np.asarray(patch['normal_grip_site'])
        assert abs(normal@anchors[side]+patch['plane_offset_m'])<1e-12
        expected=np.array([1.,0,0]) if side==0 else np.array([-1.,0,0])
        assert normal@expected>=np.cos(np.deg2rad(1))
        component=model['components'][patch['hand_component_index']]
        vertices=np.asarray(component['vertices_zero_m'])
        face=vertices[patch['source_vertex_indices']]
        assert np.max(abs(face@normal+patch['plane_offset_m']))<1e-12
        assert not component['is_pad']
        for n,offset in _inward_planes(patch['polygon_yz_m']):
            assert n@anchors[side,1:]+offset>=.00025-1e-12
    assert model==before and HAND.read_bytes()==saved


def test_physical_extent_and_old_pad_definition_are_not_changed():
    calibration=make_contact_calibration(hand())
    assert calibration['physical_fingertip_origin_grip_site_m'][2]==pytest.approx(.009249034365359496)
    assert calibration['minimum_contact_z_grip_site_m']==pytest.approx(.0044)
    assert calibration['full_finger_extent_beyond_anchor_m']==pytest.approx(.0017582125733558822)
    assert calibration['gap_zero_m']==pytest.approx(-.00021272675084723253)
    assert calibration['gap_per_opening']==pytest.approx(1.)
    assert not calibration['provenance']['physical_contact_accuracy_validated']
    assert 'groove' in calibration['provenance']['visual_geometry_caveat']


def test_calibration_hash_is_deterministic_and_bound_to_complete_saved_hand():
    model=hand();first=make_contact_calibration(model);second=make_contact_calibration(model)
    assert first==second
    digest=first.pop('calibration_sha256')
    assert digest==hashlib.sha256(json.dumps(first,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert first['hand_geometry_sha256']==hand_geometry_digest(model)
    changed=deepcopy(model)
    changed['components'][0]['vertices_zero_m'][0][0]+=.000001
    assert hand_geometry_digest(changed)!=hand_geometry_digest(model)


@pytest.mark.parametrize('kwargs',[dict(normal_angle_rad=0),dict(normal_angle_rad=np.pi),
    dict(normal_angle_rad=float('nan')),dict(edge_margin_m=-.001),dict(edge_margin_m=float('nan'))])
def test_invalid_contact_surface_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):make_contact_calibration(hand(),**kwargs)
