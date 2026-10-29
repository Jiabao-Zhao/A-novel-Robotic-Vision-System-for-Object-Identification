from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import _solid_collider, transform_points
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from . import sampler
from .contact_audit import make_contact_calibration

ROOT = Path(__file__).resolve().parents[2]


def box(lo, hi):
    from scipy.spatial import ConvexHull
    v = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])],float)
    h = ConvexHull(v)
    f = h.simplices.copy()
    for i, face in enumerate(f):
        a,b,c = v[face]
        if np.cross(b-a,c-a)@h.equations[i,:3] < 0:
            f[i] = face[::-1]
    return v,f


def test_prefix_bisection_finds_first_contact_even_if_end_is_clear():
    # A small jaw passes completely through a thin target; endpoint-only
    # bisection would miss the collision at intermediate opening.
    v,f = box([.019,-.002,-.002],[.021,.002,.002])
    target = _solid_collider(v,f,'test')
    c = dict(name='jaw',is_pad=False,vertices_zero_m=(box([-.001,-.001,-.001],[.001,.001,.001])[0]).tolist(),
             translation_per_opening=[.5,0,0])
    assert not target.intersects(np.array(c['vertices_zero_m']))
    q,info = sampler._jaw_contact(target,[c],np.eye(3),np.zeros(3),.08)
    assert info['status']=='first_contact_bracketed'
    assert abs(q-.044)<2e-9
    assert info['command_bracket_width_m']<=sampler.FIT_COMMAND_TOLERANCE_M


def test_fit_places_real_finger_witnesses_not_old_pad_anchors():
    from experiments.cad_grasp_refinement_20261029.fingertip_audit import audit_case
    from experiments.cad_grasp_refinement_20261029.sampler import pose_from_pair
    from experiments.cad_grasp_geometry_20261026.gallery import read_stl
    hand=json.loads((ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
    before=deepcopy(hand)
    v,f=read_stl(ROOT/'outputs/simulation/nist_task_board_1/assets/meshes/RGOCG8-50_8mm_m.stl')
    # Native pin axis is Z; make the gripper approach perpendicular to it.
    raw=pose_from_pair(np.array([[-.004,0,0],[.004,0,0]]),np.array([0.,-1.,0.]),hand)
    raw.update(candidate_id=0,pair_id=0,contact_face_ids=[0,0])
    fitted=sampler.fit_candidate(raw,_solid_collider(v,f,'pin'),hand,make_contact_calibration(hand))
    assert fitted['first_contact_fit']['status']=='fitted'
    points=np.array(fitted['actual_contacts_hand_m'])
    assert np.all(points[:,2]>.0044)
    assert np.all(points[:,2]<.00925)
    np.testing.assert_allclose(transform_points(points,fitted['cad_T_grip_site']),fitted['contacts_cad_m'],atol=1e-12)
    old=fingertip_calibration(hand)['contact_anchors_zero_grip_site_m']
    assert np.min(points[:,2]-old[:,2])>.002
    assert hand==before


def test_fit_fails_if_maximum_opening_already_intersects():
    v,f=box([-.1,-.1,-.1],[.1,.1,.1])
    component=dict(name='jaw',is_pad=False,vertices_zero_m=box([-.001,-.001,-.001],[.001,.001,.001])[0].tolist(),translation_per_opening=[.5,0,0])
    q,info=sampler._jaw_contact(_solid_collider(v,f,'large'),[component],np.eye(3),np.zeros(3),.08)
    assert q is None and info['status']=='maximum_opening_not_clear'


def test_cache_key_is_bound_to_complete_target_and_clearance():
    v,f=box([0,0,0],[1,1,1])
    calibration={'calibration_sha256':'fixed-hand-and-contact-definition'}
    key=sampler._bank_key(v,f,[],calibration,.001)
    assert key!=sampler._bank_key(v+[0,0,.1],f,[],calibration,.001)
    assert key!=sampler._bank_key(v,f[::-1],[],calibration,.001)
    assert key!=sampler._bank_key(v,f,[],calibration,.002)
