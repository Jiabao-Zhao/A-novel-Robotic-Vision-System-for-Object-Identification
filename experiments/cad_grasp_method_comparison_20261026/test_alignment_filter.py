from copy import deepcopy
import numpy as np
import pytest
from .alignment_filter import normal_angles,passes_alignment,retain_groups,group_revalidated


def test_opposite_signed_pad_normals_both_align_and_jaw_sign_is_irrelevant():
    normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
    angles,signed=normal_angles(normals,[1.,0.,0.])
    assert np.array_equal(angles,[0.,0.])
    assert np.array_equal(signed,[1.,-1.])
    reversed_angles,reversed_signed=normal_angles(normals,[-1.,0.,0.])
    assert np.array_equal(reversed_angles,angles)
    assert np.array_equal(reversed_signed,-signed)


def test_both_contacts_must_pass_not_just_the_best_one():
    angle=np.deg2rad(4.)
    angles,_=normal_angles([[1,0,0],[-np.cos(angle),np.sin(angle),0]],[1,0,0])
    assert np.allclose(angles,[0,4],atol=1e-10)
    evaluation=dict(angles_deg=angles.tolist())
    assert not passes_alignment(evaluation,1.)
    assert not passes_alignment(evaluation,3.)
    assert passes_alignment(evaluation,5.)


def test_degenerate_normal_is_explicit_error():
    with pytest.raises(ValueError):
        normal_angles([[1,0,0],[0,0,0]],[1,0,0])


def test_later_passing_alias_is_retained_when_parent_representative_fails():
    by_id={f'U{i}':dict(id=f'U{i}',family_id=f'P{i}',jaw_sign=1.,opening_m=.03,world_T_grip=np.eye(4).tolist(),contacts_world_m=[[0,0,0],[.01,0,0]]) for i in range(3)}
    parent=dict(id='U0',grouped_candidate_ids=['U0','U1','U2'],source_family_ids=['P0','P1','P2'])
    evaluations={'U0':dict(angles_deg=[2.,2.]),'U1':dict(angles_deg=[.5,.6]),'U2':dict(angles_deg=[.7,.8])}
    before=deepcopy(by_id)
    kept=retain_groups([parent],by_id,evaluations,1.)
    assert len(kept)==1 and kept[0]['id']=='U1'
    assert kept[0]['grouped_candidate_ids']==['U1','U2']
    assert [c['parent_id'] for c in kept[0]['jaw_aliases']]==['U1','U2']
    assert kept[0]['world_T_grip']==by_id['U1']['world_T_grip']
    assert kept[0]['contacts_world_m']==by_id['U1']['contacts_world_m']
    assert by_id==before


def test_main_revalidation_grouping_keeps_later_passing_alias():
    candidates=[dict(id='U0',pair_index=1,family_id='PAIR0001',valid=False,jaw_sign=1.,opening_m=.03),
                dict(id='U1',pair_index=1,family_id='PAIR0001',valid=True,jaw_sign=-1.,opening_m=.03)]
    groups=group_revalidated(candidates)
    assert len(groups)==1 and groups[0]['id']=='U1'
    assert groups[0]['grouped_candidate_ids']==['U1']
