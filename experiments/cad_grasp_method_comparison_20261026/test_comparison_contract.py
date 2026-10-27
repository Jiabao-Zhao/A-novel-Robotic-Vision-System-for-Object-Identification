"""Focused regression checks for independent comparison audit arithmetic."""
from copy import deepcopy
import numpy as np
from experiments.cad_grasp_geometry_20261026.verify_results import Checks
from experiments.cad_grasp_method_comparison_20261026 import verify_comparison as audit


def fixture():
    pair=np.array([[[0.,0.,.02],[.01,0.,.021]]])
    candidates=[]
    for adapter in ('midpoint','panda_table'):
        for sign in (1.,-1.):
            candidates.append(dict(id=f'U{len(candidates)+1:05d}',pair_index=1,
                adapter=adapter,jaw_sign=sign,family_id='PAIR0001',
                protocols=[adapter+'_proj15'],valid=False,rejection_reason='open_pad_hits_target'))
    row=dict(method='one_budget512',scene='N00',part='X',generated_count=4,
             primary_generated_count=2,valid_count=0,all_adapter_valid_count=0,
             grouped_count=0,grasps=[],status='no_feasible_upstream_pair_in_common_adapter',
             protocol_summary={p:audit.summarize(candidates,p) for p in audit.PROTOCOLS},
             rejection_counts={'open_pad_hits_target':2})
    return pair,candidates,row,dict(primary_protocol='panda_table_proj15')


def test_all_adapter_and_primary_counts_have_different_meanings():
    pairs,candidates,row,config=fixture();checks=Checks()
    audit.audit_counts(row,candidates,pairs,config,checks)
    assert checks.failed_count==0
    row['primary_generated_count']=4;checks=Checks()
    audit.audit_counts(row,candidates,pairs,config,checks)
    assert any('primary generated count' in value for value in checks.failures)


def test_protocol_summary_does_not_invent_one_degree_candidates():
    pairs,candidates,row,config=fixture();checks=Checks()
    row['protocol_summary']['panda_table_proj1']['checked']=2
    audit.audit_counts(row,candidates,pairs,config,checks)
    assert any('panda_table_proj1 summary' in value for value in checks.failures)


def test_grouping_cannot_drop_later_primary_passes():
    pairs,candidates,row,config=fixture()
    candidates[2].update(valid=True,rejection_reason=None)
    row.update(valid_count=1,all_adapter_valid_count=1,status='feasible_grasps_available',
               protocol_summary={p:audit.summarize(candidates,p) for p in audit.PROTOCOLS},
               rejection_counts={'open_pad_hits_target':1})
    checks=Checks();audit.audit_counts(row,candidates,pairs,config,checks)
    assert any('groups partition' in value for value in checks.failures)


def test_raw_world_conversion_keeps_tilted_endpoints_and_input_bytes():
    raw=np.array([[[0.,0.,0.],[.01,0.,.001]]]);before=raw.copy()
    pose=np.eye(4);pose[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    pose[:3,3]=[.2,.3,.4]
    config=dict(maximum_closing_projection_deg=15.,pair_sparsity_m=.005)
    sparse,nwidth,nproject=audit.replay_pairs(raw,pose,.067,config)
    assert nwidth==nproject==1
    assert np.array_equal(raw,before)
    assert np.allclose(sparse,raw@pose[:3,:3].T+pose[:3,3])
    assert sparse[0,0,2]!=sparse[0,1,2]


def test_sparsity_suppresses_only_when_both_endpoints_are_near():
    pairs=np.array([[[0,0,0],[.02,0,0]],[[.001,0,0],[.021,0,0]],
                    [[.001,0,0],[.04,0,0]]],float)
    saved=audit.sparse_replay(pairs,.005)
    assert np.array_equal(saved,pairs[[0,2]])


def test_frame_is_fixed_down_for_both_jaw_assignments():
    x=np.array([.6,.8,0.])
    for sign in (-1,1):
        rotation=audit.hand_rotation(sign*x)
        assert np.allclose(rotation.T@rotation,np.eye(3))
        assert np.isclose(np.linalg.det(rotation),1.)
        assert np.array_equal(rotation[:,2],[0,0,-1])
