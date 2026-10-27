"""Focused regressions for finite CAD axes, one height and alias-only groups."""
from copy import deepcopy
import numpy as np

from . import run
from .verify_results import Checks, audit_contract, same_physical_grip


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def fixture():
    row=dict(scene='TEST',part='BOX',world_T_cad=np.eye(4).tolist())
    row['axes']=run.cad_axes(row)
    aliases=[]
    for index,sign in enumerate((1.,-1.),1):
        pose=np.eye(4)
        pose[:3,:3]=run.hand_rotation(np.array([sign,0.,0.]))
        pose[:3,3]=[0,0,.015]
        aliases.append(dict(id=f'C{index:04d}',family_id='CAD_X_R001',axis_label='CAD_X',
          cad_axis_index=0,jaw_sign=sign,cad_axis_projection_error_deg=0.,
          world_T_grip=pose.tolist(),pad_depth_m=0.,valid=True,rejection_reason=None,
          opening_m=.033,contact_opening_m=.021))
    row.update(contact_families=[dict(family_id='CAD_X_R001',axis_label='CAD_X',prepared=True,
          height_m=.015,pad_depth_m=0.,candidate_ids=[c['id'] for c in aliases],
          valid_alias_ids=[c['id'] for c in aliases])],merged_contact_regions=1,
          prepared_contact_families=1,generated_count=2,valid_count=2,grouped_count=1,
          rejection_counts={},preparation_rejection_counts={},grasps=[run.canonical_family(aliases)])
    return row,aliases


def test_axes_follow_rotated_cad_not_arbitrary_world_yaw():
    pose=np.eye(4)
    pose[:3,:3]=rz(np.deg2rad(35))
    axes=run.cad_axes(dict(world_T_cad=pose.tolist()))
    assert [a['usable'] for a in axes]==[True,True,False]
    assert np.allclose(axes[0]['closing_axis_world'],[np.cos(np.deg2rad(35)),np.sin(np.deg2rad(35)),0])
    assert np.allclose(run.hand_rotation(axes[0]['closing_axis_world'])[:,2],[0,0,-1])


def test_tilted_cylinder_does_not_silently_project_nonhorizontal_cross_axes():
    pose=np.eye(4)
    ry90=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    pose[:3,:3]=ry90@rz(np.deg2rad(58.308))
    axes=run.cad_axes(dict(world_T_cad=pose.tolist()))
    assert [a['usable'] for a in axes]==[False,False,True]
    assert np.allclose([a['projection_error_deg'] for a in axes],[31.692,58.308,0.],atol=1e-8)


def test_positive_alias_is_a_canonical_label_not_a_quality_score():
    row,aliases=fixture()
    aliases[0]['invented_quality']=-100
    aliases[1]['invented_quality']=100
    assert run.canonical_family(aliases)['jaw_sign']==1.
    assert run.canonical_family(list(reversed(aliases)))['jaw_sign']==1.
    aliases[0]['valid']=False
    assert run.canonical_family(aliases)['jaw_sign']==-1.
    aliases[1]['valid']=False
    assert run.canonical_family(aliases) is None


def test_physical_equality_ignores_sampling_depth_and_triangle_labels():
    _,aliases=fixture()
    first=aliases[0]
    second=deepcopy(first)
    second['pad_depth_m']=-.004
    second['seed_triangle']=999
    assert same_physical_grip(first,second)
    assert not same_physical_grip(first,aliases[1])


def test_pre_expansion_placement_ignores_labels_but_retains_position_and_opening():
    first=dict(position=np.array([.01,.02,.03]),axis=np.array([1.,0.,0.]),opening=.03)
    second=dict(first,pad_depth_m=-.004,seed_triangle=999)
    assert run.same_placement(first,second)
    second['position']=first['position']+np.array([.001,0.,0.])
    assert not run.same_placement(first,second)
    second['position']=first['position'].copy()
    second['opening']+=.006
    assert not run.same_placement(first,second)


def test_different_checked_approach_openings_merge_only_after_final_contact():
    row,first=fixture()
    second=deepcopy(first)
    for index,candidate in enumerate(second,3):
        candidate.update(id=f'C{index:04d}',family_id='CAD_X_R002',pad_depth_m=-.004,
                         opening_m=.027,contact_opening_m=.0210005)
        candidate['world_T_grip'][1][3]+=.0000002
    groups=run.collapse_equivalent_families([run.canonical_family(first),run.canonical_family(second)])
    assert len(groups)==1
    group=groups[0]
    assert group['id']=='C0003'
    assert group['source_family_ids']==['CAD_X_R001','CAD_X_R002']
    assert set(group['grouped_candidate_ids'])=={'C0001','C0002','C0003','C0004'}
    family=deepcopy(row['contact_families'][0])
    family.update(family_id='CAD_X_R002',pad_depth_m=-.004,
                  candidate_ids=['C0003','C0004'],valid_alias_ids=['C0003','C0004'])
    row['contact_families'].append(family)
    row.update(merged_contact_regions=2,prepared_contact_families=2,generated_count=4,
               valid_count=4,grasps=groups)
    checks=Checks()
    audit_contract(row,first+second,run.CONFIG,checks)
    assert checks.failed_count==0,checks.failures


def test_distinct_final_hand_positions_are_not_collapsed():
    _,aliases=fixture()
    translated=deepcopy(aliases)
    for candidate in translated:
        candidate['world_T_grip'][0][3]+=.001
        candidate['family_id']='CAD_X_R002'
    groups=run.collapse_equivalent_families([run.canonical_family(aliases),run.canonical_family(translated)])
    assert len(groups)==2


def test_contract_accepts_exactly_one_height_with_two_checked_aliases():
    row,candidates=fixture()
    checks=Checks()
    audit_contract(row,candidates,run.CONFIG,checks)
    assert checks.failed_count==0,checks.failures


def test_contract_rejects_hidden_second_hand_height():
    row,candidates=fixture()
    candidates[1]['world_T_grip'][2][3]+=.001
    checks=Checks()
    audit_contract(row,candidates,run.CONFIG,checks)
    assert checks.failed_count>0
    assert any('one physical hand height' in value for value in checks.failures)


def test_contract_rejects_an_undeclared_oblique_yaw():
    row,candidates=fixture()
    rotation=run.hand_rotation(np.array([np.cos(.1),np.sin(.1),0.]))
    candidates[0]['world_T_grip'][:3]=np.c_[rotation,[0,0,.015]].tolist()
    checks=Checks()
    audit_contract(row,candidates,run.CONFIG,checks)
    assert checks.failed_count>0
    assert any('signed declared axis' in value for value in checks.failures)


def test_minimum_table_height_solves_plane_for_all_vertices_and_both_aliases():
    component=dict(vertices_zero_m=[[x,y,z] for x in (-.005,.005)
                   for y in (-.006,.006) for z in (-.007,.007)],
                   translation_per_opening=[0.,0.,0.])
    plane=np.array([.01,-.02,1.,-.002])
    plane/=np.linalg.norm(plane[:3])
    midpoint=np.array([.1,-.2,.02])
    x=np.array([1.,0.,0.])
    height=run.minimum_table_height(midpoint,x,.03,[component],plane)
    for sign in (1.,-1.):
        world=np.asarray(component['vertices_zero_m'])@run.hand_rotation(sign*x).T
        world+=np.array([midpoint[0],midpoint[1],height])
        assert (world@plane[:3]+plane[3]).min()>=run.old.CONFIG['table_clearance_m']-1e-12
