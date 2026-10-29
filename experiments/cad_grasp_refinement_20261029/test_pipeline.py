"""Cross-module selection tests using explicit synthetic contact candidates."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_pipeline import request_fixture
from experiments.cad_grasp_selection_20261029.test_fingertip import tip_candidate
from experiments.cad_grasp_selection_20261029.sampling_regions import compile_sampling_region
from . import pipeline


def request():
    result=request_fixture()
    result['protected_contact_region']=dict(type='face_ids', protected_face_ids=[])
    result['sampling_contact_region']=compile_sampling_region(
        result['vertices_cad_m'],result['faces'],result['protected_contact_region'])
    return result


@pytest.mark.parametrize('variant',['previous','refined'])
def test_both_variants_use_complete_fingertip_pose_validation_and_same_priority(monkeypatch,variant):
    data=request()
    candidate=tip_candidate(data['hand'])
    data['world_T_cad_pickup']=np.eye(4)
    data['world_T_cad_final']=np.eye(4)
    def generated(*args,**kwargs):
        assert kwargs['max_poses']==100
        return [deepcopy(candidate)],dict(proposed_count=1)
    if variant=='refined':
        monkeypatch.setattr(pipeline,'generate_candidates',generated)
    else:
        monkeypatch.setattr(pipeline.previous_sampler,'generate_candidates',generated)
    diagnostics=[]
    result=pipeline.select_grasp(data,sampler_variant=variant,diagnostics=diagnostics)
    assert result['status']=='selected',result
    selected=result['selected_grasp']
    assert selected['validity']['valid']
    assert selected['validity']['diagnostics']
    assert all(value['status']=='pass' for value in selected['validity']['diagnostics'].values())
    np.testing.assert_allclose(selected['world_T_hand_pickup'],selected['cad_T_hand'])
    np.testing.assert_allclose(selected['world_T_grip_site_final'],selected['cad_T_grip_site'])
    assert [stage['criterion'] for stage in result['selection_stages']]==['r_tau','r_surface','theta_down_rad']
    assert result['proposed_count']==result['valid_count']==len(diagnostics)==1


def test_missing_labels_and_simulator_pose_flag_never_generate_fallback(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Invalid input must fail before proposal generation')
    monkeypatch.setattr(pipeline,'generate_candidates',forbidden)
    for data in [request_fixture(),dict(request(),simulator_object_poses_read=True)]:
        result=pipeline.select_grasp(data)
        assert result['status']=='unavailable_inputs'
        assert result['selected_grasp'] is None and result['proposed_count']==0


@pytest.mark.parametrize('variant',['previous','refined'])
def test_zero_pose_budget_reports_no_valid_sampled_grasp(variant):
    result=pipeline.select_grasp(request(),sampler_variant=variant,max_poses=0,surface_budget=4)
    assert result['status']=='no_valid_grasp_within_budget',result
    assert result['selected_grasp'] is None and result['proposed_count']==0
