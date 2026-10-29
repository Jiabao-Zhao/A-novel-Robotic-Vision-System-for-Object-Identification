"""Offline saved-data reporting checks; no NIST trial or collision query."""
import copy
import json

import numpy as np
import pytest

from experiments.cad_grasp_refinement_20261029 import report as module
from experiments.cad_grasp_selection_20261028.test_gallery import records
from experiments.cad_grasp_selection_20261028.test_geometry import fixture
from experiments.cad_grasp_selection_20261029.sampler import pose_from_pair


def test_table_audit_keeps_all_components_and_both_opening_endpoints():
    kwargs, _ = fixture()
    hand = copy.deepcopy(kwargs['hand'])
    # A moving solid outside either pad can determine table clearance.
    hand['components'].append(dict(name='distal_finger', is_pad=False,
        vertices_zero_m=[[x,y,z] for x in (-.001,.001) for y in (-.001,.001)
                         for z in (-.05,-.04)], translation_per_opening=[0.,0.,-1.]))
    candidate = dict(cad_T_grip_site=np.eye(4).tolist(),
                     holding_opening_m=.01, pre_opening_m=.03)
    value, witness = module.candidate_table_clearance(candidate, np.eye(4),
                                                     np.array([0.,0.,1.,.1]), hand)
    assert value == pytest.approx(.02)
    assert witness == dict(component='distal_finger', opening_state='pre_opening')


def test_reporting_does_not_call_no_available_grasp_success(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    hand, registry, observed = tmp_path/'hand.json', tmp_path/'registry.json', tmp_path/'observed.json'
    for path in (hand,registry,observed):
        module.dump(path,{})
    monkeypatch.setattr(module, 'HAND', hand)
    monkeypatch.setattr(module, 'REGISTRY', registry)
    historical = tmp_path/'historical'
    historical.mkdir()
    module.dump(historical/'summary.json', dict(placements=96,
        outcome_counts={'selected':4}, elapsed_seconds=116.04))
    monkeypatch.setattr(module, 'HISTORICAL', historical)
    folder = tmp_path/'comparison'
    folder.mkdir()
    module.dump(folder/'launch_manifest.json', dict(settings={'max_poses':100},
        source_sha256={}, protected_sha256={}, hand_model_sha256=module.sha(hand),
        corrected_registry_sha256=module.sha(registry), observed_manifest='observed.json',
        observed_manifest_sha256=module.sha(observed)))
    summaries = {}
    for variant in module.VARIANTS:
        root = folder/variant
        root.mkdir()
        data = records()
        for row in data:
            row.update(seed=123, wall_seconds=.01, independent_check_counts={},
                       table_plane_world=[0.,0.,1.,0.])
            row['selection'] = dict(status='unavailable_inputs', selected_grasp=None,
                                    proposed_count=0, valid_count=0)
            case = root/'cases'/row['scene']/row['part']
            case.mkdir(parents=True)
            module.dump(case/'candidate_diagnostics.json', [])
        (root/'task_inputs').mkdir()
        for part in {r['part'] for r in data}:
            module.dump(root/'task_inputs'/f'{part}.json', {'source':{'same':'input'}})
        summary = dict(selected_placements=0, proposed_poses=0, valid_poses=0,
            workers=1, elapsed_wall_seconds=1., summed_case_seconds=.96)
        module.dump(root/'results.json', data)
        module.dump(root/'summary.json', summary)
        summaries[variant] = summary
    module.dump(folder/'comparison.json', {'variants':summaries})
    paths = module.report(folder)
    verification = json.loads(paths['verification'].read_text(encoding='utf-8'))
    assert verification['status'] == 'passed'
    assert verification['unavailable_is_success'] is False
    assert verification['variants']['previous']['totals']['selected'] == 0
    assert verification['variants']['refined']['per_part']['Part_00']['unavailable_placements'] == 8
    assert verification['historical_reference']['controlled_comparison'] is False
    assert verification['collision_engine_rerun'] is False
    assert 'not pickup or assembly success' in paths['report'].read_text(encoding='utf-8')
    assert len(paths['placement_counts'].read_text().splitlines()) == 193


def test_nonempty_audit_recomputes_saved_contacts_criteria_and_world_frames(tmp_path, monkeypatch):
    kwargs, _ = fixture()
    request = dict(vertices_cad_m=kwargs['vertices_cad'], faces=kwargs['faces'],
        axis_point_cad_m=[0.,0.,0.], axis_direction_cad=[0.,0.,1.],
        world_T_cad_final=np.eye(4))
    monkeypatch.setattr(module, 'build_request', lambda *a, **k:dict(status='available',request=request))
    plane = np.array([0.,0.,1.,.1])
    candidate = pose_from_pair([[-.01,0.,0.],[.01,0.,0.]], [0.,0.,-1.], kwargs['hand'])
    candidate.update(candidate_id=3, valid=True)
    candidate['pre_opening_m'] = candidate['holding_opening_m']+.001
    candidate['criteria'] = module.raw_criteria(candidate, module.reference_from_request(request),
                                               np.eye(4), plane)
    for kind in ('hand','grip_site'):
        for place in ('pickup','final'):
            candidate[f'world_T_{kind}_{place}'] = candidate[f'cad_T_{kind}']
    minimum, _ = module.candidate_table_clearance(candidate, np.eye(4), plane, kwargs['hand'])
    candidate['validity'] = dict(status='valid', tolerances={'table_clearance_m':.0001},
        diagnostics={'table':dict(status='pass',table_minimum_clearance_m=minimum)})
    reference = {k:v.tolist() if isinstance(v,np.ndarray) else v
                 for k,v in module.reference_from_request(request).items()}
    data = records()
    for index,row in enumerate(data):
        row.update(table_plane_world=plane.tolist(), wall_seconds=.01, independent_check_counts={})
        row['selection'] = dict(status='no_valid_grasp_within_budget', selected_grasp=None,
                                proposed_count=0, valid_count=0)
        case = tmp_path/'cases'/row['scene']/row['part']
        case.mkdir(parents=True)
        module.dump(case/'candidate_diagnostics.json', [candidate] if index == 0 else [])
    data[0]['selection'].update(status='selected', selected_grasp=candidate,
        proposed_count=1,valid_count=1,reference=reference)
    data[0]['independent_check_counts'] = {'table':{'pass':1}}
    module.dump(tmp_path/'results.json', data)
    module.dump(tmp_path/'summary.json',dict(selected_placements=1,proposed_poses=1,valid_poses=1))
    audit, _, _, _ = module.audit_variant(tmp_path, kwargs['hand'], {}, {})
    assert audit['totals']['all_candidate_anchor_pairs_reconstructed'] == 1
    assert audit['totals']['valid_candidate_criteria_and_world_poses_recomputed'] == 1
    assert audit['per_part']['Part_00']['table_minima_range_m'] == pytest.approx([minimum,minimum])
    assert audit['maximum_anchor_reconstruction_error_m'] < 1e-12
    bad = copy.deepcopy(candidate)
    bad['criteria']['q_surface_m'] += .001
    case = tmp_path/'cases'/data[0]['scene']/data[0]['part']/'candidate_diagnostics.json'
    module.dump(case, [bad])
    with pytest.raises(AssertionError, match='criterion differs'):
        module.audit_variant(tmp_path, kwargs['hand'], {}, {})
