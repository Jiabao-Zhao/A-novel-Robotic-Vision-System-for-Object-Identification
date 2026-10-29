"""Synthetic complete-geometry regressions; no NIST bank or robot execution."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.test_geometry import fixture, box, clear_region
from experiments.cad_grasp_selection_20261029.geometry import GeometryValidator as PreviousValidator
from experiments.cad_grasp_selection_20261029.sampler import pose_from_pair
from experiments.cad_grasp_selection_20261029.test_fingertip import tip_candidate
from .geometry import GeometryValidator, CATEGORIES
from .diagnostics import evaluate_candidate, summarize
from .test_contact_model import cylinder, model_for


def protected(points, *, contact_polygons, contact_face_ids):
    result = clear_region(points, contact_polygons=contact_polygons, contact_face_ids=contact_face_ids)
    result['eligible'][:] = False
    return result


@pytest.mark.parametrize('case', ['valid', 'table', 'palm', 'sweep', 'receiver', 'protected', 'opening', 'anchor'])
def test_no_model_keeps_previous_validity(case):
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    if case == 'table':
        kwargs['table_plane'] = [0, 0, 1, .02]
    elif case == 'palm':
        kwargs['hand']['components'][-1]['vertices_zero_m'] = box([-.003]*3, [.003]*3)[0].tolist()
    elif case == 'sweep':
        kwargs['hand']['components'].append(dict(name='moving_body', is_pad=False,
            vertices_zero_m=box([-.051, -.002, -.002], [-.049, .002, .002])[0].tolist(),
            translation_per_opening=[2., 0, 0]))
        candidate['pre_opening_m'] = .04
    elif case == 'receiver':
        vertices, faces = box([-.03, -.02, -.04], [.03, .02, -.015])
        kwargs['receivers'] = [dict(name='near', vertices=vertices, faces=faces, world_T_mesh=np.eye(4))]
    elif case == 'protected':
        kwargs['protected_contact_check'] = protected
    elif case == 'opening':
        candidate['pre_opening_m'] = .081
    elif case == 'anchor':
        candidate['contacts_cad_m'][0][1] += .001
    previous = PreviousValidator(**kwargs).validate(candidate)
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['valid'] == previous['valid'], (previous, result)
    assert set(result['diagnostics']) == set(CATEGORIES)
    if case == 'valid':
        assert result['valid']
        assert all(value['status'] == 'pass' for value in result['diagnostics'].values())
        np.testing.assert_allclose(result['world_T_hand_pickup'], previous['world_T_hand_pickup'])
    else:
        assert not result['valid']


def test_independent_failures_are_all_exposed_in_one_traversal():
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    kwargs['hand']['components'][-1]['vertices_zero_m'] = box([-.003]*3, [.003]*3)[0].tolist()
    kwargs['table_plane'] = [0, 0, 1, 0]
    kwargs['protected_contact_check'] = protected
    vertices, faces = box([-.04]*3, [.04]*3)
    kwargs['receivers'] = [dict(name='blocking', vertices=vertices, faces=faces, world_T_mesh=np.eye(4))]
    validator = GeometryValidator(**kwargs)
    counts = {'surface': 0, 'target': 0}
    surface, target = validator.target.surface_contacts, validator.target.intersects
    def counted_surface(*args, **kwargs):
        counts['surface'] += 1
        return surface(*args, **kwargs)
    def counted_target(*args, **kwargs):
        counts['target'] += 1
        return target(*args, **kwargs)
    validator.target.surface_contacts, validator.target.intersects = counted_surface, counted_target
    result = validator.validate(candidate)
    for category in ['target_closing_collision', 'table_clearance', 'final_receiver_collision', 'protected_surfaces']:
        assert result['diagnostics'][category]['status'] == 'fail', result
    assert counts == {'surface': 2, 'target': 3}
    assert evaluate_candidate(validator, candidate, result) is result['diagnostics']
    assert counts == {'surface': 2, 'target': 3}
    assert summarize([dict(validity=result)])['table_clearance']['fail'] == 1


def test_unavailable_category_remains_visible_after_physical_failure():
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    kwargs['table_plane'] = [0, 0, 1, 0]
    def unavailable_after_probe(points, **polygons):
        result = clear_region(points, **polygons)
        if len(points): result['status'] = 'unavailable_region_query'
        return result
    kwargs['protected_contact_check'] = unavailable_after_probe
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'unavailable_input'
    assert result['diagnostics']['table_clearance']['status'] == 'fail'
    assert result['diagnostics']['protected_surfaces']['status'] == 'unavailable'
    assert result['diagnostics']['target_closing_collision']['status'] == 'pass'


def test_flat_face_tolerance_stays_one_degree_and_other_checks_still_run():
    kwargs, _ = fixture()
    theta = np.deg2rad(2.)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    kwargs['vertices_cad'] = kwargs['vertices_cad']@rotation.T
    candidate = pose_from_pair([[-.01/np.cos(theta), 0, 0], [.01/np.cos(theta), 0, 0]], [0, 0, 1], kwargs['hand'])
    candidate['pre_opening_m'] = .03
    kwargs['table_plane'] = [0, 0, 1, 0]
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['diagnostics']['surface_normals']['status'] == 'fail'
    assert result['diagnostics']['table_clearance']['status'] == 'fail'
    assert result['diagnostics']['target_closing_collision']['status'] in ('pass', 'fail')
    assert result['tolerances']['normal_angle_rad'] == pytest.approx(np.deg2rad(1.))


def cylinder_case(interior=False):
    kwargs, _ = fixture()
    vertices, faces = cylinder(n=72, radius=.006)
    kwargs.update(vertices_cad=vertices, faces=faces)
    # An off-center point on one original flat facet has a smooth radial
    # normal different from that facet's true support normal. Using it as a
    # tangent plane cuts the adjacent original vertex and must remain invalid.
    angle = np.pi/(2*72) if interior else 0.
    radius = .006*np.cos(np.pi/72)/np.cos(angle-np.pi/72) if interior else .006
    axis = np.array([np.cos(angle), np.sin(angle), 0.])
    center = np.array([0., 0., -.01])
    candidate = pose_from_pair([center-radius*axis, center+radius*axis], [0, 0, 1], kwargs['hand'])
    candidate['pre_opening_m'] = .03
    return kwargs, candidate, model_for(vertices, faces)


def test_verified_round_support_can_pass_without_relaxing_flats_or_collision():
    kwargs, candidate, model = cylinder_case()
    previous = GeometryValidator(**kwargs).validate(candidate)
    result = GeometryValidator(**kwargs, contact_model=model).validate(candidate)
    assert not previous['valid']
    assert previous['diagnostics']['surface_normals']['status'] == 'fail'
    assert result['valid'], result
    assert result['diagnostics']['target_closing_collision']['status'] == 'pass'


def test_analytic_normal_cannot_hide_raw_mesh_penetration():
    kwargs, candidate, model = cylinder_case(interior=True)
    before = kwargs['vertices_cad'].tobytes(), kwargs['faces'].tobytes()
    result = GeometryValidator(**kwargs, contact_model=model).validate(candidate)
    assert result['diagnostics']['surface_normals']['status'] == 'pass', result
    assert result['diagnostics']['target_closing_collision']['status'] == 'fail', result
    assert not result['valid']
    assert (kwargs['vertices_cad'].tobytes(), kwargs['faces'].tobytes()) == before


def test_foreign_mesh_model_and_unknown_candidate_frame_are_unavailable():
    kwargs, candidate, model = cylinder_case()
    kwargs['vertices_cad'] = kwargs['vertices_cad']*2
    result = GeometryValidator(**kwargs, contact_model=model).validate(candidate)
    assert result['diagnostics']['candidate_input']['status'] == 'unavailable'
    assert result['diagnostics']['table_clearance']['status'] == 'not_evaluated'
    kwargs, _ = fixture()
    candidate = tip_candidate(kwargs['hand'])
    candidate['hand_pose_reference'] = 'unknown'
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'unavailable_input'
    assert result['diagnostics']['candidate_input']['status'] == 'unavailable'
