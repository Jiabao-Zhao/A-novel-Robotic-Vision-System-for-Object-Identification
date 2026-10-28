"""Offline checks for fixed geometry normalization and strict priorities."""
import numpy as np
import pytest

from experiments.cad_grasp_selection_20261028.criteria import (
    build_reference, evaluate_criteria, select_priority,
)


def box():
    vertices = np.array([[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.],
                         [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]])
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7], [0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return vertices, faces


def record(cid, tau, surface, angle):
    return dict(candidate_id=cid, valid=True,
                criteria=dict(r_tau=tau, r_surface=surface, theta_down_rad=angle))


def test_area_weighted_centroid_is_not_vertex_mean():
    vertices = np.array([[0.,0.,0.], [2.,0.,0.], [0.,2.,0.],
                         [10.,0.,0.], [11.,0.,0.], [10.,1.,0.]])
    faces = np.array([[0,1,2],[3,4,5]])
    result = build_reference(vertices, faces, [0,0,0], [0,0,1])
    expected = (2 * vertices[:3].mean(axis=0) + .5 * vertices[3:].mean(axis=0)) / 2.5
    assert np.allclose(result['surface_centroid_cad_m'], expected)
    assert not np.allclose(expected, vertices.mean(axis=0))


def test_mesh_subdivision_preserves_surface_centroid():
    vertices, faces = box()
    original = build_reference(vertices, faces, [0,0,0], [0,0,1])
    triangle = vertices[faces[0]]
    subdivided_vertices = np.vstack((vertices, triangle.mean(axis=0)))
    a,b,c = faces[0]
    subdivided_faces = np.vstack((faces[1:], [[a,b,8],[b,c,8],[c,a,8]]))
    result = build_reference(subdivided_vertices, subdivided_faces, [0,0,0], [0,0,1])
    assert np.allclose(result['surface_centroid_cad_m'], original['surface_centroid_cad_m'])
    assert result['surface_area_m2'] == pytest.approx(original['surface_area_m2'])


def test_normalization_bounds_for_random_surface_midpoints_and_fixed_reference():
    vertices, faces = box()
    reference = build_reference(vertices, faces, [.3,-.2,.7], [0,0,1])
    rng = np.random.RandomState(18)
    for _ in range(40):
        ids = rng.randint(len(faces), size=2)
        weights = rng.dirichlet([1.,1.,1.], size=2)
        contacts = np.einsum('ij,ijk->ik', weights, vertices[faces[ids]])
        result = evaluate_criteria(contacts, reference, [1,0,0], [0,0,1])
        assert 0 <= result['r_tau'] <= 1
        assert 0 <= result['r_surface'] <= 1
        assert result['theta_down_rad'] == pytest.approx(np.pi / 2)
    assert reference['D_tau_m'] == pytest.approx(max(np.linalg.norm(np.cross(
        vertices - [.3,-.2,.7], [0,0,1]), axis=1)))


def test_approach_preference_has_no_validity_gate():
    vertices, faces = box()
    reference = build_reference(vertices, faces, [0,0,0], [0,0,1])
    contacts = [[-1,0,0],[1,0,0]]
    down = evaluate_criteria(contacts, reference, [0,0,-1], [0,0,1])
    side = evaluate_criteria(contacts, reference, [0,1,0], [0,0,1])
    up = evaluate_criteria(contacts, reference, [0,0,1], [0,0,1])
    assert down['theta_down_rad'] == 0
    assert side['theta_down_rad'] == pytest.approx(np.pi / 2)
    assert up['theta_down_rad'] == pytest.approx(np.pi)
    assert side['r_tau'] == down['r_tau'] == up['r_tau']


def test_zero_denominator_is_inactive_and_never_nan():
    reference = dict(surface_centroid_cad_m=[0,0,0], axis_point_cad_m=[0,0,0],
                     axis_direction_cad=[0,0,1], D_tau_m=0., D_surface_m=0.)
    result = evaluate_criteria([[0,0,-1],[0,0,1]], reference, [1,0,0], [0,0,1])
    assert result['r_tau'] == result['r_surface'] == 0
    assert result['zero_tau_criterion'] and result['zero_surface_criterion']
    with pytest.raises(ValueError, match='zero CAD bound'):
        evaluate_criteria([[1,0,-1],[1,0,1]], reference, [1,0,0], [0,0,1])


def test_out_of_bound_value_is_rejected_not_clipped():
    vertices, faces = box()
    reference = build_reference(vertices, faces, [0,0,0], [0,0,1])
    with pytest.raises(ValueError, match='fixed CAD bound'):
        evaluate_criteria([[8,0,0],[10,0,0]], reference, [0,0,-1], [0,0,1])


def test_primary_priority_cannot_be_overridden_by_other_criteria():
    candidates = [record(9, .1, 1., np.pi), record(1, .10001, 0., 0.)]
    result = select_priority(candidates)
    assert result['selected_candidate']['candidate_id'] == 9


def test_secondary_then_tertiary_priorities_and_deterministic_final_tie():
    candidates = [record(9, .2, .4, 0), record(4, .2, .3, .7),
                  record(2, .2, .3, .7), record(8, .2, .3, 1.)]
    result = select_priority(candidates)
    assert result['selected_candidate']['candidate_id'] == 2
    assert result['final_tied_candidate_ids'] == [2,4]
    assert result['stages'][1]['remaining_candidate_ids'] == [2,4,8]
    assert select_priority(list(reversed(candidates)))['selected_candidate']['candidate_id'] == 2


def test_ties_are_relative_to_stage_minimum_not_neighbor_chain():
    candidates = [record(1, .1, .5, 0), record(2, .10000000008, .4, 0),
                  record(3, .10000000016, 0., 0)]
    result = select_priority(candidates, dict(r_tau=1e-10))
    assert result['stages'][0]['remaining_candidate_ids'] == [1,2]
    assert result['selected_candidate']['candidate_id'] == 2


def test_empty_candidates_return_explicit_failure_without_fallback():
    result = select_priority([])
    assert result['status'] == 'no_valid_grasp'
    assert result['selected_candidate'] is None
    assert result['message'] == 'No valid grasp found within the sampling budget.'


@pytest.mark.parametrize('candidates', [
    [record(1, np.nan, 0, 0)], [record(1, 0, 0, 0), record(1, 0, 0, 0)],
    [dict(record(1,0,0,0), valid=False)], [record(1, 1.1, 0, 0)],
])
def test_invalid_selection_inputs_are_explicit_errors(candidates):
    with pytest.raises(ValueError):
        select_priority(candidates)


def test_invalid_axis_and_mesh_are_explicit_errors():
    vertices, faces = box()
    with pytest.raises(ValueError, match='unit length'):
        build_reference(vertices, faces, [0,0,0], [0,0,2])
    with pytest.raises(ValueError, match='positive finite total area'):
        build_reference(np.zeros((3,3)), np.array([[0,1,2]]), [0,0,0], [0,0,1])
