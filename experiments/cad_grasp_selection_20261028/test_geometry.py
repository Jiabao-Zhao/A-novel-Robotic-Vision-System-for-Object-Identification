"""Offline synthetic solids; never launch a NIST trial or a simulator."""
import copy
import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_selection_20261028.geometry import GeometryValidator, KNOWN_HAND_FRAME, pad_calibration


def clear_region(points, *, contact_polygons, contact_face_ids):
    # This synthetic fixture explicitly declares the complete target surface
    # clear; this is not an inference from sampled polygon vertices.
    assert len(contact_polygons) == len(contact_face_ids)
    return dict(status='available', eligible=np.ones(len(points), bool),
                polygon_eligible=np.ones(len(contact_polygons), bool))


def box(low, high):
    points = np.asarray([[x, y, z] for x in (low[0], high[0])
                         for y in (low[1], high[1]) for z in (low[2], high[2])], float)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    for index, face in enumerate(faces):
        triangle = points[face]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        if normal @ hull.equations[index, :3] < 0:
            faces[index, [1, 2]] = faces[index, [2, 1]]
    return points, faces


def fixture():
    # Zero opening has opposing pad centers at x=0; +Z is approach. The palm
    # is behind both contact pads in -Z. Geometry is linear in joint command.
    left, _ = box([-.003, -.004, -.004], [0., .004, .004])
    right, _ = box([0., -.004, -.004], [.003, .004, .004])
    palm, _ = box([-.025, -.006, -.025], [.025, .006, -.018])
    hand = dict(frame=KNOWN_HAND_FRAME, opening_limits_m=[0., .08], components=[
        dict(name='left_pad', is_pad=True, vertices_zero_m=left.tolist(), translation_per_opening=[-.5, 0, 0]),
        dict(name='right_pad', is_pad=True, vertices_zero_m=right.tolist(), translation_per_opening=[.5, 0, 0]),
        dict(name='palm', is_pad=False, vertices_zero_m=palm.tolist(), translation_per_opening=[0, 0, 0])])
    target, faces = box([-.01, -.01, -.01], [.01, .01, .01])
    receiver, receiver_faces = box([1, 1, 1], [1.1, 1.1, 1.1])
    kwargs = dict(vertices_cad=target, faces=faces, hand=hand,
                  pickup_T=np.eye(4), final_T=np.eye(4), table_plane=[0., 0., 1., .1],
                  protected_contact_check=clear_region,
                  receivers=[dict(name='far', vertices=receiver, faces=receiver_faces, world_T_mesh=np.eye(4))])
    candidate = dict(cad_T_hand=np.eye(4), contacts_cad_m=[[-.01, 0, 0], [.01, 0, 0]],
                     holding_opening_m=.02, pre_opening_m=.03)
    return kwargs, candidate


def test_pad_calibration_and_center_reconstruction():
    kwargs, candidate = fixture()
    calibration = pad_calibration(kwargs['hand'])
    assert calibration['gap_zero_m'] == 0
    assert calibration['gap_per_opening'] == 1
    assert np.allclose(calibration['centers_zero_m']+.02*calibration['centers_per_opening'], candidate['contacts_cad_m'])
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['valid'], result
    assert result['checks']['pad_center_reconstruction_error_m'] == [0., 0.]
    bad = copy.deepcopy(candidate)
    bad['cad_T_hand'][1, 3] = .001
    rejected = GeometryValidator(**kwargs).validate(bad)
    assert not rejected['valid'] and 'pad centers' in rejected['reasons'][0]


def test_valid_side_and_upward_grasps_have_no_downward_gate():
    kwargs, candidate = fixture()
    # Identity approach is +Z: explicitly upward, and is still geometrically
    # valid. Rotate the hand 90 degrees about X for a side approach as well.
    validator = GeometryValidator(**kwargs)
    assert validator.validate(candidate)['valid']
    side = copy.deepcopy(candidate)
    side['cad_T_hand'][:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    result = validator.validate(side)
    assert result['valid'], result
    assert np.allclose(result['approach_world'], [0, -1, 0])


def test_round_side_tangent_line_does_not_require_flat_contact_area():
    kwargs, candidate = fixture()
    # A finely faceted round pin lies along Y. The two pads touch its tangent
    # lines at x=+/-radius, rather than a positive-area flat CAD face.
    angles = np.linspace(0., 2*np.pi, 256, endpoint=False)
    points = np.asarray([[.01*np.cos(a), y, .01*np.sin(a)]
                         for y in (-.01, .01) for a in angles])
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    for index, face in enumerate(faces):
        tri = points[face]
        if np.cross(tri[1]-tri[0], tri[2]-tri[0]) @ hull.equations[index, :3] < 0:
            faces[index, [1, 2]] = faces[index, [2, 1]]
    kwargs.update(vertices_cad=points, faces=faces)
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['valid'], result


def test_fine_four_mm_radius_pin_queries_actual_tangent_contact_not_nearby_band():
    kwargs, candidate = fixture()
    # A 1um expanded query band on this pin would include normals over one
    # degree from tangency, although those faces do not actually touch a pad.
    angles = np.linspace(0., 2*np.pi, 1024, endpoint=False)
    points = np.asarray([[.004*np.cos(a), y, .004*np.sin(a)]
                         for y in (-.01, .01) for a in angles])
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    for index, face in enumerate(faces):
        tri = points[face]
        if np.cross(tri[1]-tri[0], tri[2]-tri[0]) @ hull.equations[index, :3] < 0:
            faces[index, [1, 2]] = faces[index, [2, 1]]
    kwargs.update(vertices_cad=points, faces=faces)
    candidate.update(contacts_cad_m=[[-.004, 0, 0], [.004, 0, 0]],
                     holding_opening_m=.008, pre_opening_m=.018)
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['valid'], result


def test_small_flat_face_inside_larger_pad_ignores_perpendicular_boundary_only_faces():
    kwargs, candidate = fixture()
    kwargs['vertices_cad'], kwargs['faces'] = box([-.01, -.002, -.002], [.01, .002, .002])
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['valid'], result


def test_unavailable_preflight_status_and_fractional_face_indices_are_not_silently_accepted():
    kwargs, candidate = fixture()
    def missing_region(points, **polygons):
        result = clear_region(points, **polygons)
        result['status'] = 'available_but_missing_region'
        return result
    kwargs['protected_contact_check'] = missing_region
    validator = GeometryValidator(**kwargs)
    assert validator.input_errors
    assert validator.validate(candidate)['status'] == 'unavailable_input'
    kwargs, candidate = fixture()
    kwargs['faces'] = kwargs['faces'].astype(float)+.5
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'unavailable_input' and 'integers' in result['reasons'][0]


def test_whole_palm_collision_and_complete_containment_are_rejected():
    kwargs, candidate = fixture()
    kwargs['hand']['components'][-1]['vertices_zero_m'] = box([-.003]*3, [.003]*3)[0].tolist()
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and 'palm' in result['reasons'][0]


def test_closing_sweep_detects_obstacle_between_clear_endpoints():
    kwargs, candidate = fixture()
    # A non-pad finger component crosses the part while closing, although it
    # is completely outside the part at both endpoints.
    moving, _ = box([-.051, -.002, -.002], [-.049, .002, .002])
    kwargs['hand']['components'].append(dict(name='moving_body', is_pad=False,
        vertices_zero_m=moving.tolist(), translation_per_opening=[2., 0, 0]))
    candidate['pre_opening_m'] = .04
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and 'moving_body' in result['reasons'][0]
    assert result['closing_discrete_samples'] == 0


def test_table_checks_full_palm_and_final_receiver_uses_goal_transform():
    kwargs, candidate = fixture()
    kwargs['table_plane'] = [0., 0., 1., .02]
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and 'table' in result['reasons'][0]
    kwargs, candidate = fixture()
    kwargs['final_T'][0, 3] = .1
    receiver, faces = box([.075, -.006, -.025], [.125, .006, -.018])
    kwargs['receivers'] = [dict(vertices=receiver, faces=faces, world_T_mesh=np.eye(4))]
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'] and 'receiver' in result['reasons'][0]


def test_missing_inputs_and_legacy_unavailable_regions_are_not_passes():
    kwargs, candidate = fixture()
    kwargs['protected_contact_check'] = lambda points, **polygons: dict(status='unavailable_zero_engagement', eligible=np.ones(len(points), bool))
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'unavailable_input'
    kwargs, candidate = fixture()
    kwargs['receivers'] = []
    assert GeometryValidator(**kwargs).validate(candidate)['status'] == 'unavailable_input'
    kwargs, candidate = fixture()
    kwargs['hand']['frame'] = 'unknown frame'
    assert GeometryValidator(**kwargs).validate(candidate)['status'] == 'unavailable_input'


def test_reserved_surfaces_and_open_target_are_explicitly_rejected():
    kwargs, candidate = fixture()
    kwargs['protected_contact_check'] = lambda points, **polygons: dict(status='available', eligible=np.zeros(len(points), bool), polygon_eligible=np.ones(len(polygons['contact_polygons']), bool))
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'invalid' and 'protected' in result['reasons'][0]
    kwargs, candidate = fixture()
    kwargs['faces'] = kwargs['faces'][:-1]
    assert GeometryValidator(**kwargs).validate(candidate)['status'] == 'unavailable_input'


def test_closing_limit_and_pad_nonpenetration_not_just_contact_normals():
    kwargs, candidate = fixture()
    candidate['pre_opening_m'] = .081
    assert not GeometryValidator(**kwargs).validate(candidate)['valid']
    kwargs, candidate = fixture()
    # Contact centers remain on their assigned faces, but another target bar
    # protrudes into the left pad body: the whole-pad check must reject it.
    protrusion, extra_faces = box([-.012, -.002, -.002], [-.009, .002, .002])
    count = len(kwargs['vertices_cad'])
    kwargs['vertices_cad'] = np.concatenate((kwargs['vertices_cad'], protrusion))
    kwargs['faces'] = np.concatenate((kwargs['faces'], extra_faces+count))
    result = GeometryValidator(**kwargs).validate(candidate)
    assert not result['valid'], result


def test_point_only_protected_checker_cannot_certify_polygon_interiors():
    kwargs, candidate = fixture()
    kwargs['protected_contact_check'] = lambda points, **polygons: dict(
        status='available', eligible=np.ones(len(points), bool))
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'unavailable_input'
    assert 'complete contact polygon' in result['reasons'][0]


def test_protected_island_inside_contact_polygon_is_rejected():
    kwargs, candidate = fixture()
    def has_reserved_island(points, *, contact_polygons, contact_face_ids):
        # All sampled vertices are clear, but the region classifier determines
        # that a protected interior island intersects one complete polygon.
        polygons_clear = np.ones(len(contact_polygons), bool)
        if len(polygons_clear):
            polygons_clear[0] = False
        return dict(status='available', eligible=np.ones(len(points), bool),
                    polygon_eligible=polygons_clear)
    kwargs['protected_contact_check'] = has_reserved_island
    result = GeometryValidator(**kwargs).validate(candidate)
    assert result['status'] == 'invalid' and 'protected' in result['reasons'][0]
