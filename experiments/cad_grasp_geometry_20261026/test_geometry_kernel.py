"""Solid/through-hole/sweep regressions for the experiment's geometry checker."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider, _clip_triangle


def box(low, high):
    points = np.asarray([[x, y, z] for x in (low[0], high[0])
                         for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)
    return points, ConvexHull(points).simplices


def test_separate_touch_and_surface_crossing():
    vertices, triangles = box([-.01]*3, [.01]*3)
    collider = MeshCollider(vertices, triangles)
    assert not collider.intersects(box([.02, -.005, -.005], [.03, .005, .005])[0])
    assert collider.intersects(box([.01, -.005, -.005], [.03, .005, .005])[0])
    assert collider.intersects(box([.005, -.005, -.005], [.03, .005, .005])[0])


def test_complete_containment_in_both_directions():
    vertices, triangles = box([-.01]*3, [.01]*3)
    collider = MeshCollider(vertices, triangles)
    small = box([-.002]*3, [.002]*3)[0]
    assert collider.intersection_report(small)["kind"] == "hand_inside_target"
    assert collider.intersects(box([-.02]*3, [.02]*3)[0])
    assert collider.contains_points([[0, 0, 0], [.02, 0, 0]]).tolist() == [True, False]


def test_continuous_sweep_detects_thin_obstacle_between_clear_endpoints():
    vertices, triangles = box([-.001, -.01, -.01], [.001, .01, .01])
    collider = MeshCollider(vertices, triangles)
    hand = box([-.03, -.002, -.002], [-.02, .002, .002])[0]
    displacement = np.asarray([.06, 0, 0])
    assert not collider.intersects(hand)
    assert not collider.intersects(hand + displacement)
    assert collider.swept_intersects(hand, displacement)
    contact = collider.first_contact(hand, displacement, fraction_tolerance=1e-5)
    assert abs(contact["fraction"] - .019/.06) <= 1e-5
    assert contact["point_world_m"] is not None
    assert contact["spatial_bracket_m"] <= 6e-7


def test_native_target_hole_is_not_closed_by_convexification():
    # Four closed non-overlapping bars make a rectangular frame with a clear
    # central passage. Keep all target triangles instead of taking their hull.
    bars = [([-.02, -.02, -.01], [-.01, .02, .01]),
            ([.01, -.02, -.01], [.02, .02, .01]),
            ([-.01, -.02, -.01], [.01, -.01, .01]),
            ([-.01, .01, -.01], [.01, .02, .01])]
    vertices, triangles = [], []
    for low, high in bars:
        points, faces = box(low, high)
        triangles.append(faces + 8*len(vertices))
        vertices.append(points)
    collider = MeshCollider(np.concatenate(vertices), np.concatenate(triangles))
    hand = box([-.003]*3, [.003]*3)[0]
    assert not collider.intersects(hand)
    assert not collider.swept_intersects(hand - [0, 0, .03], [0, 0, .06])
    assert collider.intersects(hand + [.015, 0, 0])


def test_clearance_guard_is_explicit_and_conservative():
    vertices, triangles = box([-.01]*3, [.01]*3)
    collider = MeshCollider(vertices, triangles)
    hand = box([.01005, -.002, -.002], [.02, .002, .002])[0]
    assert not collider.intersects(hand)
    assert collider.intersects(hand, clearance_m=.0001)


def test_no_contact_and_initial_contact():
    vertices, triangles = box([-.01]*3, [.01]*3)
    collider = MeshCollider(vertices, triangles)
    hand = box([.02, -.002, -.002], [.03, .002, .002])[0]
    assert collider.first_contact_fraction(hand, [.01, 0, 0]) is None
    assert collider.first_contact_fraction(hand - [.02, 0, 0], [.01, 0, 0]) == 0.


def test_bvh_triangle_tests_agree_with_independent_convex_feasibility_oracle():
    rng = np.random.default_rng(26)
    vertices = rng.normal(size=(512, 3))
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * [.02, .013, .01]
    target_hull = ConvexHull(vertices)
    collider = MeshCollider(vertices, target_hull.simplices)
    assert len(collider.triangles) > 64
    for center in rng.uniform([-.027, -.02, -.017], [.027, .02, .017], size=(24, 3)):
        hand = box([-.004, -.003, -.002], [.004, .003, .002])[0] + center
        constraints = np.concatenate((target_hull.equations, ConvexHull(hand).equations))
        oracle = linprog(np.zeros(3), A_ub=constraints[:, :3], b_ub=-constraints[:, 3],
                         bounds=[(None, None)]*3, method="highs")
        assert collider.intersects(hand) == oracle.success


def test_overlapping_closed_components_are_a_union_not_parity_cancellation():
    first_v, first_f = box([-.02, -.01, -.01], [.005, .01, .01])
    second_v, second_f = box([-.005, -.008, -.008], [.02, .008, .008])
    collider = MeshCollider(np.concatenate((first_v, second_v)),
                            np.concatenate((first_f, second_f + len(first_v))))
    assert collider.topology["component_count"] == 2
    assert collider.topology["closed_component_count"] == 2
    assert collider.contains_points([[0, 0, 0], [-.015, 0, 0], [.015, 0, 0], [.03, 0, 0]]).tolist() == [True, True, True, False]
    small_hand = box([-.001]*3, [.001]*3)[0]
    assert collider.intersection_report(small_hand)["kind"] == "hand_inside_target"


def test_all_simultaneous_edge_contacts_are_available_regardless_of_face_order():
    vertices, triangles = box([-.01]*3, [.01]*3)
    hand = box([.02, -.002, .009], [.025, .002, .012])[0]
    displacement = np.asarray([-.02, 0, 0])
    normal_sets = []
    for faces in (triangles, triangles[::-1]):
        collider = MeshCollider(vertices, faces)
        contact = collider.first_contact(hand, displacement, fraction_tolerance=1e-5)
        touching = hand + displacement*contact["fraction"]
        contacts = collider.surface_contacts(touching)
        normals = np.abs(np.asarray([record["normal_world"] for record in contacts]))
        assert np.any(normals[:, 0] > .999)
        assert np.any(normals[:, 2] > .999)
        normal_sets.append(set(map(tuple, np.round(normals, 8))))
    assert normal_sets[0] == normal_sets[1]


def test_tolerance_boundary_never_extrapolates_outside_original_triangle():
    # Every original x is positive. Classifying at x<=1e-10 but intersecting
    # x=0 formerly extrapolated edges and produced negative y/z witnesses.
    triangle = np.asarray([[.5e-10, 0., 0.], [1.1e-10, 1., 0.], [1.1e-10, 0., 1.]])
    polygon = _clip_triangle(triangle, np.asarray([[1., 0., 0., 0.]]))
    assert len(polygon) == 3
    assert np.all(polygon[:, 0] >= .5e-10-1e-24)
    assert np.all(polygon[:, 0] <= 1e-10+1e-24)
    # For this triangle barycentric coordinates are (1-y-z, y, z).
    barycentric = np.c_[1-polygon[:, 1]-polygon[:, 2], polygon[:, 1:]]
    assert np.all(barycentric >= -1e-15)
    assert np.allclose(barycentric@triangle, polygon, rtol=0, atol=1e-15)
