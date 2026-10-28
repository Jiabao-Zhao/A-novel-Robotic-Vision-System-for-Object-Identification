"""Offline geometry checks; no simulator, learned model, or pickup execution."""
import numpy as np

from experiments.fabrica_candidates_20261027.sampler import generate_pairs


def _box(size, center=(0., 0., 0.)):
    vertices = np.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float) * np.asarray(size) / 2 + np.asarray(center)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    faces = np.asarray([(a, b, c) for a, b, c, d in quads] +
                       [(a, c, d) for a, b, c, d in quads])
    return vertices, faces


def _cylinder(radius, length, segments=32):
    angles = np.arange(segments) * 2 * np.pi / segments
    ring = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    bottom = np.column_stack((ring, np.full(segments, -length / 2)))
    top = np.column_stack((ring, np.full(segments, length / 2)))
    vertices = np.vstack((bottom, top, [0, 0, -length / 2], [0, 0, length / 2]))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([(i, j, segments + j), (i, segments + j, segments + i),
                      (2 * segments, j, i),
                      (2 * segments + 1, segments + i, segments + j)])
    return vertices, np.asarray(faces)


def _assert_contacts_on_recorded_faces(vertices, faces, result, raw=False):
    """Check triangle membership independently, without casting another ray."""
    pairs = result['raw_pairs_m' if raw else 'pairs_m']
    normals = result['raw_normals' if raw else 'normals']
    face_ids = result['raw_face_ids' if raw else 'face_ids']
    for point, normal, face_id in zip(pairs.reshape(-1, 3),
                                      normals.reshape(-1, 3), face_ids.ravel()):
        triangle = vertices[faces[face_id]]
        basis = (triangle[1:] - triangle[0]).T
        weights, _, _, _ = np.linalg.lstsq(basis, point - triangle[0], rcond=None)
        reconstructed = triangle[0] + basis @ weights
        np.testing.assert_allclose(point, reconstructed, atol=1e-10, rtol=0)
        assert weights.min() >= -1e-8
        assert weights.sum() <= 1 + 1e-8
        face_normal = np.cross(basis[:, 0], basis[:, 1])
        face_normal /= np.linalg.norm(face_normal)
        np.testing.assert_allclose(normal, face_normal, atol=1e-10, rtol=0)


def test_saved_contacts_and_normals_belong_to_recorded_box_faces():
    vertices, faces = _box((.016, .024, .05))
    result = generate_pairs(vertices, faces, seed=17)
    assert len(result['pairs_m']) > 0
    assert result['metadata']['status'] == 'raw_budget_reached'
    assert result['metadata']['units'] == 'meters'
    assert result['metadata']['input_and_output_frame'] == 'native_CAD'
    _assert_contacts_on_recorded_faces(vertices, faces, result)
    _assert_contacts_on_recorded_faces(vertices, faces, result, raw=True)

    closing = result['pairs_m'][:, 1] - result['pairs_m'][:, 0]
    widths = np.linalg.norm(closing, axis=1)
    np.testing.assert_allclose(closing / widths[:, None],
                               -result['normals'][:, 0], atol=1e-10, rtol=0)
    opposition = np.sum(-result['normals'][:, 0] * result['normals'][:, 1], axis=1)
    assert np.all(opposition > .95)
    assert np.all((widths > 0) & (widths <= .076))
    np.testing.assert_array_equal(result['pairs_m'],
                                   result['raw_pairs_m'][result['selected_raw_indices']])


def test_8_mm_round_pin_has_side_contacts_without_planar_face_requirement():
    vertices, faces = _cylinder(.004, .05)
    result = generate_pairs(vertices, faces, seed=11)
    _assert_contacts_on_recorded_faces(vertices, faces, result, raw=True)
    side = np.all(np.abs(result['raw_normals'][:, :, 2]) < 1e-10, axis=1)
    assert side.any(), 'Round side contacts must survive the opposite-normal check'
    pairs = result['raw_pairs_m'][side]
    widths = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    # The polygonal mesh is inscribed in the physical 8 mm diameter cylinder.
    assert np.all(widths >= .008 * np.cos(np.pi / 32) - 1e-10)
    assert np.all(widths <= .008 + 1e-10)
    np.testing.assert_allclose(pairs[:, 0, 2], pairs[:, 1, 2], atol=1e-10, rtol=0)


def test_all_opposed_forward_surfaces_are_considered_before_thinning():
    left, left_faces = _box((.008, .01, .012), (-.01, 0, 0))
    right, right_faces = _box((.008, .01, .012), (.01, 0, 0))
    vertices = np.vstack((left, right))
    faces = np.vstack((left_faces, right_faces + len(left)))
    result = generate_pairs(vertices, faces, seed=23)
    pairs = result['raw_pairs_m']
    # From the outside left face, one ray exits the first box and later exits
    # the second. Both exits have normals opposite to the starting face.
    starts_on_left = ((pairs[:, 0, 0] < -.013) &
                      (result['raw_normals'][:, 0, 0] < -.99))
    pairs = pairs[starts_on_left]
    assert len(pairs) > 0
    _, groups = np.unique(pairs[:, 0], axis=0, return_inverse=True)
    widths = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    assert any(np.any(widths[groups == group] < .01) and
               np.any(widths[groups == group] > .02)
               for group in np.unique(groups))
    _assert_contacts_on_recorded_faces(vertices, faces, result, raw=True)


def test_oversized_part_returns_no_contacts_after_finite_search():
    vertices, faces = _box((.1, .11, .12))
    result = generate_pairs(vertices, faces, seed=7)
    assert result['pairs_m'].shape == (0, 2, 3)
    assert result['raw_pairs_m'].shape == (0, 2, 3)
    assert result['metadata']['status'] == 'bounded_search_exhausted'
    assert result['metadata']['batches'] == 8
    assert result['metadata']['surface_samples'] == 800
    assert result['metadata']['width_rejections'] > 0


def test_raw_contact_geometry_moves_with_the_cad_rigid_transform():
    vertices, faces = _box((.016, .024, .05))
    axis = np.asarray([1., 2., 3.])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = .63
    rotation = (np.eye(3) + np.sin(angle) * skew +
                (1 - np.cos(angle)) * (skew @ skew))
    translation = np.asarray([.3, -.2, .1])
    original = generate_pairs(vertices, faces, seed=17)
    moved = generate_pairs(vertices @ rotation.T + translation, faces, seed=17)
    np.testing.assert_allclose(moved['raw_pairs_m'],
                               original['raw_pairs_m'] @ rotation.T + translation,
                               atol=1e-9, rtol=0)
    np.testing.assert_allclose(moved['raw_normals'],
                               original['raw_normals'] @ rotation.T,
                               atol=1e-9, rtol=0)
    np.testing.assert_array_equal(moved['raw_face_ids'], original['raw_face_ids'])
