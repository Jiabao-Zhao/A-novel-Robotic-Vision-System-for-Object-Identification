"""Bounded, metre-scale port of Fabrica's mesh contact construction.

Only the original nearby-pair thinning function executes verbatim. Surface
sampling and all-hit ray queries use NumPy because trimesh/FCL are not installed.
World/table and finite-hand validation belong to the caller; no physics, robot,
learned model, assembly score or VLM is imported here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.cad_grasp_method_comparison_20261026.pair_reduction import (
    SOURCE, original_pair_sparsity,
)


CONFIG = dict(
    target_raw_pairs=100,
    surface_points_per_batch=100,
    maximum_batches=8,
    opposite_normal_cosine=0.95,
    inward_origin_offset_m=0.0005,
    maximum_pair_width_m=0.076,
    minimum_pair_width_m=1e-6,
    pair_sparsity_m=0.0005,
    ray_batch_size=8,
    coincident_hit_tolerance_m=1e-9,
)
FABRICA_COMMIT = '215a30fe51e59299588a5b5e417a9cb934fb393e'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _surface_samples(triangles, areas, count, rng):
    """Uniform surface area, not uniform triangle count."""
    ids = rng.choice(len(triangles), count, p=areas / areas.sum())
    uv = rng.random_sample((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    return np.einsum('ni,nij->nj', weights, triangles[ids]), ids


def _all_ray_hits(origins, directions, triangles):
    """Moller--Trumbore queries retaining every forward surface intersection.

    Deduplicate coincident triangle-edge hits per ray in distance order, with
    lowest triangle index breaking exact ties. This replaces trimesh's unique intersection-location query;
    it does not select only the first exit or the farthest intersection.
    """
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    determinant_scale = np.linalg.norm(np.cross(e1, e2), axis=1)
    ray_ids, face_ids, distances = [], [], []
    raw_count = 0
    for start in range(0, len(origins), CONFIG['ray_batch_size']):
        o = origins[start:start + CONFIG['ray_batch_size'], None, :]
        d = directions[start:start + CONFIG['ray_batch_size'], None, :]
        h = np.cross(d, e2[None])
        det = np.einsum('mj,nmj->nm', e1, h)
        mask = np.abs(det) > 1e-12 * determinant_scale[None]
        inverse = np.divide(1., det, out=np.zeros_like(det), where=mask)
        s = o - v0[None]
        u = inverse * np.einsum('nmj,nmj->nm', s, h)
        mask &= (u >= -1e-10) & (u <= 1. + 1e-10)
        q = np.cross(s, e1[None])
        v = inverse * np.sum(d * q, axis=2)
        mask &= (v >= -1e-10) & (u + v <= 1. + 1e-10)
        t = inverse * np.einsum('mj,nmj->nm', e2, q)
        mask &= t > 1e-9
        rr, ff = np.nonzero(mask)
        tt = t[rr, ff]
        raw_count += len(tt)
        # Per-ray distance order makes coincident-hit removal independent of
        # triangle enumeration except for the explicit lowest-index tie rule.
        order = np.lexsort((ff, tt, rr))
        rr, ff, tt = rr[order], ff[order], tt[order]
        keep = np.ones(len(tt), dtype=bool)
        if len(tt) > 1:
            keep[1:] = ((rr[1:] != rr[:-1]) |
                        (np.diff(tt) > CONFIG['coincident_hit_tolerance_m']))
        ray_ids.extend((rr[keep] + start).tolist())
        face_ids.extend(ff[keep].tolist())
        distances.extend(tt[keep].tolist())
    rr = np.asarray(ray_ids, dtype=np.int64)
    ff = np.asarray(face_ids, dtype=np.int64)
    tt = np.asarray(distances, dtype=np.float64)
    hits = origins[rr] + tt[:, None] * directions[rr]
    return hits, rr, ff, raw_count - len(tt)


def generate_pairs(vertices, faces, seed=0):
    """Return object-frame contacts, all raw proposals, and explicit provenance.

    The surface budget is a bounded local-trial setting, not a target number of
    useful grasps. A complete final batch is retained even if it exceeds the raw
    pair target, matching Fabrica's accumulate-then-thin ordering.
    """
    started = perf_counter()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Expected finite (V,3) vertices and valid (F,3) triangles')
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_areas = np.linalg.norm(cross, axis=1)
    nondegenerate = twice_areas > 0.
    if not nondegenerate.any():
        raise ValueError('The mesh has no positive-area triangles')
    normals = np.divide(cross, twice_areas[:, None], out=np.zeros_like(cross),
                        where=twice_areas[:, None] > 0.)
    areas = 0.5 * twice_areas
    centroid = np.average(triangles.mean(axis=1), axis=0, weights=areas)
    rng = np.random.RandomState(seed)
    pair_batches, normal_batches, face_batches = [], [], []
    raw_count = samples = all_hits = normal_rejected = width_rejected = duplicate_hits = 0
    batches = 0
    for batches in range(1, CONFIG['maximum_batches'] + 1):
        points, first_faces = _surface_samples(
            triangles, areas, CONFIG['surface_points_per_batch'], rng)
        first_normals = normals[first_faces]
        origins = points - CONFIG['inward_origin_offset_m'] * first_normals
        hits, rr, second_faces, duplicates = _all_ray_hits(origins, -first_normals, triangles)
        samples += len(points)
        all_hits += len(hits)
        duplicate_hits += duplicates
        opposition = np.einsum('ij,ij->i', normals[second_faces], -first_normals[rr])
        good_normal = opposition > CONFIG['opposite_normal_cosine']
        width = np.linalg.norm(hits - points[rr], axis=1)
        good_width = ((width > CONFIG['minimum_pair_width_m']) &
                      (width <= CONFIG['maximum_pair_width_m']))
        normal_rejected += int((~good_normal).sum())
        width_rejected += int((good_normal & ~good_width).sum())
        good = good_normal & good_width
        pair_batches.append(np.stack((points[rr[good]], hits[good]), axis=1))
        normal_batches.append(np.stack((first_normals[rr[good]], normals[second_faces[good]]), axis=1))
        face_batches.append(np.column_stack((first_faces[rr[good]], second_faces[good])))
        raw_count += int(good.sum())
        if raw_count >= CONFIG['target_raw_pairs']:
            break
    raw_pairs = np.concatenate(pair_batches, axis=0)
    raw_normals = np.concatenate(normal_batches, axis=0)
    raw_faces = np.concatenate(face_batches, axis=0)
    sparse = original_pair_sparsity(raw_pairs, CONFIG['pair_sparsity_m'])
    # The original function returns exact input rows. Keep the corresponding
    # normals/face IDs without inferring them again from a nearest-triangle query.
    first_row = {}
    for i, pair in enumerate(raw_pairs):
        first_row.setdefault(pair.tobytes(), i)
    selected = np.asarray([first_row[p.tobytes()] for p in sparse], dtype=np.int64)
    if len(selected):
        center_distance = np.linalg.norm(raw_pairs[selected].mean(axis=1) - centroid, axis=1)
        selected = selected[np.argsort(center_distance, kind='stable')]
    status = ('raw_budget_reached' if raw_count >= CONFIG['target_raw_pairs']
              else 'bounded_search_exhausted')
    metadata = dict(
        method='fabrica_derived_bounded_CAD_contacts', source_commit=FABRICA_COMMIT,
        source_repository='https://github.com/yunshengtian/Fabrica',
        source_file=str(SOURCE), source_sha256=_sha(SOURCE),
        adapter_source_sha256=_sha(__file__), config=CONFIG.copy(), seed=int(seed),
        input_and_output_frame='native_CAD', units='meters', status=status,
        surface_samples=samples, batches=batches, unique_ray_hits=all_hits,
        coincident_triangle_hits_removed=duplicate_hits,
        opposite_normal_rejections=normal_rejected, width_rejections=width_rejected,
        raw_pairs=len(raw_pairs), sparse_pairs=len(selected),
        degenerate_input_triangles=int((~nondegenerate).sum()),
        mesh_surface_centroid_m=centroid.tolist(),
        native_opposite_normal_threshold=0.95, native_pair_sparsity_m=0.0005,
        native_sampler_default_raw_target=200,
        original_pair_thinning_function_executed=True,
        original_full_sampler_executed=False,
        original_trimesh_or_FCL_executed=False,
        adaptations=[
            'NumPy area-uniform sampler and all-forward-hit ray queries replace trimesh.',
            'All positive ray intersections are tested; no farthest-only shortcut.',
            'Native centimetre lengths are converted to metres.',
            '100 accepted-pair target, 100 surface points per batch and eight-batch cap bound the trial.',
            'Native downstream Panda 76 mm span rejection is applied before raw accumulation.',
            'Coincident per-ray triangle hits within 1 nm are removed numerically.',
            'Native 10 mm ground gate and 10 mm clearance spheres are omitted in CAD coordinates; caller must apply actual finite-hand/table checks.',
            'Original endpoint thinning precedes original midpoint-to-surface-centroid ordering.',
        ],
        no_vlm=True, no_robot_execution=True, no_physics_steps=True,
        sampling_seconds=perf_counter() - started,
    )
    return dict(pairs_m=raw_pairs[selected], normals=raw_normals[selected],
                face_ids=raw_faces[selected], raw_pairs_m=raw_pairs,
                raw_normals=raw_normals, raw_face_ids=raw_faces,
                selected_raw_indices=selected, metadata=metadata)
