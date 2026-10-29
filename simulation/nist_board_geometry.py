"""Finished ATB1 plate geometry derived from the unchanged official STL.

NIST's fabrication instructions specify four finished round holes. This helper
reconstructs those nominal dimensions, retaining all other cut boundaries and
the native metric plate frame. Some source cuts are already oversized; this is
a nominal simulation model, not a drill-toolpath simulation. Downloaded CAD and
converted source meshes are never edited.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import triangulate, unary_union


# Locations in the original 0..384 mm plate export, before centering.
ROUND_PIN_HOLES = (
    (4, (191.45, 347.176), 4.1),
    (8, (116.45, 347.176), 8.1),
    (12, (41.45, 122.176), 12.2),
    (16, (266.45, 122.176), 16.5),
)
REVISION = 1


def read_binary_stl(path):
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84 + 50 * count:
        raise ValueError(f'Expected binary STL: {path}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84, count=count)['v']
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices.astype(float), inverse.reshape(-1, 3)


def write_binary_stl(path, vertices, faces):
    triangles = np.asarray(vertices, float)[np.asarray(faces, int)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms < 1e-15):
        raise ValueError('Degenerate finished-plate triangle')
    normals /= norms[:, None]
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    records = np.zeros(len(faces), dtype=dtype)
    records['normal'], records['v'] = normals, triangles
    header = b'NIST finished holes; derived geometry, original CAD unchanged'.ljust(80, b' ')
    Path(path).write_bytes(header + len(records).to_bytes(4, 'little') + records.tobytes())


def plate_loops(vertices, faces):
    """Return ordered XY cut loops from one complete flat plate face."""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    top = vertices[:, 2].max()
    planar = faces[np.all(np.isclose(vertices[faces, 2], top, atol=1e-9), axis=1)]
    counts = Counter(tuple(sorted((int(a), int(b))))
                     for face in planar for a, b in zip(face, np.roll(face, -1)))
    graph = defaultdict(list)
    for (a, b), count in counts.items():
        if count == 1:
            graph[a].append(b)
            graph[b].append(a)
        elif count != 2:
            raise ValueError('Nonmanifold planar plate face')
    if not graph or any(len(v) != 2 for v in graph.values()):
        raise ValueError('Incomplete plate cut boundary')
    unseen, loops = set(graph), []
    while unseen:
        first = min(unseen)
        loop, previous, current = [], None, first
        while True:
            loop.append(current)
            unseen.discard(current)
            neighbours = graph[current]
            following = neighbours[0] if neighbours[0] != previous else neighbours[1]
            previous, current = current, following
            if current == first:
                break
            if current not in unseen:
                raise ValueError('Plate boundary does not form a simple loop')
        loops.append(vertices[loop, :2])
    loops.sort(key=lambda loop: Polygon(loop).area, reverse=True)
    return loops


def finished_plate(vertices, faces, circle_sides=128):
    """Build a closed plate with documented finished round-pin holes.

    Circular polygons are circumscribed: their minimum inscribed diameter equals
    the documented drill size. Other openings and plate thickness are preserved.
    """
    if circle_sides < 32 or circle_sides % 4:
        raise ValueError('Use a multiple of four, at least 32 circle sides')
    vertices = np.asarray(vertices, float)
    loops = plate_loops(vertices, faces)
    exterior, holes = loops[0], list(loops[1:])
    records, used = [], set()
    for pin_mm, source_xy_mm, diameter_mm in ROUND_PIN_HOLES:
        expected = (np.asarray(source_xy_mm) - 192.) / 1000
        candidates = []
        for index, loop in enumerate(holes):
            fit = np.linalg.lstsq(np.c_[2 * loop, np.ones(len(loop))], np.sum(loop * loop, axis=1), rcond=None)[0]
            center = fit[:2]
            radii = np.linalg.norm(loop - center, axis=1)
            if len(loop) >= 12 and np.ptp(radii) < 5e-5:
                candidates.append((np.linalg.norm(center - expected), index, center, float(np.mean(radii))))
        distance, index, center, raw_radius = min(candidates, key=lambda item: item[0])
        if distance > 2e-4 or index in used:
            raise ValueError(f'Cannot identify official {pin_mm} mm pin hole')
        used.add(index)
        radius = diameter_mm / 2000 / np.cos(np.pi / circle_sides)
        angle = np.arange(circle_sides) * 2 * np.pi / circle_sides
        holes[index] = center + radius * np.c_[np.cos(angle), np.sin(angle)]
        records.append(dict(pin_diameter_mm=pin_mm, finished_drill_diameter_mm=diameter_mm,
                            source_centered_xy_m=center.tolist(), raw_fitted_diameter_mm=raw_radius * 2000,
                            modeled_minimum_inscribed_diameter_mm=diameter_mm,
                            polygon_sides=circle_sides))
    polygon = orient(Polygon(exterior, holes), sign=1.)
    if not polygon.is_valid:
        raise ValueError('Finished plate polygon is invalid')
    planar = [t for t in triangulate(polygon) if polygon.covers(t)]
    union = unary_union(planar)
    missing = polygon.symmetric_difference(union).area
    if missing > 1e-12:
        raise ValueError(f'Plate triangulation leaves {missing} square metres uncovered')
    lower, upper = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    points, triangles = [], []
    def triangle(points3):
        offset = len(points)
        points.extend(points3)
        triangles.append([offset, offset + 1, offset + 2])
    for t in planar:
        xy = np.asarray(orient(t, sign=1.).exterior.coords)[:3]
        triangle(np.c_[xy, np.full(3, upper)])
        triangle(np.c_[xy[::-1], np.full(3, lower)])
    for ring in [polygon.exterior, *polygon.interiors]:
        xy = np.asarray(ring.coords)
        for a, b in zip(xy[:-1], xy[1:]):
            triangle([[*a, lower], [*b, lower], [*b, upper]])
            triangle([[*a, lower], [*b, upper], [*a, upper]])
    points, inverse = np.unique(np.asarray(points), axis=0, return_inverse=True)
    faces_out = inverse[np.asarray(triangles)]
    counts = Counter(tuple(sorted((int(a), int(b)))) for face in faces_out
                     for a, b in zip(face, np.roll(face, -1)))
    if any(count != 2 for count in counts.values()):
        raise ValueError('Finished plate extrusion is not closed')
    return points, faces_out, dict(finished_round_holes=records,
        thickness_m=upper - lower, unchanged_other_openings=len(holes) - 4,
        planar_triangulation_missing_area_m2=float(missing),
        frame='Original centered plate STL XYZ in metres; native CAD unchanged',
        circle_model='Circumscribed regular polygons; stated drill size is minimum inscribed diameter')


def prepare_finished_board(mesh_dir, output_dir=None):
    mesh_dir = Path(mesh_dir)
    output_dir = Path(output_dir) if output_dir else mesh_dir.parent / 'finished_board_20261029'
    source = mesh_dir / 'GMC_Laser_Plate_Virtual_m.stl'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    target = output_dir / 'GMC_Laser_Plate_Finished_m.stl'
    manifest = output_dir / 'finished_board.json'
    if target.exists() and manifest.exists():
        cached = json.loads(manifest.read_text(encoding='utf-8'))
        if (cached.get('revision') == REVISION and cached.get('source_sha256') == source_hash
                and cached.get('derived_sha256') == hashlib.sha256(target.read_bytes()).hexdigest()):
            return target, cached
    output_dir.mkdir(parents=True, exist_ok=True)
    v, f = read_binary_stl(source)
    points, triangles, record = finished_plate(v, f)
    write_binary_stl(target, points, triangles)
    record.update(revision=REVISION, source=str(source), source_sha256=source_hash,
                  derived=str(target), derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  source_mesh_unchanged=True,
                  source_reference='https://www.nist.gov/document/assemblyinstructionsv4docx')
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    return target, record
