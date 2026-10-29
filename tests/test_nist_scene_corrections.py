"""Offline plate and XML construction checks; no simulator/model loading."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

from simulation import nist_board_geometry as geometry
from simulation import nist_task_board_1 as scene

ROOT = Path(__file__).resolve().parents[1]
MESHES = ROOT / 'outputs/simulation/nist_task_board_1/assets/meshes'


@pytest.fixture(scope='module')
def original_plate():
    source = MESHES / 'GMC_Laser_Plate_Virtual_m.stl'
    if not source.is_file():
        pytest.skip('Official plate mesh is required for this offline CAD check')
    return geometry.read_binary_stl(source)


def test_finished_plate_has_documented_drill_sizes_and_preserves_other_cuts(original_plate):
    vertices, faces = original_plate
    points, triangles, record = geometry.finished_plate(vertices, faces)
    before = geometry.plate_loops(vertices, faces)
    after = geometry.plate_loops(points, triangles)
    assert len(before) == len(after) == 35
    assert record['unchanged_other_openings'] == 30
    assert record['planar_triangulation_missing_area_m2'] == pytest.approx(0, abs=1e-12)
    assert np.ptp(points[:, 2]) == pytest.approx(np.ptp(vertices[:, 2]), abs=1e-12)
    changed = []
    for item in record['finished_round_holes']:
        center = np.asarray(item['source_centered_xy_m'])
        loop = min(after[1:], key=lambda p: np.linalg.norm(np.asarray(Polygon(p).centroid.coords[0]) - center))
        edges = np.roll(loop, -1, axis=0) - loop
        rel = center - loop
        distance = np.abs(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]) / np.linalg.norm(edges, axis=1)
        assert distance.min() * 2000 == pytest.approx(item['finished_drill_diameter_mm'], abs=1e-8)
        changed.append(Polygon(loop))
    # All other cut loops are byte-for-byte coordinate-preserved in the rebuilt
    # surface, irrespective of ring orientation or first-vertex ordering.
    for loop in before[1:]:
        polygon = Polygon(loop)
        if any(polygon.centroid.distance(p.centroid) < 5e-5 for p in changed):
            continue
        match = min(after[1:], key=lambda p: polygon.centroid.distance(Polygon(p).centroid))
        assert polygon.symmetric_difference(Polygon(match)).area < 1e-15
    counts = Counter(tuple(sorted((int(a), int(b)))) for face in triangles
                     for a, b in zip(face, np.roll(face, -1)))
    assert set(counts.values()) == {2}
    solid = points[triangles]
    volume = np.einsum('ij,ij->i', solid[:, 0], np.cross(solid[:, 1], solid[:, 2])).sum() / 6
    expected_volume = Polygon(after[0], after[1:]).area * np.ptp(vertices[:, 2])
    assert volume == pytest.approx(expected_volume, rel=1e-10, abs=1e-15)


def test_derived_plate_does_not_overwrite_official_mesh(tmp_path):
    source = MESHES / 'GMC_Laser_Plate_Virtual_m.stl'
    if not source.is_file():
        pytest.skip('Official plate mesh required')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    path, record = geometry.prepare_finished_board(MESHES, tmp_path)
    assert path.parent == tmp_path
    assert record['source_sha256'] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    # Serialized float32 STL still has complete closed rings and finite bounds.
    v, f = geometry.read_binary_stl(path)
    assert len(geometry.plate_loops(v, f)) == 35
    assert np.isfinite(v).all()
    assert geometry.prepare_finished_board(MESHES, tmp_path)[1]['derived_sha256'] == record['derived_sha256']


def test_full_component_rotation_and_finished_plate_used_by_scene_xml(monkeypatch, tmp_path):
    source = MESHES / 'GMC_Laser_Plate_Virtual_m.stl'
    if not source.is_file():
        pytest.skip('Official plate mesh required')
    rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, [.01, -.02, .03]
    module = SimpleNamespace(fixture_transforms=lambda mesh_dir: [dict(
        name='Waterproof_Female', instance='nist_fixture_14', board_T_mesh=pose)])
    monkeypatch.setitem(sys.modules, 'experiments.nist_scene_corrections_20261029.native_goals', module)
    real_prepare = geometry.prepare_finished_board
    monkeypatch.setattr(geometry, 'prepare_finished_board', lambda mesh_dir: real_prepare(mesh_dir, tmp_path))
    model = SimpleNamespace(asset=ET.Element('asset'), worldbody=ET.Element('worldbody'))
    dimensions = {scene.BOARD_FILE: [.384, .384, .0089916], 'Waterproof_Female': [.04, .06, .04]}
    scene.add_board_scene(model, MESHES, dimensions, parts=())
    body = model.worldbody.find("body[@name='nist_fixture_14']")
    position = np.fromstring(body.get('pos'), sep=' ')
    q = np.fromstring(body.get('quat'), sep=' ')
    np.testing.assert_allclose(position, pose[:3, 3] + [0, .065, .0244958], atol=1e-10)
    recovered = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(recovered @ [0, 1, 0], [0, 0, 1], atol=1e-9)
    plate = model.worldbody.find("body[@name='nist_board']/body[@name='nist_board_mesh']")
    assert plate.get('quat') == '0 1 0 0'
    mesh = model.asset.find(f"mesh[@name='nist_{scene.BOARD_FILE}']")
    assert Path(mesh.get('file')).name == 'GMC_Laser_Plate_Finished_m.stl'
    assert len(plate.findall('geom')) > 100  # visual and actual hole-preserving prisms
    assert not model.worldbody.findall('.//freejoint')


def test_component_transform_rejects_reflection():
    pose = np.eye(4)
    pose[2, 2] = -1
    with pytest.raises(ValueError, match='proper rigid'):
        scene._quaternion_wxyz(pose)
    pose = np.eye(4)
    pose[0, 0] = 1.000001
    with pytest.raises(ValueError, match='proper rigid'):
        scene._quaternion_wxyz(pose)


def test_source_layout_assignments_use_official_roles():
    fixtures = {r[0]: r for r in scene.FIXTURES}
    assert fixtures['M4_Screw'][1:3] == (191.45, 122.176)
    assert fixtures['M12_Screw'][1:3] == (266.45, 347.176)
    assert fixtures['USB_Female'][1:3] == (41.45, 272.176)
    assert fixtures['DSUB_Female'][1:3] == fixtures['DB_Housing'][1:3] == (266.45, 197.176)
    assert [fixtures[f'M{d}_Screw'][3] for d in [4, 8, 12, 16]] == [-12.9916, -16.9916, -20.9916, -24.9916]


def test_actual_registered_fixtures_build_xml_with_upward_openings_and_underboard_bolt_heads():
    config = ROOT / 'experiments/nist_scene_corrections_20261029/config.json'
    if not config.is_file():
        pytest.skip('Native geometry registration must be prepared before this integration check')
    from experiments.nist_scene_corrections_20261029.native_goals import fixture_transforms

    provenance = json.loads((MESHES.parent / 'provenance.json').read_text(encoding='utf-8'))
    dimensions = provenance['dimensions_m']
    fixtures = fixture_transforms(MESHES)
    assert len(fixtures) == len(scene.FIXTURES) == 15
    model = SimpleNamespace(asset=ET.Element('asset'), worldbody=ET.Element('worldbody'))
    scene.add_board_scene(model, MESHES, dimensions, parts=())
    for fixture in fixtures:
        target = np.asarray(fixture['board_T_mesh'])
        body = model.worldbody.find(f"body[@name='{fixture['instance']}']")
        assert body is not None
        q = np.fromstring(body.get('quat'), sep=' ')
        actual_R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        actual_p = np.fromstring(body.get('pos'), sep=' ')
        np.testing.assert_allclose(actual_R, target[:3, :3], atol=1e-8, rtol=0)
        np.testing.assert_allclose(actual_p, target[:3, 3] + [0, .065, .0244958], atol=1e-8, rtol=0)
        if fixture['name'] == 'Waterproof_Female':
            np.testing.assert_allclose(actual_R @ [0, 1, 0], [0, 0, 1], atol=1e-8, rtol=0)
        if fixture['name'] == 'DSUB_Female':
            np.testing.assert_allclose(actual_R @ [0, 0, 1], [0, 0, 1], atol=1e-8, rtol=0)
        if fixture['name'] in {'M4_Screw', 'M8_Screw', 'M12_Screw', 'M16_Screw'}:
            diameter = int(fixture['name'].split('_')[0][1:]) / 1000
            vertices, _ = geometry.read_binary_stl(MESHES / (fixture['name'] + '_m.stl'))
            # Outer head vertices are wider than the nominal thread. Their
            # upper plane is the actual modeled shoulder, independent of bbox.
            radius = np.linalg.norm(vertices[:, :2], axis=1)
            shoulder = vertices[radius > diameter / 2 * 1.05, 2].max()
            board_shoulder = (target @ [0, 0, shoulder, 1])[2]
            assert board_shoulder == pytest.approx(-dimensions[scene.BOARD_FILE][2] / 2, abs=2e-6)
    assert not model.worldbody.findall('.//freejoint')
