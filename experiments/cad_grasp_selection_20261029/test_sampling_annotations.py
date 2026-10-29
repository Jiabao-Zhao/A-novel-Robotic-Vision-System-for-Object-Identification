"""Inspect actual CAD labels only; never sample real-part grasp candidates."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.nist_mating_goals_20261028.cad_tools import mesh_component_labels
from experiments.cad_grasp_selection_20261029.sampling_annotations import annotation_for
from experiments.cad_grasp_selection_20261029.task_inputs import build_request

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def actual_regions():
    path = ROOT / 'outputs/nist_scene_corrections_20261029/assembly_goals.json'
    if not path.exists():
        pytest.skip('Local corrected CAD registry is required for this annotation inspection')
    registry = json.loads(path.read_text(encoding='utf-8'))
    out = {}
    for row in registry['parts']:
        result = build_request(row['part'], np.eye(4), [0,0,1,0], registry=registry)
        assert result['status'] == 'available', result.get('message')
        out[row['part']] = row, result['request']
    return out


def test_all_twelve_current_CADs_have_nonempty_bound_sampling_regions(actual_regions):
    assert len(actual_regions) == 12
    for row, request in actual_regions.values():
        region = request['sampling_contact_region']
        assert region['eligible_area_m2'] > 0
        assert region['eligible_area_m2'] < region['provenance']['original_surface_area_m2']
        points = region['sampling_triangles_cad_m'].mean(axis=1)
        assert region['endpoint_contains'](points, region['source_face_ids']).all()


def test_prismatic_pin_area_uses_exposed_thirty_mm_shaft(actual_regions):
    for name, perimeter_mm in [('KET8_Square_8mm',30),('KET12_Square_12mm',40),('KET16_Square_16mm',52)]:
        row, request = actual_regions[name]
        region = request['sampling_contact_region']
        # A50mm pin with20mm inserted leaves30mm of full-size side surfaces.
        assert region['eligible_area_m2'] * 1e6 == pytest.approx(perimeter_mm * 30, abs=.002)
        point, axis = np.asarray(row['axis_point_cad_m']), np.asarray(row['axis_direction_cad'])
        projections = (region['sampling_triangles_cad_m']-point)@axis
        assert projections.max() < .005


def test_USB_guide_excludes_cable_but_retains_complete_CAD(actual_regions):
    _, request = actual_regions['USB_Male']
    region = request['sampling_contact_region']
    labels = mesh_component_labels(request['faces'])
    assert len(np.unique(labels)) == 6
    assert np.all(labels[region['source_face_ids']] == 0)
    assert request['vertices_cad_m'][:,0].max() > .18  # full cable remains
    assert region['sampling_triangles_cad_m'][:,:,0].max() < .026


def test_connector_body_limits_do_not_include_fastening_wings(actual_regions):
    for name, x_limit in [('DSUB_Male',.020),('Waterproof_Male',.0216)]:
        _, request = actual_regions[name]
        patches = request['sampling_contact_region']['sampling_triangles_cad_m']
        assert np.abs(patches[:,:,0]).max() <= x_limit+1.1e-8
        assert np.abs(request['vertices_cad_m'][:,0]).max() > x_limit+.005


def test_BNC_barrel_excludes_rear_stem_and_mating_end(actual_regions):
    _, request = actual_regions['BNC_Male']
    z = request['sampling_contact_region']['sampling_triangles_cad_m'][:,:,2]
    assert z.min() > -.0024950027000159025
    assert z.max() <= .003085+1.1e-8
    assert request['vertices_cad_m'][:,2].max() > .010
    assert request['vertices_cad_m'][:,2].min() < -.010


def test_annotation_rejects_changed_mesh_identity(actual_regions):
    row, request = actual_regions['USB_Male']
    changed = dict(row, part_mesh_sha256='0'*64)
    with pytest.raises(ValueError, match='CAD hash'):
        annotation_for('USB_Male', request['vertices_cad_m'], request['faces'], changed)
