"""Offline checks of target preparation and explicit NIST trial scope."""
from pathlib import Path

import numpy as np

from experiments.cad_grasp_geometry_20261026.gallery import read_stl
from experiments.cad_grasp_selection_20261028.geometry import _solid_collider
from experiments.cad_grasp_selection_20261028.run_nist import prepare_target
from experiments.cad_grasp_selection_20261028.test_geometry import box


ROOT = Path(__file__).resolve().parents[2]


def test_gear_seam_weld_closes_native_mesh_without_changing_file_or_scale():
    path = ROOT/'outputs/simulation/nist_task_board_1/assets/meshes/Gear_Large_m.stl'
    original_bytes = path.read_bytes()
    original, faces = read_stl(path)
    vertices, mapped_faces, mapping, receipt = prepare_target('Gear_Large', original, faces)
    assert receipt['merged_close_pair_count'] == 26
    assert receipt['original_vertex_count'] == 11468
    assert receipt['validation_vertex_count'] == 11442
    assert receipt['maximum_vertex_displacement_m'] < 1.1e-8
    assert np.array_equal(mapping[faces], mapped_faces)
    collider = _solid_collider(vertices, mapped_faces, 'test gear validation copy')
    assert collider.closed_components.all()
    assert path.read_bytes() == original_bytes
    assert receipt['native_metric_scale_preserved'] and not receipt['original_CAD_file_modified']


def test_other_targets_pass_through_without_vertex_motion():
    vertices, faces = box([-.01]*3, [.01]*3)
    actual, actual_faces, mapping, receipt = prepare_target('KET8_Square_8mm', vertices, faces)
    assert np.array_equal(actual, vertices) and np.array_equal(actual_faces, faces)
    assert np.array_equal(mapping, np.arange(len(vertices)))
    assert receipt['maximum_vertex_displacement_m'] == 0
