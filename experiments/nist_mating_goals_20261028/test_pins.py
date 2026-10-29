"""Checks of pin task endpoints and CAD-based aperture assignment."""
import numpy as np
import pytest

from experiments.nist_mating_goals_20261028 import pins


@pytest.fixture(scope='module')
def reviewed(tmp_path_factory):
    required = [pins.MESHES/'GMC_Laser_Plate_Virtual_m.stl',
                pins.ROOT/'experiments/runtime_antipodal_assembly_20261023/goal_audit.json']
    if not all(p.is_file() for p in required):
        pytest.skip('Requires the existing local NIST board CAD and goal inputs')
    saved_output = pins.OUTPUT
    pins.OUTPUT = tmp_path_factory.mktemp('pin_goal_review')/'pins.json'
    try:
        return pins.review()
    finally:
        pins.OUTPUT = saved_output


def test_all_six_task_endpoints_place_chamfered_tip_12mm_below_top(reviewed):
    assert {r['part'] for r in reviewed['parts']} == set(pins.PARTS)
    for row in reviewed['parts']:
        pose = np.asarray(row['proposed_board_T_part'])
        frame = np.asarray(row['part_mating_frame_cad'])
        receiver = np.asarray(row['receiver_mating_frame_board'])
        tip = pose @ np.r_[frame[:3,3],1.]
        np.testing.assert_allclose(tip[:2],receiver[:2,3],atol=1e-12)
        assert receiver[2,3]-tip[2] == pytest.approx(.012,abs=1e-12)
        np.testing.assert_allclose(pose[:3,:3]@np.array([0.,0.,1.]),[0.,0.,-1.],atol=1e-12)
        assert np.linalg.det(pose[:3,:3]) == pytest.approx(1.,abs=1e-12)


def test_user_depth_is_distinguished_from_a_cad_stop(reviewed):
    for row in reviewed['parts']:
        m = row['measurements']
        assert m['chosen_task_depth_m'] == .012
        assert m['depth_source'] == pins.DEPTH_SOURCE
        assert m['shoulder_or_axial_stop_present'] is False
        assert m['board_only_axial_stop_present'] is False
        assert row['mating_transform_parameterization']['axial_stop_CAD_defined'] is False
        assert row['status'] == 'CAD_direction_clearance_checked_user_depth_12mm'
        assert m['tip_protrusion_below_board_underside_m'] == pytest.approx(.00300839996,abs=1e-10)


def test_mating_frame_equation_matches_complete_goal(reviewed):
    for row in reviewed['parts']:
        receiver = np.asarray(row['receiver_mating_frame_board'])
        part_frame = np.asarray(row['part_mating_frame_cad'])
        chosen_depth = np.eye(4); chosen_depth[2,3] = .012
        expected = receiver @ chosen_depth @ np.linalg.inv(part_frame)
        np.testing.assert_allclose(row['proposed_board_T_part'],expected,atol=1e-12)
        np.testing.assert_allclose(row['review_board_T_part'],expected,atol=1e-12)


def test_all_full_shank_sections_fit_actual_apertures(reviewed):
    for row in reviewed['parts']:
        m=row['measurements']
        assert m['selected_minimum_polygon_side_clearance_m'] > 0
        assert m['full_shank_projection_outside_selected_aperture_m2'] < 1e-18
        assert m['through_aperture_same_top_and_bottom'] is True
        assert m['top_and_bottom_aperture_boundary_hausdorff_distance_m'] <= pins.NUMERICAL_TOL_M
        lo,hi=m['full_shank_spans_entire_board_depth_interval_m']
        assert lo < .012 < hi


def test_tighter_round12_hole_is_not_used_when_occupied_by_m12_screw(reviewed):
    row = next(x for x in reviewed['parts'] if x['part']=='RGOCG12-50_12mm')
    assert row['measurements']['selected_opening_id']=='opening_27'
    tighter=next(x for x in row['measurements']['compatible_apertures'] if x['opening_id']=='opening_26')
    assert any(x['name']=='M12_Screw' for x in tighter['existing_fixed_fixtures_at_center'])


def test_small_cad_slot_rotations_are_retained(reviewed):
    rows={x['part']:x for x in reviewed['parts']}
    assert rows['KET8_Square_8mm']['measurements']['selected_aperture_aligned_yaw_deg'] == pytest.approx(-.1014857724,abs=1e-7)
    assert rows['KET16_Square_16mm']['measurements']['selected_aperture_aligned_yaw_deg'] == pytest.approx(-.1924362914,abs=1e-7)
    assert rows['KET12_Square_12mm']['measurements']['selected_aperture_aligned_yaw_deg'] == pytest.approx(90.,abs=1e-7)
