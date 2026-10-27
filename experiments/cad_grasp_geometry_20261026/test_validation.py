"""Regress the actual hand-aware candidate validator before scene evaluation.

Run using the existing WSL experiment runtime. These tests use the frozen Panda
collision-component export; they do not instantiate/step physics or perform IK.
"""

import copy

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from experiments.cad_grasp_geometry_20261026 import run_trial
from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.cad_grasp_geometry_20261026.hand_model import vertices


@pytest.fixture(scope="module")
def actual_hand():
    run_trial.initialize()
    return run_trial.hand_geometry()


def target_box(reverse=False):
    points = np.asarray([[x, y, z] for x in (-.005, .005)
                         for y in (-.004, .004) for z in (0., .008)])
    faces = ConvexHull(points).simplices
    return MeshCollider(points, faces[::-1] if reverse else faces)


def symmetric_candidate():
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[2, 3] = .010
    return dict(world_T_grip=pose.tolist(), opening_m=.023, width_m=.010,
                pad_depth_m=.004,
                contacts_world_m=[[-.005, 0., .006], [.005, 0., .006]])


def test_symmetric_box_has_two_pad_contacts_at_correct_actual_opening(actual_hand):
    components, pads, gap_offset = actual_hand
    reason, evidence = run_trial.validate(symmetric_candidate(), target_box(), components, pads,
                                          np.asarray([0., 0., 1., 0.]))
    assert reason is None
    assert gap_offset == pytest.approx(.001, abs=1e-10)
    assert evidence["contact_opening_m"] == pytest.approx(.011, abs=5e-6)
    assert evidence["contact_opening_difference_m"] <= 5e-6
    assert len(evidence["actual_contacts"]) == 2
    assert evidence["table_clearance_m"] > run_trial.CONFIG["table_clearance_m"]


def test_same_hand_pose_lowered_is_rejected_by_the_table(actual_hand):
    components, pads, _ = actual_hand
    candidate = symmetric_candidate()
    candidate["world_T_grip"][2][3] = .008
    reason, evidence = run_trial.validate(candidate, target_box(), components, pads,
                                          np.asarray([0., 0., 1., 0.]))
    assert reason == "finger_or_palm_too_low_for_table"
    assert evidence["table_clearance_m"] < 0


@pytest.mark.parametrize("depth", [.006, .008])
def test_old_extended_finger_depths_are_outside_actual_current_pad(actual_hand, depth):
    components, pads, _ = actual_hand
    candidate = symmetric_candidate()
    candidate["pad_depth_m"] = depth
    reason, _ = run_trial.validate(candidate, target_box(), components, pads,
                                    np.asarray([0., 0., 1., 0.]))
    assert reason == "proposed_contact_outside_inner_pad"


def test_actual_inner_pad_gap_is_79mm_at_80mm_command(actual_hand):
    _, pads, gap_offset = actual_hand
    opening = run_trial.HAND["opening_limits_m"][1]
    negative, positive = pads
    measured_gap = vertices(positive, opening)[:, 0].min() - vertices(negative, opening)[:, 0].max()
    assert measured_gap == pytest.approx(.079, abs=1e-10)
    assert measured_gap == pytest.approx(opening-gap_offset, abs=1e-10)
    assert opening-gap_offset-run_trial.CONFIG["extra_opening_m"] == pytest.approx(.067, abs=1e-10)


def test_off_center_box_rejects_different_first_contact_openings(actual_hand):
    components, pads, _ = actual_hand
    candidate = symmetric_candidate()
    candidate["world_T_grip"][0][3] = .0002
    reason, evidence = run_trial.validate(candidate, target_box(), components, pads,
                                          np.asarray([0., 0., 1., 0.]))
    assert reason == "fingers_reach_target_at_different_openings"
    assert evidence["opening_difference_m"] == pytest.approx(.0008, abs=6e-6)


def test_actual_validator_is_independent_of_target_triangle_order(actual_hand):
    components, pads, _ = actual_hand
    outputs = [run_trial.validate(copy.deepcopy(symmetric_candidate()), target_box(reverse), components,
                                 pads, np.asarray([0., 0., 1., 0.])) for reverse in (False, True)]
    assert outputs[0][0] is None and outputs[1][0] is None
    assert outputs[0][1]["first_contact_openings_m"] == pytest.approx(outputs[1][1]["first_contact_openings_m"], abs=1e-10)
