"""Check depth preservation and shortlist alternatives on simple geometry."""
import numpy as np

from experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.search import (
    frames_with_full_finger_depths, small_shortlist, FINGER_DEPTHS_M,
)


def test_full_finger_depths_keep_contact_midpoint():
    pair = np.array([[[-.004, 0., .004], [.004, 0., .004]]])
    frames = frames_with_full_finger_depths(pair, np.eye(3), np.zeros(3), 1)
    assert set(round(c.pad_depth_m, 6) for c in frames) == set(FINGER_DEPTHS_M)
    for c in frames:
        np.testing.assert_allclose(c.world_T_grip[:3, 3] + c.world_T_grip[:3, 2]*c.pad_depth_m,
                                   c.contacts_world_m.mean(0), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(c.world_T_grip[:3, :3]), 1.)


def test_shortlist_is_bounded_and_keeps_both_low_pin_depths():
    pair = np.array([[[-.004, 0., .004], [.004, 0., .004]]])
    frames = frames_with_full_finger_depths(pair, np.eye(3), np.zeros(3), 1)
    low_pin_clear = [c for c in frames if c.roll_deg == 0. and c.pad_depth_m in (.006, .008)]
    selected = small_shortlist(low_pin_clear, 12)
    assert 0 < len(selected) <= 12
    assert {c.pad_depth_m for c in selected} == {.006, .008}
    assert len({id(c) for c in selected}) == len(selected)
