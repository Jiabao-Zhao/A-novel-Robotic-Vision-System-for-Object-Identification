"""Explicit near-tip contact calibration for the saved complete Panda hand.

The commanded frame is the midpoint of the physical finger distal ends. Contact
anchors are one declared pad-edge margin behind the usable pad edges. Pad centers
are not contact anchors. Every saved pad, finger and palm collision vertex is
retained, including finger geometry extending beyond the usable pad tips.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import (
    pad_calibration, rigid_transform, transform_points,
)

FINGERTIP_FRAME = 'Panda physical fingertip midpoint; X jaw closing, +Z approach'
USABLE_PAD_TIP_FRAME = 'Panda distal usable-pad-tip midpoint; X jaw closing, +Z approach'
DEFAULT_CONTACT_INSET_M = .00025


def fingertip_calibration(hand, contact_inset_m=DEFAULT_CONTACT_INSET_M):
    """Read saved geometry and derive a fixed reference and near-tip anchors.

    ``grip_site_T_fingertip`` maps fingertip-frame coordinates into the legacy
    verified grip-site frame. This is geometry calibration, not physical robot
    calibration. Opening is still the exported joint command, not pad gap.
    """
    base = pad_calibration(hand)
    bounds = np.asarray(base['usable_bounds_hand_zero_m'])
    inset = float(contact_inset_m)
    if (not np.isfinite(inset) or inset < 0
            or np.any(inset >= .5*(bounds[:, 1, 2]-bounds[:, 0, 2]))):
        raise ValueError('Near-tip contact inset must be finite, nonnegative and within each pad')
    tips = np.asarray(base['centers_zero_m']).copy()
    tips[:, 2] = bounds[:, 1, 2]
    if np.max(np.abs((tips[1]-tips[0])[1:])) > 1e-8:
        raise ValueError('Distal usable pad tips are not aligned on the closing axis')
    if np.linalg.norm(base['midpoint_per_opening']) > 1e-9:
        raise ValueError('A fixed fingertip frame requires symmetric pad opening motion')
    pad_tip_origin = tips.mean(axis=0)
    moving_fingers = [component for component in hand['components']
        if not component['is_pad']
        and abs(np.asarray(component['translation_per_opening'])[0]) > 1e-9]
    physical_components = moving_fingers or base['pads']
    finger_tips = []
    for side in (-1, 1):
        group = [component for component in physical_components
                 if side*np.asarray(component['translation_per_opening'])[0] > 0]
        if not group:
            raise ValueError('Both physical fingers need a calibrated distal end')
        points = np.concatenate([np.asarray(component['vertices_zero_m']) for component in group])
        distal_z = points[:, 2].max()
        finger_tips.append(points[np.abs(points[:, 2]-distal_z) <= 1e-9].mean(axis=0))
    origin = np.asarray(finger_tips).mean(axis=0)
    grip_tip = np.eye(4)
    grip_tip[:3, 3] = origin
    tip_grip = np.eye(4)
    tip_grip[:3, 3] = -origin
    grip_pad_tip = np.eye(4)
    grip_pad_tip[:3, 3] = pad_tip_origin
    anchors = tips.copy()
    # One representable step inward prevents roundoff from violating the same
    # declared edge margin when GeometryValidator subtracts the coordinates.
    anchors[:, 2] = np.nextafter(bounds[:, 1, 2]-inset, -np.inf)
    vertices = np.concatenate([np.asarray(c['vertices_zero_m']) for c in hand['components']])
    return dict(
        frame=FINGERTIP_FRAME,
        grip_site_T_fingertip=grip_tip,
        fingertip_T_grip_site=tip_grip,
        fingertip_origin_grip_site_m=origin,
        physical_finger_tips_zero_grip_site_m=np.asarray(finger_tips),
        physical_finger_tip_components=[c['name'] for c in physical_components],
        usable_pad_tip_frame=USABLE_PAD_TIP_FRAME,
        usable_pad_tip_origin_grip_site_m=pad_tip_origin,
        grip_site_T_usable_pad_tip=grip_pad_tip,
        distal_pad_edges_zero_grip_site_m=tips,
        contact_anchors_zero_grip_site_m=anchors,
        contact_anchors_zero_fingertip_m=anchors-origin,
        contact_anchors_per_opening=np.asarray(base['centers_per_opening']).copy(),
        contact_inset_m=inset,
        full_hand_distal_extent_beyond_pad_tip_m=float(vertices[:, 2].max()-pad_tip_origin[2]),
        gap_zero_m=base['gap_zero_m'], gap_per_opening=base['gap_per_opening'],
        opening_limits_m=np.asarray(base['opening_limits_m']).copy(),
        geometry_source='saved simulator hand export; no physical calibration claimed',
    )


def hand_in_fingertip_frame(hand, contact_inset_m=DEFAULT_CONTACT_INSET_M):
    """Return a complete transformed copy; never trim collision components."""
    calibration = fingertip_calibration(hand, contact_inset_m)
    result = deepcopy(hand)
    result['frame'] = FINGERTIP_FRAME
    result['source_frame'] = hand['frame']
    result['grip_site_T_fingertip'] = calibration['grip_site_T_fingertip'].tolist()
    result['contact_inset_m'] = calibration['contact_inset_m']
    for component in result['components']:
        component['vertices_zero_m'] = transform_points(
            component['vertices_zero_m'], calibration['fingertip_T_grip_site']).tolist()
    return result


def grip_site_pose_from_fingertip(cad_T_fingertip, hand,
                                 contact_inset_m=DEFAULT_CONTACT_INSET_M):
    calibration = fingertip_calibration(hand, contact_inset_m)
    return rigid_transform(cad_T_fingertip, 'fingertip pose') @ calibration['fingertip_T_grip_site']


def fingertip_pose_from_grip_site(cad_T_grip_site, hand,
                                 contact_inset_m=DEFAULT_CONTACT_INSET_M):
    calibration = fingertip_calibration(hand, contact_inset_m)
    return rigid_transform(cad_T_grip_site, 'grip-site pose') @ calibration['grip_site_T_fingertip']
