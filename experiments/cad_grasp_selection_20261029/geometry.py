"""Near-tip contact calibration with unchanged full-hand geometric checks."""
from __future__ import annotations

import numpy as np

from experiments.cad_grasp_selection_20261028.geometry import (
    GeometryValidator as LegacyGeometryValidator, rigid_transform,
)
from .hand_frame import FINGERTIP_FRAME, fingertip_calibration


class GeometryValidator(LegacyGeometryValidator):
    """Convert public fingertip poses to grip-site geometry without trimming it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.errors:
            return
        try:
            self.tip = fingertip_calibration(self.hand, self.config['pad_edge_margin_m'])
            # The inherited checker uses these historical key names for intended
            # contact points. They now contain explicitly calibrated anchors,
            # while all original pad bounds and collision vertices remain intact.
            self.calibration['centers_zero_m'] = self.tip['contact_anchors_zero_grip_site_m']
            self.calibration['midpoint_zero_m'] = self.calibration['centers_zero_m'].mean(axis=0)
        except (ValueError, TypeError, KeyError) as exc:
            self.errors.append(str(exc))

    def validate(self, candidate):
        try:
            if candidate.get('hand_pose_reference') != FINGERTIP_FRAME:
                raise ValueError('Candidate must explicitly name the calibrated fingertip frame')
            if self.errors:
                return super().validate(candidate)
            pose = rigid_transform(candidate['cad_T_hand'], 'fingertip pose')
            grip = pose@self.tip['fingertip_T_grip_site']
            if ('cad_T_grip_site' in candidate and not np.allclose(
                    rigid_transform(candidate['cad_T_grip_site']), grip, atol=1e-9, rtol=0)):
                raise ValueError('Fingertip and grip-site pose fields disagree')
            inset = float(candidate.get('contact_inset_from_pad_tip_m', np.nan))
            if not np.isfinite(inset) or abs(inset-self.tip['contact_inset_m']) > 1e-12:
                raise ValueError('Candidate contact inset disagrees with the validator edge margin')
            internal = dict(candidate, cad_T_hand=grip.tolist())
            result = super().validate(internal)
            checks = result['checks']
            if 'pad_center_reconstruction_error_m' in checks:
                checks['contact_anchor_reconstruction_error_m'] = checks.pop('pad_center_reconstruction_error_m')
            if 'pad_centers_contact_with_opposing_normals' in checks:
                checks['contact_anchors_contact_with_opposing_normals'] = checks.pop('pad_centers_contact_with_opposing_normals')
            result['reasons'] = [reason.replace('pad centers', 'near-tip contact anchors')
                .replace('pad center', 'near-tip contact anchor')
                .replace('pad-center', 'near-tip contact-anchor') for reason in result['reasons']]
            result.update(hand_pose_reference=FINGERTIP_FRAME,
                contact_inset_from_pad_tip_m=self.tip['contact_inset_m'],
                full_hand_distal_extent_beyond_pad_tip_m=self.tip['full_hand_distal_extent_beyond_pad_tip_m'])
            if result['valid']:
                result.update(cad_T_hand=pose.tolist(), cad_T_grip_site=grip.tolist(),
                    world_T_hand_pickup=(self.pickup_T@pose).tolist(),
                    world_T_hand_final=(self.final_T@pose).tolist(),
                    world_T_grip_site_pickup=(self.pickup_T@grip).tolist(),
                    world_T_grip_site_final=(self.final_T@grip).tolist())
            return result
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            return dict(status='unavailable_input', valid=False, reasons=[str(exc)], checks={})
