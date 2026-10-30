"""Controlled contact-angle validation on fixed hand surfaces and fixed fits.

The historical engines are reused unchanged. Finger initialization always uses
the declared hand-surface angle; only after its source planes/calibration have
been verified is the target-versus-finger acceptance angle installed. Proposal
angles belong to the sampler and are not consumed by these validators.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_refinement_20261029.support_geometry import SupportValidator
from experiments.cad_grasp_refinement_20261029.geometry import GeometryValidator as ArrayGeometryValidator
from experiments.cad_fingertip_contacts_20261029.geometry import GeometryValidator as FrozenFingerValidator
from .calibration import make_contact_calibration

HAND_SURFACE_NORMAL_ANGLE_RAD = float(np.deg2rad(1.))
CONTACT_CATEGORIES = ('candidate_input','opening_limits','contact_reconstruction',
                      'surface_normals','pad_contact_geometry','target_closing_collision')
STAGES = dict(
    contact_valid=CONTACT_CATEGORIES,
    pickup_valid=CONTACT_CATEGORIES+('table_clearance',),
    final_goal_valid=CONTACT_CATEGORIES+('final_receiver_collision','protected_surfaces'),
    joint_valid=CONTACT_CATEGORIES+('table_clearance','final_receiver_collision','protected_surfaces'),
    contact_with_protected=CONTACT_CATEGORIES+('protected_surfaces',),
    final_endpoint_clear=('final_receiver_collision',),
    protected_task_compatible=('protected_surfaces',),
)


def _validation_angle(config, explicit):
    value = (config or {}).get('normal_angle_rad',HAND_SURFACE_NORMAL_ANGLE_RAD) if explicit is None else explicit
    try:
        value=float(value)
    except (ValueError,TypeError) as exc:
        raise ValueError('Validation normal angle must be finite in [0,pi/2)') from exc
    if not np.isfinite(value) or not 0<=value<np.pi/2:
        raise ValueError('Validation normal angle must be finite in [0,pi/2)')
    return value


class FingerGeometryValidator(FrozenFingerValidator):
    """Keep real hand labels/anchors/hash fixed while varying acceptance only."""
    def __init__(self,*args,config=None,contact_calibration=None,
                 tip_contact_calibration=None,validation_normal_angle_rad=None,
                 hand_surface_normal_angle_rad=HAND_SURFACE_NORMAL_ANGLE_RAD,**kwargs):
        requested = dict(config or {})
        calibration = contact_calibration if contact_calibration is not None else tip_contact_calibration
        error=None
        try:
            acceptance=_validation_angle(requested,validation_normal_angle_rad)
            hand_angle=float(hand_surface_normal_angle_rad)
            if not np.isfinite(hand_angle) or not 0<hand_angle<np.pi/2:
                raise ValueError('Fixed hand-surface angle must be finite in (0,pi/2)')
            if (not isinstance(calibration,dict) or
                    not np.isclose(calibration.get('normal_angle_rad',np.nan),hand_angle,atol=1e-15,rtol=0)):
                raise ValueError('Contact calibration angle must match the separately fixed hand-surface angle')
        except (ValueError,TypeError) as exc:
            error=str(exc);acceptance=HAND_SURFACE_NORMAL_ANGLE_RAD;hand_angle=HAND_SURFACE_NORMAL_ANGLE_RAD
        self.hand_surface_normal_angle_rad=hand_angle
        self.validation_normal_angle_rad=acceptance
        # Bypass only the historical finger constructor's hardcoded source-file
        # provenance factory. The shared solid/input initializer is array-only;
        # every inherited contact/collision check remains unchanged.
        ArrayGeometryValidator.__init__(self,*args,
            config=dict(requested,normal_angle_rad=acceptance),**kwargs)
        self.contact_calibration=calibration
        self.finger_regions,self.finger_names={},[]
        if error:self.errors.append(error)
        if self.errors:return
        try:
            expected=make_contact_calibration(self.hand,normal_angle_rad=hand_angle,
                edge_margin_m=self.config['pad_edge_margin_m'])
            if (calibration.get('hand_geometry_sha256')!=expected['hand_geometry_sha256']
                    or calibration.get('calibration_sha256')!=expected['calibration_sha256']):
                raise ValueError('Actual finger-contact calibration must match this complete supplied hand and fixed surface policy')
            fingers=sorted([component for component in self.hand['components']
                if not component['is_pad'] and abs(np.asarray(component['translation_per_opening'])[0])>1e-9],
                key=lambda component:np.asarray(component['translation_per_opening'])[0])
            if len(fingers)!=2:raise ValueError('Exactly two original moving finger solids are required')
            self.distal_minimum_z=float(self.tip['usable_pad_tip_origin_grip_site_m'][2])
            cosine=np.cos(hand_angle)
            for index,component in enumerate(fingers):
                vertices=np.asarray(component['vertices_zero_m'],float);hull=ConvexHull(vertices)
                inward=np.array([1.,0,0]) if index==0 else np.array([-1.,0,0])
                allowed=hull.equations[:,:3]@inward>=cosine
                allowed&=vertices[hull.simplices][...,2].max(axis=1)>=self.distal_minimum_z
                if not allowed.any():raise ValueError('No permitted original inward distal surface: '+component['name'])
                self.finger_names.append(component['name'])
                self.finger_regions[component['name']]=dict(index=index,component=component,
                    equations=hull.equations.copy(),allowed_equations=hull.equations[allowed].copy())
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
            self.errors.append(str(exc))

    def validate(self,candidate):
        result=super().validate(candidate)
        result['angular_parameters']=dict(
            hand_surface_normal_angle_rad=self.hand_surface_normal_angle_rad,
            validation_normal_angle_rad=self.validation_normal_angle_rad,
            proposal_normal_angle_rad=None,
            proposal_angle_scope='supplied by sampler; independent of this fixed-fit validation',
            calibration_sha256=(self.contact_calibration or {}).get('calibration_sha256'),
            fixed_hand_surfaces_and_anchors=True)
        result['stages']=classify_stages(result)
        return result


class PadGeometryValidator(SupportValidator):
    """Historical full-pad engine; acceptance angle has no sampler side effect."""
    def __init__(self,*args,config=None,validation_normal_angle_rad=None,**kwargs):
        requested=dict(config or {})
        error=None
        try:angle=_validation_angle(requested,validation_normal_angle_rad)
        except ValueError as exc:error=str(exc);angle=HAND_SURFACE_NORMAL_ANGLE_RAD
        super().__init__(*args,config=dict(requested,normal_angle_rad=angle),**kwargs)
        if error:self.errors.append(error)
        self.validation_normal_angle_rad=angle

    def validate(self,candidate):
        result=super().validate(candidate)
        result['angular_parameters']=dict(hand_surface_normal_angle_rad=None,
            validation_normal_angle_rad=self.validation_normal_angle_rad,
            proposal_normal_angle_rad=None, fixed_hand_surfaces_and_anchors=True,
            hand_surface_scope='unchanged saved rectangular pad calibration')
        result['stages']=classify_stages(result)
        return result


def classify_stages(value):
    """Classify saved nine-category results without any geometry re-evaluation.

    Protected surfaces express assembly-interface compatibility. They are
    explicit in the goal/joint stages and separately reported. A known failure
    takes precedence; otherwise unavailable, then not_evaluated, then pass.
    Required categories and each original status remain visible for every stage.
    """
    if not isinstance(value,dict):value={}
    if 'validity' in value:value=value['validity']
    diagnostics=value.get('diagnostics',value)
    if not isinstance(diagnostics,dict):diagnostics={}
    result={}
    for stage,keys in STAGES.items():
        statuses={}
        for key in keys:
            entry=diagnostics.get(key)
            status=entry.get('status') if isinstance(entry,dict) else entry
            statuses[key]=status if status in ('pass','fail','unavailable','not_evaluated') else 'unavailable'
        values=list(statuses.values())
        status=next((item for item in ('fail','unavailable','not_evaluated') if item in values),'pass')
        result[stage]=dict(status=status,required_categories=list(keys),category_statuses=statuses)
    return result


# Explicit aliases let callers use the current finger mode or saved-diagnostic
# stage classification without depending on a particular trial runner.
GeometryValidator=FingerGeometryValidator
stage_classification=classify_stages
classify_candidate_stages=classify_stages
