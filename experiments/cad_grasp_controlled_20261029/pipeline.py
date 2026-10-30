"""Paired bounded contact-mode experiments with explicit endpoint diagnostics.

Each policy sees at most 100 candidates. A comparison can evaluate both modes
on the same 100 raw proposals and replay a preregistered 50/50 subset; that is
200 underlying benchmark evaluations, not 100 overall. No robot is executed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from experiments.cad_grasp_selection_20261028.criteria import build_reference,evaluate_criteria,select_priority
from experiments.cad_grasp_selection_20261028.geometry import component_vertices,transform_points,_solid_collider
from experiments.cad_grasp_selection_20261029.hand_frame import fingertip_calibration
from experiments.cad_grasp_selection_20261029.task_inputs import protected_checker
from experiments.cad_grasp_refinement_20261029.contact_model import build_contact_model
from experiments.cad_fingertip_contacts_20261029.sampler import fit_candidate
from .calibration import make_contact_calibration,hand_geometry_digest
from .geometry import FingerGeometryValidator,PadGeometryValidator,stage_classification

MODES=('pad','finger','combined')
FIT_KEYS=('contact_mode','contact_calibration_sha256','proposal_contacts_cad_m',
    'proposal_contact_normals_cad','first_contact_fit','cad_T_hand','cad_T_grip_site',
    'contacts_cad_m','contact_face_ids','actual_contacts_hand_m','actual_contacts_hand_frame',
    'actual_contact_component_names','holding_opening_m','pre_opening_m',
    'reconstructed_contact_anchors_cad_m','contact_anchor_convention','contact_normals_cad')
VALIDATION_KEYS=FIT_KEYS+('hand_pose_reference','contact_inset_from_pad_tip_m')


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=_json,separators=(',',':')).encode()).hexdigest()


def compact_reasons(validity):
    """Preserve repeated-message counts without megabytes of duplicate text."""
    counts=Counter(validity.get('reasons',[]))
    validity['reasons']=list(counts);validity['reason_occurrence_counts']=dict(counts)
    for entry in validity.get('diagnostics',{}).values():
        counts=Counter(entry.get('reasons',[]))
        entry['reasons']=list(counts);entry['reason_occurrence_counts']=dict(counts)
    return validity


def apply_pickup(validity,candidate,hand,pickup,table,*,final=None):
    """Recompute exact complete-hand table geometry, preserving CAD/goal checks.

    All other mandatory geometry is in CAD coordinates or the fixed final goal.
    Reuse is valid only inside one Evaluator whose CAD/hand/task inputs are fixed.
    """
    result=deepcopy(validity);checks=result['checks'];diagnostics=result['diagnostics']
    if 'component_geometry' not in checks:
        return result  # No fitted contacts; do not invent downstream passes.
    pickup,table=np.asarray(pickup),np.asarray(table)
    grip=np.asarray(candidate['cad_T_grip_site'])
    old=diagnostics['table_clearance']
    old_reasons=set(old.get('reasons',[]))
    result['reasons']=[r for r in result['reasons'] if r not in old_reasons]
    counts=result.get('reason_occurrence_counts',{})
    for reason in old_reasons:counts.pop(reason,None)
    minima={}
    for comp in hand['components']:
        local=np.concatenate([component_vertices(comp,q) for q in
            (candidate['holding_opening_m'],candidate['pre_opening_m'])])
        world=transform_points(transform_points(local,grip),pickup)
        minima[comp['name']]=float(np.min(world@table[:3]+table[3]))
    for detail in checks['component_geometry']:
        detail['table_minimum_clearance_m']=minima[detail['component']]
    value=min(minima.values());passed=value>=result['tolerances']['table_clearance_m']
    reason='Complete gripper violates table clearance during closing'
    diagnostics['table_clearance']=dict(status='pass' if passed else 'fail',
        reasons=[] if passed else [reason],failed=not passed,unavailable=False,
        reason_occurrence_counts={} if passed else {reason:1})
    if not passed:result['reasons'].append(reason);counts[reason]=1
    checks.update(table_minimum_clearance_m=value,table_clear=passed)
    statuses=[entry['status'] for entry in diagnostics.values()]
    result['valid']=all(value=='pass' for value in statuses)
    result['status']='valid' if result['valid'] else 'unavailable_input' if 'unavailable' in statuses else 'invalid'
    result['diagnostic_failure_categories']=[k for k,e in diagnostics.items() if e['status']=='fail']
    result['diagnostic_unavailable_categories']=[k for k,e in diagnostics.items() if e['status']=='unavailable']
    tip=np.asarray(candidate['cad_T_hand'])
    pose_fields=('cad_T_hand','cad_T_grip_site','world_T_hand_pickup','world_T_grip_site_pickup',
        'world_T_hand_final','world_T_grip_site_final','approach_world')
    for field in pose_fields:result.pop(field,None)
    if result['valid']:
        result.update(cad_T_hand=tip.tolist(),cad_T_grip_site=grip.tolist(),
            world_T_hand_pickup=(pickup@tip).tolist(),world_T_grip_site_pickup=(pickup@grip).tolist(),
            approach_world=(pickup[:3,:3]@grip[:3,2]).tolist())
        if final is not None:
            result.update(world_T_hand_final=(np.asarray(final)@tip).tolist(),
                world_T_grip_site_final=(np.asarray(final)@grip).tolist())
    result['stages']=stage_classification(result)
    result['pickup_table_audit_method']='all unchanged hand vertices at holding/pre-opening endpoints; exact for affine opening and a fixed plane'
    return result


class Evaluator:
    """One fixed part/hand/final task; accepts observed pickup poses per case."""
    def __init__(self,request,*,proposal_angle_deg=1.,validation_angle_deg=1.,hand_angle_deg=1.):
        if request.get('simulator_object_poses_read',False):raise ValueError('Simulator labels/poses are audit-only')
        self.request=request;self.hand=request['hand'];self.vertices=request['vertices_cad_m'];self.faces=request['faces']
        self.pickup0=np.asarray(request['world_T_cad_pickup']);self.table0=np.asarray(request['table_plane_world'])
        self.final=np.asarray(request['world_T_cad_final'])
        self.proposal_angle=float(np.deg2rad(proposal_angle_deg));self.validation_angle=float(np.deg2rad(validation_angle_deg))
        self.calibration=make_contact_calibration(self.hand,normal_angle_rad=np.deg2rad(hand_angle_deg))
        self.tip=fingertip_calibration(self.hand)
        self.region=request['sampling_contact_region']
        self.model=request.get('contact_model') or build_contact_model(self.vertices,self.faces,self.region,
            part=request.get('part'),axis_point_cad_m=request['axis_point_cad_m'],axis_direction_cad=request['axis_direction_cad'])
        self.reference=build_reference(self.vertices,self.faces,request['axis_point_cad_m'],request['axis_direction_cad'])
        self.protected=request.get('protected_contact_check') or protected_checker(
            request['protected_contact_region'],self.vertices,self.faces)
        self.target=_solid_collider(self.vertices,self.faces,'complete CAD for first contact')
        self.fit_cache={};self.validity_cache={};self.validators={}
        self.counters=Counter()
        self.binding=digest(dict(vertices=self.vertices,faces=self.faces,hand=hand_geometry_digest(self.hand),
            final=self.final,receivers=request['receivers'],protected=request['protected_contact_region'],
            calibration=self.calibration['calibration_sha256'],model=self.model.provenance))

    def _validator(self,mode,angle):
        key=mode,float(angle)
        if key not in self.validators:
            kwargs=dict(vertices_cad=self.vertices,faces=self.faces,hand=self.hand,
                pickup_T=self.pickup0,final_T=self.final,table_plane=self.table0,
                protected_contact_check=self.protected,receivers=self.request['receivers'],
                contact_model=self.model,validation_normal_angle_rad=angle)
            if mode=='finger':kwargs['contact_calibration']=self.calibration
            cls=FingerGeometryValidator if mode=='finger' else PadGeometryValidator
            validator=cls(**kwargs)
            if validator.errors:raise ValueError('; '.join(validator.errors))
            self.validators[key]=validator
        return self.validators[key]

    def fit(self,raw,mode):
        candidate=deepcopy(raw);candidate['contact_mode']=mode
        if mode=='pad':
            grip=np.asarray(candidate['cad_T_grip_site']);points=np.asarray(candidate['contacts_cad_m'])
            candidate.update(actual_contacts_hand_m=((points-grip[:3,3])@grip[:3,:3]).tolist(),
                actual_contacts_hand_frame='original saved grip-site frame')
            return candidate
        key=digest({k:raw.get(k) for k in ('contacts_cad_m','cad_T_grip_site','contact_face_ids','contact_normals_cad','approach_cad')})
        if key not in self.fit_cache:
            fitted=fit_candidate(candidate,self.target,self.hand,self.calibration,contact_model=self.model)
            self.fit_cache[key]={k:deepcopy(fitted[k]) for k in FIT_KEYS if k in fitted}
            self.counters['unique_finger_first_contact_fits']+=1
        else:self.counters['reused_finger_first_contact_fits']+=1
        candidate.update(deepcopy(self.fit_cache[key]));candidate['contact_mode']='finger'
        return candidate

    def evaluate(self,candidate,pickup,table,*,angle=None):
        mode=candidate['contact_mode'];angle=self.validation_angle if angle is None else float(angle)
        geometry={k:candidate.get(k) for k in VALIDATION_KEYS}
        key=mode,angle,digest(geometry)
        reused=key in self.validity_cache
        if not reused:
            self.validity_cache[key]=compact_reasons(self._validator(mode,angle).validate(candidate))
            self.counters['unique_local_and_final_validations']+=1
        else:self.counters['reused_local_and_final_validations']+=1
        validity=apply_pickup(self.validity_cache[key],candidate,self.hand,pickup,table,final=self.final)
        validity['geometry_reuse']=dict(reused=reused,fixed_task_binding_sha256=self.binding,
            local_CAD_and_final_goal_only=True,observed_pickup_table_recomputed=True)
        record=dict(candidate,validity=validity)
        if validity['status']=='valid':
            approach=np.asarray(pickup)[:3,:3]@np.asarray(candidate['approach_cad'])
            record.update(valid=True,criteria=evaluate_criteria(candidate['contacts_cad_m'],self.reference,approach,np.asarray(table)[:3]),
                world_T_hand_pickup=(np.asarray(pickup)@np.asarray(candidate['cad_T_hand'])).tolist(),
                world_T_hand_final=(self.final@np.asarray(candidate['cad_T_hand'])).tolist(),
                world_T_grip_site_pickup=(np.asarray(pickup)@np.asarray(candidate['cad_T_grip_site'])).tolist(),
                world_T_grip_site_final=(self.final@np.asarray(candidate['cad_T_grip_site'])).tolist(),approach_world=approach.tolist())
        return record

    def decide(self,records,*,mode,seed,strategy,sampling_reference=None):
        valid=[r for r in records if r['validity']['status']=='valid']
        decision=select_priority(valid)
        stages={name:dict(Counter(r['validity'].get('stages',stage_classification(r['validity']))[name]['status'] for r in records))
                for name in stage_classification({})}
        return dict(status='selected' if valid else 'no_valid_grasp_within_budget',
            selected_grasp=decision['selected_candidate'],proposed_count=len(records),valid_count=len(valid),
            contact_mode=mode,sampler_strategy=strategy,seed=seed,
            reference=self.reference,selection_stages=decision['stages'],tie_tolerances=decision['tie_tolerances'],
            final_tied_candidate_ids=decision['final_tied_candidate_ids'],stage_counts=stages,
            sampling_reference=sampling_reference,robot_or_physics_execution=False,
            physical_contact_status='unverified; collision hull/display surface discrepancy audited separately',
            no_VLM=True,no_pickup_or_assembly_execution=True)

    def compare(self,raw,pickup,table,*,seed,strategy,sampling_reference=None):
        pools={mode:[self.evaluate(self.fit(candidate,mode),pickup,table) for candidate in raw] for mode in ('pad','finger')}
        chosen={mode:{r['candidate_id']:r for r in records} for mode,records in pools.items()}
        pools['combined']=[chosen[candidate['combined_contact_mode']][candidate['candidate_id']] for candidate in raw]
        decisions={mode:self.decide(records,mode=mode,seed=seed,strategy=strategy,sampling_reference=sampling_reference)
                   for mode,records in pools.items()}
        return decisions,pools


def select_grasp(request,*,seed=0,mode='combined',strategy='task_informed',max_poses=100,
                 surface_budget=200,orientation_samples=10,proposal_angle_deg=1.,validation_angle_deg=1.):
    """Select one bounded mode, without evaluating the two unused policy arms."""
    from .sampler import generate_candidates
    if mode not in MODES:raise ValueError('contact mode must be pad, finger or combined')
    evaluator=Evaluator(request,proposal_angle_deg=proposal_angle_deg,validation_angle_deg=validation_angle_deg)
    raw,sampling=generate_candidates(evaluator.vertices,evaluator.faces,evaluator.hand,
        world_T_cad_pickup=request['world_T_cad_pickup'],world_T_cad_final=request['world_T_cad_final'],
        table_plane_world=request['table_plane_world'],axis_point_cad_m=request['axis_point_cad_m'],
        axis_direction_cad=request['axis_direction_cad'],sampling_contact_region=evaluator.region,
        contact_model=evaluator.model,contact_calibration=evaluator.calibration,
        seed=seed,strategy=strategy,contact_modes=('pad','finger'),
        max_poses=max_poses,surface_budget=surface_budget,orientation_samples=orientation_samples,
        proposal_normal_angle_rad=evaluator.proposal_angle,final_up_world=request.get('final_up_world',[0,0,1]))
    records=[]
    for candidate in raw:
        contact_mode=candidate['combined_contact_mode'] if mode=='combined' else mode
        records.append(evaluator.evaluate(evaluator.fit(candidate,contact_mode),request['world_T_cad_pickup'],request['table_plane_world']))
    result=evaluator.decide(records,mode=mode,seed=seed,strategy=strategy)
    result['sampling']=sampling;result['evaluation_counters']=dict(evaluator.counters)
    result['diagnostics']=records
    return result
