from copy import deepcopy
from unittest.mock import patch
import numpy as np

from experiments.cad_grasp_method_comparison_20261026 import ablate
from experiments.cad_surface_grasps_20261026.test_contacts import box_contact
from experiments.cad_surface_grasps_20261026.contacts import inspect_contacts


def test_strict_replay_uses_the_same_paired_point_decision_on_flat_box():
    candidate,collider,pads=box_contact()
    candidate['id']='TEST'
    reason,_=inspect_contacts(candidate,collider,pads)
    result=ablate.paired_variant(candidate,collider,pads,1.,None,.000025,{})
    assert result['valid']==(reason is None)
    assert result['reason']==reason


def test_no_shared_guard_still_runs_later_radial_check():
    candidate,collider,pads=box_contact()
    candidate['id']='TEST'
    fit=dict(id='C',shaft_axis_world=[0,0,1],point_on_axis_world=[0,0,0])
    cache={('TEST','C',.000025):(None,dict(band_m=.000025,
           individual_pad_region_polygons=[1,1],shared_region_exists=False))}
    with patch.object(ablate,'radial_check',return_value=ablate.decision('radial_reject')) as replay:
        result=ablate.witness_variant(candidate,collider,pads,15.,fit,False,cache)
    assert replay.call_count==1
    assert result['reason']=='radial_reject'


def test_matched_witness_variants_change_only_shared_guard_when_other_checks_pass():
    candidate,collider,pads=box_contact()
    candidate['id']='TEST'
    cache={('TEST',None,.000025):(None,dict(band_m=.000025,
           individual_pad_region_polygons=[1,1],shared_region_exists=False))}
    a=ablate.witness_variant(candidate,collider,pads,1.,None,True,cache)
    b=ablate.witness_variant(candidate,collider,pads,1.,None,False,cache)
    assert not a['valid'] and a['reason']=='no_common_opposed_surface_region_inside_pads'
    assert b['valid']


def test_removing_shared_guard_keeps_individual_guarded_pad_requirement():
    candidate,collider,pads=box_contact()
    candidate['id']='TEST'
    cache={('TEST',None,.000025):(None,dict(band_m=.000025,
           individual_pad_region_polygons=[0,1],shared_region_exists=False))}
    result=ablate.witness_variant(candidate,collider,pads,1.,None,False,cache)
    assert result['reason']=='one_pad_has_no_guarded_facing_region'
