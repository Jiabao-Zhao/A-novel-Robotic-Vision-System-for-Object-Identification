import numpy as np
from .centering import center_from_first_contacts
from .test_contacts import box_contact


def test_known_two_mm_hand_offset_is_projected_onto_contact_midpoint():
    candidate,collider,pads=box_contact()
    pose=np.asarray(candidate['world_T_grip'])
    pose[:3,3]+=.002*pose[:3,0]
    centered,evidence=center_from_first_contacts(pose,candidate['opening_m'],collider,pads)
    assert evidence['applied'] and evidence['geometry_queries']==2
    assert abs(evidence['translation_along_closing_m']+.002)<3e-6
    assert abs(centered[0,3])<3e-6
    from experiments.cad_grasp_geometry_20261026 import run_trial as old
    components,_,_=old.hand_geometry()
    candidate['world_T_grip']=centered.tolist()
    reason,details=old.validate(candidate,collider,components,pads,np.array([0.,0.,1.,0.]))
    assert reason is None
    assert details['contact_opening_difference_m']<5e-6


def test_projection_keeps_approach_height_and_orientation():
    candidate,collider,pads=box_contact()
    before=np.asarray(candidate['world_T_grip'])
    centered,evidence=center_from_first_contacts(before,candidate['opening_m'],collider,pads)
    assert np.array_equal(centered[:3,:3],before[:3,:3])
    assert centered[2,3]==before[2,3]
