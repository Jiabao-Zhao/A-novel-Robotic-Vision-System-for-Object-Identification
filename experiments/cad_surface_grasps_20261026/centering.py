"""Center the Panda hand on finite-pad first contacts, without moving the part."""
import numpy as np
from experiments.cad_grasp_geometry_20261026.hand_model import vertices as hand_vertices


def center_from_first_contacts(pose,opening,collider,pads,fraction_tolerance=1e-4):
    pose=np.asarray(pose,dtype=float)
    before=pose.copy()
    openings=[]
    for pad in pads:
        points=hand_vertices(pad,opening)@pose[:3,:3].T+pose[:3,3]
        motion=pose[:3,:3]@(-opening*np.asarray(pad['translation_per_opening']))
        first=collider.first_contact(points,motion,fraction_tolerance)
        if first is None:
            return pose,dict(world_T_grip_before=before.tolist(),first_contact_openings_before_m=openings,
                translation_along_closing_m=0.,geometry_queries=len(openings)+1,applied=False,
                reason='one_pad_has_no_first_contact_in_projection')
        openings.append(float(opening*(1-first['fraction'])))
    slopes=[float(np.asarray(p['translation_per_opening'])[0]) for p in pads]
    if not np.allclose(slopes,[-.5,.5],atol=1e-12):
        raise ValueError('This centering equation is for the declared Panda +/- opening/2 motion')
    shift=(openings[1]-openings[0])/4
    pose=pose.copy()
    pose[:3,3]+=shift*pose[:3,0]
    return pose,dict(world_T_grip_before=before.tolist(),first_contact_openings_before_m=openings,
        translation_along_closing_m=float(shift),geometry_queries=2,applied=True,
        expected_equal_contact_opening_m=float(np.mean(openings)),
        equation='delta_x = (right_contact_opening - left_contact_opening) / 4')
