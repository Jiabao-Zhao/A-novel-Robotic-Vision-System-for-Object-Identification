import numpy as np
from .contacts import intersect_polygons,signed_area,closest_point


def test_convex_overlap_and_reversed_winding():
    a=np.array([[0,0],[2,0],[2,2],[0,2]],float)
    b=np.array([[1,1],[3,1],[3,3],[1,3]],float)
    assert abs(abs(signed_area(intersect_polygons(a,b)))-1)<1e-12
    assert abs(abs(signed_area(intersect_polygons(a,b[::-1])))-1)<1e-12


def test_disjoint_polygons_do_not_invent_common_contact():
    a=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    assert not len(intersect_polygons(a,a+2))
    assert np.allclose(closest_point(a,[2,.4]),[1,.4])
    assert np.allclose(closest_point(a,[.3,.4]),[.3,.4])


def box_contact(yaw_deg=0.):
    from experiments.cad_grasp_geometry_20261026 import run_trial as old
    from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
    from experiments.cad_grasp_regions_20261026.run import hand_rotation
    from scipy.spatial import ConvexHull
    old.initialize()
    v=np.array([[x,y,z] for x in (-.01,.01) for y in (-.015,.015) for z in (.015,.045)])
    hull=ConvexHull(v)
    f=hull.simplices.copy()
    for i,face in enumerate(f):
        normal=np.cross(v[face[1]]-v[face[0]],v[face[2]]-v[face[0]])
        if normal@hull.equations[i,:3]<0:
            f[i]=face[::-1]
    collider=MeshCollider(v,f)
    angle=np.deg2rad(yaw_deg)
    pose=np.eye(4)
    pose[:3,:3]=hand_rotation([np.cos(angle),np.sin(angle),0.])
    pose[:3,3]=[0.,0.,.0264]
    candidate=dict(world_T_grip=pose.tolist(),opening_m=.045,pad_depth_m=-.0036)
    components,pads,_=old.hand_geometry()
    reason,evidence=old.validate(candidate,collider,components,pads,np.array([0.,0.,1.,0.]))
    assert reason is None,reason
    candidate.update(evidence)
    return candidate,collider,pads


def test_aligned_flat_box_has_common_paired_pad_region():
    from .contacts import inspect_contacts
    candidate,collider,pads=box_contact()
    reason,evidence=inspect_contacts(candidate,collider,pads)
    assert reason is None
    points=np.asarray(evidence['paired_surface_points_world_m'])
    pose=np.asarray(candidate['world_T_grip'])
    local=(points-pose[:3,3])@pose[:3,:3]
    assert np.allclose(local[0,1:],local[1,1:],atol=1e-10)
    assert all(p['polygons'] for p in evidence['pad_surface_regions'])


def test_old_interference_pass_does_not_accept_oblique_flat_contact():
    from .contacts import inspect_contacts
    candidate,collider,pads=box_contact(5.)
    reason,evidence=inspect_contacts(candidate,collider,pads)
    assert reason=='actual_pad_normal_not_aligned_with_surface'
    assert min(evidence['actual_pad_normal_errors_deg'])>4.99
