"""Focused regressions for retained near-contact evidence and surface directions."""
from copy import deepcopy
import numpy as np

from .verify_results import Checks, audit_contract, audit_surface_regions, audit_used_cylinders, audit_center_projection


def fixture():
    rotation=np.diag([1.,-1.,-1.])
    yz=np.array([[-.006,-.006],[0.,.006],[.006,-.006]])
    local=np.array([np.c_[np.full(3,-.005),yz],np.c_[np.full(3,.005),yz[::-1]]])
    world=local@rotation.T
    pose=np.eye(4)
    pose[:3,:3]=rotation
    pads={}
    records=[]
    witnesses=[]
    for i,(name,sign) in enumerate((('left',-1.),('right',1.))):
        xs=[-.0045,.0005] if sign<0 else [-.0005,.0045]
        vertices=np.array([[x,y,z] for x in xs for y in (-.008,.008) for z in (-.008,.008)])
        pads[name]=dict(name=name,vertices_zero_m=vertices.tolist(),translation_per_opening=[sign*.5,0,0],is_pad=True)
        normal=np.cross(local[i,1]-local[i,0],local[i,2]-local[i,0])
        normal/=np.linalg.norm(normal)
        records.append(dict(pad=name,inner_plane_x_m=sign*.005,polygons=[dict(triangle_index=i,
                       polygon_local_m=local[i].tolist(),normal_local=normal.tolist(),pad_normal_error_deg=0.)]))
        witnesses.append(dict(pad=name,opening_m=.011,normal_alignment=1.,first_contact=dict(triangle_index=i)))
    c=dict(id='C0001',family_id='D001_R001',direction_id='D001',direction_source='opposed_planar_facets',
           jaw_sign=1.,world_T_grip=pose.tolist(),pad_depth_m=0.,opening_m=.023,contact_opening_m=.011,
           valid=True,rejection_reason=None,actual_contacts=witnesses,
           actual_pad_normal_errors_deg=[0.,0.],surface_normal_tolerance_deg=1.,near_contact_band_m=.000025,
           pad_surface_regions=records,paired_surface_points_world_m=[[-.005,0.,0.],[.005,0.,0.]],
           paired_surface_triangle_indices=[0,1],common_region_polygon_yz_m=yz.tolist(),paired_surface_center_offset_m=0.,
           first_contact_openings_m=[.011,.011],center_projection=dict(world_T_grip_before=pose.tolist(),
               first_contact_openings_before_m=[.011,.011],translation_along_closing_m=0.,geometry_queries=2,applied=True,
               expected_equal_contact_opening_m=.011,equation='delta_x = (right_contact_opening - left_contact_opening) / 4'))
    return c,world,pads


def test_finite_flat_regions_and_paired_points_replay():
    c,t,p=fixture()
    checks=Checks()
    report=audit_surface_regions(c,t,p,checks)
    assert checks.failed_count==0, checks.failures
    assert report['near_contact_polygons']==2


def test_independent_normals_reject_forged_alignment_evidence():
    c,t,p=fixture()
    c['actual_pad_normal_errors_deg']=[.2,.2]
    checks=Checks()
    audit_surface_regions(c,t,p,checks)
    assert any('independently reconstructed' in f for f in checks.failures)


def test_diagonal_arbitrary_points_are_not_accepted_as_the_paired_region_points():
    c,t,p=fixture()
    c['paired_surface_points_world_m'][1][2]+=.001
    checks=Checks()
    audit_surface_regions(c,t,p,checks)
    assert any('share hand Y/Z' in f for f in checks.failures)


def test_polygon_outside_native_triangle_is_rejected_even_if_inside_pad():
    c,t,p=fixture()
    c['pad_surface_regions'][0]['polygons'][0]['polygon_local_m'][0][1]=-.007
    checks=Checks()
    audit_surface_regions(c,t,p,checks)
    assert any('vertices lie on recorded native triangle' in f for f in checks.failures)


def test_fitted_round_side_tangency_is_replayed_separately_from_mesh_normals():
    c,t,p=fixture()
    # Narrow native side triangles approximate a small cylinder sector.
    t[:,:,2]*=.01
    for record in c['pad_surface_regions']:
        for polygon in record['polygons']:
            for vertex in polygon['polygon_local_m']:
                vertex[2]*=.01
    for vertex in c['common_region_polygon_yz_m']:
        vertex[1]*=.01
    sectors=[]
    for triangle in t:
        radial=triangle.copy()
        radial[:,1]=0.
        radial/=np.linalg.norm(radial,axis=1)[:,None]
        sectors.append(float(np.degrees(np.arccos(np.clip((radial@radial.T).min(),-1,1)))))
    c.update(surface_normal_tolerance_deg=15.,cylinder_model=dict(shaft_axis_world=[0.,1.,0.],point_on_axis_world=[0.,0.,0.]),
             fitted_cylinder_tangent_errors_deg=[0.,0.],cylinder_closing_perpendicular_error_deg=0.,
             mesh_contact_sector_widths_deg=sectors,fitted_cylinder_tangent_allowances_deg=(1.+np.asarray(sectors)).tolist())
    checks=Checks()
    audit_surface_regions(c,t,p,checks)
    assert checks.failed_count==0,checks.failures
    c['cylinder_model']['point_on_axis_world']=[0.,0.,.001]
    checks=Checks()
    audit_surface_regions(c,t,p,checks)
    assert any('radial error bounded by native triangle sector' in f for f in checks.failures)


def test_round_allowance_cannot_be_silently_inflated_beyond_native_sector():
    c,t,p=fixture()
    c.update(surface_normal_tolerance_deg=15.,cylinder_model=dict(shaft_axis_world=[0.,1.,0.],point_on_axis_world=[0.,0.,0.]),
             fitted_cylinder_tangent_errors_deg=[0.,0.],cylinder_closing_perpendicular_error_deg=0.,
             mesh_contact_sector_widths_deg=[180.,180.],fitted_cylinder_tangent_allowances_deg=[181.,181.])
    checks=Checks()
    audit_surface_regions(c,t,p,checks)
    assert any('sector-width replay' in f for f in checks.failures)
    assert any('includes only native sector plus 1 degree' in f for f in checks.failures)


def contract_fixture():
    first,_,_=fixture()
    second=deepcopy(first)
    second['id']='C0002'
    second['jaw_sign']=-1.
    second['world_T_grip']=(np.diag([-1.,-1.,1.,1.])@np.asarray(first['world_T_grip'])).tolist()
    # Surface evidence is exercised separately above; this fixture tests the set contract.
    aliases=[first,second]
    group=dict(first,grouped_candidate_ids=['C0001','C0002'],jaw_aliases=aliases,source_family_ids=['D001_R001'])
    row=dict(scene='TEST',part='BOX',directions=[dict(id='D001',source='opposed_planar_facets',closing_axis_world=[1.,0.,0.])],
             contact_families=[dict(family_id='D001_R001',direction_id='D001',prepared=True,height_m=0.,pad_depth_m=0.,
                                   candidate_ids=['C0001','C0002'],valid_alias_ids=['C0001','C0002'],center_projection=first['center_projection'])],
             merged_contact_regions=1,prepared_contact_families=1,generated_count=2,valid_count=2,grouped_count=1,
             rejection_counts={},preparation_rejection_counts={},grasps=[group],center_projection_geometry_queries=2)
    return row,aliases


def test_surface_direction_contract_has_no_native_cad_axis_requirement():
    row,aliases=contract_fixture()
    checks=Checks()
    audit_contract(row,aliases,{},checks)
    assert checks.failed_count==0,checks.failures


def test_dropping_one_valid_checked_jaw_is_rejected():
    row,aliases=contract_fixture()
    row['grasps'][0]['grouped_candidate_ids']=['C0001']
    row['grasps'][0]['jaw_aliases']=[aliases[0]]
    checks=Checks()
    audit_contract(row,aliases,{},checks)
    assert any('every valid alternative retained once' in f for f in checks.failures)


def test_empty_case_is_explicit_and_has_no_fabricated_grasp():
    row=dict(scene='TEST',part='BOX',directions=[],contact_families=[],merged_contact_regions=0,
             prepared_contact_families=0,generated_count=0,valid_count=0,grouped_count=0,
             rejection_counts={},preparation_rejection_counts={},grasps=[],center_projection_geometry_queries=0)
    checks=Checks()
    audit_contract(row,[],{},checks)
    assert checks.failed_count==0,checks.failures


def cylinder_fit_fixture():
    angles=np.arange(24)*2*np.pi/24
    rim=np.c_[.005*np.cos(angles),.005*np.sin(angles)]
    triangles=[]
    for a,b in zip(rim,np.roll(rim,-1,axis=0)):
        lo_a,lo_b=np.r_[a,-.01],np.r_[b,-.01]
        hi_a,hi_b=np.r_[a,.01],np.r_[b,.01]
        triangles.extend(([lo_a,lo_b,hi_b],[lo_a,hi_b,hi_a]))
    triangles=np.asarray(triangles)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=float(np.linalg.norm(cross,axis=1).sum()/2)
    config=dict(cylinder_min_band_area_m2=16e-6,cylinder_min_surface_fraction=.015,
                cylinder_max_circle_residual_m=.000025,cylinder_max_circle_residual_fraction=.002,
                cylinder_min_axial_span_m=.003,cylinder_normal_perpendicular_deg=1.)
    model=dict(id='CYL001',shaft_axis_world=[0.,0.,1.],point_on_axis_world=[0.,0.,0.],
        fit_radius_m=.005,supporting_triangle_ids=list(range(len(triangles))),
        circle_vertex_residual95_m=0.,circle_vertex_residual_max_m=0.,axial_span_m=.02,axial_min_m=-.01,axial_max_m=.01,
        area_m2=area,direction_area_gate_m2=max(16e-6,area*.015),supports_direction_generation=True,
        classification_only=False,area_below_direction_gate=False,direction_support_cylinder_id=None)
    row=dict(scene='TEST',part='CYLINDER',directions=[dict(source='cylinder_radial',sources=[dict(type='cylinder_radial',cylinder_id='CYL001')])],
             direction_analysis=dict(cylinder_hypotheses=[model]))
    return row,triangles,config


def test_used_cylinder_fit_replays_native_vertex_residual_and_support_area():
    row,triangles,config=cylinder_fit_fixture()
    checks=Checks()
    audit_used_cylinders(row,[],triangles,config,checks)
    assert checks.failed_count==0,checks.failures
    row['direction_analysis']['cylinder_hypotheses'][0]['fit_radius_m']+=.0001
    checks=Checks()
    audit_used_cylinders(row,[],triangles,config,checks)
    assert any('every native support vertex' in f for f in checks.failures)


def test_small_classification_band_cannot_invent_an_unrelated_supporting_shaft():
    row,triangles,config=cylinder_fit_fixture()
    primary=row['direction_analysis']['cylinder_hypotheses'][0]
    small=deepcopy(primary)
    small.update(id='CYL002',supports_direction_generation=False,classification_only=True,
                 area_below_direction_gate=True,direction_support_cylinder_id='CYL001',coaxial_max_axis_distance_m=.001)
    small['point_on_axis_world']=[.001,0.,0.]
    row['direction_analysis']['cylinder_hypotheses'].append(small)
    row['directions'][0]['sources'].append(dict(type='cylinder_radial',cylinder_id='CYL002'))
    checks=Checks()
    audit_used_cylinders(row,[],triangles,config,checks)
    assert any('small band is classification-only' in f for f in checks.failures)
    assert any('both small-band endpoints' in f for f in checks.failures)


def test_center_shift_replays_first_contact_openings_not_nominal_ray_midpoint():
    c,_,_=fixture()
    c['center_projection'].update(first_contact_openings_before_m=[.013,.009],translation_along_closing_m=-.001)
    c['world_T_grip'][0][3]=-.001
    checks=Checks()
    audit_center_projection(c,checks)
    assert checks.failed_count==0,checks.failures
    c['center_projection']['translation_along_closing_m']=0.
    checks=Checks()
    audit_center_projection(c,checks)
    assert any('midpoint equation' in f for f in checks.failures)
    assert any('translation follows' in f for f in checks.failures)
