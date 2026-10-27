import numpy as np

from .directions import surface_directions


def prism(section, height):
    n=len(section)
    v=np.array([[x,y,z] for z in (-height/2,height/2) for x,y in section])
    f=[]
    for i in range(n):
        j=(i+1)%n
        f.extend(((i,j,n+j),(i,n+j,n+i)))
    for i in range(1,n-1):
        f.extend(((0,i+1,i),(n,n+i,n+i+1)))
    return v,np.asarray(f)


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def mesh_cylinder(phase=0.,radius=.008,height=.05):
    theta=np.linspace(0,2*np.pi,72,endpoint=False)+phase
    return prism(np.c_[radius*np.cos(theta),radius*np.sin(theta)],height)


def axis_set(result):
    return [np.asarray(d['closing_axis_world']) for d in result['directions']]


def matching(axis, collection, tolerance=1e-6):
    return any(abs(np.asarray(axis)@other) >= 1-tolerance for other in collection)


def test_box_uses_actual_surface_normals_after_world_rotation():
    v,f=prism(np.array([[-.008,-.005],[.008,-.005],[.008,.005],[-.008,.005]]),.05)
    rotation=rz(np.deg2rad(31.))
    result=surface_directions(v@rotation.T,f)
    axes=axis_set(result)
    assert len(axes)==2
    assert matching(rotation[:,0],axes)
    assert matching(rotation[:,1],axes)
    assert not result['cylinder_hypotheses']


def test_lying_cylinder_side_is_invariant_to_arbitrary_transverse_roll():
    tilt=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
    world=rz(.43)@tilt
    shaft=world[:,2]
    expected=np.cross([0,0,1.],shaft)
    expected/=np.linalg.norm(expected)
    for phase in (.0,.183,1.09):
        v,f=mesh_cylinder(phase)
        result=surface_directions(v@world.T+[.31,-.2,.01],f)
        assert result['cylinder_hypotheses']
        radial=[np.asarray(d['closing_axis_world']) for d in result['directions'] if d['source']=='cylinder_radial']
        assert len(radial)==1
        assert matching(expected,radial,1e-10)
        assert all(abs(d[2])<1e-12 for d in axis_set(result))


def test_vertical_cylinder_has_two_canonical_radial_representatives():
    for phase in (.0,.217,.749):
        v,f=mesh_cylinder(phase)
        result=surface_directions(v,f)
        axes=axis_set(result)
        assert len(axes)==2
        assert matching([1,0,0],axes)
        assert matching([0,1,0],axes)
        assert result['metadata']['uses_native_CAD_frame'] is False
        assert result['metadata']['no_whole_object_symmetry_claim'] is True


def test_hexagonal_flats_are_not_mistaken_for_a_round_surface():
    theta=np.arange(6)*np.pi/3+.13
    v,f=prism(np.c_[.015*np.cos(theta),.015*np.sin(theta)],.012)
    result=surface_directions(v,f)
    assert not result['cylinder_hypotheses']
    axes=axis_set(result)
    assert len(axes)==3
    for angle in theta[:3]+np.pi/6:
        assert matching([np.cos(angle),np.sin(angle),0],axes)


def test_tilted_flat_normals_report_rejection_instead_of_silent_yaw():
    v,f=prism(np.array([[-.008,-.005],[.008,-.005],[.008,.005],[-.008,.005]]),.05)
    angle=np.deg2rad(12.)
    rotation=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    result=surface_directions(v@rotation.T,f)
    assert len(axis_set(result))==1
    assert matching([1,0,0],axis_set(result))
    assert any(abs(r['projection_error_deg']-12)<1e-6 for r in result['rejected_flat_directions'])


def test_thin_tessellation_strip_is_not_called_a_broad_flat_contact():
    v,f=prism(np.array([[-.01,-.0005],[.01,-.0005],[.01,.0005],[-.01,.0005]]),.05)
    result=surface_directions(v,f)
    assert len(axis_set(result))==1
    assert matching([0,1,0],axis_set(result))
    patches=result['directions'][0]['sources'][0]['positive_planar_patches']
    assert all(min(p['support_spans_m'])>=.002 for p in patches)


def test_supporting_triangle_ids_keep_original_mesh_indices():
    v,f=mesh_cylinder()
    f=np.vstack(([0,0,0],f))
    result=surface_directions(v,f)
    assert result['cylinder_hypotheses']
    for hypothesis in result['cylinder_hypotheses']:
        ids=hypothesis['supporting_triangle_ids']
        assert min(ids)>=1
        assert max(ids)<len(f)


def test_close_coaxial_radius_bands_are_fitted_separately():
    first,fa=mesh_cylinder(radius=.0069,height=.006)
    second,fb=mesh_cylinder(radius=.0070667,height=.006)
    first[:,2]-=.003
    second[:,2]+=.003
    v=np.vstack((first,second))
    f=np.vstack((fa,fb+len(first)))
    result=surface_directions(v,f)
    radii=[h['fit_radius_m'] for h in result['cylinder_hypotheses']]
    assert any(abs(r-.0069)<1e-8 for r in radii)
    assert any(abs(r-.0070667)<1e-8 for r in radii)
    assert all(h['circle_vertex_residual_max_m']<=.000025 for h in result['cylinder_hypotheses'])


def test_elliptical_section_is_not_accepted_as_a_circular_cylinder():
    theta=np.linspace(0,2*np.pi,72,endpoint=False)
    v,f=prism(np.c_[.008*np.cos(theta),.006*np.sin(theta)],.05)
    result=surface_directions(v,f)
    assert not result['cylinder_hypotheses']


def test_small_exact_coaxial_band_only_classifies_contacts():
    large,fa=mesh_cylinder(radius=.005,height=.02)
    small,fb=mesh_cylinder(radius=.0005,height=.0032)
    small[:,2]+=.015
    v=np.vstack((large,small))
    f=np.vstack((fa,fb+len(large)))
    result=surface_directions(v,f)
    weak=[h for h in result['cylinder_hypotheses'] if h['classification_only']]
    assert weak
    assert any(abs(h['fit_radius_m']-.0005)<1e-8 for h in weak)
    assert all(h['supports_direction_generation'] is False for h in weak)
    assert all(h['area_m2']<h['direction_area_gate_m2'] for h in weak)
    assert all(h['direction_support_cylinder_id'] for h in weak)
    assert len(result['directions'])==2
    weak_ids={h['id'] for h in weak}
    assert not any(s.get('cylinder_id') in weak_ids for d in result['directions'] for s in d['sources'])
