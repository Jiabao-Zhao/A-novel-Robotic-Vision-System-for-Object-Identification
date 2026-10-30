import numpy as np
from scipy.spatial.transform import Rotation
from freezev2_repro.geometry import (depth_to_metres,backproject,project,SquareCrop,
    transform_points,observed_clouds,visible_surface,surface_sample)
from freezev2_repro.rendering import render_mesh,camera_poses,icosphere_directions


def test_bop_scaling_and_projection_direction():
    K=np.array([[500,0,320],[0,600,240],[0,0,1.]])
    z=depth_to_metres(np.array([10000]),.1)
    np.testing.assert_allclose(z,[1.])
    uv=np.array([[320,240],[420,360]])
    p=backproject(uv,np.array([1.,2.]),K)
    np.testing.assert_allclose(project(p,K)[0],uv)
    R=Rotation.from_euler('xyz',[80,-120,33],degrees=True).as_matrix();t=np.array([.1,-.2,1.])
    np.testing.assert_allclose((transform_points(p,R,t)-t)@R,p,atol=1e-12)


def test_crop_edge_padding_mapping_and_sparse_count():
    m=np.zeros((30,40),bool);m[2:12,0:20]=True
    crop=SquareCrop.from_mask(m)
    uv=crop.patch_centres();np.testing.assert_allclose(crop.to_full(crop.from_full(uv)),uv)
    assert crop.top<0 and crop.side==20
    frame=observed_clouds(np.ones(m.shape),m,np.eye(3),dense_count=30)
    assert len(frame['sparse'])<=256 and len(frame['dense'])==30
    xy=frame['pixel_uv'];assert m[xy[:,1],xy[:,0]].all()
    np.testing.assert_allclose(frame['sparse'][:,2],1)


def test_rasterizer_perspective_depth_and_visibility():
    v=np.array([[-.2,-.2,1],[.2,-.2,1],[0,.2,1.]])
    tri=np.array([[0,1,2]]);K=np.array([[50,0,31.5],[0,50,31.5],[0,0,1.]])
    im=render_mesh(v,tri,np.ones_like(v),np.eye(3),np.zeros(3),K,height=64,width=64)
    assert im['depth_m'][30,32]==1
    p=backproject(np.array([[32,30]]),np.array([im['depth_m'][30,32]]),K)
    uv,vis=visible_surface(np.vstack([p,p+[0,0,.1]]),None,np.eye(3),np.zeros(3),K,im['depth_m'],.001)
    assert vis.tolist()==[True,False]
    np.testing.assert_allclose(uv[0],[32,30])


def test_162_cameras_preserve_cad_metric_scale():
    v=np.array([[-.02,-.01,-.03],[.02,.01,.03]])
    original=v.copy();K,poses=camera_poses(v)
    assert len(poses)==162 and len(icosphere_directions())==162
    np.testing.assert_array_equal(v,original)
    for R,t in poses:
        assert np.linalg.det(R)>0.999999
        assert (transform_points(v,R,t)[:,2]>0).all()


def test_surface_sampling_native_metres():
    v=np.array([[0.,0,0],[.01,0,0],[0,.02,0]])
    p,n,f=surface_sample(v,np.array([[0,1,2]]),1000,42)
    assert p.max()<.020001 and np.all(p>=0)
    np.testing.assert_allclose(n,np.tile([0,0,1.],(1000,1)))
