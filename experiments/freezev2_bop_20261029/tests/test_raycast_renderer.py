import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from freezev2_repro.rendering import RaycastRenderer, render_mesh


def test_native_raycast_matches_metric_rasterizer():
    pytest.importorskip('open3d')
    vertices=np.array([[-.04,-.04,0],[.04,-.04,0],[.04,.04,0],[-.04,.04,0]])
    triangles=np.array([[0,1,2],[0,2,3]])
    colors=np.array([[.3,.5,.7]]*4)
    K=np.array([[100.,0,31.5],[0,100,31.5],[0,0,1]])
    R=Rotation.from_euler('xyz',[21,31,42],degrees=True).as_matrix()
    t=np.array([.001,.002,.2])
    slow=render_mesh(vertices,triangles,colors,R,t,K,height=64,width=64)
    fast=RaycastRenderer(vertices,triangles,colors).render(R,t,K,height=64,width=64)
    both=(slow['depth_m']>0)&(fast['depth_m']>0)
    assert both.sum()>100
    assert np.array_equal(slow['depth_m']>0,fast['depth_m']>0)
    assert np.max(np.abs(slow['depth_m'][both]-fast['depth_m'][both]))<1e-7
    assert np.max(np.abs(slow['rgb'].astype(float)-fast['rgb']))<=1
