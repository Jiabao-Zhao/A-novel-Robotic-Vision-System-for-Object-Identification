import numpy as np
from .pair_reduction import original_pair_sparsity,canonical_pairs
from .run_compare import project_pair


def test_original_sparsity_needs_both_endpoints_close():
    pairs=np.array([[[0,0,0],[.02,0,0]],[[.001,0,0],[.021,0,0]],[[.001,0,0],[.04,0,0]]])
    kept=original_pair_sparsity(pairs,.005)
    assert np.array_equal(kept,pairs[[0,2]])


def test_pair_canonicalization_does_not_change_geometry():
    a=np.array([[[.02,0,0],[0,0,0]]])
    b=canonical_pairs(a)
    assert np.array_equal(b,a[:,::-1])
    assert np.array_equal(a,np.array([[[.02,0,0],[0,0,0]]]))


def test_fixed_down_projection_reports_original_tilt():
    angle=np.deg2rad(10.)
    pair=np.array([[0,0,0],[.02*np.cos(angle),0,.02*np.sin(angle)]])
    x,width,tilt=project_pair(pair)
    assert np.allclose(x,[1,0,0])
    assert abs(width-.02)<1e-12
    assert abs(tilt-10.)<1e-10
