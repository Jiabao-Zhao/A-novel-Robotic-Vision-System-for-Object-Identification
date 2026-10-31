import types
import numpy as np
import pytest
import torch
from torch import nn
from gedi_valid_rows import FinitePatchForward, extract_valid_rows

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.eval()
    def forward(self, x):
        assert torch.isfinite(x).all()
        self.calls.append(len(x))
        return x.mean(dim=(1, 2))[:, None].repeat(1, 2)

class Backend:
    dim = 2
    r_lrf = 7.
    def __init__(self, invalid=None):
        self.gedi_net = Network()
        self.lrf = types.SimpleNamespace(r_lrf=8.)
        self.invalid = invalid or {}
        self.fail = False
    def compute(self, pts, pcd):
        np.random.random(5)
        if self.fail:
            raise RuntimeError("empty-neighborhood fixture")
        x = pts[:, :1].reshape(-1, 1, 1).repeat(1, 3, 4)
        for i in self.invalid.get(self.r_lrf, []):
            x[i] = float("nan")
        return self.gedi_net(x).detach().numpy()

def inputs(n=4):
    return torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3), torch.zeros(5, 3)

def test_two_scale_intersection_and_original_indices():
    b=Backend({1.:[1], 2.:[2]});q,s=inputs();net=b.gedi_net
    result=extract_valid_rows(b,q,s,[1.,2.])
    np.testing.assert_array_equal(result["original_indices"],[0,3])
    np.testing.assert_array_equal(result["descriptors"],[[0]*4,[3]*4])
    assert net.calls == [3,3]
    assert b.gedi_net is net and (b.r_lrf,b.lrf.r_lrf)==(7.,8.)
    assert result["scales"][0]["invalid_patch_indices"]==[1]
    assert result["scales"][1]["invalid_patch_indices"]==[2]

def test_all_invalid_never_calls_network():
    b=Backend({1.:list(range(4)),2.:list(range(4))});q,s=inputs()
    result=extract_valid_rows(b,q,s,[1.,2.])
    assert result["descriptors"].shape==(0,4)
    assert b.gedi_net.calls==[]

def test_exception_restores_backend_and_numpy_rng():
    b=Backend();b.fail=True;q,s=inputs();net=b.gedi_net
    np.random.seed(81);state=np.random.get_state()
    expected=np.random.random(4);np.random.set_state(state)
    with pytest.raises(RuntimeError,match="empty-neighborhood"):
        extract_valid_rows(b,q,s,[1.,2.])
    assert b.gedi_net is net and (b.r_lrf,b.lrf.r_lrf)==(7.,8.)
    np.testing.assert_array_equal(np.random.random(4),expected)

def test_success_restores_numpy_rng():
    b=Backend();q,s=inputs();np.random.seed(92);state=np.random.get_state()
    expected=np.random.random(4);np.random.set_state(state)
    extract_valid_rows(b,q,s,[1.,2.])
    np.testing.assert_array_equal(np.random.random(4),expected)

def test_training_network_rejected():
    with pytest.raises(ValueError,match="evaluation"):
        FinitePatchForward(Network().train(),2)

def test_bad_neural_output_recorded_separately():
    class Broken(Network):
        def forward(self,x):
            v=super().forward(x);v[1]=float("nan");return v
    b=Backend();b.gedi_net=Broken();q,s=inputs()
    result=extract_valid_rows(b,q,s,[1.,2.])
    assert result["scales"][0]["invalid_patch_indices"]==[]
    assert result["scales"][0]["nonfinite_neural_indices"]==[1]
    np.testing.assert_array_equal(result["original_indices"],[0,2,3])

def test_frozen_batchnorm_statistics_unchanged():
    network=nn.Sequential(nn.Flatten(),nn.Linear(12,2),nn.BatchNorm1d(2)).eval()
    before_mean=network[2].running_mean.clone();before_var=network[2].running_var.clone()
    x=torch.arange(48,dtype=torch.float32).reshape(4,3,4);x[1]=float("nan")
    with torch.inference_mode():
        expected=network(x[[0,2,3]])
        actual=FinitePatchForward(network,2)(x)
    torch.testing.assert_close(actual[[0,2,3]],expected,rtol=0,atol=0)
    assert torch.isnan(actual[1]).all()
    torch.testing.assert_close(network[2].running_mean,before_mean,rtol=0,atol=0)
    torch.testing.assert_close(network[2].running_var,before_var,rtol=0,atol=0)
