"""Experimental GeDi invalid-patch policy; never edits the original backend.

Only finite prepared patches enter the frozen network. Full original query rows,
sampling batches and both radius passes remain intact. Internal NaNs mark invalid
rows; only finite descriptors with explicit original indices leave this adapter.
This is an independent robustness policy, not verified author FreeZeV2 behavior.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class FinitePatchForward(nn.Module):
    def __init__(self, network: nn.Module, dimension: int):
        super().__init__()
        if network.training:
            raise ValueError("The descriptor network must already be in evaluation mode")
        self.network = network
        self.dimension = int(dimension)
        self.patch_validity = []
        self.neural_validity = []
        self.zero_descriptor_rows = []
        self.eval()

    def forward(self, patches):
        if patches.ndim != 3:
            raise ValueError("Expected prepared patches with shape N x 3 x samples")
        valid = torch.isfinite(patches).all(dim=(1, 2))
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        result = patches.new_full((len(patches), self.dimension), float("nan"))
        if len(indices):
            values = self.network(patches.index_select(0, indices))
            if values.shape != (len(indices), self.dimension):
                raise ValueError("Unexpected descriptor output dimensions")
            result.index_copy_(0, indices, values)
        self.patch_validity.append(valid.detach().cpu().numpy().copy())
        finite = torch.isfinite(result).all(dim=1)
        self.neural_validity.append((valid & ~finite).detach().cpu().numpy().copy())
        zero = finite & (torch.linalg.vector_norm(result, dim=1) == 0)
        self.zero_descriptor_rows.append(zero.detach().cpu().numpy().copy())
        return result

def extract_valid_rows(backend, queries, support, radii, *, seed=0):
    """Return aligned finite rows after BOTH original scale computations.

    Empty-neighborhood exceptions remain explicit failures. No jitter, tangent
    fallback, zero descriptor, changed radius or resampling is introduced.
    Callers must use original_indices for every point/visual-feature array.
    """
    radii = tuple(float(x) for x in radii)
    if not radii or any(not np.isfinite(x) or x <= 0 for x in radii):
        raise ValueError("Positive finite radii are required")
    original = backend.gedi_net
    wrapper = FinitePatchForward(original, backend.dim)
    old_radius, old_lrf_radius = backend.r_lrf, backend.lrf.r_lrf
    state = np.random.get_state()
    descriptors, diagnostics = [], []
    try:
        backend.gedi_net = wrapper
        np.random.seed(seed)
        with torch.inference_mode():
            for radius in radii:
                wrapper.patch_validity.clear()
                wrapper.neural_validity.clear()
                wrapper.zero_descriptor_rows.clear()
                backend.r_lrf = backend.lrf.r_lrf = radius
                value = np.asarray(backend.compute(pts=queries, pcd=support))
                if value.shape != (len(queries), backend.dim):
                    raise ValueError("Descriptor row count or dimensions changed")
                patch_ok = np.concatenate(wrapper.patch_validity)
                neural_bad = np.concatenate(wrapper.neural_validity)
                zero = np.concatenate(wrapper.zero_descriptor_rows)
                if len(patch_ok) != len(queries):
                    raise ValueError("Prepared-patch indices no longer align")
                diagnostics.append({
                    "radius": radius,
                    "invalid_patch_indices": np.flatnonzero(~patch_ok).tolist(),
                    "nonfinite_neural_indices": np.flatnonzero(neural_bad).tolist(),
                    "zero_descriptor_indices": np.flatnonzero(zero).tolist(),
                    "valid_rows": np.isfinite(value).all(axis=1),
                })
                descriptors.append(value)
    finally:
        backend.gedi_net = original
        backend.r_lrf, backend.lrf.r_lrf = old_radius, old_lrf_radius
        np.random.set_state(state)
    joint = np.logical_and.reduce([x["valid_rows"] for x in diagnostics])
    indices = np.flatnonzero(joint)
    values = np.concatenate(descriptors, axis=1)[indices].astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite descriptor escaped the validity policy")
    return {"original_indices": indices, "descriptors": values,
            "input_rows": len(queries), "valid_rows": len(indices),
            "invalid_rows": np.flatnonzero(~joint), "scales": diagnostics}
