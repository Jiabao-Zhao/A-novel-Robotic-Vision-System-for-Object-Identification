"""Keep finite GeDi query rows aligned without changing geometry-only clouds.

The caller selects the frozen source_snapshot on sys.path before scoring. These
helpers do not import a model, sample points, normalize descriptors or read GT.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _matrix(value: Any, name: str, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if (result.ndim != 2 or min(result.shape) < 1
            or (columns is not None and result.shape[1] != columns)):
        shape = f"(N,{columns})" if columns is not None else "(N,D)"
        raise ValueError(f"{name} must be a nonempty {shape} matrix")
    if (not np.issubdtype(result.dtype, np.number)
            or np.iscomplexobj(result) or not np.isfinite(result).all()):
        raise ValueError(f"{name} must contain finite real numeric values")
    return result


def _nonzero_descriptors(value: np.ndarray, name: str) -> None:
    # Do not normalize: this guard must preserve the exact saved descriptors.
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError(f"{name} contains nonfinite or zero-length descriptor rows")


def align_rows(points, visual, geometric, original_indices) -> dict[str, np.ndarray]:
    """Align retained geometric(M,G) to original points(N,3)/visual(N,D).

    original_indices contains M unique, strictly increasing original row IDs.
    Invalid rows have already been excluded by the versioned GeDi policy. A
    dense support cloud is deliberately not accepted or subset by this helper.
    """
    points = _matrix(points, "points", 3)
    visual = _matrix(visual, "visual")
    geometric = _matrix(geometric, "geometric")
    if len(points) != len(visual):
        raise ValueError("points and visual must have the same original row count")
    indices = np.asarray(original_indices)
    if (indices.ndim != 1 or len(indices) == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)):
        raise ValueError("original_indices must be a nonempty 1D integer array")
    if len(geometric) != len(indices):
        raise ValueError("geometric rows must match original_indices")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("original_indices must be in range for original points")
    # Comparison instead of np.diff avoids wraparound for unsigned integers.
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("original_indices must be unique and strictly increasing")
    _nonzero_descriptors(visual, "visual")
    _nonzero_descriptors(geometric, "geometric")
    indices = indices.astype(np.int64, copy=True)
    return {"points": points[indices], "visual": visual[indices],
            "geometric": geometric, "original_indices": indices}


def paired_full_geometry_score(feature_cad, sparse, R, t, diameter, C,
                               full_cad, dense, registration_config=None) -> dict:
    """Use frozen feature scoring plus original full-CAD geometry coverage.

    C indexes retained feature_cad and sparse rows. The feature denominator is
    the retained sparse count, as in the frozen score_pose call. The geometry
    denominator remains len(full_cad), including rows lacking valid GeDi. Full
    CAD and dense observation clouds must also be used unchanged by ICP upstream.
    """
    # Lazy import keeps row-alignment tests and importers independent of the
    # experiment's external source path; caller must select the sealed snapshot.
    from freezev2_repro.registration import score_pose
    from scipy.spatial import cKDTree

    full = _matrix(full_cad, "full_cad", 3).astype(np.float64, copy=False)
    observed_dense = _matrix(dense, "dense", 3).astype(np.float64, copy=False)
    score = score_pose(feature_cad, sparse, R, t, diameter, C,
                       dense_xyz=None, config=registration_config)
    # score_pose above validates the proper CAD-to-camera pose and the exact
    # configured strict inlier gate; do not duplicate or retune either here.
    transformed = full @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64).reshape(3)
    distances, _ = cKDTree(observed_dense).query(transformed, k=1)
    geometry_count = int((distances < score.threshold_m).sum())
    return {"feature_score": score.feature_score,
            "geometry_score": geometry_count / len(full),
            "feature_inlier_pairs": score.feature_inlier_pairs,
            "sparse_observed_count": score.sparse_observed_count,
            "geometry_inlier_points": geometry_count,
            "cad_count": len(full), "feature_cad_count": score.cad_count,
            "threshold_m": score.threshold_m}
