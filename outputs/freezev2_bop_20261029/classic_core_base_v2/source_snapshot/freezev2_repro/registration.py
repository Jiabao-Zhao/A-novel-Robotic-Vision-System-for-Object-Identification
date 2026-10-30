"""FreeZeV2 rigid registration, refinement and candidate scores.

All coordinates and translations are in metres.  A pose maps an original CAD
point to the camera frame: ``camera_xyz = cad_xyz @ R.T + t``.  CAD scale and
orientation are never restricted or normalised here.  Ground truth is not an
input to any function in this module.

This is a new implementation from the FreeZeV2 method description.  See
REGISTRATION_NOTES.md for specified settings versus reproduction assumptions.
The NumPy backend is an explicit CPU reference; ``torch_cuda`` is an opt-in
batched CUDA engine, and fails if CUDA is unavailable rather than falling back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RegistrationConfig:
    """Method settings and named implementation assumptions.

    The paper specifies k=10, 10,000 hypotheses, a 0.03*diameter inlier
    threshold and three unit scoring exponents.  Sampling, triangle pruning,
    batch sizes, CPU/CUDA precision and ICP stopping rules are reproducibility
    choices.  They are recorded, not presented as published hyperparameters.
    """

    k: int = 10
    iterations: int = 10_000
    inlier_fraction: float = 0.03
    seed: int = 0
    edge_length_ratio: float = 0.9
    min_triangle_area_ratio: float = 1e-6
    hypothesis_backend: str = "numpy"
    hypothesis_batch_size: int = 64
    torch_dtype: str = "float32"
    icp_max_iterations: int = 30
    icp_translation_tolerance_m: float = 1e-8
    icp_rotation_tolerance_rad: float = 1e-7
    icp_stop_when_correspondences_stable: bool = True


@dataclass(frozen=True)
class Correspondences:
    """The fixed top-k set C, one row for each sparse observed point.

    ``cad_indices[i,j]`` is a CAD index paired with observed point ``i``;
    ``similarities[i,j]`` is its cosine similarity.  Rows include *all* k
    correspondences, not just one correspondence per observed point.  This
    object is reused for coarse and refined scoring.
    """

    cad_indices: np.ndarray
    similarities: np.ndarray

    def __iter__(self):
        """Allow ``indices, similarities = topk_correspondences(...)``."""
        yield self.cad_indices
        yield self.similarities


@dataclass
class PoseResult:
    """One coarse pose or an explicit unavailable result; no identity fallback."""

    R: np.ndarray | None
    t: np.ndarray | None
    score: float | None
    status: str
    correspondences: Correspondences
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ICPResult:
    """Refinement result, including final CAD-to-observation inlier fraction."""

    R: np.ndarray
    t: np.ndarray
    geometry_score: float
    inlier_count: int
    rmse_m: float | None
    iterations: int
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseScores:
    """Eq. 5 feature score and the one-sided CAD geometry score.

    Feature score is sum(cosine over inlier *pairs*) / N_sparse.  It is not
    bounded by one when k>1, and must not be clipped or divided by N_sparse*k.
    Geometry score is the fraction of CAD points whose nearest dense observed
    neighbour is within the threshold.  Scores are not pose-error metrics.
    """

    feature_score: float
    geometry_score: float | None
    feature_inlier_pairs: int
    sparse_observed_count: int
    geometry_inlier_points: int | None
    cad_count: int
    threshold_m: float


@dataclass
class CandidateRanking:
    """Ranked original candidates and explicit suppression/failure records."""

    selected: list[Any]
    suppressed: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


def _config(config: RegistrationConfig | Mapping[str, Any] | None) -> RegistrationConfig:
    if config is None:
        value = RegistrationConfig()
    elif isinstance(config, RegistrationConfig):
        value = config
    elif isinstance(config, Mapping):
        names = {item.name for item in fields(RegistrationConfig)}
        unknown = set(config) - names
        if unknown:
            raise ValueError(f"Unknown registration settings: {sorted(unknown)}")
        value = RegistrationConfig(**config)
    else:
        raise TypeError("config must be RegistrationConfig, a mapping, or None")
    for name in ("k", "iterations", "hypothesis_batch_size", "icp_max_iterations", "seed"):
        number = getattr(value, name)
        if isinstance(number, bool) or not isinstance(number, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if value.k < 1 or value.iterations < 1 or value.hypothesis_batch_size < 1:
        raise ValueError("k, iterations and hypothesis_batch_size must be positive")
    if not np.isfinite(value.inlier_fraction) or value.inlier_fraction <= 0:
        raise ValueError("inlier_fraction must be positive and finite")
    if not 0 <= value.edge_length_ratio <= 1:
        raise ValueError("edge_length_ratio must be in [0,1]")
    if not np.isfinite(value.min_triangle_area_ratio) or value.min_triangle_area_ratio < 0:
        raise ValueError("min_triangle_area_ratio must be finite and nonnegative")
    if value.hypothesis_backend not in {"numpy", "torch_cuda"}:
        raise ValueError("hypothesis_backend must be 'numpy' or 'torch_cuda'")
    if value.torch_dtype not in {"float32", "float64"}:
        raise ValueError("torch_dtype must be float32 or float64")
    if value.icp_max_iterations < 1:
        raise ValueError("icp_max_iterations must be positive")
    for tolerance in (value.icp_translation_tolerance_m, value.icp_rotation_tolerance_rad):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("ICP tolerances must be finite and nonnegative")
    return value


def _points(value: Any, name: str, minimum: int = 1) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < minimum:
        raise ValueError(f"{name} must have shape (N,3), N>={minimum}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return result


def _features(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty (N,D) array")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError(f"{name} contains zero-length descriptor rows")
    return result / norms


def _threshold(diameter_m: float, fraction: float) -> float:
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("diameter_m must be finite and positive, in metres")
    return float(diameter_m * fraction)


def _pose(R: Any, t: Any) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape not in {(3,), (3, 1)}:
        raise ValueError("R must have shape (3,3); t must have shape (3,) or (3,1)")
    translation = translation.reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Pose contains nonfinite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("R must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0):
        raise ValueError("R must be a proper rigid rotation with determinant +1")
    return rotation, translation


def topk_correspondences(cad_features: Any, observed_features: Any, k: int = 10) -> Correspondences:
    """Return all N_sparse*k CAD/observation pairs ranked by cosine.

    Combined visual/geometric descriptors are normalised for cosine here,
    regardless of their previous branch normalisation.  Stable CAD-index tie
    breaking and float64 matrix products make the CPU reference deterministic.
    k must not exceed the CAD point count; it is never silently reduced.
    """
    cad, observed = _features(cad_features, "cad_features"), _features(observed_features, "observed_features")
    if cad.shape[1] != observed.shape[1]:
        raise ValueError("CAD and observed descriptor dimensions disagree")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(cad):
        raise ValueError("k must be a positive integer no greater than the CAD descriptor count")
    cosine = np.clip(observed @ cad.T, -1.0, 1.0)
    order = np.argsort(-cosine, axis=1, kind="stable")[:, :k]
    return Correspondences(order.astype(np.int64), np.take_along_axis(cosine, order, axis=1))


def rigid_fit(cadpoints: Any, observedpoints: Any) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares proper rigid fit of paired CAD points to camera points.

    Uses an unrestricted 3D Kabsch/SVD fit and no scale parameter.  Collinear
    or coincident inputs cannot determine a complete rotation and are rejected.
    Coplanar noncollinear points do determine a proper 3D rigid transform.
    """
    source, target = _points(cadpoints, "cadpoints", 3), _points(observedpoints, "observedpoints", 3)
    if source.shape != target.shape:
        raise ValueError("Paired point arrays must have identical shapes")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    left, right = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(left) < 2 or np.linalg.matrix_rank(right) < 2:
        raise ValueError("Collinear/coincident correspondences do not determine 3D rigid rotation")
    U, _, Vt = np.linalg.svd(left.T @ right)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(Vt.T @ U.T)
    rotation = Vt.T @ correction @ U.T
    return rotation, target_mean - rotation @ source_mean


def _check_correspondences(value: Correspondences, n_cad: int, n_observed: int) -> Correspondences:
    if not isinstance(value, Correspondences):
        raise TypeError("correspondences must be a Correspondences object")
    indices, similarities = np.asarray(value.cad_indices), np.asarray(value.similarities, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[0] != n_observed or indices.shape[1] < 1:
        raise ValueError("correspondences must have one nonempty row per sparse observed point")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= n_cad):
        raise ValueError("correspondence CAD indices are out of bounds or noninteger")
    if similarities.shape != indices.shape or not np.isfinite(similarities).all():
        raise ValueError("correspondence similarities must be finite and match index shape")
    if np.any(np.abs(similarities) > 1 + 1e-8):
        raise ValueError("correspondence similarities must be cosine values in [-1,1]")
    return Correspondences(indices.astype(np.int64, copy=False), similarities)


def _pairs(cad: np.ndarray, observed: np.ndarray, pairs: Correspondences):
    candidates = cad[pairs.cad_indices.reshape(-1)]
    paired_observed = np.repeat(observed, pairs.cad_indices.shape[1], axis=0)
    return candidates, paired_observed, pairs.similarities.reshape(-1)


def _fit_batches(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=1), target.mean(axis=1)
    covariance = np.einsum("bni,bnj->bij", source - source_mean[:, None], target - target_mean[:, None])
    U, _, Vt = np.linalg.svd(covariance)
    V, Ut = Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)
    signs = np.where(np.linalg.det(V @ Ut) < 0, -1.0, 1.0)
    V = V.copy()
    V[:, :, -1] *= signs[:, None]
    rotations = V @ Ut
    translations = target_mean - np.einsum("bij,bj->bi", rotations, source_mean)
    return rotations, translations


def _triangle_validity(source: np.ndarray, target: np.ndarray, config: RegistrationConfig):
    first, second = np.array([0, 0, 1]), np.array([1, 2, 2])
    source_lengths = np.linalg.norm(source[:, first] - source[:, second], axis=2)
    target_lengths = np.linalg.norm(target[:, first] - target[:, second], axis=2)
    maximum = np.maximum(source_lengths, target_lengths)
    edge_ok = np.all(np.minimum(source_lengths, target_lengths) >= config.edge_length_ratio * maximum, axis=1)
    edge_ok &= np.all(maximum > 0, axis=1)
    source_area = np.linalg.norm(np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]), axis=1)
    target_area = np.linalg.norm(np.cross(target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]), axis=1)
    source_scale = source_lengths.max(axis=1) ** 2
    target_scale = target_lengths.max(axis=1) ** 2
    nondegenerate = (source_area > config.min_triangle_area_ratio * source_scale) & (target_area > config.min_triangle_area_ratio * target_scale)
    return nondegenerate, edge_ok


def _numpy_hypotheses(source, target, candidate_xyz, paired_xyz, similarities, threshold_m, denominator):
    rotations, translations = _fit_batches(source, target)
    projected = np.einsum("bij,pj->bpi", rotations, candidate_xyz)
    projected += translations[:, None]
    residual = projected - paired_xyz[None]
    inliers = np.einsum("bpi,bpi->bp", residual, residual) < threshold_m ** 2
    scores = (inliers * similarities[None]).sum(axis=1) / denominator
    return rotations, translations, scores, inliers.sum(axis=1)


def _cuda_hypotheses(source, target, candidate_xyz, paired_xyz, similarities, threshold_m, denominator, config):
    """Explicit CUDA batched SVD/scoring. No hidden CPU or model fallback."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch_cuda was requested but PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch_cuda was requested but torch.cuda.is_available() is false")
    dtype = torch.float32 if config.torch_dtype == "float32" else torch.float64
    with torch.inference_mode():
        left = torch.as_tensor(source, dtype=dtype, device="cuda")
        right = torch.as_tensor(target, dtype=dtype, device="cuda")
        left_mean, right_mean = left.mean(dim=1), right.mean(dim=1)
        covariance = (left - left_mean[:, None]).transpose(1, 2) @ (right - right_mean[:, None])
        U, _, Vh = torch.linalg.svd(covariance)
        V, Ut = Vh.transpose(1, 2).clone(), U.transpose(1, 2)
        signs = torch.where(torch.linalg.det(V @ Ut) < 0, -1.0, 1.0)
        V[:, :, -1] *= signs[:, None]
        rotations = V @ Ut
        translations = right_mean - (rotations @ left_mean[:, :, None]).squeeze(2)
        candidates = torch.as_tensor(candidate_xyz, dtype=dtype, device="cuda")
        paired = torch.as_tensor(paired_xyz, dtype=dtype, device="cuda")
        cosine = torch.as_tensor(similarities, dtype=dtype, device="cuda")
        projected = candidates[None] @ rotations.transpose(1, 2) + translations[:, None]
        inliers = ((projected - paired[None]) ** 2).sum(dim=2) < threshold_m ** 2
        scores = (inliers * cosine[None]).sum(dim=1) / denominator
        return tuple(array.detach().cpu().numpy() for array in (rotations, translations, scores, inliers.sum(dim=1)))


def feature_ransac(
    cad_xyz: Any,
    observed_xyz: Any,
    cad_features: Any,
    observed_features: Any,
    diameter_m: float,
    config: RegistrationConfig | Mapping[str, Any] | None = None,
) -> PoseResult:
    """Feature-aware global rigid RANSAC with one fixed Ns*k match set.

    Each of the fixed-budget hypotheses samples three distinct observed rows
    uniformly, then one of their k CAD candidates independently and uniformly.
    Repeated CAD points and degenerate/edge-inconsistent triples are rejected
    without resampling.  Every valid hypothesis is scored by all of C.  The
    best hypothesis is returned directly; no unrequested all-inlier refit is
    added.  The seed, pruning and backend are recorded in diagnostics.
    """
    cfg = _config(config)
    cad, observed = _points(cad_xyz, "cad_xyz", 3), _points(observed_xyz, "observed_xyz", 3)
    if len(cad_features) != len(cad) or len(observed_features) != len(observed):
        raise ValueError("Each feature row must correspond to its point row")
    threshold_m = _threshold(diameter_m, cfg.inlier_fraction)
    pairs = topk_correspondences(cad_features, observed_features, cfg.k)
    # Fail explicitly before consuming hypotheses if the requested CUDA backend
    # cannot run, including cases where all sample triples might be pruned.
    if cfg.hypothesis_backend == "torch_cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch_cuda requested but PyTorch is not installed") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda requested but CUDA is unavailable")
    rng = np.random.default_rng(cfg.seed)
    observed_ids = np.stack([rng.choice(len(observed), 3, replace=False) for _ in range(cfg.iterations)])
    neighbour_ids = rng.integers(cfg.k, size=(cfg.iterations, 3))
    cad_ids = pairs.cad_indices[observed_ids, neighbour_ids]
    distinct_cad = (cad_ids[:, 0] != cad_ids[:, 1]) & (cad_ids[:, 1] != cad_ids[:, 2]) & (cad_ids[:, 0] != cad_ids[:, 2])
    source, target = cad[cad_ids], observed[observed_ids]
    nondegenerate, edge_ok = _triangle_validity(source, target, cfg)
    valid_ids = np.flatnonzero(distinct_cad & nondegenerate & edge_ok)
    diagnostics = {
        "config": asdict(cfg), "pose_direction": "CAD_to_camera", "units": "m",
        "requested_hypotheses": cfg.iterations, "valid_hypotheses": int(len(valid_ids)),
        "repeated_cad_triplets": int((~distinct_cad).sum()),
        "degenerate_triplets": int((distinct_cad & ~nondegenerate).sum()),
        "edge_rejected_triplets": int((distinct_cad & nondegenerate & ~edge_ok).sum()),
        "correspondence_pairs": int(pairs.cad_indices.size), "sparse_observed_count": len(observed),
        "feature_score_denominator": "N_sparse_observed", "threshold_m": threshold_m,
        "hypothesis_backend": cfg.hypothesis_backend,
        "numeric_dtype": "float64" if cfg.hypothesis_backend == "numpy" else cfg.torch_dtype,
        "sampling": "uniform_distinct_observed_rows_then_uniform_topk_per_row_fixed_budget_no_rejected_sample_replacement",
    }
    if not len(valid_ids):
        return PoseResult(None, None, None, "no_valid_hypothesis", pairs, diagnostics)
    candidate_xyz, paired_xyz, similarities = _pairs(cad, observed, pairs)
    best_score, best_rotation, best_translation, best_id, best_inliers = -np.inf, None, None, None, None
    for start in range(0, len(valid_ids), cfg.hypothesis_batch_size):
        ids = valid_ids[start : start + cfg.hypothesis_batch_size]
        args = (source[ids], target[ids], candidate_xyz, paired_xyz, similarities, threshold_m, len(observed))
        if cfg.hypothesis_backend == "numpy":
            rotations, translations, scores, counts = _numpy_hypotheses(*args)
        else:
            rotations, translations, scores, counts = _cuda_hypotheses(*args, cfg)
        index = int(np.argmax(scores))
        # Strict > makes equal scores use the earliest sampled hypothesis.
        if float(scores[index]) > best_score:
            best_score = float(scores[index])
            best_rotation, best_translation = rotations[index].astype(np.float64), translations[index].astype(np.float64)
            best_id, best_inliers = int(ids[index]), int(counts[index])
    diagnostics.update(best_hypothesis_index=best_id, best_inlier_pairs=best_inliers)
    return PoseResult(best_rotation, best_translation, best_score, "ok", pairs, diagnostics)


def score_pose(
    cad_xyz: Any,
    observed_xyz: Any,
    R: Any,
    t: Any,
    diameter_m: float,
    correspondences: Correspondences,
    dense_xyz: Any | None = None,
    config: RegistrationConfig | Mapping[str, Any] | None = None,
) -> PoseScores:
    """Evaluate a pose using the *saved* top-k C and optional dense geometry.

    To obtain coarse and refined feature terms, call with the same C at their
    respective poses.  Descriptor matching is not recomputed after ICP.
    """
    cfg = _config(config)
    cad, observed = _points(cad_xyz, "cad_xyz"), _points(observed_xyz, "observed_xyz")
    rotation, translation = _pose(R, t)
    pairs = _check_correspondences(correspondences, len(cad), len(observed))
    threshold_m = _threshold(diameter_m, cfg.inlier_fraction)
    candidates, paired, cosine = _pairs(cad, observed, pairs)
    residual = candidates @ rotation.T + translation - paired
    inliers = np.einsum("ni,ni->n", residual, residual) < threshold_m ** 2
    geometry_score, geometry_count = None, None
    if dense_xyz is not None:
        dense = _points(dense_xyz, "dense_xyz")
        distances, _ = cKDTree(dense).query(cad @ rotation.T + translation, k=1)
        geometry_count = int((distances < threshold_m).sum())
        geometry_score = geometry_count / len(cad)
    return PoseScores(float(cosine[inliers].sum() / len(observed)), geometry_score, int(inliers.sum()), len(observed), geometry_count, len(cad), threshold_m)


def _rotation_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def refine_icp(
    cad_xyz: Any,
    dense_xyz: Any,
    R: Any,
    t: Any,
    diameter_m: float,
    config: RegistrationConfig | Mapping[str, Any] | None = None,
) -> ICPResult:
    """Gated point-to-point ICP of the original CAD into dense observed cloud.

    Nearest neighbours are one-sided CAD-to-observation.  Each update fits the
    original inlier CAD coordinates directly to their current observed nearest
    neighbours; it never fits scale.  Insufficient/degenerate inliers leave the
    last valid pose and return an explicit status, never a new identity pose.
    """
    cfg = _config(config)
    cad, dense = _points(cad_xyz, "cad_xyz", 3), _points(dense_xyz, "dense_xyz", 3)
    rotation, translation = _pose(R, t)
    rotation, translation = rotation.copy(), translation.copy()
    threshold_m = _threshold(diameter_m, cfg.inlier_fraction)
    tree, last_indices, last_inliers, history = cKDTree(dense), None, None, []
    status, steps = "max_iterations", 0
    for step in range(cfg.icp_max_iterations):
        distances, indices = tree.query(cad @ rotation.T + translation, k=1)
        inliers = distances < threshold_m
        if int(inliers.sum()) < 3:
            status = "insufficient_inliers"
            break
        if cfg.icp_stop_when_correspondences_stable and last_indices is not None:
            if np.array_equal(inliers, last_inliers) and np.array_equal(indices[inliers], last_indices[inliers]):
                status = "converged_stable_correspondences"
                break
        try:
            new_rotation, new_translation = rigid_fit(cad[inliers], dense[indices[inliers]])
        except ValueError:
            status = "degenerate_inliers"
            break
        rotation_delta = _rotation_angle(new_rotation @ rotation.T)
        translation_delta = float(np.linalg.norm(new_translation - translation))
        history.append({"iteration": step + 1, "inlier_count": int(inliers.sum()), "rmse_before_update_m": float(np.sqrt(np.mean(distances[inliers] ** 2))), "rotation_delta_rad": rotation_delta, "translation_delta_m": translation_delta})
        rotation, translation, steps = new_rotation, new_translation, step + 1
        last_indices, last_inliers = indices, inliers
        if translation_delta <= cfg.icp_translation_tolerance_m and rotation_delta <= cfg.icp_rotation_tolerance_rad:
            status = "converged_transform_delta"
            break
    distances, _ = tree.query(cad @ rotation.T + translation, k=1)
    final_inliers = distances < threshold_m
    count = int(final_inliers.sum())
    rmse = float(np.sqrt(np.mean(distances[final_inliers] ** 2))) if count else None
    return ICPResult(rotation, translation, count / len(cad), count, rmse, steps, status, {"config": asdict(cfg), "threshold_m": threshold_m, "direction": "CAD_to_dense_observed", "objective": "point_to_point", "history": history})


def joint_score(coarse_feature: float, refined_feature: float, geometry: float) -> float:
    """FreeZeV2 base product, with all three exponents exactly one.

    Feature terms are not clipped or rescaled.  In particular this is neither
    a probability nor a confidence calibrated to [0,1].
    """
    values = np.asarray([coarse_feature, refined_feature, geometry], dtype=np.float64)
    if not np.isfinite(values).all() or not 0 <= geometry <= 1:
        raise ValueError("Score terms must be finite and geometry must be in [0,1]")
    return float(np.prod(values))


def _get(candidate: Any, name: str, default: Any = None) -> Any:
    return candidate.get(name, default) if isinstance(candidate, Mapping) else getattr(candidate, name, default)


def rank_candidates(
    poses: Sequence[Any],
    *,
    translation_nms_threshold_m: float,
    max_instances: int | None = None,
) -> CandidateRanking:
    """Joint-score ranking and same-CAD translation NMS, in metres.

    Each input mapping/object supplies R, t, score and (optionally) obj_id,
    candidate_id, status.  All omitted obj_id values form one requested-CAD
    group.  If present, only status='ok' is valid.  Scores must be supplied by
    ``joint_score`` upstream; segmentation confidence is not used here.
    Equal scores preserve input order.  Positive NMS distance is mandatory
    because the paper does not publish it; no supposedly official default is
    invented.  Different CAD identities are never suppressed against each other.
    This deliberately does not claim symmetry-aware rotational NMS.
    """
    if not np.isfinite(translation_nms_threshold_m) or translation_nms_threshold_m <= 0:
        raise ValueError("An explicit positive translation_nms_threshold_m is required")
    if max_instances is not None and (isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)) or max_instances < 1):
        raise ValueError("max_instances must be a positive integer or None")
    valid, rejected = [], []
    for index, candidate in enumerate(poses):
        identifier = _get(candidate, "candidate_id", index)
        try:
            if _get(candidate, "status", "ok") != "ok":
                raise ValueError(f"status={_get(candidate, 'status')}")
            rotation, translation = _pose(_get(candidate, "R"), _get(candidate, "t"))
            score = float(_get(candidate, "score"))
            if not np.isfinite(score):
                raise ValueError("nonfinite joint score")
        except (TypeError, ValueError) as exc:
            rejected.append({"candidate_id": identifier, "input_index": index, "reason": str(exc)})
            continue
        valid.append((index, candidate, translation, score, identifier))
    valid.sort(key=lambda entry: (-entry[3], entry[0]))
    selected_entries, suppressed = [], []
    for entry in valid:
        index, candidate, translation, score, identifier = entry
        winner = next((prior for prior in selected_entries if _get(prior[1], "obj_id") == _get(candidate, "obj_id") and np.linalg.norm(prior[2] - translation) < translation_nms_threshold_m), None)
        if winner is not None:
            suppressed.append({"candidate_id": identifier, "input_index": index, "reason": "translation_nms", "winner_candidate_id": winner[4], "distance_m": float(np.linalg.norm(winner[2] - translation))})
        elif max_instances is not None and len(selected_entries) >= max_instances:
            suppressed.append({"candidate_id": identifier, "input_index": index, "reason": "instance_count_limit"})
        else:
            selected_entries.append(entry)
    return CandidateRanking([entry[1] for entry in selected_entries], suppressed, rejected)
