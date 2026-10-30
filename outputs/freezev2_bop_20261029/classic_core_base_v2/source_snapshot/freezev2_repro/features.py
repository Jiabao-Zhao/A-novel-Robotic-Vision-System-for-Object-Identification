"""Frozen feature adapters and feature/point mapping for the FreeZeV2 study.

No model is downloaded and no substitute descriptor is produced by this module.
The paper leaves several details unspecified; see FEATURE_NOTES.md. In
particular, a DINO block must be supplied explicitly and uniform view averaging
is an explicitly declared reproduction assumption.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any

import numpy as np


class FeatureDependencyError(RuntimeError):
    """A required local trained model or its compatible dependency is absent."""


def _finite_matrix(value: Any, name: str, columns: int | None = None) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or (columns is not None and value.shape[1] != columns):
        suffix = "" if columns is None else f" with {columns} columns"
        raise ValueError(f"{name} must be a matrix{suffix}; got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return value


def sample_feature_map(
    feature_map: np.ndarray, uv: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Bilinearly sample Hf x Wf x D features at RGB pixel coordinates (u,v).

    image_size is (height,width). Pixel centers follow the half-pixel mapping
    used by bilinear interpolation with align_corners=False. Border features
    are replicated. Coordinates outside the RGB image are rejected, rather
    than silently wrapping or projecting an invisible CAD point onto a border.
    """
    fmap = np.asarray(feature_map)
    if fmap.ndim != 3 or min(fmap.shape) < 1 or not np.isfinite(fmap).all():
        raise ValueError("feature_map must be a finite nonempty Hf x Wf x D array")
    xy = _finite_matrix(uv, "uv", 2).astype(np.float64, copy=False)
    height, width = map(int, image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive (height,width)")
    if len(xy) and (
        np.any(xy[:, 0] < 0) or np.any(xy[:, 0] > width - 1)
        or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] > height - 1)
    ):
        raise ValueError("uv contains coordinates outside image_size")
    fh, fw, _ = fmap.shape
    x = np.clip((xy[:, 0] + 0.5) * fw / width - 0.5, 0, fw - 1)
    y = np.clip((xy[:, 1] + 0.5) * fh / height - 0.5, 0, fh - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, fw - 1), np.minimum(y0 + 1, fh - 1)
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    result = (
        fmap[y0, x0] * (1 - wx) * (1 - wy)
        + fmap[y0, x1] * wx * (1 - wy)
        + fmap[y1, x0] * (1 - wx) * wy
        + fmap[y1, x1] * wx * wy
    )
    return result.astype(np.result_type(fmap.dtype, np.float32), copy=False)


def aggregate_view_features(
    features: np.ndarray, visible: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weighted mean features, distinct visible-view counts).

    features has shape V x N x D and visible has shape V x N. Missing values in
    invisible slots are permitted. Unseen points receive NaN descriptors and
    must be removed before PCA/registration. weights=None means uniform visible
    weights: an assumption, since V2 specifies a weighted mean without weights.
    """
    values = np.asarray(features)
    valid = np.asarray(visible, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("features must be V x N x D and visible must be V x N")
    if not np.isfinite(values[valid]).all():
        raise ValueError("visible feature descriptors must be finite")
    if weights is None:
        w = valid.astype(np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape == (values.shape[0],):
            w = np.broadcast_to(w[:, None], valid.shape)
        if w.shape != valid.shape or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("weights must be finite nonnegative V or V x N values")
        w = np.where(valid, w, 0.0)
    totals = w.sum(axis=0)
    numerator = (np.where(valid[..., None], values, 0.0) * w[..., None]).sum(axis=0)
    mean = np.full(values.shape[1:], np.nan, dtype=np.float64)
    np.divide(numerator, totals[:, None], out=mean, where=totals[:, None] > 0)
    return mean.astype(np.float32), valid.sum(axis=0, dtype=np.int64)


class PCATransform:
    """Unwhitened CAD-fitted PCA, reused unchanged for observed descriptors.

    Fit data is deliberately named cad_features. Caller provenance must confirm
    its CAD-only origin; this class cannot infer the semantic origin of arrays.
    """

    def __init__(self, n_components: int = 64):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, cad_features: np.ndarray) -> "PCATransform":
        x = _finite_matrix(cad_features, "cad_features").astype(np.float64)
        if min(x.shape) < self.n_components:
            raise ValueError(
                f"PCA{self.n_components} requires at least that many CAD rows and columns; got {x.shape}"
            )
        mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        components = vt[: self.n_components].copy()
        # Fix the arbitrary SVD sign to make cache bytes reproducible.
        largest = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(self.n_components), largest])
        components *= np.where(signs == 0, 1.0, signs)[:, None]
        self.mean_, self.components_ = mean, components
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA has not been fitted on CAD visual features")
        x = _finite_matrix(features, "features", len(self.mean_))
        return ((x - self.mean_) @ self.components_.T).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("cannot save unfitted PCA")
        np.savez_compressed(
            path, mean=self.mean_, components=self.components_,
            n_components=np.int64(self.n_components), fit_scope=np.array("cad_only"),
            whiten=np.array(False),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCATransform":
        with np.load(path, allow_pickle=False) as data:
            if str(data["fit_scope"]) != "cad_only" or bool(data["whiten"]):
                raise ValueError("PCA cache must declare CAD-only fitting without whitening")
            result = cls(int(data["n_components"]))
            result.mean_ = np.array(data["mean"], dtype=np.float64)
            result.components_ = _finite_matrix(data["components"], "PCA components").astype(np.float64)
        if (result.mean_.ndim != 1 or not np.isfinite(result.mean_).all()
                or result.components_.shape != (result.n_components, len(result.mean_))):
            raise ValueError("invalid PCA cache dimensions or values")
        return result


def _normalize(features: np.ndarray, name: str) -> np.ndarray:
    x = _finite_matrix(features, name).astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{name} contains a zero descriptor; cannot L2 normalize")
    return x / norms


def fuse_features(visual: np.ndarray, geometric: np.ndarray) -> np.ndarray:
    """Eq(1)/(2): independent L2 normalization, then equal-dimensional concat.

    For the chosen base: visual has already undergone PCA64 and geometric is the
    two-scale GeDi64. No new fusion weight or final L2 normalization is applied.
    Cosine correspondence search must normalize the concatenated descriptor.
    """
    v, g = _normalize(visual, "visual"), _normalize(geometric, "geometric")
    if v.shape != g.shape:
        raise ValueError("visual and geometric matrices must have equal row counts and dimensions")
    return np.concatenate((v, g), axis=1)


def freeze_model(model: Any) -> Any:
    """Set eval mode and disable every parameter gradient without training."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def configure_frozen_fp32(torch: Any) -> None:
    """Use declared full-FP32 arithmetic rather than implicit Ampere TF32.

    This is a process-scoped numerical configuration assumption, since the
    paper does not specify framework TF32 defaults. It changes no environment,
    model weights, feature settings or patch sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def microbatch_frozen_model(model: Any, batch_size: int | None) -> Any:
    """Split only an already-prepared neural input, preserving patch sampling.

    Original GeDi's neighborhood/LRF batch is a sampling parameter and stays
    untouched. Its frozen network has per-example FPS, eval BatchNorm and no
    active dropout; only this forward pass may be divided to bound GPU memory.
    Actual trained-model equivalence must still be measured in its runtime.
    """
    model = freeze_model(model)
    if batch_size is None:
        return model
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("network forward microbatch must be a positive integer or None")
    import torch

    class ForwardMicrobatch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unwrapped_model = model
            self.batch_size = batch_size

        def forward(self, value):
            if self.training:
                raise RuntimeError("GeDi forward microbatch is inference-only")
            return torch.cat([self.unwrapped_model(chunk) for chunk in value.split(self.batch_size)], dim=0)

    return freeze_model(ForwardMicrobatch())


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verify_revision(repo: Path, expected_revision: str | None) -> None:
    if expected_revision is None:
        return
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeatureDependencyError(f"cannot verify local source revision in {repo}: {exc}") from exc
    if revision != expected_revision:
        raise FeatureDependencyError(f"source revision mismatch: expected {expected_revision}, got {revision}")


def _source_fingerprints(
    repo: Path, expected_source_hashes: dict[str, str] | None
) -> tuple[dict[str, str], dict[str, str]]:
    """Fingerprint selected local Python sources, optionally checking exact pins.

    Cached torch-hub ZIP archives generally lack .git. Matching these selected
    files proves their bytes, not a complete archive revision or clean Git tree.
    Relative paths are canonicalized and cannot traverse outside the local repo.
    """
    if expected_source_hashes is not None and not isinstance(expected_source_hashes, dict):
        raise ValueError("expected_source_hashes must map relative Python paths to SHA256 strings")
    expected: dict[str, str] = {}
    for relative, checksum in (expected_source_hashes or {}).items():
        if not isinstance(relative, str) or not isinstance(checksum, str):
            raise ValueError("source hash pins must use string paths and SHA256 values")
        path = PurePosixPath(relative.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts) or path.suffix != ".py":
            raise ValueError(f"source hash path must be a relative Python file inside the repo: {relative}")
        key = path.as_posix()
        if len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum):
            raise ValueError(f"source hash pin must be a complete SHA256: {relative}")
        if key in expected:
            raise ValueError(f"duplicate normalized source hash path: {relative}")
        expected[key] = checksum.lower()
    root = repo.resolve()
    actual = {}
    selected = {"hubconf.py", "dinov2/models/vision_transformer.py"} | set(expected)
    for relative in sorted(selected):
        source = (root / relative).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"source hash path resolves outside local repo: {relative}")
        if not source.is_file():
            if relative in expected:
                raise FeatureDependencyError(f"pinned DINOv2 source file missing: {source}")
            continue
        actual[relative] = _sha256(source)
        if relative in expected and actual[relative] != expected[relative]:
            raise FeatureDependencyError(
                f"DINOv2 source SHA256 mismatch for {relative}: expected {expected[relative]}, got {actual[relative]}"
            )
    return actual, expected


class FrozenDINOv2:
    """Offline official local-hub DINOv2 giant adapter, with an explicit block.

    Expects an official compatible full state dictionary. It never falls back
    to random weights, ViT-L, last-layer features, a remote hub, or downloaded
    parameters. ImageNet normalization and 224 input are recorded assumptions.
    """

    def __init__(
        self, repo_path: str | Path, checkpoint: str | Path, *, layer_index: int,
        device: str = "cuda", model_name: str = "dinov2_vitg14",
        input_size: int = 224, expected_revision: str | None = None,
        checkpoint_sha256: str | None = None,
        expected_source_hashes: dict[str, str] | None = None,
        source_revision_hint: str | None = None,
        memory_efficient_load: bool = True,
    ):
        if model_name not in {"dinov2_vitg14", "dinov2_vitg14_reg"}:
            raise ValueError("the base adapter requires an explicitly chosen DINOv2 ViT-giant/14 variant")
        if not isinstance(layer_index, int) or not 0 <= layer_index < 40:
            raise ValueError("layer_index must explicitly identify a zero-based giant block in [0,39]")
        if input_size <= 0 or input_size % 14:
            raise ValueError("DINO input_size must be a positive multiple of patch size14")
        repo, weights = Path(repo_path), Path(checkpoint)
        if not (repo / "hubconf.py").is_file():
            raise FeatureDependencyError(f"local DINOv2 source missing hubconf.py: {repo}; no download attempted")
        if not weights.is_file():
            raise FeatureDependencyError(f"local DINOv2 giant checkpoint missing: {weights}; no download attempted")
        _verify_revision(repo, expected_revision)
        source_fingerprints, verified_source_hashes = _source_fingerprints(repo, expected_source_hashes)
        actual_sha = _sha256(weights)
        if checkpoint_sha256 is not None and actual_sha != checkpoint_sha256:
            raise FeatureDependencyError(f"DINOv2 checkpoint SHA256 mismatch: {actual_sha}")
        try:
            import torch
        except ImportError as exc:
            raise FeatureDependencyError("PyTorch is required for frozen DINOv2 giant") from exc
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise FeatureDependencyError(f"requested DINO device {device}, but CUDA is unavailable")
        configure_frozen_fp32(torch)
        try:
            if memory_efficient_load:
                # The official constructor reads a scalar drop-path schedule,
                # so it cannot run on meta tensors. Memory-map the checkpoint
                # instead of allocating a second eager 4.5GB CPU copy, then
                # strictly replace all initialized parameters by trained data.
                model = torch.hub.load(str(repo), model_name, source="local", pretrained=False)
                state = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
                model.load_state_dict(state, strict=True, assign=True)
            else:
                model = torch.hub.load(str(repo), model_name, source="local", pretrained=False)
                state = torch.load(str(weights), map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
            self.model = freeze_model(model).to(device=device, dtype=torch.float32)
        except Exception as exc:
            raise FeatureDependencyError(
                f"cannot load local {model_name} with exact checkpoint {weights}: {type(exc).__name__}: {exc}"
            ) from exc
        self.torch, self.device = torch, torch.device(device)
        self.layer_index, self.input_size = layer_index, input_size
        self.provenance = {
            "backend": "official_local_dinov2", "model": model_name,
            "source_path": str(repo.resolve()), "source_revision": expected_revision,
            "verified_git_revision": expected_revision,
            "source_revision_hint": source_revision_hint,
            "source_fingerprints": source_fingerprints,
            "verified_source_hashes": verified_source_hashes,
            "source_hash_scope": "selected files only; not a complete source-tree or Git verification",
            "checkpoint": str(weights.resolve()), "checkpoint_sha256": actual_sha,
            "layer_index": layer_index, "layer_selection_is_assumption": True,
            "input_size": input_size, "input_resize_is_assumption": True,
            "normalization": "ImageNet_mean_std", "precision": "float32", "frozen": True,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "memory_efficient_checkpoint_loading": bool(memory_efficient_load),
        }

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """Return Hf x Wf x1536 normalized-block patch tokens (16×16 at224).

        RGB background is unchanged; mask-based RGB background replacement is a
        caller-level, separately recorded assumption, never implicit here.
        """
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
            raise ValueError("rgb must be a nonempty H x W x3 RGB image")
        if image.dtype != np.uint8:
            raise ValueError("rgb must be uint8 with values0..255; ambiguous input scaling is rejected")
        torch = self.torch
        with torch.inference_mode():
            x = torch.as_tensor(np.ascontiguousarray(image)).to(self.device, dtype=torch.float32)
            x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
            x = torch.nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear",
                align_corners=False, antialias=True,
            )
            mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            x = (x - mean) / std
            maps = self.model.get_intermediate_layers(
                x, n=[self.layer_index], reshape=True, return_class_token=False, norm=True
            )
            fmap = maps[0]
            expected = (1, 1536, self.input_size // 14, self.input_size // 14)
            if tuple(fmap.shape) != expected or not torch.isfinite(fmap).all():
                raise FeatureDependencyError(f"unexpected/nonfinite giant token map {tuple(fmap.shape)}; expected {expected}")
            return fmap[0].permute(1, 2, 0).cpu().numpy().copy()


class FrozenGeDi:
    """Call a hash-pinned external original GeDi implementation at two scales.

    Original GeDi is CUDA-only and depends on Open3D's torch ops, torchgeometry,
    and its Pointnet2 extension. Nothing from that unlicensed repository is
    vendored here. Patch-sampling counts below are configurable demo-derived
    assumptions, distinct from the explicit V2 radii and output dimensions.
    """

    def __init__(
        self, repo_path: str | Path, checkpoint: str | Path, *,
        expected_source_sha256: str, expected_revision: str | None = None,
        checkpoint_sha256: str | None = None, samples_per_batch: int = 500,
        samples_per_patch_lrf: int = 4000, samples_per_patch_out: int = 512,
        seed: int = 0, forward_microbatch_size: int | None = 32,
    ):
        repo, weights = Path(repo_path), Path(checkpoint)
        source = repo / "gedi.py"
        if not source.is_file():
            raise FeatureDependencyError(f"external original GeDi source missing: {source}; no replacement descriptor is available")
        if not expected_source_sha256 or _sha256(source) != expected_source_sha256:
            raise FeatureDependencyError("external GeDi source is not the exact SHA256 configured for this experiment")
        if not weights.is_file():
            raise FeatureDependencyError(f"local trained GeDi32 checkpoint missing: {weights}")
        _verify_revision(repo, expected_revision)
        actual_sha = _sha256(weights)
        if checkpoint_sha256 is not None and actual_sha != checkpoint_sha256:
            raise FeatureDependencyError(f"GeDi checkpoint SHA256 mismatch: {actual_sha}")
        if not (0 < samples_per_patch_out <= samples_per_patch_lrf and samples_per_batch > 0):
            raise ValueError("GeDi patch counts must be positive with output<=LRF count")
        try:
            import torch
            configure_frozen_fp32(torch)
            if not torch.cuda.is_available():
                raise FeatureDependencyError("original GeDi requires CUDA; this environment has no usable CUDA device")
            # A unique module name avoids accidentally importing another gedi.py.
            spec = importlib.util.spec_from_file_location(f"_freezev2_gedi_{expected_source_sha256[:12]}", source)
            if spec is None or spec.loader is None:
                raise ImportError("cannot construct external GeDi module loader")
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(repo.resolve()))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
            self.backend = module.GeDi(config={
                "dim": 32, "samples_per_batch": samples_per_batch,
                "samples_per_patch_lrf": samples_per_patch_lrf,
                "samples_per_patch_out": samples_per_patch_out,
                "r_lrf": 1.0, "fchkpt_gedi_net": str(weights.resolve()),
            })
            self._unwrapped_network = freeze_model(self.backend.gedi_net)
        except FeatureDependencyError:
            raise
        except Exception as exc:
            raise FeatureDependencyError(
                "cannot load original trained GeDi. Required: compatible PyTorch CUDA, "
                "open3d.ml.torch, torchgeometry, and the compiled GeDi Pointnet2 extension. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        self.torch, self.seed = torch, int(seed)
        self.provenance = {
            "backend": "external_original_gedi", "source_path": str(repo.resolve()),
            "source_sha256": expected_source_sha256, "source_revision": expected_revision,
            "checkpoint": str(weights.resolve()), "checkpoint_sha256": actual_sha,
            "radii_diameter_fractions": [0.3, 0.4], "output_dimension_per_scale": 32,
            "samples_per_batch": samples_per_batch,
            "samples_per_patch_lrf": samples_per_patch_lrf,
            "samples_per_patch_out": samples_per_patch_out,
            "sampling_batch_preserved": True,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "resources": {},
            "patch_counts_are_assumptions": True, "frozen": True, "seed": self.seed,
        }
        self.set_forward_microbatch_size(forward_microbatch_size)

    def set_forward_microbatch_size(self, batch_size: int | None) -> None:
        """Bound only the frozen network's memory, preserving GeDi sampling.

        Does not change samples_per_batch, RNG state, local reference frames,
        neighborhoods or prepared patches. Runtime validation records the
        numerical equivalence tolerance for supported microbatch sizes.
        """
        self.backend.gedi_net = microbatch_frozen_model(self._unwrapped_network, batch_size)
        self.forward_microbatch_size = batch_size
        self.provenance['resources']['network_forward_microbatch_size'] = batch_size

    def compute(self, points: np.ndarray, dense_cloud: np.ndarray, diameter: float) -> np.ndarray:
        points = _finite_matrix(points, "points", 3).astype(np.float32)
        dense = _finite_matrix(dense_cloud, "dense_cloud", 3).astype(np.float32)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("native-unit object diameter must be finite and positive")
        if len(points) == 0 or len(dense) < 3:
            raise ValueError("GeDi requires nonempty query points and a nondegenerate dense cloud")
        torch = self.torch
        query, support = torch.from_numpy(points), torch.from_numpy(dense)
        # Original GeDi uses NumPy's global random sampler. Preserve caller state.
        random_state = np.random.get_state()
        descriptors = []
        try:
            np.random.seed(self.seed)
            with torch.inference_mode():
                for fraction in (0.3, 0.4):
                    radius = float(fraction * diameter)
                    self.backend.r_lrf = radius
                    self.backend.lrf.r_lrf = radius
                    value = self.backend.compute(pts=query, pcd=support)
                    value = _finite_matrix(value, "trained GeDi descriptors", 32)
                    if len(value) != len(points):
                        raise ValueError("GeDi returned a different point count")
                    descriptors.append(value.astype(np.float32))
        except Exception as exc:
            raise FeatureDependencyError(
                f"original GeDi extraction failed (check local neighborhoods, LRF degeneracy and CUDA ops): {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            np.random.set_state(random_state)
        return np.concatenate(descriptors, axis=1)
