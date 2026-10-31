"""Controlled CAD-resolution ablation over unchanged frozen render caches.

This module is independent code. ``cad476`` is NOT a full replication of the
public FreeZeV2-Re implementation: CAD sampling, render photometry/cameras,
visibility, interpolation, DINO block18, observation224 and registration remain
those of the sealed baseline. Only the CAD encoder's input resolution changes.
No CAD is rerendered, no PCA is fitted, and no geometric rows are removed here.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_disk(path, minimum_free_gb, extra_bytes=0):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < minimum_free_gb * 1024**3 + extra_bytes:
        raise RuntimeError("Insufficient disk space for CAD visual cache and reserve")


class CADVisualAdapter:
    """Reuse a loaded FrozenDINOv2 without mutating it or loading new weights."""

    def __init__(self, dino, mode="baseline224"):
        if mode not in {"baseline224", "cad476"}:
            raise ValueError("mode must be baseline224 or cad476")
        if dino.layer_index != 18 or dino.input_size != 224:
            raise ValueError("Controlled ablation requires the original block18/input224 adapter")
        if dino.model.training or any(p.requires_grad for p in dino.model.parameters()):
            raise ValueError("DINO must be frozen and in evaluation mode")
        if any(p.is_floating_point() and p.dtype != dino.torch.float32
               for p in dino.model.parameters()):
            raise ValueError("Controlled ablation requires FP32 weights")
        self.dino, self.mode = dino, mode
        self.sampling_image_hw = (480, 480) if mode == "baseline224" else (476, 476)
        self.grid_size = 16 if mode == "baseline224" else 34
        self.provenance = {
            "schema": "controlled-CAD-input-resolution-v1",
            "mode": mode,
            "CAD_input_hw": [224, 224] if mode == "baseline224" else [476, 476],
            "sampling_image_hw": list(self.sampling_image_hw),
            "frozen_dino": copy.deepcopy(dino.provenance),
            "implementation_sha256": _sha(__file__),
            "scope": "CAD resolution only; observation224 and other baseline settings unchanged",
            "public_repository_replication": False,
        }

    def extract(self, rgb):
        image = np.asarray(rgb)
        if image.shape != (480, 480, 3) or image.dtype != np.uint8:
            raise ValueError("Expected an unchanged cached480x480 uint8 RGB render")
        if self.dino.model.training:
            raise RuntimeError("DINO left evaluation mode")
        if self.mode == "baseline224":
            fmap = self.dino.extract(image)
        else:
            torch = self.dino.torch
            with torch.inference_mode():
                x = torch.as_tensor(np.ascontiguousarray(image[:476, :476])).to(
                    self.dino.device, dtype=torch.float32)
                x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
                mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
                std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
                maps = self.dino.model.get_intermediate_layers(
                    (x - mean) / std, n=[self.dino.layer_index], reshape=True,
                    return_class_token=False, norm=True)
                fmap = maps[0][0].permute(1, 2, 0).cpu().numpy().copy()
        expected = (self.grid_size, self.grid_size, 1536)
        if fmap.shape != expected or fmap.dtype != np.float32 or not np.isfinite(fmap).all():
            raise ValueError(f"Invalid actual frozen CAD map; expected finite FP32{expected}")
        return fmap


def _load_map(path, identity_digest, grid_size):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["identity"]) != identity_digest:
            raise ValueError(f"CAD map identity mismatch: {path}")
        fmap = np.array(saved["map"])
    if fmap.shape != (grid_size, grid_size, 1536) or fmap.dtype != np.float32 or not np.isfinite(fmap).all():
        raise ValueError(f"Invalid CAD map: {path}")
    return fmap


def aggregate_cached_visuals(geometry_folder, output_folder, adapter, *, min_views=18,
                             baseline_map_folder=None, heartbeat=None,
                             minimum_free_gb=15, expected_views=162, sampler=None):
    """Cache/resume view maps and reproduce the baseline visible-view average.

    Returns full original-row ``visual_mean``, ``view_counts`` and
    ``original_indices``. Missing required visibility is an error, never a hidden
    row exclusion. A later GeDi repair must index all feature/point arrays using
    the same GeDi original_indices. Existing baseline224 maps can be reused by
    supplying their original feature folder (including identity/progress).
    """
    if min_views < 1 or expected_views < 1 or min_views > expected_views:
        raise ValueError("Positive visibility/view counts are required")
    if sampler is None:
        from freezev2_repro.features import sample_feature_map
        sampler = sample_feature_map
    sampler_source = inspect.getsourcefile(sampler)
    if not sampler_source or not Path(sampler_source).is_file():
        raise ValueError("Sampler must have a verifiable Python source file")
    geometry_folder, output_folder = Path(geometry_folder), Path(output_folder)
    geometry = json.loads((geometry_folder / "complete.json").read_text())
    geometry_path = geometry_folder / "geometry.npz"
    if _sha(geometry_path) != geometry["files"]["geometry.npz"]:
        raise ValueError("Original CAD geometry checksum changed")
    geometry_identity = _digest(geometry["identity"])
    with np.load(geometry_path, allow_pickle=False) as saved:
        if str(saved["identity"]) != geometry_identity:
            raise ValueError("Original CAD geometry identity mismatch")
        data = {k: np.array(saved[k]) for k in ("points_m", "visibility", "R", "t_m", "K")}
    points, visibility = data["points_m"], data["visibility"]
    n = len(points)
    if (points.shape != (n, 3) or n < 1 or visibility.shape != (expected_views, n)
            or data["R"].shape != (expected_views, 3, 3)
            or data["t_m"].shape != (expected_views, 3) or data["K"].shape != (3, 3)
            or not all(np.isfinite(data[k]).all() for k in data)):
        raise ValueError("Invalid original CAD geometry arrays")
    baseline_identity, baseline_progress, baseline_map_folder = None, None, (
        Path(baseline_map_folder) if baseline_map_folder is not None else None)
    if baseline_map_folder is not None:
        if adapter.mode != "baseline224":
            raise ValueError("Original224 maps cannot be reused by the476 ablation")
        baseline_identity = json.loads((baseline_map_folder / "identity.json").read_text())
        baseline_progress = json.loads((baseline_map_folder / "progress.json").read_text())
        if (baseline_identity["geometry_identity"] != geometry_identity
                or baseline_identity["geometry_sha256"] != geometry["files"]["geometry.npz"]):
            raise ValueError("Original224 maps refer to different CAD geometry")
        old_dino = baseline_identity["backends"]["dino"]
        for key in ("model", "layer_index", "input_size", "normalization", "precision",
                    "checkpoint_sha256", "tf32_matmul", "tf32_cudnn"):
            if old_dino.get(key) != adapter.dino.provenance.get(key):
                raise ValueError(f"Original224 DINO map provenance differs: {key}")
    identity = {
        "schema": "controlled-CAD-visual-aggregation-v1", "adapter": adapter.provenance,
        "geometry_complete_sha256": _sha(geometry_folder / "complete.json"),
        "geometry_sha256": geometry["files"]["geometry.npz"],
        "original_rows": n, "min_views": min_views, "views": expected_views,
        "sampler_source_sha256": _sha(sampler_source),
        "sampler_name": sampler.__module__ + "." + sampler.__qualname__,
        "baseline_maps_identity": _digest(baseline_identity) if baseline_identity else None,
    }
    _require_disk(output_folder, minimum_free_gb)
    output_folder.mkdir(parents=True, exist_ok=True)
    identity_path = output_folder / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Incompatible CAD visual cache; use a different output folder")
    else:
        if any(output_folder.iterdir()):
            raise ValueError("Unidentified CAD visual cache; refusing to mix outputs")
        _atomic_json(identity_path, identity)
    progress_path = output_folder / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"maps": {}, "active_runtime_s": 0.0}
    total, counts = np.zeros((n, 1536), np.float64), np.zeros(n, np.int32)
    identity_digest, start = _digest(identity), time.perf_counter()
    for view in range(expected_views):
        if heartbeat:
            heartbeat(stage="controlled_CAD_visual_features", mode=adapter.mode, view=view, views=expected_views)
        image_path = geometry_folder / "views" / f"{view:03d}.npz"
        if _sha(image_path) != geometry["view_file_hashes"][str(view)]["sha256"]:
            raise ValueError(f"Original rendered view checksum changed: {view}")
        original_map = baseline_map_folder / "visual_maps" / f"{view:03d}.npz" if baseline_map_folder else None
        cache_path = output_folder / "visual_maps" / f"{view:03d}.npz"
        use_original = original_map is not None and str(view) in baseline_progress["maps"]
        selected_path = original_map if use_original else cache_path
        map_identity = _digest(baseline_identity) if use_original else identity_digest
        if str(view) in progress["maps"]:
            if _sha(selected_path) != progress["maps"][str(view)]["sha256"]:
                raise ValueError(f"CAD map checksum changed: {view}")
            fmap = _load_map(selected_path, map_identity, adapter.grid_size)
        elif use_original:
            if _sha(original_map) != baseline_progress["maps"][str(view)]["sha256"]:
                raise ValueError(f"Original224 CAD map checksum changed: {view}")
            fmap = _load_map(original_map, map_identity, adapter.grid_size)
        else:
            _require_disk(output_folder, minimum_free_gb, adapter.grid_size**2 * 1536 * 4)
            with np.load(image_path, allow_pickle=False) as saved:
                if str(saved["identity"]) != geometry_identity:
                    raise ValueError("Original rendered view identity changed")
                image = np.array(saved["rgb"])
            fmap = adapter.extract(image)
            _atomic_npz(cache_path, identity=np.array(identity_digest), map=fmap)
        camera = points @ data["R"][view].T + data["t_m"][view]
        projected = camera @ data["K"].T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = projected[:, :2] / projected[:, 2:3]
        h, w = adapter.sampling_image_hw
        kept = visibility[view].astype(bool) & np.isfinite(uv).all(1) & (camera[:, 2] > 0)
        if adapter.mode == "cad476":
            kept &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        total[kept] += sampler(fmap, uv[kept], image_size=(h, w))
        counts[kept] += 1
        progress["maps"][str(view)] = {"sha256": _sha(selected_path), "reused_original224": use_original}
        progress["active_runtime_s"] += time.perf_counter() - start
        start = time.perf_counter()
        _atomic_json(progress_path, progress)
    if np.any(counts < min_views):
        bad = np.flatnonzero(counts < min_views)
        raise ValueError(f"CAD resolution removes required visibility for {len(bad)} original rows; first indices={bad[:10].tolist()}")
    visual = (total / counts[:, None]).astype(np.float32)
    if not np.isfinite(visual).all():
        raise ValueError("Nonfinite CAD visual descriptors")
    final_path = output_folder / "visual_mean.npz"
    indices = np.arange(n, dtype=np.int64)
    if final_path.exists():
        with np.load(final_path, allow_pickle=False) as old:
            if (str(old["identity"]) != identity_digest
                    or not np.array_equal(old["visual_mean"], visual)
                    or not np.array_equal(old["view_counts"], counts)
                    or not np.array_equal(old["original_indices"], indices)):
                raise ValueError("Final CAD visual cache differs from verified view maps")
    else:
        _require_disk(output_folder, minimum_free_gb, visual.nbytes)
        _atomic_npz(final_path, identity=np.array(identity_digest), visual_mean=visual,
                    view_counts=counts, original_indices=indices)
    complete = {"status": "complete", "identity": identity,
                "files": {"visual_mean.npz": _sha(final_path)},
                "views_completed": expected_views, "minimum_visibility": int(counts.min()),
                "runtime_s": progress["active_runtime_s"],
                "map_storage": "original224 references" if baseline_map_folder else "new versioned maps",
                "public_repository_replication": False}
    _atomic_json(output_folder / "complete.json", complete)
    return {"visual_mean": visual, "view_counts": counts, "original_indices": indices,
            "identity": identity, "manifest": complete}
