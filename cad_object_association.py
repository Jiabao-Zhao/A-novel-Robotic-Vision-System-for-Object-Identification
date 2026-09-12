"""Exhaustive, training-free association of one fixed CAD with localized objects."""

import hashlib
import json
from dataclasses import asdict
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d

from CADPointCloudRegistration import CADPointCloudRegistration


CACHE_DIR = Path("outputs/cache/cad_association")
DINO_HUB_DIR = Path("outputs/cache/torch_hub")
DINO_REPO = "facebookresearch/dinov2:e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
DINO_MODEL = "dinov2_vits14"
DINO_DEVICE = "auto"  # Set to "cpu" to force CPU even when CUDA is available.
IMAGE_SIZE = 224
CACHE_VERSION = 1  # Increment when rendering, preprocessing, or geometry code changes.
VISUAL_RENDER_VERSION = 2  # Color-aware rendering; geometry caches remain reusable.
DEFAULT_CAD_COLOR_RGB = (200 / 255, 200 / 255, 200 / 255)  # Only for unspecified materials.
VIEW_DIRECTIONS = np.vstack((np.eye(3), -np.eye(3), list(product((-1, 1), repeat=3))))
FUSION_METHOD = "weighted_mean"  # Or "geometric_mean".
# Experimental baseline only: these bounded scores are not calibrated or equivalent.
VISUAL_WEIGHT = 0.5
GEOMETRY_WEIGHT = 0.5


@lru_cache(maxsize=1)
def load_dino_encoder():
    import torch

    torch.hub.set_dir(str(DINO_HUB_DIR))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if DINO_DEVICE == "auto" else DINO_DEVICE
    model = torch.hub.load(DINO_REPO, DINO_MODEL, pretrained=True,
                           trust_repo=True, skip_validation=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def letterbox_rgb(image):
    """Return the exact 224x224 RGB canvas used before DINO normalization."""
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("DINOv2 requires a nonempty RGB image.")
    scale = IMAGE_SIZE / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_CUBIC)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    y, x = (IMAGE_SIZE - resized.shape[0]) // 2, (IMAGE_SIZE - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def extract_dino_features(images):
    """Identical RGB letterboxing/normalization for CAD views and observed crops."""
    import torch

    model = load_dino_encoder()
    tensors = []
    for image in images:
        canvas = letterbox_rgb(image)
        normalized = (canvas.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)))
    # Small batches bound CPU memory; there is no training or gradient computation.
    features = []
    with torch.inference_mode():
        for start in range(0, len(tensors), 8):
            batch = torch.stack(tensors[start:start + 8]).to(next(model.parameters()).device)
            features.append(model(batch).cpu().numpy())  # CLS descriptors, no classifier.
    return np.concatenate(features)


def render_cad_views(cad_path, base_color_rgb=None):
    """Shaded library material color via CPU ray casting; no OpenGL or display."""
    color = np.asarray(DEFAULT_CAD_COLOR_RGB if base_color_rgb is None else base_color_rgb, dtype=float)
    if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError("CAD base_color_rgb must contain three finite RGB values in [0, 1].")
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    if mesh.is_empty() or not len(mesh.triangles):
        raise ValueError(f"CAD visual association requires a triangle mesh: {cad_path}")
    center = mesh.get_axis_aligned_bounding_box().get_center()
    radius = np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("CAD mesh has no finite spatial extent.")
    mesh.translate(-center).scale(1 / radius, center=(0, 0, 0))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    grid_x, grid_y = np.meshgrid(np.linspace(-1.1, 1.1, IMAGE_SIZE),
                                np.linspace(1.1, -1.1, IMAGE_SIZE))
    images = []
    for direction in VIEW_DIRECTIONS:
        direction = direction / np.linalg.norm(direction)
        right, up = CADPointCloudRegistration.table_basis(direction)
        origins = 3 * direction + grid_x[..., None] * right + grid_y[..., None] * up
        rays = np.concatenate((origins, np.broadcast_to(-direction, origins.shape)), axis=-1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        mask = np.isfinite(hit["t_hit"].numpy())
        normals = hit["primitive_normals"].numpy()
        illumination = .35 + .65 * np.abs(normals @ direction)
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        image[mask] = np.clip(255 * illumination[mask, None] * color, 0, 255).astype(np.uint8)
        images.append(image)
    return images


def cosine_similarity(candidate_feature, cad_view_features):
    candidate = np.asarray(candidate_feature, dtype=float).reshape(-1)
    views = np.atleast_2d(np.asarray(cad_view_features, dtype=float))
    norms = np.linalg.norm(views, axis=1) * np.linalg.norm(candidate)
    if not np.isfinite(candidate).all() or not np.isfinite(views).all() or np.any(norms == 0):
        raise ValueError("Visual descriptors must be finite and nonzero.")
    return float(np.clip(np.max(views @ candidate / norms), -1, 1))


def normalize_visual_score(raw_cosine):
    if not np.isfinite(raw_cosine) or not -1 <= raw_cosine <= 1:
        raise ValueError("Cosine similarity must be in [-1, 1].")
    return float((raw_cosine + 1) / 2)


def normalize_geometry_score(registration_fitness):
    if not np.isfinite(registration_fitness) or not 0 <= registration_fitness <= 1:
        raise ValueError("Observed registration fitness must be in [0, 1].")
    return float(registration_fitness)


def fuse_scores(visual_score, geometry_score, method=None):
    visual_score = normalize_geometry_score(visual_score)
    geometry_score = normalize_geometry_score(geometry_score)
    method = FUSION_METHOD if method is None else method
    if method == "geometric_mean":
        return float(np.sqrt(visual_score * geometry_score))
    if method != "weighted_mean":
        raise ValueError(f"Unknown fusion method: {method}")
    if min(VISUAL_WEIGHT, GEOMETRY_WEIGHT) < 0 or not np.isclose(VISUAL_WEIGHT + GEOMETRY_WEIGHT, 1):
        raise ValueError("Fusion weights must be nonnegative and sum to one.")
    return float(VISUAL_WEIGHT * visual_score + GEOMETRY_WEIGHT * geometry_score)


def rank_candidates(candidates, score_key="fused_score"):
    """Keep every candidate; missing scores sort last, exact ties use object ID."""
    return sorted(candidates, key=lambda c: (
        -(c[score_key] if c[score_key] is not None else -1), c["object_id"],
    ))


def rank_and_resolve(candidates):
    """Return an experimental argmax, without confidence or presence thresholds."""
    ranking = rank_candidates(candidates)
    if not ranking:
        return ranking, None, "no_candidates", "no_localized_candidates"
    top = ranking[0]
    if top["fused_score"] is None:
        return ranking, None, "no_valid_candidates", "no_computable_fused_score"
    return ranking, top["object_id"], "ranking_only", "highest_fused_score"


def raw_observed_cloud_path(item):
    path = Path(item["pointcloud_path"])
    if path.stem.endswith("_downsampled"):
        path = path.with_name(path.stem.removesuffix("_downsampled") + path.suffix)
    return path


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(*values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def prepare_cad_features(cad_model, registrar, cache_dir):
    start = perf_counter()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_hash = content_hash(cad_model.file_path)
    key = cache_key(CACHE_VERSION, source_hash, asdict(registrar.config), o3d.__version__)
    geometry_path = cache_dir / f"{key}_geometry.npz"
    color = getattr(cad_model, "base_color_rgb", None)
    visual_key = cache_key(CACHE_VERSION, VISUAL_RENDER_VERSION, source_hash, color,
                           DINO_REPO, DINO_MODEL, IMAGE_SIZE, VIEW_DIRECTIONS.tolist(), o3d.__version__)
    visual_path = cache_dir / f"{visual_key}_visual.npy"
    geometry_hit, visual_hit = geometry_path.exists(), visual_path.exists()
    runtime = {"cad_visual_preprocessing_time_s": 0.0, "cad_geometry_preprocessing_time_s": 0.0,
               "cad_cache_load_time_s": 0.0, "cad_visual_cache_hit": visual_hit,
               "cad_geometry_cache_hit": geometry_hit}
    stage = perf_counter()
    if geometry_hit:
        with np.load(geometry_path, allow_pickle=False) as saved:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(saved["points"]))
            cloud.normals = o3d.utility.Vector3dVector(saved["normals"])
            fpfh = o3d.pipelines.registration.Feature()
            fpfh.data = saved["fpfh"]
        runtime["cad_cache_load_time_s"] += perf_counter() - stage
    else:
        cloud, fpfh = registrar.compute_fpfh(registrar.load_cad_as_point_cloud(cad_model.file_path))
        np.savez_compressed(geometry_path, points=np.asarray(cloud.points),
                            normals=np.asarray(cloud.normals), fpfh=fpfh.data)
        runtime["cad_geometry_preprocessing_time_s"] = perf_counter() - stage
    stage = perf_counter()
    if visual_hit:
        views = np.load(visual_path, allow_pickle=False)
        runtime["cad_cache_load_time_s"] += perf_counter() - stage
    else:
        views = extract_dino_features(render_cad_views(cad_model.file_path, color))
        np.save(visual_path, views, allow_pickle=False)
        runtime["cad_visual_preprocessing_time_s"] = perf_counter() - stage
    runtime["cad_preparation_total_time_s"] = perf_counter() - start
    return {"cloud": cloud, "fpfh": fpfh, "views": views,
            "geometry_cache_key": key, "visual_cache_key": visual_key}, runtime


def prepare_scene_features(localization_payload, rgb_path, registrar, scene_cache):
    """The caller owns one in-memory cache per scene, shared across all target CADs."""
    rgb_hash = content_hash(rgb_path)
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise ValueError(f"Original RGB image is unreadable: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    candidates, timing = [], {"workspace_dino_feature_time_s": 0.0,
                               "workspace_fpfh_feature_time_s": 0.0, "scene_feature_cache_hits": 0,
                               "workspace_crop_time_s": 0.0, "workspace_dino_crop_count": 0}
    pending, crops, crop_candidates = [], [], []
    ids = [item["object_id"] for item in localization_payload["objects"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Localization object IDs must be unique.")
    for item in localization_payload["objects"]:
        bbox = item.get("bbox_2d_xyxy")
        if bbox is None:
            bbox = [item["roi"][name] for name in ("x1", "y1", "x2", "y2")]
        path = raw_observed_cloud_path(item)
        key = cache_key(CACHE_VERSION, rgb_hash, item["object_id"], bbox, content_hash(path),
                         asdict(registrar.config), DINO_REPO, DINO_MODEL, IMAGE_SIZE, o3d.__version__)
        candidate_runtime = {"scene_feature_cache_hit": key in scene_cache, "crop_time_s": 0.0,
                             "dino_batch_share_time_s": 0.0, "fpfh_feature_time_s": 0.0}
        if key in scene_cache:
            candidates.append({**scene_cache[key], "runtime": candidate_runtime})
            timing["scene_feature_cache_hits"] += 1
            continue
        candidate = {"object_id": item["object_id"], "bbox_2d_xyxy": bbox,
                     "pointcloud_path": str(path), "visual_feature": None,
                     "cloud": None, "fpfh": None, "errors": [], "runtime": candidate_runtime}
        stage = perf_counter()
        x1, y1, x2, y2 = map(int, bbox)
        # Localizer ROI maxima are inclusive pixel coordinates.
        crop = rgb[max(0, y1):max(0, min(height, y2 + 1)),
                   max(0, x1):max(0, min(width, x2 + 1))]
        if crop.size == 0 or x2 < x1 or y2 < y1:
            candidate["errors"].append("invalid_rgb_crop")
        else:
            crops.append(crop)
            crop_candidates.append(candidate)
        candidate_runtime["crop_time_s"] = perf_counter() - stage
        timing["workspace_crop_time_s"] += candidate_runtime["crop_time_s"]
        pending.append((key, candidate))
        candidates.append(candidate)
    if crops:
        stage = perf_counter()
        features = extract_dino_features(crops)
        timing["workspace_dino_feature_time_s"] = perf_counter() - stage
        timing["workspace_dino_crop_count"] = len(crops)
        for candidate, feature in zip(crop_candidates, features, strict=True):
            candidate["visual_feature"] = feature
            # Amortized batch cost, not a measured single-crop inference latency.
            candidate["runtime"]["dino_batch_share_time_s"] = timing["workspace_dino_feature_time_s"] / len(crops)
    for key, candidate in pending:
        # Geometry is attempted for every uncached candidate, including invalid crops.
        stage = perf_counter()
        try:
            cloud = registrar.load_observed_cloud(candidate["pointcloud_path"])
            candidate["cloud"], candidate["fpfh"] = registrar.compute_fpfh(cloud, camera_location=[0, 0, 0])
        except ValueError as error:
            candidate["errors"].append(str(error))
        candidate["runtime"]["fpfh_feature_time_s"] = perf_counter() - stage
        timing["workspace_fpfh_feature_time_s"] += candidate["runtime"]["fpfh_feature_time_s"]
        scene_cache[key] = candidate
    return candidates, timing


def associate_cad_to_candidates(cad_model, localization_payload, rgb_path, plane_model=None,
                                 *, target_description, scene_cache=None, cache_dir=CACHE_DIR):
    """Return all scores and experimental rankings, without a confidence gate.

    plane_model is retained for handoff compatibility. Pairwise matching is rigid
    FPFH/RANSAC + ICP; final registration still applies the existing table constraints.
    """
    start = perf_counter()
    if not str(target_description).strip():
        raise ValueError("A semantic target description is required.")
    registrar = CADPointCloudRegistration()
    cad, runtime = prepare_cad_features(cad_model, registrar, cache_dir)
    scene_start = perf_counter()
    candidates, scene_timing = prepare_scene_features(
        localization_payload, rgb_path, registrar, {} if scene_cache is None else scene_cache,
    )
    runtime.update(scene_timing)
    runtime.update(pairwise_matching_time_s=0.0, visual_comparison_time_s=0.0,
                   fusion_time_s=0.0, per_candidate_matching_time_s={})
    rows = []
    for candidate in candidates:
        row = {name: candidate[name] for name in ("object_id", "bbox_2d_xyxy", "pointcloud_path")}
        row.update(visual_raw=None, visual_score=None,
                   geometry_raw={"status": "invalid", "registration_fitness": None,
                                 "ransac_inlier_ratio": None, "observed_to_cad_rmse_m": None},
                   geometry_score=None, fused_score=None, errors=list(candidate["errors"]),
                   runtime=dict(candidate["runtime"]))
        stage = perf_counter()
        if candidate["visual_feature"] is not None:
            try:
                row["visual_raw"] = cosine_similarity(candidate["visual_feature"], cad["views"])
                row["visual_score"] = normalize_visual_score(row["visual_raw"])
            except ValueError as error:
                row["errors"].append(str(error))
        row["runtime"]["visual_comparison_time_s"] = perf_counter() - stage
        runtime["visual_comparison_time_s"] += row["runtime"]["visual_comparison_time_s"]
        stage = perf_counter()
        if candidate["fpfh"] is not None:
            row["geometry_raw"] = registrar.match_fpfh(
                cad["cloud"], cad["fpfh"], candidate["cloud"], candidate["fpfh"],
            )
            row["geometry_score"] = normalize_geometry_score(row["geometry_raw"]["registration_fitness"])
        elapsed = perf_counter() - stage
        row["runtime"]["geometric_matching_time_s"] = elapsed
        runtime["per_candidate_matching_time_s"][candidate["object_id"]] = elapsed
        runtime["pairwise_matching_time_s"] += elapsed
        rows.append(row)
    stage = perf_counter()
    for row in rows:
        fusion_start = perf_counter()
        if row["visual_score"] is not None and row["geometry_score"] is not None:
            row["fused_score"] = fuse_scores(row["visual_score"], row["geometry_score"])
        row["runtime"]["fusion_time_s"] = perf_counter() - fusion_start
        row["runtime"]["accounted_time_s"] = sum(value for key, value in row["runtime"].items() if key.endswith("_time_s"))
    ranking, selected, resolution, reason = rank_and_resolve(rows)
    rankings, predictions = {}, {}
    for modality, score_key in (("visual", "visual_score"), ("geometry", "geometry_score"), ("fused", "fused_score")):
        ordered = rank_candidates(rows, score_key)
        rankings[modality] = [row["object_id"] for row in ordered]
        predictions[modality] = (ordered[0]["object_id"] if ordered and ordered[0][score_key] is not None else None)
    runtime["fusion_time_s"] = perf_counter() - stage
    runtime["scene_association_time_s"] = perf_counter() - scene_start
    runtime["visual_feature_time_s"] = runtime["workspace_dino_feature_time_s"]
    runtime["geometry_feature_time_s"] = runtime["workspace_fpfh_feature_time_s"]
    top = ranking[0] if ranking else None
    result = {
        "target_description": target_description, "cad_id": str(cad_model.cad_id),
        "cad_path": str(cad_model.file_path), "selected_object_id": selected,
        "final_object_id": selected, "resolution": resolution, "reason": reason,
        "proposed_object_id": top["object_id"] if top else None,
        "scores": {
            "visual_raw": top["visual_raw"], "visual_normalized": top["visual_score"],
            "geometry_fitness": top["geometry_raw"].get("registration_fitness"),
            "geometry_rmse_m": top["geometry_raw"].get("observed_to_cad_rmse_m"),
            "geometry_normalized": top["geometry_score"], "fused": top["fused_score"],
        } if top else {},
        "candidate_ranking": ranking, "rankings": rankings, "predictions": predictions, "runtime": runtime,
        "diagnostics": {
            "method": "frozen DINOv2-small + local FPFH/RANSAC/ICP",
            "geometry_score_direction": "observed_partial_surface_to_complete_cad",
            "cad_geometry_cache_key": cad["geometry_cache_key"], "cad_visual_cache_key": cad["visual_cache_key"],
            "encoder": DINO_MODEL, "encoder_source": DINO_REPO,
            "cad_base_color_rgb": getattr(cad_model, "base_color_rgb", None),
            "cad_color_source": ("library_material" if getattr(cad_model, "base_color_rgb", None) is not None
                                 else "unspecified_neutral_default"),
            "voxel_size_m": registrar.config.voxel_size_m,
            "fusion_method": FUSION_METHOD, "visual_weight": VISUAL_WEIGHT, "geometry_weight": GEOMETRY_WEIGHT,
            "decision_policy": "ranking_only_no_thresholds",
            "score_interpretation": "Uncalibrated, non-equivalent scores; fixed fusion is an experimental baseline.",
            "dino_candidate_timing": "Equal share of measured workspace batch time; not individual inference latency.",
        },
    }
    runtime["total_time_s"] = perf_counter() - start
    return result
