"""Metric visible-CAD surfaces for Ablation 3; production rendering is untouched."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d

import cad_object_association as association


CACHE_ROOT = Path("outputs/cache/wrist_visible_geometry")


def geometry_from_arrays(arrays):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arrays["points"]))
    cloud.normals = o3d.utility.Vector3dVector(arrays["normals"])
    features = o3d.pipelines.registration.Feature()
    features.data = arrays["fpfh"]
    return cloud, features


def visible_surfaces(cad_path, color):
    """Use production orthographic rays; undo only its center/radius normalization.

    CAD points remain in the native CAD frame, in meters. Normals interpolate the
    mesh vertex normals, as does the existing full-CAD uniform surface sampler.
    """
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    if mesh.is_empty() or not len(mesh.triangles):
        raise ValueError(f"Empty CAD mesh: {cad_path}")
    center = mesh.get_axis_aligned_bounding_box().get_center()
    radius = np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"CAD has no finite extent: {cad_path}")
    mesh.compute_vertex_normals()
    mesh.translate(-center).scale(1 / radius, center=(0, 0, 0))
    vertex_normals, triangles = np.asarray(mesh.vertex_normals), np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    grid_x, grid_y = np.meshgrid(np.linspace(-1.1, 1.1, association.IMAGE_SIZE),
                                np.linspace(1.1, -1.1, association.IMAGE_SIZE))
    for direction in association.VIEW_DIRECTIONS:
        direction = direction / np.linalg.norm(direction)
        right, up = association.CADPointCloudRegistration.table_basis(direction)
        origins = 3 * direction + grid_x[..., None] * right + grid_y[..., None] * up
        rays = np.concatenate((origins, np.broadcast_to(-direction, origins.shape)), axis=-1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        distance = hit["t_hit"].numpy()
        mask = np.isfinite(distance)
        if not mask.any():
            raise ValueError("CAD view has no visible surface.")
        points = (rays[mask, :3].astype(float) + distance[mask, None] * rays[mask, 3:].astype(float)) * radius + center
        uv = hit["primitive_uvs"].numpy()[mask].astype(float)
        barycentric = np.column_stack((1 - uv.sum(axis=1), uv))
        ids = hit["primitive_ids"].numpy()[mask]
        normals = np.sum(vertex_normals[triangles[ids]] * barycentric[..., None], axis=1)
        illumination = .35 + .65 * np.abs(hit["primitive_normals"].numpy() @ direction)
        rgb = np.full((association.IMAGE_SIZE, association.IMAGE_SIZE, 3), 255, dtype=np.uint8)
        rgb[mask] = np.clip(255 * illumination[mask, None] * color, 0, 255).astype(np.uint8)
        depth_m = np.full(mask.shape, np.nan)
        depth_m[mask] = distance[mask] * radius
        yield {"surface_points_m": points, "surface_normals": normals, "rgb": rgb,
               "ray_depth_m": depth_m, "mask": mask, "center_m": center, "radius_m": radius,
               "direction_cad": direction, "orthographic_camera_center_m": center + 3 * radius * direction}


def prepare_visible_geometry(cad_path, color, templates, registrar):
    """Cache exact surfaces, depths, and 5-mm local FPFH; no pair-score cache."""
    key = association.cache_key(
        "visible_cad_v1", association.content_hash(cad_path), color,
        association.content_hash(Path(__file__)), association.content_hash(Path(association.__file__)),
        asdict(registrar.config), o3d.__version__, association.IMAGE_SIZE, association.VIEW_DIRECTIONS.tolist(),
    )
    directory = CACHE_ROOT / key
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"view_{i:02}.npz" for i in range(1, 15)]
    cache_hit = all(p.exists() for p in paths)
    start = perf_counter()
    geometries, metadata = [], []
    surfaces = visible_surfaces(cad_path, color) if not cache_hit else None
    for index, path in enumerate(paths):
        if cache_hit:
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
        else:
            surface = next(surfaces)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surface["surface_points_m"]))
            cloud.normals = o3d.utility.Vector3dVector(surface["surface_normals"])
            # Exactly the full-CAD preprocessing call; no camera orientation override.
            down, fpfh = registrar.compute_fpfh(cloud)
            arrays = {**surface, "points": np.asarray(down.points), "normals": np.asarray(down.normals), "fpfh": fpfh.data}
            np.savez_compressed(path, **arrays)
        np.testing.assert_array_equal(arrays["rgb"], templates[index])
        geometries.append(geometry_from_arrays(arrays))
        metadata.append({"view_index_1based": index + 1, "cache_path": str(path),
                         "cache_sha256": association.content_hash(path), "cache_hit": cache_hit,
                         "surface_point_count": len(arrays["surface_points_m"]), "downsampled_point_count": len(arrays["points"]),
                         "direction_cad": arrays["direction_cad"].tolist(), "center_m": arrays["center_m"].tolist(),
                         "radius_m": float(arrays["radius_m"])})
    return geometries, metadata, perf_counter() - start


def coverage_diagnostics(cad, observed, raw):
    """Coverage on the actual registration point sets, without another voxel pass."""
    transform = raw["T_observed_from_cad"]
    if transform is None:
        return {"C_obs": raw["registration_fitness"], "C_cad": None, "C_f1": None,
                "cad_correspondence_count": None, "coverage_status": "no_alignment"}
    aligned = o3d.geometry.PointCloud(cad).transform(np.asarray(transform))
    obs_distances = np.asarray(observed.compute_point_cloud_distance(aligned))
    cad_distances = np.asarray(aligned.compute_point_cloud_distance(observed))
    radius = raw["max_correspondence_distance_m"]
    obs_count = int(np.count_nonzero(obs_distances < radius))
    cad_count = int(np.count_nonzero(cad_distances < radius))
    c_obs, c_cad = obs_count / len(observed.points), cad_count / len(cad.points)
    if obs_count != raw["correspondence_count"] or c_obs != raw["registration_fitness"]:
        raise ValueError("Coverage does not reproduce the recorded registration correspondences.")
    return {"C_obs": c_obs, "C_cad": c_cad, "C_f1": 2 * c_obs * c_cad / (c_obs + c_cad) if c_obs + c_cad else 0.,
            "cad_correspondence_count": cad_count, "coverage_status": "aligned"}
