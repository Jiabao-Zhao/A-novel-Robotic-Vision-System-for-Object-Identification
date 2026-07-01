import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from visualize_pose import build_visualization_metadata, orient_normal_to_points, show_visualization


@dataclass
class PipelineConfig:
    output_dir: Path = Path("Output/pointcloud_pose")
    warmup_frames: int = 30
    voxel_size_m: float = 0.003
    plane_distance_threshold_m: float = 0.005
    dbscan_eps_m: float = 0.02
    dbscan_min_points: int = 30
    min_cluster_points: int = 100


@dataclass
class PoseResult:
    id: str
    frame: str
    box: list[int]
    center_mm: list[float]
    dimensions_mm: list[float]
    rpy_deg: list[float]
    yaw_deg: float
    yaw_confidence: float
    rotation_matrix: list[list[float]]
    point_count: int


@dataclass
class PipelineRun:
    results: list[PoseResult]
    clouds: dict
    visualization_metadata: dict
    color_image_rgb: np.ndarray


CONFIG = PipelineConfig()

CALIBRATED_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 623.9816462749620,
    "fy": 613.8080113982506,
    "cx": 318.5163260449835,
    "cy": 237.5512378918142,
}


def capture_realsense_rgbd(config):
    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    stream_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    pipeline.start(stream_config)
    align = rs.align(rs.stream.color)

    try:
        for _ in range(config.warmup_frames):
            align.process(pipeline.wait_for_frames())

        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame.keep()
        color_frame.keep()

        return color_frame, depth_frame
    finally:
        pipeline.stop()


def rgbd_to_pointcloud(color_frame, depth_frame):
    depth_units = depth_frame.get_units()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.asarray(color_frame.get_data())),
        o3d.geometry.Image(np.asarray(depth_frame.get_data())),
        depth_scale=1.0 / depth_units,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )

    camera = o3d.camera.PinholeCameraIntrinsic(
        CALIBRATED_INTRINSICS["width"],
        CALIBRATED_INTRINSICS["height"],
        CALIBRATED_INTRINSICS["fx"],
        CALIBRATED_INTRINSICS["fy"],
        CALIBRATED_INTRINSICS["cx"],
        CALIBRATED_INTRINSICS["cy"],
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)


def remove_table_plane(pcd, config):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=config.plane_distance_threshold_m,
        ransac_n=3,
        num_iterations=1000,
    )

    object_cloud = pcd.select_by_index(inliers, invert=True)
    table_cloud = pcd.select_by_index(inliers)

    return object_cloud, plane_model, table_cloud


def cluster_objects(object_cloud, config):
    labels = np.asarray(
        object_cloud.cluster_dbscan(
            eps=config.dbscan_eps_m,
            min_points=config.dbscan_min_points,
            print_progress=False,
        )
    )

    clusters = []
    for label in sorted(set(labels.tolist()) - {-1}):
        indices = np.flatnonzero(labels == label).tolist()
        if len(indices) >= config.min_cluster_points:
            clusters.append(object_cloud.select_by_index(indices))

    clusters.sort(key=lambda cluster: len(cluster.points), reverse=True)
    return clusters


def estimate_pose(cluster, plane_model, frame):
    points = np.asarray(cluster.points)

    z_axis = orient_normal_to_points(plane_model, points)

    xy_centered = points[:, :2] - points[:, :2].mean(axis=0)
    covariance = np.cov(xy_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    lambda_1 = float(eigenvalues[order[0]])
    lambda_2 = float(eigenvalues[order[1]])

    major_axis_xy = eigenvectors[:, order[0]]
    yaw_rad = float(np.arctan2(major_axis_xy[1], major_axis_xy[0]))
    yaw_confidence = max(0.0, (lambda_1 - lambda_2) / (lambda_1 + 1e-12))

    x_axis = np.array([major_axis_xy[0], major_axis_xy[1], 0.0], dtype=float)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    local_points = points @ rotation_matrix
    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)

    center_m = ((local_min + local_max) * 0.5) @ rotation_matrix.T
    dimensions_m = local_max - local_min
    rpy_deg = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)

    return PoseResult(
        id="",
        frame=frame,
        box=project_points_to_bbox(points),
        center_mm=(center_m * 1000.0).round(3).tolist(),
        dimensions_mm=(dimensions_m * 1000.0).round(3).tolist(),
        rpy_deg=np.round(rpy_deg, 3).tolist(),
        yaw_deg=round(np.degrees(yaw_rad), 3),
        yaw_confidence=round(float(yaw_confidence), 6),
        rotation_matrix=np.round(rotation_matrix, 6).tolist(),
        point_count=int(len(points)),
    )


def project_points_to_bbox(points):
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if not np.any(valid):
        return [0, 0, 0, 0]

    visible = points[valid]
    u = CALIBRATED_INTRINSICS["fx"] * visible[:, 0] / visible[:, 2] + CALIBRATED_INTRINSICS["cx"]
    v = CALIBRATED_INTRINSICS["fy"] * visible[:, 1] / visible[:, 2] + CALIBRATED_INTRINSICS["cy"]
    u = np.clip(u, 0, CALIBRATED_INTRINSICS["width"] - 1)
    v = np.clip(v, 0, CALIBRATED_INTRINSICS["height"] - 1)

    return [
        int(np.floor(u.min())),
        int(np.floor(v.min())),
        int(np.ceil(u.max())),
        int(np.ceil(v.max())),
    ]


def clear_generated_outputs(output_dir):
    output_dir = Path(output_dir)
    visualization_dir = output_dir.parent / "visualization"

    for path in [
        output_dir / "pose_results.json",
        output_dir / "workspace_cloud.ply",
        visualization_dir / "table_plane_cloud.ply",
        visualization_dir / "visualization_metadata.json",
    ]:
        if path.exists():
            path.unlink()

    if output_dir.exists():
        for path in output_dir.glob("object_cluster_*.ply"):
            path.unlink()


def save_outputs(results, clouds, output_dir, visualization_metadata):
    output_dir = Path(output_dir)
    visualization_dir = output_dir.parent / "visualization"

    clear_generated_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(output_dir / "workspace_cloud.ply"), clouds["workspace"])
    o3d.io.write_point_cloud(str(visualization_dir / "table_plane_cloud.ply"), clouds["table"])

    for index, cluster in enumerate(clouds["clusters"], start=1):
        o3d.io.write_point_cloud(str(output_dir / f"object_cluster_{index}.ply"), cluster)

    payload = {
        "frame": results[0].frame if results else "camera",
        "object_count": len(results),
        "objects": [asdict(result) for result in results],
    }

    with (output_dir / "pose_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    with (visualization_dir / "visualization_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(visualization_metadata, file, indent=2)


def run_pipeline(config=CONFIG, save_results=True, visualize=True):
    color_frame, depth_frame = capture_realsense_rgbd(config)
    color_image_rgb = np.asarray(color_frame.get_data()).copy()
    pcd = rgbd_to_pointcloud(color_frame, depth_frame)

    workspace_cloud = pcd.voxel_down_sample(config.voxel_size_m)

    object_cloud, plane_model, table_cloud = remove_table_plane(workspace_cloud, config)
    clusters = cluster_objects(object_cloud, config)

    results = []
    for index, cluster in enumerate(clusters, start=1):
        result = estimate_pose(cluster, plane_model, "camera")
        result.id = str(index)
        results.append(result)

    clouds = {
        "workspace": workspace_cloud,
        "table": table_cloud,
        "clusters": clusters,
    }

    metadata = build_visualization_metadata(plane_model, table_cloud, clusters, "camera")

    if save_results:
        save_outputs(results, clouds, config.output_dir, metadata)
    if visualize:
        show_visualization(results, clouds, metadata)

    return PipelineRun(
        results=results,
        clouds=clouds,
        visualization_metadata=metadata,
        color_image_rgb=color_image_rgb,
    )


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
