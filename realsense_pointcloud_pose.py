import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation


_OUTPUT_FRAME = "camera"


VISUALIZATION_SCRIPT = '''import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def transform_points(points, metadata):
    origin = np.asarray(metadata["origin_m"], dtype=float)
    basis = np.asarray(metadata["basis_columns_source"], dtype=float)
    return (points - origin) @ basis


def transform_rotation(rotation, metadata):
    basis = np.asarray(metadata["basis_columns_source"], dtype=float)
    return basis.T @ rotation


def make_red_pose_axes(center, rotation, axis_length=0.08):
    points = [center]
    lines = []
    colors = []

    for axis_index in range(3):
        points.append(center + rotation[:, axis_index] * axis_length)
        lines.append([0, axis_index + 1])
        colors.append([1.0, 0.0, 0.0])

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.08)
    marker.translate(center)
    marker.paint_uniform_color([1.0, 0.0, 0.0])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return [line_set, marker]


def main():
    parser = argparse.ArgumentParser(description="Visualize segmented tabletop pose results.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "pointcloud_pose",
    )
    parser.add_argument("--axis-length", type=float, default=0.08)
    args = parser.parse_args()

    result_dir = args.result_dir
    visualization_dir = Path(__file__).resolve().parent
    plane_path = visualization_dir / "table_plane_cloud.ply"
    metadata_path = visualization_dir / "visualization_metadata.json"
    json_path = result_dir / "pose_results.json"

    if not plane_path.exists():
        raise FileNotFoundError(f"Missing segmented plane cloud: {plane_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing visualization metadata: {metadata_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing pose results: {json_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    geometries = []
    plane = o3d.io.read_point_cloud(str(plane_path))
    plane.points = o3d.utility.Vector3dVector(transform_points(np.asarray(plane.points), metadata))
    plane.paint_uniform_color([0.55, 0.55, 0.55])
    geometries.append(plane)

    for cluster_path in sorted(result_dir.glob("object_cluster_*.ply")):
        cluster = o3d.io.read_point_cloud(str(cluster_path))
        cluster.points = o3d.utility.Vector3dVector(transform_points(np.asarray(cluster.points), metadata))
        cluster.paint_uniform_color([0.0, 0.25, 1.0])
        geometries.append(cluster)

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    for result in payload["objects"]:
        center = np.asarray(result["center_mm"], dtype=float) / 1000.0
        center = transform_points(center.reshape(1, 3), metadata)[0]
        rotation = transform_rotation(np.asarray(result["rotation_matrix"], dtype=float), metadata)
        geometries.extend(make_red_pose_axes(center, rotation, args.axis_length))
        x_mm, y_mm, z_mm = result["center_mm"]
        roll_deg, pitch_deg, yaw_deg = result["rpy_deg"]
        print(
            f'{result["object_id"]}: '
            f'x={x_mm:.3f} mm, y={y_mm:.3f} mm, z={z_mm:.3f} mm, '
            f'roll={roll_deg:.3f} deg, pitch={pitch_deg:.3f} deg, yaw={yaw_deg:.3f} deg'
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Pose Visualization",
        width=1280,
        height=720,
        front=[0.0, -1.0, 0.45],
        up=[0.0, 0.0, 1.0],
        zoom=0.7,
    )


if __name__ == "__main__":
    main()
'''


@dataclass
class PoseResult:
    object_id: str
    frame: str
    center_mm: list[float]
    dimensions_mm: list[float]
    rpy_deg: list[float]
    yaw_deg: float
    yaw_confidence: float
    rotation_matrix: list[list[float]]
    point_count: int


def capture_realsense_rgbd(args) -> tuple:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        for _ in range(args.warmup_frames):
            align.process(pipeline.wait_for_frames())

        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture aligned RealSense color and depth frames.")

        depth_frame.keep()
        color_frame.keep()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return color_frame, depth_frame, intrinsics
    finally:
        pipeline.stop()


def rgbd_to_pointcloud(color_frame, depth_frame, intrinsics) -> o3d.geometry.PointCloud:
    color = np.asarray(color_frame.get_data())
    depth = np.asarray(depth_frame.get_data())
    depth_units = depth_frame.get_units()
    if depth_units <= 0:
        raise ValueError("RealSense depth units must be positive.")

    color_image = o3d.geometry.Image(color)
    depth_image = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0 / depth_units,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )
    camera = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.ppx,
        intrinsics.ppy,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)


def apply_transform(pcd, transform) -> o3d.geometry.PointCloud:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 homogeneous matrix.")

    transformed = o3d.geometry.PointCloud(pcd)
    transformed.transform(transform)
    return transformed


def crop_workspace(pcd, bounds) -> o3d.geometry.PointCloud:
    if all(value is None for value in bounds.values()):
        return pcd

    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Cannot crop an empty point cloud.")

    mask = np.ones(len(points), dtype=bool)
    axes = {"x": 0, "y": 1, "z": 2}
    for axis, index in axes.items():
        lower = bounds[f"{axis}_min"]
        upper = bounds[f"{axis}_max"]
        if lower is not None:
            mask &= points[:, index] >= lower
        if upper is not None:
            mask &= points[:, index] <= upper
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{axis}-min cannot be greater than {axis}-max.")

    cropped = pcd.select_by_index(np.flatnonzero(mask).tolist())
    if len(cropped.points) == 0:
        raise ValueError("Workspace crop removed all points.")
    return cropped


def remove_table_plane(pcd, args) -> tuple:
    if len(pcd.points) < 3:
        raise ValueError("At least 3 points are required to segment a table plane.")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=args.plane_distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    if len(inliers) == 0:
        raise RuntimeError("Plane segmentation found no table inliers.")

    table_cloud = pcd.select_by_index(inliers)
    object_cloud = pcd.select_by_index(inliers, invert=True)
    return object_cloud, plane_model, table_cloud


def cluster_objects(object_cloud, args) -> list[o3d.geometry.PointCloud]:
    if len(object_cloud.points) == 0:
        return []

    labels = np.asarray(
        object_cloud.cluster_dbscan(
            eps=args.dbscan_eps,
            min_points=args.dbscan_min_points,
            print_progress=False,
        )
    )
    clusters = []
    for label in sorted(set(labels.tolist()) - {-1}):
        indices = np.flatnonzero(labels == label).tolist()
        if len(indices) >= args.min_cluster_points:
            clusters.append(object_cloud.select_by_index(indices))

    clusters.sort(key=lambda cluster: len(cluster.points), reverse=True)
    return clusters


def orient_normal_to_points(plane_model, points) -> np.ndarray:
    normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Table plane normal has zero length.")

    z_axis = normal / normal_norm
    if len(points) == 0:
        return z_axis

    signed_distance = float(np.mean(points, axis=0) @ z_axis + plane_model[3] / normal_norm)
    if signed_distance < 0:
        z_axis = -z_axis
    return z_axis


def build_visualization_metadata(plane_model, table_cloud, clusters) -> dict:
    if clusters:
        object_points = np.vstack([np.asarray(cluster.points) for cluster in clusters])
    else:
        object_points = np.empty((0, 3))

    z_axis = orient_normal_to_points(plane_model, object_points)
    x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.array([0.0, -1.0, 0.0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    table_points = np.asarray(table_cloud.points)
    origin = table_points.mean(axis=0) if len(table_points) else np.zeros(3)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return {
        "source_frame": _OUTPUT_FRAME,
        "display_frame": "tabletop_visualization",
        "origin_m": np.round(origin, 6).tolist(),
        "basis_columns_source": np.round(basis, 6).tolist(),
    }


def transform_points_to_display(points, metadata) -> np.ndarray:
    origin = np.asarray(metadata["origin_m"], dtype=float)
    basis = np.asarray(metadata["basis_columns_source"], dtype=float)
    return (points - origin) @ basis


def transform_rotation_to_display(rotation_matrix, metadata) -> np.ndarray:
    basis = np.asarray(metadata["basis_columns_source"], dtype=float)
    return basis.T @ np.asarray(rotation_matrix, dtype=float)


def clear_generated_outputs(output_dir):
    output_dir = Path(output_dir)
    visualization_dir = output_dir.parent / "visualization"

    for path in [
        output_dir / "pose_results.json",
        output_dir / "workspace_cloud.ply",
        visualization_dir / "table_plane_cloud.ply",
        visualization_dir / "visualization_metadata.json",
        visualization_dir / "visualize_pose.py",
    ]:
        if path.exists():
            path.unlink()

    if output_dir.exists():
        for path in output_dir.glob("object_cluster_*.ply"):
            path.unlink()


def estimate_pose(cluster, plane_model) -> PoseResult:
    points = np.asarray(cluster.points)
    if len(points) < 3:
        raise ValueError("At least 3 cluster points are required to estimate pose.")

    z_axis = orient_normal_to_points(plane_model, points)

    xy = points[:, :2]
    xy_centered = xy - xy.mean(axis=0)
    covariance = np.cov(xy_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    lambda_1 = float(eigenvalues[order[0]])
    lambda_2 = float(eigenvalues[order[1]])
    major_axis_xy = eigenvectors[:, order[0]]
    yaw = float(np.arctan2(major_axis_xy[1], major_axis_xy[0]))
    yaw_confidence = max(0.0, (lambda_1 - lambda_2) / (lambda_1 + 1e-12))

    # The yaw axis comes from XY PCA, then roll/pitch come from the table normal.
    x_axis = np.array([major_axis_xy[0], major_axis_xy[1], 0.0], dtype=float)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm == 0:
        raise ValueError("PCA major axis is parallel to the table normal.")
    x_axis /= x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm == 0:
        raise ValueError("Cannot construct object frame from table normal and yaw axis.")
    y_axis /= y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation_matrix) < 0:
        y_axis = -y_axis
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    local_points = points @ rotation_matrix
    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)
    dimensions_m = local_max - local_min
    center_m = ((local_min + local_max) * 0.5) @ rotation_matrix.T

    rpy_deg = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    return PoseResult(
        object_id="",
        frame=_OUTPUT_FRAME,
        center_mm=(center_m * 1000.0).round(3).tolist(),
        dimensions_mm=(dimensions_m * 1000.0).round(3).tolist(),
        rpy_deg=np.round(rpy_deg, 3).tolist(),
        yaw_deg=round(np.degrees(yaw), 3),
        yaw_confidence=round(float(yaw_confidence), 6),
        rotation_matrix=np.round(rotation_matrix, 6).tolist(),
        point_count=int(len(points)),
    )


def make_red_pose_axes(center, rotation, axis_length=0.08):
    points = [center]
    lines = []
    colors = []

    for axis_index in range(3):
        points.append(center + rotation[:, axis_index] * axis_length)
        lines.append([0, axis_index + 1])
        colors.append([1.0, 0.0, 0.0])

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.08)
    marker.translate(center)
    marker.paint_uniform_color([1.0, 0.0, 0.0])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return [line_set, marker]


def show_visualization(results, clouds, metadata, axis_length=0.08):
    geometries = []

    table_cloud = o3d.geometry.PointCloud(clouds["table"])
    table_points = transform_points_to_display(np.asarray(table_cloud.points), metadata)
    table_cloud.points = o3d.utility.Vector3dVector(table_points)
    table_cloud.paint_uniform_color([0.55, 0.55, 0.55])
    geometries.append(table_cloud)

    for cluster in clouds["clusters"]:
        object_cloud = o3d.geometry.PointCloud(cluster)
        object_points = transform_points_to_display(np.asarray(object_cloud.points), metadata)
        object_cloud.points = o3d.utility.Vector3dVector(object_points)
        object_cloud.paint_uniform_color([0.0, 0.25, 1.0])
        geometries.append(object_cloud)

    for result in results:
        center = np.asarray(result.center_mm, dtype=float) / 1000.0
        center = transform_points_to_display(center.reshape(1, 3), metadata)[0]
        rotation = transform_rotation_to_display(result.rotation_matrix, metadata)
        geometries.extend(make_red_pose_axes(center, rotation, axis_length))
        x_mm, y_mm, z_mm = result.center_mm
        roll_deg, pitch_deg, yaw_deg = result.rpy_deg
        print(
            f"{result.object_id}: "
            f"x={x_mm:.3f} mm, y={y_mm:.3f} mm, z={z_mm:.3f} mm, "
            f"roll={roll_deg:.3f} deg, pitch={pitch_deg:.3f} deg, yaw={yaw_deg:.3f} deg"
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Pose Visualization",
        width=1280,
        height=720,
        front=[0.0, -1.0, 0.45],
        up=[0.0, 0.0, 1.0],
        zoom=0.7,
    )


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
        "frame": _OUTPUT_FRAME,
        "object_count": len(results),
        "objects": [asdict(result) for result in results],
    }
    with (output_dir / "pose_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    with (visualization_dir / "visualization_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(visualization_metadata, file, indent=2)
    with (visualization_dir / "visualize_pose.py").open("w", encoding="utf-8") as file:
        file.write(VISUALIZATION_SCRIPT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture RealSense RGB-D data and estimate tabletop object poses."
    )
    parser.add_argument("--output-dir", default="Output/pointcloud_pose")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--plane-distance-threshold", type=float, default=0.005)
    parser.add_argument("--dbscan-eps", type=float, default=0.02)
    parser.add_argument("--dbscan-min-points", type=int, default=30)
    parser.add_argument("--min-cluster-points", type=int, default=100)
    parser.add_argument("--t-base-camera", type=Path)
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--no-show-visualization", action="store_true")
    return parser.parse_args()


def main():
    global _OUTPUT_FRAME

    args = parse_args()
    if args.warmup_frames < 0:
        raise ValueError("warmup-frames must be non-negative.")
    if args.voxel_size < 0:
        raise ValueError("voxel-size must be non-negative.")
    if args.plane_distance_threshold <= 0:
        raise ValueError("plane-distance-threshold must be positive.")
    if args.dbscan_eps <= 0:
        raise ValueError("dbscan-eps must be positive.")
    if args.dbscan_min_points <= 0 or args.min_cluster_points <= 0:
        raise ValueError("DBSCAN and cluster point thresholds must be positive.")

    print("Capturing aligned RealSense RGB-D frame...")
    color_frame, depth_frame, intrinsics = capture_realsense_rgbd(args)
    pcd = rgbd_to_pointcloud(color_frame, depth_frame, intrinsics)

    if args.t_base_camera is not None:
        transform = np.load(args.t_base_camera)
        pcd = apply_transform(pcd, transform)
        _OUTPUT_FRAME = "base"

    bounds = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
    }
    workspace_cloud = crop_workspace(pcd, bounds)
    if args.voxel_size > 0:
        workspace_cloud = workspace_cloud.voxel_down_sample(args.voxel_size)

    print(f"Workspace cloud: {len(workspace_cloud.points)} points.")
    object_cloud, plane_model, table_cloud = remove_table_plane(workspace_cloud, args)
    clusters = cluster_objects(object_cloud, args)

    results = []
    for index, cluster in enumerate(clusters, start=1):
        result = estimate_pose(cluster, plane_model)
        result.object_id = f"object_{index:03d}"
        results.append(result)

    clouds = {"workspace": workspace_cloud, "table": table_cloud, "clusters": clusters}
    visualization_metadata = build_visualization_metadata(plane_model, table_cloud, clusters)
    save_outputs(results, clouds, args.output_dir, visualization_metadata)
    print(f"Detected {len(results)} object clusters.")
    print(f"Saved pose results and point clouds to {Path(args.output_dir).resolve()}.")
    print(f"Saved visualization script to {(Path(args.output_dir).parent / 'visualization').resolve()}.")
    if not args.no_show_visualization:
        print("Opening Open3D visualization window...")
        show_visualization(results, clouds, visualization_metadata)


if __name__ == "__main__":
    main()
