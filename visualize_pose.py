import json
from pathlib import Path

import numpy as np
import open3d as o3d


RESULT_DIR = Path("Output/pointcloud_pose")
VISUALIZATION_DIR = Path("Output/visualization")
AXIS_LENGTH_M = 0.08


def orient_normal_to_points(plane_model, points) -> np.ndarray:
    normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Table plane normal has zero length.")

    z_axis = normal / normal_norm
    if len(points) == 0:
        return z_axis

    signed_distance = float(np.mean(points, axis=0) @ z_axis + plane_model[3] / normal_norm)
    return -z_axis if signed_distance < 0 else z_axis


def build_visualization_metadata(plane_model, table_cloud, clusters, source_frame) -> dict:
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
    origin_m = table_points.mean(axis=0) if len(table_points) else np.zeros(3)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return {
        "source_frame": source_frame,
        "display_frame": "tabletop_visualization",
        "origin_m": np.round(origin_m, 6).tolist(),
        "basis_columns_source": np.round(basis, 6).tolist(),
    }


def transform_points(points, metadata) -> np.ndarray:
    origin = np.asarray(metadata["origin_m"], dtype=float)
    basis = np.asarray(metadata["basis_columns_source"], dtype=float)
    return (points - origin) @ basis


def transform_rotation(rotation_matrix, metadata) -> np.ndarray:
    basis = np.asarray(metadata["basis_columns_source"], dtype=float)
    return basis.T @ np.asarray(rotation_matrix, dtype=float)


def pose_field(result, name):
    if isinstance(result, dict):
        return result[name]
    return getattr(result, name)


def make_red_pose_axes(center, rotation, axis_length_m=AXIS_LENGTH_M):
    points = [center]
    lines = []
    colors = []

    for axis_index in range(3):
        points.append(center + rotation[:, axis_index] * axis_length_m)
        lines.append([0, axis_index + 1])
        colors.append([1.0, 0.0, 0.0])

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length_m * 0.08)
    marker.translate(center)
    marker.paint_uniform_color([1.0, 0.0, 0.0])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return [line_set, marker]


def show_visualization(results, clouds, metadata, axis_length_m=AXIS_LENGTH_M):
    geometries = []

    table_cloud = o3d.geometry.PointCloud(clouds["table"])
    table_cloud.points = o3d.utility.Vector3dVector(
        transform_points(np.asarray(table_cloud.points), metadata)
    )
    table_cloud.paint_uniform_color([0.55, 0.55, 0.55])
    geometries.append(table_cloud)

    for cluster in clouds["clusters"]:
        object_cloud = o3d.geometry.PointCloud(cluster)
        object_cloud.points = o3d.utility.Vector3dVector(
            transform_points(np.asarray(object_cloud.points), metadata)
        )
        object_cloud.paint_uniform_color([0.0, 0.25, 1.0])
        geometries.append(object_cloud)

    for result in results:
        center = np.asarray(pose_field(result, "center_mm"), dtype=float) / 1000.0
        center = transform_points(center.reshape(1, 3), metadata)[0]
        rotation = transform_rotation(pose_field(result, "rotation_matrix"), metadata)
        geometries.extend(make_red_pose_axes(center, rotation, axis_length_m))

        x_mm, y_mm, z_mm = pose_field(result, "center_mm")
        roll_deg, pitch_deg, yaw_deg = pose_field(result, "rpy_deg")
        print(
            f"{pose_field(result, 'object_id')}: "
            f"x={x_mm:.3f} mm, y={y_mm:.3f} mm, z={z_mm:.3f} mm, "
            f"roll={roll_deg:.3f} deg, pitch={pitch_deg:.3f} deg, yaw={yaw_deg:.3f} deg"
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Pose Visualization",
        width=1000,
        height=700,
        front=[0.0, -1.0, 0.45],
        up=[0.0, 0.0, 1.0],
        zoom=0.7,
    )


def show_saved_visualization(result_dir=RESULT_DIR, visualization_dir=VISUALIZATION_DIR):
    pose_path = result_dir / "pose_results.json"
    plane_path = visualization_dir / "table_plane_cloud.ply"
    metadata_path = visualization_dir / "visualization_metadata.json"

    if not pose_path.exists():
        raise FileNotFoundError(f"Missing pose results: {pose_path}")
    if not plane_path.exists():
        raise FileNotFoundError(f"Missing segmented table cloud: {plane_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing visualization metadata: {metadata_path}")

    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    clouds = {
        "table": o3d.io.read_point_cloud(str(plane_path)),
        "clusters": [
            o3d.io.read_point_cloud(str(path))
            for path in sorted(result_dir.glob("object_cluster_*.ply"))
        ],
    }
    show_visualization(payload["objects"], clouds, metadata)


def main():
    show_saved_visualization()


if __name__ == "__main__":
    main()
