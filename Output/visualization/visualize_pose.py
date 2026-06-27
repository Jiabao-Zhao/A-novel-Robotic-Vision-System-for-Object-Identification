import argparse
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
        width=1000,
        height=700,
        front=[0.0, -1.0, 0.45],
        up=[0.0, 0.0, 1.0],
        zoom=0.7,
    )


if __name__ == "__main__":
    main()
