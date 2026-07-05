import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from camera_capturing import CameraCapture
from helper_function import RBGAnnotation


CALIBRATED_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 623.9816462749620,
    "fy": 613.8080113982506,
    "cx": 318.5163260449835,
    "cy": 237.5512378918142,
}


@dataclass
class PointCloudConfig:
    output_dir: Path = Path("output/point_cloud_localization")
    annotation_dir: Path = Path("output/annotation")
    voxel_size_m: float = 0.003
    plane_distance_threshold_m: float = 0.005
    dbscan_eps_m: float = 0.02
    dbscan_min_points: int = 30
    min_cluster_points: int = 100
    raw_cluster_padding_m: float = 0.004


@dataclass
class LocalizedObject:
    object_id: str
    roi: dict
    pointcloud_path: str = ""


class PointCloudLocalization:
    def __init__(self, config=None):
        self.config = PointCloudConfig() if config is None else config
        self.camera = CameraCapture()
        self.localization_path = self.config.output_dir / "point_cloud_localization.json"
        self.annotated_rgb_path = self.config.annotation_dir / "RGB_point_cloud_roi_annotation.png"
        self.cluster_colors = [
            [0.90, 0.10, 0.10],
            [0.10, 0.55, 0.95],
            [0.10, 0.75, 0.25],
            [0.95, 0.60, 0.10],
            [0.60, 0.25, 0.95],
            [0.95, 0.25, 0.70],
            [0.10, 0.85, 0.85],
            [0.80, 0.80, 0.10],
        ]

    def run(self, visualize=False):
        rgb, depth, depth_scale_m = self.camera.capture_rgbd()
        rgb_path, depth_path = self.camera.save_raw_capture(rgb, depth, depth_scale_m)
        return self.run_from_arrays(
            rgb=rgb,
            depth=depth,
            depth_scale_m=depth_scale_m,
            rgb_path=rgb_path,
            depth_path=depth_path,
            visualize=visualize,
        )

    def run_from_saved_raw(
        self,
        rgb_path=Path("output/raw/RGB.png"),
        depth_path=Path("output/raw/depth_data.npz"),
        visualize=False,
    ):
        rgb = np.asarray(o3d.io.read_image(str(rgb_path)))[:, :, :3]
        with np.load(depth_path) as data:
            depth = data["depth_data"]
            depth_scale_m = float(data["depth_scale_m"])
        return self.run_from_arrays(
            rgb=rgb,
            depth=depth,
            depth_scale_m=depth_scale_m,
            rgb_path=Path(rgb_path),
            depth_path=Path(depth_path),
            visualize=visualize,
        )

    def run_from_arrays(self, rgb, depth, depth_scale_m, rgb_path, depth_path, visualize=False):
        workspace_cloud = self.rgbd_to_pointcloud(rgb, depth, depth_scale_m)
        downsampled_cloud = workspace_cloud.voxel_down_sample(self.config.voxel_size_m)
        object_cloud, plane_model, table_cloud = self.segment_table_plane(downsampled_cloud)
        clusters = self.cluster_objects(object_cloud)
        objects = self.localize_clusters(clusters)
        paths = self.save_outputs(
            rgb_path=Path(rgb_path),
            depth_path=Path(depth_path),
            workspace_cloud=workspace_cloud,
            downsampled_cloud=downsampled_cloud,
            table_cloud=table_cloud,
            clusters=clusters,
            plane_model=plane_model,
            objects=objects,
        )

        if visualize:
            self.show_segmentation(table_cloud, clusters)

        return paths

    def rgbd_to_pointcloud(self, rgb, depth, depth_scale_m):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0 / float(depth_scale_m),
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.camera_intrinsics(),
        )

    def camera_intrinsics(self):
        return o3d.camera.PinholeCameraIntrinsic(
            CALIBRATED_INTRINSICS["width"],
            CALIBRATED_INTRINSICS["height"],
            CALIBRATED_INTRINSICS["fx"],
            CALIBRATED_INTRINSICS["fy"],
            CALIBRATED_INTRINSICS["cx"],
            CALIBRATED_INTRINSICS["cy"],
        )

    def segment_table_plane(self, workspace_cloud):
        plane_model, inliers = workspace_cloud.segment_plane(
            distance_threshold=self.config.plane_distance_threshold_m,
            ransac_n=3,
            num_iterations=1000,
        )
        object_cloud = workspace_cloud.select_by_index(inliers, invert=True)
        table_cloud = workspace_cloud.select_by_index(inliers)
        return object_cloud, plane_model, table_cloud

    def cluster_objects(self, object_cloud):
        labels = np.asarray(
            object_cloud.cluster_dbscan(
                eps=self.config.dbscan_eps_m,
                min_points=self.config.dbscan_min_points,
                print_progress=False,
            )
        )
        clusters = []
        for label in sorted(set(labels.tolist()) - {-1}):
            indices = np.flatnonzero(labels == label).tolist()
            if len(indices) >= self.config.min_cluster_points:
                clusters.append(object_cloud.select_by_index(indices))
        clusters.sort(key=lambda cluster: len(cluster.points), reverse=True)
        return clusters

    def localize_clusters(self, clusters):
        objects = []
        for index, cluster in enumerate(clusters, start=1):
            points = np.asarray(cluster.points)
            if points.size == 0:
                continue

            objects.append(
                LocalizedObject(
                    object_id=f"object_{index:03d}",
                    roi=self.project_points_to_roi(points),
                )
            )
        return objects

    def project_points_to_roi(self, points):
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        if not np.any(valid):
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        visible = points[valid]
        u = CALIBRATED_INTRINSICS["fx"] * visible[:, 0] / visible[:, 2] + CALIBRATED_INTRINSICS["cx"]
        v = CALIBRATED_INTRINSICS["fy"] * visible[:, 1] / visible[:, 2] + CALIBRATED_INTRINSICS["cy"]
        u = np.clip(u, 0, CALIBRATED_INTRINSICS["width"] - 1)
        v = np.clip(v, 0, CALIBRATED_INTRINSICS["height"] - 1)

        x0 = int(np.floor(u.min()))
        y0 = int(np.floor(v.min()))
        x1 = int(np.ceil(u.max()))
        y1 = int(np.ceil(v.max()))
        return {"x1": x0, "y1": y0, "x2": x1, "y2": y1}

    def save_outputs(
        self,
        rgb_path,
        depth_path,
        workspace_cloud,
        downsampled_cloud,
        table_cloud,
        clusters,
        plane_model,
        objects,
    ):
        self.clear_outputs()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.annotation_dir.mkdir(parents=True, exist_ok=True)

        workspace_path = self.config.output_dir / "workspace_cloud.ply"
        downsampled_workspace_path = self.config.output_dir / "workspace_cloud_downsampled.ply"
        table_path = self.config.output_dir / "table_plane_cloud.ply"
        segmented_path = self.config.output_dir / "segmented_table_and_objects.ply"

        o3d.io.write_point_cloud(str(workspace_path), workspace_cloud)
        o3d.io.write_point_cloud(str(downsampled_workspace_path), downsampled_cloud)
        o3d.io.write_point_cloud(str(table_path), table_cloud)
        raw_object_cloud = self.remove_table_from_raw_cloud(workspace_cloud, plane_model)

        segmented_cloud = o3d.geometry.PointCloud()
        table_vis = o3d.geometry.PointCloud(table_cloud)
        table_vis.paint_uniform_color([0.55, 0.55, 0.55])
        segmented_cloud += table_vis

        for index, cluster in enumerate(clusters, start=1):
            object_path = self.config.output_dir / f"object_cluster_{index}.ply"
            downsampled_object_path = self.config.output_dir / f"object_cluster_{index}_downsampled.ply"
            raw_cluster = self.crop_raw_cluster(raw_object_cloud, cluster)
            cluster_vis = o3d.geometry.PointCloud(cluster)
            color = self.cluster_colors[(index - 1) % len(self.cluster_colors)]
            cluster_vis.paint_uniform_color(color)
            o3d.io.write_point_cloud(str(object_path), raw_cluster)
            o3d.io.write_point_cloud(str(downsampled_object_path), cluster_vis)
            segmented_cloud += cluster_vis
            if index <= len(objects):
                objects[index - 1].pointcloud_path = str(object_path)

        o3d.io.write_point_cloud(str(segmented_path), segmented_cloud)

        payload = {
            "frame": "camera",
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "object_count": len(objects),
            "plane_model": [float(value) for value in plane_model],
            "objects": [asdict(obj) for obj in objects],
        }
        self.localization_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        annotated_path = self.annotate_rgb(rgb_path, objects)
        return {
            "rgb": Path(rgb_path),
            "depth": Path(depth_path),
            "localization": self.localization_path,
            "annotated_rgb": annotated_path,
            "workspace_cloud": workspace_path,
            "downsampled_workspace_cloud": downsampled_workspace_path,
            "table_cloud": table_path,
            "segmented_cloud": segmented_path,
            "object_count": len(objects),
        }

    def remove_table_from_raw_cloud(self, workspace_cloud, plane_model):
        points = np.asarray(workspace_cloud.points)
        normal = np.asarray(plane_model[:3], dtype=float)
        normal_norm = np.linalg.norm(normal)
        distances = np.abs(points @ normal + float(plane_model[3])) / normal_norm
        indices = np.flatnonzero(distances > self.config.plane_distance_threshold_m).tolist()
        return workspace_cloud.select_by_index(indices)

    def crop_raw_cluster(self, raw_object_cloud, downsampled_cluster):
        bounds = downsampled_cluster.get_axis_aligned_bounding_box()
        padding = np.full(3, self.config.raw_cluster_padding_m)
        bounds.min_bound = bounds.min_bound - padding
        bounds.max_bound = bounds.max_bound + padding
        raw_cluster = raw_object_cloud.crop(bounds)
        if raw_cluster.is_empty():
            return o3d.geometry.PointCloud(downsampled_cluster)
        return raw_cluster

    def annotate_rgb(self, rgb_path, objects):
        annotator = RBGAnnotation()
        annotator.rgb_path = Path(rgb_path)
        annotator.output_dir = self.config.annotation_dir
        annotator.output_path = self.annotated_rgb_path
        image = annotator.load_rgb_image()
        candidates = [
            {
                "id": obj.object_id,
                "label": obj.object_id.removeprefix("object_"),
                "roi": obj.roi,
            }
            for obj in objects
        ]
        return annotator.save_annotation(annotator.annotate(image, candidates))

    def clear_outputs(self):
        for path in [
            self.localization_path,
            self.annotated_rgb_path,
            self.config.output_dir / "workspace_cloud.ply",
            self.config.output_dir / "workspace_cloud_downsampled.ply",
            self.config.output_dir / "table_plane_cloud.ply",
            self.config.output_dir / "segmented_table_and_objects.ply",
        ]:
            if path.exists():
                path.unlink()
        if self.config.output_dir.exists():
            for pattern in ["object_cluster_*.ply", "object_cluster_*_downsampled.ply"]:
                for path in self.config.output_dir.glob(pattern):
                    path.unlink()

    def show_segmentation(self, table_cloud, clusters):
        geometries = []
        table_vis = o3d.geometry.PointCloud(table_cloud)
        table_vis.paint_uniform_color([0.55, 0.55, 0.55])
        geometries.append(table_vis)

        for index, cluster in enumerate(clusters, start=1):
            cluster_vis = o3d.geometry.PointCloud(cluster)
            color = self.cluster_colors[(index - 1) % len(self.cluster_colors)]
            cluster_vis.paint_uniform_color(color)
            geometries.append(cluster_vis)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Point Cloud Object Segmentation",
            width=1000,
            height=700,
            front=[0.0, -1.0, 0.45],
            up=[0.0, 0.0, 1.0],
            zoom=0.7,
        )


if __name__ == "__main__":
    paths = PointCloudLocalization().run(visualize=True)
    print(f"Saved localization JSON: {paths['localization']}")
    print(f"Saved annotated RGB image: {paths['annotated_rgb']}")
    print(f"Saved segmented point cloud: {paths['segmented_cloud']}")
