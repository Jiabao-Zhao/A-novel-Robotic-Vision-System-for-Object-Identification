import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class CADRegistrationConfig:
    voxel_size_m: float = 0.005
    cad_sample_points: int = 20000
    normal_radius_factor: float = 2.0
    fpfh_radius_factor: float = 5.0
    ransac_distance_factor: float = 1.5
    ransac_attempts: int = 5
    icp_distance_factors: tuple = (8.0, 3.0, 1.5)


class CADPointCloudRegistration:
    """
    CPU CAD-to-observed-cloud registration baseline:

        T* = arg min_T d(T P_cad, P_obs)

    P_cad is sampled from the retrieved CAD model, P_obs is the partial
    RealSense object cloud, and T* maps the CAD frame into the observed camera
    frame. The transform is initialized with Open3D FPFH/RANSAC global
    registration and refined with ICP.
    """

    def __init__(self, config=None):
        self.config = CADRegistrationConfig() if config is None else config

    def load_observed_cloud(self, observed_cloud_path):
        path = Path(observed_cloud_path)
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f"Observed point cloud is empty or unreadable: {path}")
        return cloud

    def load_cad_as_point_cloud(self, cad_path):
        path = Path(cad_path)
        suffix = path.suffix.lower()
        point_cloud_suffixes = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}
        mesh_suffixes = {".stl", ".obj", ".off", ".gltf", ".glb"}

        if suffix in point_cloud_suffixes:
            cloud = o3d.io.read_point_cloud(str(path))
            if not cloud.is_empty():
                return cloud

        if suffix in mesh_suffixes or suffix == ".ply":
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError(f"CAD file is not a readable triangle mesh: {path}")
            mesh.compute_vertex_normals()
            return mesh.sample_points_uniformly(
                number_of_points=self.config.cad_sample_points
            )

        supported = ".ply, .pcd, .xyz, .xyzn, .xyzrgb, .stl, .obj, .off, .gltf, .glb"
        raise ValueError(
            f"Unsupported CAD format for Open3D direct loading: {path.suffix}. "
            f"Use one of: {supported}."
        )

    def preprocess_cloud(self, cloud):
        down = cloud.voxel_down_sample(self.config.voxel_size_m)
        if down.is_empty():
            raise ValueError("Downsampled point cloud is empty.")
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.voxel_size_m * self.config.normal_radius_factor,
                max_nn=30,
            )
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.voxel_size_m * self.config.fpfh_radius_factor,
                max_nn=100,
            ),
        )
        return down, fpfh

    def compute_global_registration(
        self,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ):
        distance_threshold = self.config.voxel_size_m * self.config.ransac_distance_factor
        registration = o3d.pipelines.registration
        return registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            registration.TransformationEstimationPointToPoint(False),
            3,
            [
                registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            registration.RANSACConvergenceCriteria(100000, 0.999),
        )

    def refine_with_icp(self, source_cloud, target_cloud, initial_transform):
        source = o3d.geometry.PointCloud(source_cloud)
        target = o3d.geometry.PointCloud(target_cloud)
        self._estimate_normals(source)
        self._estimate_normals(target)
        transform = initial_transform
        result = None
        for factor in self.config.icp_distance_factors:
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                self.config.voxel_size_m * factor,
                transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )
            transform = result.transformation
        return result

    def build_augmented_cloud(self, observed_cloud, aligned_cad_cloud):
        augmented = o3d.geometry.PointCloud(observed_cloud)
        augmented += o3d.geometry.PointCloud(aligned_cad_cloud)
        return augmented

    def run(
        self,
        cad_path,
        observed_cloud_path,
        output_dir="output/registered_point_cloud",
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cad_cloud = self.load_cad_as_point_cloud(cad_path)
        observed_cloud = self.load_observed_cloud(observed_cloud_path)
        source_down, source_fpfh = self.preprocess_cloud(cad_cloud)
        target_down, target_fpfh = self.preprocess_cloud(observed_cloud)

        ransac, icp = self.find_best_registration(
            cad_cloud,
            observed_cloud,
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
        )

        aligned_cad_cloud = o3d.geometry.PointCloud(cad_cloud)
        aligned_cad_cloud.transform(icp.transformation)
        augmented_cloud = self.build_augmented_cloud(observed_cloud, aligned_cad_cloud)

        cad_sampled_path = output_dir / "cad_sampled_cloud.ply"
        observed_path = output_dir / "observed_cloud.ply"
        aligned_path = output_dir / "cad_aligned_cloud.ply"
        augmented_path = output_dir / "augmented_object_cloud.ply"
        result_path = output_dir / "cad_registration_result.json"

        o3d.io.write_point_cloud(str(cad_sampled_path), cad_cloud)
        o3d.io.write_point_cloud(str(observed_path), observed_cloud)
        o3d.io.write_point_cloud(str(aligned_path), aligned_cad_cloud)
        o3d.io.write_point_cloud(str(augmented_path), augmented_cloud)

        result = {
            "cad_path": str(cad_path),
            "observed_cloud_path": str(observed_cloud_path),
            "cad_sampled_cloud_path": str(cad_sampled_path),
            "aligned_cad_cloud_path": str(aligned_path),
            "augmented_cloud_path": str(augmented_path),
            "transformation_matrix": np.asarray(icp.transformation).tolist(),
            "ransac_fitness": float(ransac.fitness),
            "ransac_inlier_rmse": float(ransac.inlier_rmse),
            "icp_fitness": float(icp.fitness),
            "icp_inlier_rmse": float(icp.inlier_rmse),
            "icp_distance_thresholds_m": [
                float(self.config.voxel_size_m * factor)
                for factor in self.config.icp_distance_factors
            ],
            "method": "Open3D FPFH RANSAC + ICP",
            "frame": "observed camera frame",
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["result_path"] = str(result_path)
        return result

    def find_best_registration(
        self,
        cad_cloud,
        observed_cloud,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ):
        best_ransac = None
        best_icp = None
        for _ in range(self.config.ransac_attempts):
            ransac = self.compute_global_registration(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
            )
            icp = self.refine_with_icp(cad_cloud, observed_cloud, ransac.transformation)
            if best_icp is None or self.registration_score(icp) > self.registration_score(best_icp):
                best_ransac = ransac
                best_icp = icp
        return best_ransac, best_icp

    @staticmethod
    def registration_score(result):
        return float(result.fitness), -float(result.inlier_rmse)

    def _estimate_normals(self, cloud):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.voxel_size_m * self.config.normal_radius_factor,
                max_nn=30,
            )
        )
