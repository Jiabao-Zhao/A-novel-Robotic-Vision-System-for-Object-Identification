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
    icp_stages: tuple = (
        (0.020, 0.040, 80),
        (0.012, 0.020, 60),
        (0.008, 0.012, 40),
    )
    rotation_det_tolerance: float = 1e-3
    max_translation_m: float = 2.0
    min_icp_fitness: float = 1e-6


class CADPointCloudRegistration:
    """
    Table-constrained CAD-to-observed-cloud registration.

    The selected transform is named T_observed_from_cad and maps points from
    the CAD frame into the observed RealSense camera frame.
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
        self._estimate_normals(down, self.config.voxel_size_m)
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

    def generate_initial_transforms(
        self,
        cad_cloud,
        observed_cloud,
        plane_model=None,
        cad_metadata=None,
        source_down=None,
        target_down=None,
        source_fpfh=None,
        target_fpfh=None,
    ):
        """Generate deterministic CAD-to-camera initial transform candidates."""
        cad_metadata = cad_metadata or {}
        candidates = []
        seen = set()
        cad_center = self.cloud_center(cad_cloud)
        observed_center = self.cloud_center(observed_cloud)

        def add_candidate(name, source, rotation, ransac_result=None):
            rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = observed_center - rotation @ cad_center
            key = tuple(np.round(transform, 6).reshape(-1).tolist())
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "name": name,
                    "source": source,
                    "transform": transform,
                    "ransac_result": ransac_result,
                }
            )

        add_candidate("center_identity", "center", np.eye(3))

        table_normal = self.oriented_table_normal(plane_model, observed_center)
        if table_normal is not None:
            cad_up = self.axis_from_metadata(cad_metadata.get("cad_up_axis", "Z"))
            base_rotation = self.rotation_between_vectors(cad_up, table_normal)
            yaw_values = cad_metadata.get("yaw_candidates_deg", [0, 90, 180, 270])
            table_x, table_y = self.table_basis(table_normal)
            for yaw_deg in yaw_values:
                yaw_rotation = self.rotation_about_axis(table_normal, np.deg2rad(float(yaw_deg)))
                yaw_base = yaw_rotation @ base_rotation
                add_candidate(f"table_yaw_{int(float(yaw_deg))}", "table_yaw", yaw_base)
                for axis_name, flip_axis, side in [
                    ("flip_x", np.array([1.0, 0.0, 0.0]), "right"),
                    ("flip_y", np.array([0.0, 1.0, 0.0]), "right"),
                    ("flip_z", np.array([0.0, 0.0, 1.0]), "right"),
                    ("flip_table_x", table_x, "left"),
                    ("flip_table_y", table_y, "left"),
                ]:
                    flip = self.rotation_about_axis(flip_axis, np.pi)
                    rotation = yaw_base @ flip if side == "right" else flip @ yaw_base
                    add_candidate(
                        f"table_yaw_{int(float(yaw_deg))}_{axis_name}",
                        "axis_flip",
                        rotation,
                    )

        if source_down is not None and target_down is not None:
            for index in range(self.config.ransac_attempts):
                ransac = self.compute_global_registration(
                    source_down,
                    target_down,
                    source_fpfh,
                    target_fpfh,
                )
                transform = np.asarray(ransac.transformation, dtype=float)
                key = tuple(np.round(transform, 6).reshape(-1).tolist())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "candidate_id": len(candidates) + 1,
                        "name": f"ransac_{index + 1}",
                        "source": "ransac",
                        "transform": transform,
                        "ransac_result": ransac,
                    }
                )

        return candidates

    def refine_with_icp(self, source_cloud, target_cloud, initial_transform):
        """
        Run point-to-plane ICP from coarse to fine.

        Returns the final Open3D registration result and the final transform.
        """
        transform = np.asarray(initial_transform, dtype=float).copy()
        final_result = None
        result = None
        for stage_index, (voxel_size_m, distance_m, max_iterations) in enumerate(self.config.icp_stages):
            source = source_cloud.voxel_down_sample(float(voxel_size_m))
            target = target_cloud.voxel_down_sample(float(voxel_size_m))
            if source.is_empty() or target.is_empty():
                continue
            self._estimate_normals(source, float(voxel_size_m))
            self._estimate_normals(target, float(voxel_size_m))
            target.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
            estimation = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
                if stage_index == 0
                else o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            final_result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                float(distance_m),
                transform,
                estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=int(max_iterations)
                ),
            )
            if final_result.fitness <= self.config.min_icp_fitness:
                if result is None:
                    result = final_result
                break
            result = final_result
            transform = np.asarray(result.transformation, dtype=float)

        if result is None:
            raise RuntimeError("ICP could not run because a downsampled cloud was empty.")
        return result, transform

    def score_alignment(self, cad_cloud, observed_cloud, transform):
        """Score the final CAD-to-observed alignment using bidirectional distances."""
        cad_aligned = o3d.geometry.PointCloud(cad_cloud)
        cad_aligned.transform(transform)
        cad_score = cad_aligned.voxel_down_sample(self.config.voxel_size_m)
        obs_score = observed_cloud.voxel_down_sample(self.config.voxel_size_m)
        if cad_score.is_empty():
            cad_score = cad_aligned
        if obs_score.is_empty():
            obs_score = observed_cloud

        cad_to_obs = np.asarray(cad_score.compute_point_cloud_distance(obs_score), dtype=float)
        obs_to_cad = np.asarray(obs_score.compute_point_cloud_distance(cad_score), dtype=float)
        mean_cad_to_obs = self.safe_mean(cad_to_obs)
        mean_obs_to_cad = self.safe_mean(obs_to_cad)
        if obs_to_cad.size:
            observed_outlier_ratio = float(
                np.mean(obs_to_cad > 3.0 * self.config.voxel_size_m)
            )
        else:
            observed_outlier_ratio = 1.0

        center_error = float(
            np.linalg.norm(self.cloud_center(cad_aligned) - self.cloud_center(observed_cloud))
        )
        score = (
            mean_cad_to_obs
            + mean_obs_to_cad
            + 0.02 * observed_outlier_ratio
            + 0.10 * center_error
        )
        return {
            "score": float(score),
            "mean_cad_to_obs_m": float(mean_cad_to_obs),
            "mean_obs_to_cad_m": float(mean_obs_to_cad),
            "observed_outlier_ratio": float(observed_outlier_ratio),
            "center_error_m": float(center_error),
        }

    def validate_transform(self, transform):
        """Validate a 4x4 rigid transform and reject reflections."""
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            return False, "transform is not 4x4"
        if not np.isfinite(transform).all():
            return False, "transform contains non-finite values"
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            return False, "last row is not [0, 0, 0, 1]"
        rotation = transform[:3, :3]
        determinant = float(np.linalg.det(rotation))
        if determinant < 0:
            return False, "rotation determinant indicates reflection"
        if abs(determinant - 1.0) > self.config.rotation_det_tolerance:
            return False, "rotation determinant is not close to +1"
        if np.linalg.norm(transform[:3, 3]) > self.config.max_translation_m:
            return False, "translation magnitude is too large"
        return True, None

    def build_augmented_cloud(self, observed_cloud, aligned_cad_cloud):
        augmented = o3d.geometry.PointCloud(observed_cloud)
        augmented += o3d.geometry.PointCloud(aligned_cad_cloud)
        return augmented

    def run(
        self,
        cad_path,
        observed_cloud_path,
        plane_model=None,
        cad_metadata=None,
        output_dir="output/registered_point_cloud",
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cad_cloud = self.load_cad_as_point_cloud(cad_path)
        observed_cloud = self.load_observed_cloud(observed_cloud_path)
        diagnostics = self.cloud_diagnostics(cad_cloud, observed_cloud)
        source_down, source_fpfh = self.preprocess_cloud(cad_cloud)
        target_down, target_fpfh = self.preprocess_cloud(observed_cloud)

        selected, candidate_records = self.find_best_registration(
            cad_cloud=cad_cloud,
            observed_cloud=observed_cloud,
            plane_model=plane_model,
            cad_metadata=cad_metadata,
            source_down=source_down,
            target_down=target_down,
            source_fpfh=source_fpfh,
            target_fpfh=target_fpfh,
        )

        T_observed_from_cad = np.asarray(selected["final_transform"], dtype=float)
        aligned_cad_cloud = o3d.geometry.PointCloud(cad_cloud)
        aligned_cad_cloud.transform(T_observed_from_cad)
        augmented_cloud = self.build_augmented_cloud(observed_cloud, aligned_cad_cloud)

        cad_sampled_path = output_dir / "cad_sampled_cloud.ply"
        observed_path = output_dir / "observed_cloud.ply"
        aligned_path = output_dir / "cad_aligned_cloud.ply"
        augmented_path = output_dir / "augmented_object_cloud.ply"
        result_path = output_dir / "cad_registration_result.json"
        candidates_path = output_dir / "cad_registration_candidates.json"

        o3d.io.write_point_cloud(str(cad_sampled_path), cad_cloud)
        o3d.io.write_point_cloud(str(observed_path), observed_cloud)
        o3d.io.write_point_cloud(str(aligned_path), aligned_cad_cloud)
        o3d.io.write_point_cloud(str(augmented_path), augmented_cloud)
        candidates_path.write_text(json.dumps(candidate_records, indent=2), encoding="utf-8")

        result = {
            "cad_path": str(cad_path),
            "observed_cloud_path": str(observed_cloud_path),
            "cad_sampled_cloud_path": str(cad_sampled_path),
            "aligned_cad_cloud_path": str(aligned_path),
            "augmented_cloud_path": str(augmented_path),
            "candidates_path": str(candidates_path),
            "selected_candidate_id": int(selected["candidate_id"]),
            "selected_candidate_name": selected["candidate_name"],
            "selected_candidate_source": selected["source"],
            "transformation_matrix": T_observed_from_cad.tolist(),
            "T_observed_from_cad": T_observed_from_cad.tolist(),
            "rotation_determinant": float(np.linalg.det(T_observed_from_cad[:3, :3])),
            "cad_point_count": diagnostics["cad_point_count"],
            "observed_point_count": diagnostics["observed_point_count"],
            "cad_extent_m": diagnostics["cad_extent_m"],
            "observed_extent_m": diagnostics["observed_extent_m"],
            "cad_center_m": diagnostics["cad_center_m"],
            "observed_center_m": diagnostics["observed_center_m"],
            "extent_ratio_observed_to_cad": diagnostics["extent_ratio_observed_to_cad"],
            "warnings": diagnostics["warnings"],
            "final_score": float(selected["score"]),
            "final_score_breakdown": selected["score_breakdown"],
            "icp_fitness": selected["icp_fitness"],
            "icp_inlier_rmse": selected["icp_inlier_rmse"],
            "ransac_fitness": selected.get("ransac_fitness"),
            "ransac_inlier_rmse": selected.get("ransac_inlier_rmse"),
            "method": "table-constrained multi-hypothesis ICP with RANSAC fallback",
            "frame": "observed camera frame",
            "transform_direction": "CAD frame to observed camera frame",
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["result_path"] = str(result_path)
        return result

    def find_best_registration(
        self,
        cad_cloud,
        observed_cloud,
        plane_model=None,
        cad_metadata=None,
        source_down=None,
        target_down=None,
        source_fpfh=None,
        target_fpfh=None,
    ):
        candidates = self.generate_initial_transforms(
            cad_cloud=cad_cloud,
            observed_cloud=observed_cloud,
            plane_model=plane_model,
            cad_metadata=cad_metadata,
            source_down=source_down,
            target_down=target_down,
            source_fpfh=source_fpfh,
            target_fpfh=target_fpfh,
        )
        records = []
        best = None
        rejection_reasons = []

        for candidate in candidates:
            record = self.empty_candidate_record(candidate)
            initial_valid, initial_reason = self.validate_transform(candidate["transform"])
            if not initial_valid:
                record["rejected_reason"] = initial_reason
                rejection_reasons.append(f"{record['candidate_name']}: {initial_reason}")
                records.append(record)
                continue

            try:
                icp, final_transform = self.refine_with_icp(
                    cad_cloud,
                    observed_cloud,
                    candidate["transform"],
                )
            except RuntimeError as exc:
                record["rejected_reason"] = str(exc)
                rejection_reasons.append(f"{record['candidate_name']}: {exc}")
                records.append(record)
                continue

            final_valid, final_reason = self.validate_transform(final_transform)
            record["final_transform"] = final_transform.tolist()
            record["rotation_determinant"] = float(np.linalg.det(final_transform[:3, :3]))
            record["icp_fitness"] = float(icp.fitness)
            record["icp_inlier_rmse"] = float(icp.inlier_rmse)
            if record["icp_fitness"] <= self.config.min_icp_fitness:
                reason = "ICP found no valid correspondences"
                record["rejected_reason"] = reason
                rejection_reasons.append(f"{record['candidate_name']}: {reason}")
                records.append(record)
                continue
            if not final_valid:
                record["rejected_reason"] = final_reason
                rejection_reasons.append(f"{record['candidate_name']}: {final_reason}")
                records.append(record)
                continue

            score = self.score_alignment(cad_cloud, observed_cloud, final_transform)
            record["score"] = float(score["score"])
            record["score_breakdown"] = score
            ransac = candidate.get("ransac_result")
            if ransac is not None:
                record["ransac_fitness"] = float(ransac.fitness)
                record["ransac_inlier_rmse"] = float(ransac.inlier_rmse)
            records.append(record)

            if best is None or record["score"] < best["score"]:
                best = record

        if best is None:
            raise RuntimeError(
                "All CAD registration candidates failed: "
                + " | ".join(rejection_reasons)
            )
        return best, records

    def empty_candidate_record(self, candidate):
        ransac = candidate.get("ransac_result")
        return {
            "candidate_id": int(candidate["candidate_id"]),
            "candidate_name": candidate["name"],
            "source": candidate["source"],
            "initial_transform": np.asarray(candidate["transform"]).tolist(),
            "final_transform": None,
            "rotation_determinant": None,
            "icp_fitness": None,
            "icp_inlier_rmse": None,
            "ransac_fitness": float(ransac.fitness) if ransac is not None else None,
            "ransac_inlier_rmse": float(ransac.inlier_rmse) if ransac is not None else None,
            "score": None,
            "score_breakdown": None,
            "rejected_reason": None,
        }

    def cloud_diagnostics(self, cad_cloud, observed_cloud):
        cad_box = cad_cloud.get_axis_aligned_bounding_box()
        observed_box = observed_cloud.get_axis_aligned_bounding_box()
        cad_extent = np.asarray(cad_box.get_extent(), dtype=float)
        observed_extent = np.asarray(observed_box.get_extent(), dtype=float)
        cad_max = float(np.max(cad_extent)) if cad_extent.size else 0.0
        observed_max = float(np.max(observed_extent)) if observed_extent.size else 0.0
        ratio = observed_max / cad_max if cad_max > 0 else np.inf
        warnings = []
        if ratio > 3.0:
            warnings.append("observed cluster is more than 3x larger than CAD extent")
        if ratio < 0.25:
            warnings.append("observed cluster is less than 0.25x CAD extent")
        return {
            "cad_point_count": int(len(cad_cloud.points)),
            "observed_point_count": int(len(observed_cloud.points)),
            "cad_extent_m": cad_extent.tolist(),
            "observed_extent_m": observed_extent.tolist(),
            "cad_center_m": np.asarray(cad_box.get_center(), dtype=float).tolist(),
            "observed_center_m": np.asarray(observed_box.get_center(), dtype=float).tolist(),
            "extent_ratio_observed_to_cad": float(ratio),
            "warnings": warnings,
        }

    def oriented_table_normal(self, plane_model, observed_center):
        if plane_model is None:
            return None
        values = np.asarray(plane_model, dtype=float).reshape(-1)
        if values.size != 4:
            return None
        normal = values[:3]
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            return None
        normal = normal / normal_norm
        signed_distance = float(np.dot(normal, observed_center) + values[3] / normal_norm)
        return -normal if signed_distance < 0 else normal

    @staticmethod
    def axis_from_metadata(axis_name):
        axes = {
            "X": np.array([1.0, 0.0, 0.0]),
            "-X": np.array([-1.0, 0.0, 0.0]),
            "Y": np.array([0.0, 1.0, 0.0]),
            "-Y": np.array([0.0, -1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0]),
            "-Z": np.array([0.0, 0.0, -1.0]),
        }
        return axes.get(str(axis_name).upper(), axes["Z"])

    @staticmethod
    def rotation_between_vectors(source, target):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source /= np.linalg.norm(source)
        target /= np.linalg.norm(target)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if np.isclose(dot, 1.0):
            return np.eye(3)
        if np.isclose(dot, -1.0):
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(source, helper)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            axis = np.cross(source, helper)
            axis /= np.linalg.norm(axis)
            return CADPointCloudRegistration.rotation_about_axis(axis, np.pi)
        axis = np.cross(source, target)
        skew = CADPointCloudRegistration.skew(axis)
        return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))

    @staticmethod
    def rotation_about_axis(axis, angle_rad):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ]
        )

    @staticmethod
    def table_basis(table_normal):
        normal = np.asarray(table_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        x_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x_axis, normal)) > 0.9:
            x_axis = np.array([0.0, 1.0, 0.0])
        x_axis = x_axis - np.dot(x_axis, normal) * normal
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        return x_axis, y_axis

    @staticmethod
    def skew(vector):
        x, y, z = vector
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    @staticmethod
    def cloud_center(cloud):
        return np.asarray(cloud.get_axis_aligned_bounding_box().get_center(), dtype=float)

    @staticmethod
    def safe_mean(values):
        if values.size == 0:
            return float("inf")
        return float(np.mean(values))

    def _estimate_normals(self, cloud, voxel_size_m):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(voxel_size_m) * self.config.normal_radius_factor,
                max_nn=30,
            )
        )


if __name__ == "__main__":
    CADPointCloudRegistration().run(
        cad_path="path/to/cad_model.stl",
        observed_cloud_path="output/point_cloud_localization/object_cluster_1.ply",
    )
