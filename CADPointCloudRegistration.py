import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class CADRegistrationConfig:
    voxel_size_m: float = 0.005
    cad_sample_points: int = 20000
    auto_scale_cad_to_meters: bool = True
    cad_mm_extent_threshold_m: float = 1.0
    yaw_step_deg: float = 10.0
    constrained_iterations: int = 12
    xy_step_m: float = 0.003
    yaw_step_refine_deg: float = 3.0
    min_xy_step_m: float = 0.00025
    min_yaw_step_refine_deg: float = 0.25
    random_seed: int = 0


class CADPointCloudRegistration:
    """
    Tabletop-constrained CAD-to-observed-cloud alignment.

    This estimates only the tabletop degrees of freedom: translation along the
    table plane and yaw around the table normal. The selected transform is named
    T_observed_from_cad and maps CAD-frame points into the observed RealSense
    camera frame.
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
                self.normalize_cad_units(cloud)
                return cloud

        if suffix in mesh_suffixes or suffix == ".ply":
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError(f"CAD file is not a readable triangle mesh: {path}")
            self.normalize_cad_units(mesh)
            mesh.compute_vertex_normals()
            o3d.utility.random.seed(int(self.config.random_seed))
            return mesh.sample_points_uniformly(self.config.cad_sample_points)

        supported = ".ply, .pcd, .xyz, .xyzn, .xyzrgb, .stl, .obj, .off, .gltf, .glb"
        raise ValueError(
            f"Unsupported CAD format for Open3D direct loading: {path.suffix}. "
            f"Use one of: {supported}."
        )

    def normalize_cad_units(self, geometry):
        """
        Open3D reads CAD/STL coordinates without unit metadata.

        The perception pipeline works in meters. If a CAD file has an extent
        larger than a plausible tabletop object in meters, treat it as
        millimeters and scale it by 0.001.
        """
        if not self.config.auto_scale_cad_to_meters:
            return geometry
        extent = np.asarray(geometry.get_axis_aligned_bounding_box().get_extent(), dtype=float)
        if extent.size and float(np.max(extent)) > self.config.cad_mm_extent_threshold_m:
            geometry.scale(0.001, center=(0.0, 0.0, 0.0))
        return geometry

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
        selected, candidate_records, warnings = self.find_best_tabletop_alignment(
            cad_cloud,
            observed_cloud,
            plane_model,
            cad_metadata,
        )

        T_observed_from_cad = np.asarray(selected["final_transform"], dtype=float)
        cad_cloud.paint_uniform_color([0.0, 0.45, 1.0])
        observed_cloud.paint_uniform_color([1.0, 0.0, 0.0])
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
            "result_path": str(result_path),
            "selected_candidate_id": int(selected["candidate_id"]),
            "selected_candidate_name": selected["candidate_name"],
            "selected_candidate_source": selected["source"],
            "transformation_matrix": T_observed_from_cad.tolist(),
            "T_observed_from_cad": T_observed_from_cad.tolist(),
            "cad_point_count": diagnostics["cad_point_count"],
            "observed_point_count": diagnostics["observed_point_count"],
            "cad_extent_m": diagnostics["cad_extent_m"],
            "observed_extent_m": diagnostics["observed_extent_m"],
            "cad_center_m": diagnostics["cad_center_m"],
            "observed_center_m": diagnostics["observed_center_m"],
            "extent_ratio_observed_to_cad": diagnostics["extent_ratio_observed_to_cad"],
            "warnings": diagnostics["warnings"] + warnings,
            "final_score": float(selected["score"]),
            "final_score_breakdown": selected["score_breakdown"],
            "constrained_rmse_m": selected["rmse_m"],
            "icp_fitness": None,
            "icp_inlier_rmse": selected["rmse_m"],
            "method": "tabletop-constrained x-y-yaw CAD alignment",
            "frame": "observed camera frame",
            "transform_direction": "CAD frame to observed camera frame",
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def find_best_tabletop_alignment(
        self,
        cad_cloud,
        observed_cloud,
        plane_model=None,
        cad_metadata=None,
    ):
        cad_metadata = cad_metadata or {}
        table = self.table_frame(plane_model, observed_cloud)
        warnings = table["warnings"]
        cad_up_axis_names = cad_metadata.get("cad_up_axis_candidates")
        if cad_up_axis_names is None:
            cad_up_axis_names = (
                [cad_metadata["cad_up_axis"]]
                if "cad_up_axis" in cad_metadata
                else self.default_cad_up_axes()
            )
        if isinstance(cad_up_axis_names, str):
            cad_up_axis_names = [cad_up_axis_names]
        yaw_candidates = cad_metadata.get("yaw_candidates_deg", self.default_yaw_candidates())

        records = []
        best = None
        for cad_up_axis_name in cad_up_axis_names:
            cad_up = self.axis_from_metadata(cad_up_axis_name)
            base_rotation = self.rotation_between_vectors(cad_up, table["normal"])
            for yaw_deg in yaw_candidates:
                initial = self.table_contact_transform(
                    cad_cloud,
                    observed_cloud,
                    table,
                    base_rotation,
                    float(yaw_deg),
                )
                final_transform, score_breakdown = self.refine_xy_yaw(
                    cad_cloud,
                    observed_cloud,
                    table,
                    initial,
                )
                record = {
                    "candidate_id": len(records) + 1,
                    "candidate_name": f"cad_up_{cad_up_axis_name}_yaw_{float(yaw_deg):.1f}",
                    "source": "table_up_axis_yaw",
                    "cad_up_axis": str(cad_up_axis_name),
                    "initial_yaw_deg": float(yaw_deg),
                    "initial_transform": initial.tolist(),
                    "final_transform": final_transform.tolist(),
                    "score": float(score_breakdown["rmse_m"]),
                    "rmse_m": float(score_breakdown["rmse_m"]),
                    "score_breakdown": score_breakdown,
                    "rejected_reason": None,
                }
                records.append(record)
                if best is None or record["score"] < best["score"]:
                    best = record

        if best is None:
            raise RuntimeError("No tabletop yaw candidates were generated.")
        return best, records, warnings

    def table_contact_transform(
        self,
        cad_cloud,
        observed_cloud,
        table,
        base_rotation,
        yaw_deg,
    ):
        normal = table["normal"]
        yaw_rotation = self.rotation_about_axis(normal, np.deg2rad(float(yaw_deg)))
        rotation = yaw_rotation @ base_rotation
        rotated = self.transform_points(np.asarray(cad_cloud.points), rotation, np.zeros(3))
        observed_points = np.asarray(observed_cloud.points)

        cad_plane_distances = rotated @ normal + table["d"]
        contact_shift = -float(np.min(cad_plane_distances)) * normal

        cad_center_on_plane = self.project_to_plane(np.mean(rotated + contact_shift, axis=0), table)
        observed_center_on_plane = self.project_to_plane(np.mean(observed_points, axis=0), table)
        plane_shift = observed_center_on_plane - cad_center_on_plane

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = contact_shift + plane_shift
        return transform

    def refine_xy_yaw(self, cad_cloud, observed_cloud, table, initial_transform):
        """
        Constrained ICP-like local search.

        The update is limited to table-x translation, table-y translation, and
        yaw around the table normal. Z/contact, roll, and pitch are not allowed
        to drift during refinement.
        """
        transform = np.asarray(initial_transform, dtype=float).copy()
        xy_step = float(self.config.xy_step_m)
        yaw_step = np.deg2rad(float(self.config.yaw_step_refine_deg))
        best_rmse = self.observed_to_cad_rmse(cad_cloud, observed_cloud, transform)
        iterations = 0

        while iterations < int(self.config.constrained_iterations):
            iterations += 1
            improved = False
            proposals = [
                self.delta_transform(table["x_axis"] * xy_step, np.eye(3)),
                self.delta_transform(-table["x_axis"] * xy_step, np.eye(3)),
                self.delta_transform(table["y_axis"] * xy_step, np.eye(3)),
                self.delta_transform(-table["y_axis"] * xy_step, np.eye(3)),
                self.delta_transform(np.zeros(3), self.rotation_about_axis(table["normal"], yaw_step)),
                self.delta_transform(np.zeros(3), self.rotation_about_axis(table["normal"], -yaw_step)),
            ]
            for delta in proposals:
                candidate = delta @ transform
                rmse = self.observed_to_cad_rmse(cad_cloud, observed_cloud, candidate)
                if rmse < best_rmse:
                    transform = candidate
                    best_rmse = rmse
                    improved = True

            if not improved:
                xy_step *= 0.5
                yaw_step *= 0.5
                if (
                    xy_step < self.config.min_xy_step_m
                    and np.rad2deg(yaw_step) < self.config.min_yaw_step_refine_deg
                ):
                    break

        return transform, {
            "rmse_m": float(best_rmse),
            "score_direction": "observed_surface_to_cad_surface",
            "iterations": int(iterations),
            "degrees_of_freedom": "table_x, table_y, yaw_about_table_normal",
            "yaw_refinement": "constrained local search",
        }

    def observed_to_cad_rmse(self, cad_cloud, observed_cloud, transform):
        """Measure how well the observed partial surface is supported by the CAD surface."""
        cad_eval = o3d.geometry.PointCloud(cad_cloud)
        cad_eval.transform(transform)
        cad_eval = cad_eval.voxel_down_sample(self.config.voxel_size_m)
        observed_eval = observed_cloud.voxel_down_sample(self.config.voxel_size_m)
        if cad_eval.is_empty():
            cad_eval = o3d.geometry.PointCloud(cad_cloud).transform(transform)
        if observed_eval.is_empty():
            observed_eval = observed_cloud
        distances = np.asarray(observed_eval.compute_point_cloud_distance(cad_eval), dtype=float)
        if distances.size == 0:
            return float("inf")
        return float(np.sqrt(np.mean(distances * distances)))

    def table_frame(self, plane_model, observed_cloud):
        points = np.asarray(observed_cloud.points)
        warnings = []
        if plane_model is None:
            z_min = float(np.min(points[:, 2]))
            normal = np.array([0.0, 0.0, 1.0])
            d = -z_min
            warnings.append("plane_model missing; used horizontal plane at observed minimum z")
        else:
            values = np.asarray(plane_model, dtype=float).reshape(-1)
            if values.size != 4:
                raise ValueError("plane_model must contain [a, b, c, d].")
            normal = values[:3]
            norm = np.linalg.norm(normal)
            if norm == 0:
                raise ValueError("plane_model normal has zero length.")
            normal = normal / norm
            d = float(values[3]) / norm
            observed_center = np.mean(points, axis=0)
            if float(np.dot(normal, observed_center) + d) < 0:
                normal = -normal
                d = -d

        x_axis, y_axis = self.table_basis(normal)
        return {
            "normal": normal,
            "d": float(d),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "warnings": warnings,
        }

    def project_to_plane(self, point, table):
        signed_distance = float(np.dot(table["normal"], point) + table["d"])
        return np.asarray(point, dtype=float) - signed_distance * table["normal"]

    def build_augmented_cloud(self, observed_cloud, aligned_cad_cloud):
        augmented = o3d.geometry.PointCloud(observed_cloud)
        augmented += o3d.geometry.PointCloud(aligned_cad_cloud)
        return augmented

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

    def default_yaw_candidates(self):
        step = float(self.config.yaw_step_deg)
        return np.arange(0.0, 360.0, step).tolist()

    @staticmethod
    def default_cad_up_axes():
        return ["Z", "-Z", "X", "-X", "Y", "-Y"]

    @staticmethod
    def delta_transform(translation, rotation):
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(translation, dtype=float)
        return transform

    @staticmethod
    def transform_points(points, rotation, translation):
        return points @ rotation.T + np.asarray(translation, dtype=float)

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
                [z * x - y * s - z * x * c, z * y * one_c + x * s, c + z * z * one_c],
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


if __name__ == "__main__":
    CADPointCloudRegistration().run(
        cad_path="path/to/cad_model.stl",
        observed_cloud_path="output/point_cloud_localization/object_cluster_1.ply",
    )
