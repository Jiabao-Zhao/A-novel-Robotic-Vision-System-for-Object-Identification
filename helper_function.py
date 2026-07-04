import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


class PointCloudPlot:
    def __init__(self):
        self.result_dir = Path("Output/pointcloud_pose")
        self.visualization_dir = Path("Output/visualization")
        self.axis_length_m = 0.08

    def orient_normal_to_points(self, plane_model, points) -> np.ndarray:
        normal = np.asarray(plane_model[:3], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            raise ValueError("Table plane normal has zero length.")

        z_axis = normal / normal_norm
        if len(points) == 0:
            return z_axis

        signed_distance = float(np.mean(points, axis=0) @ z_axis + plane_model[3] / normal_norm)
        return -z_axis if signed_distance < 0 else z_axis

    def build_visualization_metadata(self, plane_model, table_cloud, clusters, source_frame) -> dict:
        if clusters:
            object_points = np.vstack([np.asarray(cluster.points) for cluster in clusters])
        else:
            object_points = np.empty((0, 3))

        z_axis = self.orient_normal_to_points(plane_model, object_points)
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

    def transform_points(self, points, metadata) -> np.ndarray:
        origin = np.asarray(metadata["origin_m"], dtype=float)
        basis = np.asarray(metadata["basis_columns_source"], dtype=float)
        return (points - origin) @ basis

    def transform_rotation(self, rotation_matrix, metadata) -> np.ndarray:
        basis = np.asarray(metadata["basis_columns_source"], dtype=float)
        return basis.T @ np.asarray(rotation_matrix, dtype=float)

    @staticmethod
    def pose_field(result, name):
        if isinstance(result, dict):
            if name in result:
                return result[name]
            aliases = {
                "id": "object_id",
                "object_id": "id",
            }
            alias = aliases.get(name)
            if alias and alias in result:
                return result[alias]
            raise KeyError(name)
        return getattr(result, name)

    def make_red_pose_axes(self, center, rotation, axis_length_m=None):
        axis_length_m = self.axis_length_m if axis_length_m is None else axis_length_m
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

    def show_visualization(self, results, clouds, metadata, axis_length_m=None):
        axis_length_m = self.axis_length_m if axis_length_m is None else axis_length_m
        geometries = []

        table_cloud = o3d.geometry.PointCloud(clouds["table"])
        table_cloud.points = o3d.utility.Vector3dVector(
            self.transform_points(np.asarray(table_cloud.points), metadata)
        )
        table_cloud.paint_uniform_color([0.55, 0.55, 0.55])
        geometries.append(table_cloud)

        for cluster in clouds["clusters"]:
            object_cloud = o3d.geometry.PointCloud(cluster)
            object_cloud.points = o3d.utility.Vector3dVector(
                self.transform_points(np.asarray(object_cloud.points), metadata)
            )
            object_cloud.paint_uniform_color([0.0, 0.25, 1.0])
            geometries.append(object_cloud)

        for result in results:
            center = np.asarray(self.pose_field(result, "center_mm"), dtype=float) / 1000.0
            center = self.transform_points(center.reshape(1, 3), metadata)[0]
            rotation = self.transform_rotation(self.pose_field(result, "rotation_matrix"), metadata)
            geometries.extend(self.make_red_pose_axes(center, rotation, axis_length_m))

            x_mm, y_mm, z_mm = self.pose_field(result, "center_mm")
            roll_deg, pitch_deg, yaw_deg = self.pose_field(result, "rpy_deg")
            print(
                f"{self.pose_field(result, 'id')}: "
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

    def show_saved_visualization(self):
        pose_path = self.result_dir / "pose_results.json"
        plane_path = self.visualization_dir / "table_plane_cloud.ply"
        metadata_path = self.visualization_dir / "visualization_metadata.json"

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
                for path in sorted(self.result_dir.glob("object_cluster_*.ply"))
            ],
        }
        self.show_visualization(payload["objects"], clouds, metadata)


class RBGAnnotation:
    def __init__(self):
        self.rgb_path = Path("output/raw/RGB.png")
        self.roi_path = Path("output/roi/roi_candidates.json")
        self.output_dir = Path("output/annotation")
        self.output_path = self.output_dir / "RGB_roi_annotation.png"
        self.box_color = np.array([255, 40, 40], dtype=np.uint8)
        self.label_background_color = np.array([255, 255, 255], dtype=np.uint8)
        self.label_text_color = np.array([0, 0, 0], dtype=np.uint8)
        self.box_thickness_px = 3
        self.label_scale = 3

    def load_rgb_image(self):
        image = np.asarray(o3d.io.read_image(str(self.rgb_path)))
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Expected RGB image at {self.rgb_path}")
        return image[:, :, :3].copy()

    def load_roi_candidates(self):
        payload = json.loads(self.roi_path.read_text(encoding="utf-8"))
        return payload.get("objects", [])

    def annotate(self, image, candidates):
        annotated = np.ascontiguousarray(image).copy()
        for candidate in candidates:
            roi = candidate.get("roi", {})
            x = int(roi.get("x", roi.get("x1", 0)))
            y = int(roi.get("y", roi.get("y1", 0)))
            width = int(roi.get("width", int(roi.get("x2", x)) - x))
            height = int(roi.get("height", int(roi.get("y2", y)) - y))
            if width <= 0 or height <= 0:
                continue

            self._draw_box(annotated, x, y, width, height)
            self._draw_label(annotated, str(candidate.get("label", candidate.get("id", ""))), x, y)

        return annotated

    def save_annotation(self, annotated):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image = o3d.geometry.Image(np.ascontiguousarray(annotated.astype(np.uint8)))
        o3d.io.write_image(str(self.output_path), image)
        return self.output_path

    def run(self):
        image = self.load_rgb_image()
        candidates = self.load_roi_candidates()
        annotated = self.annotate(image, candidates)
        return self.save_annotation(annotated)

    def _draw_box(self, image, x, y, width, height):
        image_height, image_width = image.shape[:2]
        x0 = int(np.clip(x, 0, image_width - 1))
        y0 = int(np.clip(y, 0, image_height - 1))
        x1 = int(np.clip(x + width, 0, image_width))
        y1 = int(np.clip(y + height, 0, image_height))
        thickness = max(1, int(self.box_thickness_px))

        image[y0:min(y0 + thickness, y1), x0:x1] = self.box_color
        image[max(y1 - thickness, y0):y1, x0:x1] = self.box_color
        image[y0:y1, x0:min(x0 + thickness, x1)] = self.box_color
        image[y0:y1, max(x1 - thickness, x0):x1] = self.box_color

    def _draw_label(self, image, text, x, y):
        if not text:
            return

        scale = max(1, int(self.label_scale))
        digit_width = 3 * scale
        digit_height = 5 * scale
        spacing = scale
        label_width = len(text) * digit_width + max(0, len(text) - 1) * spacing + 2 * scale
        label_height = digit_height + 2 * scale

        image_height, image_width = image.shape[:2]
        x0 = int(np.clip(x, 0, max(0, image_width - label_width)))
        y0 = int(np.clip(y - label_height, 0, max(0, image_height - label_height)))
        image[y0:y0 + label_height, x0:x0 + label_width] = self.label_background_color

        cursor_x = x0 + scale
        cursor_y = y0 + scale
        for char in text:
            self._draw_digit(image, char, cursor_x, cursor_y, scale)
            cursor_x += digit_width + spacing

    def _draw_digit(self, image, char, x, y, scale):
        pattern = self._digit_patterns().get(char)
        if pattern is None:
            return

        for row_index, row in enumerate(pattern):
            for col_index, value in enumerate(row):
                if value != "1":
                    continue
                y0 = y + row_index * scale
                x0 = x + col_index * scale
                image[y0:y0 + scale, x0:x0 + scale] = self.label_text_color

    @staticmethod
    def _digit_patterns():
        return {
            "0": ["111", "101", "101", "101", "111"],
            "1": ["010", "110", "010", "010", "111"],
            "2": ["111", "001", "111", "100", "111"],
            "3": ["111", "001", "111", "001", "111"],
            "4": ["101", "101", "111", "001", "001"],
            "5": ["111", "100", "111", "001", "111"],
            "6": ["111", "100", "111", "101", "111"],
            "7": ["111", "001", "010", "010", "010"],
            "8": ["111", "101", "111", "101", "111"],
            "9": ["111", "101", "111", "001", "111"],
        }


@dataclass
class CADModel:
    cad_id: str
    cad_name: str
    file_path: str
    description: str
    category: str


class CADRetrieval:
    """
    Text-only CAD retrieval:

        m* = arg max over m_k in M [ max over q_i in Q sim(E(q_i), E(t_k)) ]

    M is the CAD model library, m_k is one CAD model record, t_k is the
    combined text metadata for m_k, Q is the expanded query set from the VLM
    classification result, E(.) is an embedding function, sim(.) is cosine
    similarity, and m* is the selected CAD model.
    """

    equation = "m* = arg max_{m_k in M} [ max_{q_i in Q} sim(E(q_i), E(t_k)) ]"

    @staticmethod
    def load_library(library_path):
        records = json.loads(Path(library_path).read_text(encoding="utf-8"))
        return [
            CADModel(
                cad_id=record["cad_id"],
                cad_name=record["cad_name"],
                file_path=record["file_path"],
                description=record["description"],
                category=record["category"],
            )
            for record in records
        ]

    @staticmethod
    def build_model_text(cad_model):
        parts = [
            cad_model.cad_name,
            cad_model.description,
            cad_model.category,
            cad_model.file_path,
        ]
        return " ".join(str(part).strip() for part in parts if str(part).strip())

    @staticmethod
    def cosine_similarity(query_embedding, model_embedding):
        query = np.asarray(query_embedding, dtype=float).reshape(-1)
        model = np.asarray(model_embedding, dtype=float).reshape(-1)
        query_norm = np.linalg.norm(query)
        model_norm = np.linalg.norm(model)
        if query_norm == 0 or model_norm == 0:
            return 0.0
        return float(np.dot(query, model) / (query_norm * model_norm))

    def retrieve(
        self,
        classification_text,
        cad_models,
        query_alternatives=None,
        embedding_function=None,
    ):
        queries = [classification_text] + list(query_alternatives or [])
        query_embeddings = [
            (query, embedding_function(query))
            for query in queries
        ]

        best_model = None
        best_score = -np.inf
        best_query = ""

        for cad_model in cad_models:
            model_text = self.build_model_text(cad_model)
            model_embedding = embedding_function(model_text)
            for query, query_embedding in query_embeddings:
                score = self.cosine_similarity(query_embedding, model_embedding)
                if score > best_score:
                    best_model = cad_model
                    best_score = score
                    best_query = query

        result = {
            "selected_cad_id": best_model.cad_id,
            "selected_cad_name": best_model.cad_name,
            "selected_file_path": best_model.file_path,
            "score": float(best_score),
            "matched_query": best_query,
            "equation": self.equation,
        }
        return best_model, result


if __name__ == "__main__":
    PointCloudPlot().show_saved_visualization()
