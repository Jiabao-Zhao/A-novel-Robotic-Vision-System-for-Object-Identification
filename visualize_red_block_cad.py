"""Render the recovered red-block CAD sample with a centered CAD coordinate frame."""

import json

import cv2
import numpy as np

from visualize_rgbd_workspace import (
    CAPTURE_DIR, reconstruct_workspace, render_point_cloud, save_ply, workspace_view,
)


REGISTRATION_DIR = CAPTURE_DIR / "registered_point_cloud/object_002"
OUTPUT_DIR = CAPTURE_DIR / "red_block_cad_visualization"
IMAGE_SIZE_PX = (1800, 1500)
AXIS_LENGTH_M = 0.037
AXIS_COLORS_RGB = [(225, 35, 35), (15, 200, 45), (35, 70, 240)]


def load_cad_sample():
    path = REGISTRATION_DIR / "cad_sampled_cloud.ply"
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii").strip()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            header.append(line)
            if line == "end_header":
                break
        properties = [line for line in header if line.startswith("property ")]
        expected = [f"property double {name}" for name in ("x", "y", "z", "nx", "ny", "nz")]
        expected += [f"property uchar {name}" for name in ("red", "green", "blue")]
        if "format binary_little_endian 1.0" not in header or properties != expected:
            raise ValueError("Expected the original Open3D CAD sample with XYZ, normals, and RGB.")
        vertices = np.fromfile(stream, dtype=[
            ("point", "<f8", (3,)), ("normal", "<f8", (3,)), ("color", "u1", (3,)),
        ])
    count = int(next(line.split()[-1] for line in header if line.startswith("element vertex")))
    if len(vertices) != count or not count:
        raise ValueError("CAD sample point count does not match its header.")
    return vertices["point"], vertices["normal"]


def draw_cad_axes(image, axis_pixels):
    start = tuple(np.rint(axis_pixels[0]).astype(int))
    for label, endpoint, color in zip("XYZ", axis_pixels[1:], AXIS_COLORS_RGB):
        end = tuple(np.rint(endpoint).astype(int))
        offset = endpoint - axis_pixels[0]
        arrow_tip = max(0.055, 20 / np.linalg.norm(offset))
        cv2.arrowedLine(image, start, end, (255, 255, 255), 13, cv2.LINE_AA, tipLength=arrow_tip)
        cv2.arrowedLine(image, start, end, color, 9, cv2.LINE_AA, tipLength=arrow_tip)
        label_position = endpoint + offset / np.linalg.norm(offset) * 28
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        label_position += [-text_size[0] / 2, text_size[1] / 2]
        cv2.putText(image, label, tuple(label_position.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(image, label, tuple(label_position.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


def render_aligned_workspace(native_points_m, normals, registration):
    scene_points_m, scene_colors, T_table_from_camera = reconstruct_workspace()
    T_table_from_cad = T_table_from_camera @ np.asarray(registration["T_observed_from_cad"])
    rotation, translation = T_table_from_cad[:3, :3], T_table_from_cad[:3, 3]
    cad_points_m = native_points_m @ rotation.T + translation
    center_native_m = np.asarray(registration["cad_center_m"])
    axes_native_m = center_native_m + np.vstack([np.zeros(3), np.eye(3) * AXIS_LENGTH_M])
    axes_table_m = axes_native_m @ rotation.T + translation

    light = np.array([0.25, -0.4, 1.0])
    light /= np.linalg.norm(light)
    brightness = 0.4 + 0.6 * np.maximum((normals @ rotation.T) @ light, 0.0)
    cad_colors = np.rint(np.array([225, 30, 35]) * brightness[:, None]).astype(np.uint8)
    image_size_px = (2400, 1600)
    image, projection = render_point_cloud(
        np.vstack([scene_points_m, cad_points_m]), np.vstack([scene_colors, cad_colors]),
        workspace_view(), image_size_px, np.sqrt(5), axes_table_m,
    )
    axis_pixels = np.column_stack([axes_table_m, np.ones(4)]) @ projection.T
    draw_cad_axes(image, axis_pixels)

    # Call out the CAD overlay without covering the measured objects.
    label_position = np.rint(axis_pixels[0] + [-420, -220]).astype(int)
    cv2.putText(image, "Aligned CAD", tuple(label_position), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(image, "Aligned CAD", tuple(label_position), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (35, 35, 35), 2, cv2.LINE_AA)
    leader_start = tuple(label_position + [205, 12])
    leader_end = tuple(np.rint(axis_pixels[0] + [-55, -55]).astype(int))
    cv2.line(image, leader_start, leader_end, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.line(image, leader_start, leader_end, (65, 65, 65), 2, cv2.LINE_AA)
    image_path = OUTPUT_DIR / "red_block_aligned_workspace_axes.png"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(image_path)
    metadata = {
        "source_registration": str(REGISTRATION_DIR / "cad_registration_result.json"),
        "alignment": "original saved CAD alignment, applied without refitting",
        "workspace_frame": "table",
        "axis_frame": "centered CAD axes transformed into the table frame",
        "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
        "axis_length_mm": AXIS_LENGTH_M * 1000,
        "axis_origin_table_mm": (axes_table_m[0] * 1000).tolist(),
        "axis_endpoints_table_mm": {label: (point * 1000).tolist() for label, point in zip("XYZ", axes_table_m[1:])},
        "T_table_from_native_cad_m": T_table_from_cad.tolist(),
        "scene_point_count": len(scene_points_m),
        "cad_point_count": len(cad_points_m),
        "image_size_px": list(image_size_px),
        "pixel_projection_from_table_m": projection.tolist(),
        "rendering": "Measured RGB-D cloud plus sampled red CAD geometry with a shared depth buffer. Coordinate arrows are overlaid for visibility.",
    }
    (OUTPUT_DIR / "red_block_aligned_workspace_axes.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved aligned CAD workspace view: {image_path}")


def main():
    native_points_m, normals = load_cad_sample()
    registration = json.loads((REGISTRATION_DIR / "cad_registration_result.json").read_text())
    center_m = np.asarray(registration["cad_center_m"])
    points_m = native_points_m - center_m
    red = np.tile(np.array([175, 22, 27], dtype=np.uint8), (len(points_m), 1))

    # Y-up view for the template figure. This changes the viewpoint, not CAD axes.
    toward_eye = np.array([1.5, 1.05, 1.8])
    toward_eye /= np.linalg.norm(toward_eye)
    right = np.cross([0.0, 1.0, 0.0], toward_eye)
    right /= np.linalg.norm(right)
    view = np.stack([right, np.cross(toward_eye, right), toward_eye])
    axes_m = np.vstack([np.zeros(3), np.eye(3) * AXIS_LENGTH_M])
    # Shade point markers by CAD surface normal to make the three faces legible.
    light = np.array([0.1, 0.9, 0.45])
    light /= np.linalg.norm(light)
    brightness = 0.2 + 0.8 * np.maximum(normals @ light, 0.0)
    render_colors = np.rint(red * brightness[:, None]).astype(np.uint8)
    visible = normals @ toward_eye > 0.0
    image, projection = render_point_cloud(
        points_m[visible], render_colors[visible], view, IMAGE_SIZE_PX, 3.0, axes_m,
    )
    axis_pixels = np.column_stack([axes_m, np.ones(4)]) @ projection.T
    draw_cad_axes(image, axis_pixels)
    cv2.putText(image, "Template", (110, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (25, 25, 25), 3, cv2.LINE_AA)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "red_block_cad_axes.png"
    if not cv2.imwrite(str(png_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(png_path)
    save_ply(
        OUTPUT_DIR / "red_block_cad_centered.ply", points_m, red,
        "Sampled CAD geometry; centered CAD frame; XYZ in meters; display color red",
    )
    T_native_from_centered = np.eye(4)
    T_native_from_centered[:3, 3] = center_m
    metadata = {
        "source_cad": "CAD/models/square_block.stl",
        "source_sample": str(REGISTRATION_DIR / "cad_sampled_cloud.ply"),
        "source_git_commit": "57e59a02f829f17efcf96f1845b85e12922a7f43",
        "object_id": "object_002",
        "target_description": "red block",
        "point_count": len(points_m),
        "dimensions_mm": (np.ptp(points_m, axis=0) * 1000).tolist(),
        "frame": "centered CAD frame; axes parallel to native CAD axes",
        "units": "meters",
        "origin_in_native_cad_mm": (center_m * 1000).tolist(),
        "axis_length_mm": AXIS_LENGTH_M * 1000,
        "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
        "axis_origin_m": axes_m[0].tolist(),
        "axis_endpoints_m": {label: point.tolist() for label, point in zip("XYZ", axes_m[1:])},
        "T_native_cad_from_centered_cad_m": T_native_from_centered.tolist(),
        "T_camera_from_centered_cad_m": (np.asarray(registration["T_observed_from_cad"]) @ T_native_from_centered).tolist(),
        "image_size_px": list(IMAGE_SIZE_PX),
        "view_rotation": view.tolist(),
        "pixel_projection_from_centered_cad_m": projection.tolist(),
        "rendering": "View-facing CAD samples rendered as red points with normal-based shading; rear-facing samples hidden for legibility. PLY retains all original samples. Coordinate arrows overlaid at the CAD center; axes are annotations, not PLY geometry.",
        "orientation_note": "Axes define a chosen CAD coordinate frame. Cube symmetry does not provide a unique observable object orientation; the saved alignment is one symmetry-equivalent solution.",
    }
    (OUTPUT_DIR / "red_block_cad_axes.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(points_m):,} CAD points and centered axes to {OUTPUT_DIR}")
    render_aligned_workspace(native_points_m, normals, registration)


if __name__ == "__main__":
    main()
