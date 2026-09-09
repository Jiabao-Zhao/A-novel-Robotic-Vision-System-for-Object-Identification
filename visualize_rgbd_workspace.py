"""Render the recovered RGB-D workspace and export its measured colored points."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "outputs/recovered/original_four_object_setup_57e59a0"
OUTPUT_DIR = CAPTURE_DIR / "rgbd_visualization"
# White work surface boundary, selected from this capture's 1280 x 720 RGB image.
WORKSPACE_POLYGON_PX = np.array([[245, 0], [1119, 0], [1076, 642], [195, 615]])
IMAGE_SIZE_PX = (2400, 1600)
VIEW_AZIMUTH_DEG = -65.0
VIEW_ELEVATION_DEG = 38.0


def reconstruct_workspace():
    rgb_path = CAPTURE_DIR / "raw/RGB.png"
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    with np.load(CAPTURE_DIR / "raw/depth_data.npz") as data:
        depth_m = data["depth_data"].astype(float) * float(data["depth_scale_m"])
        width, height, fx, fy, cx, cy = data["camera_intrinsics"]
    if rgb.shape[:2] != depth_m.shape or depth_m.shape != (int(height), int(width)):
        raise ValueError("RGB, depth, and camera intrinsics have different dimensions.")
    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [WORKSPACE_POLYGON_PX.astype(np.int32)], 1)
    v, u = np.nonzero((mask > 0) & (depth_m > 0) & np.isfinite(depth_m))
    z_m = depth_m[v, u]
    points_camera_m = np.column_stack(((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m))
    if len(points_camera_m) == 0:
        raise ValueError("The selected workspace has no valid depth samples.")

    localization = json.loads(
        (CAPTURE_DIR / "point_cloud_localization/point_cloud_localization.json").read_text()
    )
    plane = np.asarray(localization["plane_model"], dtype=float)
    plane /= np.linalg.norm(plane[:3])
    # Up faces the camera. Table X follows projected camera X; Y completes the frame.
    up = -plane[:3] if plane[3] < 0 else plane[:3]
    x_axis = np.array([1.0, 0.0, 0.0]) - up[0] * up
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack((x_axis, np.cross(up, x_axis), up))
    center_px = WORKSPACE_POLYGON_PX.mean(axis=0)
    ray = np.array([(center_px[0] - cx) / fx, (center_px[1] - cy) / fy, 1.0])
    origin_camera_m = ray * (-plane[3] / np.dot(plane[:3], ray))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin_camera_m
    points_table_m = (points_camera_m - origin_camera_m) @ rotation.T
    return points_table_m, rgb[v, u], transform


def save_ply(path, points_m, colors, description="RGB-D observations; table frame; XYZ in meters"):
    vertices = np.empty(
        len(points_m),
        dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points_m[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment {description}\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def render_point_cloud(points_m, colors, view, image_size_px, point_radius_px, frame_points_m=None):
    """Render point splats and return the matching affine 3D-to-image projection."""
    projected = points_m @ view.T
    bounds = projected
    if frame_points_m is not None:
        bounds = np.vstack([bounds, frame_points_m @ view.T])
    lo, hi = bounds[:, :2].min(axis=0), bounds[:, :2].max(axis=0)
    width, height = np.array(image_size_px) * 2
    scale = min(width * 0.88 / (hi[0] - lo[0]), height * 0.85 / (hi[1] - lo[1]))
    center = (lo + hi) / 2

    def project(points):
        p = np.asarray(points) @ view.T
        return np.column_stack([
            width / 2 + (p[:, 0] - center[0]) * scale,
            height / 2 - (p[:, 1] - center[1]) * scale,
        ])

    pixels = np.rint(project(points_m)).astype(int)
    point_depth = projected[:, 2].astype(np.float32)
    z_buffer = np.full(width * height, -np.inf, dtype=np.float32)
    image = np.full((width * height, 3), 255, dtype=np.uint8)
    # A small disk per measured sample; no meshing, surface completion, or recoloring.
    radius = int(np.ceil(point_radius_px * 2))
    offsets = [(dx, dy) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if dx*dx + dy*dy <= (point_radius_px * 2)**2]
    for dx, dy in offsets:
        x, y = pixels[:, 0] + dx, pixels[:, 1] + dy
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = y[valid] * width + x[valid]
        np.maximum.at(z_buffer, indices, point_depth[valid])
    for dx, dy in offsets:
        x, y = pixels[:, 0] + dx, pixels[:, 1] + dy
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        selected = np.flatnonzero(valid)
        indices = y[valid] * width + x[valid]
        visible = point_depth[valid] == z_buffer[indices]
        image[indices[visible]] = colors[selected[visible]]
    image = cv2.resize(image.reshape(height, width, 3), image_size_px, interpolation=cv2.INTER_AREA)
    projection = np.column_stack([
        view[:2] * np.array([[scale / 2], [-scale / 2]]),
        [width / 4 - center[0] * scale / 2, height / 4 + center[1] * scale / 2],
    ])
    return image, projection


def workspace_view():
    azimuth, elevation = np.deg2rad([VIEW_AZIMUTH_DEG, VIEW_ELEVATION_DEG])
    toward_eye = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    right = np.cross([0.0, 0.0, 1.0], toward_eye)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(toward_eye, right), toward_eye])


def render_workspace(points_m, colors):
    view = workspace_view()
    image, _ = render_point_cloud(points_m, colors, view, IMAGE_SIZE_PX, np.sqrt(5))

    # A separate inset uses the same view rotation; it is not an object-pose estimate.
    origin = np.array([185.0, IMAGE_SIZE_PX[1] - 150.0])
    directions = np.eye(3) @ view.T
    for axis, direction, color in zip("XYZ", directions, [(205, 45, 45), (25, 145, 65), (40, 85, 215)]):
        end = origin + np.array([direction[0], -direction[1]]) * 115
        cv2.arrowedLine(image, tuple(origin.astype(int)), tuple(end.astype(int)), color, 4, cv2.LINE_AA, tipLength=0.14)
        label = end + (end - origin) / np.linalg.norm(end - origin) * 17
        cv2.putText(image, axis, tuple(label.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(image, "Table frame", (95, IMAGE_SIZE_PX[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 80), 2, cv2.LINE_AA)
    return image


def main():
    points_m, colors, transform = reconstruct_workspace()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ply_path = OUTPUT_DIR / "workspace_rgbd_table_frame.ply"
    image_path = OUTPUT_DIR / "workspace_rgbd_oblique.png"
    save_ply(ply_path, points_m, colors)
    image = render_workspace(points_m, colors)
    if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {image_path}")
    metadata = {
        "source_capture": str(CAPTURE_DIR),
        "source_git_commit": "57e59a02f829f17efcf96f1845b85e12922a7f43",
        "point_count": len(points_m),
        "cloud_frame": "table",
        "cloud_units": "meters",
        "workspace_polygon_px": WORKSPACE_POLYGON_PX.tolist(),
        "T_table_from_camera_m": transform.tolist(),
        "T_camera_from_table_m": np.linalg.inv(transform).tolist(),
        "table_frame_definition": "Origin: ROI-center camera ray intersected with saved table plane. X: projected camera X. Z: plane normal toward camera. Y: Z cross X.",
        "view_azimuth_deg": VIEW_AZIMUTH_DEG,
        "view_elevation_deg": VIEW_ELEVATION_DEG,
        "image_size_px": list(IMAGE_SIZE_PX),
        "projection": "orthographic",
        "color_processing": "original RGB; point coverage antialiasing only",
        "depth_processing": "source averages 15 filtered depth frames",
        "geometry_processing": "image-space crop and rigid camera-to-table transform only; no downsampling, outlier removal, meshing, or completion",
    }
    (OUTPUT_DIR / "workspace_rgbd_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(points_m):,} colored points: {ply_path}")
    print(f"Saved RGB-D view: {image_path}")


if __name__ == "__main__":
    main()
