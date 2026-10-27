"""Post-run camera/estimated-CAD pictures and video from recorded camera frames.

Only public controller estimates, raw sensor files, native CAD and original RGB
frame recordings are read. No simulator model, state, labels or replay is used.
"""
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from scripts.surface_verification import CadSurface


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/observed_held_assembly_20261025/runs/20261025T201252Z"
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
OUTPUT = RUN / "gallery/visualizations"
VIDEO_OUTPUT = RUN / "gallery/videos"
CAMERAS = ("overhead", "inspection_front", "inspection_back", "board_inspection")
PARTS = ("Gear_Large", "Gear_Medium", "M16_Hex_Nut")
PANEL_SIZE = 560
VIDEO_FPS = 2.5


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def permitted(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if directory not in path.parents and path != directory:
        raise ValueError("Visualization input is outside its permitted public directory")
    return path


def read_json(path, tracked):
    tracked.add(path)
    return json.loads(path.read_text())


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def stage_title(label):
    names = {"initial": "Before pickup", "after_pickup": "After pickup and hold",
             "after_turn": "After turning the hand", "above_board": "Above the board",
             "after_release": "After opening the fingers", "after_retreat": "After the hand moved away"}
    if label.startswith("board_step_"):
        return f"After board-lowering step {int(label.rsplit('_', 1)[-1]) + 1}"
    return names.get(label, label.replace("_", " "))


def project(points_world, calibration):
    transform = np.asarray(calibration["world_T_camera"])
    intrinsics = np.asarray(calibration["intrinsics"])
    points = (np.atleast_2d(points_world) - transform[:3, 3]) @ transform[:3, :3]
    pixels = points @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:], points[:, 2]


def estimated_crop(vertices, pose, calibration, image_size):
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    pixels, depth = project(world, calibration)
    width, height = image_size
    if np.any(depth <= 0) or not np.isfinite(pixels).all():
        return (0, 0, width, height), False
    lower, upper = pixels.min(0), pixels.max(0)
    if upper[0] < 0 or upper[1] < 0 or lower[0] >= width or lower[1] >= height:
        return (0, 0, width, height), False
    center = (lower + upper) / 2
    side = min(max(160, int(np.ceil(2.4 * max(upper - lower)))), width, height)
    x = int(np.clip(round(center[0] - side / 2), 0, width - side))
    y = int(np.clip(round(center[1] - side / 2), 0, height - side))
    return (x, y, x + side, y + side), True


def annotated_camera(surface, pose, calibration, rgb, observed_depth):
    image = Image.fromarray(rgb)
    result = {"estimated_CAD_in_frame": False,
              "crop_source": "Full camera image because no fitted CAD pose is available"}
    if pose is None:
        return image, (0, 0, image.width, image.height), result
    bounds, visible = estimated_crop(surface.vertices, pose, calibration, image.size)
    result.update(estimated_CAD_in_frame=visible,
                  crop_source="Projected bounds of native CAD at the saved estimated pose" if visible
                              else "Full camera image because projected estimated CAD is outside this camera")
    if not visible:
        return image, bounds, result
    camera_T_cad = np.linalg.inv(np.asarray(calibration["world_T_camera"])) @ pose
    rendered = surface.render(camera_T_cad, np.asarray(calibration["intrinsics"]), observed_depth.shape, pixel_offset=0.)
    overlay = rgb.copy()
    if rendered is not None:
        x1, y1, x2, y2 = rendered["roi"]
        expected = rendered["depth"]
        mask = np.isfinite(expected) & (expected > 0)
        edge = mask & ~binary_erosion(mask)
        foreground = np.isfinite(observed_depth[y1:y2, x1:x2]) & (observed_depth[y1:y2, x1:x2] < expected - .002)
        cyan = binary_dilation(edge & ~foreground)
        orange = binary_dilation(edge & foreground)
        patch = overlay[y1:y2, x1:x2]
        patch[cyan] = [0, 225, 245]
        patch[orange] = [255, 150, 45]
        result.update(projected_CAD_pixels=int(mask.sum()),
                      projected_CAD_pixels_hidden_by_observed_foreground=int((mask & foreground).sum()))
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    axis_length = max(.008, min(.022, .35 * np.ptp(surface.vertices, axis=0).max()))
    axes = np.vstack((pose[:3, 3], pose[:3, 3] + pose[:3, :3].T * axis_length))
    pixels, depths = project(axes, calibration)
    if np.all(depths > 0):
        for index, (name, color) in enumerate((("X", "#ff4444"), ("Y", "#28d65c"), ("Z", "#438aff")), 1):
            start, end = tuple(pixels[0]), tuple(pixels[index])
            draw.line((start, end), fill=color, width=2)
            draw.text(end, name, fill=color, font=font(15))
    return image, bounds, result


def stage_picture(part, entry, surface, tracked):
    label = entry["label"]
    source_folder = permitted(entry["observation_folder"], RUN / "observations")
    estimate_path = permitted(entry["estimate_path"], RUN / "controller")
    estimate = read_json(estimate_path, tracked)
    pose = np.asarray(estimate["world_T_cad"], float) if estimate["status"] == "available" else None
    width, top, row_height = PANEL_SIZE * 2, 118, PANEL_SIZE + 66
    sheet = Image.new("RGB", (width, top + row_height * 2), "#f3f5f7")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), f"{part.replace('_', ' ')} · {stage_title(label)}", fill="#17293f", font=font(27))
    draw.text((16, 49), f"Saved camera-based CAD fit: {estimate['status'].replace('_', ' ')}", fill="#283c51", font=font(21))
    draw.text((16, 80), "Cyan: estimated CAD outline. Orange: outline hidden by observed foreground. XYZ: estimated CAD axes.", fill="#283c51", font=font(17))
    views = []
    for index, name in enumerate(CAMERAS):
        rgb_path, depth_path = source_folder / f"{name}_rgb.png", source_folder / f"{name}_depth_m.npy"
        calibration_path = source_folder / f"{name}_calibration.json"
        tracked.update((rgb_path, depth_path))
        calibration = read_json(calibration_path, tracked)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path, allow_pickle=False)
        image, bounds, details = annotated_camera(surface, pose, calibration, rgb, depth)
        x, y = (index % 2) * PANEL_SIZE, top + (index // 2) * row_height
        crop = image.crop(bounds).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
        sheet.paste(crop, (x, y + 52))
        draw.text((x + 12, y + 5), name.replace("_", " "), fill="#17293f", font=font(23))
        note = "Crop from estimated CAD projection" if details["estimated_CAD_in_frame"] else "Full raw camera view; no in-frame CAD overlay"
        draw.text((x + 12, y + 33), note, fill="#3e5267", font=font(15))
        views.append({"camera": name, "crop_xyxy": list(bounds), **details,
                      "raw_rgb": str(rgb_path.relative_to(RUN)),
                      "raw_depth": str(depth_path.relative_to(RUN)),
                      "calibration": str(calibration_path.relative_to(RUN))})
    folder = OUTPUT / part
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}_estimated_pose.png"
    sheet.save(path)
    return {"label": label, "title": stage_title(label), "fit_status": estimate["status"],
            "overlay": str(path.relative_to(RUN)), "overlay_sha256": digest(path),
            "estimate": str(estimate_path.relative_to(RUN)), "views": views,
            "current_pose_source": "Saved camera-based CAD estimate", "goal_pose_overlay_used": False}


def encode_recorded_video(part, tracked):
    frames = sorted((RUN / "recorded_camera" / part).glob("*.png"))
    if not frames:
        return {"available": False, "reason": "No camera frames were recorded during commands"}
    encoder = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not encoder.is_file():
        raise RuntimeError("The existing bundled video encoder is unavailable")
    VIDEO_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = VIDEO_OUTPUT / f"{part}.mp4"
    size = Image.open(frames[0]).size
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=VIDEO_FPS, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7, ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart", "-threads", "2"])
    writer.send(None)
    try:
        for frame in frames:
            permitted(frame, RUN / "recorded_camera")
            image = Image.open(frame).convert("RGB")
            if image.size != size:
                raise ValueError("Recorded camera frame size changed")
            tracked.add(frame)
            writer.send(np.asarray(image))
    finally:
        writer.close()
    import cv2
    capture = cv2.VideoCapture(str(path))
    decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    first_ok, _ = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, decoded_count - 1)
    last_ok, _ = capture.read()
    capture.release()
    if decoded_count != len(frames) or abs(decoded_fps - VIDEO_FPS) > 1e-6 or not first_ok or not last_ok:
        raise AssertionError("Encoded video did not preserve all recorded camera frames")
    return {"available": True, "path": str(path.relative_to(RUN)), "sha256": digest(path),
            "frame_count": len(frames), "fps": VIDEO_FPS, "duration_seconds": len(frames) / VIDEO_FPS,
            "first_frame": str(frames[0].relative_to(RUN)), "last_frame": str(frames[-1].relative_to(RUN)),
            "source": "Original camera PNGs saved during actual commands; no state reconstruction",
            "state_replay_used": False, "interpolated_frames": 0,
            "decoded_first_and_last_frames_checked": True}


def main():
    tracked = set()
    results_path = permitted(RUN / "results.json", RUN)
    results = read_json(results_path, tracked)
    rpc_path = permitted(RUN / "client_rpc_log.json", RUN)
    rpc = read_json(rpc_path, tracked)
    if {row["part"] for row in results} != set(PARTS) or not rpc or rpc[-1]["action"] != "shutdown":
        raise RuntimeError("Post-run pictures/video require the completed three-part trial and server shutdown")
    controller_results = {}
    for part in PARTS:
        result_path = permitted(RUN / "controller" / part / "result.json", RUN / "controller")
        result = read_json(result_path, tracked)
        controller_results[part] = result
        tracked.add(MESHES / f"{part}_m.stl")
        tracked.update((RUN / "recorded_camera" / part).glob("*.png"))
        for entry in result["observed_estimates"]:
            tracked.add(permitted(entry["estimate_path"], RUN / "controller"))
            folder = permitted(entry["observation_folder"], RUN / "observations")
            for name in CAMERAS:
                tracked.update((folder / f"{name}_rgb.png", folder / f"{name}_depth_m.npy",
                                folder / f"{name}_calibration.json"))
    before = {path: digest(path) for path in tracked}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for part in PARTS:
        result = controller_results[part]
        mesh_path = MESHES / f"{part}_m.stl"
        tracked.add(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        surface = CadSurface(mesh)
        stages = [stage_picture(part, entry, surface, tracked) for entry in result["observed_estimates"]]
        video = encode_recorded_video(part, tracked)
        parts.append({"part": part, "controller_status": result["status"], "stages": stages, "video": video})
        print(json.dumps({"part": part, "stages": len(stages), "video_frames": video.get("frame_count", 0)}), flush=True)
    if tracked != set(before) or any(digest(path) != value for path, value in before.items()):
        raise AssertionError("A public source file changed while building post-run views")
    hashes = {str(path.relative_to(ROOT)): before[path] for path in sorted(tracked)}
    manifest = {"run": str(RUN), "parts": parts, "source_file_sha256": hashes,
        "helper_sha256": digest(Path(__file__)), "post_run_only": True,
        "visual_geometry_inputs": "Raw RGB-D, native CAD, saved estimated poses and camera calibration",
        "source_files_unchanged": True, "simulator_state_or_object_labels_read": False,
        "simulator_model_or_state_replay_used": False, "old_pickup_closeups_used": False,
        "cropping": "Saved estimated native-CAD projection only; full image when fit unavailable/outside view",
        "outline_colors": {"cyan": "Estimated CAD silhouette", "orange": "Estimated silhouette behind measured raw foreground"},
        "axis_colors": {"red": "CAD X", "green": "CAD Y", "blue": "CAD Z"},
        "pose_caveat": "Symmetric CAD yaw/tooth/thread phase may remain ambiguous; overlays do not establish insertion or engagement",
        "physics_steps": 0, "model_inference_calls": 0, "VLM_calls": 0}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(OUTPUT / "manifest.json"), "source_hashes_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
