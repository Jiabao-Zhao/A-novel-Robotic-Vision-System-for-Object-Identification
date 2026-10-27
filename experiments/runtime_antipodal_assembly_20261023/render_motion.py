"""Render the saved contact simulation; never synthesize or repair motion."""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import imageio_ffmpeg
import mujoco
import numpy as np


def render_part(run_path, name, stride=2):
    return render_folder(Path(run_path) / name, name, stride)


def render_folder(folder, name=None, stride=2):
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    name = result["part"] if name is None else name
    trajectory = np.load(folder / "trajectory.npz")
    if len(trajectory["qpos"]) <= 1:
        return None
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    body = model.body("nist_part_" + name).id
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = .26, 130, -25
    video_path = folder / "recorded_motion.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "1280x600", "-r", "25", "-i", "-", "-an",
               "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart",
               str(video_path)]
    indices = list(range(0, len(trajectory["qpos"]), stride))
    if indices[-1] != len(trajectory["qpos"]) - 1:
        indices.append(len(trajectory["qpos"]) - 1)
    with (folder / "video_encoder.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for index in indices:
                data.qpos[:] = trajectory["qpos"][index]
                data.qvel[:] = trajectory["qvel"][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="agentview", scene_option=option)
                wide = renderer.render().copy()
                camera.lookat[:] = data.xpos[body] + [0, 0, .015]
                renderer.update_scene(data, camera=camera, scene_option=option)
                close = renderer.render().copy()
                frame = np.full((600, 1280, 3), 247, np.uint8)
                frame[80:560, :640], frame[80:560, 640:] = wide, close
                phase = str(trajectory["phase"][index])
                cv2.putText(frame, f"{name} | {phase} | t={trajectory['time_s'][index]:.2f} s",
                            (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (25, 25, 25), 2, cv2.LINE_AA)
                cv2.putText(frame, "Training-free runtime grasp | supplied initial CAD pose | actual finger contacts | no attachment",
                            (18, 60), cv2.FONT_HERSHEY_SIMPLEX, .52, (80, 55, 30), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Recorded states at {stride}x playback | Outcome: {result['status']}",
                            (18, 585), cv2.FONT_HERSHEY_SIMPLEX, .57, (40, 40, 40), 1, cv2.LINE_AA)
                process.stdin.write(frame.tobytes())
            for _ in range(25):
                process.stdin.write(frame.tobytes())
            cv2.imwrite(str(folder / "motion_outcome.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"Video encoder failed: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    expected = len(indices) + 25
    if decoded != expected:
        raise RuntimeError(f"Video decode count mismatch: {decoded} vs {expected}")
    metadata = {"source": "saved actual MuJoCo qpos/qvel, no interpolated poses",
                "source_recording_dt_s": .04, "frame_stride": stride, "playback_speed": stride,
                "expected_decoded_frames": expected, "decoded_frames": decoded,
                "result_status": result["status"]}
    (folder / "video_verification.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return video_path


if __name__ == "__main__":
    for part in sys.argv[2:]:
        print(render_part(Path(sys.argv[1]), part), flush=True)
