"""Show the complete prepared scene before the first trial hid other parts."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np


def main():
    root = Path(__file__).resolve().parents[3]
    source = root / "outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml"
    output = root / "outputs/runtime_antipodal_assembly_20261023/clarification_20261023"
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("objects_centered").id)
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(1280, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(800, model.vis.global_.offheight)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, height=800, width=1280) as renderer:
        renderer.update_scene(data, camera="agentview", scene_option=option)
        rgb = renderer.render().copy()
    canvas = np.full((870, 1280, 3), 248, np.uint8)
    canvas[70:] = rgb
    cv2.putText(canvas, "Scene used: older 14-part objects-centered task board", (20, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .78, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "All parts shown here. The previous trial hid the 13 parts that were not being picked.",
                (20, 56), cv2.FONT_HERSHEY_SIMPLEX, .55, (45, 45, 45), 1, cv2.LINE_AA)
    path = output / "full_scene.png"
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write {path}")
    metadata = {"source_scene": str(source),
                "source_scene_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "saved_state": "objects_centered", "camera": "agentview",
                "all_14_loose_parts_visible": True,
                "note": "This overview shows the prepared input before per-part visibility/contact changes."}
    (output / "full_scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
