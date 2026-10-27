"""Public-camera-only post-run views for the separate board-feedback follow-up."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .. import visualization as shared


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/observed_held_assembly_20261025/board_feedback_followup_20261025/runs/20261025T202719Z"
POSE_CAVEAT = ("Repeated visible geometry can leave the full CAD angular phase unverified. "
               "Exact CAD symmetry, tooth alignment and thread phase are not established by these views.")


def main():
    bindings = {name: getattr(shared, name) for name in ("RUN", "OUTPUT", "VIDEO_OUTPUT")}
    try:
        shared.RUN = RUN
        shared.OUTPUT = RUN / "gallery/visualizations"
        shared.VIDEO_OUTPUT = RUN / "gallery/videos"
        shared.main()
        path = shared.OUTPUT / "manifest.json"
        manifest = json.loads(path.read_text())
        for part in manifest["parts"]:
            for stage in part["stages"]:
                picture_path = RUN / stage["overlay"]
                original = Image.open(picture_path).convert("RGB")
                picture = Image.new("RGB", (original.width, original.height + 64), "#f3f5f7")
                picture.paste(original, (0, 0))
                draw = ImageDraw.Draw(picture)
                draw.text((16, original.height + 7), "Repeated visible geometry can leave the full CAD orientation unverified.",
                          fill="#283c51", font=shared.font(21))
                draw.text((16, original.height + 36), "Axes show the saved estimate. Gear-tooth and nut-thread phase remain unverified.",
                          fill="#283c51", font=shared.font(20))
                picture.save(picture_path)
                stage["overlay_sha256"] = shared.digest(picture_path)
        manifest.update(wrapper_sha256=shared.digest(Path(__file__)),
                        shared_helper_sha256=shared.digest(Path(shared.__file__)),
                        pose_caveat=POSE_CAVEAT,
                        original_visualization_source_edited=False,
                        original_or_followup_runtime_sources_edited=False)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"followup_manifest": str(path), "pose_caveat": POSE_CAVEAT}), flush=True)
    finally:
        for name, value in bindings.items():
            setattr(shared, name, value)


if __name__ == "__main__":
    main()
