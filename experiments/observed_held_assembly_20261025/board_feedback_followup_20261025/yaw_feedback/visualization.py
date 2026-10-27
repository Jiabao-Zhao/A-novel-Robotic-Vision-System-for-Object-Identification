"""Post-run public-camera views for the fresh yaw-feedback trial."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ... import visualization as shared


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "outputs/observed_held_assembly_20261025/board_feedback_followup_20261025/yaw_feedback/runs/20261025T203906Z"
POSE_CAVEAT = ("Visible CAD surface and depth agreement rank the orientation candidates. "
               "The scores are uncalibrated and do not give a measured probability that a pose is correct. "
               "Repeated visible geometry can leave full CAD angular phase unverified; exact CAD symmetry, "
               "gear-tooth alignment and nut-thread phase are not established by these views.")


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
                picture = Image.new("RGB", (original.width, original.height + 92), "#f3f5f7")
                picture.paste(original, (0, 0))
                draw = ImageDraw.Draw(picture)
                draw.text((16, original.height + 7), "The fit score ranks CAD matches; it is not a measured chance that the pose is correct.",
                          fill="#283c51", font=shared.font(20))
                draw.text((16, original.height + 36), "Repeated visible geometry can leave the full CAD orientation unverified.",
                          fill="#283c51", font=shared.font(21))
                draw.text((16, original.height + 65), "Axes show the saved estimate. Gear-tooth and nut-thread phase remain unverified.",
                          fill="#283c51", font=shared.font(20))
                picture.save(picture_path)
                stage["overlay_sha256"] = shared.digest(picture_path)
        manifest.update(wrapper_sha256=shared.digest(Path(__file__)),
                        shared_helper_sha256=shared.digest(Path(shared.__file__)),
                        pose_caveat=POSE_CAVEAT,
                        geometric_rank_is_calibrated_confidence=False,
                        original_visualization_source_edited=False,
                        frozen_runtime_sources_edited=False)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"yaw_feedback_manifest": str(path), "pose_caveat": POSE_CAVEAT}), flush=True)
    finally:
        for name, value in bindings.items():
            setattr(shared, name, value)


if __name__ == "__main__":
    main()
