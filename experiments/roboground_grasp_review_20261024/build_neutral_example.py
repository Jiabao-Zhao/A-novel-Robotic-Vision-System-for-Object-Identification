"""Make neutral, shuffled example VLM inputs from saved initial-grasp poses.

No model is called and no simulation is run. Display images and their IDs carry
no prior movement checks, chosen-grasp marker, ranking score or pickup result.
The separate mapping preserves the exact source poses and image hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.runtime_antipodal_assembly_20261023.grasp_pose_gallery_20261023.render_grasps import (
    SOURCE as SCENE_PATH, add_contact_markers, mark_contacts_on_image,
    setup_model, show_only_part_and_gripper,
)
from experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.gallery import move_gripper_drawing


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "outputs/runtime_antipodal_assembly_20261023/smaller_search"
          / "20261024T030119Z/shortlist_gallery")
RESULT = (SOURCE.parent / "small/Gear_Large/result.json")
OUTPUT = ROOT / "outputs/roboground_grasp_review_20261024/neutral_input_example"
SEED = 20261024
VIEWS = {"camera_a": {"azimuth_deg": -55., "elevation_deg": -30.},
         "camera_b": {"azimuth_deg": 35., "elevation_deg": -60.}}
CAMERA_DISTANCE_M = .34


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size, bold=False):
    weight = "-Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{weight}.ttf", size)


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def render_views(mapping, expected_target):
    """Two cameras have identical parameters for every candidate; no physics."""
    model, data = setup_model()
    mujoco.mj_forward(model, data)
    target_body = model.body("nist_part_Gear_Large").id
    initial_target = np.eye(4)
    initial_target[:3, :3] = data.xmat[target_body].reshape(3, 3)
    initial_target[:3, 3] = data.xpos[target_body]
    np.testing.assert_allclose(initial_target, expected_target, atol=1e-7)
    hand = show_only_part_and_gripper(model, "Gear_Large")
    if hand is not None:
        model.geom_rgba[hand, 3] = .15
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    site = model.site("gripper0_grip_site").id
    lookat = initial_target[:3, 3] + [0., 0., .04]
    (OUTPUT / "views").mkdir()
    drawing_audit = []
    with mujoco.Renderer(model, height=520, width=800) as renderer:
        for candidate in mapping:
            for number, sign in ((1, 1), (2, -1)):
                joint = model.joint(f"gripper0_finger_joint{number}").id
                data.qpos[model.jnt_qposadr[joint]] = sign*candidate["opening_m"]/2
            mujoco.mj_forward(model, data)
            old_pose = np.eye(4)
            old_pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
            old_pose[:3, 3] = data.site_xpos[site]
            pose = np.asarray(candidate["world_T_grip"])
            contacts = np.asarray(candidate["contacts_world_m"])
            candidate["rendered_images"] = {}
            for view, parameters in VIEWS.items():
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = lookat
                camera.distance = CAMERA_DISTANCE_M
                camera.azimuth = parameters["azimuth_deg"]
                camera.elevation = parameters["elevation_deg"]
                renderer.update_scene(data, camera=camera, scene_option=option)
                audit = move_gripper_drawing(renderer.scene, model, old_pose, pose)
                add_contact_markers(renderer.scene, contacts, pose, False)
                picture = Image.fromarray(renderer.render().copy())
                pixels = mark_contacts_on_image(picture, renderer.scene, contacts)
                name = f"views/candidate_{candidate['display_id']}_{view}.png"
                picture.save(OUTPUT / name)
                candidate["rendered_images"][view] = {"file": name, "sha256": digest(OUTPUT/name)}
                drawing_audit.append({"display_id": candidate["display_id"], "view": view,
                    "contact_marker_pixels": pixels, **audit})
    np.testing.assert_array_equal(data.xpos[target_body], initial_target[:3, 3])
    np.testing.assert_array_equal(data.xmat[target_body].reshape(3, 3), initial_target[:3, :3])
    return {"lookat_world_m": lookat.tolist(), "distance_m": CAMERA_DISTANCE_M,
            "views": VIEWS, "parameters_fixed_across_candidates": True,
            "target_pose_unchanged": True, "physics_steps": 0,
            "drawing_audit": drawing_audit}


def build():
    record_path = SOURCE / "Gear_Large/render_record.json"
    record = json.loads(record_path.read_text())
    saved = json.loads(RESULT.read_text())
    assert len(record["grasps"]) == len(saved["shortlisted_grasps"]) == 12
    source_paths = [record_path, RESULT, SCENE_PATH,
        ROOT / "experiments/runtime_antipodal_assembly_20261023/grasp_pose_gallery_20261023/render_grasps.py",
        ROOT / "experiments/runtime_antipodal_assembly_20261023/smaller_search_20261023/gallery.py"]
    before = {relative(path): digest(path) for path in source_paths}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    order = list(range(12))
    random.Random(SEED).shuffle(order)
    mapping = []
    for display_index, source_index in enumerate(order, 1):
        candidate = record["grasps"][source_index]
        original = saved["shortlisted_grasps"][source_index]
        assert candidate["world_T_grip"] == original["world_T_grip"]
        assert candidate["contacts_world_m"] == original["contacts_world_m"]
        # Explicitly copy only geometry/provenance. Never copy result records.
        mapping.append({
            "display_id": f"{display_index:02d}",
            "world_T_grip": candidate["world_T_grip"],
            "contacts_world_m": candidate["contacts_world_m"],
            "width_m": original["width_m"], "opening_m": original["opening_m"],
            "pad_depth_m": original["pad_depth_m"],
        })
    camera_record = render_views(mapping, saved["world_T_cad_input"])
    sheets = {}
    for view, label in (("camera_a", "Fixed camera A"), ("camera_b", "Fixed camera B")):
        width, height = 800, 520
        tile_height, header = height + 54, 155
        sheet = Image.new("RGB", (width*3, header+tile_height*4), "#f3f6f9")
        draw = ImageDraw.Draw(sheet)
        draw.text((24, 12), "Proposed VLM input, no model call", font=font(37, True), fill="#172b40")
        draw.text((24, 62), "Initial grasp views only; assembly-goal views not supplied", font=font(29), fill="#42576a")
        draw.text((24, 106), f"Large gear | {label} | Green rings: planned finger touches | Blue arrow: approach", font=font(25), fill="#42576a")
        for index, candidate in enumerate(mapping):
            image = Image.open(OUTPUT / candidate["rendered_images"][view]["file"]).convert("RGB")
            assert image.size == (width, height)
            x, y = (index % 3)*width, header+(index//3)*tile_height
            sheet.paste(image, (x, y+54))
            draw.rectangle((x, y, x+width-1, y+tile_height-1), outline="#98a8b7", width=2)
            draw.text((x+17, y+8), "Candidate " + candidate["display_id"], font=font(30, True), fill="#172b40")
        name = f"neutral_{view}_sheet.png"
        sheet.save(OUTPUT / name)
        sheets[view] = {"file": name, "sha256": digest(OUTPUT / name),
                        "size_pixels": list(sheet.size)}
    mapping_path = OUTPUT / "display_id_mapping.json"
    mapping_path.write_text(json.dumps({"part": "Gear_Large", "random_seed": SEED,
        "world_T_cad_input": saved["world_T_cad_input"], "candidates": mapping}, indent=2)+"\n")
    after = {relative(path): digest(path) for path in source_paths}
    assert before == after
    manifest = {
        "purpose": "Proposed candidate-image VLM review adaptation; not a RoboGround reproduction.",
        "model_calls": 0, "simulation_steps": 0, "random_seed": SEED,
        "candidate_count": 12, "layout": "3 columns by 4 rows", "sheets": sheets,
        "display_id_mapping": {"file": mapping_path.name, "sha256": digest(mapping_path)},
        "source_sha256": before, "protected_source_files_unchanged": before == after,
        "builder_source_sha256": digest(Path(__file__)),
        "fixed_camera_rendering": camera_record,
        "scope": "Initial grasp views only; assembly-goal views not supplied.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (OUTPUT / "README.md").write_text(
        "# Proposed VLM input, no model call\n\n"
        "These two sheets show the same 12 large-gear grasp candidates in a fixed shuffled order "
        "with neutral IDs. All candidates have the same border and text style.\n\n"
        "[Fixed camera A](neutral_camera_a_sheet.png) · [Fixed camera B](neutral_camera_b_sheet.png)\n\n"
        "The same two world-frame camera positions and the same zoom are used for every candidate. "
        "Initial grasp views only; assembly-goal views not supplied. This illustrates a proposed "
        "candidate-image review step and is not a RoboGround reproduction. No model was called "
        "and no simulation was run. The sheets contain no earlier selected-grasp markers, movement "
        "check results, ranking scores or pickup results. The exact source poses, contact points "
        "and image hashes are kept separately in display_id_mapping.json and manifest.json.\n")
    print(json.dumps({"output": str(OUTPUT), "sheets": sheets,
                      "protected_source_files_unchanged": True}, indent=2))


if __name__ == "__main__":
    build()
