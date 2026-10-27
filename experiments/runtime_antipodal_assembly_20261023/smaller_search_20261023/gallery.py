"""Render every small-search shortlist, without robot IK or physics steps.

The original CAD and collidable Panda finger/hand meshes are rendered. The
gripper drawing is moved rigidly into each saved world_T_grip frame after scene
construction. This is a pose illustration, never a claim of robot reachability.
"""
import html
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..grasp_pose_gallery_20261023.render_grasps import (
    GROUPS, add_contact_markers, camera_for, digest, font,
    mark_contacts_on_image, setup_model, show_only_part_and_gripper,
)


LABELS = {part: label.replace(" — successful follow-up", "")
          for group in GROUPS.values() for part, label in group}


def failed_check(grasp):
    return next((check for check in grasp.get("path_checks", [])
                 if check.get("status") != "passed"), {})


def plain_body(name):
    replacements = {"gripper0_leftfinger": "left finger", "gripper0_rightfinger": "right finger",
                    "gripper0_hand": "gripper body", "world": "table"}
    if name in replacements:
        return replacements[name]
    if name.startswith("robot0_link"):
        return "arm link " + name[len("robot0_link"):]
    return name.replace("_", " ")


def collision_cause(check):
    contacts = check.get("contacts", [])
    if not contacts:
        return "robot joint limit exceeded"
    contact = max(contacts, key=lambda c: c.get("penetration_m", 0))
    bodies = contact.get("bodies", [])
    if contact.get("kind") == "robot_table":
        robot = next((body for body in bodies if body.startswith(("robot0_", "gripper0_"))), "robot")
        return plain_body(robot) + " hits table"
    return " hits ".join(plain_body(body) for body in bodies)


def path_detail(grasp):
    check = failed_check(grasp)
    if not check:
        return ""
    status = check.get("status")
    if status == "ik_failed":
        reason = str(check.get("reason", ""))
        metrics = dict(re.findall(r"(position_error_m|rotation_error_rad)=([+\-\d.eE]+)", reason))
        if "position_error_m" in metrics and "rotation_error_rad" in metrics:
            return (f"Gripper position error {float(metrics['position_error_m'])*1000:.3f} mm "
                    f"and direction error {np.rad2deg(float(metrics['rotation_error_rad'])):.3f} degrees; "
                    "solver limits are 0.100 mm and 0.086 degrees.")
        return reason
    maximum = max((c.get("penetration_m", 0) for c in check.get("contacts", [])), default=0)*1000
    if status == "target_obstruction":
        return f"Open fingers overlap the part by up to {maximum:.3f} mm while moving toward the grasp; rejection limit 0.050 mm."
    if status == "collision":
        return f"{collision_cause(check).capitalize()}; largest overlap {maximum:.3f} mm; rejection limit 0.050 mm."
    return str(check.get("reason", status))


def pose_status(result, index, grasp):
    selected = result.get("selected_shortlist_index") == index
    if selected:
        return "Selected; pickup " + ("passed" if result.get("status") == "pickup_passed" else "attempted")
    state = str(grasp.get("path_check_status", "not_checked"))
    if state in {"passed", "pickup_path_passed"}:
        return "Robot movement checked"
    if state in {"not_checked", "not_tested", "untried", "untested", "pending"}:
        return "Robot movement not checked"
    failed = failed_check(grasp)
    phase = {"approach": "Approach", "descend": "Lowering", "lift": "Lifting"}.get(
        failed.get("phase"), str(failed.get("phase", "Movement")).capitalize())
    cause = failed.get("status")
    if cause == "target_obstruction":
        maximum = max((c.get("penetration_m", 0) for c in failed.get("contacts", [])), default=0)*1000
        return f"{phase}: open fingers hit part ({maximum:.2f} mm)"
    if cause == "ik_failed":
        return f"{phase}: no robot joint angles found"
    if cause == "collision":
        return f"{phase}: {collision_cause(failed)}"
    return "Robot movement rejected: " + state.replace("_", " ")


def move_gripper_drawing(scene, model, old_pose, new_pose):
    """Move each rendered collision mesh exactly; model and object stay fixed."""
    rotation = new_pose[:3, :3] @ old_pose[:3, :3].T
    moved, worst = [], 0.
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid < 0:
            continue
        name = model.geom(int(geom.objid)).name or ""
        if not name.startswith("gripper0_"):
            continue
        first_pos, first_mat = geom.pos.copy(), geom.mat.copy()
        expected_pos = rotation @ (first_pos - old_pose[:3, 3]) + new_pose[:3, 3]
        expected_mat = rotation @ first_mat
        geom.pos[:] = expected_pos
        geom.mat[:] = expected_mat
        error = max(float(np.max(np.abs(geom.pos - expected_pos))),
                    float(np.max(np.abs(geom.mat - expected_mat))))
        worst = max(worst, error)
        moved.append(name)
    if not moved:
        raise AssertionError("No collidable gripper meshes were drawn")
    return {"moved_gripper_geoms": moved, "max_scene_transform_roundoff": worst}


def make_tile(label, index, image, status, grasp, selected):
    tile = Image.new("RGB", (600, 465), "#ffffff")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 599, 464), outline="#16824d" if selected else "#cbd5df", width=4 if selected else 1)
    draw.text((13, 10), f"Grasp {index+1:02d}", font=font(22, True), fill="#192e43")
    draw.text((13, 42), status[:61], font=font(15), fill="#16824d" if selected else "#526171")
    tile.paste(image.resize((596, 338), Image.Resampling.LANCZOS), (2, 70))
    draw.text((13, 417), f"Finger depth {grasp['pad_depth_m']*1000:+.0f} mm  |  Open jaws {grasp['opening_m']*1000:.1f} mm",
              font=font(16), fill="#526171")
    return tile


def render_part(run, output, part):
    result_path = run / "small" / part / "result.json"
    result = json.loads(result_path.read_text())
    short = result["shortlisted_grasps"]
    folder = output / part
    folder.mkdir(parents=True, exist_ok=True)
    model, data = setup_model()
    mujoco.mj_forward(model, data)
    expected = np.asarray(result["world_T_cad_input"])
    bid = model.body("nist_part_" + part).id
    actual = np.eye(4)
    actual[:3, 3] = data.xpos[bid]
    actual[:3, :3] = data.xmat[bid].reshape(3, 3)
    if not np.allclose(expected, actual, atol=1e-7):
        raise AssertionError(f"{part}: input target differs from prepared scene")
    initial_target = actual.copy()
    original_qpos = data.qpos.copy()
    visual = model.geom("nist_part_" + part + "_visual").id
    mesh = model.geom_dataid[visual]
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[first:first+count] @ data.geom_xmat[visual].reshape(3, 3).T + data.geom_xpos[visual]
    extent = np.ptp(vertices, axis=0)
    hand = show_only_part_and_gripper(model, part)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    site = model.site("gripper0_grip_site").id
    records, tiles = [], []
    with mujoco.Renderer(model, height=520, width=800) as renderer:
        for index, grasp in enumerate(short):
            pose = np.asarray(grasp["world_T_grip"])
            contacts = np.asarray(grasp["contacts_world_m"])
            for number, sign in ((1, 1), (2, -1)):
                joint = model.joint(f"gripper0_finger_joint{number}").id
                data.qpos[model.jnt_qposadr[joint]] = sign*grasp["opening_m"]/2
            mujoco.mj_forward(model, data)
            old_pose = np.eye(4)
            old_pose[:3, 3] = data.site_xpos[site]
            old_pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
            images, pixels, audits = {}, {}, {}
            for view, detailed in (("overview", False), ("detail", True)):
                if hand is not None:
                    model.geom_rgba[hand, 3] = .07 if detailed else .20
                renderer.update_scene(data, camera=camera_for(pose, contacts, extent, detailed), scene_option=option)
                audits[view] = move_gripper_drawing(renderer.scene, model, old_pose, pose)
                add_contact_markers(renderer.scene, contacts, pose, detailed)
                picture = Image.fromarray(renderer.render().copy())
                pixels[view] = mark_contacts_on_image(picture, renderer.scene, contacts)
                filename = f"grasp_{index+1:02d}_{view}.png"
                picture.save(folder / filename)
                images[view] = f"{part}/{filename}"
                if detailed:
                    detail_picture = picture
            selected = index == result.get("selected_shortlist_index")
            status = pose_status(result, index, grasp)
            tile = make_tile(LABELS[part], index, detail_picture, status, grasp, selected)
            tile_name = f"grasp_{index+1:02d}_tile.png"
            tile.save(folder / tile_name)
            images["tile"] = f"{part}/{tile_name}"
            tiles.append(tile)
            records.append({"index": index, "candidate_id": grasp.get("candidate_id", index),
                "selected": selected, "status_label": status, "images": images,
                "pad_depth_m": grasp["pad_depth_m"], "opening_m": grasp["opening_m"],
                "table_clearance_m": grasp["table_clearance_m"],
                "path_checks": grasp.get("path_checks", []),
                "path_check_detail": path_detail(grasp),
                "world_T_grip": pose.tolist(), "contacts_world_m": contacts.tolist(),
                "contact_markers_pixels": pixels, "drawing_transform_audits": audits})
    actual_after = np.eye(4)
    actual_after[:3, 3] = data.xpos[bid]
    actual_after[:3, :3] = data.xmat[bid].reshape(3, 3)
    np.testing.assert_array_equal(actual_after, initial_target)
    target_joint = model.joint("nist_part_" + part + "_joint").id
    adr = model.jnt_qposadr[target_joint]
    np.testing.assert_array_equal(data.qpos[adr:adr+7], original_qpos[adr:adr+7])
    rows = max(1, (len(tiles)+2)//3)
    sheet = Image.new("RGB", (1800, 95+465*rows), "#f1f5f9")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 11), LABELS[part] + f" — all {len(tiles)} shortlisted grasps", font=font(31, True), fill="#192e43")
    draw.text((16, 55), "Green rings: finger touches. Blue: open gripper. Green border: selected for the pickup attempt.", font=font(21), fill="#526171")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index%3)*600, 95+(index//3)*465))
    sheet.save(folder / "all_shortlisted_grasps.png")
    record = {"part": part, "label": LABELS[part], "result_source": str(result_path),
              "result_sha256": digest(result_path), "pickup_status": result.get("status"),
              "selected_shortlist_index": result.get("selected_shortlist_index"),
              "world_T_cad_input": expected.tolist(), "target_pose_unchanged": True,
              "physics_steps": 0, "shortlist_count": len(records), "grasps": records,
              "sheet": f"{part}/all_shortlisted_grasps.png"}
    (folder / "render_record.json").write_text(json.dumps(record, indent=2)+"\n")
    return record


def build_groups(output, records):
    mapped = {record["part"]: record for record in records}
    group_files = {}
    for name, members in GROUPS.items():
        sheet = Image.new("RGB", (1800, 110+len(members)*510), "#f0f5fa")
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 10), "Smaller search: " + name.replace("_", " "), font=font(32, True), fill="#192e43")
        draw.text((16, 57), "Three examples per part. Open the gallery to see every shortlisted pose and its checking status.", font=font(23), fill="#526171")
        for row, (part, _) in enumerate(members):
            record = mapped[part]
            top = 110+row*510
            draw.text((16, top), record["label"] + f" — {record['shortlist_count']} in shortlist", font=font(24, True), fill="#192e43")
            for column, grasp in enumerate(record["grasps"][:3]):
                tile = Image.open(output / grasp["images"]["tile"])
                sheet.paste(tile, (column*600, top+39))
        filename = name + "_shortlist_examples.png"
        sheet.save(output / filename)
        group_files[name] = filename
    return group_files


def build_html(output, records):
    cards = []
    for part_index, record in enumerate(records):
        rows = []
        for grasp in record["grasps"]:
            image = html.escape(grasp["images"]["tile"], quote=True)
            rows.append(f'<button class="pose" onclick="show({part_index},{grasp["index"]},\'detail\')"><img src="{image}" loading="lazy" alt="{html.escape(record["label"])} grasp {grasp["index"]+1}"></button>')
        cards.append(f'<section id="{html.escape(record["part"])}"><h2>{html.escape(record["label"])} <small>{record["shortlist_count"]} grasps</small></h2><p>Pickup result: {html.escape(str(record["pickup_status"]).replace("_", " "))}. <a href="{html.escape(record["sheet"])}">Open all grasps as one picture</a></p><div class="poses">'+"".join(rows)+"</div></section>")
    data = json.dumps(records).replace("<", "\\u003c")
    document = HTML.replace("__CARDS__", "\n".join(cards)).replace("__DATA__", data)
    (output / "index.html").write_text(document, encoding="utf-8")


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All 14 parts: smaller-search grasp shortlists</title>
<style>*{box-sizing:border-box}body{margin:0;font:16px/1.5 system-ui,sans-serif;color:#182d42;background:#f4f7fa}main{max-width:1500px;padding:28px;margin:auto}h1{font-size:34px;line-height:1.2;margin:0 0 14px}p{max-width:1050px;color:#526171}section{padding:18px 0;border-top:1px solid #cbd5df;margin-top:28px}h2{font-size:26px;margin:0}small{font-size:16px;color:#526171;font-weight:400;margin-left:15px}.poses{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.pose{display:block;padding:0;border:0;background:white;cursor:pointer}.pose img{display:block;width:100%}button:focus-visible,a:focus-visible{outline:3px solid #2086d5}dialog{border:1px solid #b9c8d5;border-radius:12px;width:min(1200px,96vw);max-height:96vh;padding:14px}dialog::backdrop{background:#112233bb}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.tools button,.tools a{padding:7px 12px;border:1px solid #becbd7;border-radius:6px;background:white;font:inherit;color:#245776;text-decoration:none}.tools h2{font-size:21px;flex:1}#big{display:block;max-width:100%;max-height:73vh;margin:8px auto}#caption{font-size:14px;margin:0}@media(max-width:1050px){.poses{grid-template-columns:repeat(3,1fr)}}@media(max-width:680px){main{padding:16px}.poses{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}h2{font-size:23px}}</style>
<main><h1>Smaller-search grasp shortlists for all 14 parts</h1><p>Every shortlisted gripper pose is shown individually. Orange is the part, transparent blue is the actual gripper geometry, green rings are the two planned finger touches, and the blue arrow shows the approach. The gripper is open before closing. Select any picture to inspect it.</p><p><strong>Finger depth</strong> gives the touch position along the fingers. A larger positive value is nearer the finger ends. With downward-pointing fingers, a larger value places the gripper higher.</p><p>Each shortlist passed the table check. The tile states whether robot movement was checked. Only the pose with a green border was selected for a physical pickup attempt in simulation. The other pictures do not establish that the robot can reach or pick up the part there. Other loose parts are omitted from these close-up illustrations; they remained visible, with contacts disabled, during the pickup trial. No assembly was attempted in this comparison.</p>__CARDS__</main>
<dialog id="zoom"><div class="tools"><h2 id="title"></h2><button onclick="step(-1)">Previous grasp</button><button onclick="step(1)">Next grasp</button><button onclick="show(activePart,activePose,'overview')">Whole gripper</button><button onclick="show(activePart,activePose,'detail')">Fingers close-up</button><a id="open" target="_blank">Open image</a><button onclick="document.getElementById('zoom').close()">Close</button></div><img id="big" alt="Generated grasp pose"><p id="caption"></p></dialog>
<script id="data" type="application/json">__DATA__</script><script>const records=JSON.parse(document.getElementById('data').textContent);let activePart=0,activePose=0,view='detail';function show(part,pose,nextView){activePart=part;activePose=pose;view=nextView;const record=records[part],g=record.grasps[pose];document.getElementById('title').textContent=record.label+' · grasp '+(pose+1)+' / '+record.grasps.length;document.getElementById('big').src=g.images[view];document.getElementById('open').href=g.images[view];document.getElementById('caption').textContent=g.status_label+'. '+g.path_check_detail+' Finger depth: '+(g.pad_depth_m*1000).toFixed(0)+' mm; open jaws: '+(g.opening_m*1000).toFixed(1)+' mm; table clearance: '+(g.table_clearance_m*1000).toFixed(2)+' mm. Static drawing of the saved pose; no physics steps.';const d=document.getElementById('zoom');if(!d.open)d.showModal()}function step(delta){const count=records[activePart].grasps.length;show(activePart,(activePose+delta+count)%count,view)}document.getElementById('zoom').addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();step(-1)}if(e.key==='ArrowRight'){e.preventDefault();step(1)}});</script></html>'''


def main():
    run = Path(sys.argv[1]).resolve()
    output = run / "shortlist_gallery"
    output.mkdir(exist_ok=False)
    records = []
    for members in GROUPS.values():
        for part, _ in members:
            print("RENDER " + part, flush=True)
            records.append(render_part(run, output, part))
    groups = build_groups(output, records)
    manifest = {"source_run": str(run), "render_source_sha256": digest(Path(__file__)),
        "rendering": "Exact model collision meshes moved rigidly into saved world_T_grip drawing frames; no robot IK and no physics.",
        "parts": records, "group_images": groups, "physics_steps": 0,
        "part_count": len(records), "grasp_count": sum(r["shortlist_count"] for r in records)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    build_html(output, records)
    (output / "README.md").write_text("# Smaller-search grasp shortlists\n\n[Open every part and every shortlisted pose](index.html).\n\nEach tile shows one generated pose with its checking status. Green borders identify selected pickup attempts. The original gripper collision meshes are drawn in the saved gripper frame; no robot inverse-kinematics solution is implied for untested poses. No physics steps were run to make these pictures. Object positions are unchanged. All transforms and source hashes are in manifest.json.\n")
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
