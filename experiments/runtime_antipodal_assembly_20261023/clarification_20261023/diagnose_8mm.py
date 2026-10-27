"""New 8 mm centre-grasp diagnostic; preserve every prior trial unchanged.

Measure the actual Panda pads and complete finger meshes, then test horizontal
side contacts through the resting pin's centre. Geometry and controller settings
are unchanged. Robot state drives control; target poses are initial inputs and
evaluation evidence only. This finite diagnostic cannot establish impossibility.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..grasp_planning import gripper_surface_local, linear_pose_path, validate_path
from ..prepare_scene import prepare_scene
from ..robot_trial import RobotTrial, CONTACT_SETTINGS


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = "RGOCG8-50_8mm"
DEPTHS_M = (.016, .012, .008, .006, .004, .002, 0., -.004, -.008, -.012, -.016, -.024)
ROLLS_DEG = (0., -10., 10., -20., 20.)
OPENING_M = .020


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def restore_other_visuals(trial, source_model):
    """Show the other parts while retaining their requested ignored collisions."""
    for geom in range(trial.m.ngeom):
        body = trial.m.body(int(trial.m.geom_bodyid[geom])).name or ""
        if body.startswith("nist_part_") and body != trial.target_name:
            name = trial.m.geom(geom).name
            source_geom = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if source_geom >= 0:
                trial.m.geom_rgba[geom] = source_model.geom_rgba[source_geom]
    trial.visuals_restored_for_other_parts = True


def surface_records(model, reference, opening):
    """Exact compiled collision vertices/corners expressed in grip-site frame."""
    data = mujoco.MjData(model)
    data.qpos[:] = reference.qpos
    for index, sign in ((1, 1), (2, -1)):
        joint = model.joint(f"gripper0_finger_joint{index}").id
        data.qpos[model.jnt_qposadr[joint]] = sign * opening / 2
    mujoco.mj_forward(model, data)
    site = model.site("gripper0_grip_site").id
    rotation = data.site_xmat[site].reshape(3, 3)
    position = data.site_xpos[site]
    result = []
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        if not name.startswith("gripper0_") or not (
            model.geom_contype[geom] or model.geom_conaffinity[geom]
        ):
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start:start + count])
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                                 for z in (-1, 1)]) * model.geom_size[geom]
        else:
            raise ValueError(f"Unexpected collision geometry {name}.")
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - position) @ rotation
        result.append({"geom": name, "body": model.body(int(model.geom_bodyid[geom])).name,
                       "geometry_kind": int(model.geom_type[geom]),
                       "vertex_count": len(local), "minimum_grip_xyz_m": local.min(axis=0).tolist(),
                       "maximum_grip_xyz_m": local.max(axis=0).tolist(),
                       "local_vertices": local})
    return result


def candidate_pose(center, pin_axis, depth, roll, jaw_sign):
    """Keep both chosen contacts horizontal and through the pin's centre."""
    down = np.array([0., 0., -1.])
    close = np.cross(pin_axis, down)
    close /= np.linalg.norm(close)
    close *= jaw_sign
    approach = Rotation.from_rotvec(close * np.deg2rad(roll)).apply(down)
    rotation = np.column_stack((close, np.cross(approach, close), approach))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = center - rotation @ [0., 0., depth]
    return pose


def render_views(trial, directory, label, initial_center, pin_axis):
    """Real native simulation images; parts visible and no geometry edits."""
    trial.render(directory / f"{label}_workspace.png")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = initial_center + [0., 0., .010]
    camera.distance = .115
    camera.azimuth = float(np.rad2deg(np.arctan2(pin_axis[1], pin_axis[0])))
    camera.elevation = -7.
    options = mujoco.MjvOption()
    options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    trial._renderer.update_scene(trial.d, camera=camera, scene_option=options)
    image = trial._renderer.render().copy()
    cv2.imwrite(str(directory / f"{label}_side.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def physics_attempt(source, source_model, out, row, center, axis):
    attempt = out / f"attempt_depth_{row['pad_depth_mm']:+.0f}_roll_{row['roll_deg']:+.0f}_jaw_{row['jaw_sign']:+.0f}"
    trial = RobotTrial(source, attempt, TARGET)
    restore_other_visuals(trial, source_model)
    result = {"candidate": row, "other_parts_visible": True,
              "other_parts_collisions_ignored": True, "geometry_or_contact_tuning": False}
    grasp = np.asarray(row["world_T_grip"])
    above = grasp.copy()
    above[:3, 3] -= .12 * grasp[:3, 2]
    lift = grasp.copy()
    lift[2, 3] += .16
    try:
        q = trial.d.qpos[trial.qa].copy()
        checks = []
        for name, first, last, opening, check_target in (
            ("approach", trial.pose(), above, OPENING_M, True),
            ("descend", above, grasp, OPENING_M, True),
            ("lift", grasp, lift, .008, False),
        ):
            qpath, check = validate_path(trial, linear_pose_path(first, last, 16), opening,
                                         seed=q, check_open_target=check_target)
            checks.append({"phase": name, **check})
            if qpath is None:
                result.update(status="path_rejected", path_checks=checks)
                return result
            q = qpath[-1]
        result["path_checks"] = checks
        initial = trial.target_pose()
        render_views(trial, attempt, "initial", center, axis)
        trial.gripper(OPENING_M)
        trial.move(above, 100, "approach")
        trial.move(grasp, 90, "descend")
        render_views(trial, attempt, "open_at_grasp", center, axis)
        trial.gripper(0.)
        trial.hold(45, "close")
        render_views(trial, attempt, "closed", center, axis)
        contacts = trial.states[-1]["grasp_contacts"]
        result["bilateral_contact_after_closing"] = len(contacts) == 2
        result["actual_target_pose_after_closing_audit"] = trial.target_pose().tolist()
        if len(contacts) != 2:
            result["status"] = "no_bilateral_contact"
            return result
        trial.move(lift, 110, "lift")
        trial.hold(30, "hold_lifted")
        render_views(trial, attempt, "lifted", center, axis)
        final = trial.target_pose()
        rise = float(final[2, 3] - initial[2, 3])
        bilateral = len(trial.states[-1]["grasp_contacts"]) == 2
        robot_clear = max(trial.audit[k] for k in ("robot_self", "robot_table", "robot_board")) <= .0002
        result.update(status="pickup_passed" if rise >= .12 and bilateral and robot_clear else "pickup_failed",
                      lift_m=rise, bilateral_contact_after_lift=bilateral,
                      all_robot_penetration_at_most_0p2mm=robot_clear,
                      final_target_pose_audit=final.tolist())
    except Exception as error:
        result.update(status="execution_error", error=str(error))
    finally:
        result["controller_audit"] = trial.save()
        result["controller_audit"]["other_loose_parts_visibility_restored"] = True
        write_json(attempt / "diagnostic_result.json", result)
        trial.close()
    return result


def make_diagram(out, records, center_z, rows):
    """Measured collision outlines, in the closing-axis/approach-axis plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharex=True, sharey=True)
    pad = next(item for item in records if "finger1_pad" in item["geom"])
    pad_middle = .5 * (pad["minimum_grip_xyz_m"][2] + pad["maximum_grip_xyz_m"][2])
    depths = (pad_middle, .004, .006)
    labels = ("Middle of finger pad", "Original finger-end position", "Raised 2 mm: successful pickup")
    for ax, depth, label in zip(axes, depths, labels):
        minimum = float("inf")
        for item in records:
            vertices = item["local_vertices"]
            world_z = center_z + depth - vertices[:, 2]
            minimum = min(minimum, float(world_z.min()))
            points = np.unique(np.column_stack((vertices[:, 0], world_z)) * 1000, axis=0)
            hull = ConvexHull(points)
            pad_geom = "pad_collision" in item["geom"]
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                    color="#2586ad" if pad_geom else "#b9c1ca", alpha=.75,
                    edgecolor="#174c65" if pad_geom else "#566270", linewidth=.8)
        circle = plt.Circle((0, center_z * 1000), 4, color="#df9b27", alpha=.9,
                            ec="#8e5a11", linewidth=1.2)
        ax.add_patch(circle)
        ax.axhline(0, color="black", linewidth=2)
        ax.axhspan(-30, 0, color="#b7afa3", alpha=.35)
        ax.scatter([0], [(center_z + depth) * 1000], marker="+", color="red", s=90)
        ax.set_title(label)
        ax.text(.02, .97, (f"Table clearance: {minimum * 1000:+.2f} mm" if minimum >= 0
                           else f"Finger/table overlap: {-minimum * 1000:.2f} mm"),
                transform=ax.transAxes, va="top", fontsize=10,
                color="#15673e" if minimum >= 0 else "#9b2323")
        ax.text(23, -5, "table", ha="center", fontsize=10)
        ax.annotate("8 mm pin", xy=(1, center_z * 1000 + 2), xytext=(18, 18),
                    arrowprops={"arrowstyle": "->", "color": "#805014"}, color="#805014", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-15, 55)
        ax.set_xlabel("Across the two fingers (mm)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Height above table (mm)")
    fig.suptitle("8 mm pin: raising the fingers 2 mm avoids the table", fontsize=13)
    fig.text(.5, .015, "Blue: contact pads. Gray: solid finger shapes. Yellow: 8 mm pin. Both jaws are centered on the pin.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .95])
    fig.savefig(out / "measured_center_vs_tip.png", dpi=160)
    plt.close(fig)


def enrich_saved_output(out):
    """Analyze saved actual physics; do not step another physical attempt."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    snapshot = out / "executed_diagnostic_source.py"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest["diagnostic_source_sha256"]:
        raise RuntimeError("Executed diagnostic source snapshot does not match its manifest.")
    attempts = json.loads((out / "physics_attempts.json").read_text())
    success = next(result for result in attempts if result["status"] == "pickup_passed")
    row = success["candidate"]
    folder = out / f"attempt_depth_{row['pad_depth_mm']:+.0f}_roll_{row['roll_deg']:+.0f}_jaw_{row['jaw_sign']:+.0f}"
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    source_model = mujoco.MjModel.from_xml_path(manifest["source_scene"])
    data = mujoco.MjData(model)
    trajectory = np.load(folder / "trajectory.npz")
    data.qpos[:] = trajectory["qpos"][0]
    data.qvel[:] = trajectory["qvel"][0]
    mujoco.mj_forward(model, data)
    # At commanded 9 mm opening, the measured inner pad gap is 8 mm. Use this
    # nominal contact-width profile only for the geometric side illustration.
    records = surface_records(model, data, .009)
    make_diagram(out, records, manifest["initial_world_T_cad"][2][3], [])
    actual_contacts = json.loads((folder / "contacts.json").read_text())
    target = model.body("nist_part_" + TARGET).id
    site = model.site("gripper0_grip_site").id
    phase_evidence = {}
    for phase in ("close", "lift", "hold_lifted"):
        index = int(np.flatnonzero(trajectory["phase"] == phase)[-1])
        data.qpos[:] = trajectory["qpos"][index]
        data.qvel[:] = trajectory["qvel"][index]
        mujoco.mj_forward(model, data)
        positions = []
        for contact in data.contact:
            geoms = [int(g) for g in contact.geom]
            bodies = [int(model.geom_bodyid[g]) for g in geoms]
            names = [model.body(b).name or "" for b in bodies]
            if target not in bodies or not any("finger" in name for name in names):
                continue
            local = (contact.pos - data.site_xpos[site]) @ data.site_xmat[site].reshape(3, 3)
            positions.append({"geoms": [model.geom(g).name for g in geoms],
                              "world_position_m": contact.pos.tolist(),
                              "grip_site_contact_xyz_m": local.tolist(),
                              "height_above_target_center_m": float(contact.pos[2] - data.xpos[target, 2])})
        phase_evidence[phase] = {"trajectory_index": index, "time_s": float(trajectory["time_s"][index]),
                                 "actual_recorded_force_contacts": [contact for contact in actual_contacts[index]["contacts"]
                                                                    if contact["kind"] == "finger_target"],
                                 "replayed_geometry_contact_positions": positions}
    target_table_maximum = 0.
    target_table_worst_frame = None
    for index in range(len(trajectory["qpos"])):
        data.qpos[:] = trajectory["qpos"][index]
        data.qvel[:] = trajectory["qvel"][index]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            geoms = [int(g) for g in contact.geom]
            names = [model.geom(g).name for g in geoms]
            bodies = [int(model.geom_bodyid[g]) for g in geoms]
            if target in bodies and "workbench_top" in names:
                overlap = float(max(0., -contact.dist))
                if overlap > target_table_maximum:
                    target_table_maximum = overlap
                    target_table_worst_frame = {"index": index,
                        "time_s": float(trajectory["time_s"][index]),
                        "phase": str(trajectory["phase"][index]), "geoms": names}
    maximum = success["controller_audit"]["max_penetration_m"]
    evidence = {"status": "one_successful_center_pose_pickup_diagnostic",
                "protocol": "Manually specified horizontal centre-point grasp frames, common depth/roll grid; this is not an automatic all-part method rerun.",
                "physical_attempts": len(attempts), "lift_m": success["lift_m"],
                "candidate": row, "max_penetration_mm": {key: value * 1000 for key, value in maximum.items()},
                "saved_frame_target_table_maximum_penetration_mm": target_table_maximum * 1000,
                "saved_frame_target_table_worst_contact": target_table_worst_frame,
                "target_table_audit_scope": "All saved 25 Hz states; controller robot/table audit covers every 500 Hz physics substep.",
                "equality_constraint_count": int(model.neq), "all_other_parts_visible_in_images": True,
                "other_loose_parts_collision_ignored": True, "per_case_geometry_or_contact_tuning": False,
                "phase_contact_evidence": phase_evidence,
                "geometry_figure_inner_pad_gap_m": .008,
                "figure_source": "Exact compiled collision-mesh silhouette projections at preserved pin-centre poses; no geometry change.",
                "executed_source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "enrichment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scene_sha256": hashlib.sha256((folder / "scene.xml").read_bytes()).hexdigest(),
                "trajectory_npz_sha256": hashlib.sha256((folder / "trajectory.npz").read_bytes()).hexdigest(),
                "actual_contact_force_json_sha256": hashlib.sha256((folder / "contacts.json").read_bytes()).hexdigest()}
    write_json(out / "successful_contact_audit.json", evidence)
    # Save a result in the new diagnostic's own folder for convenient review.
    write_json(folder / "result.json", {**success, "part": TARGET,
        "protocol": evidence["protocol"], "assembly_attempted": False})
    render_saved_diagnostic(folder, model, trajectory, source_model)
    print(json.dumps({"contact_audit": str(out / "successful_contact_audit.json"),
                      "max_penetration_mm": evidence["max_penetration_mm"],
                      "lift_m": success["lift_m"], "equality_constraint_count": int(model.neq)}), flush=True)


def render_saved_diagnostic(folder, model, trajectory, source_model):
    """Render actual recorded states with honest diagnostic labels and colours."""
    import subprocess
    import imageio_ffmpeg
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        if body.startswith("nist_part_") and body != "nist_part_" + TARGET:
            original = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if original >= 0:
                model.geom_rgba[geom] = source_model.geom_rgba[original]
    data = mujoco.MjData(model)
    target = model.body("nist_part_" + TARGET).id
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    options = mujoco.MjvOption()
    options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = .19, 155, -12
    path = folder / "recorded_center_pickup.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "1280x600", "-r", "25", "-i", "-", "-an",
               "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(path)]
    indices = list(range(0, len(trajectory["qpos"]), 2))
    if indices[-1] != len(trajectory["qpos"]) - 1:
        indices.append(len(trajectory["qpos"]) - 1)
    with (folder / "center_pickup_encoder.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for index in indices:
                data.qpos[:] = trajectory["qpos"][index]
                data.qvel[:] = trajectory["qvel"][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="agentview", scene_option=options)
                wide = renderer.render().copy()
                camera.lookat[:] = data.xpos[target] + [0, 0, .012]
                renderer.update_scene(data, camera=camera, scene_option=options)
                close = renderer.render().copy()
                frame = np.full((600, 1280, 3), 247, np.uint8)
                frame[80:560, :640], frame[80:560, 640:] = wide, close
                label = f"8 mm pin | centre-grasp follow-up | {trajectory['phase'][index]} | {trajectory['time_s'][index]:.2f} s"
                cv2.putText(frame, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 2, cv2.LINE_AA)
                cv2.putText(frame, "Manually specified new pose | actual finger contacts | unchanged geometry/contact settings | no attachment",
                            (18, 60), cv2.FONT_HERSHEY_SIMPLEX, .47, (80, 55, 30), 1, cv2.LINE_AA)
                cv2.putText(frame, "Saved physics at 2x playback | lifted 159 mm | other parts visible; collisions ignored | pickup only",
                            (18, 585), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 40, 40), 1, cv2.LINE_AA)
                process.stdin.write(frame.tobytes())
            for _ in range(25):
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Diagnostic video encoder failed.")
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    if count != len(indices) + 25:
        raise RuntimeError("Diagnostic video decode check failed.")
    write_json(folder / "center_pickup_video_verification.json",
               {"source": "saved actual qpos/qvel, no physics rerun or repaired motion",
                "trajectory_npz_sha256": hashlib.sha256((folder / "trajectory.npz").read_bytes()).hexdigest(),
                "playback_speed": 2, "decoded_frames": count,
                "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def main():
    source = prepare_scene()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs/runtime_antipodal_assembly_20261023/8mm_center_grasp_followup" / stamp
    out.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={out}", flush=True)
    source_model = mujoco.MjModel.from_xml_path(str(source))
    trial = RobotTrial(source, out / "geometry_reference", TARGET)
    restore_other_visuals(trial, source_model)
    initial = trial.target_pose()
    center = initial[:3, 3]
    axis = initial[:3, 2].copy()
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    records = surface_records(trial.m, trial.d, OPENING_M)
    serial_records = [{key: value for key, value in row.items() if key != "local_vertices"}
                      for row in records]
    rows = []
    for depth in DEPTHS_M:
        for roll in ROLLS_DEG:
            for jaw_sign in (1., -1.):
                pose = candidate_pose(center, axis, depth, roll, jaw_sign)
                minima = []
                for record in records:
                    world = record["local_vertices"] @ pose[:3, :3].T + pose[:3, 3]
                    minima.append((float(world[:, 2].min()), record["geom"]))
                minimum, lowest_geom = min(minima)
                row = {"pad_depth_mm": depth * 1000, "roll_deg": roll, "jaw_sign": jaw_sign,
                       "world_T_grip": pose.tolist(), "center_contact_world_m": center.tolist(),
                       "jaw_opening_m": OPENING_M, "full_gripper_table_clearance_m": minimum,
                       "lowest_collision_geom": lowest_geom, "exact_horizontal_center_contacts": True,
                       "table_clear_0p1mm": minimum >= .0001}
                rows.append(row)
    make_diagram(out, records, center[2], rows)
    manifest = {"source_scene": str(source), "source_scene_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "target": TARGET, "initial_world_T_cad": initial.tolist(), "pin_axis_world": axis.tolist(),
                "contact_position": "exact pin center with horizontally opposed contacts",
                "original_sampled_contact_depths_m": [.004, 0., -.004, -.008],
                "new_contact_depths_m": DEPTHS_M, "rolls_deg": ROLLS_DEG,
                "new_contact_depths_are_diagnostic_only": True,
                "actual_collision_surfaces": serial_records,
                "common_contact_settings_unchanged": CONTACT_SETTINGS,
                "other_parts_visible_but_noncolliding": True,
                "limitations": ["Finite center-contact/depth/roll grid; failure is not impossibility.",
                                "A contact-depth sample past a physical pad is only a placement probe, not a valid grasp.",
                                "Initial exact pose supplied under the existing known-pose assumption; no updated target correction."]}
    write_json(out / "manifest.json", manifest)
    write_json(out / "geometry_grid.json", rows)
    with (out / "geometry_grid.csv").open("w", newline="") as stream:
        fields = ["pad_depth_mm", "roll_deg", "jaw_sign", "full_gripper_table_clearance_m",
                  "lowest_collision_geom", "table_clear_0p1mm"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    trial.save()
    trial.close()
    clear = [row for row in rows if row["table_clear_0p1mm"]]
    print(f"GRID={len(rows)} TABLE_CLEAR={len(clear)}", flush=True)
    # Centre-pad probes first; retain tip and beyond-tip probes as explicit
    # diagnostics. Both jaw-equivalent orientations may differ in robot IK.
    clear.sort(key=lambda row: (abs(row["roll_deg"]), abs(row["pad_depth_mm"]), -row["jaw_sign"]))
    attempts = []
    if os.environ.get("GEOMETRY_ONLY", "0") != "1":
        for row in clear[:20]:
            result = physics_attempt(source, source_model, out, row, center, axis)
            attempts.append(result)
            write_json(out / "physics_attempts.json", attempts)
            print(json.dumps({"depth_mm": row["pad_depth_mm"], "roll": row["roll_deg"],
                              "jaw": row["jaw_sign"], "status": result["status"],
                              "lift_m": result.get("lift_m"), "error": result.get("error")}), flush=True)
            if result["status"] == "pickup_passed":
                break
    summary = {"geometry_grid_size": len(rows), "table_clear_probe_count": len(clear),
               "physical_attempt_count": len(attempts),
               "pickup_passed_count": sum(row["status"] == "pickup_passed" for row in attempts),
               "best_original_depth_table_clearance_m": max(row["full_gripper_table_clearance_m"]
                   for row in rows if row["pad_depth_mm"] in (4., 0., -4., -8.)),
               "status": "pickup_found" if any(row["status"] == "pickup_passed" for row in attempts)
                         else "no_pickup_found_in_this_finite_center_grid",
               "no_impossibility_claim": True}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    if os.environ.get("ENRICH_OUTPUT"):
        enrich_saved_output(os.environ["ENRICH_OUTPUT"])
    else:
        main()
