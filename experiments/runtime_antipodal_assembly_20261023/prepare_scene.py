"""Freeze a known-CAD input scene, correcting the retained USB variation only.

The external objects-centered source contains a USB-with-cable visual asset and
disabled USB collisions. This new trial deliberately uses the native catalog USB
instead. All other part poses and geometry are preserved. No external source or
historical output is written by this module.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = Path("/mnt/c/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to"
              "/nist-taskboard-workspace/objects-centered/scene.xml")
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
OUTPUT = ROOT / "outputs/runtime_antipodal_assembly_20261023/input_scene"
USB_NAME = "nist_part_USB_Male"
SUPPORT_CLEARANCE_M = .001
PENETRATION_LIMIT_M = .0002


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pose(model, data, name):
    body = model.body(name).id
    result = np.eye(4)
    result[:3, :3] = data.xmat[body].reshape(3, 3)
    result[:3, 3] = data.xpos[body]
    return result


def _reset_keyframe(model):
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "objects_centered")
    if key < 0:
        raise ValueError("The source requires an objects_centered keyframe.")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def _resolve_asset_paths(root, source_directory):
    compiler = root.find("compiler")
    if compiler is not None:
        for name in ("meshdir", "texturedir"):
            value = compiler.get(name)
            if value and not Path(value).is_absolute():
                compiler.set(name, str((source_directory / value).resolve()))
    for asset in root.findall("asset/*"):
        file_name = asset.get("file")
        if file_name and not Path(file_name).is_absolute():
            directory_name = "meshdir" if asset.tag == "mesh" else "texturedir"
            directory = (Path(compiler.get(directory_name)) if compiler is not None
                         and compiler.get(directory_name) else source_directory)
            asset.set("file", str((directory / file_name).resolve()))


def _verify_existing(scene_path, audit_path):
    if not scene_path.is_file() or not audit_path.is_file():
        raise FileExistsError("Input-scene freeze is incomplete; it will not be overwritten.")
    audit = json.loads(audit_path.read_text())
    if _sha256(scene_path) != audit["prepared_scene_sha256"]:
        raise RuntimeError("Frozen input-scene hash changed.")
    if _sha256(SOURCE) != audit["external_source_sha256"]:
        raise RuntimeError("External source changed after the input scene was frozen.")
    for record in audit["parts"]:
        if _sha256(record["native_cad_path"]) != record["native_cad_sha256"]:
            raise RuntimeError(f"Frozen CAD hash changed: {record['part']}")
    if len(audit["parts"]) != 14 or not audit["all_14_visual_cad_matches"]:
        raise RuntimeError("Frozen input-scene CAD audit is invalid.")
    if audit["maximum_initial_part_table_penetration_m"] > PENETRATION_LIMIT_M:
        raise RuntimeError("Frozen input-scene initial support check failed.")
    return scene_path


def prepare_scene():
    """Create the input once and return its scene path; validate an existing freeze."""
    scene_path = OUTPUT / "scene.xml"
    audit_path = OUTPUT / "scene_input_audit.json"
    if OUTPUT.exists():
        return _verify_existing(scene_path, audit_path)
    source_hash = _sha256(SOURCE)
    original = mujoco.MjModel.from_xml_path(str(SOURCE))
    original_data = _reset_keyframe(original)
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    _resolve_asset_paths(root, SOURCE.parent)
    bodies = [body for body in root.findall("worldbody/body")
              if body.get("name", "").startswith("nist_part_")]
    if len(bodies) != 14:
        raise ValueError(f"Expected 14 retained loose parts, found {len(bodies)}.")
    body = next(body for body in bodies if body.get("name") == USB_NAME)
    mesh_asset = root.find("asset/mesh[@name='nist_USB_Male']")
    if mesh_asset is None:
        raise ValueError("USB mesh asset is unavailable.")
    old_usb_asset = dict(mesh_asset.attrib)
    native_usb_path = (MESHES / "USB_Male_m.stl").resolve()
    mesh_asset.set("file", str(native_usb_path))
    mesh_asset.attrib.pop("scale", None)
    collision = body.find("geom[@name='nist_part_USB_Male_collision']")
    if collision is None:
        raise ValueError("USB collision geometry is unavailable.")
    old_usb_collision = dict(collision.attrib)
    collision.set("contype", "1")
    collision.set("conaffinity", "1")
    collision.set("group", "0")
    # Keep the existing body-frame XY and rotation. Recompute only height from
    # every exact native CAD vertex, retaining the source scene's 1 mm gap.
    usb_pose_before = _pose(original, original_data, USB_NAME)
    native_usb = o3d.io.read_triangle_mesh(str(native_usb_path))
    vertices = np.asarray(native_usb.vertices)
    if not len(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("Native USB CAD mesh is empty or invalid.")
    rotated = vertices @ usb_pose_before[:3, :3].T
    usb_pose_after = usb_pose_before.copy()
    usb_pose_after[2, 3] = SUPPORT_CLEARANCE_M - float(rotated[:, 2].min())
    body.set("pos", " ".join(format(value, ".17g") for value in usb_pose_after[:3, 3]))
    usb_joint = original.joint("nist_part_USB_Male_joint").id
    usb_qadr = int(original.jnt_qposadr[usb_joint])
    key = root.find("keyframe/key[@name='objects_centered']")
    key_qpos = np.fromstring(key.get("qpos", ""), sep=" ")
    if len(key_qpos) != original.nq:
        raise ValueError("Source keyframe qpos length does not match its model.")
    key_qpos[usb_qadr + 2] = usb_pose_after[2, 3]
    key.set("qpos", " ".join(format(value, ".17g") for value in key_qpos))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    tree.write(scene_path, encoding="unicode")
    prepared = mujoco.MjModel.from_xml_path(str(scene_path))
    prepared_data = _reset_keyframe(prepared)
    if prepared.nq != original.nq or prepared.nv != original.nv:
        raise RuntimeError("Input correction unexpectedly changed state dimensions.")
    other_qpos = np.ones(original.nq, dtype=bool)
    other_qpos[usb_qadr + 2] = False
    np.testing.assert_array_equal(prepared_data.qpos[other_qpos], original_data.qpos[other_qpos])
    all_assets = {asset.get("name"): asset for asset in root.findall("asset/mesh")}
    records = []
    for part_body in bodies:
        name = part_body.get("name")
        part = name.removeprefix("nist_part_")
        visual = part_body.find(f"geom[@name='{name}_visual']")
        asset = all_assets[visual.get("mesh")]
        native_path = (MESHES / f"{part}_m.stl").resolve()
        actual_path = Path(asset.get("file")).resolve()
        exact = (actual_path == native_path
                 and np.array_equal(np.fromstring(asset.get("scale", "1 1 1"), sep=" "), [1., 1., 1.])
                 and not visual.get("pos") and not visual.get("quat"))
        if not exact:
            raise RuntimeError(f"Source visual and native CAD frame differ: {part}")
        transform = _pose(prepared, prepared_data, name)
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
        if not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-8):
            raise RuntimeError(f"Invalid part rotation: {part}")
        native_mesh = o3d.io.read_triangle_mesh(str(native_path))
        part_vertices = np.asarray(native_mesh.vertices)
        world_vertices = part_vertices @ transform[:3, :3].T + transform[:3, 3]
        active = [geom.get("name") for geom in part_body.findall(".//geom")
                  if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1"))]
        if not active:
            raise RuntimeError(f"Part has no active collision geometry: {part}")
        records.append({"part": part, "native_cad_path": str(native_path),
                        "native_cad_sha256": _sha256(native_path), "visual_cad_exact_match": exact,
                        "active_collision_count": len(active), "active_collision_geoms": active,
                        "world_T_cad_initial": transform.tolist(),
                        "native_mesh_minimum_world_z_m": float(world_vertices[:, 2].min()),
                        "pose_changed_from_source": part == "USB_Male"})
    target_bodies = {prepared.body("nist_part_" + record["part"]).id for record in records}
    contacts = []
    for contact in prepared_data.contact:
        geoms = [int(geom) for geom in contact.geom]
        names = [prepared.geom(geom).name for geom in geoms]
        body_ids = [int(prepared.geom_bodyid[geom]) for geom in geoms]
        if "workbench_top" in names and any(body_id in target_bodies for body_id in body_ids):
            contacts.append({"geoms": names, "penetration_m": float(max(0., -contact.dist))})
    maximum = max((record["penetration_m"] for record in contacts), default=0.)
    if maximum > PENETRATION_LIMIT_M:
        raise RuntimeError(f"Initial part/table penetration exceeds 0.2 mm: {maximum} m")
    audit = {"schema_version": 1, "external_source": str(SOURCE),
             "external_source_sha256": source_hash, "prepared_scene": str(scene_path),
             "prepared_scene_sha256": _sha256(scene_path), "part_count": len(records),
             "all_14_visual_cad_matches": all(record["visual_cad_exact_match"] for record in records),
             "all_other_qpos_bitwise_preserved": True,
             "only_changed_qpos_index": usb_qadr + 2,
             "usb_source_mesh_before": old_usb_asset,
             "usb_collision_attributes_before": old_usb_collision,
             "usb_world_T_cad_before": usb_pose_before.tolist(),
             "usb_world_T_cad_after": _pose(prepared, prepared_data, USB_NAME).tolist(),
             "usb_support_method": "Minimum world Z of every native CAD vertex under preserved initial rotation; retain 1 mm clearance above world Z=0 table.",
             "usb_native_vertex_count": len(vertices), "usb_support_clearance_m": SUPPORT_CLEARANCE_M,
             "maximum_initial_part_table_penetration_m": maximum,
             "initial_part_table_contacts": contacts, "penetration_limit_m": PENETRATION_LIMIT_M,
             "parts": records,
             "caveats": ["Original USB-with-cable shape replaced by the native catalog USB for this explicitly known-CAD condition.",
                         "Only USB body height and its objects_centered keyframe Z changed; all other state entries, parts, fixtures, board and robot are preserved.",
                         "USB contact geometry uses MuJoCo's convex mesh collision, not connector mating or cable physics.",
                         "This input audit does not validate grasping, collision-free paths or assembly."]}
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    if _sha256(SOURCE) != source_hash:
        raise RuntimeError("External source changed during scene preparation.")
    return _verify_existing(scene_path, audit_path)


if __name__ == "__main__":
    print(prepare_scene(), flush=True)
