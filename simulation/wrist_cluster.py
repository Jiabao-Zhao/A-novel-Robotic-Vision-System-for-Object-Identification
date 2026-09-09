"""Scattered pickup scene with NIST and representative CAD."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .nist_task_board_1 import (NistTaskBoardEnvironment, prepare_assets, _values,
                              BOARD_CENTER_XY_M, BOARD_SIZE_M, FLOOR_Z_M)


CAMERA = "robot0_eye_in_hand"
SEED = 20260909
ROOT = Path(__file__).resolve().parents[1] / "outputs/simulation/wrist_cluster"
NIST_PARTS = {
    "Gear_Large": "large gear", "Gear_Medium": "medium gear",
    "M12_Hex_Nut": "M12 hex nut", "M16_Hex_Nut": "M16 hex nut",
    "DSUB_Male": "D-sub connector", "RGOCG16-50_16mm": "round peg",
    "KET16_Square_16mm": "white square peg",
}
COMMON = ("bbq_sauce", "cream_cheese")
ADDED = ("bearing", "pulley", "spacer", "red_block", "blue_block")
DESCRIPTIONS = {**NIST_PARTS, **{name: name.replace("_", " ") for name in (*COMMON, *ADDED)}}
WORKSPACE_MIN = (-.34, -.37, -.01)
WORKSPACE_MAX = (.04, .37, .17)
# Fixed workcell layout prior, not a live object pose or simulator segmentation.
BOARD_EXCLUSION_XY = np.array([BOARD_CENTER_XY_M - BOARD_SIZE_M[:2]/2 - .006,
                               BOARD_CENTER_XY_M + BOARD_SIZE_M[:2]/2 + .006])
MIN_OBJECT_GAP_M = .012
MIN_BOARD_GAP_M = .045
OBSERVATION_CAMERA_POSITION_M = (-.09, 0., .585)


def ring_mesh(inner, outer, height, segments=64):
    """Closed annular CAD mesh in meters, centered at the origin."""
    import open3d as o3d
    vertices = [[r * np.cos(a), r * np.sin(a), z]
                for z in (-height / 2, height / 2) for r in (inner, outer)
                for a in np.linspace(0, 2 * np.pi, segments, endpoint=False)]
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        for a, b, c, d in ((i, j, segments + j, segments + i),
                           (2*segments+i, 3*segments+i, 3*segments+j, 2*segments+j),
                           (i, 2*segments+i, 2*segments+j, j),
                           (segments+i, segments+j, 3*segments+j, 3*segments+i)):
            faces.extend(((a, b, c), (a, c, d)))
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(faces))


def prepare_catalog():
    import open3d as o3d
    from .libero_cad import retrieve_libero_cad
    asset_dir = ROOT / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir, dimensions = prepare_assets()
    catalog = {}
    for name in NIST_PARTS:
        catalog[name] = {"cad_path": str(mesh_dir / f"{name}_m.stl"), "scale_to_m": 1.,
                         "cad_up_axis": "Z", "extent_m": dimensions[name],
                         "source": "official NIST ATB1 mesh", "max_registration_rmse_m": .004}
        if name in ("Gear_Large", "Gear_Medium"):
            catalog[name]["collision_model"] = "separate_lower_section_and_hub"
            catalog[name]["grasp_yaw_free"] = True
        if name == "KET16_Square_16mm":
            catalog[name]["visual_rgba"] = [.95, .95, .95, 1.]
    for name in COMMON:
        record = retrieve_libero_cad(name)
        mesh = o3d.io.read_triangle_mesh(record["cad_path"])
        if record["cad_up_axis"] != "Z":
            raise ValueError("The wrist scene's common assets require CAD Z-up support.")
        record["extent_m"] = (np.ptp(np.asarray(mesh.vertices), axis=0) * record["scale_to_m"]).tolist()
        record["cad_support_z_m"] = float(mesh.get_min_bound()[2] * record["scale_to_m"])
        catalog[name] = record
    for name in ADDED:
        source = "procedural representative geometry; not an official NIST part"
        if name.endswith("block"):
            mesh = o3d.geometry.TriangleMesh.create_box(.035, .035, .035).translate([-.0175]*3)
        else:
            inner, outer, height = {"bearing": (.012, .027, .016), "spacer": (.006, .014, .020),
                                    "pulley": (.008, .030, .020)}[name]
            mesh = ring_mesh(inner, outer, height)
            if name == "pulley":
                # Flanges and a recessed belt track; no belt dynamics are claimed.
                mesh = ring_mesh(inner, outer-.004, height)
                for z in (-.008, .008):
                    mesh += ring_mesh(outer-.004, outer, .004).translate([0, 0, z])
        mesh.compute_triangle_normals()
        path = asset_dir / f"{name}.stl"
        if not o3d.io.write_triangle_mesh(str(path), mesh):
            raise RuntimeError(f"Cannot save representative CAD: {name}")
        catalog[name] = {"cad_path": str(path), "scale_to_m": 1., "cad_up_axis": "Z",
                         "extent_m": np.ptp(np.asarray(mesh.vertices), axis=0).tolist(),
                         "source": source,
                         "max_registration_rmse_m": .004}
    for name, record in catalog.items():
        record.update(description=DESCRIPTIONS[name],
                      cad_sha256=hashlib.sha256(Path(record["cad_path"]).read_bytes()).hexdigest())
    (asset_dir / "catalog.json").write_text(json.dumps(catalog, indent=2))
    return catalog


def random_placements(catalog, seed=SEED):
    """Seeded space-filling scatter around the board, inside the pickup workspace."""
    rng = np.random.default_rng(seed)
    radii = {name: float(np.linalg.norm(np.asarray(record["extent_m"])[:2]) / 2)
             for name, record in catalog.items()}
    placements = {}
    for name in sorted(catalog, key=radii.get, reverse=True):
        radius = radii[name]
        candidates = rng.uniform(np.array(WORKSPACE_MIN[:2]) + radius,
                                 np.array(WORKSPACE_MAX[:2]) - radius, size=(5000, 2))
        nearest_board = np.clip(candidates, *BOARD_EXCLUSION_XY)
        candidates = candidates[np.linalg.norm(candidates-nearest_board, axis=1) >= radius + MIN_BOARD_GAP_M]
        if not len(candidates):
            raise RuntimeError(f"Cannot fit {name} in the declared cluster workspace.")
        if placements:
            clearance = np.min(np.column_stack([
                np.linalg.norm(candidates-entry["xy_m"], axis=1) - radius - radii[other]
                for other, entry in placements.items()]), axis=1)
            index = int(np.argmax(clearance))
            if clearance[index] < MIN_OBJECT_GAP_M:
                raise RuntimeError(f"Cannot scatter {name} with the required object clearance.")
            xy = candidates[index]
        else:
            xy = candidates[0]
        placements[name] = {"xy_m": xy.tolist(), "yaw_deg": float(rng.uniform(-180, 180))}
    return placements


def replace_gear_collision(model, name, cad_path):
    """Preserve the CAD rim/hub step, keeping the gear one rigid body."""
    import open3d as o3d
    from scipy.spatial import ConvexHull

    vertices = np.asarray(o3d.io.read_triangle_mesh(str(cad_path)).vertices)
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    hub_radius = max(radius[vertices[:, 2] > 1e-6])
    pieces = (vertices[vertices[:, 2] <= 1e-6],
              vertices[(vertices[:, 2] >= -1e-6) & (radius <= hub_radius + 1e-6)])
    body = model.worldbody.find(f"./body[@name='nist_part_{name}']")
    original = body.find(f"./geom[@name='nist_part_{name}_collision']")
    attributes = dict(original.attrib)
    # Retain the original mesh solely for its nominal mass and inertia.
    # It must not contribute the artificial sloping collision surface.
    original.set("name", f"nist_part_{name}_inertial_geometry")
    original.set("contype", "0")
    original.set("conaffinity", "0")
    for label, points in zip(("lower", "hub"), pieces):
        key = f"nist_part_{name}_{label}_collision"
        hull = points[ConvexHull(points).vertices]
        ET.SubElement(model.asset, "mesh", name=key, vertex=_values(hull.ravel()))
        ET.SubElement(body, "geom", **{**attributes, "name": key, "mesh": key, "mass": "0"})


def make_environment(catalog, placements, image_size=768):
    common_parts = tuple(name for name, record in catalog.items() if "asset_relative_path" in record)
    added_parts = tuple(name for name in catalog if name not in (*NIST_PARTS, *common_parts))
    def add_shapes(model):
        for name, record in catalog.items():
            if record.get("collision_model") == "separate_lower_section_and_hub":
                replace_gear_collision(model, name, record["cad_path"])
            if "visual_rgba" in record:
                visual = model.worldbody.find(f"./body[@name='nist_part_{name}']/geom[@name='nist_part_{name}_visual']")
                visual.set("rgba", _values(record["visual_rgba"]))
        # Use the same upright support assumption as CAD alignment and scatter
        # footprints. LIBERO's BBQ preset instead rotates the bottle onto its side.
        for name in common_parts:
            if "cad_support_z_m" not in catalog[name]:
                continue  # Historical catalogs retain their original LIBERO preset.
            body = model.worldbody.find(f"./body[@name='nist_part_{name}']")
            placement = placements[name]
            angle = np.deg2rad(placement["yaw_deg"]) / 2
            body.set("quat", _values([np.cos(angle), 0, 0, np.sin(angle)]))
            body.set("pos", _values([*placement["xy_m"],
                FLOOR_Z_M - catalog[name]["cad_support_z_m"] + .001]))
        for name in added_parts:
            extent = np.asarray(catalog[name]["extent_m"])
            placement = placements[name]
            angle = np.deg2rad(placement["yaw_deg"]) / 2
            body = ET.SubElement(model.worldbody, "body", name=f"nist_part_{name}",
                pos=_values([*placement["xy_m"], extent[2]/2 + .001]),
                quat=_values([np.cos(angle), 0, 0, np.sin(angle)]))
            ET.SubElement(body, "freejoint", name=f"{name}_joint")
            ET.SubElement(model.asset, "mesh", name=name, file=catalog[name]["cad_path"])
            color = {"red_block": ".85 .025 .025 1", "blue_block": ".025 .12 .85 1",
                     "bearing": ".63 .67 .72 1", "pulley": ".28 .30 .32 1"}.get(name, ".55 .59 .63 1")
            ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh", mesh=name,
                          rgba=color, contype="0", conaffinity="0", group="1", mass="0")
            common = {"group": "0", "density": "1800", "friction": "0.8 0.005 0.0001"}
            if name == "cable_shark_device":
                # Preserve replay of historical catalogs that included this object.
                ET.SubElement(body, "geom", name=f"{name}_collision", type="mesh",
                              mesh=name, **common)
            elif name in ("bearing", "pulley", "spacer"):
                inner, outer = {"bearing": (.012, .027), "pulley": (.008, .030), "spacer": (.006, .014)}[name]
                # Convex wedges preserve the center opening in collision geometry.
                for i in range(32):
                    angles = 2 * np.pi * np.array([i, i+1]) / 32
                    vertices = [[r*np.cos(a), r*np.sin(a), z]
                                for z in (-extent[2]/2, extent[2]/2) for r in (inner, outer) for a in angles]
                    key = f"{name}_wedge_{i}"
                    ET.SubElement(model.asset, "mesh", name=key, vertex=_values(np.ravel(vertices)))
                    ET.SubElement(body, "geom", name=key, type="mesh", mesh=key, **common)
            elif name == "l_bracket":
                for i, (size, pos) in enumerate((((.030, .020, .004), (0, 0, -.016)),
                                               ((.004, .020, .016), (-.026, 0, .004)))):
                    ET.SubElement(body, "geom", name=f"{name}_box_{i}", type="box",
                                  size=_values(size), pos=_values(pos), **common)
            else:
                ET.SubElement(body, "geom", name=f"{name}_collision", type="box",
                              size=_values(extent/2), **common)
    original_names = (*NIST_PARTS, *common_parts)
    env = NistTaskBoardEnvironment(parts=tuple(NIST_PARTS), common_parts=common_parts,
        placements={name: placements[name] for name in original_names},
        camera_names=(CAMERA,), additional_scene=add_shapes, image_size=image_size)
    env.active_parts = tuple(catalog)
    env.task_name = "wrist_only_scattered_object_pickup"
    return env
