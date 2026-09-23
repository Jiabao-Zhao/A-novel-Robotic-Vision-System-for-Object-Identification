"""Industrial workbench preview using existing CAD assets; no association policy."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .nist_task_board_1 import prepare_assets, _color, _values


ROOT = Path(__file__).resolve().parents[1]
# Designed simulation dimensions, not a physical camera/robot calibration.
TABLE_SIZE_M = (1.30, 1.00, .05)
TABLE_CENTER_XY_M = (-.30, 0.)
TABLE_TOP_Z_M = 0.
FLOOR_Z_M = -.85
NIST_PARTS = (
    "Gear_Large", "Gear_Medium", "Gear_Small", "Waterproof_Male",
    "M8_Hex_Nut", "M12_Hex_Nut", "M16_Hex_Nut", "DSUB_Male",
    "RGOCG16-50_16mm", "KET16_Square_16mm",
)
PARTS = (*NIST_PARTS, "lmo_drill", "lmo_glue", "ycbv_power_drill")


def prepare_catalog():
    """Read existing CADs without regenerating historical experiment assets."""
    from .benchmark_products import prepare_products

    mesh_dir, dimensions = prepare_assets()
    catalog = {}
    for name in NIST_PARTS:
        catalog[name] = {
            "cad_path": str(mesh_dir / f"{name}_m.stl"), "scale_to_m": 1.,
            "extent_m": dimensions[name], "source": "official NIST ATB1 STL",
            "visual_rgba": list(map(float, _color(name).split())),
        }
        if name.startswith("Gear_"):
            catalog[name]["collision_model"] = "separate_lower_section_and_hub"
    catalog.update(prepare_products())
    for name, record in catalog.items():
        record["cad_sha256"] = hashlib.sha256(Path(record["cad_path"]).read_bytes()).hexdigest()
        record["cad_up_axis"] = "Z"
        record.setdefault("description", name.replace("_", " "))
    return {name: catalog[name] for name in PARTS}


def preview_placements():
    """Separated inspection grid only; not a randomized benchmark split."""
    placements = {name: {"xy_m": [-.255 + .11 * (i // 4), -.165 + .11 * (i % 4)],
                   "yaw_deg": (-25., 15., 40., -10.)[i % 4]}
            for i, name in enumerate(PARTS)}
    # Preserve native product dimensions and separate the three larger footprints.
    placements.update(lmo_drill={"xy_m": [.12, -.24], "yaw_deg": 0.},
                      lmo_glue={"xy_m": [.12, 0.], "yaw_deg": 0.},
                      ycbv_power_drill={"xy_m": [.12, .25], "yaw_deg": 0.})
    return placements


def add_workbench(model):
    """Keep tabletop at old support Z=0; place the actual floor 850 mm below."""
    floor = model.worldbody.find(".//geom[@name='floor']")
    if floor is None:
        raise ValueError("Expected the existing LIBERO floor plane.")
    floor.set("pos", _values([0, 0, FLOOR_Z_M]))
    floor.attrib.pop("material", None)
    floor.set("rgba", ".29 .31 .33 1")
    for wall in model.worldbody.findall("./geom"):
        if wall.get("name", "").startswith("wall_"):
            position = np.fromstring(wall.get("pos"), sep=" ")
            position[2] += FLOOR_Z_M
            wall.set("pos", _values(position))
    for body in list(model.worldbody):
        if body.get("name", "") == "nist_board" or body.get("name", "").startswith("nist_fixture_"):
            model.worldbody.remove(body)
    ET.SubElement(model.asset, "material", name="workbench_matte", rgba=".46 .49 .48 1",
                  specular=".08", shininess=".05", reflectance="0")
    ET.SubElement(model.worldbody, "light", name="workbench_task_light",
                  pos="-.30 0 1.5", dir="0 0 -1", directional="true",
                  ambient=".15 .15 .15", diffuse=".45 .45 .45",
                  specular=".1 .1 .1", castshadow="false")
    bench = ET.SubElement(model.worldbody, "body", name="industrial_workbench")
    ET.SubElement(bench, "geom", name="workbench_top", type="box", group="1",
                  size=_values(np.array(TABLE_SIZE_M) / 2),
                  pos=_values([*TABLE_CENTER_XY_M, -.025]), material="workbench_matte",
                  friction=".8 .005 .0001", priority="1",
                  solref=".006 1", solimp=".99 .99 .001")
    def box(name, pos, half_size, color):
        ET.SubElement(bench, "geom", name=name, type="box", group="1",
                      pos=_values(pos), size=_values(half_size), rgba=color)
    steel = ".12 .20 .25 1"
    for i, x in enumerate((-.88, .28)):
        for j, y in enumerate((-.43, .43)):
            box(f"bench_leg_{i}_{j}", [x, y, -.45], [.035, .035, .40], steel)
    for i, y in enumerate((-.43, .43)):
        box(f"bench_crossbar_{i}", [-.30, y, -.65], [.58, .025, .035], steel)
    box("bench_lower_shelf", [-.30, 0, -.68], [.57, .40, .012], ".27 .31 .34 1")
    # Storage stays outside the part inspection area and wrist field of view.
    for i, x in enumerate((-.73, -.50, -.27)):
        box(f"storage_bin_{i}", [x, .30, -.59], [.09, .075, .075], ".10 .24 .37 1")
    # A separate presentation camera; the wrist sensor remains the perception input.
    eye, target = np.array([1.55, -1.90, 1.65]), np.array([-.30, 0, -.05])
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(model.worldbody, "camera", name="workbench_preview", mode="fixed",
                  pos=_values(eye), xyaxes=_values(np.r_[x, y]), fovy="45")


def configure_scene(model, catalog, placements):
    from scipy.spatial.transform import Rotation
    import open3d as o3d

    add_workbench(model)
    # Lay both long pins on their sides. Other parts use their existing Z-up state.
    for name in ("RGOCG16-50_16mm", "KET16_Square_16mm"):
        rotation = (Rotation.from_euler("z", placements[name]["yaw_deg"], degrees=True)
                    * Rotation.from_euler("y", 90, degrees=True))
        mesh = o3d.io.read_triangle_mesh(catalog[name]["cad_path"])
        vertices = np.asarray(mesh.vertices) @ rotation.as_matrix().T
        body = model.worldbody.find(f"./body[@name='nist_part_{name}']")
        body.set("quat", _values(rotation.as_quat()[[3, 0, 1, 2]]))
        body.set("pos", _values([*placements[name]["xy_m"], -vertices[:, 2].min() + .001]))
