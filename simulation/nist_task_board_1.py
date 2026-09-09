"""Official NIST ATB1 meshes in a custom LIBERO perception scene.

Run in the existing WSL LIBERO environment. This is not an assembly benchmark:
removable parts use convex collision hulls, without insertion/threading models.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .libero_env import LiberoIntegrationError, LiberoTaskEnvironment, configure_mujoco_rendering


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/simulation/nist_task_board_1"
ASSET_DIR = OUTPUT_DIR / "assets"
STL_URL = "https://www.nist.gov/document/taskboard1stlzip"
BOARD_FILE = "GMC_Laser_Plate_Virtual"
BOARD_SIZE_M = np.array([0.384, 0.384, 0.0089916])
# Scene placement, in the simulated world frame; not a physical calibration.
BOARD_CENTER_XY_M = np.array([0.0, 0.065])
# The actual EmptyArena plane is at world Z=0. LIBERO's z_offset=-0.025
# belongs to its placement sampler and is not the physical floor elevation.
FLOOR_Z_M = 0.0
BOARD_BOTTOM_Z_M = FLOOR_Z_M + 0.020

# Centers measured from the official plate STL, in its millimetre XY frame.
FIXTURES = (
    ("Gear_Plate", 171.45, 47.176, 0.0, -90),
    ("Gear_Shaft", 141.45, 47.176, 5.0, 0),
    ("Gear_Shaft", 191.45, 47.176, 5.0, 0),
    ("Gear_Shaft", 221.45, 47.176, 5.0, 0),
    ("M4_Screw", 191.45, 347.176, -7.0, 0),
    ("M8_Screw", 41.45, 47.176, -9.0, 0),
    ("M12_Screw", 41.45, 122.176, -11.0, 0),
    ("M16_Screw", 341.45, 47.176, -16.0, 0),
    ("BNC_Female", 191.45, 279.676, -2.85, 0),
    ("USB_Female", 266.45, 197.176, 0.0, 0),
    ("RJ45_Housing", 341.45, 347.176, 0.0, 0),
    ("RJ45_Female", 341.45, 347.176, 2.0, 0),
    ("DB_Housing", 41.45, 272.176, 0.0, -90),
    ("DSUB_Female", 41.45, 272.176, 9.0, -90),
    ("Waterproof_Female", 116.45, 197.176, 0.0, 0),
)
PARTS = (
    "Gear_Large", "Gear_Medium", "Gear_Small", "M16_Hex_Nut", "M12_Hex_Nut",
    "M8_Hex_Nut", "M4_Hex_Nut", "BNC_Male", "USB_Male", "RJ45_Male",
    "RGOCG16-50_16mm", "RGOCG12-50_12mm", "RGOCG8-50_8mm", "RGOCG4-50_Round_4mm",
    "KET16_Square_16mm", "KET12_Square_12mm", "KET8_Square_8mm", "KET4_Square_4mm",
    "Waterproof_Male", "DSUB_Male",
)


def prepare_assets():
    """Convert the official millimetre STLs to centered metre binary STLs."""
    archive = ASSET_DIR / "stl.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Download {STL_URL} to {archive}; see simulation/README.md.")
    mesh_dir = ASSET_DIR / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    names = {BOARD_FILE, *PARTS, *(item[0] for item in FIXTURES)}
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    provenance_path = ASSET_DIR / "provenance.json"
    if provenance_path.exists():
        cached = json.loads(provenance_path.read_text())
        if (cached.get("conversion_revision") == 1 and cached["archive_sha256"] == archive_hash
                and all((mesh_dir / f"{name}_m.stl").is_file() for name in names)):
            return mesh_dir, cached["dimensions_m"]
    import open3d as o3d

    dimensions = {}
    with zipfile.ZipFile(archive) as source:
        for name in sorted(names):
            stl_path = mesh_dir / f"{name}.STL"
            stl_path.write_bytes(source.read(f"{name}.STL"))
            mesh = o3d.io.read_triangle_mesh(str(stl_path))
            mesh.remove_duplicated_vertices()
            vertices = np.asarray(mesh.vertices)
            if not len(vertices) or not np.all(np.isfinite(vertices)):
                raise ValueError(f"Invalid NIST mesh: {name}")
            # Most assembled component exports point along -Z. The separately
            # exported pegs, waterproof and DSUB parts use their own local frame.
            if name != BOARD_FILE and not name.startswith(("KET", "RGOCG", "Waterproof", "DSUB")):
                vertices[:, 1:] *= -1
            extent = np.ptp(vertices, axis=0) * 0.001
            vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            vertices *= 0.001
            dimensions[name] = extent.tolist()
            # MuJoCo's binary STL decoder accepts at most 200,000 triangles.
            if len(mesh.triangles) > 200000:
                mesh = mesh.simplify_quadric_decimation(190000)
            mesh.compute_triangle_normals()
            if not o3d.io.write_triangle_mesh(str(mesh_dir / f"{name}_m.stl"), mesh):
                raise RuntimeError(f"Could not write converted mesh: {name}")
    np.testing.assert_allclose(dimensions[BOARD_FILE], BOARD_SIZE_M, atol=1e-6)
    provenance = {
        "source_url": STL_URL,
        "archive_sha256": archive_hash, "conversion_revision": 1,
        "source_units": "mm", "mesh_units": "m", "dimensions_m": dimensions,
        "fixture_xy_source": "mounting-hole centers measured from the official plate STL",
        "fixture_z_note": "scene mounting approximations; fastener engagement is not validated",
        "collision_model": "plate triangle prisms; convex hulls for components",
        "mesh_simplification": "DSUB male reduced to 190000 triangles for MuJoCo STL limit",
        "limitations": ["no insertion or threading validation", "no cables or connector latch physics",
                        "no official NIST task success metric", "uniform illustrative materials",
                        "STL pilot holes are preserved; no post-fabrication drilling is modeled"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))
    return mesh_dir, dimensions


def _values(values):
    return " ".join(f"{value:.9g}" for value in values)


def _color(name):
    if name == BOARD_FILE:
        return "0.90 0.90 0.85 1"
    if name in {"Gear_Large", "Gear_Medium", "Gear_Small"}:
        return "0.92 0.92 0.88 1"
    if "Housing" in name or name in {"USB_Female", "USB_Male", "DSUB_Female", "DSUB_Male"}:
        return "0.09 0.10 0.12 1"
    if "Waterproof" in name:
        return "0.68 0.70 0.57 1"
    return "0.52 0.57 0.61 1"


def add_board_scene(model, mesh_dir, dimensions, parts=PARTS, placements=None):
    """Add fixed hardware and the selected movable parts before compilation."""
    import open3d as o3d

    for name in dimensions:
        ET.SubElement(model.asset, "mesh", name=f"nist_{name}", file=str(mesh_dir / f"{name}_m.stl"))
    board_height = dimensions[BOARD_FILE][2]
    board_top = BOARD_BOTTOM_Z_M + board_height
    board = ET.SubElement(model.worldbody, "body", name="nist_board",
                          pos=_values([*BOARD_CENTER_XY_M, BOARD_BOTTOM_Z_M + board_height / 2]))
    ET.SubElement(board, "geom", name="nist_board_visual", type="mesh", mesh=f"nist_{BOARD_FILE}",
                  rgba=_color(BOARD_FILE), contype="0", conaffinity="0", group="1")
    # A single MuJoCo mesh collision would close all holes with its convex hull.
    # Extrude each bottom-surface triangle instead to preserve the plate openings.
    plate = o3d.io.read_triangle_mesh(str(mesh_dir / f"{BOARD_FILE}_m.stl"))
    vertices = np.asarray(plate.vertices)
    triangles = np.asarray(plate.triangles)
    bottom = vertices[:, 2].min()
    triangles = triangles[np.all(np.isclose(vertices[triangles, 2], bottom, atol=1e-8), axis=1)]
    for index, triangle in enumerate(triangles):
        lower = vertices[triangle].copy()
        if np.linalg.norm(np.cross(lower[1] - lower[0], lower[2] - lower[0])) < 1e-12:
            continue
        upper = lower + [0, 0, board_height]
        name = f"nist_plate_collision_{index}"
        ET.SubElement(model.asset, "mesh", name=name, vertex=_values(np.vstack((lower, upper)).ravel()))
        ET.SubElement(board, "geom", type="mesh", mesh=name, group="0", friction="0.6 0.005 0.0001")
    for x in (-0.175, 0.175):
        for y in (-0.175, 0.175):
            ET.SubElement(board, "geom", type="cylinder", size="0.003 0.01",
                          pos=_values([x, y, -board_height / 2 - 0.01]), rgba="0.5 0.55 0.6 1", group="1")

    def component(name, instance, position, yaw=0, movable=False):
        angle = np.deg2rad(yaw) / 2
        body = ET.SubElement(model.worldbody, "body", name=instance, pos=_values(position),
                             quat=_values([np.cos(angle), 0, 0, np.sin(angle)]))
        if movable:
            ET.SubElement(body, "freejoint", name=f"{instance}_joint")
        ET.SubElement(body, "geom", name=f"{instance}_visual", type="mesh", mesh=f"nist_{name}",
                      rgba=_color(name), contype="0", conaffinity="0", group="1", mass="0")
        ET.SubElement(body, "geom", name=f"{instance}_collision", type="mesh", mesh=f"nist_{name}",
                      group="0", density="1200" if "Gear" in name else "3000", friction="0.8 0.005 0.0001")

    for index, (name, x_mm, y_mm, base_mm, yaw) in enumerate(FIXTURES):
        xy = (np.array([x_mm, y_mm]) - 192) * 0.001 + BOARD_CENTER_XY_M
        component(name, f"nist_fixture_{index}",
                  [*xy, board_top + base_mm * 0.001 + dimensions[name][2] / 2], yaw)
    for index, name in enumerate(parts):
        row, column = divmod(index, 10)
        placement = (placements[name] if placements else
                     {"xy_m": [-0.28 + column * 0.064, -0.215 - row * 0.082], "yaw_deg": 0.})
        component(name, f"nist_part_{name}",
                  [*placement["xy_m"], FLOOR_Z_M + dimensions[name][2] / 2 + 0.001],
                  yaw=placement["yaw_deg"], movable=True)


class NistTaskBoardEnvironment(LiberoTaskEnvironment):
    """Reuse the RGB-D bridge and robot controls for a custom board scene."""

    def __init__(self, image_size=768, parts=PARTS, placements=None, overhead_view=False, common_parts=(),
                 camera_names=("agentview", "robot0_eye_in_hand"), additional_scene=None):
        if not parts or len(set(parts)) != len(parts) or not set(parts).issubset(PARTS):
            raise ValueError("NIST scene requires distinct known component names.")
        active_parts = (*parts, *common_parts)
        if len(set(active_parts)) != len(active_parts):
            raise ValueError("Scene component names must be distinct.")
        if placements is not None and set(placements) != set(active_parts):
            raise ValueError("Every selected NIST component needs one scenario placement.")
        self.active_parts = active_parts
        self.rendering_backend = configure_mujoco_rendering()
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem

        mesh_dir, dimensions = prepare_assets()
        self.part_dimensions_m = dimensions

        @register_problem
        class Nist_Task_Board_One(TASK_MAPPING["libero_floor_manipulation"]):
            def _load_model(self):
                super()._load_model()
                add_board_scene(self.model, mesh_dir, dimensions, parts, placements)
                if common_parts:
                    from libero.libero.envs.objects import get_object_fn
                    from scipy.spatial.transform import Rotation

                    for name in common_parts:
                        obj = get_object_fn(name)(name=f"mixed_{name}")
                        body = obj.get_obj()
                        body.set("name", f"nist_part_{name}")
                        placement = placements[name]
                        rotation = Rotation.identity()
                        axes = (obj.rotation if isinstance(obj.rotation, dict) else
                                {obj.rotation_axis: obj.rotation})
                        for axis, angles in axes.items():
                            if angles[0] != angles[1]:
                                raise ValueError("Mixed-scene assets require a fixed supplied base orientation.")
                            rotation = Rotation.from_euler(axis, angles[0]) * rotation
                        rotation = Rotation.from_euler("z", placement["yaw_deg"], degrees=True) * rotation
                        body.set("quat", _values(rotation.as_quat()[[3, 0, 1, 2]]))
                        body.set("pos", _values([*placement["xy_m"], FLOOR_Z_M - obj.bottom_offset[2] + .001]))
                        self.model.merge_assets(obj)
                        self.model.worldbody.append(body)
                if additional_scene is not None:
                    additional_scene(self.model)
                if overhead_view:
                    camera = self.model.worldbody.find(".//camera[@name='agentview']")
                    if camera is None:
                        raise ValueError("NIST classification camera is missing.")
                    camera.set("pos", "0 -0.065 1.4")
                    camera.set("quat", "1 0 0 0")
                    camera.set("fovy", "32")

            def _check_success(self):
                return False  # This capture scene defines no assembly success predicate.

        self.suite_name = "custom_nist_task_board_1"
        self.task_index = None
        self.task_name = "nist_task_board_1_perception"
        self.instruction = "Inspect the NIST Task Board 1 and its loose assembly components."
        self.camera_names = tuple(camera_names)
        self.image_width = self.image_height = int(image_size)
        self.last_observation = None
        self.last_reset_seed = self.last_init_state_index = self.last_init_state_sha256 = None
        self.control_mode = None
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(Path(__file__).parent / "assets/nist_task_board_1.bddl"),
            camera_names=list(self.camera_names), camera_heights=image_size, camera_widths=image_size,
            camera_depths=True, control_freq=20, hard_reset=False,
        )

    def _load_task_initial_states(self):
        raise LiberoIntegrationError("The custom NIST scene has no official LIBERO fixed states.")
