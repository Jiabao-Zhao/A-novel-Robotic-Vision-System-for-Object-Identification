"""Prepare textured tool assets and matching metric geometry for the workbench."""

import hashlib
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import zipfile

import numpy as np


ASSETS = Path(__file__).resolve().parents[1] / "outputs/simulation/industrial_workbench/assets"
SOURCES = {
    "drill": {
        "archive": "035_power_drill_google_16k.tgz", "member": "035_power_drill",
        "url": "https://ycb-benchmarks.s3.amazonaws.com/data/google/035_power_drill_google_16k.tgz",
        "source": "YCB 035_power_drill Google 16k scanned model", "license": "CC BY 4.0",
        "attribution": "Yale-CMU-Berkeley Object and Model Set, Calli et al.",
    },
    "screwdriver": {
        "archive": "043_phillips_screwdriver_google_16k.tgz", "member": "043_phillips_screwdriver",
        "url": "https://ycb-benchmarks.s3.amazonaws.com/data/google/043_phillips_screwdriver_google_16k.tgz",
        "source": "YCB 043_phillips_screwdriver Google 16k scanned model", "license": "CC BY 4.0",
        "attribution": "Yale-CMU-Berkeley Object and Model Set, Calli et al.",
    },
    "tape_measure": {
        "archive": "tool_pack_1.zip", "member": "tape_measure.glb",
        "url": "https://opengameart.org/sites/default/files/tool_pack_1.zip",
        "source": "Tool Pack 1 tape measure; authored asset, not a measured physical part",
        "license": "CC0", "attribution": "LonesomeDucky, https://opengameart.org/content/tool-pack-1",
    },
}


def prepare_tools():
    import open3d as o3d

    ASSETS.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for name, source in SOURCES.items():
        output = ASSETS / name
        output.mkdir(exist_ok=True)
        record_path = output / "provenance.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if all(Path(record[key]).is_file() for key in ("cad_path", "visual_mesh_path", "texture_path")):
                catalog[name] = record
                continue
        archive = ASSETS / source["archive"]
        if not archive.exists():
            urlretrieve(source["url"], archive)
        if name == "tape_measure":
            with zipfile.ZipFile(archive) as files:
                path = output / "source.glb"
                path.write_bytes(files.read(source["member"]))
            model = o3d.io.read_triangle_model(str(path))
            if len(model.meshes) != 1 or len(model.materials) != 1:
                raise ValueError("Expected one tape-measure mesh and material.")
            mesh = model.meshes[0].mesh
            texture = model.materials[0].albedo_img
        else:
            with tarfile.open(archive) as files:
                for filename in ("textured.obj", "textured.mtl", "texture_map.png"):
                    data = files.extractfile(f"{source['member']}/google_16k/{filename}").read()
                    if filename == "textured.mtl":
                        # Some YCB MTL lines have trailing blanks in the texture filename.
                        data = ("\n".join(line.rstrip() for line in data.decode().splitlines()) + "\n").encode()
                    (output / filename).write_bytes(data)
            mesh = o3d.io.read_triangle_mesh(str(output / "textured.obj"))
            texture = o3d.io.read_image(str(output / "texture_map.png"))
        vertices = np.asarray(mesh.vertices)
        rotation = np.eye(3)
        if name == "screwdriver":
            # Align its long axis with X without changing physical size or UVs.
            _, axes = np.linalg.eigh(np.cov(vertices[:, :2].T))
            angle = -np.arctan2(axes[1, -1], axes[0, -1])
            rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        rotated = vertices @ rotation.T
        center = (rotated.min(0) + rotated.max(0)) / 2
        mesh.vertices = o3d.utility.Vector3dVector(rotated - center)
        mesh.compute_vertex_normals()
        mesh.textures = [texture]
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh.triangles), dtype=np.int32))
        visual, geometry = output / "visual.obj", output / "geometry.stl"
        # Open3D flushes OBJ lines individually; write on Linux before copying
        # complete files to the Windows-mounted workspace.
        with TemporaryDirectory(prefix="workbench_tool_") as temporary:
            temp = Path(temporary)
            if not o3d.io.write_triangle_mesh(str(temp / visual.name), mesh):
                raise RuntimeError(f"Cannot write textured tool: {name}")
            if not o3d.io.write_triangle_mesh(str(temp / geometry.name), mesh):
                raise RuntimeError(f"Cannot write tool geometry: {name}")
            for path in temp.iterdir():
                (output / path.name).write_bytes(path.read_bytes())
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, -center
        record = {**source, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "cad_path": str(geometry), "visual_mesh_path": str(visual),
            "texture_path": str(output / "visual_0.png"),
            "scale_to_m": 1., "cad_up_axis": "Z", "extent_m": mesh.get_axis_aligned_bounding_box().get_extent().tolist(),
            "cad_T_source": transform.tolist(), "visual_rgba": [1., 1., 1., 1.],
            "collision_model": "convex_mesh", "description": name.replace("_", " "),
            "texture_policy": "Original supplied base-color texture; no recoloring",
            "scale_policy": "Native metre coordinates retained; no fitting to grid cells"}
        record_path.write_text(json.dumps(record, indent=2))
        catalog[name] = record
    return catalog
