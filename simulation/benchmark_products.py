"""Prepare the three selected BOP products at their published metric scale."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/cache/bop_model_catalogs"
ASSETS = ROOT / "outputs/simulation/industrial_workbench/assets/bop_products"
PRODUCTS = {
    "lmo_drill": ("lmo", 8, "LM-O drill"),
    "lmo_glue": ("lmo", 11, "LM-O glue bottle"),
    "ycbv_power_drill": ("ycbv", 15, "YCB-V power drill"),
}


def vertex_color_texture(mesh):
    """Bake interpolated PLY vertex colors into padded per-triangle UV tiles."""
    import open3d as o3d

    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)[faces]
    side = int(np.ceil(np.sqrt(len(faces))))
    tile = 8
    yy, xx = np.mgrid[:tile, :tile]
    weights = np.stack([1 - (xx - 1) / 5 - (yy - 1) / 5,
                        (xx - 1) / 5, (yy - 1) / 5], axis=-1).clip(0, 1)
    weights /= weights.sum(-1, keepdims=True)
    image = np.zeros((side * tile, side * tile, 3), dtype=np.uint8)
    uv = np.empty((len(faces), 3, 2))
    for i, face_colors in enumerate(colors):
        x, y = (i % side) * tile, (i // side) * tile
        image[y:y + tile, x:x + tile] = np.rint(weights @ face_colors * 255).astype(np.uint8)
        uv[i, :, 0] = (x + np.array([1.5, 6.5, 1.5])) / image.shape[1]
        uv[i, :, 1] = 1 - (y + np.array([1.5, 1.5, 6.5])) / image.shape[0]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv.reshape(-1, 2))
    return o3d.geometry.Image(image)


def prepare_products():
    import open3d as o3d
    from scripts.show_bop_model_catalogs import texture_coordinates

    catalog = {}
    for name, (dataset, object_id, description) in PRODUCTS.items():
        source = CACHE / dataset / "models" / f"obj_{object_id:06d}.ply"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        output = ASSETS / name
        record_path = output / "provenance.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("preparation_version") == 2 and record["source_sha256"] == source_hash and all(
                    Path(record[key]).exists() for key in ("cad_path", "visual_mesh_path", "texture_path")):
                catalog[name] = record
                continue
        output.mkdir(parents=True, exist_ok=True)
        mesh = o3d.io.read_triangle_mesh(str(source))
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]) if name == "lmo_drill" else np.eye(3)
        vertices = np.asarray(mesh.vertices) * .001 @ rotation.T
        center = (vertices.min(0) + vertices.max(0)) / 2
        mesh.vertices = o3d.utility.Vector3dVector(vertices - center)
        mesh.compute_vertex_normals()
        if dataset == "ycbv":
            uv, pixels = texture_coordinates(source, len(vertices))
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uv[np.asarray(mesh.triangles)].reshape(-1, 2))
            texture = o3d.geometry.Image(np.ascontiguousarray(pixels))
            texture_policy = "Original BOP UV texture retained"
        else:
            texture = vertex_color_texture(mesh)
            texture_policy = "Original BOP vertex colors baked to interpolated UV tiles; no recoloring"
        mesh.textures = [texture]
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh.triangles), dtype=np.int32))
        with TemporaryDirectory(prefix="bop_product_") as temporary:
            temp = Path(temporary)
            for filename in ("visual.obj", "geometry.stl"):
                if not o3d.io.write_triangle_mesh(str(temp / filename), mesh):
                    raise RuntimeError(f"Cannot export {name}: {filename}")
            for path in temp.iterdir():
                (output / path.name).write_bytes(path.read_bytes())
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -center
        record = {"cad_path": str(output / "geometry.stl"), "visual_mesh_path": str(output / "visual.obj"),
                  "texture_path": str(output / "visual_0.png"), "scale_to_m": 1.,
                  "extent_m": np.ptp(vertices, axis=0).tolist(), "cad_up_axis": "Z",
                  "visual_rgba": [1., 1., 1., 1.], "collision_model": "convex_mesh",
                  "description": description, "source": f"Official BOP {dataset} object {object_id}",
                  "source_path": str(source), "source_sha256": source_hash,
                  "source_download": json.loads((CACHE / dataset / "download.json").read_text())["url"],
                  "source_units": "mm", "source_scale_to_m": .001, "preparation_version": 2,
                  "cad_T_source_m": transform.tolist(), "texture_policy": texture_policy,
                  "scale_policy": "Millimetres converted to metres; no fit-to-cell scaling"}
        if name == "lmo_glue":
            record["description_view_policy"] = {
                "support": "upright on base, nozzle up; no tip-down placement",
                "suggested_views": ["front", "side", "upper oblique"],
                "excluded_views": ["bottom"], "operator_approved_view_count": None,
            }
        record_path.write_text(json.dumps(record, indent=2))
        catalog[name] = record
    return catalog
