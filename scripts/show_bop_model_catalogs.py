"""Render inspection galleries from the official cached BOP model meshes."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/cache/bop_model_catalogs"
OUTPUT = ROOT / "outputs/bop_model_catalogs"
DATASETS = {"lmo": ("LM-O", 8), "tudl": ("TUD-L", 3), "icbin": ("IC-BIN", 2),
            "itodd": ("ITODD", 28), "hb": ("HB", 33), "ycbv": ("YCB-V", 21)}
HB_SUBSET = {1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33}
LMO_NAMES = {1: "Ape", 5: "Watering can", 6: "Cat", 8: "Drill", 9: "Duck",
             10: "Eggbox", 11: "Glue bottle", 12: "Hole punch"}
YCB_NAMES = ["Master chef can", "Cracker box", "Sugar box", "Tomato soup can",
             "Mustard bottle", "Tuna fish can", "Pudding box", "Gelatin box",
             "Potted meat can", "Banana", "Pitcher base", "Bleach cleanser",
             "Bowl", "Mug", "Power drill", "Wood block", "Scissors", "Large marker",
             "Large clamp", "Extra-large clamp", "Foam brick"]
EYES = ((2, -3, 1.8), (-2, 3, 1.8))


def texture_coordinates(path, vertex_count):
    """Read the per-vertex texture coordinates in the official ASCII YCB PLYs."""
    with path.open() as stream:
        header = []
        while True:
            line = stream.readline().strip()
            header.append(line)
            if line == "end_header":
                break
        assert "format ascii 1.0" in header
        begin = header.index(f"element vertex {vertex_count}") + 1
        properties = []
        for line in header[begin:]:
            if not line.startswith("property "):
                break
            properties.append(line.split()[-1])
        uv = np.loadtxt(stream, max_rows=vertex_count,
                        usecols=[properties.index("texture_u"), properties.index("texture_v")])
        filename = next(line.split()[-1] for line in header if line.startswith("comment TextureFile"))
    return uv, np.asarray(Image.open(path.parent / filename).convert("RGB"))


def render_views(path, textured):
    mesh = o3d.io.read_triangle_mesh(str(path))
    assert mesh.has_triangles()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    center = (vertices.min(0) + vertices.max(0)) / 2
    radius = np.linalg.norm(vertices - center, axis=1).max()
    normalized = ((vertices - center) / radius).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(normalized),
                        o3d.core.Tensor(triangles.astype(np.uint32)))
    colors = np.asarray(mesh.vertex_colors) * 255 if mesh.has_vertex_colors() else None
    uv, texture = texture_coordinates(path, len(vertices)) if textured else (None, None)
    size = 360
    yy, xx = np.mgrid[:size, :size]
    images = []
    for eye in EYES:
        forward = -np.asarray(eye, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        origin = (-3 * forward + (xx[..., None] + .5 - size / 2) * (2.15 / size) * right
                  - (yy[..., None] + .5 - size / 2) * (2.15 / size) * up)
        rays = np.concatenate([origin, np.broadcast_to(forward, origin.shape)], axis=-1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        mask = np.isfinite(hit["t_hit"].numpy())
        assert mask.any() and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        faces = triangles[hit["primitive_ids"].numpy()[mask]]
        ab = hit["primitive_uvs"].numpy()[mask]
        weights = np.column_stack([1 - ab.sum(1), ab])
        if texture is not None:
            coords = (uv[faces] * weights[..., None]).sum(1)
            tx = np.clip(np.rint(coords[:, 0] * (texture.shape[1] - 1)), 0, texture.shape[1] - 1).astype(int)
            ty = np.clip(np.rint((1 - coords[:, 1]) * (texture.shape[0] - 1)), 0, texture.shape[0] - 1).astype(int)
            base = texture[ty, tx].astype(float)
        elif colors is not None:
            base = (colors[faces] * weights[..., None]).sum(1)
        else:
            base = np.tile([169., 181., 195.], (len(faces), 1))
        normal = (normals[faces] * weights[..., None]).sum(1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        light = -forward + .5 * up - .3 * right
        light /= np.linalg.norm(light)
        intensity = .65 + .35 * np.abs(normal @ light)
        rgba = np.zeros((size, size, 4), np.uint8)
        rgba[mask, :3] = np.clip(base * intensity[:, None], 0, 255).astype(np.uint8)
        rgba[mask, 3] = 255
        image = Image.fromarray(rgba)
        images.append(image.crop(image.getbbox()))
    return images


def gallery(dataset, title, paths):
    columns = min(5, len(paths))
    rows = (len(paths) + columns - 1) // columns
    width, height, gap = 340, 246, 14
    canvas = Image.new("RGB", (32 + columns * (width + gap) - gap,
                               114 + rows * (height + gap)), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    fonts = Path("/mnt/c/Windows/Fonts")
    label = ImageFont.truetype(str(fonts / "segoeuib.ttf"), 22)
    small = ImageFont.truetype(str(fonts / "segoeui.ttf"), 18)
    heading = ImageFont.truetype(str(fonts / "segoeuib.ttf"), 32)
    draw.text((16, 12), f"{title}: all {len(paths)} object models", font=heading, fill="#182938")
    subtitle = "Green border: the 16 models in the classic BOP HB subset." if dataset == "hb" else "Two views per model; labels use BOP object IDs."
    draw.text((16, 57), subtitle, font=small, fill="#465766")
    for index, path in enumerate(paths):
        object_id = int(path.stem.split("_")[-1])
        views = render_views(path, dataset == "ycbv")
        row, col = divmod(index, columns)
        x, y = 16 + col * (width + gap), 94 + row * (height + gap)
        selected = dataset == "hb" and object_id in HB_SUBSET
        draw.rounded_rectangle((x, y, x + width, y + height), radius=9, fill="white",
                               outline="#19866b" if selected else "#d5dee7", width=3 if selected else 1)
        name = LMO_NAMES[object_id] if dataset == "lmo" else YCB_NAMES[object_id - 1] if dataset == "ycbv" else ""
        draw.text((x + 12, y + 8), f"ID {object_id:02}" + (f"  {name}" if name else ""), font=label, fill="#182938")
        for j, image in enumerate(views):
            tile = ImageOps.contain(image, (152, 178), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x + 12 + 164 * j + (152 - tile.width) // 2,
                               y + 46 + (178 - tile.height) // 2), tile)
        print(dataset, object_id, "rendered", flush=True)
    draw.text((16, canvas.height - 24), "Official BOP models. Objects resized independently.", font=small, fill="#465766")
    canvas.save(OUTPUT / f"{dataset}_all_models.png")


def main():
    OUTPUT.mkdir(exist_ok=True)
    counts = {}
    for dataset, (title, expected) in DATASETS.items():
        paths = sorted((CACHE / dataset / "models").glob("obj_*.ply"))
        assert len(paths) == expected, (dataset, len(paths), expected)
        counts[dataset] = expected
        if not (OUTPUT / f"{dataset}_all_models.png").exists():
            gallery(dataset, title, paths)
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "model_counts": counts, "source": "https://bop.felk.cvut.cz/datasets/",
        "ycb_names": "https://github.com/yuxng/YCB_Video_toolbox/blob/master/classes.txt",
        "hb_classic_subset_ids": sorted(HB_SUBSET), "views_native_model_axes": EYES,
        "renderer": "Open3D first-hit rays, vertex colors or UV texture, simple normal shading",
        "purpose": "Inspection galleries only; no association experiment or production change"
    }, indent=2))


if __name__ == "__main__":
    main()
