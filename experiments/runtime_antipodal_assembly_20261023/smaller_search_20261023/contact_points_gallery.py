"""Draw the saved small-search contact points on native CAD; no physics runs."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import string
import struct

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/runtime_antipodal_assembly_20261023/smaller_search/20261024T030119Z"
MESHES = ROOT / "outputs/simulation/nist_task_board_1/assets/meshes"
GROUPS = {
    "gears_and_nuts": [("Gear_Large", "Large gear"), ("Gear_Medium", "Medium gear"),
                       ("Gear_Small", "Small gear"), ("M12_Hex_Nut", "M12 nut"),
                       ("M16_Hex_Nut", "M16 nut")],
    "pins": [("RGOCG16-50_16mm", "16 mm round pin"), ("KET16_Square_16mm", "16 mm rectangular pin"),
             ("RGOCG12-50_12mm", "12 mm round pin"), ("KET12_Square_12mm", "12 mm rectangular pin"),
             ("RGOCG8-50_8mm", "8 mm round pin"), ("KET8_Square_8mm", "8 mm rectangular pin")],
    "connectors": [("Waterproof_Male", "Waterproof connector"), ("USB_Male", "USB connector"),
                   ("DSUB_Male", "D-sub connector")],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def read_stl(path):
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + 50 * count, "Expected native binary STL"
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].astype(float)


def contact_groups(grasps):
    grouped = {}
    for index, grasp in enumerate(grasps, 1):
        points = np.asarray(grasp["contacts_world_m"])
        ordered = sorted(tuple(np.round(point, 5)) for point in points)
        key = tuple(ordered)
        if key not in grouped:
            grouped[key] = {"points": points, "grasps": []}
        grouped[key]["grasps"].append(index)
    return list(grouped.values())


def projected_view(triangles, groups, azimuth):
    az, el = np.deg2rad([azimuth, 32.])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    up = np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
    toward = np.cross(right, up)
    basis = np.column_stack((right, up, toward))
    projected = triangles @ basis
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([.3, -.4, .85]); light /= np.linalg.norm(light)
    shade = .65 + .35*np.clip(normals @ light, 0, 1)
    colors = np.array([.76, .81, .87])[None, :] * shade[:, None]
    order = np.argsort(projected[:, :, 2].mean(axis=1))
    fig = Figure(figsize=(5, 3.6), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([.01, .01, .98, .98])
    ax.add_collection(PolyCollection(projected[order, :, :2], facecolors=colors[order],
                                     edgecolors="none", antialiased=False, rasterized=True))
    bounds = projected[:, :, :2].reshape(-1, 2)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    centre = (low + high)/2
    span = max((high-low)[0], (high-low)[1] * 5/3.6) * 1.25
    ax.set_xlim(centre[0]-span/2, centre[0]+span/2)
    ax.set_ylim(centre[1]-span*.36, centre[1]+span*.36)
    ax.set_aspect("equal"); ax.axis("off")
    for index, group in enumerate(groups):
        points = group["points_mm"] @ basis
        color = matplotlib.colormaps["tab20"]((2*index) % 20)
        ax.plot(points[:, 0], points[:, 1], "--", color=color, lw=1.1, alpha=.8, zorder=3)
        ax.scatter(points[:, 0], points[:, 1], s=44, c=[color], edgecolors="white", lw=1, zorder=4)
        for point in points:
            ax.annotate(string.ascii_uppercase[index], point[:2], xytext=(5, 5),
                        textcoords="offset points", fontsize=11, fontweight="bold", zorder=5,
                        bbox={"boxstyle": "round,pad=.12", "fc": "white", "ec": "none", "alpha": .92})
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB")


def make_part(part, label, output):
    source = RUN / "small" / part / "result.json"
    before = digest(source)
    result = json.loads(source.read_text())
    mesh_path = MESHES / (part + "_m.stl")
    assert digest(mesh_path) == result["mesh_sha256"]
    triangles = read_stl(mesh_path)
    transform = np.asarray(result["world_T_cad_input"])
    triangles = triangles @ transform[:3, :3].T + transform[:3, 3]
    centre = (triangles.reshape(-1, 3).min(0) + triangles.reshape(-1, 3).max(0))/2
    triangles = (triangles-centre)*1000
    groups = contact_groups(result["shortlisted_grasps"])
    for group in groups:
        group["points_mm"] = (group["points"]-centre)*1000
    canvas = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 999, 619), outline="#d7e0e9", width=2)
    draw.text((18, 14), label, font=font(28, True), fill="#17324c")
    count = len(result["shortlisted_grasps"])
    draw.text((18, 52), f"{count} kept poses | {len(groups)} contact pairs | {2*len(groups)} contact points",
              font=font(20), fill="#4d6175")
    for column, azimuth in enumerate((-60, 120)):
        canvas.paste(projected_view(triangles, groups, azimuth), (500*column, 89))
    draw.text((210, 84), "View A", font=font(17), fill="#61758a")
    draw.text((664, 84), "Opposite side", font=font(17), fill="#61758a")
    for index, group in enumerate(groups):
        column, row = index % 2, index // 2
        ids = ", ".join(f"G{n:02d}" for n in group["grasps"])
        text = f"{string.ascii_uppercase[index]}: {ids}"
        color = matplotlib.colors.to_hex(matplotlib.colormaps["tab20"]((2*index) % 20))
        draw.text((20+column*495, 457+row*24), text, font=font(17, True), fill=color)
    filename = part + "_contacts.png"
    canvas.save(output / filename)
    assert before == digest(source)
    return {"part": part, "label": label, "poses": count, "pairs": len(groups),
            "points": len(groups)*2, "source": str(source), "source_sha256": before,
            "mesh_sha256": result["mesh_sha256"], "image": filename,
            "groups": [{"letter": string.ascii_uppercase[i], "grasp_numbers": g["grasps"],
                        "contacts_world_m": g["points"].tolist()} for i, g in enumerate(groups)]}


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs/runtime_antipodal_assembly_20261023/contact_point_views" / stamp
    output.mkdir(parents=True)
    records, sheets = [], []
    for name, members in GROUPS.items():
        current = [make_part(part, label, output) for part, label in members]
        records.extend(current)
        rows = (len(current)+1)//2
        sheet = Image.new("RGB", (2000, 126+620*rows), "#f2f6fa")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 14), "All kept contact points: " + name.replace("_", " "),
                  font=font(34, True), fill="#17324c")
        draw.text((20, 62), "Same letter / color = one finger-contact pair. Dashed line joins the pair; it is not a motion path.",
                  font=font(23), fill="#4d6175")
        draw.text((20, 92), "All contacts are shown through the CAD, including hidden-side contacts. G numbers match the original grasp gallery.",
                  font=font(23), fill="#4d6175")
        for i, record in enumerate(current):
            sheet.paste(Image.open(output / record["image"]), (1000*(i%2), 126+620*(i//2)))
        filename = name + ".png"; sheet.save(output / filename); sheets.append(filename)
        print(f"Rendered {name}: {sum(r['poses'] for r in current)} saved poses", flush=True)
    assert len(records) == 14
    assert sum(r["poses"] for r in records) == 156
    assert sum(r["pairs"] for r in records) == 85
    original = RUN / "shortlist_gallery/index.html"
    links = [f'<a href="{original.as_uri()}#{r["part"]}">{html.escape(r["label"])}: all {r["poses"]} hand poses</a>' for r in records]
    pictures = ''.join(f'<a href="{name}"><img src="{name}" alt="All saved contact points"></a>' for name in sheets)
    page = ('<!doctype html><meta charset="utf-8"><title>Smaller search: all kept contacts</title>'
            '<style>body{font:18px Arial;margin:24px;background:#f2f6fa;color:#17324c}img{width:100%;height:auto}'
            'nav{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}a{color:#145ba0}</style>'
            '<h1>All kept finger-contact points on all 14 parts</h1>'
            '<p>156 saved poses share 85 contact pairs (170 distinct points). Letters group identical pairs; G numbers identify the original poses.</p>'
            '<p>These are saved candidate contacts. One pickup was attempted per part. No new grasp generation or physics was run.</p>'
            '<nav>' + ' | '.join(links) + '</nav>' + pictures)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "validation.json").write_text(json.dumps({"source_run": str(RUN), "physics_steps": 0,
        "part_count": len(records), "pose_count": sum(r["poses"] for r in records),
        "contact_pair_count": sum(r["pairs"] for r in records), "contact_point_count": sum(r["points"] for r in records),
        "deduplication_m": .00001, "source_hashes_unchanged": True, "parts": records}, indent=2)+"\n", encoding="utf-8")
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
