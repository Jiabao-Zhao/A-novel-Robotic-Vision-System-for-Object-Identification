"""Extract one saved CAD/table example for an interactive explanation; no fitting."""

import hashlib
import json

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from scripts.inspect_bop_mask_cad_oracles import ROOT, SOURCE, read_json


def main():
    case = next(r for r in read_json(SOURCE / "weights_and_table_diagnostic.json")["cases"]
                if r["scene_id"] == 11 and r["true_cad_id"] == 8)
    name = "scene_000011_000004"
    data = ROOT / "outputs/cache/bop_tless/data/tless"
    scene = data / "test_primesense/000011"
    mesh_path = data / "models_cad/obj_000008.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    center = (vertices.min(0) + vertices.max(0)) / 2
    local = vertices - center
    hull = local[ConvexHull(local).vertices]
    display = mesh.simplify_quadric_decimation(1600)
    plane = np.array(read_json(ROOT / "outputs/bop_localization_20260917" / name / "localization.json")["plane_model"])
    plane /= np.linalg.norm(plane[:3])
    if plane[3] < 0:
        plane *= -1
    pose_rows = read_json(scene / "scene_gt.json")["4"]
    info = read_json(scene / "scene_gt_info.json")["4"]
    pose_rows = [p for p, i in zip(pose_rows, info) if i["px_count_visib"] > 0]
    anns = [a for a in read_json(SOURCE / "coco_ground_truth.json")["annotations"] if a["image_id"] == 11]
    gt = pose_rows[next(i for i, a in enumerate(anns) if a["id"] == case["gt_id"])]
    assert gt["obj_id"] == 8
    Rgt, tgt = np.array(gt["cam_R_m2c"]).reshape(3, 3), np.array(gt["cam_t_m2c"])
    views = [r for r in read_json(SOURCE / "matching" / name / "regions/sam_096.json")["views"]
             if r["cad_id"] == "obj_000008"]
    assert len(views) == 42 and all(r["pose_status"] == "table_penetration" for r in views)
    chosen = next(r for r in views if r["view_index_1based"] == 42)
    initial = np.array(chosen["camera_T_cad"])
    fitted_center = initial[:3, :3] @ center + initial[:3, 3] * 1000
    origin = fitted_center - plane[:3] * (fitted_center @ plane[:3] + 1000 * plane[3])
    longest = Rgt[:, np.argmax(np.ptp(vertices, axis=0))]
    x = longest - plane[:3] * (longest @ plane[:3])
    x /= np.linalg.norm(x)
    B = np.stack((x, np.cross(plane[:3], x), plane[:3]))
    assert np.linalg.det(B) > .999999
    saved = []
    for row in sorted(views, key=lambda v: v["view_index_1based"]):
        T = np.array(row["camera_T_cad"])
        rotation = B @ T[:3, :3]
        location = B @ (T[:3, :3] @ center + T[:3, 3] * 1000 - origin)
        minimum = float((local @ rotation.T + location)[:, 2].min())
        np.testing.assert_allclose((hull @ rotation.T + location)[:, 2].min(), minimum, atol=1e-10)
        saved.append({"view": row["view_index_1based"], "q": Rotation.from_matrix(rotation).as_quat().tolist(),
                      "center": location.tolist(), "minimum_mm": minimum})
    start = saved[41]
    target_q = Rotation.from_matrix(B @ Rgt).as_quat()
    target_center = B @ (Rgt @ center + tgt - origin)
    rotated_min = float((local @ (B @ Rgt).T + np.array(start["center"]))[:, 2].min())
    translated_min = float((local @ (B @ initial[:3, :3]).T + target_center)[:, 2].min())
    np.testing.assert_allclose(start["minimum_mm"], case["pose_diagnostic"]["fitted_clearance_mm"], atol=1e-8)
    camera = read_json(scene / "scene_camera.json")["4"]
    K = np.array(camera["cam_K"]).reshape(3, 3)
    raw = cv2.imread(str(scene / "depth/000004.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(SOURCE / "sam" / name / "masks/sam_096.png"), 0) > 0
    yy, xx = np.nonzero(mask & (raw > 0))
    z = raw[yy, xx].astype(float) * camera["depth_scale"]
    observed = np.column_stack(((xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z))
    observed = (observed - origin) @ B.T
    observed = observed[np.linspace(0, len(observed) - 1, min(1000, len(observed))).astype(int)]
    extent_points = [observed]
    for pose in saved:
        extent_points.append(hull @ Rotation.from_quat(pose["q"]).as_matrix().T + pose["center"])
    interpolator = Slerp([0, 1], Rotation.from_quat([start["q"], target_q]))
    for progress in np.linspace(0, 1, 51):
        extent_points.append(hull @ interpolator(progress).as_matrix().T + start["center"])
    extents = np.concatenate(extent_points)
    result = {"scene": 11, "frame": 4, "cad": 8, "region": "sam_096", "units": "mm",
        "display_vertices": np.round(np.asarray(display.vertices) - center, 4).tolist(),
        "triangles": np.asarray(display.triangles).tolist(), "hull_vertices": np.round(hull, 6).tolist(),
        "observed_points": np.round(observed, 3).tolist(), "saved": saved, "target_q": target_q.tolist(),
        "target_center": target_center.tolist(), "tolerance_mm": 5.,
        "expected": {"saved_minimum_mm": start["minimum_mm"], "rotation_corrected_minimum_mm": rotated_min,
                     "center_corrected_minimum_mm": translated_min, "best_of_42_mm": max(p["minimum_mm"] for p in saved)},
        "bounds": [extents.min(0).tolist(), extents.max(0).tolist()],
        "scope": "Actual saved placements and CAD. Corrected rotation/center endpoints use GT only for diagnosis. Motion is an illustrative interpolation, not an estimator or registration result. Display mesh simplified; signed minimum uses the original CAD convex-hull vertices, mathematically identical for a plane.",
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest()}
    (SOURCE / "table_filter_visual_data.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print({"expected": result["expected"], "vertices": len(result["display_vertices"]),
           "faces": len(result["triangles"]), "exact_hull_vertices": len(hull),
           "bounds": result["bounds"]}, flush=True)


if __name__ == "__main__":
    main()
