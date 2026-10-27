"""Supplied CAD/crop associations and observation-fitted poses for N00--N07.

Identity correspondence is assumed correct for this isolated grasp study. Audit
labels supply only that correspondence; no simulator object pose is read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY = REPO / "experiments/nist12_qwen_clip_reranking_20261023"
FITTER = STUDY / "depth_verification_20261023"
OUTPUT = REPO / "outputs/cad_grasp_geometry_20261026"
MANIFEST = OUTPUT / "inputs.json"
ASSOCIATIONS = REPO / "outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures/audit_labels.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    value = str(path).replace("\\", "/")
    retired = "C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to"
    for prefix in (retired, "/mnt/c" + retired[2:]):
        if value == prefix or value.startswith(prefix + "/"):
            suffix = value[len(prefix):].lstrip("/")
            if ".." in suffix.split("/") or ":" in suffix:
                raise ValueError("Unsafe retired-workspace path suffix")
            return REPO / "outputs/datasets/nist_task_board/legacy_workspace_inputs" / suffix
    if len(value) > 1 and value[1] == ":" and sys.platform != "win32":
        value = "/mnt/" + value[0].lower() + value[2:]
    elif sys.platform == "win32" and value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
        value = value[5].upper() + ":" + value[6:]
    result = Path(value)
    return result if result.is_absolute() else REPO / result


def relative(path):
    try:
        return Path(path).relative_to(REPO).as_posix()
    except ValueError:
        value = str(path).replace("\\", "/")
        if value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
            return value[5].upper() + ":" + value[6:]
        return value


def write_new(path, data):
    path = Path(path)
    if path.exists():
        if read(path) != data:
            raise RuntimeError(f"Refusing to overwrite saved input evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_missing(scene, part, proposal_id):
    """Run the unchanged observed-depth fitter, saving only to this new study."""
    path = OUTPUT / "extra_fits" / f"{scene}__{part}__p{proposal_id:03d}.json"
    if path.exists():
        return path
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(FITTER))
    spec = importlib.util.spec_from_file_location("grasp_input_original_fitter", FITTER / "run_trial.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import geometry
    import open3d as o3d

    runner.worker_init()
    pair = {"state": scene, "target": part, "proposal_id": proposal_id}
    points, plane, intrinsics, depth, mask = runner.observation(pair)
    mesh_path = local(runner._INPUTS["cads"][part]["mesh_file"])
    start = perf_counter()
    cad = geometry.prepare_cad(o3d.io.read_triangle_mesh(str(mesh_path)))
    result = geometry.fit_pair(cad, points, plane, intrinsics, depth, mask, clearances=(0.,))
    record = {
        **pair,
        "geometry": result,
        "observed_points": len(points),
        "wall_seconds": perf_counter() - start,
        "existing_fitting_protocol_sha256": sha256(FITTER / "protocol.json"),
        "fitter_source_sha256": {name: sha256(FITTER / name) for name in ("geometry.py", "run_trial.py")},
        "clearances_m": [0.],
        "correspondence_assumed": True,
        "simulator_object_poses_read": False,
        "note": "Missing correct-crop fit for this isolated grasp study; old shortlist/fit evidence is unchanged.",
    }
    write_new(path, record)
    print(json.dumps({"new_observation_fit": relative(path), "seconds": record["wall_seconds"]}), flush=True)
    return path


def prepare_inputs(refit_missing=True):
    captures_path = STUDY / "data/capture_manifest.json"
    cads_path = STUDY / "data/cad_inputs.json"
    captures = read(captures_path)["states"]
    cads = read(cads_path)
    associations = read(ASSOCIATIONS)["identities"]
    sources = {relative(path): sha256(path) for path in (captures_path, cads_path, ASSOCIATIONS)}
    records = []
    for scene, capture in sorted(captures.items()):
        camera_path = local(capture["world_T_camera_file"])
        world_T_camera = np.load(camera_path)
        camera_plane = np.asarray(capture["plane_model_camera"], float)
        normal = world_T_camera[:3, :3] @ camera_plane[:3]
        distance = float(camera_plane[3] - normal @ world_T_camera[:3, 3])
        plane = np.r_[normal, distance]
        plane /= np.linalg.norm(plane[:3])
        if plane[2] < 0:
            plane *= -1
        sources[relative(camera_path)] = sha256(camera_path)
        for candidate, part in sorted(associations[scene].items(), key=lambda item: item[1]):
            proposal_id = int(candidate)
            mesh_path = local(cads[part]["mesh_file"])
            assert sha256(mesh_path) == cads[part]["mesh_sha256"]
            filename = f"{scene}__{part}__p{proposal_id:03d}.json"
            fit_path = FITTER / "fits" / filename
            if not fit_path.exists() and refit_missing:
                fit_path = fit_missing(scene, part, proposal_id)
            row = {
                "scene": scene, "part": part,
                "mesh_path": relative(mesh_path), "mesh_sha256": cads[part]["mesh_sha256"],
                "world_T_cad": None,
                "table_z_m": float(-plane[3] / plane[2]),
                "table_plane_world": plane.tolist(),
                "input_status": "unavailable_missing_observed_fit",
                "pose_source": relative(fit_path), "candidate_id": proposal_id,
                "associations_assumed": True,
                "pose_arm": "contact_0mm",
                "world_T_camera_source": relative(camera_path),
                "simulator_object_poses_read": False,
            }
            if fit_path.exists():
                fit = read(fit_path)
                arms = fit["geometry"].get("arms", [])
                arm = next((value for value in arms if value["arm_name"] == "contact_0mm"), None)
                sources[relative(fit_path)] = sha256(fit_path)
                if arm and arm["status"] == "available":
                    best = arm["best_geometry"]
                    pose = world_T_camera @ np.asarray(best["T_camera_from_cad"], float)
                    assert np.allclose(pose[3], [0, 0, 0, 1])
                    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8)
                    row.update(world_T_cad=pose.tolist(), input_status="available",
                               pose_geometry_score=best["verification"]["geometry_score"],
                               pose_depth_mae_m=best["verification"]["depth_mae_m"],
                               observed_points=fit["observed_points"])
                else:
                    row["input_status"] = "unavailable_observed_fit"
            records.append(row)
    assert len(records) == 96
    assert len({(row["scene"], row["part"]) for row in records}) == 96
    manifest = {
        "study": "N00-N07 pre-VLM CAD grasp geometry",
        "scenes": sorted(captures), "parts": sorted(cads),
        "records": records, "source_sha256": sources,
        "association_assumption": "Correct identity-to-depth-crop correspondence is supplied. Audit identities are used only to instantiate this assumption, never as an estimated classification result or object pose.",
        "pose_estimation": "Saved native-scale CAD/observed-depth contact_0mm best-geometry fits. Missing correct-crop fit is recomputed with unchanged zero-clearance fitter into new outputs.",
        "camera_transform": "Existing saved simulation camera calibration; not a new physical calibration.",
        "table": "Observed camera-frame depth plane transformed to world, normalized and upward-facing. table_z_m is plane intercept at world x=y=0; use full plane for distance checks.",
        "limitations": ["Assumed successful association; classification/retrieval errors are outside this study.",
                        "Observed CAD fits are finite table-constrained orientation hypotheses, not certified unrestricted 6D estimates.",
                        "Symmetric part orientation can remain unobservable.",
                        "N00-N07 contain twelve products, not the older fourteen-part inventory.",
                        "Static synthetic captures use ideal depth; no physical calibration or robot motion is performed."],
        "simulator_object_poses_read": False,
    }
    write_new(MANIFEST, manifest)
    return records


def load_inputs():
    """Return the 96 frozen input records; prepare them if not already saved."""
    return read(MANIFEST)["records"] if MANIFEST.exists() else prepare_inputs()


if __name__ == "__main__":
    rows = load_inputs()
    print(json.dumps({"input_records": len(rows), "available": sum(row["input_status"] == "available" for row in rows),
                      "manifest": relative(MANIFEST)}), flush=True)
