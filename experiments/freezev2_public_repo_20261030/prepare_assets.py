"""Build a no-copy, GT-free BOP inference view for unmodified public code.

Only this new experiment's output tree is written. Original RGB/depth/CAD,
checkpoints and existing environments remain read-only. TIFF/JPEG bytes may be
aliased under the public CLI's .png names; Pillow detects the actual format.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
from PIL import Image

DATASETS = ("lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv")
SPLITS = {name: "test_primesense" if name in {"tless", "hb"} else "test" for name in DATASETS}
FORBIDDEN_GT_NAMES = {"scene_gt.json", "scene_gt_info.json", "scene_gt_coco.json"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def local_path(value) -> Path:
    value = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    return Path(value).expanduser().absolute()


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_identity(path, expected):
    path = Path(path)
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("Asset view identity differs; use a new versioned output directory")
    else:
        atomic_json(path, expected)


def ensure_alias(source, destination):
    """Create or validate an exact symlink; never overwrite ordinary files."""
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source.name in FORBIDDEN_GT_NAMES or destination.name in FORBIDDEN_GT_NAMES:
        raise ValueError("GT annotation links are forbidden in the inference view")
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError("Alias points to a different source: " + str(destination))
    elif destination.exists():
        raise ValueError("Refusing to overwrite a non-symlink asset: " + str(destination))
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    if not destination.exists() or destination.resolve() != source.resolve():
        raise ValueError("Alias verification failed: " + str(destination))
    return destination


def pixel_sha(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def verify_image_alias(source, alias, *, rgb=False):
    """Decode both source and alias and compare every pixel and original mode."""
    source, alias = Path(source), Path(alias)
    if not alias.is_symlink() or alias.resolve() != source.resolve():
        raise ValueError("An image alias must resolve to its exact original source")
    with Image.open(source) as a, Image.open(alias) as b:
        raw_source, raw_alias = np.asarray(a).copy(), np.asarray(b).copy()
        source_mode, alias_mode = a.mode, b.mode
        source_format, alias_format = a.format, b.format
        if (source_mode != alias_mode or a.size != b.size or raw_source.dtype != raw_alias.dtype
                or raw_source.shape != raw_alias.shape
                or not np.array_equal(raw_source, raw_alias, equal_nan=True)):
            raise ValueError("Image alias changed source mode, dimensions or pixels")
        converted = np.asarray(a.convert("RGB")).copy() if rgb else None
        converted_alias = np.asarray(b.convert("RGB")).copy() if rgb else None
    if rgb and not np.array_equal(converted, converted_alias):
        raise ValueError("RGB conversion differs between original and aliased image")
    return {"source_path": str(source), "alias_path": str(alias),
            "source_sha256": sha256(source), "source_mode": source_mode,
            "alias_mode": alias_mode, "source_format": source_format,
            "alias_detected_format": alias_format, "shape": list(raw_source.shape),
            "dtype": str(raw_source.dtype), "native_pixel_sha256": pixel_sha(raw_source),
            "RGB_converted_pixel_sha256": pixel_sha(converted) if rgb else None,
            "all_pixels_equal": True, "transcoding_performed": False}


def find_image(scene_dir, folders, image_id):
    """Resolve explicit BOP filenames without interpreting extension as format."""
    scene_dir = Path(scene_dir)
    for folder in folders:
        for suffix in IMAGE_SUFFIXES:
            candidate = scene_dir / folder / f"{image_id:06d}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No source image {image_id:06d} in {scene_dir}; folders={folders}")


def exact_targets(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Official target rows must be a nonempty list")
    targets, seen = [], set()
    for row in rows:
        result = {}
        for key, minimum in (("scene_id", 0), ("im_id", 0), ("obj_id", 1), ("inst_count", 1)):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("Invalid official target " + key)
            result[key] = value
        key = result["scene_id"], result["im_id"], result["obj_id"]
        if key in seen:
            raise ValueError("Duplicate official target object-image row")
        seen.add(key)
        targets.append(result)
    return sorted(targets, key=lambda t: (t["scene_id"], t["im_id"], t["obj_id"]))


def reject_gt_links(view):
    """Check only the materialized view; directory model links are not traversed."""
    for base, dirs, files in os.walk(view, followlinks=False):
        for name in dirs + files:
            if name in FORBIDDEN_GT_NAMES:
                raise ValueError("Unexpected GT metadata in inference view: " + str(Path(base) / name))


def prepare_dataset(dataset_root, view_root, dataset, entry, *, progress=None):
    if dataset not in DATASETS:
        raise ValueError("Unsupported Classic-Core dataset")
    dataset_root, view_root = Path(dataset_root), Path(view_root)
    source = dataset_root / dataset
    models = source / ("models_cad" if dataset == "tless" else "models")
    if not (models / "models_info.json").is_file():
        raise FileNotFoundError(models / "models_info.json")
    target_source = local_path(entry["targets"]["path"])
    if sha256(target_source) != entry["targets"]["sha256"]:
        raise ValueError("Official target source changed")
    targets = exact_targets(read_json(target_source))
    expected_rows = entry.get("target_records", len(targets))
    if len(targets) != expected_rows:
        raise ValueError("Target row count differs from retained asset manifest")
    images = sorted({(t["scene_id"], t["im_id"]) for t in targets})
    if len(images) != entry.get("official_target_images", len(images)):
        raise ValueError("Target image count differs from retained asset manifest")
    mask_record = None
    if entry.get("masks"):
        original_mask = entry["masks"]
        checked_mask = _checked_pin(original_mask["path"], original_mask["sha256"], "Predicted masks")
        mask_record = {**original_mask, "path": checked_mask["path"],
                       "actual_sha256": checked_mask["sha256"], "bytes": checked_mask["bytes"]}
    destination = view_root / dataset
    ensure_alias(models, destination / "models")
    ensure_alias(target_source, destination / "test_targets_bop19.json")
    camera_cache, frames = {}, []
    model_records = []
    for model in sorted(models.glob("obj_*.ply")):
        model_records.append({"path": str(model), "sha256": sha256(model), "bytes": model.stat().st_size})
    model_ids = {int(p.stem.removeprefix("obj_")) for p in models.glob("obj_*.ply")}
    required_ids = sorted({t["obj_id"] for t in targets})
    if not set(required_ids) <= model_ids:
        raise ValueError("A targeted CAD object is absent from original models")
    for number, (scene, image) in enumerate(images, 1):
        original_scene = source / SPLITS[dataset] / f"{scene:06d}"
        new_scene = destination / "test" / f"{scene:06d}"
        camera = original_scene / "scene_camera.json"
        ensure_alias(camera, new_scene / "scene_camera.json")
        if scene not in camera_cache:
            camera_cache[scene] = (read_json(camera), sha256(camera))
        cameras, camera_sha = camera_cache[scene]
        calibration = cameras.get(str(image), cameras.get(f"{image:06d}"))
        if calibration is None:
            raise ValueError("Target image calibration missing")
        K = np.asarray(calibration.get("cam_K"), dtype=np.float64)
        depth_scale = float(calibration.get("depth_scale", 1.0))
        if (K.size != 9 or not np.isfinite(K).all() or K.reshape(3, 3)[0, 0] <= 0
                or K.reshape(3, 3)[1, 1] <= 0 or not np.isfinite(depth_scale) or depth_scale <= 0):
            raise ValueError("Invalid physical calibration or depth scale")
        rgb = find_image(original_scene, ("rgb", "gray"), image)
        depth = find_image(original_scene, ("depth",), image)
        rgb_alias = ensure_alias(rgb, new_scene / "rgb" / f"{image:06d}.png")
        depth_alias = ensure_alias(depth, new_scene / "depth" / f"{image:06d}.png")
        rgb_record = verify_image_alias(rgb, rgb_alias, rgb=True)
        depth_record = verify_image_alias(depth, depth_alias)
        if rgb_record["shape"][:2] != depth_record["shape"][:2]:
            raise ValueError("RGB/gray and depth dimensions differ")
        frames.append({"scene_id": scene, "im_id": image, "rgb": rgb_record,
                       "depth": depth_record, "camera_source_path": str(camera),
                       "camera_sha256": camera_sha, "cam_K": K.reshape(3, 3).tolist(),
                       "depth_scale_mm": depth_scale})
        if progress and (number % 50 == 0 or number == len(images)):
            progress(dataset=dataset, images_verified=number, expected_images=len(images))
    reject_gt_links(destination)
    return {"source_root": str(source), "source_split": SPLITS[dataset], "view_split": "test",
            "inference_view": str(destination), "target_source_path": str(target_source),
            "target_sha256": sha256(target_source), "target_rows": len(targets),
            "requested_instances": sum(t["inst_count"] for t in targets), "images": len(images),
            "required_object_ids": required_ids, "models_source": str(models), "masks": mask_record,
            "models_info_sha256": sha256(models / "models_info.json"),
            "model_files": model_records, "native_CAD_units": "millimetres",
            "image_aliases_verified": len(frames), "frames": frames,
            "GT_annotation_links": 0, "image_transcoding_or_dataset_copy": False}


def _checked_pin(path, expected, label):
    path = local_path(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(label + " checksum differs from trained runtime pin")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def prepare(root, output, datasets=DATASETS):
    if os.name != "posix":
        raise RuntimeError("Create this symlink view with the existing WSL Python; native Windows is read-only for this helper")
    root, output = local_path(root), local_path(output)
    datasets = tuple(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in DATASETS for d in datasets):
        raise ValueError("Choose unique supported Classic-Core datasets")
    old = root / "outputs/freezev2_bop_20261029"
    asset_path, runtime_path, eval_path = (old / "assets_manifest.json", old / "runtime_ready.json", old / "evaluation_runtime.json")
    assets, runtime, evaluation = read_json(asset_path), read_json(runtime_path), read_json(eval_path)
    data_root = local_path(root / "experiments/freezev2_bop_20261029/protocol.json")
    dataset_root = local_path(read_json(data_root)["paths"]["dataset_root"])
    identity = {"schema": "public-repo-source-alias-view-v1", "datasets": list(datasets),
                "assets_manifest_sha256": sha256(asset_path), "runtime_ready_sha256": sha256(runtime_path),
                "evaluation_runtime_sha256": sha256(eval_path), "dataset_root": str(dataset_root),
                "implementation_sha256": sha256(__file__), "GT_link_policy": "none",
                "image_policy": "exact original bytes and pixels; extension aliases only"}
    output.mkdir(parents=True, exist_ok=True)
    ensure_identity(output / "asset_view_identity.json", identity)
    if shutil.disk_usage(output).free < 15 * 1024**3:
        raise RuntimeError("Below15GiB conservative disk reserve")
    def progress(**values):
        atomic_json(output / "asset_preparation_progress.json", {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), **values})
        print(json.dumps(values), flush=True)
    start = time.perf_counter()
    pins = runtime["checkpoint_paths_sha256"]
    dino = _checked_pin(pins["dinov2_giant"]["wsl_path"], pins["dinov2_giant"]["sha256"], "DINO giant")
    gedi = _checked_pin(pins["gedi_3dmatch"]["wsl_path"], pins["gedi_3dmatch"]["sha256"], "GeDi checkpoint")
    dino_source = local_path(runtime["scientific_backend_settings"]["dinov2"]["source_path"])
    dino_files = {}
    for relative, expected in runtime["scientific_backend_settings"]["dinov2"]["verified_source_hashes"].items():
        dino_files[relative] = _checked_pin(dino_source / relative, expected, "DINO source")
    gedi_settings = runtime["scientific_backend_settings"]["gedi"]
    gedi_source = local_path(gedi_settings["source_path"])
    gedi_file = _checked_pin(gedi_source / "gedi.py", gedi_settings["source_sha256"], "GeDi source")
    gedi_revision = _git_head(gedi_source)
    if gedi_revision != gedi_settings["source_revision"]:
        raise ValueError("GeDi actual Git revision differs")
    toolkit = local_path(evaluation["toolkit_root"])
    toolkit_revision = _git_head(toolkit)
    if toolkit_revision != evaluation["toolkit_revision"]:
        raise ValueError("Official evaluator toolkit revision differs")
    torch_home = output / "runtime/torch"
    giant_alias = ensure_alias(dino["path"], torch_home / "hub/checkpoints/dinov2_vitg14_pretrain.pth")
    prepared = {name: prepare_dataset(dataset_root, output / "bop_view", name, assets["datasets"][name], progress=progress) for name in datasets}
    totals = {"images": sum(d["images"] for d in prepared.values()),
              "target_rows": sum(d["target_rows"] for d in prepared.values()),
              "requested_instances": sum(d["requested_instances"] for d in prepared.values())}
    if set(datasets) == set(DATASETS) and (totals["images"] != 3871 or totals["target_rows"] != 13635):
        raise ValueError("All-seven exact target coverage differs from3871images/13635object-image rows")
    reject_gt_links(output / "bop_view")
    result = {**identity, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "verified_inference_assets_only_not_pose_performance", "runtime_s": time.perf_counter() - start,
              "bop_root": str(output / "bop_view"), "datasets": prepared, "totals": totals,
              "DINO_checkpoint": dino, "GeDi_checkpoint": gedi,
              "DINO_source": {"path": str(dino_source), "selected_files": dino_files,
                  "revision_hint": runtime["scientific_backend_settings"]["dinov2"].get("source_revision_hint"),
                  "scope": "Selected file hashes, not complete Git-tree verification"},
              "GeDi_source": {"path": str(gedi_source), "gedi_py": gedi_file, "actual_git_revision": gedi_revision},
              "evaluation_runtime": {"path": str(eval_path), "sha256": sha256(eval_path),
                  "toolkit_root": str(toolkit), "actual_toolkit_revision": toolkit_revision},
              "torch_home": str(torch_home), "torch_checkpoint_alias": str(giant_alias),
              "model_dataset_copies_downloads_or_moves": 0, "GT_annotation_links": 0,
              "GT_pose_access": False, "global_environment_changes": False,
              "pinned_old_assets_manifest": {"path": str(asset_path), "sha256": sha256(asset_path)}}
    atomic_json(output / "assets_manifest.json", result)
    progress(status=result["status"], totals=totals)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    root = local_path(args.root)
    output = local_path(args.output) if args.output else root / "outputs/freezev2_public_repo_20261030/base_public_v1"
    result = prepare(root, output, args.datasets)
    print(json.dumps({"status": result["status"], "output": str(output), "bop_root": result["bop_root"],
                      "totals": result["totals"], "torch_home": result["torch_home"]}, indent=2))

if __name__ == "__main__":
    main()