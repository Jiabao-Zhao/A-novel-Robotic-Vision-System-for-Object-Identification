import hashlib
import json
import re
from pathlib import Path

import numpy as np

from CADPointCloudRegistration import CADPointCloudRegistration


LIBERO_CAD_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1] / "CAD" / "libero_object_library.json"
)


def retrieve_libero_cad(object_type, catalog_path=LIBERO_CAD_LIBRARY_PATH, assets_root=None):
    """Resolve a semantic VLM class to a declared LIBERO CAD prior."""
    records = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    normalized_type = _normalize_name(object_type)
    matches = []
    for record in records:
        names = [record["cad_name"], *record.get("aliases", [])]
        if any(_name_matches(normalized_type, _normalize_name(name)) for name in names):
            matches.append(record)
    if len(matches) != 1:
        available = ", ".join(record["cad_name"] for record in records)
        raise LookupError(
            f"Expected exactly one CAD match for {object_type!r}; found {len(matches)}. "
            f"Available CAD models: {available or 'none'}."
        )

    result = dict(matches[0])
    candidate_roots = _asset_roots(assets_root)
    candidate_paths = [root / result["asset_relative_path"] for root in candidate_roots]
    cad_path = next((path for path in candidate_paths if path.is_file()), None)
    if cad_path is None:
        attempted = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            "Declared LIBERO CAD asset is missing. Checked: "
            f"{attempted}. Instantiate a LIBERO environment once to download its assets."
        )
    expected_sha256 = result.get("asset_sha256")
    if not expected_sha256:
        raise ValueError(f"CAD catalog entry {result['cad_id']!r} has no asset_sha256.")
    actual_sha256 = _sha256(cad_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"LIBERO CAD asset checksum mismatch for {cad_path}. "
            f"Expected {expected_sha256}, received {actual_sha256}. "
            "The installed asset revision may differ from the tested catalog."
        )
    result["cad_path"] = str(cad_path)
    result["verified_asset_sha256"] = actual_sha256
    return result


def register_libero_cad_to_observation(
    object_type,
    object_id,
    localization,
    world_T_camera,
    output_dir,
    catalog_path=LIBERO_CAD_LIBRARY_PATH,
    assets_root=None,
    registrar=None,
):
    """Align a semantic CAD prior to one depth-localized camera-frame cloud."""
    if localization.get("frame") != "camera":
        raise ValueError(
            "CAD registration requires localization in the camera frame; "
            f"received {localization.get('frame')!r}."
        )
    plane_model = np.asarray(localization.get("plane_model"), dtype=float)
    if (
        plane_model.shape != (4,)
        or not np.all(np.isfinite(plane_model))
        or np.linalg.norm(plane_model[:3]) <= 0.0
    ):
        raise ValueError("localization plane_model must be finite [a, b, c, d].")

    localized_object = _localized_object(localization, object_id)
    observed_cloud_path = Path(localized_object["pointcloud_path"])
    if not observed_cloud_path.is_file():
        raise FileNotFoundError(f"Localized object cloud is missing: {observed_cloud_path}")

    world_T_camera = np.asarray(world_T_camera, dtype=float)
    if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
        raise ValueError("world_T_camera must be a finite 4x4 matrix.")

    cad = retrieve_libero_cad(
        object_type,
        catalog_path=catalog_path,
        assets_root=assets_root,
    )
    registrar = CADPointCloudRegistration() if registrar is None else registrar
    output_dir = Path(output_dir)
    registration = registrar.run(
        cad_path=cad["cad_path"],
        observed_cloud_path=observed_cloud_path,
        plane_model=plane_model.tolist(),
        cad_metadata=cad,
        output_dir=output_dir,
    )

    camera_T_cad = np.asarray(registration["T_observed_from_cad"], dtype=float)
    if camera_T_cad.shape != (4, 4) or not np.all(np.isfinite(camera_T_cad)):
        raise RuntimeError("CAD registration returned an invalid T_observed_from_cad matrix.")
    cad_center_m = np.asarray(registration["cad_center_m"], dtype=float)
    if cad_center_m.shape != (3,) or not np.all(np.isfinite(cad_center_m)):
        raise RuntimeError("CAD registration returned an invalid CAD center.")
    rmse_m = float(registration["constrained_rmse_m"])
    max_rmse_m = float(cad["max_registration_rmse_m"])
    warnings = list(registration.get("warnings", []))
    if not np.isfinite(rmse_m) or rmse_m > max_rmse_m or warnings:
        raise RuntimeError(
            "Milk CAD registration was rejected before task execution: "
            f"RMSE={rmse_m:.6f} m (maximum {max_rmse_m:.6f} m), "
            f"warnings={warnings}."
        )

    world_T_cad = world_T_camera @ camera_T_cad
    registered_center_camera_m = _transform_point(camera_T_cad, cad_center_m)
    registered_center_world_m = _transform_point(world_T_cad, cad_center_m)
    result = {
        "object_id": object_id,
        "object_type": object_type,
        "cad_id": cad["cad_id"],
        "cad_name": cad["cad_name"],
        "cad_path": cad["cad_path"],
        "cad_source": cad["source"],
        "cad_asset_repository": cad["asset_repository"],
        "cad_asset_revision": cad["asset_revision"],
        "cad_asset_sha256": cad["verified_asset_sha256"],
        "cad_license": cad["license"],
        "cad_scale_to_m": float(cad["scale_to_m"]),
        "cad_up_axis": cad["cad_up_axis"],
        "evaluation_condition": cad["evaluation_condition"],
        "exact_rendering_cad_prior_used": bool(cad["exact_rendering_cad_prior_used"]),
        "observed_cloud_path": str(observed_cloud_path),
        "camera_T_cad": camera_T_cad.tolist(),
        "world_T_cad": world_T_cad.tolist(),
        "cad_center_cad_m": cad_center_m.tolist(),
        "registered_center_camera_m": registered_center_camera_m.tolist(),
        "registered_center_world_m": registered_center_world_m.tolist(),
        "registered_center_definition": "transformed center of the sampled CAD axis-aligned box",
        "registration_method": registration.get("method"),
        "selected_candidate_name": registration.get("selected_candidate_name"),
        "constrained_rmse_m": rmse_m,
        "max_accepted_rmse_m": max_rmse_m,
        "registration_accepted": True,
        "cad_extent_m": registration.get("cad_extent_m"),
        "observed_extent_m": registration.get("observed_extent_m"),
        "registration_warnings": warnings,
        "registration_result_path": registration["result_path"],
        "aligned_cad_cloud_path": registration["aligned_cad_cloud_path"],
        "augmented_cloud_path": registration["augmented_cloud_path"],
        "simulator_object_identity_or_pose_used": False,
        "pick_position_source": "registered CAD axis-aligned-box center",
        "transform_convention": {
            "camera_T_cad": (
                "maps CAD-local points after scale_to_m into the agent camera frame"
            ),
            "world_T_cad": "world_T_camera @ camera_T_cad",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cad_to_observation.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _localized_object(localization, object_id):
    matches = [
        item
        for item in localization.get("objects", [])
        if item.get("object_id") == object_id
    ]
    if len(matches) != 1:
        raise LookupError(
            f"Expected one localized object for {object_id!r}; found {len(matches)}."
        )
    return matches[0]


def _asset_roots(explicit_root):
    if explicit_root is not None:
        return [Path(explicit_root)]
    try:
        from libero.libero import get_libero_path
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "LIBERO is required to resolve the installed CAD asset directory."
        ) from error
    roots = [
        Path.home() / ".cache" / "libero" / "assets",
        Path(get_libero_path("assets")),
    ]
    return list(dict.fromkeys(roots))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform_point(transform, point):
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    return (np.asarray(transform, dtype=float) @ homogeneous)[:3]


def _normalize_name(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _name_matches(object_type, alias):
    return object_type == alias or f" {alias} " in f" {object_type} "
