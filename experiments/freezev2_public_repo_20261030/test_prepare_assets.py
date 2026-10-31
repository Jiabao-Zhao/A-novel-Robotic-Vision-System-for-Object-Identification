from pathlib import Path
import json
import os

import numpy as np
from PIL import Image
import pytest

import prepare_assets as subject

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Run symlink asset-view tests in the existing WSL venv")


def image(path, values, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, mode=mode).save(path)
    return path


def fixture_dataset(tmp_path, dataset="lmo"):
    source = tmp_path / "source"
    models = source / dataset / ("models_cad" if dataset == "tless" else "models")
    models.mkdir(parents=True)
    (models / "models_info.json").write_text('{"1":{"diameter":100}}')
    (models / "obj_000001.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    scene = source / dataset / subject.SPLITS[dataset] / "000002"
    scene.mkdir(parents=True)
    camera = {"3": {"cam_K": [100, 0, 1, 0, 100, 1, 0, 0, 1], "depth_scale": .1}}
    (scene / "scene_camera.json").write_text(json.dumps(camera))
    (scene / "scene_gt.json").write_text("This GT pose file must never be read")
    (scene / "scene_gt_info.json").write_text("This GT info file must never be read")
    image(scene / "rgb/000003.png", np.arange(36, dtype=np.uint8).reshape(3, 4, 3))
    image(scene / "depth/000003.png", np.arange(12, dtype=np.uint16).reshape(3, 4))
    targets = tmp_path / "official_targets.json"
    targets.write_text(json.dumps([{"scene_id": 2, "im_id": 3, "obj_id": 1, "inst_count": 1}]))
    entry = {"targets": {"path": str(targets), "sha256": subject.sha256(targets)},
             "target_records": 1, "official_target_images": 1}
    return source, scene, entry


def test_tiff_float_depth_is_not_transcoded_when_named_png(tmp_path):
    values = np.asarray([[1.25, 8.5], [0, float("nan")]], dtype=np.float32)
    source = image(tmp_path / "original.tif", values)
    alias = subject.ensure_alias(source, tmp_path / "depth/000001.png")
    report = subject.verify_image_alias(source, alias)
    assert report["source_format"] == report["alias_detected_format"] == "TIFF"
    assert report["source_mode"] == report["alias_mode"] == "F"
    assert report["dtype"] == "float32"
    assert report["all_pixels_equal"] and not report["transcoding_performed"]
    assert subject.sha256(source) == subject.sha256(alias)


def test_gray_tiff_rgb_conversion_is_exact(tmp_path):
    values = np.arange(12, dtype=np.uint8).reshape(3, 4)
    source = image(tmp_path / "gray.tif", values)
    alias = subject.ensure_alias(source, tmp_path / "rgb/000001.png")
    report = subject.verify_image_alias(source, alias, rgb=True)
    assert report["source_mode"] == report["alias_mode"] == "L"
    assert report["RGB_converted_pixel_sha256"] == subject.pixel_sha(np.repeat(values[:, :, None], 3, axis=2))


def test_jpeg_format_magic_survives_png_alias(tmp_path):
    source = image(tmp_path / "rgb.jpg", np.full((3, 4, 3), 80, np.uint8))
    alias = subject.ensure_alias(source, tmp_path / "000003.png")
    report = subject.verify_image_alias(source, alias, rgb=True)
    assert report["alias_detected_format"] == "JPEG"
    assert report["source_sha256"] == subject.sha256(alias)


def test_alias_reuse_is_idempotent(tmp_path):
    source = tmp_path / "source.txt"; source.write_text("preserve")
    destination = tmp_path / "view/target.txt"
    subject.ensure_alias(source, destination)
    subject.ensure_alias(source, destination)
    assert destination.is_symlink() and destination.read_text() == "preserve"


def test_alias_refuses_different_original(tmp_path):
    a, b, destination = tmp_path / "a", tmp_path / "b", tmp_path / "alias"
    a.write_text("a"); b.write_text("b")
    subject.ensure_alias(a, destination)
    with pytest.raises(ValueError, match="different source"):
        subject.ensure_alias(b, destination)
    assert destination.resolve() == a.resolve()


def test_alias_does_not_overwrite_regular_file(tmp_path):
    a, destination = tmp_path / "a", tmp_path / "existing"
    a.write_text("a"); destination.write_text("historical evidence")
    with pytest.raises(ValueError, match="non-symlink"):
        subject.ensure_alias(a, destination)
    assert destination.read_text() == "historical evidence"


@pytest.mark.parametrize("name", ["scene_gt.json", "scene_gt_info.json", "scene_gt_coco.json"])
def test_ground_truth_alias_is_forbidden(tmp_path, name):
    source = tmp_path / name; source.write_text("do not link")
    with pytest.raises(ValueError, match="GT annotation"):
        subject.ensure_alias(source, tmp_path / "view" / name)


def test_identity_refusal_preserves_prior_record(tmp_path):
    path = tmp_path / "identity.json"
    subject.ensure_identity(path, {"source": "a"})
    before = path.read_bytes()
    subject.ensure_identity(path, {"source": "a"})
    with pytest.raises(ValueError, match="identity differs"):
        subject.ensure_identity(path, {"source": "b"})
    assert path.read_bytes() == before


def test_gt_free_view_exact_source_pixels_and_camera(tmp_path):
    source, scene, entry = fixture_dataset(tmp_path)
    view = tmp_path / "bop_view"
    report = subject.prepare_dataset(source, view, "lmo", entry)
    assert report["images"] == report["target_rows"] == report["requested_instances"] == 1
    assert report["GT_annotation_links"] == 0
    assert not (view / "lmo/test/000002/scene_gt.json").exists()
    assert not (view / "lmo/test/000002/scene_gt_info.json").exists()
    assert (view / "lmo/models").resolve() == (source / "lmo/models").resolve()
    assert (view / "lmo/test/000002/scene_camera.json").resolve() == (scene / "scene_camera.json").resolve()
    frame = report["frames"][0]
    assert frame["depth_scale_mm"] == .1
    assert frame["rgb"]["all_pixels_equal"] and frame["depth"]["all_pixels_equal"]
    assert frame["rgb"]["source_sha256"] == subject.sha256(scene / "rgb/000003.png")


def test_tless_split_and_cad_models_alias(tmp_path):
    source, scene, entry = fixture_dataset(tmp_path, "tless")
    view = tmp_path / "bop_view"
    report = subject.prepare_dataset(source, view, "tless", entry)
    assert report["source_split"] == "test_primesense"
    assert report["view_split"] == "test"
    assert (view / "tless/models").resolve() == (source / "tless/models_cad").resolve()
    assert (view / "tless/test/000002/rgb/000003.png").resolve() == (scene / "rgb/000003.png").resolve()


def test_target_hash_change_is_rejected(tmp_path):
    source, scene, entry = fixture_dataset(tmp_path)
    entry["targets"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target source changed"):
        subject.prepare_dataset(source, tmp_path / "view", "lmo", entry)


def test_duplicate_target_is_rejected():
    row = {"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 1}
    with pytest.raises(ValueError, match="Duplicate"):
        subject.exact_targets([row, row])


@pytest.mark.parametrize("key,value", [("scene_id", -1), ("im_id", True), ("obj_id", 0), ("inst_count", 0)])
def test_invalid_target_values_are_rejected(key, value):
    row = {"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 1}
    row[key] = value
    with pytest.raises(ValueError, match="Invalid"):
        subject.exact_targets([row])


def test_pixel_hash_records_dtype_and_shape():
    a = np.asarray([1, 2, 3, 4], dtype=np.uint8)
    assert subject.pixel_sha(a) != subject.pixel_sha(a.reshape(2, 2))
    assert subject.pixel_sha(a) != subject.pixel_sha(a.astype(np.uint16))


def test_inference_view_rejects_unexpected_gt(tmp_path):
    (tmp_path / "scene_gt.json").write_text("should be absent")
    with pytest.raises(ValueError, match="Unexpected GT"):
        subject.reject_gt_links(tmp_path)


def test_windows_path_translation():
    assert subject.local_path(r"D:\BOP_SAM3_Pilot\data") == Path("/mnt/d/BOP_SAM3_Pilot/data")

def test_existing_mask_source_is_pinned_without_copy(tmp_path):
    source, scene, entry = fixture_dataset(tmp_path)
    masks = tmp_path / "public_masks.json"
    masks.write_text("[]")
    entry["masks"] = {"path": str(masks), "sha256": subject.sha256(masks), "segmentor": "CNOS-FastSAM", "source_kind": "predicted"}
    report = subject.prepare_dataset(source, tmp_path / "view", "lmo", entry)
    assert report["masks"]["path"] == str(masks)
    assert report["masks"]["actual_sha256"] == entry["masks"]["sha256"]
    assert masks.read_text() == "[]"


def test_changed_mask_source_is_rejected(tmp_path):
    source, scene, entry = fixture_dataset(tmp_path)
    masks = tmp_path / "public_masks.json"
    masks.write_text("[]")
    entry["masks"] = {"path": str(masks), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Predicted masks checksum"):
        subject.prepare_dataset(source, tmp_path / "view", "lmo", entry)