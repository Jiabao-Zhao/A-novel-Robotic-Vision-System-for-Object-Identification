"""External predictions and BOP output tests; no GT pose is used for inference."""
from pathlib import Path
import csv
import json
import os

import numpy as np
import pytest

from freezev2_repro.masks import (
    MaskCandidate, MaskInputError, PredictedMasksUnavailable, decode_rle,
    encode_rle, load_localization_targets, load_mask_candidates,
    select_localization_candidates,
)
from freezev2_repro.evaluation import (
    PosePrediction, build_evaluation_command, preflight_official_evaluation, read_bop_csv, validate_bop_csv,
    write_bop_csv,
)


@pytest.mark.parametrize("mask", [
    np.zeros((3, 5), dtype=bool), np.ones((3, 5), dtype=bool),
    np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
])
def test_rle_non_square_and_initial_foreground(mask):
    rle = encode_rle(mask)
    assert rle["size"] == list(mask.shape)
    np.testing.assert_array_equal(decode_rle(rle), mask)
    if mask[0, 0]:
        assert rle["counts"][0] == 0


def test_compressed_rle_interoperates_with_public_pycocotools():
    mask_utils = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(17)
    # Irregular runs exercise signed deltas and a non-square image.
    for mask in [rng.random((23, 31)) > .65, np.ones((7, 13), dtype=bool)]:
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        np.testing.assert_array_equal(decode_rle(encoded), mask)
        encoded["counts"] = encoded["counts"].decode("ascii")
        np.testing.assert_array_equal(decode_rle(encoded), mask)


@pytest.mark.parametrize("rle", [
    {"size": [3, 4], "counts": [11]},
    {"size": [3, 4], "counts": [-1, 13]},
    {"size": [3, 4], "counts": [12.0]},
    {"size": [3, 4], "counts": "P"},  # continuation without terminating byte
    {"size": [3, 4], "counts": "!"},
    {"size": [0, 4], "counts": [0]},
])
def test_invalid_rle_is_rejected(rle):
    with pytest.raises(MaskInputError):
        decode_rle(rle)


def _record(**updates):
    record = dict(scene_id=1, image_id=2, category_id=3, score=.8,
                  segmentation=encode_rle(np.eye(4, dtype=bool)))
    record.update(updates)
    return record


def test_external_prediction_provenance_and_immutable_mask(tmp_path):
    path = tmp_path / "cnos.json"
    path.write_text(json.dumps([_record()]))
    candidate, = load_mask_candidates(path, segmentor="CNOS", source_id="public-release")
    assert len(candidate.source_file_sha256) == 64
    assert candidate.provenance()["source_kind"] == "predicted"
    assert candidate.mask.dtype == bool
    with pytest.raises(ValueError):
        candidate.mask[0, 0] = False
    with pytest.raises(PredictedMasksUnavailable, match="no segmentation or GT fallback"):
        load_mask_candidates(tmp_path / "absent.json", segmentor="CNOS", source_id="public-release")


def test_dataset_wide_masks_filter_before_dense_decode(tmp_path):
    path = tmp_path / "dataset_predictions.json"
    # An unselected image's RLE is deliberately malformed: it must not be decoded.
    path.write_text(json.dumps([_record(), _record(image_id=99, segmentation={})]))
    selected = load_mask_candidates(path, segmentor="CNOS", source_id="public",
                                    scene_id=1, image_id=2, object_id=3)
    assert len(selected) == 1 and selected[0].image_id == 2
    cohort = load_mask_candidates(path, segmentor="CNOS", source_id="public", target_keys={(1, 2, 3)})
    assert len(cohort) == 1 and cohort[0].record_index == 0


def test_ground_truth_and_oracle_inputs_do_not_silently_enter_benchmark(tmp_path):
    gt_path = tmp_path / "full_eval_gt" / "masks.json"
    gt_path.parent.mkdir()
    gt_path.write_text(json.dumps([_record()]))
    with pytest.raises(MaskInputError, match="ground-truth path"):
        load_mask_candidates(gt_path, segmentor="oracle", source_id="GT")
    with pytest.raises(MaskInputError, match="explicit allow_oracle"):
        load_mask_candidates(gt_path, segmentor="oracle", source_id="GT", source_kind="oracle")
    oracle = load_mask_candidates(gt_path, segmentor="oracle", source_id="GT",
                                  source_kind="oracle", allow_oracle=True)
    with pytest.raises(MaskInputError, match="cannot enter"):
        select_localization_candidates(oracle, scene_id=1, image_id=2, object_id=3, instance_count=1)
    path = tmp_path / "misdeclared.json"
    path.write_text(json.dumps([_record(cam_R_m2c=np.eye(3).ravel().tolist())]))
    with pytest.raises(MaskInputError, match="Ground-truth annotation"):
        load_mask_candidates(path, segmentor="CNOS", source_id="unknown")


def test_localization_targets_need_official_object_ids_and_counts(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps([dict(scene_id=1, im_id=2)]))
    with pytest.raises(MaskInputError, match="image-only lists are insufficient"):
        load_localization_targets(path)
    row = dict(scene_id=1, im_id=2, obj_id=3, inst_count=2)
    path.write_text(json.dumps([row]))
    target, = load_localization_targets(path)
    assert target.object_id == 3 and target.instance_count == 2
    path.write_text(json.dumps([row, row]))
    with pytest.raises(MaskInputError, match="Duplicate"):
        load_localization_targets(path)


def test_n_plus_one_per_segmentor_uses_only_segmentation_confidence():
    candidates = []
    for segmentor in ("SAM", "FastSAM"):
        for index, confidence in enumerate((.2, .9, .8, .7, .1)):
            candidates.append(MaskCandidate(1, 2, 3, confidence, np.ones((3, 4)),
                                            segmentor, "public", index))
    candidates += [MaskCandidate(1, 2, 99, 1., np.ones((3, 4)), "SAM", "public", 11),
                   MaskCandidate(1, 2, 3, 1., np.zeros((3, 4)), "SAM", "public", 12)]
    selected = select_localization_candidates(candidates, scene_id=1, image_id=2,
                                               object_id=3, instance_count=2)
    assert len(selected) == 6
    for segmentor in ("SAM", "FastSAM"):
        assert [c.score for c in selected if c.segmentor == segmentor] == [.9, .8, .7]
    # Identical masks remain independent candidates: there is no added mask NMS.
    assert all(np.array_equal(c.mask, selected[0].mask) for c in selected)


def test_localization_targets_to_filtered_prediction_fixture_end_to_end(tmp_path):
    targets_file = tmp_path / "fixed_targets.json"
    target_rows = [dict(scene_id=5, im_id=image, obj_id=1, inst_count=1) for image in (7, 20, 25)]
    target_rows.append(dict(scene_id=9, im_id=0, obj_id=1, inst_count=3))
    targets_file.write_text(json.dumps(target_rows))
    rows = [_record(scene_id=5, image_id=image, category_id=1) for image in (7, 20, 25)]
    rows += [_record(scene_id=9, image_id=0, category_id=1, score=score) for score in (.4, .9, .5, .8, .7)]
    rows += [_record(scene_id=99, image_id=0, category_id=1, segmentation={})]
    masks_file = tmp_path / "public_prediction_fixture.json"
    masks_file.write_text(json.dumps({"predictions": rows}))
    targets = load_localization_targets(targets_file)
    keys = {(t.scene_id, t.image_id, t.object_id) for t in targets}
    masks = load_mask_candidates(masks_file, segmentor="CNOS-FastSAM", source_id="fixture",
                                 target_keys=keys)
    assert len(masks) == 8  # unselected dataset-wide mask was never decoded
    pools = [select_localization_candidates(masks, scene_id=t.scene_id, image_id=t.image_id,
              object_id=t.object_id, instance_count=t.instance_count) for t in targets]
    assert [len(pool) for pool in pools] == [1, 1, 1, 4]
    assert [m.record_index for m in pools[-1]] == [4, 6, 7, 5]


def _pose(*, image_time=2.5, rotation=None, translation=None, obj=3):
    # A roll/pitch rotation demonstrates that output is unrestricted 6D.
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]) if rotation is None else rotation
    translation = np.array([.012, -.034, .567]) if translation is None else translation
    return PosePrediction(1, 2, obj, .73, rotation, translation, image_time)


def test_bop_round_trip_frame_units_and_full_image_time(tmp_path):
    path = write_bop_csv([_pose(), _pose(obj=4)], tmp_path / "freezev2_tless-test.csv")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["t"].split() == ["12", "-34", "567"]
    assert rows[0]["time"] == rows[1]["time"] == "2.5"
    actual, _ = read_bop_csv(path)
    np.testing.assert_allclose(actual.R_m2c, _pose().R_m2c)
    np.testing.assert_allclose(actual.t_m, _pose().t_m)
    with pytest.raises(FileExistsError):
        write_bop_csv([_pose()], path)
    with pytest.raises(ValueError, match="complete image wall time"):
        write_bop_csv([_pose(), _pose(image_time=3., obj=4)], tmp_path / "inconsistent.csv")


@pytest.mark.parametrize("pose", [
    _pose(rotation=np.diag([1., 1., -1.])), _pose(rotation=np.zeros((3, 3))),
    _pose(translation=np.array([np.nan, 0, 1])), _pose(image_time=-1),
])
def test_bop_invalid_pose_is_rejected(pose, tmp_path):
    with pytest.raises(ValueError):
        write_bop_csv([pose], tmp_path / "invalid.csv")
    assert not (tmp_path / "invalid.csv").exists()


def test_no_empty_prediction_success(tmp_path):
    with pytest.raises(ValueError, match="No valid poses"):
        write_bop_csv([], tmp_path / "empty.csv")


def test_official_evaluation_invocation_is_separate_from_inference(tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "scripts").mkdir(parents=True)
    (toolkit / "scripts/eval_bop19_pose.py").write_text("# official checkout required in a real run")
    result = write_bop_csv([_pose()], tmp_path / "freezev2_tless-test.csv")
    command, environment = build_evaluation_command(toolkit_root=toolkit,
        results_file=result, datasets_root=tmp_path / "bop", evaluation_root=tmp_path / "eval",
        targets_filename="test_targets_bop19.json", python_executable="/selected/bin/python")
    assert command[0] == "/selected/bin/python"
    assert command[1].endswith("eval_bop19_pose.py")
    assert command[command.index("--targets_filename") + 1] == "test_targets_bop19.json"
    assert environment["BOP_PATH"] == str((tmp_path / "bop").resolve())
    assert environment["PATH"].startswith(str(Path("/selected/bin/python").parent) + os.pathsep)
    with pytest.raises(ValueError, match="dataset-local"):
        build_evaluation_command(toolkit_root=toolkit, results_file=result,
            datasets_root=tmp_path, evaluation_root=tmp_path, targets_filename="../GT.json")


def test_actual_official_toolkit_format_check_when_configured(tmp_path):
    root = os.environ.get("FREEZEV2_BOP_TOOLKIT")
    if not root:
        pytest.skip("Set FREEZEV2_BOP_TOOLKIT to the pinned official checkout for integration check")
    result = write_bop_csv([_pose()], tmp_path / "freezev2_tless-test.csv")
    support = os.environ.get("FREEZEV2_BOP_SUPPORT_PATH")
    report = validate_bop_csv(result, root, [support] if support else [])
    assert report["official_format_valid"]
    assert report["status"] == "valid_predictions_format_only"
    assert len(report["toolkit_revision"]) == 40


def test_official_evaluation_preflight_refuses_cad_substitution_when_configured(tmp_path):
    root = os.environ.get("FREEZEV2_BOP_TOOLKIT")
    if not root:
        pytest.skip("Set FREEZEV2_BOP_TOOLKIT to the pinned official checkout for integration check")
    support = os.environ.get("FREEZEV2_BOP_SUPPORT_PATH")
    dataset = tmp_path / "view/tless"
    (dataset / "models_cad").mkdir(parents=True)
    # Files may exist, but no contents are read as GT during preflight.
    (dataset / "test_targets_bop19.json").write_text(json.dumps([
        dict(scene_id=1, im_id=2, obj_id=3, inst_count=1)]))
    result = write_bop_csv([_pose()], tmp_path / "freezev2_tless-test.csv")
    report = preflight_official_evaluation(toolkit_root=root, results_file=result,
        datasets_root=dataset.parent, targets_filename="test_targets_bop19.json", dataset="tless",
        extra_python_paths=[support] if support else [])
    assert report["status"] == "blocked"
    assert report["gt_pose_contents_read"] is False
    assert report["targets"] == 1
    model_blocker = next(b for b in report["blockers"] if b["component"] == "evaluation_models")
    assert model_blocker["missing_count"] == 30
    assert all("models_eval" in p for p in model_blocker["missing_paths"])


def test_evaluation_cli_dry_run_reports_exact_command_and_missing_models_when_configured(tmp_path, capsys):
    root = os.environ.get("FREEZEV2_BOP_TOOLKIT")
    if not root:
        pytest.skip("Set FREEZEV2_BOP_TOOLKIT to the pinned official checkout for integration check")
    import evaluate
    paths = dict(bop_toolkit=root, output_root=str(tmp_path / "output"))
    support = os.environ.get("FREEZEV2_BOP_SUPPORT_PATH")
    if support:
        paths["bop_eval_deps"] = support
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(dataset="tless", paths=paths)))
    result = write_bop_csv([_pose()], tmp_path / "freezev2_tless-test.csv")
    status = evaluate.main(["--config", str(config), "--results-file", str(result),
        "--dataset-view", str(tmp_path / "evaluation_view"), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert status == 2 and report["status"] == "blocked_evaluation_inputs"
    assert report["command"][1].endswith("eval_bop19_pose.py")
    assert report["format"]["official_format_valid"]
    assert any(b["component"] == "evaluation_models" for b in report["preflight"]["blockers"])
    assert not (tmp_path / "output").exists()
