"""Evaluation-only audits with synthetic saved ledgers; never runs a GPU metric."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import complete_evaluation as helper
from freezev2_repro.durable import atomic_json, worker_lock
from freezev2_repro.owned_process import start_ticks


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, dataset="tless"):
    run = tmp_path / "run"
    targets = [dict(scene_id=1, im_id=1, obj_id=1, inst_count=1),
               dict(scene_id=1, im_id=2, obj_id=2, inst_count=2),
               dict(scene_id=1, im_id=2, obj_id=27, inst_count=2)]
    target = tmp_path / "official_targets.json"
    _json(target, targets)
    asset = dict(targets=dict(path=str(target), sha256=helper.sha256(target)),
                 masks=dict(sha256="m" * 64), target_records=3,
                 requested_instances=5, official_target_images=2,
                 cad_objects=3,
                 evaluation=dict(ready=dataset in helper.PUBLIC))
    manifest = dict(method_identity={"method": "unchanged-base", "code": {"frozen.py": "a" * 64}},
                    original_input_manifest_sha256="i" * 64, camera_snapshot_digest="c" * 64)
    run_identity = dict(method_digest=helper.digest(manifest["method_identity"]), dataset=dataset,
        targets_sha256=asset["targets"]["sha256"], mask_sha256=asset["masks"]["sha256"],
        original_input_manifest_sha256=manifest["original_input_manifest_sha256"],
        camera_snapshot_digest=manifest["camera_snapshot_digest"])
    def pose(image, obj, elapsed):
        return dict(scene_id=1, image_id=image, object_id=obj, score=.75,
                    R_m2c=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    t_m=[.012, -.034, .567], image_time_s=elapsed)
    first = dict(identity=dict(run=run_identity, scene_id=1, image_id=1, target_rows=[[1, 1]]),
        runtime_s=1.2, targets=[dict(object_id=1, requested_instances=1, status="completed", valid_predictions=1)],
        predictions=[pose(1, 1, 1.2)])
    second = dict(identity=dict(run=run_identity, scene_id=1, image_id=2, target_rows=[[2, 2], [27, 2]]),
        runtime_s=2.5, targets=[dict(object_id=2, requested_instances=2, status="failed", valid_predictions=1, reason="candidate_failure"),
                              dict(object_id=27, requested_instances=2, status="blocked", valid_predictions=0, reason="CAD_feature_cache_rejected")],
        predictions=[pose(2, 2, 2.5)])
    first_path = run / "datasets" / dataset / "images/000001_000001/complete.json"
    second_path = run / "datasets" / dataset / "images/000001_000002/partial.json"
    _json(first_path, first)
    _json(second_path, second)
    directory = run / "datasets" / dataset / "predictions"
    directory.mkdir()
    result = directory / f"freezev2base_{dataset}-test.csv"
    with result.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for saved in first["predictions"] + second["predictions"]:
            writer.writerow([saved["scene_id"], saved["image_id"], saved["object_id"], format(saved["score"], ".17g"),
                " ".join(format(v, ".17g") for row in saved["R_m2c"] for v in row),
                " ".join(format(v * 1000., ".17g") for v in saved["t_m"]), format(saved["image_time_s"], ".17g")])
    export = directory / "export_receipt.json"
    _json(export, dict(source="our_inference", csv_sha256=helper.sha256(result), target_sha256=asset["targets"]["sha256"],
                       expected_targets=3, ledger_targets=3, poses=2))
    return SimpleNamespace(run=run, asset=asset, manifest=manifest, target=target, csv=result,
                           export=export, first_path=first_path, second_path=second_path, dataset=dataset)


def audit(value):
    return helper.audit_dataset(value.run, value.dataset, value.asset, value.manifest)


def test_full_denominator_includes_failed_blocked_and_valid_partial_poses(tmp_path):
    value = fixture(tmp_path)
    result = audit(value)
    assert result["coverage_complete"] and result["export_valid"]
    assert (result["expected_targets"], result["attempted_targets"], result["expected_instances"]) == (3, 3, 5)
    assert (result["completed_targets"], result["failed_targets"], result["blocked_targets"]) == (1, 1, 1)
    assert result["valid_pose_predictions"] == 2 and result["recorded_inference_failures"]
    assert result["target_reason_counts"] == [
        dict(status="blocked", reason="CAD_feature_cache_rejected", targets=1),
        dict(status="failed", reason="candidate_failure", targets=1)]
    assert not result["strict_execution_complete"]


def test_effective_complete_receipt_overrides_old_partial(tmp_path):
    value = fixture(tmp_path)
    _json(value.first_path.with_name("partial.json"), {"ignored": "old interrupted image receipt"})
    assert audit(value)["coverage_complete"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "instance_count", "camera_identity"])
def test_coverage_protocol_violations_are_blocked(tmp_path, mutation):
    value = fixture(tmp_path)
    row = helper.read_json(value.second_path)
    if mutation == "missing":
        value.second_path.unlink()
    elif mutation == "duplicate":
        _json(value.first_path.parent.with_name("duplicate") / "complete.json", helper.read_json(value.first_path))
    elif mutation == "unknown":
        row["targets"][-1]["object_id"] = 99
        row["identity"]["target_rows"][-1][0] = 99
        _json(value.second_path, row)
    elif mutation == "instance_count":
        row["targets"][-1]["requested_instances"] = 3
        row["identity"]["target_rows"][-1][1] = 3
        _json(value.second_path, row)
    else:
        row["identity"]["run"]["camera_snapshot_digest"] = "changed"
        _json(value.second_path, row)
    assert not audit(value)["coverage_complete"]


@pytest.mark.parametrize("update_export_hash", [False, True])
def test_csv_must_match_both_export_hash_and_effective_ledger_poses(tmp_path, update_export_hash):
    value = fixture(tmp_path)
    value.csv.write_text(value.csv.read_text().replace("567", "568"))
    if update_export_hash:
        export = helper.read_json(value.export)
        export["csv_sha256"] = helper.sha256(value.csv)
        _json(value.export, export)
    assert not audit(value)["export_valid"]


def make_receipt(value, *, original=True):
    score = value.run / "datasets" / value.dataset / "official_evaluation/probe/scores_bop19.json"
    scores = {key: .25 for key in helper.METRICS}
    _json(score, scores)
    receipt = value.run / "datasets" / value.dataset / ("evaluation_receipt.json" if original else "evaluation_with_recorded_failures.json")
    _json(receipt, dict(status="complete", csv_sha256=helper.sha256(value.csv), target_sha256=value.asset["targets"]["sha256"],
                        toolkit_revision="revision", score_file=str(score), score_sha256=helper.sha256(score),
                        official_pose_metrics={key: .9 for key in helper.METRICS}, runtime_s=1.))
    return receipt, score


@pytest.mark.parametrize("field", [None, "csv_sha256", "target_sha256", "toolkit_revision", "score_sha256"])
def test_original_metric_receipt_reuse_requires_all_pins(tmp_path, field):
    value = fixture(tmp_path)
    receipt, score = make_receipt(value)
    if field:
        contents = helper.read_json(receipt)
        contents[field] = "wrong"
        _json(receipt, contents)
    result = helper.reusable_receipt(receipt, audit(value), {"toolkit_revision": "revision"}, receipt.parent)
    if field:
        assert result is None
    else:
        assert result["official_pose_metrics"]["bop19_average_recall"] == .25  # actual score file, not edited receipt numbers


def test_invalid_receipt_json_is_not_reused(tmp_path):
    value = fixture(tmp_path)
    receipt, _ = make_receipt(value)
    receipt.write_text("{broken")
    assert helper.reusable_receipt(receipt, audit(value), {"toolkit_revision": "revision"}, receipt.parent) is None


def test_protocol_keeps_official_inst_count_visibility_semantics():
    protocol = dict(datasets=list(helper.DATASETS), task="BOP19_model_based_6D_localization")
    runtime = dict(targets_filename="test_targets_bop19.json", renderer_type="vispy", protocol=dict(visib_gt_min=-1))
    helper.check_protocol(protocol, runtime)
    runtime["protocol"]["visib_gt_min"] = .1
    with pytest.raises(ValueError, match="k-most-visible"):
        helper.check_protocol(protocol, runtime)


def test_onboarding_runtime_separates_completed_and_partial_checkpointed_work(tmp_path):
    root = tmp_path / "run/datasets/tless/onboarding"
    _json(root / "obj_000001/geometry/complete.json", dict(runtime_s=10.))
    _json(root / "obj_000001/features/complete.json", dict(runtime_s=20.))
    _json(root / "obj_000027/geometry/complete.json", dict(runtime_s=11.))
    _json(root / "obj_000027/features/progress.json", dict(active_runtime_s=12.))
    result = helper.onboarding_runtime_summary(tmp_path / "run", "tless", 30)
    assert result["completed_stage_active_runtime_s"] == 41.
    assert result["partial_checkpointed_active_runtime_s"] == 12.
    assert result["stages"]["geometry"]["completed_CADs"] == 2
    assert result["stages"]["features"]["completed_CADs"] == 1
    assert "not fully logged" in result["scope"]


def evaluator(value, tmp_path):
    result = helper.CompletionEvaluator.__new__(helper.CompletionEvaluator)
    result.root, result.run_manifest = value.run, value.manifest
    result.assets = {"datasets": {value.dataset: value.asset}}
    result.toolkit = tmp_path / "toolkit"
    result.pins = {"helper_sha256": helper.sha256(helper.__file__), "primary_source_manifest_sha256": "frozen", "toolkit_revision": "revision"}
    view = tmp_path / "view"
    _json(view / value.dataset / "test_targets_bop19.json", helper.read_json(value.target))
    # Identity is byte-level; the official view must use the exact target bytes.
    (view / value.dataset / "test_targets_bop19.json").write_bytes(value.target.read_bytes())
    result.runtime = dict(toolkit_revision="revision", dataset_view=str(view), targets_filename="test_targets_bop19.json", python="evaluation-python", environment={})
    result.atomic, result.heartbeat = atomic_json, lambda **kwargs: None
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return ["official", "--targets_filename", kwargs["targets_filename"]], {}
    def owned(command, environment, folder, lock_stream, heartbeat):
        assert lock_stream == "inherited-worker-lock"
        assert environment["OMP_NUM_THREADS"] == "4"
        _json(Path(folder) / value.csv.stem / "scores_bop19.json", {key: .25 for key in helper.METRICS})
        return 0
    result.frozen = dict(evaluation=SimpleNamespace(build_evaluation_command=build), owned=SimpleNamespace(run_owned=owned))
    return result, calls, view


def test_failure_inclusive_metrics_keep_full_targets_and_original_outputs_unchanged(tmp_path):
    value = fixture(tmp_path)
    _json(value.run / "dataset_states.json", {value.dataset: {"CAD_blockers": {"27": "strict finite-output guard"}}})
    runner, calls, _ = evaluator(value, tmp_path)
    original_csv, original_export = value.csv.read_bytes(), value.export.read_bytes()
    result = runner.dataset(value.dataset, "inherited-worker-lock")
    assert result["status"] == "complete_with_recorded_inference_failures"
    assert result["expected_instances"] == 5 and result["valid_pose_predictions"] == 2
    assert result["original_CAD_blockers"] == {"27": "strict finite-output guard"}
    assert "Our finite-output guard" in result["CAD_blocker_interpretation"]
    assert calls[0]["targets_filename"] == "test_targets_bop19.json"
    assert calls[0]["results_file"] == str(value.csv)
    assert value.csv.read_bytes() == original_csv and value.export.read_bytes() == original_export
    assert not (value.csv.parent.parent / "evaluation_receipt.json").exists()


def test_valid_primary_receipt_avoids_duplicate_calculation(tmp_path):
    value = fixture(tmp_path)
    original, _ = make_receipt(value)
    original_bytes = original.read_bytes()
    runner, calls, _ = evaluator(value, tmp_path)
    result = runner.dataset(value.dataset, "inherited-worker-lock")
    assert result["reused_receipt"] == str(original) and not calls
    assert original.read_bytes() == original_bytes


def test_missing_coverage_or_changed_view_denominator_never_runs_evaluator(tmp_path):
    value = fixture(tmp_path)
    runner, calls, view = evaluator(value, tmp_path)
    value.second_path.unlink()
    assert runner.dataset(value.dataset, "inherited-worker-lock")["status"] == "blocked_ledger_coverage"
    assert not calls
    (view / value.dataset / "test_targets_bop19.json").write_text("[]")
    assert runner.dataset(value.dataset, "inherited-worker-lock")["status"] == "blocked_official_target_identity"
    assert not calls


def test_private_GT_never_runs_metrics_or_submits_remotely(tmp_path):
    value = fixture(tmp_path, "hb")
    runner, calls, _ = evaluator(value, tmp_path)
    assert runner.dataset("hb", "inherited-worker-lock")["status"] == "unavailable_private_GT"
    assert not calls


@pytest.mark.skipif(os.name != "posix", reason="Frozen suite lock ownership is Linux/WSL")
def test_queue_probe_ignores_main_locks_and_once_fails_closed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    frozen = dict(durable=SimpleNamespace(worker_lock=worker_lock, atomic_json=atomic_json),
                  owned=SimpleNamespace(start_ticks=start_ticks), source_manifest_sha256="frozen")
    monkeypatch.setattr(helper, "load_frozen", lambda root: frozen)
    original_status = run / "status.json"
    original_status.write_text("primary unchanged")
    with worker_lock(run / "supervisor.lock"):
        assert helper.main(["--run-root", str(run), "--probe-queue-lock"]) == 0
        assert not (run / "failure_inclusive_evaluation_process.json").exists()
        assert helper.main(["--run-root", str(run), "--once"]) == 3
    assert helper.read_json(run / "failure_inclusive_evaluation_status.json")["status"] == "blocked_live_main_owner"
    assert original_status.read_text() == "primary unchanged"
    with worker_lock(run / "post_evaluation.lock"):
        assert helper.main(["--run-root", str(run), "--probe-queue-lock"]) == 3
