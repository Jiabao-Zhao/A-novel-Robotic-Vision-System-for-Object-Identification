"""Reporting-only gates using synthetic ledgers and saved score JSONs; no GPU/GT."""
from pathlib import Path

import pytest

import complete_evaluation as helper
import finalize_report as report
from test_complete_evaluation import fixture, audit, make_receipt, _json


def test_all_seven_exact_coverage_passes_and_partial_ycbv_blocks_without_outputs(tmp_path):
    assets, fixtures = {"datasets": {}}, {}
    for name in report.DATASETS:
        value = fixture(tmp_path, name)
        assets["datasets"][name], fixtures[name] = value.asset, value
    value = fixtures["ycbv"]
    assert set(report.complete_audits(value.run, assets, value.manifest, helper)) == set(report.DATASETS)
    value.second_path.unlink()
    with pytest.raises(ValueError, match="ycbv.*coverage/export not finalized"):
        report.complete_audits(value.run, assets, value.manifest, helper)
    assert not (value.run / "FINAL_REPORT.md").exists()
    assert not (value.run / "final_results.json").exists()


def test_duplicate_target_receipt_cannot_be_reported_as_complete(tmp_path, monkeypatch):
    value = fixture(tmp_path)
    duplicate = value.first_path.parent.with_name("duplicate") / "complete.json"
    _json(duplicate, helper.read_json(value.first_path))
    monkeypatch.setattr(report, "DATASETS", ("tless",))
    with pytest.raises(ValueError, match="Duplicate effective ledger target"):
        report.complete_audits(value.run, {"datasets": {"tless": value.asset}}, value.manifest, helper)


def test_cross_platform_ledger_digest_is_same_bound_file_set(tmp_path):
    value = fixture(tmp_path)
    result = audit(value)
    windows = [dict(row, path=row["path"].replace("/", "\\")) for row in result["ledger_files"]]
    assert report.ledger_digest_matches(helper.digest(windows), result, helper)
    windows[0]["sha256"] = "changed"
    assert not report.ledger_digest_matches(helper.digest(windows), result, helper)


def test_requested_instances_and_partial_poses_are_separate_from_target_rows(tmp_path):
    value = fixture(tmp_path)
    details = report.ledger_details(value.run, audit(value), helper)
    assert details["requested_instances_by_target_status"] == {"completed": 1, "failed": 2, "blocked": 2}
    assert details["retained_poses_by_target_status"] == {"completed": 1, "failed": 1, "blocked": 0}
    assert details["missing_requested_predictions"] == 3


def test_official_runtime_uses_csv_image_denominator_not_all_attempted_images(tmp_path):
    value = fixture(tmp_path)
    result = audit(value)
    result.update(expected_images=3, attempted_images=3, inference_runtime_s=6.)
    timing = report.timing_summary(result, helper, {"bop19_average_time_per_image": 1.85})
    assert timing["CSV_images_with_predictions"] == 2 and timing["images_without_CSV_predictions"] == 1
    assert timing["all_attempted_image_mean_s"] == 2.
    assert timing["official_BOP_image_mean_s"] == 1.85
    with pytest.raises(ValueError, match="time metric"):
        report.timing_summary(result, helper, {"bop19_average_time_per_image": 2.})


def failure_receipt(value):
    original, _ = make_receipt(value)
    result = audit(value)
    saved = dict(**result, **{k:v for k,v in helper.read_json(original).items() if k not in result},
                 source_pins={"primary_source_manifest_sha256": "primary", "helper_sha256": "helper"})
    path = original.with_name("evaluation_with_recorded_failures.json")
    _json(path, saved)
    return path, result


@pytest.mark.parametrize("corrupt", [None, "score_sha256", "target_sha256", "toolkit_revision", "ledger_digest", "expected_instances", "source_pins"])
def test_failure_inclusive_receipt_reuse_requires_counts_bytes_and_source_pins(tmp_path, corrupt):
    value = fixture(tmp_path)
    path, result = failure_receipt(value)
    if corrupt:
        saved = helper.read_json(path)
        saved[corrupt] = {} if corrupt == "source_pins" else "changed"
        _json(path, saved)
    kwargs = dict(root=value.run, name=value.dataset, audit=result,
                  runtime={"toolkit_revision": "revision"}, helper=helper,
                  source_pins={"primary_source_manifest_sha256": "primary", "helper_sha256": "helper"})
    if corrupt:
        with pytest.raises(ValueError):
            report.verify_failure_inclusive_receipt(**kwargs)
    else:
        assert report.verify_failure_inclusive_receipt(**kwargs)["official_pose_metrics"]["bop19_average_recall"] == .25


def test_visual_evidence_checks_reviewed_raw_artifact_bytes(tmp_path):
    value = fixture(tmp_path)
    result = audit(value)
    _json(value.run / "tless_full_pass_audit_20261030.json", {"audit": result})
    cases = []
    for index in range(3):
        image = value.run / f"overlay{index}.png"
        image.write_bytes(b"synthetic artifact")
        cases.append(dict(overlay=str(image), sha256=helper.sha256(image)))
    _json(value.run / "validation_visual_review.json", dict(scope="synthetic", cases=cases))
    assert len(report.reviewed_evidence(value.run, "tless", result, helper)["reviewed_artifact_bindings"]) == 3
    Path(cases[0]["overlay"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="visual artifact changed"):
        report.reviewed_evidence(value.run, "tless", result, helper)


def test_main_refuses_partial_report_without_writing_or_changing_primary_files(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    primary = root / "status.json"
    primary.write_text("unchanged primary")
    monkeypatch.setattr(report, "load_helper", lambda root: (helper, {}))
    def blocked(*args):
        raise ValueError("ycbv: coverage/export not finalized")
    monkeypatch.setattr(report, "build_report", blocked)
    assert report.main(["--run-root", str(root)]) == 2
    assert set(path.name for path in root.iterdir()) == {"status.json"}
    assert primary.read_text() == "unchanged primary"
