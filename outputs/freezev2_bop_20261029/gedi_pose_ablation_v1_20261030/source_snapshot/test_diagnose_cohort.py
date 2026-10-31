"""CPU tests of saved-match counting, failure denominator and official agreement."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_cohort as dc


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _targets():
    return [dict(scene_id=1, im_id=i, obj_id=i + 1, inst_count=1) for i in range(7)]


def _matches(targets, success):
    rows = [dict(scene_id=t["scene_id"], im_id=t["im_id"], obj_id=t["obj_id"],
                 gt_id=0, valid=True, est_id=0 if success(i) else -1, error=-1)
            for i, t in enumerate(targets)]
    # Non-target GT entries can be present in official match files but must not
    # enter this cohort's denominator. No GT poses are needed for this fixture.
    return rows + [dict(scene_id=1, im_id=0, obj_id=99, gt_id=9,
                        valid=False, est_id=-1, error=-1)]


@pytest.fixture
def official(tmp_path):
    root = tmp_path / "official_evaluation/freezev2cohort_tless-test"
    targets = _targets()
    for metric, total in dc.METRIC_COUNTS.items():
        for j in range(total):
            if metric == "vsd":
                folder = f"error=vsd_ntop=-1_delta=15.000_tau={(j // 10 + 1) * .05:.3f}"
                threshold = (j % 10 + 1) * .05
            else:
                folder = f"error={metric}_ntop=-1"
                threshold = (j + 1) * (5.0 if metric == "mspd" else .05)
            success = lambda i: (i == 1 or (i == 2 and j % 2 == 0)
                                 or (i == 3 and metric == "vsd")
                                 or (i == 4 and metric == "mssd")
                                 or (i == 5 and metric == "mspd")
                                 or (i == 6 and j == 0))
            matches = _matches(targets, success)
            tp = sum(row["valid"] and row["est_id"] >= 0 for row in matches)
            tail = f"th={threshold:.3f}_min-visib=-1.000.json"
            _write(root / folder / ("matches_" + tail), matches)
            _write(root / folder / ("scores_" + tail),
                   dict(targets_count=7, tp_count=tp, gt_count=len(matches), recall=tp / 7))
    # Expected values are independent analytical fractions of the fixture:
    # one always-correct target, a half-correct target, one metric-only target,
    # and one success out of all thresholds. Zero-prediction target stays zero.
    values = {dc.METRIC_KEYS["vsd"]: 251 / 700,
              dc.METRIC_KEYS["mssd"]: 26 / 70,
              dc.METRIC_KEYS["mspd"]: 26 / 70}
    values[dc.AR_KEY] = sum(values.values()) / 3
    _write(root / "scores_bop19.json", values)
    return root, targets, values


def test_exact_per_target_recalls_preserve_zero_output_and_official_denominators(official):
    root, targets, values = official
    result = dc.analyze_matches(root, targets, values)
    assert len(result["per_target"]) == 7
    ar = [row["AR"] for row in result["per_target"]]
    assert ar == pytest.approx([0., 1., .5, 1/3, 1/3, 1/3, .07], abs=1e-12)
    assert len(result["threshold_files"]) == 120
    assert result["metric_numerators_denominators"]["vsd"] == {
        "true_positives_across_thresholds": 251, "threshold_instance_trials": 700,
        "match_files": 100, "recall": 251 / 700}
    assert result["computed_metrics"] == pytest.approx(values, abs=1e-12)


def test_missing_zero_prediction_target_is_an_error_not_silently_dropped(official):
    root, targets, values = official
    path = next(root.rglob("matches_th*.json"))
    matches = dc.read_json(path)
    _write(path, matches[1:])
    with pytest.raises(ValueError, match="GT counts differ"):
        dc.analyze_matches(root, targets, values)


def test_missing_threshold_file_cannot_count_as_complete(official):
    root, targets, values = official
    next(root.rglob("matches_th*.json")).unlink()
    with pytest.raises(ValueError, match="exactly100 VSD"):
        dc.analyze_matches(root, targets, values)


@pytest.mark.parametrize("fault", ["duplicate_gt", "unknown_valid", "invalid_flag"])
def test_invalid_official_match_rows_fail_instead_of_changing_denominator(fault):
    targets = _targets()
    rows = _matches(targets, lambda i: False)
    if fault == "duplicate_gt": rows.append(deepcopy(rows[0]))
    elif fault == "unknown_valid": rows[-1]["valid"] = True
    else: rows[0]["valid"] = 1
    with pytest.raises(ValueError):
        dc.count_matches(rows, dc.target_counts(targets))


def test_wrong_official_numerator_is_detected(official):
    root, targets, values = official
    path = next(root.rglob("scores_th*.json"))
    scores = dc.read_json(path); scores["tp_count"] += 1; _write(path, scores)
    with pytest.raises(ValueError, match="numerator/denominator"):
        dc.analyze_matches(root, targets, values)


def test_aggregate_official_score_must_agree_to_tight_tolerance(official):
    root, targets, values = official
    wrong = dict(values); wrong[dc.METRIC_KEYS["mssd"]] += 1e-5
    with pytest.raises(ValueError, match="aggregate mssd"):
        dc.analyze_matches(root, targets, wrong)


def test_receipt_pin_location_matches_real_evaluator_and_hash_is_enforced(official):
    root, targets, values = official
    evaluation = root.parent
    receipt = dict(status="complete", source_pins={"cohort_sha256": "fixed-cohort",
                   "toolkit_revision": dc.TOOLKIT}, score_file=str(root / "scores_bop19.json"),
                   score_sha256=dc.sha256(root / "scores_bop19.json"), metrics=values,
                   saved_predictions=6)
    _write(evaluation / "evaluation_receipt.json", receipt)
    result = dc.read_arm(evaluation, targets, "fixed-cohort")
    assert result["saved_prediction_count"] == 6
    assert result["computed_metrics"] == pytest.approx(values)
    receipt["score_sha256"] = "changed"; _write(evaluation / "evaluation_receipt.json", receipt)
    with pytest.raises(ValueError, match="score file changed"):
        dc.read_arm(evaluation, targets, "fixed-cohort")


def test_pairing_and_report_keep_all_seven_targets_with_case_names(official, tmp_path, monkeypatch):
    root, targets, values = official
    analyzed = dc.analyze_matches(root, targets, values)
    cohort_path = tmp_path / "cohort.json"
    _write(cohort_path, {"targets": targets, "cases": [
        {"target": t, "case_id": f"case_{i}"} for i, t in enumerate(targets)]})
    monkeypatch.setattr(dc, "read_arm", lambda *_: deepcopy(analyzed))
    result = dc.diagnose(tmp_path, cohort_path)
    assert len(result["paired_targets"]) == 7
    assert result["paired_targets"][0]["case_id"] == "case_0"
    assert all(row["AR_delta_percentage_points"]["repair224_vs_baseline"] == 0
               for row in result["paired_targets"])
    dc.write_reports(tmp_path, result)
    dc.write_reports(tmp_path, result)  # exact repeat is idempotent
    markdown = (tmp_path / "PAIRED_TARGET_METRICS.md").read_text()
    assert "0.000000%" in markdown and "Repair476 AR" in markdown
    changed = deepcopy(result); changed["scope"] = "changed"
    with pytest.raises(ValueError, match="Existing paired report differs"):
        dc.write_reports(tmp_path, changed)
