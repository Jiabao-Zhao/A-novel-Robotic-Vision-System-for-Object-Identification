"""Analyze and calibrate frozen LIBERO raw VLM label likelihoods.

This script never calls a VLM. It fits class-agnostic calibration mappings on
the frozen calibration split, selects one mapping on validation Brier score,
and evaluates that frozen mapping once on the held-out test split.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.analyze_vlm_uncertainty_scores import (
    risk_coverage_curve,
    threshold_metrics,
)


EXPERIMENT_ROOT = Path(
    "experiments/put_all_objects_into_basket/proposed_framework/vlm_confidence_500"
)
RECORDS_PATH = EXPERIMENT_ROOT / "association_records.json"
OUTPUT_ROOT = EXPERIMENT_ROOT / "likelihood_calibration"
SPLIT_COUNTS = {"calibration": 300, "validation": 100, "test": 100}
CALIBRATION_METHODS = ("isotonic", "platt")
RELIABILITY_BIN_COUNT = 10
DEVELOPMENT_GATE_THRESHOLD = 0.9999832372181827
SYSTEMATIC_FAILURE_TARGETS = ("alphabet soup", "tomato sauce")
OPERATING_THRESHOLDS = (
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99995,
    0.99998,
    DEVELOPMENT_GATE_THRESHOLD,
    0.99999,
    0.999995,
    0.999999,
)


def load_frozen_records(path=RECORDS_PATH):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("Expected exactly 500 frozen association records.")

    partitions = {
        name: [record for record in records if record.get("partition") == name]
        for name in SPLIT_COUNTS
    }
    counts = {name: len(items) for name, items in partitions.items()}
    if counts != SPLIT_COUNTS:
        raise ValueError(f"Expected frozen split counts {SPLIT_COUNTS}; received {counts}.")

    for record in records:
        score = record.get("raw_association_likelihood")
        raw_log_probability = record.get("raw_log_probability")
        if score is None or raw_log_probability is None:
            raise ValueError(
                f"Missing raw likelihood for {record.get('sample_id')!r}."
            )
        score = float(score)
        raw_log_probability = float(raw_log_probability)
        if not math.isfinite(score) or not 0.0 < score <= 1.0:
            raise ValueError(f"Invalid raw likelihood: {score}")
        if not math.isclose(
            math.exp(raw_log_probability), score, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"Raw log probability and likelihood disagree for "
                f"{record.get('sample_id')!r}."
            )
        if not isinstance(record.get("correct"), bool):
            raise ValueError("Every frozen correctness outcome must be boolean.")
    return records, partitions


def record_arrays(records):
    log_likelihoods = np.asarray(
        [float(record["raw_log_probability"]) for record in records], dtype=float
    )
    likelihoods = np.asarray(
        [float(record["raw_association_likelihood"]) for record in records],
        dtype=float,
    )
    correctness = np.asarray(
        [int(record["correct"]) for record in records], dtype=int
    )
    return log_likelihoods, likelihoods, correctness


def fit_calibrators(calibration_records):
    log_likelihoods, _, correctness = record_arrays(calibration_records)
    if np.unique(correctness).size != 2:
        raise ValueError("Calibration split must contain correct and incorrect cases.")

    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(log_likelihoods, correctness)

    feature_mean = float(log_likelihoods.mean())
    feature_scale = float(log_likelihoods.std())
    if feature_scale == 0.0:
        raise ValueError("Calibration log likelihoods have zero variance.")
    standardized = ((log_likelihoods - feature_mean) / feature_scale).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    platt.fit(standardized, correctness)
    return {
        "isotonic": isotonic,
        "platt": {
            "model": platt,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        },
    }


def calibrated_probabilities(calibrator_name, calibrator, records):
    log_likelihoods, _, _ = record_arrays(records)
    if calibrator_name == "isotonic":
        probabilities = calibrator.predict(log_likelihoods)
    elif calibrator_name == "platt":
        standardized = (
            (log_likelihoods - calibrator["feature_mean"])
            / calibrator["feature_scale"]
        ).reshape(-1, 1)
        probabilities = calibrator["model"].predict_proba(standardized)[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {calibrator_name}")
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def reliability_bins(probabilities, correctness, bin_count=RELIABILITY_BIN_COUNT, strategy="quantile"):
    probabilities = np.asarray(probabilities, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    if probabilities.size != correctness.size or probabilities.size == 0:
        raise ValueError("Reliability inputs must be nonempty and have equal length.")
    if strategy == "quantile":
        edges = np.unique(
            np.quantile(probabilities, np.linspace(0.0, 1.0, bin_count + 1))
        )
        if edges.size == 1:
            bin_indices = np.zeros(probabilities.size, dtype=int)
        else:
            bin_indices = np.searchsorted(edges[1:-1], probabilities, side="right")
    elif strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, bin_count + 1)
        bin_indices = np.minimum(
            np.searchsorted(edges[1:-1], probabilities, side="right"),
            bin_count - 1,
        )
    else:
        raise ValueError(f"Unknown reliability bin strategy: {strategy}")

    bins = []
    for bin_index in np.unique(bin_indices):
        mask = bin_indices == bin_index
        bins.append(
            {
                "bin": int(bin_index),
                "count": int(mask.sum()),
                "minimum_score": float(probabilities[mask].min()),
                "maximum_score": float(probabilities[mask].max()),
                "mean_score": float(probabilities[mask].mean()),
                "empirical_accuracy": float(correctness[mask].mean()),
            }
        )
    return bins


def expected_calibration_error(bins):
    total = sum(item["count"] for item in bins)
    return sum(
        item["count"]
        / total
        * abs(item["mean_score"] - item["empirical_accuracy"])
        for item in bins
    )


def probability_metrics(records, probabilities):
    _, _, correctness = record_arrays(records)
    probabilities = np.asarray(probabilities, dtype=float)
    quantile_bins = reliability_bins(probabilities, correctness, strategy="quantile")
    equal_width_bins = reliability_bins(
        probabilities, correctness, strategy="equal_width"
    )
    return {
        "trials": len(records),
        "accuracy": float(correctness.mean()),
        "correctness_ranking_auroc": float(
            roc_auc_score(correctness, probabilities)
        ),
        "brier_score": float(brier_score_loss(correctness, probabilities)),
        "ece_quantile_10": float(expected_calibration_error(quantile_bins)),
        "ece_equal_width_10": float(expected_calibration_error(equal_width_bins)),
        "reliability_quantile_10": quantile_bins,
        "reliability_equal_width_10": equal_width_bins,
    }


def calibration_spec(calibrator_name, calibrator):
    if calibrator_name == "isotonic":
        return {
            "method": "isotonic",
            "input": "raw_log_probability",
            "x_thresholds": [float(value) for value in calibrator.X_thresholds_],
            "y_thresholds": [float(value) for value in calibrator.y_thresholds_],
            "out_of_bounds": "clip",
        }
    model = calibrator["model"]
    return {
        "method": "platt",
        "input": "standardized_raw_log_probability",
        "feature_mean": calibrator["feature_mean"],
        "feature_scale": calibrator["feature_scale"],
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def systematic_failure_analysis(records, calibrated, threshold):
    analysis = {}
    for target in SYSTEMATIC_FAILURE_TARGETS:
        indices = [
            index
            for index, record in enumerate(records)
            if record["target_description"] == target
        ]
        target_records = [records[index] for index in indices]
        incorrect_indices = [
            index for index in indices if not bool(records[index]["correct"])
        ]
        raw_incorrect = [
            float(records[index]["raw_association_likelihood"])
            for index in incorrect_indices
        ]
        calibrated_incorrect = [float(calibrated[index]) for index in incorrect_indices]
        analysis[target] = {
            "trials": len(target_records),
            "accuracy": (
                sum(bool(record["correct"]) for record in target_records)
                / len(target_records)
                if target_records
                else None
            ),
            "incorrect_predictions": len(incorrect_indices),
            "incorrect_accepted_by_raw_development_gate": sum(
                score >= threshold for score in raw_incorrect
            ),
            "incorrect_raw_likelihood": _distribution(raw_incorrect),
            "incorrect_calibrated_probability": _distribution(calibrated_incorrect),
            "incorrect_cases": [
                {
                    "sample_id": records[index]["sample_id"],
                    "predicted_object_id": records[index]["predicted_object_id"],
                    "ground_truth_object_id": records[index][
                        "ground_truth_object_id"
                    ],
                    "raw_association_likelihood": float(
                        records[index]["raw_association_likelihood"]
                    ),
                    "calibrated_probability": float(calibrated[index]),
                    "accepted_by_raw_development_gate": (
                        float(records[index]["raw_association_likelihood"])
                        >= threshold
                    ),
                }
                for index in incorrect_indices
            ],
        }
    return analysis


def analyze(records, partitions):
    calibrators = fit_calibrators(partitions["calibration"])
    raw_metrics = {}
    risk_curves = []
    operating_points = []
    reliability_rows = []
    for split, split_records in partitions.items():
        _, raw_scores, _ = record_arrays(split_records)
        raw_metrics[split] = probability_metrics(split_records, raw_scores)
        curve = risk_coverage_curve(split_records, "raw_association_likelihood")
        raw_metrics[split]["risk_coverage"] = {
            key: value for key, value in curve.items() if key != "points"
        }
        for point in curve["points"]:
            risk_curves.append({"split": split, **point})
        for threshold in OPERATING_THRESHOLDS:
            operating_points.append(
                {
                    "split": split,
                    **threshold_metrics(
                        split_records, "raw_association_likelihood", threshold
                    ),
                }
            )
        reliability_rows.extend(
            _tag_reliability(
                raw_metrics[split]["reliability_quantile_10"],
                split,
                "raw",
            )
        )

    validation_candidates = {}
    calibration_probabilities = {}
    validation_probabilities = {}
    for method in CALIBRATION_METHODS:
        calibration_probabilities[method] = calibrated_probabilities(
            method, calibrators[method], partitions["calibration"]
        )
        validation_probabilities[method] = calibrated_probabilities(
            method, calibrators[method], partitions["validation"]
        )
        validation_candidates[method] = probability_metrics(
            partitions["validation"], validation_probabilities[method]
        )
        reliability_rows.extend(
            _tag_reliability(
                validation_candidates[method]["reliability_quantile_10"],
                "validation",
                method,
            )
        )

    selected_method = min(
        CALIBRATION_METHODS,
        key=lambda method: (
            validation_candidates[method]["brier_score"],
            validation_candidates[method]["ece_quantile_10"],
        ),
    )
    selected_calibrator = calibrators[selected_method]
    test_probabilities = calibrated_probabilities(
        selected_method, selected_calibrator, partitions["test"]
    )
    calibrated_test_metrics = probability_metrics(
        partitions["test"], test_probabilities
    )
    reliability_rows.extend(
        _tag_reliability(
            calibrated_test_metrics["reliability_quantile_10"],
            "test",
            selected_method,
        )
    )
    raw_test_metrics = raw_metrics["test"]
    systematic = systematic_failure_analysis(
        partitions["test"], test_probabilities, DEVELOPMENT_GATE_THRESHOLD
    )
    return {
        "schema_version": 1,
        "input_records": str(RECORDS_PATH),
        "input_sha256": hashlib.sha256(RECORDS_PATH.read_bytes()).hexdigest(),
        "frozen_trials": len(records),
        "split_counts": SPLIT_COUNTS,
        "raw_score_definition": (
            "exp(sum(provider log probabilities of generated decision-bearing "
            "label tokens)); not a calibrated correctness probability"
        ),
        "calibration_protocol": {
            "fit_split": "calibration only",
            "selection_split": "validation only",
            "selection_rule": (
                "minimum validation Brier score; quantile-bin ECE breaks exact ties"
            ),
            "test_usage": "one evaluation after freezing method and parameters",
            "class_used_as_input": False,
        },
        "raw_metrics": raw_metrics,
        "raw_operating_points": operating_points,
        "validation_calibration_candidates": validation_candidates,
        "selected_calibration_method": selected_method,
        "selected_calibrator": calibration_spec(
            selected_method, selected_calibrator
        ),
        "held_out_test": {
            "raw": _compact_probability_metrics(raw_test_metrics),
            "calibrated": _compact_probability_metrics(calibrated_test_metrics),
            "ranking_auroc_change": (
                calibrated_test_metrics["correctness_ranking_auroc"]
                - raw_test_metrics["correctness_ranking_auroc"]
            ),
        },
        "development_raw_gate_threshold": DEVELOPMENT_GATE_THRESHOLD,
        "systematic_high_confidence_failures_test": systematic,
    }, {
        "calibrators": calibrators,
        "calibration_probabilities": calibration_probabilities,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "risk_curves": risk_curves,
        "operating_points": operating_points,
        "reliability_rows": reliability_rows,
    }


def save_analysis(report, artifacts, partitions, output_root=OUTPUT_ROOT):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_path = output_root / "analysis.json"
    analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _write_csv(output_root / "risk_coverage.csv", artifacts["risk_curves"])
    _write_csv(
        output_root / "autonomous_accuracy_by_threshold.csv",
        artifacts["operating_points"],
    )
    _write_csv(output_root / "reliability_bins.csv", artifacts["reliability_rows"])
    _save_discrimination_figure(
        partitions,
        artifacts["risk_curves"],
        output_root / "raw_likelihood_discrimination.png",
    )
    _save_validation_reliability_figure(
        partitions["validation"],
        artifacts["validation_probabilities"],
        output_root / "validation_calibration.png",
    )
    _save_test_reliability_figure(
        partitions["test"],
        artifacts["test_probabilities"],
        report["selected_calibration_method"],
        output_root / "test_reliability.png",
    )
    report_path = output_root / "REPORT.md"
    report_path.write_text(_markdown_report(report), encoding="utf-8")
    return analysis_path, report_path


def _compact_probability_metrics(metrics):
    return {
        key: metrics[key]
        for key in (
            "trials",
            "accuracy",
            "correctness_ranking_auroc",
            "brier_score",
            "ece_quantile_10",
            "ece_equal_width_10",
            "reliability_quantile_10",
        )
    }


def _tag_reliability(bins, split, score_source):
    return [
        {"split": split, "score_source": score_source, **item} for item in bins
    ]


def _distribution(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty analysis table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_discrimination_figure(partitions, risk_curves, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    colors = {"calibration": "#0072B2", "validation": "#D55E00", "test": "#009E73"}
    for split in SPLIT_COUNTS:
        points = sorted(
            (item for item in risk_curves if item["split"] == split),
            key=lambda item: item["autonomous_coverage"],
        )
        coverage = [0.0, *(item["autonomous_coverage"] for item in points)]
        risk = [0.0, *(item["selective_risk"] for item in points)]
        accuracy = [1.0, *(item["autonomous_accuracy"] for item in points)]
        axes[0].plot(coverage, risk, label=split, color=colors[split], linewidth=2)
        axes[1].plot(
            coverage, accuracy, label=split, color=colors[split], linewidth=2
        )

    for split, records in partitions.items():
        _, scores, correctness = record_arrays(records)
        transformed = -np.log10(np.maximum(1.0 - scores, 1e-15))
        axes[2].hist(
            transformed[correctness == 1],
            bins=20,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=colors[split],
            label=f"{split}: correct",
        )
        axes[2].hist(
            transformed[correctness == 0],
            bins=20,
            density=True,
            histtype="step",
            linestyle="--",
            linewidth=1.4,
            color=colors[split],
            label=f"{split}: incorrect",
        )
    axes[0].set(
        title="Risk–coverage",
        xlabel="Autonomous coverage",
        ylabel="Selective risk",
    )
    axes[1].set(
        title="Accuracy–coverage",
        xlabel="Autonomous coverage",
        ylabel="Autonomous accuracy",
        ylim=(0.0, 1.02),
    )
    axes[2].set(
        title="Near-one score resolution",
        xlabel=r"$-\log_{10}(1-c_{raw})$",
        ylabel="Density",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False, fontsize=7)
    figure.suptitle("Frozen LIBERO raw generated-label likelihood")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_validation_reliability_figure(records, calibrated, output_path):
    _, raw_scores, correctness = record_arrays(records)
    series = {"raw": raw_scores, **calibrated}
    _save_reliability_figure(
        correctness,
        series,
        "Validation-only calibrator selection",
        output_path,
    )


def _save_test_reliability_figure(records, calibrated, method, output_path):
    _, raw_scores, correctness = record_arrays(records)
    _save_reliability_figure(
        correctness,
        {"raw": raw_scores, method: calibrated},
        f"Held-out test reliability: frozen {method} mapping",
        output_path,
    )


def _save_reliability_figure(correctness, series, title, output_path):
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.plot([0, 1], [0, 1], color="0.45", linestyle="--", label="ideal")
    colors = {"raw": "#0072B2", "isotonic": "#D55E00", "platt": "#009E73"}
    for name, probabilities in series.items():
        bins = reliability_bins(probabilities, correctness, strategy="quantile")
        axis.plot(
            [item["mean_score"] for item in bins],
            [item["empirical_accuracy"] for item in bins],
            marker="o",
            linewidth=1.8,
            label=name,
            color=colors[name],
        )
    axis.set(
        title=title,
        xlabel="Mean predicted correctness probability (quantile bins)",
        ylabel="Empirical correctness",
        xlim=(0.0, 1.02),
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(report):
    raw = report["held_out_test"]["raw"]
    calibrated = report["held_out_test"]["calibrated"]
    method = report["selected_calibration_method"]
    validation = report["validation_calibration_candidates"]
    failures = report["systematic_high_confidence_failures_test"]
    lines = [
        "# Raw VLM likelihood calibration on frozen LIBERO trials",
        "",
        "The analysis uses the existing 500 predictions only; no VLM inference was rerun.",
        "The 300/100/100 calibration, validation, and test partitions were preserved.",
        "Raw likelihood is used as a ranking signal here, while probability "
        "calibration is evaluated separately.",
        "",
        "## Raw-score discrimination",
        "",
        "| Split | Association accuracy | Correctness AUROC | Risk–coverage area |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in SPLIT_COUNTS:
        metrics = report["raw_metrics"][split]
        lines.append(
            f"| {split} | {metrics['accuracy']:.6f} | "
            f"{metrics['correctness_ranking_auroc']:.6f} | "
            f"{metrics['risk_coverage']['aurc_over_operational_coverage']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Held-out autonomous accuracy by raw threshold",
            "",
            "| Threshold | Coverage | Autonomous accuracy | False accepts |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    displayed_thresholds = {
        0.90,
        0.99,
        0.999,
        0.9999,
        0.99995,
        DEVELOPMENT_GATE_THRESHOLD,
        0.999999,
    }
    for point in report["raw_operating_points"]:
        if point["split"] == "test" and point["threshold"] in displayed_thresholds:
            lines.append(
                f"| {point['threshold']:.15g} | "
                f"{point['autonomous_coverage']:.3f} | "
                f"{point['autonomous_accuracy']:.3f} | "
                f"{point['false_autonomous_acceptance_count']} |"
            )
    lines.extend(
        [
            "",
            "## Validation selection",
            "",
            "| Mapping | Validation Brier | Validation quantile ECE | Validation AUROC |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for candidate in CALIBRATION_METHODS:
        metrics = validation[candidate]
        lines.append(
            f"| {candidate} | {metrics['brier_score']:.6f} | "
            f"{metrics['ece_quantile_10']:.6f} | "
            f"{metrics['correctness_ranking_auroc']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected on validation: **{method}** (lowest Brier score).",
            "",
            "## Held-out test",
            "",
            "| Score | ECE (10 quantile bins) | ECE (10 equal-width bins) | Brier | AUROC |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Raw likelihood | {raw['ece_quantile_10']:.6f} | "
            f"{raw['ece_equal_width_10']:.6f} | {raw['brier_score']:.6f} | "
            f"{raw['correctness_ranking_auroc']:.6f} |",
            f"| {method} calibrated | {calibrated['ece_quantile_10']:.6f} | "
            f"{calibrated['ece_equal_width_10']:.6f} | "
            f"{calibrated['brier_score']:.6f} | "
            f"{calibrated['correctness_ranking_auroc']:.6f} |",
            "",
            f"The frozen monotonic mapping changed test AUROC by "
            f"{report['held_out_test']['ranking_auroc_change']:+.6f}; isotonic "
            "plateaus create ties, so ranking is not expected to improve.",
            "",
            "### Held-out quantile reliability bins",
            "",
            "| Score | Count | Mean score | Empirical correctness |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in (("raw", raw), (method, calibrated)):
        for item in metrics["reliability_quantile_10"]:
            lines.append(
                f"| {name} | {item['count']} | {item['mean_score']:.6f} | "
                f"{item['empirical_accuracy']:.6f} |"
            )
    lines.extend(
        [
            "",
            "Ten quantile bins were requested. Exact score ties and isotonic "
            "plateaus reduce these to fewer distinct non-empty bins.",
            "",
            "Raw likelihood is not itself calibrated: values concentrated near one "
            "substantially overstate empirical correctness. The selected mapping "
            "supports an approximate class-agnostic correctness-probability "
            "interpretation for this frozen model/prompt/simulation setup. The test "
            "sample is only 100 trials and the isotonic output is coarse, so this "
            "interpretation must not be generalized to new models, prompts, "
            "candidate counts, or domains without fresh validation.",
            "",
            "## Systematic held-out failures",
            "",
            "| Target | Trials | Accuracy | Incorrect | Incorrect accepted by raw gate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in SYSTEMATIC_FAILURE_TARGETS:
        item = failures[target]
        lines.append(
            f"| {target} | {item['trials']} | {item['accuracy']:.3f} | "
            f"{item['incorrect_predictions']} | "
            f"{item['incorrect_accepted_by_raw_development_gate']} |"
        )
    lines.extend(
        [
            "",
            "These failures remain important because a single class-agnostic scalar "
            "mapping cannot identify a class-specific semantic confusion when its raw "
            "likelihood resembles that of correct predictions.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    records, partitions = load_frozen_records()
    report, artifacts = analyze(records, partitions)
    analysis_path, report_path = save_analysis(
        report, artifacts, partitions
    )
    test = report["held_out_test"]
    print(f"Frozen associations analyzed: {len(records)}")
    print(f"Selected calibrator: {report['selected_calibration_method']}")
    print(f"Test raw AUROC: {test['raw']['correctness_ranking_auroc']:.6f}")
    print(f"Test raw ECE: {test['raw']['ece_quantile_10']:.6f}")
    print(f"Test calibrated ECE: {test['calibrated']['ece_quantile_10']:.6f}")
    print(f"Test raw Brier: {test['raw']['brier_score']:.6f}")
    print(f"Test calibrated Brier: {test['calibrated']['brier_score']:.6f}")
    print(f"Saved analysis: {analysis_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
