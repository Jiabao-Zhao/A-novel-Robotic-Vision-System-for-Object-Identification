"""Compare saved VLM uncertainty scores without rerunning model inference."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = Path(
    "experiments/put_all_objects_into_basket/proposed_framework/vlm_confidence_500"
)
RECORDS_PATH = EXPERIMENT_ROOT / "association_records.json"
OUTPUT_ROOT = EXPERIMENT_ROOT / "uncertainty_score_comparison"
SCORE_METHODS = {
    "raw_label_likelihood": "raw_association_likelihood",
    "candidate_normalized": "association_score",
    "candidate_margin": "association_margin",
}
TARGET_ACCURACIES = (0.90, 0.95)
SYSTEMATIC_FAILURE_TARGETS = ("alphabet soup", "tomato sauce")


def threshold_metrics(records, score_field, threshold):
    """Evaluate a saved-score gate; unavailable scores are always deferred."""
    scored = []
    for record in records:
        score = record.get(score_field)
        if score is None:
            continue
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid {score_field}: {score}")
        scored.append((score, bool(record["correct"])))

    accepted = [correct for score, correct in scored if score >= threshold]
    false_accepts = accepted.count(False)
    total = len(records)
    return {
        "threshold": float(threshold),
        "total_trials": total,
        "score_available_trials": len(scored),
        "score_unavailable_trials": total - len(scored),
        "autonomous_decisions": len(accepted),
        "deferred_decisions": total - len(accepted),
        "autonomous_coverage": _ratio(len(accepted), total),
        "autonomous_accuracy": _ratio(accepted.count(True), len(accepted)),
        "selective_risk": _ratio(false_accepts, len(accepted)),
        "false_autonomous_acceptance_count": false_accepts,
        "false_autonomous_acceptance_rate_all_trials": _ratio(
            false_accepts, total
        ),
        "false_autonomous_acceptance_rate_autonomous": _ratio(
            false_accepts, len(accepted)
        ),
    }


def select_development_operating_point(
    calibration_records,
    validation_records,
    score_field,
    minimum_accuracy,
):
    """Maximize validation coverage while both development splits meet risk."""
    thresholds = sorted(
        {
            float(record[score_field])
            for record in [*calibration_records, *validation_records]
            if record.get(score_field) is not None
        }
    )
    eligible = []
    for threshold in thresholds:
        calibration = threshold_metrics(
            calibration_records, score_field, threshold
        )
        validation = threshold_metrics(validation_records, score_field, threshold)
        if (
            calibration["autonomous_decisions"]
            and validation["autonomous_decisions"]
            and calibration["autonomous_accuracy"] >= minimum_accuracy
            and validation["autonomous_accuracy"] >= minimum_accuracy
        ):
            eligible.append((threshold, calibration, validation))
    if not eligible:
        return None

    threshold, calibration, validation = max(
        eligible,
        key=lambda item: (
            item[2]["autonomous_coverage"],
            item[1]["autonomous_coverage"],
            -item[0],
        ),
    )
    return {
        "minimum_autonomous_accuracy": minimum_accuracy,
        "threshold": threshold,
        "calibration": calibration,
        "validation": validation,
    }


def risk_coverage_curve(records, score_field):
    """Return tied-threshold selective-risk points and normalized AURC."""
    values = sorted(
        {
            float(record[score_field])
            for record in records
            if record.get(score_field) is not None
        },
        reverse=True,
    )
    points = [threshold_metrics(records, score_field, value) for value in values]
    previous_coverage = 0.0
    previous_risk = 0.0
    area = 0.0
    for point in points:
        coverage = point["autonomous_coverage"]
        risk = point["selective_risk"]
        area += (coverage - previous_coverage) * (risk + previous_risk) / 2.0
        previous_coverage = coverage
        previous_risk = risk
    attainable_coverage = _ratio(
        sum(record.get(score_field) is not None for record in records),
        len(records),
    )
    return {
        "points": points,
        "attainable_coverage": attainable_coverage,
        "aurc_over_operational_coverage": area,
        "aurc_normalized_to_attainable_coverage": (
            area / attainable_coverage if attainable_coverage else None
        ),
    }


def score_distribution(records, score_field):
    available = [
        (float(record[score_field]), bool(record["correct"]))
        for record in records
        if record.get(score_field) is not None
    ]
    return {
        "correct": _distribution([score for score, correct in available if correct]),
        "incorrect": _distribution(
            [score for score, correct in available if not correct]
        ),
        "correctness_ranking_auc": _ranking_auc(available),
    }


def analyze(records):
    partitions = {
        name: [record for record in records if record.get("partition") == name]
        for name in ("calibration", "validation", "test")
    }
    if {name: len(items) for name, items in partitions.items()} != {
        "calibration": 300,
        "validation": 100,
        "test": 100,
    }:
        raise ValueError("Expected the frozen 300/100/100 experiment split.")

    development = [*partitions["calibration"], *partitions["validation"]]
    methods = {}
    curves = []
    for method, score_field in SCORE_METHODS.items():
        availability = {
            split: _ratio(
                sum(record.get(score_field) is not None for record in split_records),
                len(split_records),
            )
            for split, split_records in partitions.items()
        }
        availability["all"] = _ratio(
            sum(record.get(score_field) is not None for record in records),
            len(records),
        )
        operating_points = {
            str(target): select_development_operating_point(
                partitions["calibration"],
                partitions["validation"],
                score_field,
                target,
            )
            for target in TARGET_ACCURACIES
        }
        curve_summaries = {}
        for split in ("calibration", "validation"):
            curve = risk_coverage_curve(partitions[split], score_field)
            curve_summaries[split] = {
                key: value for key, value in curve.items() if key != "points"
            }
            for point in curve["points"]:
                curves.append({"method": method, "split": split, **point})
        methods[method] = {
            "score_field": score_field,
            "availability": availability,
            "development_distribution": score_distribution(
                development, score_field
            ),
            "risk_coverage": curve_summaries,
            "development_operating_points": operating_points,
            "systematic_failure_analysis": _systematic_failure_analysis(
                development,
                score_field,
                operating_points,
            ),
        }

    selected_method, selected_point = _select_final_method(methods)
    selected_field = SCORE_METHODS[selected_method]
    frozen_threshold = selected_point["threshold"]
    test_metrics = threshold_metrics(
        partitions["test"], selected_field, frozen_threshold
    )
    test_metrics["systematic_failure_targets"] = {
        target: threshold_metrics(
            [
                record
                for record in partitions["test"]
                if record["target_description"] == target
            ],
            selected_field,
            frozen_threshold,
        )
        for target in SYSTEMATIC_FAILURE_TARGETS
    }
    return {
        "schema_version": 1,
        "input_records": str(RECORDS_PATH),
        "prediction_rule": (
            "Every method uses the same saved predicted_object_id and correctness "
            "label; only the deferral ranking score changes."
        ),
        "selection_protocol": (
            "For each requested accuracy, select the threshold with highest "
            "validation coverage among thresholds meeting the accuracy on both "
            "calibration and validation. Select the final method by highest "
            "validation coverage at >=95% accuracy. Test is evaluated only for "
            "that frozen method and threshold."
        ),
        "methods": methods,
        "frozen_selection": {
            "method": selected_method,
            "score_field": selected_field,
            "target_accuracy": 0.95,
            "threshold": frozen_threshold,
            "calibration": selected_point["calibration"],
            "validation": selected_point["validation"],
        },
        "held_out_test": test_metrics,
    }, curves


def _select_final_method(methods):
    candidates = []
    for method, result in methods.items():
        point = result["development_operating_points"].get("0.95")
        if point is not None:
            candidates.append((method, point))
    if not candidates:
        raise RuntimeError("No score method supports a >=95% development point.")
    return max(
        candidates,
        key=lambda item: (
            item[1]["validation"]["autonomous_coverage"],
            item[1]["calibration"]["autonomous_coverage"],
        ),
    )


def _systematic_failure_analysis(records, score_field, operating_points):
    analysis = {}
    for target in SYSTEMATIC_FAILURE_TARGETS:
        target_records = [
            record for record in records if record["target_description"] == target
        ]
        item = {
            "trials": len(target_records),
            "incorrect_predictions": sum(
                not bool(record["correct"]) for record in target_records
            ),
            "score_distribution": score_distribution(target_records, score_field),
            "operating_points": {},
        }
        for accuracy, point in operating_points.items():
            if point is None:
                item["operating_points"][accuracy] = None
            else:
                item["operating_points"][accuracy] = threshold_metrics(
                    target_records, score_field, point["threshold"]
                )
        analysis[target] = item
    return analysis


def _distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q10": _quantile(ordered, 0.10),
        "q25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "q75": _quantile(ordered, 0.75),
        "q90": _quantile(ordered, 0.90),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ranking_auc(scored_correctness):
    correct = [score for score, is_correct in scored_correctness if is_correct]
    incorrect = [score for score, is_correct in scored_correctness if not is_correct]
    if not correct or not incorrect:
        return None
    favorable = sum(
        sum(value > other for other in incorrect)
        + 0.5 * sum(value == other for other in incorrect)
        for value in correct
    )
    return favorable / (len(correct) * len(incorrect))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def save_analysis(report, curves, output_root=OUTPUT_ROOT):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    curve_path = output_root / "risk_coverage_curves.csv"
    fields = (
        "method",
        "split",
        "threshold",
        "total_trials",
        "score_available_trials",
        "score_unavailable_trials",
        "autonomous_decisions",
        "deferred_decisions",
        "autonomous_coverage",
        "autonomous_accuracy",
        "selective_risk",
        "false_autonomous_acceptance_count",
        "false_autonomous_acceptance_rate_all_trials",
        "false_autonomous_acceptance_rate_autonomous",
    )
    with curve_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curves)
    figure_path = output_root / "risk_coverage.png"
    _save_risk_coverage_figure(curves, figure_path)
    return report_path, curve_path, figure_path


def _save_risk_coverage_figure(curves, output_path):
    labels = {
        "raw_label_likelihood": "Raw label likelihood",
        "candidate_normalized": "Candidate-normalized",
        "candidate_margin": "Candidate margin",
    }
    colors = {
        "raw_label_likelihood": "#0072B2",
        "candidate_normalized": "#D55E00",
        "candidate_margin": "#009E73",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for axis, split in zip(axes, ("calibration", "validation")):
        for method in SCORE_METHODS:
            points = sorted(
                (
                    item
                    for item in curves
                    if item["method"] == method and item["split"] == split
                ),
                key=lambda item: item["autonomous_coverage"],
            )
            x = [0.0, *(item["autonomous_coverage"] for item in points)]
            y = [0.0, *(item["selective_risk"] for item in points)]
            axis.plot(
                x,
                y,
                label=labels[method],
                color=colors[method],
                linewidth=2.0,
            )
        axis.axhline(0.10, color="0.45", linestyle="--", linewidth=1.0)
        axis.axhline(0.05, color="0.45", linestyle=":", linewidth=1.0)
        axis.set_title(split.capitalize())
        axis.set_xlabel("Operational autonomous coverage")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Selective risk (1 - autonomous accuracy)")
    maximum_risk = max(
        item["selective_risk"]
        for item in curves
        if item["selective_risk"] is not None
    )
    axes[0].set_ylim(0.0, min(1.0, max(0.25, maximum_risk * 1.05)))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "LIBERO VLM uncertainty: risk versus operational coverage", y=0.98
    )
    figure.tight_layout(rect=(0.0, 0.11, 1.0, 0.92))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    payload = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("Expected exactly 500 saved association records.")
    report, curves = analyze(records)
    report_path, curve_path, figure_path = save_analysis(report, curves)
    selection = report["frozen_selection"]
    test = report["held_out_test"]
    print(f"Frozen method: {selection['method']}")
    print(f"Frozen threshold: {selection['threshold']}")
    print(f"Test coverage: {test['autonomous_coverage']}")
    print(f"Test autonomous accuracy: {test['autonomous_accuracy']}")
    print(f"Test false accepts: {test['false_autonomous_acceptance_count']}")
    print(f"Saved analysis: {report_path}")
    print(f"Saved risk-coverage curves: {curve_path}")
    print(f"Saved risk-coverage figure: {figure_path}")


if __name__ == "__main__":
    main()
