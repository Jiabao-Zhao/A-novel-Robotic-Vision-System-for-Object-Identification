"""Evaluate raw VLM association likelihoods on reproducible saved scenes.

Input is a JSON manifest containing a ``split`` name and a ``samples`` list.
Threshold selection is deliberately not performed here: use calibration or
validation output to choose a threshold, then evaluate that fixed value on a
separate test split.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from vlm_module import (
    GeminiThenOpenAI,
    GeminiVLM,
    OpenAIVLM,
    infer_target_association,
    load_localized_objects,
)


DEFAULT_THRESHOLDS = tuple(value / 100 for value in range(50, 100, 5))
REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "image_path",
    "localization_path",
    "target_description",
    "ground_truth_object_id",
)
CSV_FIELDS = (
    "sample_id",
    "target_description",
    "provider",
    "model",
    "predicted_object_id",
    "ground_truth_object_id",
    "correct",
    "raw_log_probability",
    "association_score",
    "candidate_scores",
    "association_margin",
    "score_unavailable_reason",
)


def load_evaluation_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("Evaluation manifest must contain a 'samples' list.")

    samples = []
    seen_ids = set()
    for raw_sample in payload["samples"]:
        if not isinstance(raw_sample, dict):
            raise ValueError("Every evaluation sample must be a JSON object.")
        missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in raw_sample]
        if missing:
            raise ValueError(f"Evaluation sample is missing fields: {missing}")
        sample_id = str(raw_sample["sample_id"]).strip()
        target_description = str(raw_sample["target_description"]).strip()
        if not sample_id or not target_description:
            raise ValueError("sample_id and target_description must be nonempty.")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate evaluation sample_id: {sample_id}")
        seen_ids.add(sample_id)

        ground_truth = raw_sample["ground_truth_object_id"]
        if ground_truth is not None:
            ground_truth = str(ground_truth).strip()
            if ground_truth.lower() == "none":
                ground_truth = None
            elif not ground_truth:
                raise ValueError(
                    f"ground_truth_object_id is empty for sample {sample_id!r}."
                )
        samples.append(
            {
                "sample_id": sample_id,
                "image_path": _manifest_relative_path(
                    manifest_path.parent, raw_sample["image_path"]
                ),
                "localization_path": _manifest_relative_path(
                    manifest_path.parent, raw_sample["localization_path"]
                ),
                "target_description": target_description,
                "ground_truth_object_id": ground_truth,
            }
        )

    if not samples:
        raise ValueError("Evaluation manifest contains no samples.")
    return str(payload.get("split") or "unspecified"), samples


def evaluate_samples(samples, provider):
    """Run one raw VLM association per sample without human correction."""
    records = []
    for sample in samples:
        detections = load_localized_objects(sample["localization_path"])
        candidate_ids = {str(item["object_id"]) for item in detections}
        ground_truth = sample["ground_truth_object_id"]
        if ground_truth is not None and ground_truth not in candidate_ids:
            raise ValueError(
                f"Ground-truth ID {ground_truth!r} for sample "
                f"{sample['sample_id']!r} is not in its localization candidates."
            )

        result = infer_target_association(
            target_description=sample["target_description"],
            image_path=sample["image_path"],
            detections=detections,
            provider=provider,
        )
        diagnostics = result["diagnostics"]
        prediction_available = diagnostics.get("choice_label") is not None
        score = result["association_score"]
        if score is not None and not math.isclose(
            score,
            diagnostics["raw_association_likelihood"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "association_score must equal raw_association_likelihood."
            )

        unavailable_reason = None
        if score is None:
            unavailable_reason = diagnostics.get("logprob_error")
            if unavailable_reason is None and not prediction_available:
                unavailable_reason = "The provider output was not a valid choice label."
            if unavailable_reason is None:
                unavailable_reason = "Decision-bearing token log probabilities unavailable."

        records.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": str(sample["image_path"]),
                "localization_path": str(sample["localization_path"]),
                "target_description": sample["target_description"],
                "provider": diagnostics.get("provider"),
                "model": diagnostics.get("model"),
                "predicted_object_id": result["vlm_object_id"],
                "ground_truth_object_id": ground_truth,
                "correct": bool(
                    prediction_available and result["vlm_object_id"] == ground_truth
                ),
                "raw_log_probability": diagnostics["raw_log_probability"],
                "association_score": score,
                "candidate_scores": diagnostics.get("candidate_scores"),
                "association_margin": diagnostics.get("association_margin"),
                "score_unavailable_reason": unavailable_reason,
            }
        )
    return records


def threshold_sweep(records, thresholds=DEFAULT_THRESHOLDS):
    """Analyze saved numeric scores; this function never performs VLM inference."""
    scored_records = []
    for record in records:
        score = record.get("association_score")
        if score is None:
            continue
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid association_score: {score}")
        scored_records.append({**record, "association_score": score})

    total_trials = len(records)
    scored_trials = len(scored_records)
    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid threshold: {threshold}")
        autonomous = [
            record
            for record in scored_records
            if record["association_score"] >= threshold
        ]
        deferred_count = scored_trials - len(autonomous)
        correct_autonomous = sum(bool(record["correct"]) for record in autonomous)
        false_acceptances = len(autonomous) - correct_autonomous
        rows.append(
            {
                "threshold": threshold,
                "total_trials": total_trials,
                "scored_trials": scored_trials,
                "score_unavailable_trials": total_trials - scored_trials,
                "autonomous_decisions": len(autonomous),
                "deferred_decisions": deferred_count,
                "autonomous_coverage": _safe_ratio(len(autonomous), scored_trials),
                "deferral_rate": _safe_ratio(deferred_count, scored_trials),
                "autonomous_accuracy": _safe_ratio(
                    correct_autonomous, len(autonomous)
                ),
                "false_autonomous_acceptance_count": false_acceptances,
                "false_autonomous_acceptance_rate_all_scored_trials": _safe_ratio(
                    false_acceptances, scored_trials
                ),
                "false_autonomous_acceptance_rate_autonomous": _safe_ratio(
                    false_acceptances, len(autonomous)
                ),
            }
        )
    return rows


def save_evaluation(records, dataset_split, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "association_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_split": dataset_split,
                "score_definition": (
                    "association_score = exp(sum(log probabilities of "
                    "decision-bearing output tokens))"
                ),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "association_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            csv_record = {field: record.get(field) for field in CSV_FIELDS}
            candidate_scores = csv_record["candidate_scores"]
            csv_record["candidate_scores"] = (
                json.dumps(candidate_scores, separators=(",", ":"))
                if candidate_scores is not None
                else ""
            )
            writer.writerow(csv_record)

    saved_records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
    sweep = threshold_sweep(saved_records)
    sweep_path = output_dir / "threshold_sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_split": dataset_split,
                "analysis_population": "samples_with_numeric_association_score",
                "threshold_selection_performed": False,
                "metrics": sweep,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return records_path, csv_path, sweep_path


def provider_from_name(name):
    if name == "openai":
        return OpenAIVLM()
    if name == "gemini":
        return GeminiVLM()
    if name == "auto":
        return GeminiThenOpenAI()
    raise ValueError(f"Unknown provider: {name}")


def _manifest_relative_path(manifest_directory, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_directory / path).resolve()


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate raw VLM association likelihood on saved RGB-D scenes."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--provider", choices=("openai", "gemini", "auto"), default="openai")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physical/vlm_confidence_evaluation"),
    )
    arguments = parser.parse_args()

    dataset_split, samples = load_evaluation_manifest(arguments.manifest)
    records = evaluate_samples(samples, provider_from_name(arguments.provider))
    records_path, csv_path, sweep_path = save_evaluation(
        records, dataset_split, arguments.output_dir
    )
    scored_count = sum(record["association_score"] is not None for record in records)
    print(f"Evaluated associations: {len(records)}")
    print(f"Associations with numeric raw likelihood: {scored_count}")
    print(f"Saved association JSON: {records_path}")
    print(f"Saved association CSV: {csv_path}")
    print(f"Saved threshold sweep: {sweep_path}")


if __name__ == "__main__":
    main()
