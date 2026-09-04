"""Run and checkpoint the 500-scene LIBERO VLM confidence experiment."""

import hashlib
import json
import os
import time
from pathlib import Path

from scripts.evaluate_vlm_confidence import (
    evaluate_samples,
    load_evaluation_manifest,
    save_evaluation,
)
from scripts.capture_libero_confidence_dataset import OUTPUT_ROOT
from vlm_module import OpenAIVLM


MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
# Keep new raw-likelihood runs separate from the historical
# candidate-normalized checkpoint/results directory.
RESULTS_ROOT = OUTPUT_ROOT / "raw_label_likelihood_results"
CHECKPOINT_PATH = RESULTS_ROOT / "association_checkpoint.json"
MAX_PROVIDER_ATTEMPTS = 6


def run_experiment(
    manifest_path=MANIFEST_PATH,
    results_root=RESULTS_ROOT,
    provider=None,
):
    manifest_path = Path(manifest_path)
    results_root = Path(results_root)
    dataset_split, samples = load_evaluation_manifest(manifest_path)
    if len(samples) != 500:
        raise RuntimeError(
            f"The full LIBERO-Object experiment requires 500 samples; found {len(samples)}."
        )

    active_provider = OpenAIVLM() if provider is None else provider
    signature = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "requested_model": getattr(
            active_provider,
            "model_name",
            os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini"),
        ),
        "score_type": "raw_label_likelihood",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    records = _load_checkpoint(results_root / CHECKPOINT_PATH.name, signature)
    completed_ids = {record["sample_id"] for record in records}

    for sample in samples:
        if sample["sample_id"] in completed_ids:
            continue
        record = _evaluate_with_retry(sample, active_provider)
        records.append(record)
        completed_ids.add(record["sample_id"])
        _save_checkpoint(results_root / CHECKPOINT_PATH.name, signature, records)
        print(
            f"Scored {len(records):03d}/500: {record['sample_id']} "
            f"prediction={record['predicted_object_id']} "
            f"ground_truth={record['ground_truth_object_id']} "
            f"score={record['association_score']} correct={record['correct']}"
        )

    records.sort(key=lambda item: (item["task_index"], item["initial_state_index"]))
    saved = {"all": save_evaluation(records, dataset_split, results_root / "all")}
    for partition in ("calibration", "validation", "test"):
        partition_records = [
            record for record in records if record.get("partition") == partition
        ]
        expected_count = 300 if partition == "calibration" else 100
        if len(partition_records) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {partition} records; found "
                f"{len(partition_records)}."
            )
        saved[partition] = save_evaluation(
            partition_records,
            f"{dataset_split}_{partition}",
            results_root / partition,
        )

    summary = _experiment_summary(records, signature)
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Experiment summary: {summary_path}")
    return summary, saved


def run_task_worker(
    task_indices,
    worker_root,
    manifest_path=MANIFEST_PATH,
    provider=None,
):
    """Score a disjoint task subset and save a resumable worker checkpoint."""
    task_indices = tuple(sorted({int(value) for value in task_indices}))
    if not task_indices:
        raise ValueError("A confidence worker requires at least one task index.")
    manifest_path = Path(manifest_path)
    _, all_samples = load_evaluation_manifest(manifest_path)
    samples = [item for item in all_samples if item.get("task_index") in task_indices]
    expected_count = 50 * len(task_indices)
    if len(samples) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} samples for tasks {task_indices}; found "
            f"{len(samples)}."
        )

    active_provider = OpenAIVLM() if provider is None else provider
    signature = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "requested_model": getattr(
            active_provider,
            "model_name",
            os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1-mini"),
        ),
        "score_type": "raw_label_likelihood",
        "task_indices": list(task_indices),
    }
    worker_root = Path(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = worker_root / "association_checkpoint.json"
    records = _load_checkpoint(checkpoint_path, signature)
    completed_ids = {record["sample_id"] for record in records}
    for sample in samples:
        if sample["sample_id"] in completed_ids:
            continue
        record = _evaluate_with_retry(sample, active_provider)
        records.append(record)
        completed_ids.add(record["sample_id"])
        _save_checkpoint(checkpoint_path, signature, records)
        print(
            f"Worker {task_indices} {len(records):03d}/{expected_count}: "
            f"{record['sample_id']} prediction={record['predicted_object_id']} "
            f"score={record['association_score']} correct={record['correct']}"
        )
    return checkpoint_path


def finalize_worker_checkpoints(
    checkpoint_paths,
    manifest_path=MANIFEST_PATH,
    results_root=RESULTS_ROOT,
):
    """Combine disjoint worker outputs and calculate saved threshold tables."""
    manifest_path = Path(manifest_path)
    dataset_split, samples = load_evaluation_manifest(manifest_path)
    sample_ids = {sample["sample_id"] for sample in samples}
    records_by_id = {}
    for checkpoint_path in checkpoint_paths:
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            sample_id = record.get("sample_id")
            if sample_id not in sample_ids:
                raise RuntimeError(f"Unknown checkpoint sample ID: {sample_id}")
            if sample_id in records_by_id:
                raise RuntimeError(
                    f"Duplicate sample ID across confidence checkpoints: {sample_id}"
                )
            records_by_id[sample_id] = record
    if set(records_by_id) != sample_ids:
        missing = sorted(sample_ids - set(records_by_id))
        raise RuntimeError(
            f"Confidence checkpoints are missing {len(missing)} samples: {missing[:5]}"
        )

    records = list(records_by_id.values())
    records.sort(key=lambda item: (item["task_index"], item["initial_state_index"]))
    results_root = Path(results_root)
    save_evaluation(records, dataset_split, results_root / "all")
    for partition in ("calibration", "validation", "test"):
        partition_records = [
            record for record in records if record.get("partition") == partition
        ]
        save_evaluation(
            partition_records,
            f"{dataset_split}_{partition}",
            results_root / partition,
        )
    signatures = [
        json.loads(Path(path).read_text(encoding="utf-8"))["signature"]
        for path in checkpoint_paths
    ]
    summary = _experiment_summary(
        records,
        {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "requested_model": signatures[0]["requested_model"],
            "score_type": "raw_label_likelihood",
        },
    )
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _load_checkpoint(checkpoint_path, signature):
    if not checkpoint_path.is_file():
        return []
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise RuntimeError(
            "Existing confidence checkpoint does not match the current manifest, "
            "model, or score definition. Move it aside before starting a new run."
        )
    records = list(payload.get("records", []))
    if len({record.get("sample_id") for record in records}) != len(records):
        raise RuntimeError("Confidence checkpoint contains duplicate sample IDs.")
    print(f"Resuming after {len(records)} completed associations.")
    return records


def _save_checkpoint(checkpoint_path, signature, records):
    temporary_path = checkpoint_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "signature": signature,
                "completed_count": len(records),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def _evaluate_with_retry(sample, provider, max_attempts=MAX_PROVIDER_ATTEMPTS):
    """Retry transient provider failures without changing completed records."""
    for attempt in range(1, max_attempts + 1):
        try:
            return evaluate_samples([sample], provider)[0]
        except Exception as error:
            if not _is_retryable_provider_error(error) or attempt == max_attempts:
                raise
            delay_s = min(2 ** (attempt - 1), 30)
            print(
                f"Transient provider failure for {sample['sample_id']} "
                f"(attempt {attempt}/{max_attempts}); retrying in {delay_s}s: "
                f"{type(error).__name__}"
            )
            time.sleep(delay_s)


def _is_retryable_provider_error(error):
    status_code = getattr(error, "status_code", None)
    return (
        status_code == 429
        or isinstance(status_code, int) and status_code >= 500
        or type(error).__name__
        in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
    )


def _experiment_summary(records, signature):
    scored = [record for record in records if record["association_score"] is not None]
    correct = sum(bool(record["correct"]) for record in records)
    scored_correct = sum(bool(record["correct"]) for record in scored)
    return {
        "schema_version": 1,
        "signature": signature,
        "total_associations": len(records),
        "raw_label_likelihood_available": len(scored),
        "score_unavailable": len(records) - len(scored),
        "ungated_association_accuracy": correct / len(records) if records else None,
        "scored_association_accuracy": (
            scored_correct / len(scored) if scored else None
        ),
        "partition_counts": {
            partition: sum(record.get("partition") == partition for record in records)
            for partition in ("calibration", "validation", "test")
        },
        "method_input_excludes_simulator_ground_truth": True,
        "threshold_selected": False,
    }


if __name__ == "__main__":
    worker_tasks = os.environ.get("LIBERO_CONFIDENCE_TASKS")
    if worker_tasks:
        task_indices = tuple(int(value) for value in worker_tasks.split(","))
        worker_root = os.environ.get("LIBERO_CONFIDENCE_WORKER_ROOT")
        if not worker_root:
            raise SystemExit(
                "LIBERO_CONFIDENCE_WORKER_ROOT is required for a task worker."
            )
        run_task_worker(task_indices, Path(worker_root))
    else:
        run_experiment()
