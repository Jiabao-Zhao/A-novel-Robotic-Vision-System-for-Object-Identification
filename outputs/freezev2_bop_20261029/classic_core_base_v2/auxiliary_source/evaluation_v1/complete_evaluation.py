"""Queued, evaluation-only scoring of the unchanged baseline including failures.

This auxiliary script never edits inference source, masks, descriptors, CSVs,
primary status, or primary evaluation receipts. Both main OS locks must be held
before any evaluator is launched; blocked/failed targets remain in its official
target denominator.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import numpy as np

DATASETS = ("lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv")
PUBLIC = ("lmo", "tless", "tudl", "icbin", "ycbv")
METRICS = ("bop19_average_recall", "bop19_average_recall_vsd",
           "bop19_average_recall_mssd", "bop19_average_recall_mspd")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def local_path(value):
    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:].replace("\\", "/")
    return Path(text).expanduser().resolve()


def load_frozen(run_root):
    """Check every sealed primary source file before importing its small helpers."""
    snapshot = Path(run_root) / "source_snapshot"
    auxiliary_manifest = Path(__file__).with_name("source_manifest.json")
    if auxiliary_manifest.is_file():
        auxiliary = read_json(auxiliary_manifest)
        if (auxiliary.get("files", {}).get(Path(__file__).name) != sha256(__file__)
                or auxiliary.get("primary_source_manifest_sha256") != sha256(snapshot / "source_manifest.json")):
            raise ValueError("Auxiliary source seal differs from its helper or primary snapshot")
    manifest = read_json(snapshot / "source_manifest.json")
    for relative, expected in manifest["files"].items():
        actual = (snapshot / relative).resolve()
        if not actual.is_relative_to(snapshot.resolve()) or sha256(actual) != expected:
            raise ValueError(f"Primary source snapshot checksum mismatch: {relative}")
    for name, module in list(sys.modules.items()):
        if name == "freezev2_repro" or name.startswith("freezev2_repro."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(snapshot.resolve()):
                raise RuntimeError("A mutable reproduction package is already imported; use a fresh helper process")
    sys.path.insert(0, str(snapshot))
    return dict(durable=importlib.import_module("freezev2_repro.durable"),
                owned=importlib.import_module("freezev2_repro.owned_process"),
                evaluation=importlib.import_module("freezev2_repro.evaluation"),
                source_manifest=manifest,
                source_manifest_sha256=sha256(snapshot / "source_manifest.json"))


def try_main_locks(run_root, worker_lock):
    """Return an acquired ExitStack or None; never clear or infer locks from JSON."""
    stack = ExitStack()
    try:
        stack.enter_context(worker_lock(Path(run_root) / "supervisor.lock"))
        worker_stream = stack.enter_context(worker_lock(Path(run_root) / "worker.lock"))
        return stack, worker_stream
    except RuntimeError:
        stack.close()
        return None
    except BaseException:
        stack.close()
        raise


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {label}")
    return value


def _pose_tuple(row, *, csv_units=False):
    scene = _integer(row["scene_id"], "prediction scene")
    image = _integer(row["image_id"], "prediction image")
    obj = _integer(row["object_id"], "prediction object", 1)
    rotation = np.asarray(row["R_m2c"], dtype=float)
    translation = np.asarray(row["t_m"], dtype=float).reshape(-1)
    score, elapsed = float(row["score"]), float(row["image_time_s"])
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Invalid saved pose dimensions")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not math.isfinite(score) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Nonfinite saved pose or negative runtime")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
        raise ValueError("Saved rotation is not proper SO(3)")
    # Compare the original CSV millimetres directly against the writer's metre
    # conversion. Avoid a second /1000 round trip introducing an extra ULP.
    millimetres = translation if csv_units else translation * 1000.
    return (scene, image, obj, score, *rotation.reshape(-1).tolist(), *millimetres.tolist(), elapsed)


def read_csv_poses(path):
    poses = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]:
            raise ValueError("Unexpected BOP CSV header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed BOP CSV row")
            poses.append(_pose_tuple(dict(scene_id=int(row["scene_id"]), image_id=int(row["im_id"]),
                object_id=int(row["obj_id"]), score=float(row["score"]),
                R_m2c=np.asarray([float(x) for x in row["R"].split()]).reshape(3, 3),
                t_m=[float(x) for x in row["t"].split()], image_time_s=float(row["time"])), csv_units=True))
    # An empty prediction set is a possible recorded outcome, never successful
    # inference. The full target file still defines the evaluator denominator.
    return poses


def onboarding_runtime_summary(run_root, dataset, expected):
    """Completed-stage active time only; partial/failed tails are not invented."""
    root = Path(run_root) / "datasets" / dataset / "onboarding"
    stages = {}
    for stage in ("geometry", "features"):
        completed, partial = [], []
        for folder in sorted(root.glob(f"obj_*/{stage}")):
            complete = folder / "complete.json"
            if complete.is_file():
                value = read_json(complete)
                seconds = float(value["runtime_s"])
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("Invalid completed CAD stage runtime")
                completed.append(seconds)
            elif (folder / "progress.json").is_file():
                seconds = float(read_json(folder / "progress.json").get("active_runtime_s", 0.))
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("Invalid partial CAD checkpoint runtime")
                partial.append(seconds)
        stages[stage] = dict(completed_CADs=len(completed), expected_CADs=expected,
                             completed_stage_active_runtime_s=sum(completed),
                             partial_CADs_with_progress=len(partial), partial_checkpointed_active_runtime_s=sum(partial))
    return dict(stages=stages,
                completed_stage_active_runtime_s=sum(row["completed_stage_active_runtime_s"] for row in stages.values()),
                partial_checkpointed_active_runtime_s=sum(row["partial_checkpointed_active_runtime_s"] for row in stages.values()),
                scope="Sum of saved completed-stage active runtimes; partial progress is separate. Failed/unfinished CAD work after its last checkpoint, including a failing GeDi tail, is not fully logged and is not included.")


def audit_dataset(run_root, dataset, asset, run_manifest):
    """Audit exact target/count coverage and the unchanged CSV against ledgers."""
    root = Path(run_root)
    target_path = local_path(asset["targets"]["path"])
    if sha256(target_path) != asset["targets"]["sha256"]:
        raise ValueError("Official target source checksum changed")
    expected = {}
    for row in read_json(target_path):
        key = (_integer(row["scene_id"], "target scene"), _integer(row["im_id"], "target image"),
               _integer(row["obj_id"], "target object", 1))
        if key in expected:
            raise ValueError("Duplicate official target key")
        expected[key] = _integer(row["inst_count"], "target instance count", 1)
    if len(expected) != asset["target_records"] or sum(expected.values()) != asset["requested_instances"]:
        raise ValueError("Official target records/instance denominator differs from assets manifest")
    expected_images = {(key[0], key[1]) for key in expected}
    if len(expected_images) != asset["official_target_images"]:
        raise ValueError("Official target image denominator differs from assets manifest")
    expected_run = dict(method_digest=digest(run_manifest["method_identity"]), dataset=dataset,
        targets_sha256=asset["targets"]["sha256"], mask_sha256=asset["masks"]["sha256"],
        original_input_manifest_sha256=run_manifest["original_input_manifest_sha256"],
        camera_snapshot_digest=run_manifest["camera_snapshot_digest"])
    coverage_errors, export_errors = [], []
    found, ledger_poses, ledger_files = {}, [], []
    counts = Counter(completed=0, failed=0, blocked=0)
    target_reasons = Counter()
    image_count, runtime = 0, 0.
    image_root = root / "datasets" / dataset / "images"
    for folder in sorted(image_root.glob("*")):
        if not folder.is_dir():
            continue
        file = folder / "complete.json"
        if not file.is_file():
            file = folder / "partial.json"
        if not file.is_file():
            continue
        value = read_json(file)
        ledger_files.append(dict(path=str(file.relative_to(root)), sha256=sha256(file)))
        identity = value["identity"]
        scene, image = _integer(identity["scene_id"], "ledger scene"), _integer(identity["image_id"], "ledger image")
        if identity.get("run") != expected_run:
            coverage_errors.append(f"Ledger run/source identity mismatch: {file}")
        image_count += 1
        elapsed = float(value["runtime_s"])
        if not math.isfinite(elapsed) or elapsed < 0:
            coverage_errors.append(f"Invalid image runtime: {file}")
        else:
            runtime += elapsed
        rows = value["targets"]
        requested_rows = sorted([[row["object_id"], row["requested_instances"]] for row in rows])
        if sorted(identity.get("target_rows", [])) != requested_rows:
            coverage_errors.append(f"Ledger target identity/counts disagree with its rows: {file}")
        by_object = Counter()
        for pose in value["predictions"]:
            try:
                item = _pose_tuple(pose)
                if item[:2] != (scene, image):
                    raise ValueError("Prediction belongs to another image")
                if item[:3] not in expected:
                    raise ValueError("Prediction is outside official target keys")
                if not math.isclose(item[-1], elapsed, abs_tol=1e-9, rel_tol=0):
                    raise ValueError("Prediction time differs from complete image time")
                ledger_poses.append(item)
                by_object[item[2]] += 1
            except (ValueError, KeyError, TypeError) as exc:
                export_errors.append(f"Invalid ledger prediction {file}: {exc}")
        for row in rows:
            obj = _integer(row["object_id"], "ledger object", 1)
            key = scene, image, obj
            if key in found:
                coverage_errors.append(f"Duplicate effective ledger target {key}")
            found[key] = row
            if key not in expected:
                coverage_errors.append(f"Unknown effective ledger target {key}")
            elif row["requested_instances"] != expected[key]:
                coverage_errors.append(f"Requested instance-count mismatch {key}")
            status = row["status"]
            if row.get("reason") is not None:
                target_reasons[(status, str(row["reason"]))] += 1
            if status not in counts:
                coverage_errors.append(f"Unknown target status {key}: {status}")
            else:
                counts[status] += 1
            if row["valid_predictions"] != by_object[obj] or by_object[obj] > row["requested_instances"]:
                export_errors.append(f"Target pose count disagrees with ledger predictions {key}")
            if status == "completed" and by_object[obj] != row["requested_instances"]:
                export_errors.append(f"Completed target lacks its requested poses {key}")
    missing = sorted(set(expected) - set(found))
    unknown = sorted(set(found) - set(expected))
    if missing:
        coverage_errors.append(f"Missing official ledger targets: {len(missing)}")
    csv_path = root / "datasets" / dataset / "predictions" / f"freezev2base_{dataset}-test.csv"
    export_path = csv_path.parent / "export_receipt.json"
    csv_sha = None
    csv_poses = []
    if not csv_path.is_file() or not export_path.is_file():
        export_errors.append("Saved prediction CSV/export receipt is missing")
    else:
        receipt = read_json(export_path)
        csv_sha = sha256(csv_path)
        if receipt.get("csv_sha256") != csv_sha or receipt.get("target_sha256") != asset["targets"]["sha256"]:
            export_errors.append("CSV/export target or CSV checksum mismatch")
        if receipt.get("source") != "our_inference":
            export_errors.append("Export does not identify our generated inference")
        if receipt.get("expected_targets") != len(expected) or receipt.get("ledger_targets") != len(found):
            export_errors.append("Export target coverage receipt differs from effective ledgers")
        try:
            csv_poses = read_csv_poses(csv_path)
            if Counter(csv_poses) != Counter(ledger_poses) or receipt.get("poses") != len(csv_poses):
                export_errors.append("CSV poses do not exactly match effective saved ledger poses")
        except (ValueError, KeyError, TypeError) as exc:
            export_errors.append(f"Invalid BOP CSV: {exc}")
    return dict(dataset=dataset, expected_images=len(expected_images), attempted_images=image_count,
        expected_targets=len(expected), attempted_targets=len(set(found) & set(expected)),
        expected_instances=sum(expected.values()), completed_targets=counts["completed"],
        failed_targets=counts["failed"], blocked_targets=counts["blocked"],
        target_reason_counts=[dict(status=status, reason=reason, targets=count)
                              for (status, reason), count in sorted(target_reasons.items())],
        missing_targets=len(missing), unknown_targets=len(unknown), valid_pose_predictions=len(csv_poses),
        valid_partial_poses_retained=True, inference_runtime_s=runtime,
        strict_execution_complete=not coverage_errors and counts["blocked"] == 0,
        recorded_inference_failures=bool(counts["failed"] or counts["blocked"]),
        coverage_complete=not coverage_errors, export_valid=not export_errors,
        coverage_errors=coverage_errors, export_errors=export_errors,
        missing_keys=missing, unknown_keys=unknown, ledger_files=ledger_files,
        ledger_digest=digest(ledger_files), csv_path=str(csv_path), csv_sha256=csv_sha,
        export_receipt_path=str(export_path), export_receipt_sha256=sha256(export_path) if export_path.exists() else None,
        official_target_path=str(target_path), target_sha256=asset["targets"]["sha256"],
        mask_sha256=asset["masks"]["sha256"],
        denominator_policy="Unmodified complete official localization targets, including blocked and failed rows")


def checked_scores(path):
    result = read_json(path)
    if any(name not in result or isinstance(result[name], bool) or not isinstance(result[name], (int, float))
           or not math.isfinite(result[name]) or not 0. <= result[name] <= 1. for name in METRICS):
        raise ValueError("Official score JSON is incomplete or invalid")
    return result


def reusable_receipt(path, audit, runtime, dataset_root):
    """Reuse only cryptographically bound original/new complete metric receipts."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        prior = read_json(path)
    except (ValueError, OSError):
        return None
    if prior.get("status") not in {"complete", "complete_with_recorded_inference_failures"}:
        return None
    if (prior.get("csv_sha256") != audit["csv_sha256"] or prior.get("target_sha256") != audit["target_sha256"]
            or prior.get("toolkit_revision") != runtime["toolkit_revision"]):
        return None
    score = local_path(prior.get("score_file", "__missing__"))
    if not score.is_relative_to(Path(dataset_root).resolve()) or not score.is_file() or prior.get("score_sha256") != sha256(score):
        return None
    try:
        metrics = checked_scores(score)
    except (ValueError, OSError):
        return None
    return dict(official_pose_metrics=metrics, score_file=str(score), score_sha256=sha256(score),
                reused_receipt=str(path), reused_receipt_sha256=sha256(path),
                evaluation_runtime_s=prior.get("evaluation_runtime_s", prior.get("runtime_s")),
                command=prior.get("command"), toolkit_revision=runtime["toolkit_revision"])


def check_protocol(protocol, runtime):
    if (set(protocol["datasets"]) != set(DATASETS) or len(protocol["datasets"]) != 7
            or protocol.get("task") != "BOP19_model_based_6D_localization"):
        raise ValueError("Requires the original exact all-seven BOP19 localization protocol")
    if runtime.get("targets_filename") != "test_targets_bop19.json" or runtime.get("renderer_type") != "vispy":
        raise ValueError("Unexpected official target or renderer protocol")
    if runtime.get("protocol", {}).get("visib_gt_min") != -1:
        raise ValueError("Official localization must keep inst_count k-most-visible semantics")


class CompletionEvaluator:
    def __init__(self, root, frozen):
        self.root, self.frozen = Path(root), frozen
        self.atomic = frozen["durable"].atomic_json
        self.protocol = read_json(self.root / "protocol.json")
        self.run_manifest = read_json(self.root / "run_manifest.json")
        if digest(self.protocol) != digest(self.run_manifest["protocol"]):
            raise ValueError("Original run protocol changed")
        self.assets_path = local_path(self.protocol["paths"]["assets_manifest"])
        self.runtime_path = local_path(self.protocol["paths"]["evaluation_runtime"])
        self.assets, self.runtime = read_json(self.assets_path), read_json(self.runtime_path)
        check_protocol(self.protocol, self.runtime)
        self.toolkit = local_path(self.runtime["toolkit_root"])
        actual = subprocess.check_output(["git", "-C", str(self.toolkit), "rev-parse", "HEAD"], text=True).strip()
        if actual != self.runtime["toolkit_revision"]:
            raise ValueError("Actual official BOP Toolkit revision differs from pinned runtime")
        self.pins = dict(helper_sha256=sha256(__file__), primary_source_manifest_sha256=frozen["source_manifest_sha256"],
            original_protocol_sha256=sha256(self.root / "protocol.json"), run_manifest_sha256=sha256(self.root / "run_manifest.json"),
            assets_manifest_sha256=sha256(self.assets_path), evaluation_runtime_sha256=sha256(self.runtime_path),
            toolkit_revision=actual)
        self.result = dict(schema="freezev2-recorded-failure-evaluation-v1", datasets={}, source_pins=self.pins,
            method=self.protocol["method"], mask_configuration=self.protocol["mask_configuration"],
            base_settings=self.protocol["base"], original_protocol_path=str(self.root / "protocol.json"),
            original_protocol_sha256=self.pins["original_protocol_sha256"],
            predictions_origin="our_unchanged_frozen_baseline_inference", inference_variant_changed=False,
            all_seven_aggregate=None, all_seven_aggregate_unavailable_reason="HB and ITODD official test GT is private",
            claim="All requested instances stay in the unmodified official denominator. Surviving saved poses are retained; missing predictions reduce the score. No dummy/zero predictions are inserted. Original execution limitations remain explicit; no paper-performance parity claim.")

    def heartbeat(self, **fields):
        value = dict(pid=os.getpid(), linux_start_ticks=self.frozen["owned"].start_ticks(os.getpid()),
                     updated_epoch=time.time(), source_pins=self.pins, **fields)
        self.atomic(self.root / "failure_inclusive_evaluation_status.json", value)

    def dataset(self, name, lock_stream):
        asset = self.assets["datasets"][name]
        audit = audit_dataset(self.root, name, asset, self.run_manifest)
        directory = self.root / "datasets" / name
        receipt = directory / "evaluation_with_recorded_failures.json"
        entry = dict(**audit, source_pins=self.pins, official_pose_metrics=None, evaluation_runtime_s=None,
                     toolkit_revision=self.runtime["toolkit_revision"], GT_selection="official inst_count k-most-visible; visib_gt_min=-1",
                     original_execution_incomplete=audit["blocked_targets"] > 0,
                     onboarding=onboarding_runtime_summary(self.root, name, asset["cad_objects"]),
                     original_primary_evaluation_receipt_unchanged=True)
        states_path = self.root / "dataset_states.json"
        entry["original_CAD_blockers"] = (read_json(states_path).get(name, {}).get("CAD_blockers", {})
                                          if states_path.is_file() else {})
        entry["CAD_blocker_interpretation"] = ("The original GeDi forward produced nonfinite point descriptors. "
            "Our finite-output guard rejected the affected entire CAD feature cache. "
            "Author handling is unknown; no paper-performance parity is claimed.")
        view_targets = local_path(self.runtime["dataset_view"]) / name / self.runtime["targets_filename"]
        if not view_targets.is_file() or sha256(view_targets) != audit["target_sha256"]:
            entry.update(status="blocked_official_target_identity", reason="Evaluation view does not contain the exact official target denominator")
        elif name not in PUBLIC:
            entry.update(status="unavailable_private_GT", reason="Official HB/ITODD test GT is withheld; no remote submission attempted")
        elif not audit["coverage_complete"]:
            entry.update(status="blocked_ledger_coverage", reason="Every official target/count must appear exactly once, including failures")
        elif not audit["export_valid"]:
            entry.update(status="blocked_prediction_export", reason="Saved CSV/export receipt must match the exact effective ledgers")
        elif not asset.get("evaluation", {}).get("ready", False):
            entry.update(status="blocked_evaluation_assets", reason=asset.get("evaluation", {}).get("blockers"))
        else:
            reused = (reusable_receipt(directory / "evaluation_receipt.json", audit, self.runtime, directory)
                      or reusable_receipt(receipt, audit, self.runtime, directory))
            if reused:
                entry.update(reused)
            else:
                identity = digest(dict(csv=audit["csv_sha256"], target=audit["target_sha256"], pins=self.pins))[:24]
                evaluation_root = directory / "official_evaluation_with_recorded_failures" / identity
                evaluation_root.mkdir(parents=True, exist_ok=True)
                command, environment = self.frozen["evaluation"].build_evaluation_command(
                    toolkit_root=self.toolkit, results_file=audit["csv_path"],
                    datasets_root=local_path(self.runtime["dataset_view"]), evaluation_root=evaluation_root,
                    targets_filename=self.runtime["targets_filename"], python_executable=self.runtime["python"],
                    renderer_type="vispy", num_workers=1)
                environment.update(self.runtime.get("environment", {}))
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = "4"
                started = time.perf_counter()
                returncode = self.frozen["owned"].run_owned(command, environment, evaluation_root, lock_stream,
                    lambda pid: self.heartbeat(stage="official_failure_inclusive_evaluation", dataset=name, evaluator_guard_pid=pid))
                entry.update(command=command, evaluation_runtime_s=time.perf_counter() - started,
                             evaluator_returncode=returncode, log=str(evaluation_root / "execution.log"))
                score = evaluation_root / Path(audit["csv_path"]).stem / "scores_bop19.json"
                if returncode == 0 and score.is_file():
                    entry.update(official_pose_metrics=checked_scores(score), score_file=str(score), score_sha256=sha256(score))
                else:
                    entry.update(status="failed_official_evaluator", reason="Official evaluator did not complete successfully")
            if entry["official_pose_metrics"] is not None:
                entry["status"] = "complete_with_recorded_inference_failures" if audit["recorded_inference_failures"] else "complete"
        self.atomic(receipt, entry)
        return entry

    def save_report(self):
        rows = self.result["datasets"]
        measured = all(rows.get(name, {}).get("official_pose_metrics") is not None for name in PUBLIC)
        failures = any(row.get("recorded_inference_failures") for row in rows.values())
        self.result["status"] = ("complete_with_recorded_inference_failures" if measured and failures
                                 else "complete_public_datasets_private_GT_unavailable" if measured else "blocked_or_incomplete_evaluation")
        self.result["original_execution_incomplete"] = any(row.get("blocked_targets", 0) or row.get("missing_targets", 0) for row in rows.values())
        self.result["updated_epoch"] = time.time()
        self.atomic(self.root / "results_with_recorded_failures.json", self.result)
        text = ["# Failure-inclusive evaluation of the unchanged FreeZeV2 baseline", "", self.result["claim"], "",
            f"Method: `{self.result['method']}`; masks: `{self.result['mask_configuration']}`. Exact original settings: `{self.result['original_protocol_path']}` (SHA256 `{self.result['original_protocol_sha256']}`).", "",
            "No inference was rerun or altered. Original status, CSVs and primary evaluation receipts remain unchanged. A measured score does not establish that strict inference execution completed.", "",
            "| Dataset | Expected targets | Attempted | Completed | Failed | Blocked | Missing | Poses | Official AR | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name in DATASETS:
            row = rows.get(name)
            if row is None:
                continue
            metric = row.get("official_pose_metrics")
            ar = f"{100 * metric['bop19_average_recall']:.4f}%" if metric else "unavailable"
            text.append(f"| {name} | {row.get('expected_targets', 'unknown')} | {row.get('attempted_targets', 'unknown')} | {row.get('completed_targets', 'unknown')} | {row.get('failed_targets', 'unknown')} | {row.get('blocked_targets', 'unknown')} | {row.get('missing_targets', 'unknown')} | {row.get('valid_pose_predictions', 'unknown')} | {ar} | {row['status']} |")
        text += ["", "| Dataset | Geometry CADs | Feature CADs | Completed onboarding active s | Partial checkpointed active s | Saved inference active s | Evaluation elapsed s |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for name in DATASETS:
            row = rows.get(name, {})
            onboard = row.get("onboarding")
            if onboard is None:
                continue
            geometry, features = onboard["stages"]["geometry"], onboard["stages"]["features"]
            evaluation_time = row.get("evaluation_runtime_s")
            evaluation_time = f"{evaluation_time:.3f}" if isinstance(evaluation_time, (int, float)) else "unavailable"
            text.append(f"| {name} | {geometry['completed_CADs']}/{geometry['expected_CADs']} | {features['completed_CADs']}/{features['expected_CADs']} | {onboard['completed_stage_active_runtime_s']:.3f} | {onboard['partial_checkpointed_active_runtime_s']:.3f} | {row['inference_runtime_s']:.3f} | {evaluation_time} |")
        text += ["", "Completed onboarding time sums only saved complete-stage active runtimes. Partial checkpointed time is separate and does not include all failed/unfinished work after its last checkpoint, including a failing GeDi tail. Saved inference time follows the original image-ledger timing policy; evaluator elapsed time is separate and may be reused from a valid primary receipt.",
            "", "All failed/blocked targets and requested instances remain in the official target denominator. Surviving saved poses are retained, even within a failed/blocked target; missing predictions reduce the official score. No dummy predictions are inserted. Evaluation uses inst_count most-visible GT selection (visib_gt_min=-1), with GT confined to the official evaluator.",
            "", "HB/ITODD local scores and the seven-dataset aggregate are unavailable because their test GT is private. No five-dataset macro or paper-performance parity claim is made.", "",
            "Per-dataset VSD/MSSD/MSPD, inference/evaluation runtimes, exact ledger/CSV/target hashes, source pins and blockers are in results_with_recorded_failures.json and datasets/<dataset>/evaluation_with_recorded_failures.json.", "",
            f"Auxiliary helper SHA256: `{self.pins['helper_sha256']}`.",
            f"Primary frozen source-manifest SHA256: `{self.pins['primary_source_manifest_sha256']}`.",
            f"Official toolkit revision: `{self.pins['toolkit_revision']}`."]
        for name in DATASETS:
            row = rows.get(name, {})
            if row.get("original_CAD_blockers"):
                text += ["", f"{name} CAD feature-cache blockers: `{json.dumps(row['original_CAD_blockers'], sort_keys=True)}`.",
                         row["CAD_blocker_interpretation"]]
            if row.get("target_reason_counts"):
                text += ["", f"{name} recorded target reasons: `{json.dumps(row['target_reason_counts'], sort_keys=True)}`."]
        temporary = self.root / "FAILURE_INCLUSIVE_EVALUATION.md.tmp"
        temporary.write_text("\n".join(text) + "\n", encoding="utf-8")
        temporary.replace(self.root / "FAILURE_INCLUSIVE_EVALUATION.md")

    def run(self, lock_stream):
        for name in DATASETS:
            try:
                if sha256(__file__) != self.pins["helper_sha256"]:
                    raise RuntimeError("Auxiliary helper source changed during evaluation")
                self.heartbeat(stage="auditing_saved_predictions", dataset=name)
                entry = self.dataset(name, lock_stream)
            except Exception as exc:
                entry = dict(dataset=name, status="blocked_evaluation_audit", official_pose_metrics=None,
                             reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), source_pins=self.pins)
                self.atomic(self.root / "datasets" / name / "evaluation_with_recorded_failures.json", entry)
            self.result["datasets"][name] = entry
            self.save_report()
        self.heartbeat(stage="finished", status=self.result["status"], all_seven_aggregate=None)
        return 0 if self.result["status"].startswith("complete") else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--once", action="store_true", help="Fail closed if either main lock is held; never wait or start competing evaluation")
    parser.add_argument("--probe-queue-lock", action="store_true", help="Probe only the auxiliary queue OS lock; no status writes")
    args = parser.parse_args(argv)
    if os.name != "posix":
        raise RuntimeError("Run the queued evaluator in its recorded WSL environment")
    root = local_path(args.run_root)
    frozen = load_frozen(root)
    lock = frozen["durable"].worker_lock
    try:
        with lock(root / "post_evaluation.lock"):
            if args.probe_queue_lock:
                print(json.dumps(dict(live_queue_owner=False)))
                return 0
            identity = dict(pid=os.getpid(), linux_start_ticks=frozen["owned"].start_ticks(os.getpid()),
                actual_cmdline=Path("/proc/self/cmdline").read_bytes().decode().split("\0")[:-1],
                started_epoch=time.time(), helper_sha256=sha256(__file__),
                primary_source_manifest_sha256=frozen["source_manifest_sha256"])
            frozen["durable"].atomic_json(root / "failure_inclusive_evaluation_process.json", identity)
            while True:
                acquired = try_main_locks(root, lock)
                if acquired is not None:
                    stack, worker_stream = acquired
                    with stack:
                        try:
                            return CompletionEvaluator(root, frozen).run(worker_stream)
                        except Exception as exc:
                            frozen["durable"].atomic_json(root / "failure_inclusive_evaluation_status.json",
                                dict(**identity, updated_epoch=time.time(), stage="failed_initialization_or_evaluation",
                                     status="blocked_or_failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
                            return 2
                frozen["durable"].atomic_json(root / "failure_inclusive_evaluation_status.json",
                    dict(**identity, updated_epoch=time.time(), stage="waiting_for_main_OS_locks",
                         status="blocked_live_main_owner" if args.once else "queued", poll_seconds=15))
                if args.once:
                    return 3
                time.sleep(15)
    except RuntimeError as exc:
        if "owns the OS lock" in str(exc):
            print(json.dumps(dict(live_queue_owner=True, reason=str(exc))))
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
