"""Reporting only: finalize verified saved FreeZeV2 results, never run inference/metrics.

The only generated files are FINAL_REPORT.md and final_results.json. Missing
full-target coverage, evaluation receipts or reviewed evidence blocks both writes.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
DATASETS = ("lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv")
PUBLIC = ("lmo", "tless", "tudl", "icbin", "ycbv")
TOTALS = dict(images=3871, target_rows=13635, requested_instances=19048,
              available_CADs=125, target_required_CADs=108)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bound_file(path, helper):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing required final evidence: {path}")
    return dict(path=str(path), sha256=helper.sha256(path), bytes=path.stat().st_size)


def load_helper(root):
    folder = root / "auxiliary_source/evaluation_v1"
    source, manifest_path = folder / "complete_evaluation.py", folder / "source_manifest.json"
    manifest = read(manifest_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    primary_sha = hashlib.sha256((root / "source_snapshot/source_manifest.json").read_bytes()).hexdigest()
    if (manifest.get("schema") != "freezev2-auxiliary-evaluation-source-v1"
            or manifest.get("files", {}).get(source.name) != source_sha
            or manifest.get("primary_source_manifest_sha256") != primary_sha):
        raise ValueError("Auxiliary evaluator source/primary binding changed")
    spec = importlib.util.spec_from_file_location("sealed_failure_inclusive_evaluation", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Verifies all19 primary source files, then imports only small CPU helpers.
    frozen = helper.load_frozen(root)
    return helper, frozen


def ledger_details(root, audit, helper):
    instances, retained = Counter(), Counter()
    examples = {}
    for item in audit["ledger_files"]:
        file = root / item["path"]
        if helper.sha256(file) != item["sha256"]:
            raise ValueError("A target ledger changed during final reporting")
        value = read(file)
        for row in value["targets"]:
            status = row["status"]
            instances[status] += row["requested_instances"]
            retained[status] += row["valid_predictions"]
            if row.get("reason") == "no_predicted_masks" and "no_predicted_masks" not in examples:
                examples["no_predicted_masks"] = dict(ledger=bound_file(file, helper), object_id=row["object_id"],
                    requested_instances=row["requested_instances"], reason=row["reason"])
            if "observed_GeDi_failure" not in examples:
                for candidate in row.get("candidates", []):
                    if candidate["status"] != "failed":
                        continue
                    diagnostic = file.parent / candidate["directory"] / "diagnostic.json"
                    if helper.sha256(diagnostic) != candidate["diagnostic_sha256"]:
                        raise ValueError("Saved candidate diagnostic no longer matches its ledger")
                    trace = read(diagnostic)
                    if "GeDi" in str(trace.get("reason", "")):
                        examples["observed_GeDi_failure"] = dict(diagnostic=bound_file(diagnostic, helper),
                            candidate_id=candidate["candidate_id"], reason=trace["reason"],
                            ground_truth_comparison=False)
                        break
    if sum(instances.values()) != audit["expected_instances"] or sum(retained.values()) != audit["valid_pose_predictions"]:
        raise ValueError("Instance/partial-pose reconciliation failed")
    return dict(requested_instances_by_target_status=dict(instances), retained_poses_by_target_status=dict(retained),
                missing_requested_predictions=audit["expected_instances"] - audit["valid_pose_predictions"],
                representative_recorded_failures=examples)


def timing_summary(audit, helper, metrics):
    times = {}
    for pose in helper.read_csv_poses(audit["csv_path"]):
        key, elapsed = pose[:2], pose[-1]
        if key in times and times[key] != elapsed:
            raise ValueError("CSV timing differs within one image")
        times[key] = elapsed
    csv_mean = sum(times.values()) / len(times) if times else None
    official_mean = metrics.get("bop19_average_time_per_image") if metrics else None
    if official_mean is not None and (csv_mean is None or not math.isclose(official_mean, csv_mean, rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError("Official time metric does not match the CSV-image denominator")
    return dict(CSV_images_with_predictions=len(times), official_target_images=audit["expected_images"],
        images_without_CSV_predictions=audit["expected_images"] - len(times),
        CSV_image_mean_s=csv_mean, official_BOP_image_mean_s=official_mean,
        all_attempted_image_active_s=audit["inference_runtime_s"],
        all_attempted_image_mean_s=audit["inference_runtime_s"] / audit["attempted_images"],
        policy="Saved active current image wall time plus retained candidate compute; interruption downtime excluded.",
        exclusions=["external CNOS/FastSAM segmentation", "model loading", "CAD onboarding", "official evaluation", "unlogged failed/unfinished onboarding tail"],
        warning="Official mean includes only images represented in the CSV, not every official target image.")


def ledger_digest_matches(saved, audit, helper):
    # Existing scoped audits were produced on Windows; evaluation runs in WSL.
    # Hash the exact same relative paths/file bytes with either separator spelling.
    options = {audit["ledger_digest"]}
    for separator in ("/", "\\"):
        options.add(helper.digest([dict(item, path=item["path"].replace("\\", "/").replace("/", separator))
                                   for item in audit["ledger_files"]]))
    return saved in options


def verify_failure_inclusive_receipt(root, name, audit, runtime, helper, source_pins=None):
    receipt = root / "datasets" / name / "evaluation_with_recorded_failures.json"
    value = read(receipt)
    if (value.get("predictions_origin") not in (None, "our_unchanged_frozen_baseline_inference")
            or not value.get("coverage_complete") or not value.get("export_valid")
            or not ledger_digest_matches(value.get("ledger_digest"), audit, helper)):
        raise ValueError(f"{name}: failure-inclusive receipt is not bound to the complete current ledgers")
    if source_pins and any(value.get("source_pins", {}).get(key) != expected for key,expected in source_pins.items()):
        raise ValueError(f"{name}: evaluation receipt source/runtime identity changed")
    for field in ("expected_images", "expected_targets", "expected_instances", "attempted_targets",
                  "completed_targets", "failed_targets", "blocked_targets", "valid_pose_predictions"):
        if value.get(field) != audit[field]:
            raise ValueError(f"{name}: evaluation receipt count mismatch: {field}")
    measured = helper.reusable_receipt(receipt, audit, runtime, receipt.parent)
    if measured is None:
        raise ValueError(f"{name}: missing/incompatible CSV/target/toolkit/score hash receipt")
    return dict(receipt=bound_file(receipt, helper), **measured)


def reviewed_evidence(root, name, audit, helper):
    full_path = root / f"{name}_full_pass_audit_20261030.json"
    full = read(full_path)
    saved = full.get("audit", {})
    if (saved.get("csv_sha256") != audit["csv_sha256"] or not ledger_digest_matches(saved.get("ledger_digest"), audit, helper)
            or saved.get("target_sha256") != audit["target_sha256"]):
        raise ValueError(f"{name}: final scoped audit does not bind current CSV/target/ledger bytes")
    visual = root / ("validation_visual_review.json" if name == "tless" else f"{name}_visual_review_20261030.json")
    review = read(visual)
    if len(review.get("cases", [])) < 3:
        raise ValueError(f"{name}: requires at least three recorded visual review cases")
    raw = []
    for case in review["cases"]:
        if "diagnostic" in case:
            diagnostic = root / case["diagnostic"].replace("\\", "/")
            files = [(diagnostic, case["diagnostic_sha256"])]
            files += [(diagnostic.parent / relative, expected) for relative,expected in case.get("artifact_sha256", {}).items()]
        else:
            files = [(helper.local_path(case["overlay"]), case["sha256"])]
        for path,expected in files:
            binding = bound_file(path, helper)
            if binding["sha256"] != expected:
                raise ValueError(f"{name}: reviewed visual artifact changed: {path}")
            raw.append(binding)
    return dict(full_audit=bound_file(full_path, helper), visual_review=bound_file(visual, helper),
        visually_reviewed_cases=len(review["cases"]), visual_scope=review.get("scope"),
        reviewed_artifact_bindings=raw,
        warning="Representative visual sanity checks are not complete-dataset GT pose verification.")


def complete_audits(root, assets, manifest, helper):
    audits = {}
    for name in DATASETS:
        audit = helper.audit_dataset(root, name, assets["datasets"][name], manifest)
        if not audit["coverage_complete"] or not audit["export_valid"] or audit["attempted_images"] != audit["expected_images"]:
            raise ValueError(f"{name}: final report unavailable: coverage/export not finalized: {audit['coverage_errors'] + audit['export_errors']}")
        audits[name] = audit
    return audits


def build_report(root, helper, frozen):
    protocol, manifest = read(root / "protocol.json"), read(root / "run_manifest.json")
    assets_path = helper.local_path(protocol["paths"]["assets_manifest"])
    runtime_path = helper.local_path(protocol["paths"]["evaluation_runtime"])
    assets, runtime = read(assets_path), read(runtime_path)
    helper.check_protocol(protocol, runtime)
    if helper.digest(protocol) != helper.digest(manifest["protocol"]):
        raise ValueError("Run protocol no longer matches its frozen manifest")
    actual_toolkit = subprocess.check_output(["git", "-C", str(helper.local_path(runtime["toolkit_root"])), "rev-parse", "HEAD"], text=True).strip()
    if actual_toolkit != runtime["toolkit_revision"]:
        raise ValueError("Actual BOP Toolkit revision changed")
    expected_pins = dict(helper_sha256=helper.sha256(helper.__file__),
        primary_source_manifest_sha256=frozen["source_manifest_sha256"], toolkit_revision=actual_toolkit,
        original_protocol_sha256=helper.sha256(root/"protocol.json"), run_manifest_sha256=helper.sha256(root/"run_manifest.json"),
        assets_manifest_sha256=helper.sha256(assets_path), evaluation_runtime_sha256=helper.sha256(runtime_path))
    # First finish coverage preflight for ALL datasets; no final artifacts on a partial run.
    audits = complete_audits(root, assets, manifest, helper)
    rows = {}
    for name in DATASETS:
        asset, audit = assets["datasets"][name], audits[name]
        targets = read(helper.local_path(asset["targets"]["path"]))
        required = sorted({row["obj_id"] for row in targets})
        inventory = sorted(asset["object_ids"])
        if not set(required).issubset(inventory):
            raise ValueError(f"{name}: required CAD is absent from inventory")
        view_target = helper.local_path(runtime["dataset_view"]) / name / runtime["targets_filename"]
        if helper.sha256(view_target) != audit["target_sha256"]:
            raise ValueError(f"{name}: evaluation view target denominator changed")
        evaluation = verify_failure_inclusive_receipt(root, name, audit, runtime, helper, expected_pins) if name in PUBLIC else None
        metrics = evaluation["official_pose_metrics"] if evaluation else None
        evidence = reviewed_evidence(root, name, audit, helper)
        onboard = helper.onboarding_runtime_summary(root, name, len(required))
        physical = {}
        for stage in ("geometry", "features"):
            physical[stage] = sorted(int(path.parents[1].name[4:]) for path in
                (root / "datasets" / name / "onboarding").glob(f"obj_*/{stage}/complete.json"))
            if not set(physical[stage]).issubset(required):
                raise ValueError(f"{name}: unexpected onboarded CAD outside target-required scope")
        details = ledger_details(root, audit, helper)
        rows[name] = dict(**{k:v for k,v in audit.items() if k not in ("ledger_files", "missing_keys", "unknown_keys")},
            **details, evaluation=evaluation, official_pose_metrics=metrics,
            evaluation_status="complete_with_recorded_failures" if evaluation and audit["recorded_inference_failures"] else "complete" if evaluation else "unavailable_private_GT",
            CADs=dict(available=len(inventory), target_required=len(required), target_required_ids=required,
                unused_inventory_ids=sorted(set(inventory)-set(required)), physical_completion_ids=physical,
                missing_required_feature_ids=sorted(set(required)-set(physical["features"]))),
            onboarding=onboard, timing=timing_summary(audit, helper, metrics), evidence=evidence)
        rows[name]["original_CAD_blockers"] = read(root/"dataset_states.json").get(name, {}).get("CAD_blockers", {})
    totals = dict(images=sum(row["expected_images"] for row in rows.values()),
        target_rows=sum(row["expected_targets"] for row in rows.values()),
        requested_instances=sum(row["expected_instances"] for row in rows.values()),
        available_CADs=sum(row["CADs"]["available"] for row in rows.values()),
        target_required_CADs=sum(row["CADs"]["target_required"] for row in rows.values()))
    if totals != TOTALS:
        raise ValueError(f"Final report is not the exact Classic Core target scope: {totals}")
    for field in ("completed_targets", "failed_targets", "blocked_targets", "valid_pose_predictions"):
        totals[field] = sum(row[field] for row in rows.values())
    models_path = root / "frozen_models.json"
    models = read(models_path)
    ready_path = root.parent / "runtime_ready.json"
    ready = read(ready_path)
    features = manifest["method_identity"]["base"]["features"]
    scientific = models["scientific_identity"]
    for model, field, ready_key in (("dino", "dino_checkpoint_sha256", "dinov2_giant"), ("gedi", "gedi_checkpoint_sha256", "gedi_3dmatch")):
        if scientific[model]["checkpoint_sha256"] != features[field] or ready["checkpoint_paths_sha256"][ready_key]["sha256"] != features[field]:
            raise ValueError("Model checkpoint identity differs between frozen run and trained-runtime evidence")
    evidence = {key:bound_file(path, helper) for key,path in dict(
        run_manifest=root/"run_manifest.json", protocol=root/"protocol.json", frozen_models=models_path,
        trained_runtime=ready_path, assets_manifest=assets_path, evaluation_runtime=runtime_path,
        primary_source_manifest=root/"source_snapshot/source_manifest.json",
        auxiliary_source_manifest=root/"auxiliary_source/evaluation_v1/source_manifest.json",
        sealed_auxiliary_helper=root/"auxiliary_source/evaluation_v1/complete_evaluation.py",
        reporting_script=Path(__file__),
        CAD_scope=root/"hb_target_scope_and_sample_audit_20261030.json",
        private_GT=root.parent/"validation/private_gt_availability.json",
        GeDi_failure=root.parent/"validation/gedi_lrf_20261030/GEDI_LRF_FAILURE_DIAGNOSIS_20261030.md").items()}
    return dict(schema="freezev2-final-report-v1", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status="finalized_with_recorded_inference_failures", all_target_passes_recorded=True,
        all_requested_poses_available=totals["valid_pose_predictions"] == totals["requested_instances"],
        method=protocol["method"], mask_configuration=protocol["mask_configuration"], task=protocol["task"],
        recipe=protocol["base"], totals=totals, datasets=rows, file_bindings=evidence,
        source_identity=dict(primary_files_verified=len(frozen["source_manifest"]["files"]),
            primary_manifest_sha256=frozen["source_manifest_sha256"], helper_sha256=helper.sha256(helper.__file__),
            toolkit_revision=actual_toolkit, models=scientific,
            DINO_scope="Selected source-file SHA256s and revision hint only; no verified whole-tree Git revision.",
            checkpoint_scope="Hash-bound recorded runtime/model evidence; large checkpoint bytes are not rehashed by this report."),
        claims=dict(our_generated_localization_predictions=True, authors_predictions_evaluated=False,
            published_performance_reproduced=False, CAD_identity_classification_measured=False,
            segmentation_AP_measured=False, pose_detection_AP_measured=False,
            GT_pose_role="official evaluation only; visib_gt_min=-1 official inst_count semantics",
            all_seven_aggregate=None, public_five_macro=None,
            private_GT="HB/ITODD local pose scores unavailable; no remote submission performed by this reporter"),
        preservation="Only NEW FINAL_REPORT.md and final_results.json are written. Frozen inference/evaluation sources, predictions, receipts and status remain unchanged.")


def markdown_link_path(path):
    """Display WSL-mounted evidence as native Windows links without rebinding it."""
    text = str(path).replace("\\", "/")
    if len(text) > 7 and text.startswith("/mnt/") and text[5].isalpha() and text[6] == "/":
        return text[5].upper() + ":/" + text[7:]
    return text


def markdown(result):
    lines = ["# Final saved-result report: independent FreeZeV2 base", "", f"Verified {result['created_utc']}.", "",
        f"Method `{result['method']}` with `{result['mask_configuration']}`. All7 official target passes are recorded; individual failed/blocked targets remain explicit. This does not reproduce the authors' published performance.", "",
        "| Dataset | Images | Target rows | Instances | Completed | Failed | Blocked | Saved poses | AR_VSD | AR_MSSD | AR_MSPD | AR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in result["datasets"].items():
        score = row["official_pose_metrics"]
        values = [f"{100*score[key]:.4f}%" if score else "private GT" for key in
                  ("bop19_average_recall_vsd", "bop19_average_recall_mssd", "bop19_average_recall_mspd", "bop19_average_recall")]
        lines.append(f"| {name} | {row['expected_images']} | {row['expected_targets']} | {row['expected_instances']} | {row['completed_targets']} | {row['failed_targets']} | {row['blocked_targets']} | {row['valid_pose_predictions']} | " + " | ".join(values) + " |")
    lines += ["", "Completed means the requested number of valid-format distinct poses was saved, not GT-correct poses. Failed means fewer saved poses; surviving partial poses remain in the CSV. Every requested instance stays in the unchanged official denominator. No dummy predictions or missing-instance removal are used.", "",
        "This is model-based localization with supplied official object IDs/counts, not unknown-CAD classification, segmentation AP or pose-detection AP. HB/ITODD test GT is private. No seven-dataset aggregate or five-dataset macro is reported.", "",
        "125 CADs are available;108 occur in the official targets. HB has16/16 required CAD feature caches; its17 unused inventory CADs are not pending work. T-LESS has27/30 required feature caches: original GeDi nonfinite outputs caused our strict finite-output guard to reject CAD27/28/30 caches. Author handling is unknown. Frozen16/33 inventory summaries are preserved.", "",
        "| Dataset | Required feature CADs | Available CADs | CSV timing images | Official CSV mean s | All-image active mean s | Completed onboarding s | Evaluator elapsed s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in result["datasets"].items():
        timing=row["timing"]; mean=timing["official_BOP_image_mean_s"]
        official=f"{mean:.6f}" if mean is not None else "unavailable"
        evaluation=row["evaluation"]; elapsed=evaluation.get("evaluation_runtime_s") if evaluation else None
        elapsed=f"{elapsed:.3f}" if isinstance(elapsed,(float,int)) else "unavailable"
        lines.append(f"| {name} | {len(row['CADs']['physical_completion_ids']['features'])}/{row['CADs']['target_required']} | {row['CADs']['available']} | {timing['CSV_images_with_predictions']}/{timing['official_target_images']} | {official} | {timing['all_attempted_image_mean_s']:.6f} | {row['onboarding']['completed_stage_active_runtime_s']:.3f} | {elapsed} |")
    lines += ["", "Official timing averages only images represented by a CSV pose. The all-image column includes each attempted image's saved active work. These exclude external precomputed segmentation, model loading, CAD onboarding and evaluation. Onboarding sums saved completed stages; checkpointed partial work is separate in JSON and does not include the entire failed GeDi tail. These are not full cold-start end-to-end runtimes.", "",
        "Source, model, runtime, recipe assumptions and all file hashes are in final_results.json. The primary19-file snapshot and sealed evaluation helper are verified. DINO provenance verifies selected files and a revision hint, not a complete Git tree. Checkpoint hashes are linked to recorded trained-runtime evidence; this reporting pass does not rehash multi-GB weights.", "",
        "Representative evidence (visual sanity checks do not establish full6D correctness):"]
    for name,row in result["datasets"].items():
        full,visual=row["evidence"]["full_audit"],row["evidence"]["visual_review"]
        lines.append(f"- {name}: [{Path(full['path']).name}]({markdown_link_path(full['path'])}); [{Path(visual['path']).name}]({markdown_link_path(visual['path'])}) ({row['evidence']['visually_reviewed_cases']} cases). Failure examples and per-status instance/pose counts are bound in JSON.")
    diagnosis=result["file_bindings"]["GeDi_failure"]
    lines += [f"- [Original GeDi LRF diagnosis]({markdown_link_path(diagnosis['path'])}): raw point-level nonfinite output and our whole-cache rejection are distinct.", "",
        "Recovery commands are for an unexpected process failure only, after verifying all original Windows/Linux owners have exited. A finished worker's old timestamp is normal. Deterministic T-LESS cache rejection is preserved evidence, not a reason to restart or silently repair this baseline.", "", "```powershell",
        "& 'D:\\Python\\python.exe' 'D:\\GitHub\\A-novel-Robotic-Vision-System-for-Object-Identification\\experiments\\freezev2_bop_20261029\\launch_suite.py' --config 'D:\\GitHub\\A-novel-Robotic-Vision-System-for-Object-Identification\\experiments\\freezev2_bop_20261029\\protocol.json'",
        "& 'D:\\Python\\python.exe' 'D:\\GitHub\\A-novel-Robotic-Vision-System-for-Object-Identification\\experiments\\freezev2_bop_20261029\\launch_evaluation.py' --config 'D:\\GitHub\\A-novel-Robotic-Vision-System-for-Object-Identification\\experiments\\freezev2_bop_20261029\\protocol.json'", "```", "",
        "No inference/evaluation was launched by this reporter. Frozen run reports and receipts remain unchanged."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate all prerequisites; write nothing")
    args=parser.parse_args(argv)
    root=Path(args.run_root).expanduser().resolve()
    try:
        helper,frozen=load_helper(root)
        result=build_report(root,helper,frozen)
        output=markdown(result)
        if not args.check_only:
            # Fail closed if bindings changed while preparing the two new files.
            for binding in result["file_bindings"].values():
                if helper.sha256(binding["path"]) != binding["sha256"]:
                    raise ValueError("Final evidence changed during reporting")
            for name,row in result["datasets"].items():
                bindings = [row["evidence"]["full_audit"], row["evidence"]["visual_review"]] + row["evidence"]["reviewed_artifact_bindings"]
                if row["evaluation"]:
                    bindings += [row["evaluation"]["receipt"], dict(path=row["evaluation"]["score_file"], sha256=row["evaluation"]["score_sha256"])]
                for binding in bindings:
                    if helper.sha256(binding["path"]) != binding["sha256"]:
                        raise ValueError("Scoped audit/review changed during reporting")
                if helper.sha256(row["csv_path"]) != row["csv_sha256"]:
                    raise ValueError(f"{name}: CSV changed during reporting")
            for path,value in ((root/"final_results.json", json.dumps(result,indent=2,allow_nan=False)+"\n"), (root/"FINAL_REPORT.md", output)):
                if path.exists():
                    raise ValueError(f"Refusing to overwrite existing final evidence: {path}; inspect it explicitly")
            with (root/"final_results.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
            with (root/"FINAL_REPORT.md").open("x", encoding="utf-8") as stream:
                stream.write(output)
        print(json.dumps(dict(status="verified_check_only" if args.check_only else "finalized", totals=result["totals"])))
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status="final_report_not_available", reason=f"{type(exc).__name__}: {exc}")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
