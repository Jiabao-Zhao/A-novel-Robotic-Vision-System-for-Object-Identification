"""Evaluation-only paired target attribution from saved official BOP matches.

Reads no GT pose files, invokes no evaluator/inference and imports no model.
Every selected target, including zero-prediction failures, must be present in
all120 official threshold match files for each of the three arms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

ARMS = ("baseline", "repair224", "repair476")
METRIC_COUNTS = {"vsd": 100, "mssd": 10, "mspd": 10}
METRIC_KEYS = {name: "bop19_average_recall_" + name for name in METRIC_COUNTS}
AR_KEY = "bop19_average_recall"
TOOLKIT = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"


def local_path(value) -> Path:
    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:].replace("\\", "/")
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    return Path(text).expanduser().resolve()


def fs(path) -> Path:
    text = str(Path(path).absolute())
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        text = "\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text
    return Path(text)


def read_json(path):
    return json.loads(fs(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(fs(path).read_bytes()).hexdigest()


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid integer {label}")
    return value


def target_key(row):
    return tuple(_integer(row.get(n), n, 1 if n == "obj_id" else 0)
                 for n in ("scene_id", "im_id", "obj_id"))


def target_counts(targets):
    if not isinstance(targets, list) or not targets:
        raise ValueError("A nonempty fixed target list is required")
    result = {}
    for row in targets:
        key = target_key(row)
        if key in result:
            raise ValueError("Duplicate fixed target key")
        result[key] = _integer(row.get("inst_count"), "inst_count", 1)
    return result


def _close(actual, expected, label, tolerance=1e-12):
    if (isinstance(actual, bool) or isinstance(expected, bool)
            or not math.isfinite(float(actual)) or not math.isfinite(float(expected))
            or not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0)):
        raise ValueError(f"Mismatch in {label}: {actual!r} versus {expected!r}")


def count_matches(matches, counts):
    """Return successful official matches for every selected key, including0."""
    if not isinstance(matches, list):
        raise ValueError("Official matches must be a JSON list")
    valid_counts, successes = Counter(), Counter()
    seen_gt, seen_estimates = set(), set()
    for row in matches:
        if not isinstance(row, dict) or not isinstance(row.get("valid"), bool):
            raise ValueError("Each official match must have a Boolean valid flag")
        if not row["valid"]:
            continue
        key = target_key(row)
        if key not in counts:
            raise ValueError("Unknown valid GT target in official matches")
        gt = _integer(row.get("gt_id"), "gt_id")
        if (key, gt) in seen_gt:
            raise ValueError("Duplicate valid GT match row")
        seen_gt.add((key, gt))
        valid_counts[key] += 1
        est = _integer(row.get("est_id"), "est_id", -1)
        if est >= 0:
            if (key, est) in seen_estimates:
                raise ValueError("One estimate matched multiple valid GT rows")
            seen_estimates.add((key, est))
            successes[key] += 1
    if dict(valid_counts) != counts:
        raise ValueError(f"Official valid GT counts differ from fixed inst_count: {dict(valid_counts)}")
    return {key: successes[key] for key in counts}


def analyze_matches(score_root, targets, official_metrics):
    """Audit120 saved match/score pairs and compute per-target BOP recalls."""
    counts = target_counts(targets)
    total_instances = sum(counts.values())
    root = fs(score_root)
    groups, folder_counts = defaultdict(list), Counter()
    for path in sorted(root.rglob("matches_th*.json")):
        relative = path.relative_to(root)
        metric = path.parent.name.split("_", 1)[0].removeprefix("error=")
        if metric not in METRIC_COUNTS or len(relative.parts) != 2:
            raise ValueError("Unexpected official match-file location")
        if not path.name.endswith("_min-visib=-1.000.json"):
            raise ValueError("Expected BOP19 localization visib_gt_min=-1 matches")
        groups[metric].append(path)
        folder_counts[(metric, path.parent.name)] += 1
    if {m: len(groups[m]) for m in METRIC_COUNTS} != METRIC_COUNTS:
        raise ValueError("Expected exactly100 VSD,10 MSSD and10 MSPD match files")
    for metric, expected_folders in (("vsd", 10), ("mssd", 1), ("mspd", 1)):
        selected = [n for (m, _), n in folder_counts.items() if m == metric]
        if len(selected) != expected_folders or any(n != 10 for n in selected):
            raise ValueError("Unexpected official error/threshold grid")
    sums = {key: Counter() for key in counts}
    trial_counts, bindings = {}, []
    computed = {}
    for metric, number in METRIC_COUNTS.items():
        for path in groups[metric]:
            matches = read_json(path)
            successes = count_matches(matches, counts)
            scores_path = path.with_name(path.name.replace("matches_", "scores_", 1))
            scores = read_json(scores_path)
            tp = sum(successes.values())
            if scores.get("targets_count") != total_instances or scores.get("tp_count") != tp:
                raise ValueError("Official per-threshold numerator/denominator differs from matches")
            if scores.get("gt_count") != len(matches):
                raise ValueError("Official GT-row count differs from saved match list")
            _close(scores["recall"], tp / total_instances, "threshold recall")
            for key, value in successes.items():
                sums[key][metric] += value
            bindings.append({"matches": path.relative_to(root).as_posix(),
                             "matches_sha256": sha256(path),
                             "scores": scores_path.relative_to(root).as_posix(),
                             "scores_sha256": sha256(scores_path),
                             "tp_count": tp, "targets_count": total_instances})
        tp = sum(value[metric] for value in sums.values())
        denominator = number * total_instances
        value = tp / denominator
        _close(value, official_metrics[METRIC_KEYS[metric]], f"aggregate {metric}")
        computed[METRIC_KEYS[metric]] = value
        trial_counts[metric] = {"true_positives_across_thresholds": tp,
                                "threshold_instance_trials": denominator,
                                "match_files": number, "recall": value}
    computed[AR_KEY] = sum(computed[METRIC_KEYS[m]] for m in METRIC_COUNTS) / 3
    _close(computed[AR_KEY], official_metrics[AR_KEY], "aggregate AR")
    per_target = []
    for key, count in sorted(counts.items()):
        metrics = {m.upper(): sums[key][m] / (count * METRIC_COUNTS[m]) for m in METRIC_COUNTS}
        metrics["AR"] = sum(metrics.values()) / 3
        per_target.append({"scene_id": key[0], "im_id": key[1], "obj_id": key[2],
                           "inst_count": count, **metrics,
                           "successful_threshold_matches": dict(sums[key]),
                           "threshold_files": dict(METRIC_COUNTS)})
    # All actual cohort rows request1 instance; the ordinary row mean must agree.
    if all(count == 1 for count in counts.values()):
        for metric, official_key in [(m.upper(), METRIC_KEYS[m]) for m in METRIC_COUNTS] + [("AR", AR_KEY)]:
            _close(sum(t[metric] for t in per_target) / len(per_target),
                   computed[official_key], f"mean per-target {metric}")
    return {"per_target": per_target, "computed_metrics": computed,
            "metric_numerators_denominators": trial_counts,
            "threshold_files": sorted(bindings, key=lambda row: row["matches"])}


def read_arm(evaluation, targets, cohort_sha):
    receipt_path = Path(evaluation) / "evaluation_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete":
        raise ValueError("All three official evaluation receipts must be complete")
    if receipt.get("source_pins", {}).get("cohort_sha256") != cohort_sha:
        raise ValueError("Evaluation receipt uses a different fixed cohort")
    if receipt.get("source_pins", {}).get("toolkit_revision") != TOOLKIT:
        raise ValueError("Evaluation receipt uses an unpinned official toolkit")
    score = local_path(receipt["score_file"])
    if sha256(score) != receipt.get("score_sha256"):
        raise ValueError("Official aggregate score file changed")
    values = read_json(score)
    for name in [AR_KEY, *METRIC_KEYS.values()]:
        _close(values[name], receipt["metrics"][name], "receipt " + name)
    analyzed = analyze_matches(score.parent, targets, values)
    return {**analyzed, "official_metrics": {name: values[name] for name in [AR_KEY, *METRIC_KEYS.values()]},
            "evaluation_receipt_sha256": sha256(receipt_path),
            "aggregate_score_sha256": sha256(score),
            "saved_prediction_count": receipt.get("saved_predictions")}


def diagnose(output, cohort_path):
    """Read all completed official artifacts before creating either new report."""
    cohort = read_json(cohort_path)
    targets = cohort["targets"]
    counts = target_counts(targets)
    if len(counts) != 7 or any(value != 1 for value in counts.values()):
        raise ValueError("This paired diagnostic is fixed to seven single-instance target rows")
    cohort_sha = sha256(cohort_path)
    arms = {arm: read_arm(Path(output) / arm / "evaluation", targets, cohort_sha) for arm in ARMS}
    grid = [row["matches"] for row in arms["baseline"]["threshold_files"]]
    if any([row["matches"] for row in arms[arm]["threshold_files"]] != grid for arm in ARMS):
        raise ValueError("The three arms use different official threshold grids")
    case_names = {target_key(c["target"]): c["case_id"] for c in cohort.get("cases", [])}
    by_arm = {arm: {target_key(row): row for row in arms[arm]["per_target"]} for arm in ARMS}
    paired = []
    for key, count in sorted(counts.items()):
        row = {"scene_id": key[0], "im_id": key[1], "obj_id": key[2], "inst_count": count,
               "case_id": case_names.get(key), "arms": {a: by_arm[a][key] for a in ARMS}}
        row["AR_delta_percentage_points"] = {
            "repair224_vs_baseline": 100 * (by_arm["repair224"][key]["AR"] - by_arm["baseline"][key]["AR"]),
            "repair476_vs_repair224": 100 * (by_arm["repair476"][key]["AR"] - by_arm["repair224"][key]["AR"])}
        paired.append(row)
    result = {"schema": "freezev2-seven-target-paired-official-matches-v1",
              "scope": "Seven preselected official TLESS object-image rows only; every selected instance remains in every arm's denominator.",
              "postprocessing_role": "Saved official threshold matches only; no GT poses read and no inference, evaluation, parameter tuning or model loading invoked.",
              "interpretation": "Per-target recalls average official threshold success indicators. Positive AR is partial threshold credit, not proof of correctness at every threshold or full-dataset performance.",
              "cohort_sha256": cohort_sha, "helper_sha256": sha256(Path(__file__)),
              "target_rows": 7, "requested_instances": 7,
              "official_match_files_per_arm": dict(METRIC_COUNTS), "arms": arms, "paired_targets": paired}
    return result


def render_markdown(result):
    lines = ["# Paired metrics on the fixed seven-target cohort", "",
             "These are per-target BOP19 localization threshold recalls from saved official matches. All seven requested instances, including missing predictions and failures, remain in the denominator. This is a diagnostic subset, not the full T-LESS benchmark.", "",
             "| Scene/image/CAD | Baseline AR | Repair224 AR | Repair476 AR | 224âˆ’baseline pp | 476âˆ’224 pp |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in result["paired_targets"]:
        values = row["arms"]; delta = row["AR_delta_percentage_points"]
        lines.append(f"| {row['scene_id']}/{row['im_id']}/{row['obj_id']} | "
                     + " | ".join(f"{100*values[a]['AR']:.6f}%" for a in ARMS)
                     + f" | {delta['repair224_vs_baseline']:+.6f} | {delta['repair476_vs_repair224']:+.6f} |")
    lines += ["", "| Scene/image/CAD | Arm | VSD recall | MSSD recall | MSPD recall | AR |",
              "|---|---|---:|---:|---:|---:|"]
    for row in result["paired_targets"]:
        for arm in ARMS:
            values = row["arms"][arm]
            lines.append(f"| {row['scene_id']}/{row['im_id']}/{row['obj_id']} | {arm} | "
                         + " | ".join(f"{100*values[m]:.6f}%" for m in ("VSD", "MSSD", "MSPD", "AR")) + " |")
    lines += ["", "Each arm was checked against100 VSD,10 MSSD and10 MSPD official match/score pairs. Every per-threshold true-positive count, target denominator and recall agrees with its official score; metric aggregates and the mean of the seven per-target ARs agree with official scores_bop19.json to absolute tolerance1e-12.", "",
              "A zero-to-positive AR change means some official threshold matches were gained. It does not establish success at every threshold. The numerical GeDi repair and CAD476 preprocessing remain separate comparisons. No GT pose files were read by this report.", ""]
    return "\n".join(lines)


def write_reports(output, result):
    output = Path(output)
    contents = {"paired_target_metrics.json": (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
                "PAIRED_TARGET_METRICS.md": render_markdown(result).encode()}
    # Refuse a different existing report rather than mix a new analysis version.
    for name, content in contents.items():
        path = fs(output / name)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Existing paired report differs; preserve it and use a versioned output")
    for name, content in contents.items():
        path = fs(output / name)
        if not path.exists():
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--cohort", type=Path, default=Path(__file__).with_name("cohort.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = local_path(args.output or read_json(args.config)["output"])
    root = Path(__file__).resolve().parents[2]
    baseline = local_path(root / "outputs/freezev2_bop_20261029/classic_core_base_v2")
    if output == baseline or output.is_relative_to(baseline):
        raise ValueError("Paired reporting must not write into the frozen baseline")
    result = diagnose(output, args.cohort)
    write_reports(output, result)
    print(json.dumps({"status": "verified", "target_rows": 7,
                      "reports": [str(output / n) for n in ("paired_target_metrics.json", "PAIRED_TARGET_METRICS.md")]}))


if __name__ == "__main__":
    main()
