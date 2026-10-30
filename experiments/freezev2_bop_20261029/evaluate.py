"""Explicit, evaluation-only official BOP19 pose-metric entry point.

Inference never imports this entry point or receives GT paths from it. Dry-run
checks paths/CSV and prints the exact command; only --execute launches metrics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from freezev2_repro.evaluation import (
    build_evaluation_command, preflight_official_evaluation,
    run_official_evaluation, validate_bop_csv,
)


def _probe_evaluator_environment(environment, renderer_type):
    """Probe the actual metric subprocess environment, without opening GT."""
    code = ("import importlib,json,numpy;from bop_toolkit_lib import inout,pose_error;"
            "importlib.import_module('bop_toolkit_lib.rendering.renderer_" +
            {"python": "py", "vispy": "vispy", "cpp": "cpp"}[renderer_type] +
            "');print(json.dumps({'status':'importable','numpy':numpy.__version__}))")
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        return dict(status="blocked", reason=result.stderr.strip(),
                    action="Use a separate compatible evaluator environment with these dependencies; do not modify a shared environment or mix native Python with cached Linux binaries")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _local_path(value):
    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:].replace("\\", "/")
    return Path(text).expanduser().resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--dataset-view", required=True,
                        help="Explicit evaluation-only BOP root containing <dataset>/models_eval and GT/depth files")
    parser.add_argument("--targets-filename", default="test_targets_bop19.json")
    parser.add_argument("--evaluation-root", help="Separate output directory for official metric/error files")
    parser.add_argument("--renderer-type", choices=("python", "vispy", "cpp"), default="python")
    parser.add_argument("--num-workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print exact command/preflight only (default)")
    mode.add_argument("--execute", action="store_true", help="Explicitly run official metrics after all input checks pass")
    args = parser.parse_args(argv)
    try:
        config = json.loads(_local_path(args.config).read_text(encoding="utf-8-sig"))
        paths = config["paths"]
        toolkit, results = _local_path(paths["bop_toolkit"]), _local_path(args.results_file)
        support_paths = [_local_path(paths["bop_eval_deps"])] if paths.get("bop_eval_deps") else []
        evaluation_root = (_local_path(args.evaluation_root) if args.evaluation_root
                           else _local_path(paths["output_root"]) / "evaluation")
        kwargs = dict(toolkit_root=toolkit, results_file=results,
                      datasets_root=_local_path(args.dataset_view), evaluation_root=evaluation_root,
                      targets_filename=args.targets_filename, renderer_type=args.renderer_type,
                      num_workers=args.num_workers)
        command, environment = build_evaluation_command(**kwargs)
        report = dict(mode="execute" if args.execute else "dry_run", command=command,
                      command_interpreter=sys.executable, task="model-based_6D_localization",
                      metrics=["VSD", "MSSD", "MSPD"],
                      format=validate_bop_csv(results, toolkit, support_paths),
                      preflight=preflight_official_evaluation(toolkit_root=toolkit,
                          results_file=results, datasets_root=kwargs["datasets_root"],
                          targets_filename=args.targets_filename, dataset=config["dataset"],
                          extra_python_paths=support_paths),
                      claim="Format/preflight checks do not reproduce a benchmark score")
        report["evaluator_environment"] = _probe_evaluator_environment(environment, args.renderer_type)
        if (report["preflight"]["status"] != "ready"
                or report["evaluator_environment"]["status"] != "importable"):
            report["status"] = "blocked_evaluation_inputs"
            print(json.dumps(report, indent=2))
            return 2
        if not args.execute:
            report["status"] = "ready_dry_run_only"
            print(json.dumps(report, indent=2))
            return 0
        # The full evaluator and its subprocesses use the selected environment.
        # Extra installed dependencies must already be importable in that process;
        # no packages or shared environments are modified by this command.
        report["evaluation"] = run_official_evaluation(
            diagnostics_log=evaluation_root / "official_evaluator.log", **kwargs)
        report["status"] = "official_evaluator_completed"
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError, KeyError) as exc:
        print(json.dumps(dict(status="blocked_or_failed", type=type(exc).__name__, reason=str(exc),
                              claim="No benchmark score or published-performance reproduction claim"), indent=2))
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
