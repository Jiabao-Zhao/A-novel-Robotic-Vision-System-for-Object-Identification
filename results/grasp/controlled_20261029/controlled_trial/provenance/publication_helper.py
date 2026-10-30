"""Post-launch publication of the complete controlled policy matrix.

This adapter only copies/encodes saved evidence. It never imports a geometry
engine, proposes contacts, fits fingers, ranks candidates or chooses a seed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .publish import MAX_FILE_BYTES, _copy, _display, _relative, _sha

STRATEGIES = ("random", "task_informed")
MODES = ("pad", "finger", "combined")
ROOT_FILES = (
    "launch_manifest.json", "contact_calibration.json", "summary.json", "results.json",
    "frozen_angle_sensitivity.json", "run_verification.json", "REPORT.md",
    "comparison.json", "per_part_comparison.csv", "independent_verification.json",
    "verification.json", "detailed_surface_audit.json", "detailed_surface_audit.md",
    "contact_surface_audit.json", "contact_surface_audit.md", "source_provenance.json",
)
LARGE_ROOT_JSON = {"results.json", "frozen_angle_sensitivity.json", "detailed_surface_audit.json"}
GALLERY_FILES = {"index.html", "gallery_manifest.json", "qa_receipt.json", "render_receipt.json",
                 "controlled_render_receipt.json", "finger_contact_render_receipt.json"}


def plan_controlled_run(source: str | Path) -> dict:
    source = Path(source).resolve(strict=True)
    for name in ("launch_manifest.json", "summary.json", "results.json", "run_verification.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    all_files = [p for p in source.rglob("*") if p.is_file()]
    chosen = {source / name for name in ROOT_FILES if (source / name).is_file()}
    for strategy in STRATEGIES:
        for name in ("sampling.json.gz", "candidate_diagnostics.json.gz"):
            chosen.update(source.glob(f"banks/{strategy}/seed_*/N0[0-7]/*/{name}"))
        for mode in MODES:
            chosen.update(source.glob(f"cases/{strategy}/seed_*/{mode}/N0[0-7]/*/selection.json"))
    for folder in ("task_inputs", "part_receipts"):
        chosen.update(source.glob(f"{folder}/*.json"))
    for name in ("results.json", "summary.json"):
        if (source / "primary" / name).is_file():
            chosen.add(source / "primary" / name)
    for p in (source / "primary/gallery").glob("*"):
        if p.is_file() and (p.name in GALLERY_FILES or
                           (p.suffix == ".png" and p.stem[:3] in {f"N{i:02d}" for i in range(8)})):
            chosen.add(p)
    files = []
    for p in sorted(chosen):
        if not p.resolve(strict=True).is_relative_to(source):
            raise ValueError(f"Saved evidence escapes run: {p}")
        relative = p.relative_to(source)
        compressed = relative.name == "selection.json" or (
            relative.suffix == ".json" and relative.name in LARGE_ROOT_JSON)
        files.append({"source": str(p), "source_relative": relative.as_posix(),
                      "stored_relative": relative.as_posix() + (".gz" if compressed else ""),
                      "encoding": "gzip" if compressed else "identity",
                      "source_bytes": p.stat().st_size,
                      "source_already_gzip": relative.suffix == ".gz"})
    return {"source_directory": _display(source), "tree_files": len(all_files),
            "tree_bytes": sum(p.stat().st_size for p in all_files),
            "allowed_files": len(files), "allowed_source_bytes": sum(r["source_bytes"] for r in files),
            "files": files,
            "excluded_files": sorted(p.relative_to(source).as_posix() for p in all_files if p not in chosen)}


def _matrix(source: Path, *, require_complete: bool) -> tuple[list, dict, list]:
    launch = json.loads((source / "launch_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a list of saved policy-placement rows")
    keys = {(r["sampler_strategy"], r["seed_index"], r["contact_mode"], r["scene"], r["part"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate policy-placement rows")
    expected = {(strategy, seed_index, mode, f"N{scene:02d}", part)
                for strategy in launch["strategies"] for seed_index in range(len(launch["seeds"]))
                for mode in launch["modes"] for scene in range(8) for part in launch["parts"]}
    if require_complete and keys != expected:
        raise ValueError(f"Incomplete policy matrix: {len(keys)} saved, {len(expected)} expected")
    bank_paths = set()
    decisions = []
    for r in rows:
        if r["sampler_strategy"] not in STRATEGIES or r["contact_mode"] not in MODES:
            raise ValueError("Unknown saved strategy or policy")
        selection = r["selection"]
        if not 0 <= selection["valid_count"] <= selection["proposed_count"] <= 100:
            raise ValueError("Saved policy exceeds its declared candidate budget")
        case = Path("cases") / r["sampler_strategy"] / f"seed_{r['seed_index']}" / r["contact_mode"] / r["scene"] / r["part"] / "selection.json"
        bank = _relative(r["diagnostic_bank"])
        sampling = _relative(selection["sampling_reference"])
        for relative in (case, bank, sampling):
            if not (source / relative).is_file():
                raise FileNotFoundError(source / relative)
        bank_paths.add(bank.as_posix())
        grasp = selection.get("selected_grasp") or {}
        criteria = grasp.get("criteria", {})
        if grasp and grasp.get("contact_mode") not in ("pad", "finger"):
            raise ValueError("Selected grasp must retain its actual contact mode")
        decisions.append({"strategy": r["sampler_strategy"], "seed_index": r["seed_index"],
                          "seed": r["seed"], "policy": r["contact_mode"], "scene": r["scene"], "part": r["part"],
                          "status": selection["status"], "proposed": selection["proposed_count"],
                          "valid": selection["valid_count"], "selected_candidate_id": grasp.get("candidate_id", ""),
                          "selected_contact_mode": grasp.get("contact_mode", ""),
                          **{k: criteria.get(k, "") for k in ("r_tau", "r_surface", "theta_down_rad")},
                          **{f"{stage}_passing_candidates": selection["stage_counts"][stage].get("pass", 0)
                             for stage in ("contact_valid", "pickup_valid", "final_goal_valid", "joint_valid")},
                          "selection": case.as_posix() + ".gz", "diagnostic_bank": bank.as_posix(),
                          "diagnostic_mode": r["diagnostic_mode"], "sampling": sampling.as_posix()})
    return rows, {"expected_policy_rows": len(expected), "saved_policy_rows": len(rows),
                  "matrix_complete": keys == expected, "diagnostic_banks": len(bank_paths),
                  "per_policy_candidate_cap": 100,
                  "combined_assignment": launch["combined_mode_assignment"],
                  "benchmark_work": launch["underlying_comparison_evaluations"]}, decisions


def publish_controlled(source: str | Path, destination: str | Path, *,
                       supplements=(), require_complete: bool = True) -> dict:
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Publication and source trees must be separate")
    plan = plan_controlled_run(source)  # full inventory precedes any write
    rows, matrix, decisions = _matrix(source, require_complete=require_complete)
    extra = [(Path(src).resolve(strict=True), _relative(dst)) for src, dst in supplements]
    names = [r["stored_relative"] for r in plan["files"]] + [dst.as_posix() for _, dst in extra]
    reserved = {"publication_manifest.json", "README.md", "decisions.csv", "provenance/publication_helper.py"}
    if len(names) != len(set(names)) or reserved.intersection(names):
        raise ValueError("Publication paths collide")
    destination.mkdir(parents=True)
    files = []
    for row in plan["files"]:
        entry = _copy(Path(row["source"]), destination / row["stored_relative"],
                      encoding=row["encoding"], max_file_bytes=MAX_FILE_BYTES)
        if row["source_already_gzip"]:
            decoded_sha, decoded_bytes = _sha(destination / row["stored_relative"], decompress=True)
            entry.update(source_already_gzip=True, decoded_source_sha256=decoded_sha,
                         decoded_source_bytes=decoded_bytes,
                         source_gzip_bytes_preserved=True)
        files.append({"stored_relative": row["stored_relative"], **entry})
    for src, relative in [*extra, (Path(__file__), Path("provenance/publication_helper.py"))]:
        entry = _copy(src, destination / relative, encoding="identity", max_file_bytes=MAX_FILE_BYTES)
        files.append({"stored_relative": relative.as_posix(), **entry, "explicit_supplement": True})
    with (destination / "decisions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    manifest = {"schema": "complete-controlled-grasp-publication-v1", "source_directory": _display(source),
                "postprocessing_added_after_launch": True, "matrix": matrix,
                "inventory_before_copy": {k: v for k, v in plan.items() if k != "files"},
                "files": files, "stored_evidence_bytes": sum(r["stored_bytes"] for r in files),
                "source_evidence_bytes": sum(r["source_bytes"] for r in files),
                "source_summary_groups": summary["groups"],
                "underlying_mode_candidate_evaluations": summary["underlying_mode_candidate_evaluations"],
                "logical_policy_candidate_evaluations": summary["logical_policy_candidate_evaluations"],
                "publication_helper_sha256": _sha(Path(__file__))[0],
                "frozen_copy_helper_sha256": _sha(Path(__file__).with_name("publish.py"))[0],
                "derived_files": {"decisions.csv": _sha(destination / "decisions.csv")[0]},
                "historical_source_modified": False, "grasp_generation_performed": False,
                "policy_or_seed_selection_performed": False, "git_stage_commit_push_performed": False,
                "physical_success_measured": False}
    (destination / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(_readme(matrix), encoding="utf-8")
    return manifest


def _readme(matrix: dict) -> str:
    return f"""# Complete controlled grasp evidence

This preserves **{matrix['saved_policy_rows']} policy-placement records** across the saved NIST placements, seeds, two proposal strategies and three contact policies. Read the [saved report](REPORT.md), [summary](summary.json), [paired comparisons](comparison.json), [per-part table](per_part_comparison.csv), [compact complete decisions](decisions.csv), and [exact publication manifest](publication_manifest.json). Modeled geometry availability does not establish pickup or assembly success. VLM selection is excluded; pickup and assembly execution remain paused.

The [launch source lock](launch_manifest.json), [run verification](run_verification.json), and [saved contact calibration](contact_calibration.json) retain their original bytes. Copied report links to uncompressed result/angle JSON refer to original local filenames; the working published gzip links are provided below.

The [primary gallery](primary/gallery/index.html) and [primary summary](primary/summary.json) show task-informed combined mode with seed 0 declared before results. No better seed or different policy is substituted for failed placements. Views show selected contacts on the CAD and optional pickup side views, with no final assembly views.

Every policy has at most 100 candidates. The benchmark evaluates pad and finger versions of one shared raw bank, up to 200 underlying candidate evaluations. Combined mode uses its predefined 50:50 assignment, not the union of two validated 100-candidate sets. A saved diagnostic bank contains `pad`, `finger`, and `combined_assignment`; reconstruct combined records by looking up each assigned candidate ID in the specified pad/finger list. These comparisons reuse placements and CAD banks, so logical policy rows and candidate evaluations are not independent objects or physical trials. Independent policy wall times were not measured.

[Complete root results](results.json.gz), [primary results](primary/results.json.gz), and [frozen-angle records](frozen_angle_sensitivity.json.gz) are exact original JSON bytes stored as gzip. Each decision-table row links its complete policy selection, shared full diagnostic bank and sampling metadata. Existing bank gzip files retain their exact compressed bytes; the manifest additionally records their decoded JSON SHA256 and byte count. Copied reports keep original text and source links; use this index for the published compressed filenames.

The fixed-pose sensitivity changes validation tolerance only; hand labels, calibration, fitted poses and proposal settings stay fixed. Contact, pickup endpoint, final endpoint/task and combined geometry counts are separate. Source visual-versus-collision contact discrepancies are separate model-consistency evidence, not physical finger calibration.

Read the [independent saved-result audit](independent_audit/verification.json) and [primary finger-contact source-surface report](independent_audit/primary_detailed_source_compatibility/detailed_surface_audit.md), with its [full distance records](independent_audit/primary_detailed_source_compatibility/detailed_surface_audit.json) and [upstream subset provenance](independent_audit/task_informed_seed0_combined_finger_view/upstream_provenance.json). These are post-launch audits of saved outcomes. The derived audit view can be reconstructed from the complete published upstream banks and does not add a grasp search or certify hardware contact.

The publication helper and other explicitly included postprocessing sources were added after benchmark launch. Their presence does not change the launch source lock. The exact publisher source is saved under `provenance/publication_helper.py`. No historical source/output JSON is modified, and publication performs no generation, fitting, ranking, robot action, git stage, commit or push.

Verify with `python -m experiments.cad_grasp_controlled_20261029.publish_controlled verify <this-directory>`. Python `gzip.open(path, 'rt', encoding='utf-8')` and `json.load` can read each compressed JSON directly.
"""


def verify_controlled(destination: str | Path) -> dict:
    destination = Path(destination).resolve(strict=True)
    manifest = json.loads((destination / "publication_manifest.json").read_text(encoding="utf-8"))
    decoded_banks = 0
    for row in manifest["files"]:
        path = destination / _relative(row["stored_relative"])
        if not path.resolve().is_relative_to(destination):
            raise ValueError("Publication file escapes destination")
        if _sha(path) != (row["stored_sha256"], row["stored_bytes"]):
            raise ValueError(f"Stored file changed: {row['stored_relative']}")
        if _sha(path, decompress=row["encoding"] == "gzip") != (row["source_sha256"], row["source_bytes"]):
            raise ValueError(f"Original source bytes differ: {row['stored_relative']}")
        if row.get("source_already_gzip"):
            if _sha(path, decompress=True) != (row["decoded_source_sha256"], row["decoded_source_bytes"]):
                raise ValueError(f"Decoded bank differs: {row['stored_relative']}")
            decoded_banks += 1
    for name, expected in manifest["derived_files"].items():
        if _sha(destination / _relative(name))[0] != expected:
            raise ValueError(f"Derived table changed: {name}")
    return {"verified": True, "source_files": len(manifest["files"]),
            "original_gzip_files_decoded_and_verified": decoded_banks,
            "policy_rows": manifest["matrix"]["saved_policy_rows"]}


def update_publication_index(destination: str | Path) -> Path:
    """Append this saved comparison without rewriting the historical index rows."""
    destination = Path(destination).resolve(strict=True)
    manifest = json.loads((destination / "publication_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    index = destination.parent / "README.md"
    marker = "<!-- controlled-trial-publication -->"
    previous = index.read_text(encoding="utf-8") if index.is_file() else "# October 29 grasp evidence\n"
    previous = previous.split(marker, 1)[0].rstrip()
    name = destination.name
    lines = ["", marker, "", "## Complete controlled comparison", "",
             f"The [complete controlled evidence]({name}/README.md) preserves {manifest['matrix']['saved_policy_rows']} policy-placement records, including every failure, all shared banks, the fixed primary gallery and exact publication hashes. These counts describe modeled geometry availability; VLM selection is excluded and no pickup or assembly was executed.", "",
             "| Proposal strategy | Contact policy | Selected placements over three seeds /288 | Valid modeled candidates |",
             "|---|---|---:|---:|"]
    for row in summary["groups"]:
        lines.append(f"| {row['strategy']} | {row['contact_mode']} | {row['selected']} | {row['valid']} |")
    lines += ["", f"Read the [paired report]({name}/REPORT.md), [all policy decisions]({name}/decisions.csv), [per-part comparison]({name}/per_part_comparison.csv), [primary selected-grasp gallery]({name}/primary/gallery/index.html), and [manifest]({name}/publication_manifest.json). The displayed branch is task-informed combined mode, seed 0 fixed before results, rather than a selected best seed.", "",
              "The 288 rows per arm repeat the same 96 placements under three seeds. The benchmark evaluates two contact versions of each shared raw bank; its total work differs from each policy's 100-candidate budget. Its wall time covers the complete shared comparison and frozen-angle sweep, not an independently measured runtime or speedup for one policy. Detailed visual/collision surface mismatch remains a separate model limitation.", ""]
    index.write_text(previous + "\n" + "\n".join(lines), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "publish", "verify"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan_controlled_run(args.source)
    elif args.command == "verify":
        result = verify_controlled(args.source)
    else:
        if args.destination is None:
            parser.error("publish requires destination")
        result = publish_controlled(args.source, args.destination)
        result = {k: v for k, v in result.items() if k != "files"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
