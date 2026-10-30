"""Copy an explicit grasp-evidence allowlist; never run grasp generation.

Large JSON files are stored as deterministic gzip streams and verified against
the exact original bytes. Original study files are never edited or reformatted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 1024 * 1024
MAX_FILE_BYTES = 95 * 1024 * 1024
ROOT_FILES = (
    "summary.json", "results.json", "report.md", "verification.json",
    "independent_verification.json", "run_verification.json", "derivation.json",
    "placement_counts.csv", "selected_contacts.csv", "bank_identity.json",
    "contact_calibration.json", "contact_report.json", "failure_analysis.md",
    "launch_manifest.json", "stage_counts.json", "stage_counts.csv",
    "stage_report.md", "comparison.json", "candidate_identity.json",
)
GALLERY_JSON = {
    "gallery_manifest.json", "qa_receipt.json", "render_receipt.json",
    "finger_contact_render_receipt.json",
}


def _sha(path: Path, *, decompress: bool = False) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    opener = gzip.open if decompress else open
    with opener(path, "rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _relative(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Expected a safe bundle-relative path: {value}")
    return path


def plan_run(source: str | Path) -> dict:
    """Inventory the tree, then enumerate only the allowed evidence files."""
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Source must be a saved run directory")
    for name in ("summary.json", "results.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    all_files = [p for p in source.rglob("*") if p.is_file()]
    chosen = {source / name for name in ROOT_FILES if (source / name).is_file()}
    for p in source.glob("cases/*/*/*.json"):
        if p.name in {"selection.json", "candidate_diagnostics.json"}:
            chosen.add(p)
    chosen.update(source.glob("task_inputs/*.json"))
    for p in (source / "gallery").glob("*"):
        if p.is_file() and (
            p.name == "index.html" or p.name in GALLERY_JSON
            or (p.suffix == ".png" and p.stem[:3] in {f"N{i:02d}" for i in range(8)})
        ):
            chosen.add(p)
    files = []
    for path in sorted(chosen):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source):
            raise ValueError(f"Allowlisted source escapes its run: {path}")
        relative = path.relative_to(source)
        compressed = relative.name in {"results.json", "selection.json", "candidate_diagnostics.json"}
        files.append({"source": str(resolved), "source_relative": relative.as_posix(),
                      "stored_relative": relative.as_posix() + (".gz" if compressed else ""),
                      "encoding": "gzip" if compressed else "identity",
                      "source_bytes": path.stat().st_size})
    return {
        "source_directory": _display(source),
        "tree_file_count": len(all_files),
        "tree_bytes": sum(p.stat().st_size for p in all_files),
        "allowed_file_count": len(files),
        "allowed_source_bytes": sum(row["source_bytes"] for row in files),
        "files": files,
        "excluded_files": sorted(p.relative_to(source).as_posix() for p in all_files if p not in chosen),
    }


def _copy(source: Path, destination: Path, *, encoding: str, max_file_bytes: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    digest, size = hashlib.sha256(), 0
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            if encoding == "gzip":
                sink = gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0, compresslevel=6)
            else:
                sink = outgoing
            try:
                while block := incoming.read(CHUNK):
                    digest.update(block)
                    size += len(block)
                    sink.write(block)
            finally:
                if sink is not outgoing:
                    sink.close()
        if temporary.stat().st_size > max_file_bytes:
            raise ValueError(f"Stored file exceeds {max_file_bytes} bytes: {destination}")
        reproduced_sha, reproduced_bytes = _sha(temporary, decompress=encoding == "gzip")
        if (reproduced_sha, reproduced_bytes) != (digest.hexdigest(), size):
            raise ValueError(f"Exact-byte round-trip failed: {source}")
        stored_sha, stored_size = _sha(temporary)
        temporary.replace(destination)
        return {"source": _display(source), "source_bytes": size,
                "source_sha256": digest.hexdigest(), "stored_bytes": stored_size,
                "stored_sha256": stored_sha, "encoding": encoding,
                "exact_decompressed_bytes_verified": True}
    finally:
        temporary.unlink(missing_ok=True)


def _decisions(source: Path, destination: Path) -> dict:
    """A small, explicitly derived table; original selection JSON remains intact."""
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected the saved list of placement results")
    fields = ["scene", "part", "status", "proposed_count", "valid_count", "selected_candidate_id",
              "r_tau", "r_surface", "theta_down_rad", "case_selection", "candidate_diagnostics"]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            selection = row["selection"]
            grasp = selection.get("selected_grasp") or {}
            criteria = grasp.get("criteria") or {}
            case = f"cases/{row['scene']}/{row['part']}"
            writer.writerow({"scene": row["scene"], "part": row["part"],
                             "status": selection["status"], "proposed_count": selection["proposed_count"],
                             "valid_count": selection["valid_count"],
                             "selected_candidate_id": grasp.get("candidate_id", ""),
                             **{key: criteria.get(key, "") for key in ("r_tau", "r_surface", "theta_down_rad")},
                             "case_selection": f"{case}/selection.json.gz",
                             "candidate_diagnostics": f"{case}/candidate_diagnostics.json.gz"})
    return {"placement_count": len(rows), "selected_placements": sum(r["selection"].get("selected_grasp") is not None for r in rows),
            "valid_poses": sum(r["selection"]["valid_count"] for r in rows),
            "proposed_poses": sum(r["selection"]["proposed_count"] for r in rows)}


def publish_inputs(inputs: Iterable[tuple[str | Path, str | Path]], destination: str | Path,
                   *, label: str = "Shared exact task and hand references") -> dict:
    """Publish only caller-enumerated shared references, without CAD duplication."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    inputs = [(Path(src).resolve(strict=True), _relative(dst)) for src, dst in inputs]
    if len(inputs) != len({dst.as_posix() for _, dst in inputs}):
        raise ValueError("Duplicate shared-input destination")
    if any(dst == Path("publication_manifest.json") for _, dst in inputs):
        raise ValueError("Input collides with publication metadata")
    inventory = [{"source": _display(src), "bytes": src.stat().st_size} for src, _ in inputs]
    destination.mkdir(parents=True)
    files = []
    for src, relative in inputs:
        encoding = "gzip" if relative.suffix == ".gz" else "identity"
        files.append({"stored_relative": relative.as_posix(),
                      **_copy(src, destination / relative, encoding=encoding, max_file_bytes=MAX_FILE_BYTES)})
    manifest = {"schema": "exact-grasp-shared-inputs-v1", "label": label,
                "source_directory": "explicit caller-enumerated shared references", "files": files,
                "inventory_before_copy": inventory, "historical_source_files_modified": False,
                "git_operations_performed": False, "derived_files": []}
    (destination / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def publish_run(source: str | Path, destination: str | Path, *, label: str,
                supplements: Iterable[tuple[str | Path, str | Path]] = (),
                max_file_bytes: int = MAX_FILE_BYTES) -> dict:
    """Prepare a new review bundle. Does not stage, commit, push or run a trial."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Source and destination trees must be separate")
    if destination.exists():
        raise FileExistsError(f"Publication destination already exists: {destination}")
    plan = plan_run(source)  # inventory is deliberately completed before writing
    extra = [(Path(src).resolve(strict=True), _relative(dst)) for src, dst in supplements]
    destinations = [r["stored_relative"] for r in plan["files"]] + [p.as_posix() for _, p in extra]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate publication destination")
    if {"publication_manifest.json", "README.md", "decisions.csv"}.intersection(destinations):
        raise ValueError("Supplement collides with generated publication metadata")
    destination.mkdir(parents=True)
    entries = []
    for row in plan["files"]:
        entry = _copy(Path(row["source"]), destination / row["stored_relative"],
                      encoding=row["encoding"], max_file_bytes=max_file_bytes)
        entries.append({"stored_relative": row["stored_relative"], **entry})
    for src, relative in extra:
        encoding = "gzip" if relative.suffix == ".gz" else "identity"
        entry = _copy(src, destination / relative, encoding=encoding, max_file_bytes=max_file_bytes)
        entries.append({"stored_relative": relative.as_posix(), **entry, "explicit_supplement": True})
    counts = _decisions(source, destination / "decisions.csv")
    manifest = {
        "schema": "exact-grasp-evidence-publication-v1", "label": label,
        "source_directory": _display(source), "inventory_before_copy": {k: v for k, v in plan.items() if k != "files"},
        "counts_recounted_from_saved_results": counts, "max_stored_file_bytes": max_file_bytes,
        "files": entries, "stored_evidence_bytes": sum(e["stored_bytes"] for e in entries),
        "uncompressed_evidence_bytes": sum(e["source_bytes"] for e in entries),
        "publication_helper": _display(Path(__file__)), "publication_helper_sha256": _sha(Path(__file__))[0],
        "grasp_generation_performed": False, "physical_execution_performed": False,
        "VLM_selection": "excluded", "git_operations_performed": False,
        "historical_source_files_modified": False,
        "derived_files": [{"path": "decisions.csv", "sha256": _sha(destination / "decisions.csv")[0],
                           "description": "Compact saved-decision table; no rescoring or selection changes"}],
    }
    (destination / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _readme(destination, manifest)
    return manifest


def _readme(destination: Path, manifest: dict) -> None:
    counts = manifest["counts_recounted_from_saved_results"]
    text = f"""# {manifest['label']}

This is saved geometric evidence: **{counts['selected_placements']}/{counts['placement_count']} placements** retain a selected grasp, with {counts['valid_poses']} valid modeled candidates from {counts['proposed_poses']} proposals. These counts do not measure pickup or assembly success. VLM selection is excluded; pickup and assembly execution remain paused.

- [Selected-grasp gallery](gallery/index.html) and [report](report.md).
- [Compact per-placement decisions](decisions.csv), [saved summary](summary.json), and [byte-preservation manifest](publication_manifest.json).
- [Complete saved results](results.json.gz). Each decision row links its complete selection and full candidate diagnostics, stored as `.json.gz` files.

Every copied source file is byte-identical after decompression. The manifest records original paths, exact source and stored SHA256 values, byte counts, and the pre-copy tree inventory. The decisions table is derived directly from saved values and does not run or change selection. Copied historical reports retain their original text and local source links; use this index for included files. Gallery HTML and PNGs are copied together and require no compressed JSON to open locally.

Read compressed JSON in Python without rewriting the source:

```python
import gzip, json
with gzip.open('results.json.gz', 'rt', encoding='utf-8') as f:
    placements = json.load(f)
```

Run `python -m experiments.cad_grasp_controlled_20261029.publish verify <this-directory>` from the repository root to verify every stored file and its decompressed source hash. Source paths in historical evidence may name the original local machine; they are provenance, not required extraction paths. Native CAD meshes remain at their existing tracked repository locations. No weights, raw captures or installed dependencies are copied.
"""
    (destination / "README.md").write_text(text, encoding="utf-8")


def verify_bundle(destination: str | Path) -> dict:
    destination = Path(destination).resolve(strict=True)
    manifest = json.loads((destination / "publication_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for row in manifest["files"]:
        path = destination / _relative(row["stored_relative"])
        if not path.resolve().is_relative_to(destination):
            raise ValueError(f"Stored file escapes bundle: {path}")
        if _sha(path) != (row["stored_sha256"], row["stored_bytes"]):
            raise ValueError(f"Stored bytes changed: {row['stored_relative']}")
        if _sha(path, decompress=row["encoding"] == "gzip") != (row["source_sha256"], row["source_bytes"]):
            raise ValueError(f"Source round-trip changed: {row['stored_relative']}")
        checked += 1
    for row in manifest.get("derived_files", []):
        if _sha(destination / _relative(row["path"]))[0] != row["sha256"]:
            raise ValueError(f"Derived table changed: {row['path']}")
    return {"verified": True, "exact_source_files": checked, "source_directory": manifest["source_directory"]}


def write_index(publication_root: str | Path) -> Path:
    root = Path(publication_root)
    rows = []
    for path in sorted(root.glob("*/publication_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        counts = manifest["counts_recounted_from_saved_results"]
        rows.append(f"| [{manifest['label']}]({path.parent.name}/README.md) | {counts['selected_placements']}/{counts['placement_count']} | {counts['valid_poses']} | {manifest['stored_evidence_bytes'] / 1048576:.2f} MiB | [{path.parent.name}/gallery]({path.parent.name}/gallery/index.html) |")
    output = root / "README.md"
    output.write_text("# October 29 grasp evidence\n\nThese are preserved geometric trials and a controlled follow-up. Availability of a valid modeled grasp does not establish pickup or assembly success. VLM selection is excluded; pickup and assembly execution remain paused. Each run has its own exact-byte publication manifest and complete compressed candidate diagnostics. Historical numerical outcomes are unchanged.\n\n| Saved trial | Placements selected | Valid candidates | Stored evidence | Gallery |\n|---|---:|---:|---:|---|\n" + "\n".join(rows) + "\n\nThe 35/96 support baseline combines 72 unchanged cases and 24 corrected round-pin cases; its derivation and source manifests are included. It has no standalone 96-case elapsed time. The 37/96 finger-contact trial changes the contact model, so its lower number of valid candidates is not an isolated improvement or regression under an identical checker. The new controlled comparison reports its own shared conditions and timing scope.\n\nShared task and hand references are included separately when listed in their manifest. Native CAD files already tracked by the repository are not duplicated. The publication process performs no trial, robot action, git stage, commit or push.\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("source", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("destination", type=Path)
    publish = sub.add_parser("publish")
    publish.add_argument("source", type=Path)
    publish.add_argument("destination", type=Path)
    publish.add_argument("--label", required=True)
    index = sub.add_parser("index")
    index.add_argument("publication_root", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan_run(args.source)
    elif args.command == "verify":
        result = verify_bundle(args.destination)
    elif args.command == "publish":
        result = publish_run(args.source, args.destination, label=args.label)
        result = {k: v for k, v in result.items() if k != "files"}
    else:
        result = str(write_index(args.publication_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
