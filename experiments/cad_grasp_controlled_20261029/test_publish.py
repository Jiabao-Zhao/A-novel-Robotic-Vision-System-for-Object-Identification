from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from experiments.cad_grasp_controlled_20261029 import publish


def _run(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    case = source / "cases" / "N00" / "KET8_Square_8mm"
    case.mkdir(parents=True)
    selection = {"status": "selected", "proposed_count": 2, "valid_count": 1,
                 "selected_grasp": {"candidate_id": 7, "criteria": {"r_tau": 0.1, "r_surface": 0.4, "theta_down_rad": 0.0}}}
    raw = json.dumps(selection, indent=2).encode() + b"\r\n"
    (case / "selection.json").write_bytes(raw)
    (case / "candidate_diagnostics.json").write_bytes(b'{"candidates": [' + b'{"valid":false},' * 2000 + b'{"valid":true}]}\r\n')
    (source / "results.json").write_text(json.dumps([{"scene": "N00", "part": "KET8_Square_8mm", "selection": selection}]), encoding="utf-8")
    (source / "summary.json").write_text('{"selected_placements":1}', encoding="utf-8")
    (source / "report.md").write_bytes(b"# Saved report\r\nA historical statement.\r\n")
    gallery = source / "gallery"
    gallery.mkdir()
    (gallery / "index.html").write_text('<img src="N00.png">', encoding="utf-8")
    (gallery / "N00.png").write_bytes(b"example image bytes")
    (source / "private_unknown.json").write_text("not allowlisted", encoding="utf-8")
    (source / "unused_weights.pt").write_bytes(b"not copied")
    return source


def test_plan_inventory_and_allowlist(tmp_path):
    source = _run(tmp_path)
    plan = publish.plan_run(source)
    assert plan["tree_file_count"] == 9
    assert plan["allowed_file_count"] == 7
    assert plan["excluded_files"] == ["private_unknown.json", "unused_weights.pt"]
    assert plan["tree_bytes"] == sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    assert next(r for r in plan["files"] if r["source_relative"] == "results.json")["encoding"] == "gzip"


def test_exact_gzip_roundtrip_and_small_decisions(tmp_path):
    source = _run(tmp_path)
    originals = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "published"
    manifest = publish.publish_run(source, destination, label="Test saved run")
    assert publish.verify_bundle(destination)["verified"]
    assert manifest["counts_recounted_from_saved_results"] == {"placement_count": 1, "selected_placements": 1, "valid_poses": 1, "proposed_poses": 2}
    for row in manifest["files"]:
        stored = destination / row["stored_relative"]
        raw = gzip.decompress(stored.read_bytes()) if row["encoding"] == "gzip" else stored.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["source_sha256"]
    assert (destination / "report.md").read_bytes() == originals[Path("report.md")]
    assert all((source / p).read_bytes() == raw for p, raw in originals.items())
    assert "cases/N00/KET8_Square_8mm/candidate_diagnostics.json.gz" in (destination / "decisions.csv").read_text()
    assert "not measure pickup or assembly success" in (destination / "README.md").read_text()


def test_compression_deterministic_and_tamper_detected(tmp_path):
    source = _run(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    publish.publish_run(source, a, label="A")
    publish.publish_run(source, b, label="B")
    assert (a / "results.json.gz").read_bytes() == (b / "results.json.gz").read_bytes()
    with (a / "results.json.gz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Stored bytes changed"):
        publish.verify_bundle(a)


def test_existing_destination_and_escape_are_rejected(tmp_path):
    source = _run(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish.publish_run(source, destination, label="Forbidden overwrite")
    with pytest.raises(ValueError, match="safe bundle-relative"):
        publish.publish_run(source, tmp_path / "other", label="Traversal", supplements=[(source / "summary.json", "../escaped.json")])
    with pytest.raises(ValueError, match="separate"):
        publish.publish_run(source, source / "nested", label="Recursive copy")
    assert not (tmp_path / "other").exists()


def test_size_limit_and_shared_references(tmp_path):
    source = _run(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        publish.publish_run(source, tmp_path / "too_small", label="Bounded", max_file_bytes=1)
    assert not list((tmp_path / "too_small").rglob("*.partial"))
    inputs = tmp_path / "shared"
    manifest = publish.publish_inputs([(source / "summary.json", "saved_summary.json")], inputs)
    assert publish.verify_bundle(inputs)["exact_source_files"] == 1
    assert manifest["inventory_before_copy"][0]["bytes"] == (source / "summary.json").stat().st_size


def test_decision_table_tamper_detected(tmp_path):
    source = _run(tmp_path)
    destination = tmp_path / "bundle"
    publish.publish_run(source, destination, label="Audit")
    (destination / "decisions.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Derived table changed"):
        publish.verify_bundle(destination)


def test_index_skips_shared_input_manifest(tmp_path):
    source = _run(tmp_path)
    root = tmp_path / "publication"
    publish.publish_run(source, root / "saved_run", label="Saved trial")
    publish.publish_inputs([(source / "summary.json", "summary.json")], root / "shared_inputs")
    index = publish.write_index(root).read_text(encoding="utf-8")
    assert "[Saved trial](saved_run/README.md)" in index
    assert "| 1/1 | 1 |" in index


def test_stage_table_keeps_counts_and_detects_inconsistent_totals(tmp_path):
    stages = ("contact_valid", "pickup_valid", "final_goal_valid", "joint_valid")
    counts = dict(zip(stages, (5, 4, 3, 2)))
    placements = dict.fromkeys(stages, 1)
    document = {"definitions": {s: ["explicit_pass"] for s in stages}, "banks": {"saved": {
        "per_part": {"pin": {"counts": counts.copy(), "placements": placements.copy(), "candidate_evaluations": 100}},
        "passing_candidate_evaluations": counts.copy(), "placements_with_at_least_one": placements.copy()}}}
    source = tmp_path / "stages.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    receipt = publish.write_stage_tables(source, tmp_path / "tables")
    assert not receipt["geometry_reevaluated"] and not receipt["counts_changed"]
    assert "| pin | 5 / 1 | 4 / 1 | 3 / 1 | 2 / 1 |" in (tmp_path / "tables/geometry_stages.md").read_text()
    document["banks"]["saved"]["passing_candidate_evaluations"]["joint_valid"] = 7
    source.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="totals disagree"):
        publish.write_stage_tables(source, tmp_path / "bad_tables")
