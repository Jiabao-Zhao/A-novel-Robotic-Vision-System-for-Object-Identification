"""Offline tests for the publisher added after controlled-trial launch."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiments.cad_grasp_controlled_20261029 import publish_controlled as publication


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "saved_controlled"
    source.mkdir()
    launch = {"strategies": ["random"], "seeds": [42], "modes": ["pad", "finger", "combined"],
              "parts": ["pin"], "combined_mode_assignment": "Before validation; same raw bank",
              "underlying_comparison_evaluations": "Two versions per raw configuration"}
    (source / "launch_manifest.json").write_text(json.dumps(launch))
    (source / "summary.json").write_text(json.dumps({"groups": [], "underlying_mode_candidate_evaluations": 16,
                                                  "logical_policy_candidate_evaluations": 24}))
    (source / "run_verification.json").write_text('{"complete_policy_matrix":true}')
    (source / "frozen_angle_sensitivity.json").write_bytes(b"[]\r\n")
    rows = []
    for scene_index in range(8):
        scene = f"N{scene_index:02d}"
        bank = Path("banks/random/seed_0") / scene / "pin"
        (source / bank).mkdir(parents=True)
        for name, value in [("sampling", {"shared_raw_bank_sha256": "declared"}),
                            ("candidate_diagnostics", {"pad": [{"candidate_id": 0}], "finger": [{"candidate_id": 0}],
                                                       "combined_assignment": [{"candidate_id": 0, "contact_mode": "pad"}]})]:
            (source / bank / f"{name}.json.gz").write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
        for mode in launch["modes"]:
            selection = {"status": "selected", "proposed_count": 1, "valid_count": 1,
                         "sampling_reference": str(bank / "sampling.json.gz"),
                         "selected_grasp": {"candidate_id": 0, "contact_mode": "finger" if mode == "finger" else "pad", "criteria": {"r_tau": 0.0}},
                         "stage_counts": {name: {"pass": 1} for name in ("contact_valid", "pickup_valid", "final_goal_valid", "joint_valid")}}
            row = {"sampler_strategy": "random", "seed_index": 0, "seed": 42, "contact_mode": mode,
                   "scene": scene, "part": "pin", "selection": selection,
                   "diagnostic_bank": str(bank / "candidate_diagnostics.json.gz"), "diagnostic_mode": mode}
            rows.append(row)
            case = source / "cases/random/seed_0" / mode / scene / "pin"
            case.mkdir(parents=True)
            (case / "selection.json").write_bytes(json.dumps(row, indent=2).encode() + b"\r\n")
    (source / "results.json").write_text(json.dumps(rows))
    (source / "ignored_payload.bin").write_bytes(b"not evidence")
    return source


def test_complete_matrix_and_original_gzip_are_preserved(tmp_path):
    source = _source(tmp_path)
    before = (source / "banks/random/seed_0/N00/pin/candidate_diagnostics.json.gz").read_bytes()
    destination = tmp_path / "publication"
    manifest = publication.publish_controlled(source, destination)
    assert manifest["matrix"]["saved_policy_rows"] == 24
    assert manifest["matrix"]["diagnostic_banks"] == 8
    assert manifest["inventory_before_copy"]["excluded_files"] == ["ignored_payload.bin"]
    assert publication.verify_controlled(destination)["original_gzip_files_decoded_and_verified"] == 16
    assert (destination / "banks/random/seed_0/N00/pin/candidate_diagnostics.json.gz").read_bytes() == before
    assert gzip.decompress((destination / "cases/random/seed_0/pad/N00/pin/selection.json.gz").read_bytes()) == (source / "cases/random/seed_0/pad/N00/pin/selection.json").read_bytes()
    assert "combined_assignment" in (destination / "README.md").read_text()
    assert manifest["postprocessing_added_after_launch"]
    assert (destination / "provenance/publication_helper.py").read_bytes() == Path(publication.__file__).read_bytes()


def test_incomplete_duplicate_and_over_budget_are_rejected_before_write(tmp_path):
    source = _source(tmp_path)
    rows = json.loads((source / "results.json").read_text())
    for bad, message in [(rows[:-1], "Incomplete"), (rows + rows[:1], "Duplicate")]:
        (source / "results.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=message):
            publication.publish_controlled(source, tmp_path / "missing")
        assert not (tmp_path / "missing").exists()
    rows[0]["selection"]["proposed_count"] = 101
    (source / "results.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="budget"):
        publication.publish_controlled(source, tmp_path / "budget")


def test_bank_tamper_and_existing_destination_are_rejected(tmp_path):
    source = _source(tmp_path)
    destination = tmp_path / "published"
    publication.publish_controlled(source, destination)
    with pytest.raises(FileExistsError):
        publication.publish_controlled(source, destination)
    bank = destination / "banks/random/seed_0/N00/pin/sampling.json.gz"
    bank.write_bytes(gzip.compress(b"changed", mtime=0))
    with pytest.raises(ValueError, match="Stored file changed"):
        publication.verify_controlled(destination)


def test_controlled_index_preserves_historical_rows_and_is_repeatable(tmp_path):
    source = _source(tmp_path)
    destination = tmp_path / "published"
    publication.publish_controlled(source, destination)
    old = "# Old index\n\nPreserved support baseline 35/96 and finger trial 37/96.\n"
    (tmp_path / "README.md").write_text(old)
    path = publication.update_publication_index(destination)
    first = path.read_text()
    assert first.startswith(old.rstrip())
    assert "24 policy-placement records" in first
    publication.update_publication_index(destination)
    assert path.read_text() == first
