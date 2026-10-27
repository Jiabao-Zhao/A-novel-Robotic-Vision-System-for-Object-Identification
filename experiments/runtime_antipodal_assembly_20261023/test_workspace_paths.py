"""Offline checks of the retained inputs after the C: workspace migration."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.runtime_antipodal_assembly_20261023 import extract_goal_audit
from experiments.runtime_antipodal_assembly_20261023 import prepare_scene
from experiments.runtime_antipodal_assembly_20261023 import run_experiment


def test_scene_consumers_share_the_migrated_input():
    expected = (prepare_scene.ROOT / "outputs/datasets/nist_task_board/legacy_workspace_inputs"
                / "nist-taskboard-workspace/objects-centered/scene.xml")
    assert prepare_scene.SOURCE == expected
    assert run_experiment.SOURCE == expected
    assert expected.is_file()


def test_goal_audit_provenance_inputs_are_preserved():
    assert (extract_goal_audit.WORKSPACE / "scene.json").is_file()
    for name in ("geometry_preflight.json", "README.md", "insertion_demo.py", "full_sequence.py"):
        assert (extract_goal_audit.PILOT / name).is_file(), name


def test_runtime_sources_no_longer_address_the_retired_workspace():
    folder = Path(__file__).resolve().parent
    for name in ("prepare_scene.py", "run_experiment.py", "extract_goal_audit.py", "test_runtime_grasp.py"):
        source = (folder / name).read_text()
        assert "Documents/Codex/2026-06-02" not in source, name
        assert "can-you-add-this-skill-to" not in source, name


def migrated_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_scene, "ROOT", tmp_path)
    scene = tmp_path / "input/scene.xml"
    scene.parent.mkdir()
    scene.write_text('<mujoco><mesh file="D:/assets/part.stl"/></mujoco>')
    old = sha256(b'<mujoco><mesh file="C:/assets/part.stl"/></mujoco>').hexdigest()
    receipt = {"status": "verified", "changes": [{
        "file": "input/scene.xml", "before_sha256": old,
        "after_sha256": sha256(scene.read_bytes()).hexdigest(), "kind": "xml_asset_paths_only"}]}
    return scene, old, receipt


def write_receipt(root, receipt):
    path = root / "outputs/c_workspace_cleanup_20261027/path_migration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))


def test_xml_hash_accepts_original_bytes_without_a_receipt(tmp_path, monkeypatch):
    scene, _, _ = migrated_xml(tmp_path, monkeypatch)
    prepare_scene._verify_xml_hash(scene, sha256(scene.read_bytes()).hexdigest())


def test_xml_hash_accepts_the_exact_verified_migration(tmp_path, monkeypatch):
    scene, old, receipt = migrated_xml(tmp_path, monkeypatch)
    write_receipt(tmp_path, receipt)
    prepare_scene._verify_xml_hash(scene, old)


@pytest.mark.parametrize("failure", ["unlisted_path", "wrong_prior_hash", "tampered_new_hash",
                                     "unverified_receipt", "non_xml_change"])
def test_xml_hash_rejects_unverified_changes(tmp_path, monkeypatch, failure):
    scene, old, receipt = migrated_xml(tmp_path, monkeypatch)
    if failure == "unlisted_path":
        receipt["changes"][0]["file"] = "different/scene.xml"
    elif failure == "wrong_prior_hash":
        receipt["changes"][0]["before_sha256"] = "0" * 64
    elif failure == "tampered_new_hash":
        scene.write_text("<mujoco/> ")
    elif failure == "unverified_receipt":
        receipt["status"] = "planned"
    else:
        receipt["changes"][0]["kind"] = "json_path_values_only"
    write_receipt(tmp_path, receipt)
    with pytest.raises(RuntimeError, match="verified path migration"):
        prepare_scene._verify_xml_hash(scene, old)
