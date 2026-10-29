import pytest

from experiments.nist_scene_corrections_20261029.history import ROOT, historical_source, verify_frozen_run


def test_recorded_snapshot_contains_exact_original_scene_source():
    expected = '404a83d670afba7a0b971cc928d5a514c1469e53b23f257164610dbdb88a6982'
    resolved = historical_source('simulation/nist_task_board_1.py', expected)
    assert resolved.is_file()
    assert resolved == ROOT / 'experiments/nist_scene_corrections_20261029/frozen_sources/simulation/nist_task_board_1.py'


def test_unrecorded_hash_is_not_accepted():
    with pytest.raises(ValueError, match='No matching historical source'):
        historical_source('simulation/nist_task_board_1.py', '0' * 64)


def test_frozen_numeric_evidence_and_launched_code_are_preserved():
    receipt = verify_frozen_run()
    assert receipt['publication_files_verified'] == 249
    assert receipt['launched_sources_and_protected_inputs_verified'] == 28
    changed = [r for r in receipt['preserved_source_resolution'] if r['used_snapshot']]
    assert [r['original_path'] for r in changed] == ['simulation/nist_task_board_1.py']
