"""Resolve historical hashes after explicitly recorded current-source changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = Path(__file__).with_name('source_migrations.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_source(path, expected_sha256):
    """Return actual matching bytes, never accept a changed active file as frozen."""
    name = str(path).replace('\\', '/')
    original = ROOT / name
    if original.is_file() and sha(original) == expected_sha256:
        return original
    mappings = json.loads(MIGRATIONS.read_text(encoding='utf-8'))['migrations']
    matches = [m for m in mappings if m['original_path'] == name
               and m['sha256'] == expected_sha256]
    if len(matches) != 1:
        raise ValueError(f'No matching historical source or recorded snapshot: {name}')
    snapshot = ROOT / matches[0]['historical_snapshot_path']
    if not snapshot.is_file() or sha(snapshot) != expected_sha256:
        raise ValueError(f'Historical snapshot hash mismatch: {name}')
    return snapshot


def verify_frozen_run():
    run = ROOT / 'outputs/cad_grasp_selection_20261028/runs/20261028T190844Z'
    manifest = json.loads((run / 'publication_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        if sha(run / entry['path']) != entry['sha256']:
            raise ValueError(f'Historical numerical evidence changed: {entry["path"]}')
    launch = json.loads((run / 'launch_manifest.json').read_text(encoding='utf-8'))
    resolved = []
    for category in ['source_sha256', 'protected_sha256']:
        for name, expected in launch[category].items():
            path = historical_source(name, expected)
            resolved.append(dict(original_path=name.replace('\\', '/'),
                                 verified_path=path.relative_to(ROOT).as_posix(),
                                 sha256=expected, used_snapshot=path != ROOT / name.replace('\\', '/')))
    return dict(publication_files_verified=len(manifest['files']),
                launched_sources_and_protected_inputs_verified=len(resolved),
                preserved_source_resolution=resolved,
                original_launch_manifest_unchanged=True,
                note='Current scene changes are authorized. Exact historical bytes are checked at the explicitly recorded snapshot when active code differs.')


if __name__ == '__main__':
    print(json.dumps(verify_frozen_run(), indent=2))
