"""Opaque byte checksums only; never parse historical labels or model outputs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, digest, path, read, save


def main():
    frozen_file = HERE/'initial_protected_sources.json'
    expected = read(frozen_file)
    actual, changed, missing = {}, [], []
    for name, sha256 in expected.items():
        file = path(name)
        if not file.is_file():
            missing.append(name)
            continue
        actual[name] = digest(file)
        if actual[name] != sha256:
            changed.append(dict(file=name, expected_sha256=sha256, current_sha256=actual[name]))
    protocol = read(HERE/'protocol.json')
    geometry_hash = protocol['source_protection']['prior_geometry_sha256']
    geometry_name = 'experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/geometry.py'
    geometry_actual = digest(path(geometry_name))
    geometry_unchanged = geometry_actual == geometry_hash
    prompt_name = 'docs/cad_view_prompt.txt'
    assert expected[prompt_name] == protocol['source_protection']['active_CAD_prompt_sha256']
    timestamp = datetime.now(timezone.utc)
    receipt = dict(checked_utc=timestamp.isoformat(), passed=not changed and not missing and geometry_unchanged,
        protected_inventory_sha256=digest(frozen_file), expected_files=len(expected),
        verified_files=len(actual)-len(changed), changed=changed, missing=missing,
        active_CAD_prompt=dict(file=prompt_name, unchanged=actual.get(prompt_name) == expected[prompt_name]),
        prior_geometry=dict(file=geometry_name, expected_sha256=geometry_hash,
                            current_sha256=geometry_actual, unchanged=geometry_unchanged,
                            expected_hash_source='Frozen robustness protocol; additional to initial historical inventory.'),
        current_sha256=actual, protected_files_written=0,
        annotations_parsed=False, model_calls=0, runtime='Windows-native Python; no WSL or model loading.')
    folder = HERE/'evaluation/protected_checks'
    save(folder/(timestamp.strftime('%Y%m%dT%H%M%S%fZ')+'.json'), receipt)
    save(folder/'current.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'current_sha256'}, indent=2))
    assert receipt['passed'], 'Historical evidence or active prompt changed; receipt saved. Do not continue until investigated.'


if __name__ == '__main__':
    main()
