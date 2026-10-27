"""Acquire the pinned official identification sources, without model weights."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVISIONS = {'cnos': ('nv-nguyen/cnos', '298d1f3366171464ca271659f0e2f7a6eb8e39b4'),
             'sam6d': ('JiehongLin/SAM-6D', '1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2')}


def main():
    for name, (repository, revision) in REVISIONS.items():
        target = ROOT/'sources'/name
        receipt = ROOT/'sources'/f'{name}_provenance.json'
        if receipt.exists():
            evidence = json.loads(receipt.read_text())
            assert evidence['revision'] == revision
            for relative, expected in evidence['source_sha256'].items():
                assert hashlib.sha256((target/relative).read_bytes()).hexdigest() == expected
            print(f'{name}: existing pinned source verified', flush=True)
            continue
        assert not target.exists(), f'Preserve existing incomplete source: {target}'
        url = f'https://codeload.github.com/{repository}/zip/{revision}'
        print(f'Fetching official {repository}@{revision}', flush=True)
        payload = urllib.request.urlopen(url, timeout=180).read()
        archive = zipfile.ZipFile(io.BytesIO(payload))
        prefix = archive.namelist()[0].split('/')[0]+'/'
        target.mkdir(parents=True)
        for member in archive.infolist():
            relative = member.filename.removeprefix(prefix)
            if not relative or member.is_dir():
                continue
            path = (target/relative).resolve()
            assert path.is_relative_to(target.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(member))
        hashes = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(target.rglob('*')) if p.is_file()}
        evidence = dict(repository=f'https://github.com/{repository}', revision=revision,
                        archive_url=url, archive_sha256=hashlib.sha256(payload).hexdigest(),
                        source_sha256=hashes, model_weights_downloaded=False)
        receipt.write_text(json.dumps(evidence, indent=2)+'\n')
        print(f'{name}: {len(hashes)} official files saved', flush=True)


if __name__ == '__main__':
    main()
