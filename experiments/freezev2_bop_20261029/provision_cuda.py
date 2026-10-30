"""Install selected NVIDIA compiler redistributables into one local directory."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(destination, archive_cache):
    destination, archive_cache = Path(destination).resolve(), Path(archive_cache).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_cache.mkdir(parents=True, exist_ok=True)
    base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    manifest_url = base + 'redistrib_12.1.1.json'
    manifest = json.load(urllib.request.urlopen(manifest_url, timeout=45))
    record = {'source_manifest': manifest_url, 'components': {}, 'destination': str(destination)}
    for component in ('cuda_nvcc', 'cuda_cudart', 'cuda_cccl'):
        item = manifest[component]['linux-x86_64']
        archive = archive_cache / Path(item['relative_path']).name
        if not archive.exists():
            temporary = archive.with_name(archive.name + '.part')
            with urllib.request.urlopen(base + item['relative_path'], timeout=60) as response, temporary.open('wb') as output:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    output.write(block)
            temporary.rename(archive)
        if archive.stat().st_size != int(item['size']) or sha256(archive) != item['sha256']:
            raise RuntimeError(f'Official CUDA redistributable integrity mismatch: {archive}')
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                parts = PurePosixPath(member.name).parts
                if len(parts) <= 1:
                    continue
                relative = PurePosixPath(*parts[1:])
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unsafe redistributable archive path')
                target = destination / str(relative)
                if not target.resolve().is_relative_to(destination):
                    raise RuntimeError('Redistributable member resolves outside destination')
                stripped = copy.copy(member)
                stripped.name = str(relative)
                source.extract(stripped, destination, filter='data')
        record['components'][component] = dict(item, version=manifest[component]['version'], license=manifest[component]['license'])
        print(json.dumps({'component': component, 'verified_sha256': item['sha256']}), flush=True)
    record['compiler_version'] = subprocess.check_output([str(destination / 'bin/nvcc'), '--version'], text=True)
    (destination / 'freezev2_provisioning.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    parser.add_argument('--archive-cache', required=True)
    args = parser.parse_args()
    main(args.destination, args.archive_cache)
