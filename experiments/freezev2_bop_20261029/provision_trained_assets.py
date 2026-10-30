"""Resume one missing official pretrained asset; never overwrite a saved model.

This experiment-local acquisition helper does not install into any environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def acquire(url, destination, expected_bytes):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination.with_name(destination.name + '.acquisition.json')
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise RuntimeError(f'Existing model has unexpected size; preserve it: {destination}')
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['source_url'] != url or saved['bytes'] != expected_bytes:
                raise RuntimeError('Existing model acquisition identity differs')
            print(json.dumps({'status': 'existing_asset_reused', 'path': str(destination), 'sha256': saved['sha256']}), flush=True)
            return saved
        sha = checksum(destination)
        metadata = {'status': 'existing_asset_inventoried', 'source_url': url, 'path': str(destination),
                    'bytes': expected_bytes, 'sha256': sha, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata
    partial = destination.with_name(destination.name + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise RuntimeError(f'Partial model exceeds expected size; preserve it: {partial}')
    headers = {'User-Agent': 'FreeZeV2-official-asset-acquisition'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    started, last_report = time.monotonic(), time.monotonic()
    response_headers = {}
    if offset < expected_bytes:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status != 206:
                raise RuntimeError('Server ignored resume Range; partial file preserved')
            if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise RuntimeError('Resume range does not match partial size')
            response_headers = dict(response.headers.items())
            with partial.open('ab' if offset else 'wb') as output:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    offset += len(block)
                    if offset > expected_bytes:
                        raise RuntimeError('Download exceeded official expected size')
                    now = time.monotonic()
                    if now - last_report > 10:
                        print(json.dumps({'status': 'downloading', 'bytes': offset, 'total': expected_bytes,
                                          'percent': round(offset / expected_bytes * 100, 2)}), flush=True)
                        last_report = now
    if offset != expected_bytes:
        raise RuntimeError(f'Incomplete asset: {offset}/{expected_bytes}; resume the same command')
    sha = checksum(partial)
    partial.rename(destination)
    metadata = {'status': 'downloaded', 'source_url': url, 'path': str(destination),
                'bytes': offset, 'sha256': sha, 'response_headers': response_headers,
                'elapsed_s': time.monotonic() - started, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--expected-bytes', required=True, type=int)
    args = parser.parse_args()
    acquire(args.url, args.destination, args.expected_bytes)
