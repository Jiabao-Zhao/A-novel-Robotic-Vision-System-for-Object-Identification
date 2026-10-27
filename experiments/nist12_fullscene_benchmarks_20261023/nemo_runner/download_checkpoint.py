"""Download the pinned official NeMO checkpoint shards for this experiment."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request

COMMIT = '76478815baf2c99002cccbee98426773fa4e59cd'
HERE = Path(__file__).resolve().parent
DESTINATION = HERE / 'weights'


def fetch(name):
    destination = DESTINATION / name
    url = f'https://raw.githubusercontent.com/DLR-RM/NeMO/{COMMIT}/checkpoints/{name}'
    if destination.exists():
        return destination
    temporary = destination.with_suffix('.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'nist12-authorized-benchmark'})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as output:
        length = response.headers.get('Content-Length')
        print(f'download {name} expected_bytes={length}', flush=True)
        while block := response.read(4 * 1024 * 1024):
            output.write(block)
    if length is not None:
        assert temporary.stat().st_size == int(length)
    temporary.replace(destination)
    print(f'completed {name} bytes={destination.stat().st_size}', flush=True)
    return destination


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    names = ['part_a' + chr(ord('a') + index) for index in range(16)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        shards = list(executor.map(fetch, names))
    checkpoint = DESTINATION / 'checkpoint.pth'
    metadata = {'official_source_commit': COMMIT, 'shards': [], 'checkpoint': str(checkpoint)}
    sha = hashlib.sha256()
    with checkpoint.open('wb') as output:
        for shard in shards:
            shard_sha = hashlib.sha256()
            with shard.open('rb') as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
                    sha.update(block)
                    shard_sha.update(block)
            metadata['shards'].append({'name': shard.name, 'bytes': shard.stat().st_size,
                                       'sha256': shard_sha.hexdigest()})
    metadata['bytes'] = checkpoint.stat().st_size
    metadata['sha256'] = sha.hexdigest()
    (DESTINATION / 'checkpoint_source.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
