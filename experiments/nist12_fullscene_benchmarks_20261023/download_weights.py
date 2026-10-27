"""Download official frozen checkpoints requested for this benchmark only."""
import concurrent.futures
from pathlib import Path
import urllib.request
import os
from time import monotonic
from common import HERE, WEIGHTS, digest, save

URLS = {
    'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'groundingdino_swint_ogc.pth': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
    'groundingdino_swinb_cogcoor.pth': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth',
    'sam2.1_hiera_large.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
}


def fetch(item):
    name, url = item
    WEIGHTS.mkdir(exist_ok=True)
    destination = WEIGHTS / name
    record_path = WEIGHTS / (name + '.json')
    if destination.exists() and record_path.exists():
        print('Cached ' + name, flush=True)
        return
    partial = WEIGHTS / (name + '.partial')
    start = monotonic()
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as f:
        expected = int(response.headers.get('Content-Length', 0))
        total, last = 0, start
        while block := response.read(8 * 1024 * 1024):
            f.write(block)
            total += len(block)
            if monotonic() - last > 30:
                print(f'{name}: {total / 1e6:.0f} / {expected / 1e6:.0f} MB', flush=True)
                last = monotonic()
        assert not expected or total == expected
    os.replace(partial, destination)
    save(record_path, {'file': name, 'url': url, 'bytes': destination.stat().st_size,
                       'sha256': digest(destination), 'seconds': monotonic() - start})
    print(f'Completed {name}: {destination.stat().st_size / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, URLS.items()))
