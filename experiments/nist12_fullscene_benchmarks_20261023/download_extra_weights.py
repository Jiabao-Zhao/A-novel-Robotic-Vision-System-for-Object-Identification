"""Official text-encoder and Ultralytics SAM2.1 assets for native proposals."""
import concurrent.futures
import json
import urllib.request
from common import WEIGHTS, digest, save


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + '.receipt.json')
    if destination.exists() and receipt.exists():
        return
    partial = destination.with_name(destination.name + '.partial')
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
        expected = int(response.headers.get('Content-Length', 0))
        total = 0
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            total += len(block)
        assert expected == 0 or total == expected
    partial.replace(destination)
    save(receipt, {'url': url, 'bytes': total, 'sha256': digest(destination)})
    print(f'Completed {destination.name}: {total / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    with urllib.request.urlopen('https://huggingface.co/api/models/google-bert/bert-base-uncased') as response:
        revision = json.load(response)['sha']
    save(WEIGHTS / 'bert_revision.json', {'repository': 'google-bert/bert-base-uncased', 'revision': revision})
    files = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors']
    jobs = [('https://huggingface.co/google-bert/bert-base-uncased/resolve/' + revision + '/' + name,
             WEIGHTS / 'bert-base-uncased' / name) for name in files]
    jobs.append(('https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt',
                 WEIGHTS / 'yolosam/sam2.1_l.pt'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda x: download(*x), jobs))
