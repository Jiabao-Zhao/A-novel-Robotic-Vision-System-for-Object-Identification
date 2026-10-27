"""Unchanged Qwen prompts on fresh crops; reuse only identical pixel evidence."""
import base64
import copy
import importlib.util
from pathlib import Path
import sys
from time import perf_counter
from common import HERE, REPO, read, save, digest, path, relative

OLD = HERE.parent / 'nist12_qwen_clip_reranking_20261023'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    helper = load('fullscene_frozen_qwen_helpers', OLD / 'run_qwen.py')
    local = helper.local
    manifest = local.model_manifest('qwen', generation=True)
    old_inputs = read(OLD / 'inputs.json')
    old_descriptions = read(OLD / 'descriptions.json')
    by_input = {r['input_id']: r for r in old_inputs}
    cached = {(by_input[r['input_id']]['sha256'], r['mode']): r for r in old_descriptions}
    inputs = [copy.deepcopy(r) for r in old_inputs if r['kind'] == 'CAD']
    for r in read(HERE / 'ours/proposals.json'):
        inputs.append({'input_id': f"crop__{r['scene_id']}__p{r['proposal_id']:03d}", 'kind': 'crop',
            'state': r['scene_id'], 'proposal_id': r['proposal_id'], 'file': r['crop'], 'sha256': r['crop_sha256']})
    save(HERE / 'ours/description_inputs.json', inputs)
    prompts = {mode: (OLD / f'prompts/{mode}.txt').read_text(encoding='utf-8') for mode in ('keywords', 'phrases')}
    save(HERE / 'ours/description_protocol.json', {'qwen': manifest, 'system': '', 'prompts': prompts,
         'reuse': 'Only exact RGB SHA256 and identical frozen prompt/model controls; provenance retained.',
         'inputs_sha256': digest(HERE / 'ours/description_inputs.json'),
         'source_sha256': {relative(OLD / n): digest(OLD / n) for n in ('run_qwen.py', 'descriptions.json', 'protocol.json')}})
    provenance = local.runtime_for(manifest, HERE / 'ours/qwen_runtime')
    rows = []
    reused, fresh = 0, 0
    try:
        for item in inputs:
            for mode in ('keywords', 'phrases'):
                key = item['sha256'], mode
                if key in cached:
                    row = copy.deepcopy(cached[key])
                    row.update(input_id=item['input_id'], reused=True,
                               reused_description_file=relative(OLD / 'descriptions.json'),
                               reused_input_id=cached[key]['input_id'], image_sha256=item['sha256'])
                    reused += 1
                else:
                    destination = HERE / 'ours/qwen_raw' / mode / (item['input_id'] + '.json')
                    if destination.exists():
                        stored = read(destination)
                        assert stored['image_sha256'] == item['sha256'] and stored['prompt'] == prompts[mode]
                    else:
                        start = perf_counter()
                        encoded = base64.b64encode(path(item['file']).read_bytes()).decode()
                        images = [{'file': str(path(item['file']).resolve()), 'sha256': item['sha256']}]
                        metadata = local.request_for('qwen', manifest['controls'], '', prompts[mode], images)
                        wire = local.request_for('qwen', manifest['controls'], '', prompts[mode], images, [encoded])
                        response = local.infer('qwen', wire)
                        stored = {'response': response, 'image_sha256': item['sha256'], 'prompt': prompts[mode],
                                  'request': metadata, 'runtime': provenance, 'raw_file': relative(destination),
                                  'duration_seconds': perf_counter()-start, 'generation_error': None}
                        save(destination, stored)
                    row = helper.parsed_record(stored, item, mode, manifest)
                    row.update(reused=False, image_sha256=item['sha256'])
                    fresh += 1
                    print(f"Qwen fresh {fresh}: {item['input_id']} {mode} {row['status']}", flush=True)
                rows.append(row)
                save(HERE / 'ours/descriptions_progress.json', {'reused': reused, 'new_or_resumed': fresh, 'processed': len(rows), 'expected': len(inputs)*2})
        save(HERE / 'ours/descriptions.json', rows)
        save(HERE / 'ours/description_completion.json', {'reused': reused, 'new_or_resumed': fresh, 'total': len(rows), 'all_available': all(r['status'] == 'ok' for r in rows)})
    finally:
        local.release('qwen', manifest)


if __name__ == '__main__':
    main()
