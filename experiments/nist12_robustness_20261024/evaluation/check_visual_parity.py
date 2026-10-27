"""Label-free source/cache audit and full replay of the old saved CLIP bank.

Loads NumPy arrays only. It does not load a model, use CUDA, read object labels,
or modify the root visual runner or any historical artifact.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, OLD, PRIOR, digest, read, relative, save


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    folder = HERE/'evaluation/visual_parity'
    folder.mkdir(parents=True, exist_ok=True)
    original_protocol = read(OLD/'protocol.json')
    prior_protocol = read(PRIOR/'ours/description_protocol.json')
    assert prior_protocol['qwen'] == original_protocol['qwen']
    assert original_protocol.get('system', '') == prior_protocol['system'] == ''
    prompts = {mode: (OLD/'prompts'/(mode+'.txt')).read_text(encoding='utf-8')
               for mode in ('keywords', 'phrases')}
    assert prior_protocol['prompts'] == prompts
    protected = read(HERE/'protocol.json')['source_protection']
    assert all(digest(OLD/'prompts'/(mode+'.txt')) == protected[mode+'_prompt_sha256'] for mode in prompts)
    cache, conflicts, records = {}, [], 0
    for base, input_name in ((OLD, 'inputs.json'), (PRIOR/'ours', 'description_inputs.json')):
        inputs = {r['input_id']: r for r in read(base/input_name)}
        for row in read(base/'descriptions.json'):
            key = inputs[row['input_id']]['sha256'], row['mode']
            semantic = {name: row.get(name) for name in
                        ('status', 'fields', 'raw_output', 'finished', 'reason')}
            if key in cache and cache[key] != semantic:
                conflicts.append(dict(image_sha256=key[0], mode=key[1]))
            cache[key] = semantic
            records += 1
    assert not conflicts, ('Hash-identical cached images have conflicting outputs', conflicts)
    new_inputs = load('visual_wrapper_audit', HERE/'run_visual.py').inputs_now()
    assert len({r['input_id'] for r in new_inputs}) == len(new_inputs)
    cad_views = {}
    for row in new_inputs:
        if row['kind'] == 'CAD':
            cad_views.setdefault(row['target'], set()).add(row['view_index'])
    assert len(cad_views) == 46 and all(len(v) == 20 for v in cad_views.values())

    # Replay the current wrapper unchanged against historical frozen embeddings.
    replay_root = folder/'replay'
    visual = replay_root/'visual'
    visual.mkdir(parents=True, exist_ok=True)
    old_inputs = read(OLD/'inputs.json')
    save(visual/'inputs.json', old_inputs)
    feature_file = visual/'features.npz'
    if not feature_file.exists():
        os.link(OLD/'clip/features.npz', feature_file)
    assert digest(feature_file) == digest(OLD/'clip/features.npz')
    states = sorted({r['state'] for r in old_inputs if r['kind'] == 'crop'})
    save(replay_root/'manifest.json', {'scenes': [{'scene_id': s} for s in states]})
    runner = load('visual_wrapper_replay', HERE/'run_visual.py')
    runner.HERE = replay_root
    runner.score(relative(replay_root/'manifest.json'), relative(HERE/'catalog.json'), 'old_replay')
    actual_pairs = read(visual/'old_replay/pairs.json')
    old_pairs = {(r['state'], r['target'], r['proposal_id']): r['clip_full']
                 for r in read(OLD/'clip/candidate_scores.json')}
    maximum_error = 0.
    for pair in actual_pairs:
        expected = old_pairs[pair['scene_id'], pair['target'], pair['proposal_id']]
        assert pair['score_available'] == expected['score_available']
        if pair['score_available']:
            maximum_error = max(maximum_error, abs(pair['score']-expected['score']))
            assert pair['best_CAD_view'] == expected['selected_CLIP_CAD_view']
    assert maximum_error == 0.
    actual_tasks = {(r['scene_id'], r['target']): r for r in read(visual/'old_replay/shortlists.json')}
    old_shortlists = [r for r in read(OLD/'clip/shortlist.json') if r['method'] == 'clip_full']
    comparable = 0
    for old in old_shortlists:
        if old['status'] == 'ok':
            actual = actual_tasks[old['state'], old['target']]
            assert actual['status'] == 'ok' and actual['candidate_ids'] == old['candidate_ids']
            comparable += 1
    report = dict(passed=True, runner_sha256=digest(HERE/'run_visual.py'),
        original_CLIP_source_sha256=digest(OLD/'run_clip.py'),
        prior_Qwen_manifest_and_prompts_exact=True, cached_records_checked=records,
        unique_cache_keys=len(cache), conflicting_cache_keys=conflicts,
        CAD_targets=len(cad_views), views_per_target=20, CAD_input_ID_collisions=False,
        replayed_candidate_pairs=len(actual_pairs), maximum_absolute_score_error=maximum_error,
        selected_view_mapping_exact=True, matching_fully_available_shortlists=comparable,
        intentional_new_policy='Rank finite available scene candidates instead of historical all-candidate veto; require all CAD views per crop. Allow fewer than five crops when fewer are available.',
        model_forward_calls=0, CUDA_used=False, annotations_read=False)
    save(folder/'completion.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
