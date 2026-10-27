"""Audit frozen label-free evidence, then evaluate retrieved-CAD-to-crop matches."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'analysis'
METHODS = ('dino', 'clip_full_dino', 'clip_shape_dino', 'clip_full_only', 'clip_shape_only')
TOL = 1e-12
LABEL_SOURCE = REPO / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures'
DISPLAY = dict(dino='DINO alone', clip_full_dino='CLIP full → DINO',
    clip_shape_dino='CLIP shape text → DINO', clip_full_only='CLIP full only',
    clip_shape_only='CLIP shape text only')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def local(value):
    value = str(value).replace('\\', '/')
    retired = 'C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to'
    for prefix in (retired, '/mnt/c' + retired[2:]):
        if value == prefix or value.startswith(prefix + '/'):
            suffix = value[len(prefix):].lstrip('/')
            if '..' in suffix.split('/') or ':' in suffix:
                raise ValueError('Unsafe retired-workspace path suffix')
            return REPO / 'outputs/datasets/nist_task_board/legacy_workspace_inputs' / suffix
    if sys.platform == 'win32' and value.startswith('/mnt/'):
        value = value[5].upper() + ':' + value[6:]
    elif sys.platform != 'win32' and len(value) > 1 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + value[2:]
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def csv_file(path, rows):
    assert rows
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def close(actual, expected, atol=1e-12):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def rank(rows):
    """Independently retain missing competitors and unresolved numerical ties."""
    available = sorted((row for row in rows if row['available']), key=lambda row: (-row['score'], row['id']))
    missing = sorted(row['id'] for row in rows if not row['available'])
    if missing or not available:
        return dict(status='unavailable', top_ids=[], missing=missing, top_score=None, gap=None)
    top = [row['id'] for row in available if abs(row['score'] - available[0]['score']) <= TOL]
    gap = 0. if len(top) > 1 else available[0]['score'] - available[1]['score'] if len(available) > 1 else None
    return dict(status='tie' if len(top) > 1 else 'ok', top_ids=top, missing=[], top_score=available[0]['score'], gap=gap)


def shortlist(rows, k=5):
    available = sorted((row for row in rows if row['available']), key=lambda row: (-row['score'], row['id']))
    if any(not row['available'] for row in rows) or len(available) < k:
        return dict(status='unavailable_incomplete_CLIP_evidence', candidate_ids=[])
    if len(available) > k and abs(available[k - 1]['score'] - available[k]['score']) <= TOL:
        return dict(status='unavailable_ambiguous_top5', candidate_ids=[])
    return dict(status='ok', candidate_ids=[row['id'] for row in available[:k]])


def outcome(row, truth):
    ids = row.get('top_ids', row.get('top_proposal_ids', []))
    if row.get('complete', True) is False or row.get('status', '').startswith('unavailable') or not ids:
        return 'unavailable'
    if len(ids) > 1:
        return 'tied'
    return 'correct' if ids[0] == truth else 'wrong'


def family(target):
    if target.startswith('RGOCG'):
        return 'round_pin'
    if target.startswith('KET'):
        return 'rectangular_pin'
    return target


def paired_change(baseline, alternative):
    if baseline == 'wrong' and alternative == 'correct':
        return 'fixed_error'
    if baseline == 'correct' and alternative == 'wrong':
        return 'introduced_error'
    if baseline == 'correct' and alternative != 'correct':
        return 'lost_correct_to_' + alternative
    if baseline != 'correct' and alternative == 'correct':
        return 'resolved_' + baseline
    return 'unchanged_' + baseline if baseline == alternative else baseline + '_to_' + alternative


def verify_completions():
    """This must succeed before any audit identity file is opened."""
    required = ('qwen/completion.json', 'dino/completion.json', 'clip/completion.json',
                'clip/label_free_rankings.json', 'clip/shortlist.json', 'clip/candidate_scores.json',
                'dino/label_free_rankings.json', 'dino/scores.json')
    assert all((HERE / name).is_file() for name in required), 'Complete all label-free generation/scoring before evaluation.'
    protocol = read(HERE / 'protocol.json')
    qwen, dino, clip = [read(HERE / name) for name in required[:3]]
    assert qwen['saved_requests'] == qwen['expected_requests'] == 672
    assert qwen['descriptions_sha256'] == digest(HERE / 'descriptions.json')
    assert qwen['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert dino['passed'] and dino['tasks'] == 96 and dino['comparisons'] == 1152
    assert dino['scores_written_before_identity_audit'] and dino['no_identity_labels_read']
    assert dino['scores_sha256'] == digest(HERE / 'dino/scores.json')
    assert dino['rankings_sha256'] == digest(HERE / 'dino/label_free_rankings.json')
    assert clip['completed'] and clip['scoring_completed'] and clip['task_count'] == 96
    assert set(clip['methods']) == set(METHODS)
    frozen = protocol.get('protected', protocol.get('immutable_inputs'))
    assert frozen
    for file, expected in frozen.items():
        assert digest(local(file)) == expected, 'Frozen source changed: ' + file
    for file, expected in clip['input_hashes'].items():
        assert digest(local(file)) == expected, 'Scoring source changed: ' + file
    for file, expected in clip['artifact_hashes'].items():
        assert digest(HERE / 'clip' / file) == expected, 'CLIP evidence changed: ' + file
    for file, expected in read(HERE / 'dino/protocol.json')['protected'].items():
        assert digest(local(file)) == expected, 'DINO source changed: ' + file
    return protocol, dict(completion_markers_checked=True, protected_sources=len(frozen),
        all_source_and_completion_hashes_pass=True, identity_labels_opened=False)


def verify_qwen(inputs, descriptions):
    import base64
    import copy
    protocol = read(HERE / 'protocol.json')
    assert len(descriptions) == 672
    indexed = {row['input_id']: row for row in inputs}
    seen = set()
    for row in descriptions:
        key = row['input_id'], row['mode']
        assert key not in seen and key[0] in indexed and key[1] in ('keywords', 'phrases')
        seen.add(key)
        raw = read(HERE / row['raw_file'])
        assert raw['input_id'] == key[0] and raw['mode'] == key[1]
        assert raw['image_sha256'] == indexed[key[0]]['sha256'] and raw['image_count'] == 1
        request = raw['request']
        assert request['messages'][0] == dict(role='system', content='')
        assert request['messages'][1]['content'] == protocol['prompts'][key[1]]
        assert len(request['messages'][1]['images']) == 1
        assert request['messages'][1]['images'][0]['sha256'] == indexed[key[0]]['sha256']
        wire = copy.deepcopy(request)
        wire['messages'][1]['images'] = [base64.b64encode(local(indexed[key[0]]['file']).read_bytes()).decode()]
        assert raw['wire_sha256'] == hashlib.sha256(json.dumps(wire).encode()).hexdigest()
        if raw.get('response') is not None:
            response_hash = hashlib.sha256(json.dumps(raw['response']).encode()).hexdigest()
            assert raw['response_sha256'] == response_hash
            assert (raw['response']['message'].get('content') or '') == row['raw_output']
        assert raw['protocol_sha256'] == digest(HERE / 'protocol.json')
        if row['status'] == 'ok':
            import re
            for field in ('COLOR', 'SHAPE', 'MATERIAL'):
                values = re.findall('<' + field + r'\s*>(.*?)</' + field + r'\s*>',
                    row['raw_output'], flags=re.IGNORECASE | re.DOTALL)
                assert len(values) == 1 and values[0].strip() == row['fields'][field]
    assert len(seen) == 2 * len(inputs)
    return dict(raw_requests_checked=len(seen), descriptions_unavailable=sum(row['status'] != 'ok' for row in descriptions))


def verify_clip_features(inputs):
    arrays = dict(np.load(HERE / 'clip/features.npz'))
    mapping = read(HERE / 'clip/tensor_indices.json')
    assert len(mapping['images']) == len(inputs) == 336
    for i, (left, right) in enumerate(zip(mapping['images'], inputs)):
        assert left['image_index'] == i and all(left[key] == value for key, value in right.items())
    np.testing.assert_allclose(arrays['image_normalized'], np.stack([unit(row) for row in arrays['image_raw']]), atol=TOL, rtol=0)
    if len(arrays['chunk_raw']):
        np.testing.assert_allclose(arrays['chunk_normalized'], np.stack([unit(row) for row in arrays['chunk_raw']]), atol=TOL, rtol=0)
    else:
        assert len(arrays['chunk_normalized']) == 0
    for i, field in enumerate(mapping['fields']):
        assert bool(arrays['field_available'][i]) == (field['status'] == 'ok')
        if field['status'] != 'ok':
            continue
        chunks = [mapping['chunks'][j] for j in field['chunk_indices']]
        assert [token for chunk in chunks for token in chunk['content_token_ids']] == field['content_token_ids']
        assert all(0 < len(chunk['content_token_ids']) <= 75 and len(chunk['input_ids']) <= 77 for chunk in chunks)
        np.testing.assert_allclose(arrays['field_normalized'][i], unit(np.mean(
            arrays['chunk_normalized'][field['chunk_indices']], axis=0)), atol=TOL, rtol=0)
    for pool in mapping['pools']:
        i, name = pool['image_index'], pool['pool']
        assert bool(arrays['text_' + name + '_available'][i]) == (pool['status'] == 'ok')
        if pool['status'] == 'ok':
            np.testing.assert_allclose(arrays['text_' + name][i], unit(np.mean(
                arrays['field_normalized'][pool['field_indices']], axis=0)), atol=TOL, rtol=0)
    return arrays, dict(CLIP_image_vectors=336, CLIP_text_fields=len(mapping['fields']),
        CLIP_text_chunks=len(mapping['chunks']), all_content_tokens_preserved=True,
        image_chunk_field_pool_normalization_replayed=True)


def verify_dino_features(inputs, scores):
    """Recompute CPU arithmetic from cached tensors; never load a model."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    records = read(HERE / 'data/input_records.json')
    features = np.load(HERE / 'dino/features.npz')
    prepared = np.load(HERE / 'dino/inputs_and_masks.npz')
    np.testing.assert_array_equal(features['encoded_record_indices'], prepared['encoded_record_indices'])
    indices = {int(record): i for i, record in enumerate(features['encoded_record_indices'])}
    status = read(HERE / 'dino/descriptor_statuses.json')
    reference = {}
    observed = {}
    for i, record in enumerate(records):
        if record['kind'].lower() == 'cad' and status[i]['status'] == 'available':
            reference.setdefault(record['target'], []).append(i)
        elif record['kind'] == 'crop':
            observed[record['state'], record['proposal_id']] = i
    cls = F.normalize(torch.as_tensor(features['cls'], dtype=torch.float32), dim=-1)
    patches = []
    for tokens, mask in zip(features['patches'], prepared['masks']):
        occupancy = mask.reshape(16, 14, 16, 14).mean(axis=(1, 3)).reshape(256)
        gate = torch.as_tensor(occupancy > .5)
        patches.append(F.normalize(torch.as_tensor(tokens, dtype=torch.float32) * gate[:, None], dim=-1))
    checked = 0
    for row in scores:
        if row['score'] is None:
            continue
        query = indices[observed[row['state'], row['proposal_id']]]
        refs = reference[row['target']]
        sims = F.cosine_similarity(cls[query][None], cls[[indices[i] for i in refs]], dim=-1)
        clamped = sims.clamp(0, 1)
        selected = refs[int(clamped.argmax())]
        assert row['selected_CAD_view'] == records[selected]['view_index']
        global_score = float(torch.topk(clamped, 5).values.mean())
        q, r = patches[query], patches[indices[selected]]
        maxima = (q @ r.T).max(dim=-1).values
        patch_score = float((maxima.sum() / (torch.count_nonzero(q.sum(-1)) + 1e-6)).clamp(0, 1))
        close(row['global_score'], global_score, 3e-7)
        close(row['patch_score'], patch_score, 3e-7)
        close(row['score'], .5 * (global_score + patch_score), 3e-7)
        reverse = (q @ r.T).max(dim=0).values
        coverage = float(torch.count_nonzero(reverse * (reverse > .5)) / (torch.count_nonzero(reverse) + 1e-6))
        close(row['coverage'], coverage, 3e-7)
        checked += 1
    return dict(DINO_feature_pairs_replayed=checked, DINO_replay_absolute_tolerance=3e-7,
        DINO_global_and_foreground_patch_arithmetic_independently_replayed=True)


def verify_rankings(inputs, arrays, pairs, views, shortlists, rankings, dino):
    image_index = {row['input_id']: i for i, row in enumerate(inputs)}
    view_by_pair = {}
    for row in views:
        ci, ri = image_index[row['crop_input_id']], image_index[row['CAD_input_id']]
        pool = row['pool']
        crop_ok = bool(arrays['text_' + pool + '_available'][ci])
        cad_ok = bool(arrays['text_' + pool + '_available'][ri])
        expected = dict(image_image=float(arrays['image_normalized'][ci] @ arrays['image_normalized'][ri]),
            text_text=None, crop_image_to_CAD_text=None, crop_text_to_CAD_image=None)
        if cad_ok:
            expected['crop_image_to_CAD_text'] = float(arrays['image_normalized'][ci] @ arrays['text_' + pool][ri])
        if crop_ok:
            expected['crop_text_to_CAD_image'] = float(arrays['text_' + pool][ci] @ arrays['image_normalized'][ri])
        if crop_ok and cad_ok:
            expected['text_text'] = float(arrays['text_' + pool][ci] @ arrays['text_' + pool][ri])
        for field in expected:
            close(row['components'][field], expected[field])
        score = float(np.mean(list(expected.values()))) if crop_ok and cad_ok else None
        close(row['score'], score)
        assert row['score_available'] == (score is not None)
        view_by_pair.setdefault((row['state'], row['target'], row['proposal_id'], pool), []).append(row)
    dino_index = {(row['state'], row['target'], row['proposal_id']): row for row in dino}
    tasks = {}
    for pair in pairs:
        key = pair['state'], pair['target'], pair['proposal_id']
        source = dino_index[key]
        for a, b in (('dino_score', 'score'), ('dino_global_score', 'global_score'), ('dino_patch_score', 'patch_score')):
            close(pair[a], source[b])
        assert pair['dino_selected_CAD_view'] == source['selected_CAD_view']
        for pool in ('full', 'shape'):
            rows = view_by_pair[key + (pool,)]
            assert len(rows) == 20 and len({r['CAD_view_index'] for r in rows}) == 20
            available = sorted((r for r in rows if r['score_available']), key=lambda r: (-r['score'], r['CAD_view_index']))
            best = available[0] if available else None
            saved = pair['clip_' + pool]
            assert saved['score_available'] == (len(available) == 20)
            close(saved['score'], best['score'] if best else None)
            assert saved['selected_CLIP_CAD_view'] == (best['CAD_view_index'] if best else None)
            assert saved['selected_CLIP_view_tie_ids'] == ([] if best is None else [r['CAD_view_index'] for r in available if abs(r['score'] - best['score']) <= TOL])
        tasks.setdefault(key[:2], []).append(pair)
    short_index = {(r['state'], r['target'], r['method']): r for r in shortlists}
    rank_index = {(r['state'], r['target'], r['method']): r for r in rankings}
    retention = []
    def check_status(expected, saved):
        assert expected['top_ids'] == saved['top_ids']
        close(expected.get('top_score'), saved['top_score'])
        close(expected.get('gap'), saved['gap'])
        if expected['status'] == 'unavailable':
            assert saved['status'].startswith('unavailable') and saved['complete'] is False
        else:
            assert saved['status'] == expected['status'] and saved['complete'] is True
    for (state, target), task in tasks.items():
        assert len(task) == 12
        baseline = rank([dict(id=r['proposal_id'], score=r['dino_score'], available=r['dino_score_available']) for r in task])
        check_status(baseline, rank_index[state, target, 'dino'])
        for pool in ('full', 'shape'):
            rows = [dict(id=r['proposal_id'], score=r['clip_' + pool]['score'],
                available=r['clip_' + pool]['score_available']) for r in task]
            short = shortlist(rows)
            saved_short = short_index[state, target, 'clip_' + pool]
            assert short['status'] == saved_short['status'] and short['candidate_ids'] == saved_short['candidate_ids']
            clip_rank = rank(rows)
            saved_clip = rank_index[state, target, 'clip_' + pool + '_only']
            check_status(clip_rank, saved_clip)
            final = rank([dict(id=r['proposal_id'], score=r['dino_score'], available=r['dino_score_available'])
                for r in task if r['proposal_id'] in short['candidate_ids']]) if short['status'] == 'ok' else dict(top_ids=[], status='unavailable')
            saved_final = rank_index[state, target, 'clip_' + pool + '_dino']
            check_status(final, saved_final)
            retained = baseline['status'] == 'ok' and baseline['top_ids'][0] in short['candidate_ids']
            if retained:
                assert saved_final['top_ids'] == baseline['top_ids'], 'DINO winner retained but changed: implementation violation.'
            retention.append(dict(state=state, target=target, pool=pool,
                baseline_unique=baseline['status'] == 'ok', shortlist_status=short['status'],
                baseline_winner_retained=retained, invariant_pass=True))
    assert len(tasks) == 96 and len(pairs) == 1152 and len(views) == 46080 and len(rankings) == 480
    return retention, dict(CLIP_four_cosine_rows_replayed=len(views), candidate_maxima_replayed=len(pairs) * 2,
        shortlist_decisions_replayed=len(shortlists), final_decisions_replayed=len(rankings),
        DINO_winner_retention_invariant_pass=True)


def read_labels_after_verification(inputs, protocol):
    labels, sources = {}, []
    crops = {(row['state'], row['proposal_id']): row for row in inputs if row['kind'] == 'crop'}
    for state in protocol['scenes']:
        path = LABEL_SOURCE / 'scenes' / state / 'candidates.json'
        source = read(path)
        assert len(source) == 12
        labels[state] = {}
        for row in source:
            key = state, row['proposal_id']
            assert row['roi'] == crops[key]['roi'], 'Audit proposal mapping changed.'
            assert row['posthoc_identity'] in protocol['parts']
            labels[state][row['proposal_id']] = row['posthoc_identity']
        assert set(labels[state].values()) == set(protocol['parts'])
        sources.append(dict(file=str(path.relative_to(REPO)).replace('\\', '/'), sha256=digest(path)))
    return labels, sources


def evaluate(protocol, rankings, shortlists, labels):
    short_index = {(r['state'], r['target'], r['method']): r for r in shortlists}
    decisions = []
    for row in rankings:
        truth, = [i for i, label in labels[row['state']].items() if label == row['target']]
        ids = row['top_ids']
        predicted = labels[row['state']][ids[0]] if len(ids) == 1 else None
        pool = 'shape' if 'shape' in row['method'] else 'full'
        short = short_index[row['state'], row['target'], 'clip_' + pool]
        result = outcome(row, truth)
        decisions.append(dict(method=row['method'], state=row['state'], target=row['target'],
            outcome=result, status=row['status'], truth_proposal_id=truth, selected_ids=ids,
            predicted_product=predicted, top_score=row['top_score'], gap=row['gap'],
            shortlist_status=short['status'], shortlist_ids=short['candidate_ids'],
            truth_in_top5=truth in short['candidate_ids'] if short['status'] == 'ok' else None,
            same_family_wrong_size=result == 'wrong' and family(row['target']) in ('round_pin', 'rectangular_pin') and family(predicted) == family(row['target']),
            DINO_CAD_views=row.get('selected_DINO_CAD_views'), CLIP_CAD_views=row.get('selected_CLIP_CAD_views')))
    indexed = {(r['state'], r['target'], r['method']): r for r in decisions}
    changes = []
    for state in protocol['scenes']:
        for target in protocol['parts']:
            baseline = indexed[state, target, 'dino']
            for method in METHODS[1:]:
                alternative = indexed[state, target, method]
                changes.append(dict(state=state, target=target, method=method,
                    change=paired_change(baseline['outcome'], alternative['outcome']),
                    baseline_outcome=baseline['outcome'], alternative_outcome=alternative['outcome'],
                    baseline_ids=baseline['selected_ids'], alternative_ids=alternative['selected_ids'],
                    baseline_product=baseline['predicted_product'], alternative_product=alternative['predicted_product'],
                    truth_in_top5=alternative['truth_in_top5'], shortlist_status=alternative['shortlist_status'],
                    baseline_winner_in_shortlist=bool(len(baseline['selected_ids']) == 1 and baseline['selected_ids'][0] in alternative['shortlist_ids'])))
    per_object = []
    for target in protocol['parts']:
        record = dict(target=target, family=family(target), tasks=8)
        for method in METHODS:
            rows = [r for r in decisions if r['target'] == target and r['method'] == method]
            for status in ('correct', 'wrong', 'tied', 'unavailable'):
                record[method + '_' + status] = sum(r['outcome'] == status for r in rows)
            record[method + '_same_family_wrong_size'] = sum(r['same_family_wrong_size'] for r in rows)
        for pool in ('full', 'shape'):
            rows = [r for r in decisions if r['target'] == target and r['method'] == 'clip_' + pool + '_dino']
            record['clip_' + pool + '_top5_hits'] = sum(r['truth_in_top5'] is True for r in rows)
            record['clip_' + pool + '_top5_unavailable'] = sum(r['truth_in_top5'] is None for r in rows)
        per_object.append(record)
    summary = dict(tasks=96, scenes=8, objects_per_scene=12, CAD_views=240, crops=96,
        lighting='Approved middle condition: balanced lighting, original colors', methods={})
    for method in METHODS:
        rows = [r for r in decisions if r['method'] == method]
        stats = {status: sum(r['outcome'] == status for r in rows) for status in ('correct', 'wrong', 'tied', 'unavailable')}
        stats['correct_percent'] = 100 * stats['correct'] / 96
        stats['same_family_wrong_size'] = sum(r['same_family_wrong_size'] for r in rows)
        if method != 'dino':
            changes_here = [r for r in changes if r['method'] == method]
            stats.update(errors_fixed=sum(r['change'] == 'fixed_error' for r in changes_here),
                errors_introduced=sum(r['change'] == 'introduced_error' for r in changes_here),
                previously_correct_lost=sum(r['baseline_outcome'] == 'correct' and r['alternative_outcome'] != 'correct' for r in changes_here),
                shortlist_hits=sum(r['truth_in_top5'] is True for r in rows),
                shortlist_misses=sum(r['truth_in_top5'] is False for r in rows),
                shortlist_unavailable=sum(r['truth_in_top5'] is None for r in rows))
        summary['methods'][method] = stats
    return decisions, changes, per_object, summary


def image_link(file):
    return Path(os.path.relpath(local(file), OUT)).as_posix()


def readable(target):
    names = {'Gear_Large': 'Large gear', 'M16_Hex_Nut': 'M16 nut', 'Waterproof_Male': 'Waterproof',
             'USB_Male': 'USB', 'DSUB_Male': 'D-sub', 'BNC_Male': 'BNC'}
    if target in names:
        return names[target]
    if target.startswith('RGOCG'):
        return target.split('_')[-1].replace('mm', ' mm') + ' round pin'
    if target.startswith('KET'):
        return target.split('_')[-1].replace('mm', ' mm') + ' rectangular pin'
    return target


def figures(protocol, decisions, changes, per_object, summary, inputs, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    chosen = ('dino', 'clip_full_dino', 'clip_shape_dino')
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2]})
    colors = ['#17527d', '#dc7b32', '#4d926e']
    vals = [summary['methods'][m]['correct'] for m in chosen]
    left.bar(range(3), vals, color=colors)
    left.set_ylim(0, 105); left.set_ylabel('Correct CAD-to-crop matches / 96')
    left.set_xticks(range(3), ['DINO alone', 'CLIP full\n→ DINO', 'CLIP shape text\n→ DINO'])
    for i, value in enumerate(vals):
        left.text(i, value + 2, str(value), ha='center', fontsize=14)
    xs = np.arange(12)
    for i, method in enumerate(chosen):
        right.barh(xs + (i - 1) * .24, [r[method + '_correct'] for r in per_object], height=.24, color=colors[i], label=DISPLAY[method])
    right.set_yticks(xs, [readable(r['target']) for r in per_object]); right.invert_yaxis()
    right.set_xlim(0, 8.6); right.set_xticks(range(9)); right.set_xlabel('Correct matches / eight scenes'); right.legend(loc='lower right', fontsize=8)
    fig.suptitle('CLIP retrieves five crops; unchanged DINO scores select the final crop', fontsize=15)
    fig.text(.5, .01, 'Same eight N layouts · 12 products · balanced lighting/original colors · frozen pretrained weights', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .03, 1, .94]); fig.savefig(OUT / 'comparison.png', dpi=170); fig.savefig(OUT / 'comparison.pdf'); plt.close(fig)
    crops = {(r['state'], r['proposal_id']): r for r in inputs if r['kind'] == 'crop'}
    cad = {(r['target'], r['view_index']): r for r in inputs if r['kind'] == 'CAD'}
    pair_index = {(r['state'], r['target'], r['proposal_id']): r for r in pairs}
    decision_index = {(r['state'], r['target'], r['method']): r for r in decisions}
    changed = [r for r in changes if r['method'] == 'clip_full_dino' and r['baseline_ids'] != r['alternative_ids']]
    if not changed:
        changed = [dict(state=protocol['scenes'][0], target=protocol['parts'][0], change='No changed primary decisions')]
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    def put(sheet, file, box):
        with Image.open(local(file)).convert('RGB') as image:
            image.thumbnail((box[2], box[3])); sheet.paste(image, (box[0] + (box[2] - image.width) // 2, box[1] + (box[3] - image.height) // 2))
    sheet = Image.new('RGB', (1240, 260 * len(changed) + 50), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), 'Every changed primary decision | target CAD / correct crop / DINO / CLIP → DINO', fill='black', font=font)
    for i, change in enumerate(changed):
        y = 50 + i * 260; state, target = change['state'], change['target']
        base = decision_index[state, target, 'dino']; hybrid = decision_index[state, target, 'clip_full_dino']
        draw.text((12, y), state + ' · ' + readable(target) + ' · ' + change['change'], fill='black', font=font)
        truth = base['truth_proposal_id']; truth_pair = pair_index[state, target, truth]
        view = truth_pair['dino_selected_CAD_view'] or 1
        put(sheet, cad[target, view]['file'], (10, y + 30, 290, 190))
        put(sheet, crops[state, truth]['file'], (320, y + 30, 290, 190))
        for x, decision in ((630, base), (940, hybrid)):
            if len(decision['selected_ids']) == 1:
                put(sheet, crops[state, decision['selected_ids'][0]]['file'], (x, y + 30, 290, 190))
            draw.text((x, y + 225), decision['outcome'] + ': ' + readable(decision['predicted_product'] or decision['status']), fill='black', font=font)
    sheet.save(OUT / 'changed_cases.png')
    # A concrete primary task is also shown as the full inference pipeline.
    case = changed[0]; state, target = case['state'], case['target']
    hybrid = decision_index[state, target, 'clip_full_dino']; truth = hybrid['truth_proposal_id']
    canvas = Image.new('RGB', (1620, 360), 'white'); d = ImageDraw.Draw(canvas)
    d.text((12, 10), state + ' · ' + readable(target) + ' | CLIP top five → DINO final selection', fill='black', font=font)
    p = pair_index[state, target, truth]
    put(canvas, cad[target, p['clip_full']['selected_CLIP_CAD_view'] or 1]['file'], (10, 70, 220, 200))
    d.text((20, 300), 'Supplied CAD', fill='black', font=font)
    for i, proposal in enumerate(hybrid['shortlist_ids']):
        x = 280 + i * 230; put(canvas, crops[state, proposal]['file'], (x, 70, 210, 200))
        pair = pair_index[state, target, proposal]
        caption = f'p{proposal:03d} | DINO {pair["dino_score"]:.3f}' if pair['dino_score'] is not None else f'p{proposal:03d} unavailable'
        d.text((x, 300), caption, fill='#167543' if proposal == truth else 'black', font=font)
        if proposal in hybrid['selected_ids']:
            d.rectangle((x - 2, 60, x + 214, 330), outline='#dc7b32', width=4)
    canvas.save(OUT / 'pipeline_case.png')


def gallery(protocol, decisions, pairs, inputs, descriptions, labels):
    by_input = {r['input_id']: r for r in inputs}
    crops = {(r['state'], r['proposal_id']): r for r in inputs if r['kind'] == 'crop'}
    cad = {(r['target'], r['view_index']): r for r in inputs if r['kind'] == 'CAD'}
    desc = {(r['input_id'], r['mode']): r for r in descriptions}
    di = {(r['state'], r['target'], r['method']): r for r in decisions}
    task_pairs = {}
    for pair in pairs:
        task_pairs.setdefault((pair['state'], pair['target']), []).append(pair)
    esc = html.escape
    def image_text(record, label):
        blocks = [f'<b>{esc(label)}</b><br><img loading="lazy" src="{esc(image_link(record["file"]))}">']
        for mode in ('keywords', 'phrases'):
            response = desc[record['input_id'], mode]
            blocks.append('<details><summary>' + mode + ' · ' + esc(response['status']) + '</summary><pre>' + esc(response['raw_output']) + '</pre></details>')
        return ''.join(blocks)
    chunks = ['<!doctype html><meta charset="utf-8"><title>CLIP retrieval + DINO selection evidence</title>',
        '<style>body{font:15px Arial;margin:26px;color:#192b3b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d9e1e7;padding:8px;vertical-align:top}img{width:150px;height:150px;object-fit:contain;background:#eee}pre{white-space:pre-wrap;font:12px monospace;max-width:330px}summary{cursor:pointer;padding:7px}details.case{margin:12px 0;border:1px solid #bac6cf;padding:6px}.gt{background:#edf8ef}.selected{outline:2px solid #d9823c}small{color:#526270}</style>',
        '<h1>CLIP retrieves five crops; DINO makes the final selection</h1><p>Every task and all twelve candidates are retained. Labels below are post-hoc audit information. Expand candidate descriptions for exact raw Qwen responses.</p>',
        '<p><a href="README.md">Report</a> · <a href="decisions.csv">All decisions</a> · <a href="paired_changes.csv">Paired changes</a></p>']
    for state in protocol['scenes']:
        for target in protocol['parts']:
            base, hybrid = di[state, target, 'dino'], di[state, target, 'clip_full_dino']
            truth = base['truth_proposal_id']
            title = f'{state} · {readable(target)} | DINO: {base["outcome"]} ({base["predicted_product"]}) | CLIP→DINO: {hybrid["outcome"]} ({hybrid["predicted_product"]}) | true crop p{truth:03d}'
            chunks.append('<details class="case"><summary>' + esc(title) + '</summary>')
            chunks.append('<p>' + ' · '.join(esc(DISPLAY[m] + ': ' + di[state, target, m]['outcome'] + ' ' + str(di[state, target, m]['selected_ids'])) for m in METHODS) + '</p>')
            chunks.append('<table><tr><th>Crop and exact Qwen text</th><th>DINO corresponding CAD view</th><th>CLIP full best CAD view</th><th>Scores / shortlist</th></tr>')
            for pair in sorted(task_pairs[state, target], key=lambda r: -(r['clip_full']['score'] or -1e9)):
                proposal = pair['proposal_id']; crop = crops[state, proposal]
                classes = ('gt ' if proposal == truth else '') + ('selected' if proposal in hybrid['selected_ids'] else '')
                chunks.append('<tr class="' + classes + '"><td>' + image_text(crop, f'p{proposal:03d} · audit: {readable(labels[state][proposal])}') + '</td>')
                for view, label in ((pair['dino_selected_CAD_view'], 'DINO CAD'), (pair['clip_full']['selected_CLIP_CAD_view'], 'CLIP full CAD')):
                    chunks.append('<td>' + (image_text(cad[target, view], f'{label} V{view:02d}') if view else 'unavailable') + '</td>')
                score_data = dict(DINO=pair['dino_score'], DINO_global=pair['dino_global_score'], DINO_patch=pair['dino_patch_score'],
                    CLIP_full=pair['clip_full']['score'], CLIP_components=pair['clip_full']['components'],
                    CLIP_shape=pair['clip_shape']['score'], full_top5=proposal in hybrid['shortlist_ids'],
                    shape_top5=proposal in di[state, target, 'clip_shape_dino']['shortlist_ids'])
                chunks.append('<td><pre>' + esc(json.dumps(score_data, indent=2)) + '</pre></td></tr>')
            chunks.append('</table></details>')
    (OUT / 'index.html').write_text('\n'.join(chunks), encoding='utf-8')


def report_markdown(protocol, summary, per_object, changes):
    lines = ['# CLIP top-five retrieval + DINO selection', '',
        'Eight retained N00–N07 simulation layouts contain twelve products each. All images use the approved middle lighting condition with original colors preserved. Qwen3.5-9B generates attributes and descriptive phrases; frozen CLIP ViT-B/32 retrieves five crops, then the unchanged DINOv2 global/foreground-patch score makes the final selection. DINO alone ranks all twelve crops.', '',
        '| Method | Correct /96 | Wrong | Tied | Unavailable | Fixed DINO errors | Introduced errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        r = summary['methods'][method]
        lines.append(f'| {DISPLAY[method]} | {r["correct"]} | {r["wrong"]} | {r["tied"]} | {r["unavailable"]} | {r.get("errors_fixed", "—")} | {r.get("errors_introduced", "—")} |')
    lines += ['', 'Full-attribute CLIP→DINO is the primary comparison. SHAPE-text is a separately frozen control and retains RGB image features. CLIP-only selection is diagnostic.', '',
        '| Product | DINO /8 | CLIP full→DINO /8 | CLIP shape→DINO /8 | Full recall@5 /8 | Shape recall@5 /8 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in per_object:
        lines.append(f'| {readable(r["target"])} | {r["dino_correct"]} | {r["clip_full_dino_correct"]} | {r["clip_shape_dino_correct"]} | {r["clip_full_top5_hits"]} | {r["clip_shape_top5_hits"]} |')
    lines += ['', '## Errors fixed or introduced', '']
    primary = [r for r in changes if r['method'] == 'clip_full_dino' and r['change'] in ('fixed_error', 'introduced_error')]
    if not primary:
        lines.append('No wrong→correct or correct→wrong primary changes.')
    for r in primary:
        lines.append(f'- **{r["state"]} · {readable(r["target"])} · {r["change"]}:** {r["baseline_product"]} → {r["alternative_product"]}; true crop in top five: {r["truth_in_top5"]}.')
    lines += ['', '## Interpretation and limits', '',
        'The hybrid is a filter, not a new DINO score: if CLIP retains the unique DINO winner, the final answer must remain identical. An error can be rescued only when CLIP removes the wrong DINO winner while retaining a better surviving match. A correct match is lost when retrieval removes it, or evidence is unavailable. This invariant is checked independently for every task and both text conditions.', '',
        'The four CLIP terms are image–image, text–text, crop-image/CAD-text and crop-text/CAD-image cosine similarities, equally averaged per CAD view and maximized over all twenty views. The five highest distinct scene crops are shortlisted. Ties at the fifth/sixth boundary are unavailable; final ties remain unresolved. No per-case fallback, score fusion, assignment, weight fitting or tuning is used.', '',
        'CLIP and Qwen receive the same native raw RGB files. DINO uses exact isolated-CAD silhouettes and depth-plane foreground masks with white, aspect-preserving 224-pixel preprocessing. Both primary methods reuse the exact same DINO candidate scores and CAD-view mappings; adding CLIP does not change this preprocessing.', '',
        'These are reused known-CAD simulator scenes with one instance of each product and previously accepted depth-localizer rectangles. They do not measure localization quality, absent targets, duplicate instances, unseen CADs, real-image robustness, robot execution or assembly success. The relit twelve-object condition differs from the earlier eight-object J study; its 58/64 result is not this baseline. Scores are uncalibrated and all pretrained weights remain unchanged.', '',
        'Evidence: [all 96 tasks and candidate images/text](index.html), [per-object CSV](per_object.csv), [all decisions](decisions.csv), [paired changes](paired_changes.csv), [within-family errors](within_family.csv), [verification](verification.json).', '',
        '![Comparison](comparison.png)', '', '![Every changed primary decision](changed_cases.png)', '', '![One full pipeline example](pipeline_case.png)']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run():
    assert not (OUT / 'completion.json').exists(), 'Preserve completed report evidence.'
    protocol, verification = verify_completions()
    inputs, descriptions = read(HERE / 'inputs.json'), read(HERE / 'descriptions.json')
    pairs = read(HERE / 'clip/candidate_scores.json'); views = read(HERE / 'clip/label_free_view_scores.json')
    shortlists = read(HERE / 'clip/shortlist.json'); rankings = read(HERE / 'clip/label_free_rankings.json')
    dino = read(HERE / 'dino/scores.json')
    verification.update(verify_qwen(inputs, descriptions))
    arrays, feature_check = verify_clip_features(inputs); verification.update(feature_check)
    verification.update(verify_dino_features(inputs, dino))
    retention, rank_check = verify_rankings(inputs, arrays, pairs, views, shortlists, rankings, dino)
    verification.update(rank_check)
    OUT.mkdir(exist_ok=True)
    save(OUT / 'label_free_verification.json', verification)
    # Only this line and later access audit identities; all ranking replay is done.
    labels, label_sources = read_labels_after_verification(inputs, protocol)
    verification.update(identity_labels_opened=True, identity_labels_opened_after_complete_label_free_replay=True,
        audit_sources=label_sources, audit_ROI_mapping_exact=True)
    decisions, changes, per_object, summary = evaluate(protocol, rankings, shortlists, labels)
    summary['verification'] = 'passed'
    for name, rows in (('decisions', decisions), ('paired_changes', changes), ('per_object', per_object),
                       ('winner_retention', retention)):
        save(OUT / (name + '.json'), rows); csv_file(OUT / (name + '.csv'), rows)
    within = [dict(method=r['method'], state=r['state'], target=r['target'], selected_product=r['predicted_product'],
        within_family_wrong_size=r['same_family_wrong_size']) for r in decisions if r['outcome'] == 'wrong']
    save(OUT / 'within_family.json', within)
    if within:
        csv_file(OUT / 'within_family.csv', within)
    else:
        (OUT / 'within_family.csv').write_text('method,state,target,selected_product,within_family_wrong_size\n', encoding='utf-8')
    save(OUT / 'summary.json', summary); save(OUT / 'verification.json', verification)
    figures(protocol, decisions, changes, per_object, summary, inputs, pairs)
    gallery(protocol, decisions, pairs, inputs, descriptions, labels)
    report_markdown(protocol, summary, per_object, changes)
    save(OUT / 'completion.json', dict(passed=True, tasks=96, methods=list(METHODS),
        label_free_verified_before_labels=True, report_source_sha256=digest(Path(__file__)),
        artifact_hashes={p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != 'completion.json'}))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
