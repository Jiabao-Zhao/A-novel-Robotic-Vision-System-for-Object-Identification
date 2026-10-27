"""CLIP retrieves scene crops; unchanged DINO scores choose within the shortlist."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIELDS = ('COLOR', 'SHAPE', 'MATERIAL')
MODES = ('keywords', 'phrases')
METHODS = ('dino', 'clip_full_dino', 'clip_shape_dino',
           'clip_full_only', 'clip_shape_only')
OUTPUT = HERE / 'clip'
TOP_K = 5
TOLERANCE = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_path(value):
    text = str(value).replace('\\', '/')
    if len(text) > 1 and text[1] == ':' and sys.platform != 'win32':
        text = '/mnt/' + text[0].lower() + text[2:]
    if text.startswith('/mnt/') and sys.platform == 'win32':
        text = text[5].upper() + ':' + text[6:]
    path = Path(text)
    return path if path.is_absolute() else REPO / path


def verify_sources(protocol):
    protected = protocol.get('protected', protocol.get('immutable_inputs', {}))
    assert protected, 'A frozen source-hash map is required.'
    for file, expected in protected.items():
        assert digest(source_path(file)) == expected, 'Frozen source changed: ' + file


def split_content_ids(ids, capacity=75):
    """Keep every content token in order; no decoding or textual rewriting."""
    assert capacity > 0
    values = list(ids)
    return [values[first:first + capacity] for first in range(0, len(values), capacity)]


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 1e-12:
        return None
    return value / norm


def pool_unit(vectors):
    """Equal weighting after normalizing each item, followed by normalization."""
    if not vectors or any(value is None for value in vectors):
        return None
    normalized = [unit(value) for value in vectors]
    if any(value is None for value in normalized):
        return None
    return unit(np.mean(normalized, axis=0))


def text_plan(inputs, descriptions, tokenizer, capacity):
    indexed = {}
    for row in descriptions:
        key = (row['input_id'], row['mode'])
        assert key not in indexed, 'Duplicate description: ' + str(key)
        indexed[key] = row
    expected = {(row['input_id'], mode) for row in inputs for mode in MODES}
    assert set(indexed) <= expected, 'Description outside the frozen inputs/modes.'
    fields, chunks = [], []
    for image in inputs:
        for mode in MODES:
            description = indexed.get((image['input_id'], mode))
            for field in FIELDS:
                record = dict(input_id=image['input_id'], mode=mode, field=field,
                              status='unavailable', unavailable_reason=None,
                              raw_text=None, content_token_ids=[], chunk_indices=[])
                if description is None:
                    record['unavailable_reason'] = 'missing_description'
                elif description.get('status') != 'ok':
                    record['unavailable_reason'] = 'description_unavailable'
                else:
                    value = description.get('fields', {}).get(field)
                    record['raw_text'] = value
                    if not isinstance(value, str) or not value.strip():
                        record['unavailable_reason'] = 'missing_or_empty_string_field'
                    else:
                        ids = tokenizer.encode(value, add_special_tokens=False, truncation=False)
                        pieces = split_content_ids(ids, capacity)
                        if not pieces:
                            record['unavailable_reason'] = 'empty_tokenized_field'
                        else:
                            record.update(status='ok', content_token_ids=ids)
                            assert [token for piece in pieces for token in piece] == ids
                            for piece in pieces:
                                tokens = tokenizer.build_inputs_with_special_tokens(piece)
                                record['chunk_indices'].append(len(chunks))
                                chunks.append(dict(field_index=len(fields),
                                    input_id=image['input_id'], mode=mode, field=field,
                                    content_token_ids=piece, input_ids=tokens))
                fields.append(record)
    return fields, chunks


def load_helper():
    file = REPO / 'scripts/clip_encoders.py'
    spec = importlib.util.spec_from_file_location('frozen_clip_encoder_helpers', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(protocol, inputs, descriptions):
    import torch
    from PIL import Image

    config = protocol['clip']
    assert config['device'] == 'cuda' and config['dtype'] == 'float32'
    assert config['fields'] == list(FIELDS) and config['modes'] == list(MODES)
    assert config['text_capacity'] == 77 and config['max_content_tokens'] == 75
    assert config['seed'] == 0 and config['tie_tolerance'] == TOLERANCE
    assert torch.cuda.is_available(), 'The reserved CUDA runtime is required.'
    helper = load_helper()
    assert source_path(config['asset_config']).resolve() == (helper.HERE / 'encoder_assets.json').resolve()
    assert config['asset_method'] == 'clip'
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    asset = helper.asset_record('clip')
    model, processor, load_metadata = helper.load_model(asset, 'cuda')
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert model.config.text_config.max_position_embeddings == 77
    assert processor.tokenizer.num_special_tokens_to_add(pair=False) == 2
    batch_size = config['batch_size']
    assert batch_size > 0
    fields, chunks = text_plan(inputs, descriptions, processor.tokenizer, 75)
    assert all(0 < len(row['input_ids']) <= 77 for row in chunks)
    images, image_batches, chunk_batches = [], [], []
    stage = perf_counter()
    with torch.inference_mode():
        for first in range(0, len(inputs), batch_size):
            before = perf_counter()
            frames = []
            for row in inputs[first:first + batch_size]:
                with Image.open(source_path(row['file'])) as frame:
                    frames.append(frame.convert('RGB'))
            tensors = processor(images=frames, return_tensors='pt').to('cuda')
            raw = helper.feature_tensor(model.get_image_features(**tensors))
            torch.cuda.synchronize()
            images.append(raw.float().cpu().numpy())
            image_batches.append(dict(first=first, count=len(frames), seconds=perf_counter() - before))
            print(f'CLIP RGB images {min(first + batch_size, len(inputs))}/{len(inputs)}', flush=True)
        raw_parts = []
        for first in range(0, len(chunks), batch_size):
            before = perf_counter()
            sequence = [row['input_ids'] for row in chunks[first:first + batch_size]]
            tensors = processor.tokenizer.pad(dict(input_ids=sequence), padding=True,
                return_attention_mask=True, return_tensors='pt').to('cuda')
            for row, ids in zip(tensors['input_ids'].tolist(), sequence):
                assert row[:len(ids)] == ids, 'Padding changed content token IDs.'
            raw = helper.feature_tensor(model.get_text_features(**tensors))
            torch.cuda.synchronize()
            raw_parts.append(raw.float().cpu().numpy())
            chunk_batches.append(dict(first=first, count=len(sequence), seconds=perf_counter() - before))
        image_raw = np.concatenate(images)
        width = image_raw.shape[1]
        chunk_raw = np.concatenate(raw_parts) if raw_parts else np.empty((0, width), dtype=np.float32)
    assert image_raw.shape == (len(inputs), width) and chunk_raw.shape == (len(chunks), width)
    assert np.isfinite(image_raw).all() and np.isfinite(chunk_raw).all()
    image_unit = np.stack([unit(row) for row in image_raw])
    chunk_unit = np.stack([unit(row) for row in chunk_raw]) if len(chunk_raw) else np.empty((0, width))
    field_unit = np.zeros((len(fields), width), dtype=np.float64)
    for index, row in enumerate(fields):
        if row['status'] != 'ok':
            continue
        value = pool_unit([chunk_unit[i] for i in row['chunk_indices']])
        if value is None:
            row.update(status='unavailable', unavailable_reason='degenerate_chunk_pool')
        else:
            field_unit[index] = value
    field_index = {(r['input_id'], r['mode'], r['field']): i for i, r in enumerate(fields)}
    pools = {name: np.zeros((len(inputs), width)) for name in ('full', 'shape')}
    pool_records = []
    for image_index, row in enumerate(inputs):
        for name, included in (('full', FIELDS), ('shape', ('SHAPE',))):
            ids = [field_index[row['input_id'], mode, field] for mode in MODES for field in included]
            problems = [dict(mode=fields[i]['mode'], field=fields[i]['field'],
                reason=fields[i]['unavailable_reason']) for i in ids if fields[i]['status'] != 'ok']
            value = None if problems else pool_unit([field_unit[i] for i in ids])
            if value is None and not problems:
                problems.append(dict(reason='degenerate_field_pool'))
            if value is not None:
                pools[name][image_index] = value
            pool_records.append(dict(input_id=row['input_id'], pool=name,
                image_index=image_index, field_indices=ids,
                status='ok' if value is not None else 'unavailable', unavailable_reasons=problems))
    arrays = dict(image_raw=image_raw, image_normalized=image_unit,
        chunk_raw=chunk_raw, chunk_normalized=chunk_unit, field_normalized=field_unit,
        field_available=np.array([row['status'] == 'ok' for row in fields]),
        text_full=pools['full'], text_shape=pools['shape'],
        text_full_available=np.array([row['status'] == 'ok' for row in pool_records if row['pool'] == 'full']),
        text_shape_available=np.array([row['status'] == 'ok' for row in pool_records if row['pool'] == 'shape']))
    metadata = dict(**load_metadata, device='cuda', dtype='float32', seed=config['seed'],
        torch_version=torch.__version__, transformers_version=helper.transformers.__version__,
        cuda_device=torch.cuda.get_device_name(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        model_asset=asset, model_weights_frozen=True, training_performed=False,
        processed_images='Exact raw input RGB converted by the cached standard CLIPProcessor; no DINO foreground mask.',
        preprocessing=processor.image_processor.to_dict(), image_batches=image_batches,
        text_batches=chunk_batches, embedding_width=width, encoding_seconds=perf_counter() - stage,
        text_capacity=77, content_capacity=75, text_truncation=False,
        fields_with_multiple_chunks=sum(len(row['chunk_indices']) > 1 for row in fields),
        field_token_lengths=[len(row['content_token_ids']) for row in fields],
        total_chunks=len(chunks), unavailable_fields=sum(row['status'] != 'ok' for row in fields),
        pool_status_counts=dict(Counter(row['pool'] + ':' + row['status'] for row in pool_records)),
        normalization='Raw FP32 model vectors preserved; cosine/pooling in NumPy float64; equal normalized chunk mean, normalized field mean.',
        labels_or_simulator_poses_read=False)
    mapping = dict(images=[dict(image_index=i, **row) for i, row in enumerate(inputs)],
                   fields=fields, chunks=chunks, pools=pool_records)
    del model
    torch.cuda.empty_cache()
    return arrays, mapping, metadata


def validate_inputs(inputs, require_trial_size=True, verify_files=True):
    """Read only frozen RGB inputs and their experimental CAD/crop mapping."""
    by_id = {row['input_id']: row for row in inputs}
    assert len(by_id) == len(inputs), 'Duplicate image input IDs.'
    cad, crops = {}, {}
    for row in inputs:
        assert row['kind'] in ('CAD', 'crop')
        if verify_files:
            assert digest(source_path(row['file'])) == row['sha256'], row['file']
        if row['kind'] == 'CAD':
            cad.setdefault(row['target'], []).append(row)
        else:
            crops.setdefault(row['state'], []).append(row)
    for views in cad.values():
        assert len(views) == len({row['view_index'] for row in views})
    for instances in crops.values():
        assert len(instances) == len({row['proposal_id'] for row in instances})
    if require_trial_size:
        assert len(cad) == 12 and len(crops) == 8
        assert all(len(views) == 20 for views in cad.values())
        assert all(len(instances) == 12 for instances in crops.values())
        assert len(inputs) == 336
    return by_id, cad, crops


def dino_index(rows, cad, crops, method=None):
    methods = {row.get('method') for row in rows}
    if method is None:
        assert len(methods) == 1, 'Declare the existing DINO score method explicitly.'
        method = next(iter(methods))
    selected = [row for row in rows if row.get('method') == method]
    assert selected, ('DINO method absent', method, methods)
    indexed = {}
    for row in selected:
        key = (row['state'], row['target'], row['proposal_id'])
        assert key not in indexed, ('Duplicate DINO pair', key)
        indexed[key] = row
    expected = {(state, target, crop['proposal_id'])
                for state, instances in crops.items()
                for target in cad for crop in instances}
    assert set(indexed) == expected, 'DINO must score all scene crops for every CAD.'
    for (state, target, _), row in indexed.items():
        score = row.get('score')
        if score is not None:
            assert np.isfinite(score)
            assert row['selected_CAD_view'] in {view['view_index'] for view in cad[target]}
    return indexed, method


def rank_scores(rows, tolerance=TOLERANCE, missing_status='unavailable_incomplete_candidates'):
    """Keep ties unresolved and never drop a missing competing candidate."""
    assert rows and len(rows) == len({row['proposal_id'] for row in rows})
    available = sorted([row for row in rows if row['score_available']],
                       key=lambda row: (-row['score'], row['proposal_id']))
    missing = sorted(row['proposal_id'] for row in rows if not row['score_available'])
    provisional, score, gap = [], None, None
    if available:
        score = available[0]['score']
        provisional = [row['proposal_id'] for row in available
                       if abs(row['score'] - score) <= tolerance]
        gap = 0. if len(provisional) > 1 else (
            score - available[1]['score'] if len(available) > 1 else None)
    complete = not missing
    top = provisional if complete else []
    return dict(top_ids=top, top_proposal_ids=top,
                top_score=score if complete else None, gap=gap if complete else None,
                eligible_candidates=len(available), unavailable_ids=missing,
                status=missing_status if missing else ('tie' if len(top) > 1 else 'ok'),
                complete=complete, provisional_top_ids=provisional,
                provisional_top_score=score, provisional_gap=gap)


def make_shortlist(rows, k=TOP_K, tolerance=TOLERANCE):
    """A cutoff tie makes top-k ambiguous; sorting only orders exported evidence."""
    assert len(rows) >= k and len(rows) == len({row['proposal_id'] for row in rows})
    available = sorted([row for row in rows if row['score_available']],
                       key=lambda row: (-row['score'], row['proposal_id']))
    missing = sorted(row['proposal_id'] for row in rows if not row['score_available'])
    provisional = [row['proposal_id'] for row in available[:k]]
    if missing:
        status = 'unavailable_incomplete_CLIP_evidence'
    elif len(available) > k and abs(available[k - 1]['score'] - available[k]['score']) <= tolerance:
        status = 'unavailable_ambiguous_top5'
    else:
        status = 'ok'
    cutoff = available[k - 1]['score'] if len(available) >= k else None
    boundary = [] if cutoff is None else [row['proposal_id'] for row in available
                                         if abs(row['score'] - cutoff) <= tolerance]
    return dict(status=status, complete=status == 'ok', k=k,
                candidate_ids=provisional if status == 'ok' else [],
                provisional_candidate_ids=provisional, unavailable_ids=missing,
                cutoff_score=cutoff, cutoff_tie_ids=boundary,
                ranked_candidate_ids=[row['proposal_id'] for row in available],
                ranked_scores=[row['score'] for row in available])


def view_components(crop, cad, arrays, pool):
    image_crop, image_cad = arrays['image_normalized'][crop], arrays['image_normalized'][cad]
    valid_crop = bool(arrays['text_' + pool + '_available'][crop])
    valid_cad = bool(arrays['text_' + pool + '_available'][cad])
    values = dict(image_image=float(image_crop @ image_cad),
        text_text=None, crop_image_to_CAD_text=None, crop_text_to_CAD_image=None)
    if valid_cad:
        values['crop_image_to_CAD_text'] = float(image_crop @ arrays['text_' + pool][cad])
    if valid_crop:
        values['crop_text_to_CAD_image'] = float(arrays['text_' + pool][crop] @ image_cad)
    if valid_crop and valid_cad:
        values['text_text'] = float(arrays['text_' + pool][crop] @ arrays['text_' + pool][cad])
    score = float(np.mean(list(values.values()))) if valid_crop and valid_cad else None
    assert score is None or np.isfinite(score)
    return values, score


def score_all(inputs, arrays, dino_rows, method=None, require_trial_size=True):
    """Score all CAD views, shortlist by CLIP, then reuse exact DINO pair scores."""
    _, cad, crops = validate_inputs(inputs, require_trial_size, verify_files=False)
    indexed, dino_method = dino_index(dino_rows, cad, crops, method)
    image_index = {row['input_id']: i for i, row in enumerate(inputs)}
    view_rows, pair_rows, shortlists, rankings = [], [], [], []
    for state, instances in sorted(crops.items()):
        for target, views in sorted(cad.items()):
            task_id = state + ':' + target
            task_pairs = []
            for crop in sorted(instances, key=lambda row: row['proposal_id']):
                di = indexed[state, target, crop['proposal_id']]
                dino_score = di.get('score')
                dino_available = dino_score is not None and di.get('score_available', True)
                pair = dict(task_id=task_id, state=state, target=target,
                    proposal_id=crop['proposal_id'], crop_input_id=crop['input_id'],
                    dino_score=dino_score, dino_score_available=bool(dino_available),
                    dino_selected_CAD_view=di.get('selected_CAD_view'),
                    dino_global_score=di.get('global_score'), dino_patch_score=di.get('patch_score'),
                    dino_source_method=dino_method)
                for pool in ('full', 'shape'):
                    available, unavailable = [], []
                    for view in sorted(views, key=lambda row: row['view_index']):
                        component, score = view_components(image_index[crop['input_id']],
                                                         image_index[view['input_id']], arrays, pool)
                        row = dict(task_id=task_id, state=state, target=target,
                            proposal_id=crop['proposal_id'], crop_input_id=crop['input_id'],
                            CAD_input_id=view['input_id'], CAD_view_index=view['view_index'],
                            pool=pool, score=score, score_available=score is not None,
                            components=component)
                        view_rows.append(row)
                        (available if score is not None else unavailable).append(row)
                    available.sort(key=lambda row: (-row['score'], row['CAD_view_index']))
                    best = available[0] if available else None
                    ties = [] if best is None else [row['CAD_view_index'] for row in available
                                                  if abs(row['score'] - best['score']) <= TOLERANCE]
                    # Strict availability is independent of the value of partial maximum scores.
                    pair['clip_' + pool] = dict(
                        score=best['score'] if best else None,
                        score_available=bool(best is not None and not unavailable),
                        status='ok' if best is not None and not unavailable else 'unavailable_CAD_or_crop_text',
                        selected_CLIP_CAD_view=best['CAD_view_index'] if best else None,
                        selected_CLIP_CAD_input_id=best['CAD_input_id'] if best else None,
                        selected_CLIP_view_tie_ids=ties,
                        components=best['components'] if best else None,
                        required_views=len(views), available_views=len(available),
                        unavailable_CAD_views=[row['CAD_view_index'] for row in unavailable])
                pair_rows.append(pair)
                task_pairs.append(pair)
            dino_rank = rank_scores([dict(proposal_id=row['proposal_id'], score=row['dino_score'],
                score_available=row['dino_score_available']) for row in task_pairs])
            rankings.append(dict(method='dino', task_id=task_id, state=state, target=target,
                shortlist_ids=None, selected_DINO_CAD_views={str(row['proposal_id']): row['dino_selected_CAD_view']
                    for row in task_pairs if row['proposal_id'] in dino_rank['top_ids']}, **dino_rank))
            for pool in ('full', 'shape'):
                clip_rows = [dict(proposal_id=row['proposal_id'], score=row['clip_' + pool]['score'],
                    score_available=row['clip_' + pool]['score_available']) for row in task_pairs]
                shortlist = make_shortlist(clip_rows)
                shortlists.append(dict(task_id=task_id, state=state, target=target,
                    method='clip_' + pool, **shortlist))
                clip_rank = rank_scores(clip_rows, missing_status='unavailable_incomplete_CLIP_evidence')
                rankings.append(dict(method='clip_' + pool + '_only', task_id=task_id,
                    state=state, target=target, shortlist_ids=shortlist['candidate_ids'],
                    selected_CLIP_CAD_views={str(row['proposal_id']): row['clip_' + pool]['selected_CLIP_CAD_view']
                        for row in task_pairs if row['proposal_id'] in clip_rank['top_ids']}, **clip_rank))
                if shortlist['status'] == 'ok':
                    selected = [row for row in task_pairs if row['proposal_id'] in shortlist['candidate_ids']]
                    assert len(selected) == TOP_K
                    # No CLIP/DINO fusion and no re-selection of DINO reference views here.
                    final_rank = rank_scores([dict(proposal_id=row['proposal_id'], score=row['dino_score'],
                        score_available=row['dino_score_available']) for row in selected],
                        missing_status='unavailable_incomplete_DINO_shortlist')
                else:
                    final_rank = dict(top_ids=[], top_proposal_ids=[], top_score=None, gap=None,
                        eligible_candidates=0, unavailable_ids=shortlist['unavailable_ids'],
                        status=shortlist['status'], complete=False, provisional_top_ids=[],
                        provisional_top_score=None, provisional_gap=None)
                rankings.append(dict(method='clip_' + pool + '_dino', task_id=task_id,
                    state=state, target=target, shortlist_ids=shortlist['candidate_ids'],
                    selected_DINO_CAD_views={str(row['proposal_id']): row['dino_selected_CAD_view']
                        for row in task_pairs if row['proposal_id'] in final_rank['top_ids']},
                    selected_CLIP_CAD_views={str(row['proposal_id']): row['clip_' + pool]['selected_CLIP_CAD_view']
                        for row in task_pairs if row['proposal_id'] in final_rank['top_ids']}, **final_rank))
    if require_trial_size:
        assert len(view_rows) == 8 * 12 * 12 * 20 * 2
        assert len(pair_rows) == 8 * 12 * 12
        assert len(shortlists) == 8 * 12 * 2
        assert len(rankings) == 8 * 12 * len(METHODS)
    return view_rows, pair_rows, shortlists, rankings


def run():
    assert not (OUTPUT / 'completion.json').exists(), 'Preserve completed frozen evidence.'
    protocol = read(HERE / 'protocol.json')
    verify_sources(protocol)
    inputs, descriptions = read(HERE / 'inputs.json'), read(HERE / 'descriptions.json')
    validate_inputs(inputs)
    dino_file = HERE / 'dino' / 'scores.json'
    dino_rows = read(dino_file)
    source_files = [HERE / name for name in ('protocol.json', 'inputs.json', 'descriptions.json', 'run_clip.py')]
    source_files += [dino_file, HERE / 'dino' / 'completion.json']
    before = {str(path.relative_to(REPO)).replace('\\', '/'): digest(path) for path in source_files}
    start = perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    arrays, mapping, metadata = encode(protocol, inputs, descriptions)
    np.savez_compressed(OUTPUT / 'features.npz', **arrays)
    save(OUTPUT / 'tensor_indices.json', mapping)
    save(OUTPUT / 'encoding.json', metadata)
    view_scores, pairs, shortlists, rankings = score_all(inputs, arrays, dino_rows,
        protocol.get('dino', {}).get('score_method'))
    save(OUTPUT / 'label_free_view_scores.json', view_scores)
    save(OUTPUT / 'candidate_scores.json', pairs)
    save(OUTPUT / 'shortlist.json', shortlists)
    save(OUTPUT / 'label_free_rankings.json', rankings)
    verify_sources(protocol)
    for path, expected in before.items():
        assert digest(source_path(path)) == expected, 'Source changed during encoding/scoring: ' + path
    names = ('features.npz', 'tensor_indices.json', 'encoding.json', 'label_free_view_scores.json',
             'candidate_scores.json', 'shortlist.json', 'label_free_rankings.json')
    completion = dict(completed=True, scoring_completed=True, unique_inputs=len(inputs),
        CAD_inputs=sum(row['kind'] == 'CAD' for row in inputs), crop_inputs=96,
        task_count=96, candidate_pairs=len(pairs), per_view_score_rows=len(view_scores),
        ranking_rows=len(rankings), methods=list(METHODS), total_seconds=perf_counter() - start,
        input_hashes=before, artifact_hashes={name: digest(OUTPUT / name) for name in names},
        ranking_status_counts=dict(Counter(row['method'] + ':' + row['status'] for row in rankings)),
        retrieval='CLIP equal four-component mean per view; maximum over all twenty CAD views; top five scene crops.',
        final_selection='Exact separately saved DINO candidate scores and reference mappings; no CLIP/DINO score fusion.',
        missing_evidence_policy='All twenty CAD views and all twelve scene crops must have the arm text evidence; no dropping competitors.',
        cutoff_tie_policy='Within 1e-12 across fifth/sixth rank: unavailable_ambiguous_top5.',
        no_training_download_label_or_pose_reads=True, active_runtime_unchanged=True)
    save(OUTPUT / 'completion.json', completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 1, 'Run unit tests separately; this entry point runs the frozen trial.'
    run()


