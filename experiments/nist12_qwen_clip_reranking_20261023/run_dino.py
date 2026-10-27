"""Cached DINO control and reusable candidate scores for the relit N scenes.

This runner never reads simulator identities. The CLIP shortlist is applied by
the experiment analysis, after these scores for all twelve crops are frozen.
"""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'dino'
sys.path.insert(0, str(REPO))
from scripts import cad_descriptors as dino
from scripts import sam6d_patch_matching as visual

CHECKPOINT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
TIE_ATOL = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    """Completed or resumed evidence may be replayed, never silently replaced."""
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Frozen evidence changed: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def local(path):
    value = str(path).replace('\\', '/')
    retired = 'C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to'
    for prefix in (retired, '/mnt/c' + retired[2:]):
        if value == prefix or value.startswith(prefix + '/'):
            suffix = value[len(prefix):].lstrip('/')
            if '..' in suffix.split('/') or ':' in suffix:
                raise ValueError('Unsafe retired-workspace path suffix')
            return REPO / 'outputs/datasets/nist_task_board/legacy_workspace_inputs' / suffix
    if len(value) > 1 and value[1] == ':':
        if sys.platform != 'win32':
            value = '/mnt/' + value[0].lower() + value[2:]
        return Path(value)
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO / candidate


def relative(path):
    return str(Path(path).relative_to(REPO)).replace('\\', '/')


def foreground(depth, intrinsics, roi, plane, threshold=.002):
    """Use the observed fitted plane, native inclusive ROI, and camera meters."""
    assert threshold == .002
    plane = np.asarray(plane, dtype=np.float64)
    assert plane.shape == (4,) and np.isfinite(plane).all() and np.linalg.norm(plane[:3]) > 0
    x1, y1, x2, y2 = [int(roi[key]) for key in ('x1', 'y1', 'x2', 'y2')]
    assert 0 <= x1 <= x2 < depth.shape[1] and 0 <= y1 <= y2 < depth.shape[0]
    yy, xx = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    z = depth[y1:y2 + 1, x1:x2 + 1]
    with np.errstate(invalid='ignore'):
        xyz = np.stack([(xx-intrinsics[0, 2])*z/intrinsics[0, 0],
                        (yy-intrinsics[1, 2])*z/intrinsics[1, 1], z], axis=-1)
        distance = np.abs(xyz @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3])
    return (z > 0) & np.isfinite(z) & (distance > threshold)


def prepare(rgb, mask):
    """Shared established masked tight crop and aspect-preserving letterbox."""
    assert rgb.shape[:2] == mask.shape and mask.dtype == np.bool_
    image, _, bounds = dino.prepare_masked_input(rgb, mask)
    x1, y1, x2, y2 = bounds
    return image, dino.letterbox_mask(mask[y1:y2 + 1, x1:x2 + 1]), list(bounds)


def rank_candidates(candidates):
    available = [row for row in candidates if row['score'] is not None]
    ordered = sorted(available, key=lambda row: (-row['score'], row['proposal_id']))
    if not ordered:
        return dict(available=False, top_proposal_ids=[], winner_proposal_id=None,
                    top_score=None, gap=None, ordered_proposal_ids=[], eligible_candidates=0,
                    unavailable_ids=[row['proposal_id'] for row in candidates])
    top = ordered[0]['score']
    ties = [row['proposal_id'] for row in ordered if abs(row['score'] - top) <= TIE_ATOL]
    return dict(available=True, top_proposal_ids=ties,
                winner_proposal_id=ties[0] if len(ties) == 1 else None,
                top_score=top, gap=top-ordered[1]['score'] if len(ordered) > 1 else None,
                ordered_proposal_ids=[row['proposal_id'] for row in ordered],
                eligible_candidates=len(ordered),
                unavailable_ids=[row['proposal_id'] for row in candidates if row['score'] is None])


def prepare_images(records, capture, catalog):
    images, masks, encoded_indices, preparation = [], [], [], []
    consumed = set()
    scenes = {}
    for state, scene in capture['states'].items():
        depth_path, intrinsics_path = local(scene['depth_file']), local(scene['intrinsics_file'])
        scenes[state] = (np.load(depth_path), np.load(intrinsics_path), scene)
        consumed.update([depth_path, intrinsics_path])
    for index, record in enumerate(records):
        path = local(record['file'])
        assert digest(path) == record['sha256'], f'Input image changed: {path}'
        consumed.add(path)
        rgb = np.asarray(Image.open(path).convert('RGB'))
        if record['kind'].lower() == 'cad':
            view, = [view for view in catalog[record['target']]['views']
                     if view['view_index'] == record['view_index']]
            mask_path = local(record.get('mask_file', view['mask_file']))
            expected = record.get('mask_sha256', view.get('mask_sha256'))
            assert expected and digest(mask_path) == expected
            native = np.asarray(Image.open(mask_path).convert('L'))
            assert np.isin(native, [0, 255]).all(), 'Expected exact isolated-CAD silhouette.'
            mask = native > 0
            consumed.add(mask_path)
        else:
            depth, intrinsics, scene = scenes[record['state']]
            mask = foreground(depth, intrinsics, record['roi'], scene['plane_model_camera'],
                              scene['plane_distance_threshold_m'])
        assert rgb.shape[:2] == mask.shape
        metadata = dict(record_index=index, foreground_pixels=int(mask.sum()))
        if mask.any():
            image, mask224, bounds = prepare(rgb, mask)
            encoded_indices.append(index)
            images.append(image)
            masks.append(mask224)
            metadata.update(status='available', native_tight_bbox_in_input=bounds)
        else:
            metadata.update(status='unavailable_empty_native_foreground')
        preparation.append(metadata)
    return np.stack(images), np.stack(masks), encoded_indices, preparation, consumed


def encode(images, masks, encoded_indices, checkpoint, hub):
    assert torch.cuda.is_available(), 'Use the existing CUDA runtime; do not install a replacement.'
    model = torch.hub.load(str(hub), 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.eval().requires_grad_(False).to('cuda')
    assert model.patch_size == 14 and model.embed_dim == 384
    cls, patches = [], []
    started = perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            batch = [torch.from_numpy(((image.astype(np.float32)/255-[.485, .456, .406])/
                     [.229, .224, .225]).transpose(2, 0, 1).astype(np.float32))
                     for image in images[start:start + 8]]
            features = model.forward_features(torch.stack(batch).to('cuda'))
            for key, output in [('x_norm_clstoken', cls), ('x_norm_patchtokens', patches)]:
                array = features[key].cpu().numpy().astype(np.float64)
                array /= np.linalg.norm(array, axis=-1, keepdims=True)
                output.extend(array)
            if start % 64 == 0 or start + 8 >= len(images):
                print(f'DINO encoded {min(start + 8, len(images))}/{len(images)} images', flush=True)
    arrays = dict(cls=np.array(cls), patches=np.array(patches),
                  encoded_record_indices=np.array(encoded_indices, dtype=np.int64))
    assert arrays['cls'].shape == (len(images), 384)
    assert arrays['patches'].shape == (len(images), 256, 384)
    for name in ['cls', 'patches']:
        assert np.isfinite(arrays[name]).all()
        np.testing.assert_allclose(np.linalg.norm(arrays[name], axis=-1), 1., atol=1e-12)
    np.savez_compressed(OUT / 'features.npz', **arrays)
    np.savez_compressed(OUT / 'inputs_and_masks.npz', images=images, masks=masks,
                        encoded_record_indices=np.array(encoded_indices, dtype=np.int64))
    save(OUT / 'encoding_completion.json', dict(passed=True, new_images_encoded=len(images),
         all_CAD_and_crop_features_recomputed=True, seconds=perf_counter()-started,
         device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
         checkpoint_sha256=digest(checkpoint), no_training_or_download=True))
    del model
    torch.cuda.empty_cache()


def score(records):
    features = np.load(OUT / 'features.npz')
    inputs = np.load(OUT / 'inputs_and_masks.npz')
    np.testing.assert_array_equal(features['encoded_record_indices'], inputs['encoded_record_indices'])
    indices = features['encoded_record_indices'].tolist()
    feature_index = {record_index: encoded_index for encoded_index, record_index in enumerate(indices)}
    descriptors = [visual.masked_patch_descriptors(patch, dino.patch_occupancy(mask))
                   for patch, mask in zip(features['patches'], inputs['masks'])]
    counts = {record_index: int(torch.count_nonzero(descriptors[feature_index[record_index]].sum(-1)))
              if record_index in feature_index else 0 for record_index in range(len(records))}
    statuses = [dict(**record, record_index=i, foreground_patches=counts[i],
                    status='available' if counts[i] else 'unavailable_no_majority_foreground_patch')
                for i, record in enumerate(records)]
    save(OUT / 'descriptor_statuses.json', statuses)
    parts = list(dict.fromkeys(record['target'] for record in records if record['kind'].lower() == 'cad'))
    pools = {target: [i for i, record in enumerate(records) if record['kind'].lower() == 'cad'
                     and record['target'] == target and counts[i] > 0] for target in parts}
    observed = {(record['state'], record['proposal_id']): i for i, record in enumerate(records)
                if record['kind'] == 'crop'}
    scores, rankings = [], []
    for state in sorted({state for state, _ in observed}):
        for target in parts:
            refs = pools[target]
            candidates = []
            for (scene, proposal), index in observed.items():
                if scene != state:
                    continue
                row = dict(state=state, target=target, proposal_id=proposal,
                           score=None, global_score=None, patch_score=None,
                           selected_CAD_view=None, coverage=None,
                           observed_foreground_patches=counts[index],
                           reference_views=[records[i]['view_index'] for i in refs])
                if counts[index] == 0:
                    row['status'] = 'unavailable_crop_foreground_descriptor'
                elif len(refs) < 5:
                    row['status'] = 'unavailable_fewer_than_five_CAD_views'
                else:
                    query = feature_index[index]
                    sem = visual.semantic_score(features['cls'][query],
                                                features['cls'][[feature_index[i] for i in refs]])
                    ref = refs[sem['best_view_index_1based'] - 1]
                    app = visual.appearance_score(descriptors[query], descriptors[feature_index[ref]])
                    row.update(status='available',
                        score=visual.combined_visual_score(sem['semantic_score'], app['appearance_score']),
                        global_score=sem['semantic_score'], patch_score=app['appearance_score'],
                        selected_CAD_view=records[ref]['view_index'],
                        selected_CAD_record_index=ref, cls_view_cosines=sem['cls_view_cosines'],
                        clamped_top5_CLS_scores=sem['top5_scores'],
                        CAD_foreground_patches=counts[ref], coverage=app['visible_ratio_diagnostic'])
                scores.append(row)
                candidates.append(row)
            rankings.append(dict(method='dino', state=state, target=target, **rank_candidates(candidates)))
    assert len(parts) == 12 and len(observed) == 96 and len(scores) == 1152 and len(rankings) == 96
    save(OUT / 'scores.json', scores)
    save(OUT / 'label_free_rankings.json', rankings)
    return scores, rankings, statuses


def run():
    torch.set_num_threads(1)
    assert (ROOT / 'protocol.json').is_file(), 'Freeze the root experiment protocol before inference.'
    assert not (OUT / 'completion.json').exists(), 'Preserve completed DINO evidence.'
    checkpoint = REPO / 'outputs/cache/torch_hub/checkpoints/dinov2_vits14_pretrain.pth'
    hub = REPO / 'outputs/cache/torch_hub/facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35'
    assert checkpoint.is_file() and hub.is_dir() and digest(checkpoint) == CHECKPOINT_SHA
    records = read(DATA / 'input_records.json')
    capture = read(DATA / 'capture_manifest.json')
    catalog = read(DATA / 'cad_inputs.json')
    assert len(records) == 336 and sum(record['kind'].lower() == 'cad' for record in records) == 240
    images, masks, indices, preparation, consumed = prepare_images(records, capture, catalog)
    protected_paths = [Path(__file__), ROOT / 'protocol.json', DATA / 'input_records.json',
                       DATA / 'capture_manifest.json', DATA / 'cad_inputs.json', checkpoint,
                       REPO / 'scripts/cad_descriptors.py', REPO / 'scripts/sam6d_patch_matching.py',
                       REPO / 'docs/cad_view_prompt.txt', *sorted(consumed)]
    protected = {relative(path): digest(path) for path in protected_paths}
    protocol = dict(model='Cached DINOv2 ViT-S/14; unchanged pretrained weights',
        checkpoint_sha256=CHECKPOINT_SHA, states=sorted(capture['states']), parts=list(catalog),
        intended_CAD_views_per_target=20, tie_absolute_tolerance=TIE_ATOL,
        preprocessing='Exact native CAD silhouette; observed depth localizer fitted plane absolute distance >0.002m, no scene labels. Inclusive masked tight crop, white background, aspect-preserving 224x224 letterbox.',
        scoring='Mean top-five clamped CLS cosines over available CAD views. Foreground-patch appearance at best CLS view. Equal mean of these terms.',
        template_availability='Strict >0.5 patch occupancy; exclude unavailable references and preserve statuses. At least five available CAD views required.',
        separate_baseline='Rank all twelve scene crops for each retrieved CAD.',
        hybrid='These identical pair scores select only within the independently supplied CLIP top-five shortlist. No DINO prefilter or changed scoring.',
        no_identity_labels_read=True, no_training_download_or_VLM_calls=True,
        scores_are_uncalibrated=True, protected=protected)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'input_records.json', records)
    save(OUT / 'preparation_statuses.json', preparation)
    if not (OUT / 'encoding_completion.json').exists():
        encode(images, masks, indices, checkpoint, hub)
    scores, rankings, statuses = score(records)
    assert all(digest(REPO / path) == sha for path, sha in protected.items())
    save(OUT / 'completion.json', dict(passed=True, tasks=96, comparisons=len(scores),
         CAD_images=240, crop_images=96, input_images=336, scores_written_before_identity_audit=True,
         unavailable_pairs=sum(row['status'] != 'available' for row in scores),
         unavailable_descriptors=[row for row in statuses if row['status'] != 'available'],
         tied_rankings=sum(len(row['top_proposal_ids']) > 1 for row in rankings),
         protected_hashes_verified=True, no_identity_labels_read=True,
         scores_sha256=digest(OUT / 'scores.json'),
         rankings_sha256=digest(OUT / 'label_free_rankings.json')))
    print(json.dumps(read(OUT / 'completion.json'), indent=2), flush=True)


if __name__ == '__main__':
    run()
