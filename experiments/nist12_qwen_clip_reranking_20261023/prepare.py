"""Freeze the approved CLIP-retrieval then DINO experiment without audit labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROMPT_SHA = '180dc9df66ab2cbd0b9b8057a054314bc1252d74b47db9d5dc98a15c391d2019'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def freeze(path, value):
    if path.exists():
        assert read(path) == value, 'Do not replace frozen evidence: ' + str(path)
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    assert not (HERE / 'protocol.json').exists(), 'Protocol is already frozen.'
    records = read(HERE / 'data/input_records.json')
    cad = [r for r in records if r['kind'].lower() == 'cad']
    crops = [r for r in records if r['kind'] == 'crop']
    assert len(records) == 336 and len(cad) == 240 and len(crops) == 96
    parts = list(dict.fromkeys(r['target'] for r in cad))
    scenes = sorted({r['state'] for r in crops})
    assert len(parts) == 12 and scenes == [f'N{i:02d}' for i in range(8)]
    assert all(len([r for r in cad if r['target'] == t]) == 20 for t in parts)
    assert all(len([r for r in crops if r['state'] == s]) == 12 for s in scenes)
    inputs = []
    for row in records:
        item = dict(row)
        item['kind'] = 'CAD' if row['kind'].lower() == 'cad' else 'crop'
        item['input_id'] = (f'cad__{row["target"]}__v{row["view_index"]:02d}' if item['kind'] == 'CAD'
                            else f'crop__{row["state"]}__p{row["proposal_id"]:03d}')
        assert digest(REPO / row['file']) == row['sha256'], row['file']
        inputs.append(item)
    assert len({r['input_id'] for r in inputs}) == 336
    tasks = [dict(task_id=f'{s}__{t}', state=s, target=t, candidates=[
        dict(proposal_id=r['proposal_id'], crop_input_id=r['input_id']) for r in inputs
        if r['kind'] == 'crop' and r['state'] == s]) for s in scenes for t in parts]
    assert len(tasks) == 96 and all(len(t['candidates']) == 12 for t in tasks)
    freeze(HERE / 'inputs.json', inputs)
    freeze(HERE / 'tasks.json', tasks)
    sys.path.insert(0, str(HERE))
    import local_backend as local
    files = [Path(__file__), HERE / 'build_inputs.py', HERE / 'run_qwen.py',
             HERE / 'run_dino.py', HERE / 'focusedtest_dino.py', HERE / 'run_clip.py', HERE / 'test_clip.py',
             HERE / 'inputs.json', HERE / 'tasks.json', HERE / 'data/input_records.json',
             HERE / 'data/capture_manifest.json', HERE / 'data/cad_inputs.json',
             HERE / 'lighting_preview/preview_lighting.py', HERE / 'lighting_preview/settings.json',
             HERE / 'prompts/keywords.txt', HERE / 'prompts/phrases.txt', HERE / 'prompts/source_notes.md',
             REPO / 'docs/cad_view_prompt.txt', REPO / 'scripts/cad_descriptors.py',
             REPO / 'scripts/sam6d_patch_matching.py',
             REPO / 'scripts/clip_encoders.py',
             REPO / 'scripts/encoder_assets.json',
             *local.source_files()]
    for row in inputs:
        files.append(REPO / row['file'])
        if row.get('mask_file'):
            files.append(REPO / row['mask_file'])
    assert digest(REPO / 'docs/cad_view_prompt.txt') == PROMPT_SHA
    protected = {relative(p): digest(p) for p in dict.fromkeys(files)}
    clip = dict(asset_config='scripts/encoder_assets.json',
        asset_method='clip', device='cuda', dtype='float32', seed=0, batch_size=16,
        fields=['COLOR', 'SHAPE', 'MATERIAL'], modes=['keywords', 'phrases'],
        text_capacity=77, max_content_tokens=75, tie_tolerance=1e-12,
        chunk_policy='Consecutive content-token chunks <=75; preserve all tokens. Normalize chunks then equal mean+normalize per field; equal mean+normalize fields across both modes.',
        fusion='mean(image_image,text_text,crop_image_to_CAD_text,crop_text_to_CAD_image)',
        view_aggregation='Maximum across all twenty actual CAD views, no DINO view prefilter.',
        top_k=5,
        availability='All twenty CAD and twelve crop descriptions required independently per full/shape arm. No dropping unavailable competitors; retain partial diagnostics.',
        boundary_tie='If fifth and sixth scores tie within1e-12, retrieval is ambiguous/unavailable; no arbitrary winner or label-dependent cutoff.',
        image_preprocessing='Standard CLIPProcessor on the same raw RGB files Qwen sees; no foreground edits for CLIP.')
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), frozen_before_model_inference=True,
        task='Retrieved CAD -> CLIP image+text top-five scene crops -> DINO final matching; DINO alone independent across all twelve crops.',
        primary_methods=['dino', 'clip_full_dino'], control_methods=['clip_shape_dino'],
        diagnostic_methods=['clip_full_only', 'clip_shape_only'],
        parts=parts, scenes=scenes, tasks=96, candidates_per_task=12, shortlist_size=5,
        candidate_pairs=1152, selected_pairs=480, CAD_views_per_part=20,
        expected_Qwen_requests=672, inputs_file=relative(HERE / 'inputs.json'),
        tasks_file=relative(HERE / 'tasks.json'), inputs_sha256=digest(HERE / 'inputs.json'),
        qwen=local.model_manifest('qwen', generation=True), system='', clip=clip,
        prompts={m: (HERE / f'prompts/{m}.txt').read_text(encoding='utf-8') for m in ('keywords', 'phrases')},
        authoring='Two independent source-paper prompts per single unnamed image. Opening-only adaptation documented in source_notes. No true names/dimensions/audit labels in model-facing text. Preserve raw responses and unavailable statuses; no content/length retries or corrections.',
        dino=dict(checkpoint='outputs/cache/torch_hub/checkpoints/dinov2_vits14_pretrain.pth',
            checkpoint_sha256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9',
            foreground='Observed camera-depth distance to observed plane strictly >0.002m; native inclusiveROI; fixed pixel offset0; tightmasked white224letterbox.',
            CAD_mask='Exact isolated-render mesh silhouette, not scene audit segmentation.',
            score='Equal mean of top-five clamped CLS view similarities and foreground-patch appearance at the best CLS view. Keep global and patch components; no geometry score.',
            final='Use exactly the same per-candidate DINO score/view for both methods. Hybrid considers only CLIP-selected five; no CLIP-DINO score fusion or fallback.',
            corresponding_view='DINO chooses its best-CLS corresponding CAD view among its available views. Record this separately from the CLIP maximum-score CAD view.'),
        lighting=dict(condition='middle_balanced_original_colors', headlight_ambient=.6,
            headlight_diffuse=.25, headlight_specular=.02, fill_diffuse=.12, fill_specular=.02,
            fill_count=3, cast_shadows=False, recolor=False,
            source='Original twelve-product N00-N07 layouts, fixed cameras/poses; approved preview condition in lighting_preview/settings.json.'),
        text_controls='Full COLOR/SHAPE/MATERIAL is primary; SHAPE-text filter is separately reported. Both retain RGB image similarity and are not geometry-only image methods.',
        decisions='Independent ranking per target, final ties within1e-12 unresolved, complete comparison only. No calibrated acceptance threshold or global one-to-one assignment.',
        evaluation='Write all label-free scores, shortlists and rankings before reading audit identity labels. All96 tasks retained; report retrievalrecall, perproduct outcomes, fixed/introduced errors, and within-family size confusion.',
        training='All existing cached pretrained weights unchanged; no paid APIs/download/install/robot execution.',
        scope='Reused known-CAD simulation layouts with one product instance each and frozen valid depth-localizer ROIs. This does not measure localization, unseenCAD/generalization, real images, absent targets, duplicate instances, full6D or assembly success.',
        protected=protected, immutable_inputs=protected)
    freeze(HERE / 'protocol.json', protocol)
    print(json.dumps(dict(frozen=True, images=len(inputs), Qwen_requests=672, tasks=96,
                          primary_methods=protocol['primary_methods'], source_hashes=len(protected))), flush=True)


if __name__ == '__main__':
    main()
