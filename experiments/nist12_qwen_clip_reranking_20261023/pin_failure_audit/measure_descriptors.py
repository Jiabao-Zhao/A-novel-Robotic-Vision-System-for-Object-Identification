"""Post-hoc descriptor diagnostics; never changes frozen matching decisions."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    decisions = read(STUDY / 'analysis/decisions.json')
    pairs = read(STUDY / 'clip/candidate_scores.json')
    indexed = {(p['state'], p['target'], p['proposal_id']): p for p in pairs}
    errors = [d for d in decisions if d['method'] == 'clip_full_dino'
              and d['same_family_wrong_size']]
    diagnostics = []
    for row in errors:
        true = indexed[row['state'], row['target'], row['truth_proposal_id']]
        wrong = indexed[row['state'], row['target'], row['selected_ids'][0]]
        diagnostics.append(dict(state=row['state'], target=row['target'],
            predicted=row['predicted_product'], truth_in_top5=row['truth_in_top5'],
            true_global=true['dino_global_score'], wrong_global=wrong['dino_global_score'],
            true_patch=true['dino_patch_score'], wrong_patch=wrong['dino_patch_score'],
            true_final=true['dino_score'], wrong_final=wrong['dino_score'],
            true_DINO_view=true['dino_selected_CAD_view'], wrong_DINO_view=wrong['dino_selected_CAD_view']))
    records = read(STUDY / 'dino/input_records.json')
    features = np.load(STUDY / 'dino/features.npz')
    prepared = np.load(STUDY / 'dino/inputs_and_masks.npz')
    record_to_feature = {int(record): i for i, record in enumerate(features['encoded_record_indices'])}
    catalog = read(STUDY / 'data/cad_inputs.json')
    direction = {(target, view['view_index']): view['camera_direction_cad']
                 for target, item in catalog.items() for view in item['views']}
    cls = features['cls']
    comparisons = []
    for prefix in ('RGOCG', 'KET'):
        targets = [target for target in catalog if target.startswith(prefix)]
        for a, target_a in enumerate(targets):
            for target_b in targets[a+1:]:
                for view in range(1, 21):
                    ia = next(i for i, r in enumerate(records) if r['kind'].lower() == 'cad'
                              and r['target'] == target_a and r['view_index'] == view)
                    ib = next(i for i, r in enumerate(records) if r['kind'].lower() == 'cad'
                              and r['target'] == target_b and r['view_index'] == view)
                    fa, fb = record_to_feature[ia], record_to_feature[ib]
                    comparisons.append(dict(target_a=target_a, target_b=target_b, view=view,
                        camera_direction_cad=direction[target_a, view],
                        CLS_cosine=float(cls[fa] @ cls[fb]),
                        preprocessed_images_exactly_equal=bool(np.array_equal(prepared['images'][fa], prepared['images'][fb])),
                        prepared_mean_abs_RGB_difference=float(np.abs(prepared['images'][fa].astype(float)
                                                                      - prepared['images'][fb].astype(float)).mean())))
    summary = dict(posthoc_only=True, new_model_inference=False, scoring_or_prompt_changes=False,
        pin_failures=len(errors), true_crop_retained_in_top5=sum(d['truth_in_top5'] for d in errors),
        wrong_beats_true_global=sum(d['wrong_global'] > d['true_global'] for d in diagnostics),
        wrong_beats_true_patch=sum(d['wrong_patch'] > d['true_patch'] for d in diagnostics),
        wrong_beats_true_both=sum(d['wrong_global'] > d['true_global'] and d['wrong_patch'] > d['true_patch']
                                  for d in diagnostics),
        pin_pair_diagnostics=diagnostics, same_family_same_view_CAD_comparisons=comparisons,
        source_hashes={str(path.relative_to(REPO)): digest(path) for path in
            (STUDY / 'analysis/decisions.json', STUDY / 'clip/candidate_scores.json',
             STUDY / 'dino/features.npz', STUDY / 'dino/inputs_and_masks.npz',
             STUDY / 'dino/input_records.json', STUDY / 'data/cad_inputs.json')})
    (HERE / 'descriptor_diagnostics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key not in
        ('pin_pair_diagnostics', 'same_family_same_view_CAD_comparisons', 'source_hashes')}, indent=2))
    best = sorted(comparisons, key=lambda r: -r['CLS_cosine'])[:12]
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
