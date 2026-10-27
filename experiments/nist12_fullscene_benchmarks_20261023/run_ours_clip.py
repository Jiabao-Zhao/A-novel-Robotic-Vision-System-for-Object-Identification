"""Unchanged four-term CLIP retrieval on the newly localized crop set."""
import importlib.util
from common import HERE, read, save, digest
import numpy as np

OLD = HERE.parent / 'nist12_qwen_clip_reranking_20261023'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    clip = load('fullscene_clip_native', OLD / 'run_clip.py')
    compat = load('fullscene_clip_tokenizer_compat', OLD / 'run_clip_tokenizer_fix.py')
    original = clip.load_helper

    def load_helper():
        helper = original()
        loader = helper.load_model
        def load_model(asset, device):
            model, processor, info = loader(asset, device)
            compat.attach_and_check(processor.tokenizer)
            return model, processor, info
        helper.load_model = load_model
        return helper
    clip.load_helper = load_helper
    inputs = read(HERE / 'ours/description_inputs.json')
    protocol = read(OLD / 'protocol.json')
    descriptions = read(HERE / 'ours/descriptions.json')
    arrays, mapping, metadata = clip.encode(protocol, inputs, descriptions)
    np.savez_compressed(HERE / 'ours/clip_features.npz', **arrays)
    save(HERE / 'ours/clip_mapping.json', mapping)
    save(HERE / 'ours/clip_encoding.json', metadata)
    _, cads, crops = clip.validate_inputs(inputs, require_trial_size=False)
    image_index = {row['input_id']: i for i, row in enumerate(inputs)}
    shortlists, pairs = [], []
    for state, instances in sorted(crops.items()):
        for target, views in cads.items():
            candidates = []
            for crop in instances:
                values = []
                for view in views:
                    components, score = clip.view_components(image_index[crop['input_id']], image_index[view['input_id']], arrays, 'full')
                    values.append({'view_index': view['view_index'], 'score': score, 'components': components})
                available = all(r['score'] is not None for r in values)
                best = max(values, key=lambda r: -float('inf') if r['score'] is None else r['score'])
                candidate = {'proposal_id': crop['proposal_id'], 'score': best['score'], 'score_available': available}
                candidates.append(candidate)
                pairs.append({'scene_id': state, 'target': target, **candidate, 'views': values})
            shortlist = clip.make_shortlist(candidates)
            shortlists.append({'scene_id': state, 'target': target, **shortlist})
    save(HERE / 'ours/clip_pair_scores.json', pairs)
    save(HERE / 'ours/shortlists.json', shortlists)
    save(HERE / 'ours/clip_completion.json', {'tasks': len(shortlists), 'available': sum(r['status']=='ok' for r in shortlists),
         'input_sha256': digest(HERE / 'ours/description_inputs.json'), 'scoring_source_sha256': digest(OLD / 'run_clip.py')})


if __name__ == '__main__':
    main()
