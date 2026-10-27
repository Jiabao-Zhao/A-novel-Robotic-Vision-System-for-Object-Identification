"""Restore the frozen BOS/content/EOS policy for Transformers 5 CLIPTokenizer."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'run_clip.py'
SPEC = importlib.util.spec_from_file_location('frozen_clip_trial', SOURCE)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
REPORT = HERE / 'clip' / 'tokenizer_compatibility.json'
EXPECTED_RUNNER_HASH = 'b18e411d06df4a52771e97e285077f05677810b4a861584d8ce54ec18f2160e7'
TEST_STRINGS = ('', 'red cylindrical metal part', '  gray rectangular bracket  ',
                'ring-shaped component, 16 mm; hollow (center).',
                'café 齿轮 ⌀8', 'blue\nplastic\tconnector',
                'A surface with grooves—possibly threaded; unknown material.')


def attach_and_check(tokenizer):
    """Compare the adapter to the native backend encoding before attaching it."""
    import torch

    assert tokenizer.num_special_tokens_to_add(pair=False) == 2
    assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None and tokenizer.padding_side == 'right'
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    cases = []
    for text in TEST_STRINGS:
        content = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        native = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        expected = [bos, *content, eos]
        assert native == expected, ('Native special-token sequence differs', text, native, expected)
        cases.append(dict(text=text, content_ids=content, native_full_ids=native,
                          expected_full_ids=expected, exact_match=True))

    had_method = hasattr(tokenizer, 'build_inputs_with_special_tokens')
    if had_method:
        assert all(tokenizer.build_inputs_with_special_tokens(case['content_ids']) ==
                   case['native_full_ids'] for case in cases)
    else:
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            assert token_ids_1 is None, 'The frozen trial only encodes individual fields.'
            return [self.bos_token_id, *list(token_ids_0), self.eos_token_id]
        tokenizer.build_inputs_with_special_tokens = MethodType(build_inputs_with_special_tokens, tokenizer)
    for case in cases:
        assert tokenizer.build_inputs_with_special_tokens(case['content_ids']) == case['native_full_ids']

    sequences = [case['native_full_ids'] for case in cases]
    padded = tokenizer.pad(dict(input_ids=sequences), padding=True,
                           return_attention_mask=True, return_tensors='pt')
    assert isinstance(padded['input_ids'], torch.Tensor)
    max_length = max(map(len, sequences))
    assert tuple(padded['input_ids'].shape) == (len(sequences), max_length)
    for ids, tokens, mask in zip(sequences, padded['input_ids'].tolist(),
                                padded['attention_mask'].tolist()):
        assert tokens[:len(ids)] == ids
        assert mask == [1] * len(ids) + [0] * (max_length - len(ids))
        assert tokens[len(ids):] == [tokenizer.pad_token_id] * (max_length - len(ids))
    return dict(tokenizer_class=type(tokenizer).__name__, native_method_present=had_method,
        adapter_attached=not had_method, bos_token_id=bos, eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id, padding_side=tokenizer.padding_side,
        special_token_count=2, text_capacity=77, content_capacity=75,
        full_native_encoding_matches=True, native_padding_matches=True,
        test_cases=cases, package_or_global_environment_changes=False)


def verify_frozen_sources():
    assert RUNNER.digest(SOURCE) == EXPECTED_RUNNER_HASH
    protocol = RUNNER.read(HERE / 'protocol.json')
    RUNNER.verify_sources(protocol)
    failed = HERE / 'clip' / 'failed_attempt.json'
    failure = RUNNER.read(failed)
    assert failure['original_runner_sha256'] == EXPECTED_RUNNER_HASH
    assert failure['saved_feature_arrays'] is False and failure['scoring_completed'] is False
    return dict(wrapper_sha256=RUNNER.digest(Path(__file__)),
        original_runner_sha256=RUNNER.digest(SOURCE),
        protocol_sha256=RUNNER.digest(HERE / 'protocol.json'),
        preserved_failure_sha256=RUNNER.digest(failed), preserved_failure=failure)


def readiness():
    """Load only the cached processor/tokenizer; perform no model forward passes."""
    assert not REPORT.exists(), 'Preserve the pre-inference readiness evidence.'
    protected = verify_frozen_sources()
    helper = RUNNER.load_helper()
    asset = helper.read(helper.HERE / 'encoder_assets.json')['models']['clip']
    processor = helper.AutoProcessor.from_pretrained(asset['directory'], local_files_only=True)
    checks = attach_and_check(processor.tokenizer)
    tokenizer_files = ['config.json', 'preprocessor_config.json', 'tokenizer.json',
                       'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt', 'vocab.json']
    report = dict(**protected, **checks, transformers_version=helper.transformers.__version__,
        cached_asset_directory=asset['directory'], cached_asset_revision=asset['revision'],
        tokenizer_asset_hashes={name: RUNNER.digest(Path(asset['directory']) / name)
                                for name in tokenizer_files},
        readiness_completed=True, model_weights_loaded=False, image_or_text_inference_performed=False,
        frozen_before_model_inference=True,
        purpose='Compatibility-only adapter; the original source, prompts, token chunks, score equations and candidate selection remain frozen.')
    RUNNER.save(REPORT, report)
    print('Native cached CLIP tokenizer readiness passed; no model weights or forward passes.', flush=True)
    print('Wrapper SHA256: ' + protected['wrapper_sha256'], flush=True)
    print('Readiness SHA256: ' + RUNNER.digest(REPORT), flush=True)


def run():
    protected = verify_frozen_sources()
    readiness_record = RUNNER.read(REPORT)
    assert readiness_record['readiness_completed'] and readiness_record['frozen_before_model_inference']
    for key, value in protected.items():
        assert readiness_record[key] == value, ('Compatibility source changed', key)
    readiness_hash = RUNNER.digest(REPORT)
    original_load_helper = RUNNER.load_helper

    def load_helper():
        helper = original_load_helper()
        original_load_model = helper.load_model

        def load_model(asset, device):
            model, processor, metadata = original_load_model(asset, device)
            actual = attach_and_check(processor.tokenizer)
            assert helper.transformers.__version__ == readiness_record['transformers_version']
            for key, value in actual.items():
                assert readiness_record[key] == value, ('Readiness/runtime tokenizer mismatch', key)
            metadata['tokenizer_compatibility'] = dict(**protected,
                transformers_version=helper.transformers.__version__,
                readiness_file='clip/tokenizer_compatibility.json',
                readiness_sha256=readiness_hash, native_checks=actual)
            return model, processor, metadata

        helper.load_model = load_model
        return helper

    RUNNER.load_helper = load_helper
    RUNNER.run()
    assert RUNNER.digest(REPORT) == readiness_hash
    assert verify_frozen_sources() == protected


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('readiness', 'run')
    readiness() if sys.argv[1] == 'readiness' else run()
