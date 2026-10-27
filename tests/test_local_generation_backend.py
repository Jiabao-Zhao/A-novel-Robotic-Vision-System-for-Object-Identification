"""Offline checks for the retained local description/selection generator."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HELPER = (Path(__file__).resolve().parents[1] / 'experiments' /
          'nist12_qwen_clip_reranking_20261023' / 'local_backend.py')


def load_helper():
    spec = importlib.util.spec_from_file_location('generation_backend_test', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_manifest(key='qwen'):
    return dict(model_key=key, model=key + ':test', model_files={}, context_capacity=4096,
                expected_backend_build=None,
                controls=dict(model=key + ':test', stream=False, think=False, logprobs=False,
                              keep_alive='10m',
                              expected_model_digest='frozen-digest', expected_runtime_version='frozen-version',
                              options=dict(temperature=0, seed=0, num_ctx=4096, num_predict=1536,
                                           repeat_penalty=1.0, top_k=0, top_p=1.0)))


class LocalGenerationBackendTests(unittest.TestCase):
    def setUp(self):
        self.local = load_helper()

    def manifest(self, key='qwen'):
        with patch.object(self.local, 'read', return_value={key: frozen_manifest(key)}):
            return self.local.model_manifest(key, generation=True)

    def test_import_is_offline_and_has_no_retired_driver_import(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')):
            local = load_helper()
        self.assertFalse(hasattr(local, 'prior'))
        self.assertFalse(hasattr(local, 'parse_response'))
        self.assertFalse(hasattr(local, 'halt_for_context'))
        self.assertEqual(local.MODELS, ('qwen', 'gemma'))
        self.assertEqual(local.source_files()[0], HELPER)
        self.assertEqual([p.name for p in local.source_files()[1:]], ['generation_models.json'])

    def test_retained_runners_import_only_the_generation_helper_offline(self):
        runners = [HELPER.parent / 'run_qwen.py']
        for index, file in enumerate(runners):
            with self.subTest(file=file.name), patch('urllib.request.urlopen',
                    side_effect=AssertionError('unexpected network')), patch.object(sys, 'path', list(sys.path)):
                spec = importlib.util.spec_from_file_location('generation_runner_test_' + str(index), file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(Path(module.local.__file__).resolve(), HELPER)
                self.assertFalse(hasattr(module.local, 'parse_response'))

    def test_generation_metadata_is_loaded_without_mutating_source(self):
        for key in self.local.MODELS:
            with self.subTest(key=key):
                old = {key: frozen_manifest(key)}
                unchanged = copy.deepcopy(old)
                with patch.object(self.local, 'read', return_value=old) as read:
                    actual = self.local.model_manifest(key, generation=True)
                self.assertEqual(actual, old[key])
                self.assertEqual(old, unchanged)
                read.assert_called_once_with(self.local.SOURCE)
        with self.assertRaises(AssertionError):
            self.local.model_manifest('qwen', generation=False)

    def test_metadata_and_encoded_wire_preserve_prompt_images_and_controls(self):
        controls = self.manifest()['controls']
        unchanged = copy.deepcopy(controls)
        images = [dict(file='cad.png', sha256='image-hash')]
        for key in self.local.MODELS:
            metadata = self.local.request_for(key, controls, '', 'Exact approved prompt\n', images)
            wire = self.local.request_for(key, controls, '', 'Exact approved prompt\n', images, ['pixels'])
            expected = copy.deepcopy(controls)
            expected.pop('expected_model_digest')
            expected.pop('expected_runtime_version')
            expected['messages'] = [dict(role='system', content=''),
                                    dict(role='user', content='Exact approved prompt\n', images=images)]
            self.assertEqual(metadata, expected)
            expected['messages'][1]['images'] = ['pixels']
            self.assertEqual(wire, expected)
        self.assertEqual(controls, unchanged)
        with self.assertRaises(AssertionError):
            self.local.request_for('qwen', controls, '', '', images, [])
        with self.assertRaises(AssertionError):
            self.local.request_for('qwen', dict(frozen_manifest()['controls'], logprobs=True), '', '', images)

    def test_http_wire_keeps_local_endpoint_json_encoding_and_timeout(self):
        payload = dict(messages=[dict(content='CAD \u00e9')], stream=False)
        response = io.StringIO('{"ok": true}')
        with patch.object(self.local, 'urlopen', return_value=response) as opening:
            self.assertEqual(self.local.api('/api/chat', payload), dict(ok=True))
        request = opening.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/chat')
        self.assertEqual(request.data, json.dumps(payload).encode())
        self.assertEqual(opening.call_args.kwargs, dict(timeout=900))
        with patch.object(self.local, 'urlopen', return_value=io.StringIO('{}')) as opening:
            self.local.api('/api/version')
        self.assertIsNone(opening.call_args.args[0].data)

    def test_runtime_checks_and_saved_provenance_match_frozen_contract(self):
        manifest = self.manifest()
        tag = dict(name=manifest['model'], digest='frozen-digest')
        with tempfile.TemporaryDirectory(dir=HELPER.parent) as folder:
            with patch.object(self.local, 'api', side_effect=[dict(version='frozen-version'),
                              dict(models=[tag]), dict(parameters='native')]) as api:
                result = self.local.runtime_for(manifest, folder)
            self.assertEqual(api.call_args_list[0].args, ('/api/version',))
            self.assertEqual(api.call_args_list[1].args, ('/api/tags',))
            self.assertEqual(api.call_args_list[2].args, ('/api/show', dict(model=manifest['model'])))
            saved = self.local.read(Path(folder) / 'runtime.json')
            self.assertEqual(saved['provenance'], result)
            self.assertEqual(saved['controls'], manifest['controls'])
            self.assertEqual(result, dict(model=manifest['model'], version='frozen-version',
                                          model_digest='frozen-digest'))
            self.assertFalse((Path(folder) / 'runtime.json.tmp').exists())
        for version, digest in [('wrong-version', 'frozen-digest'), ('frozen-version', 'wrong-digest')]:
            with patch.object(self.local, 'api', side_effect=[dict(version=version),
                              dict(models=[dict(name=manifest['model'], digest=digest)])]):
                with self.assertRaises(AssertionError):
                    self.local.runtime_for(manifest, 'unused')

    def test_extract_preserves_raw_output_and_native_finish_status(self):
        response = dict(message=dict(content='  raw overlength description\n'), done=True,
                        done_reason='stop', prompt_eval_count=25, eval_count=12)
        for key in self.local.MODELS:
            parsed = self.local.extract_text(key, response)
            self.assertEqual(parsed, dict(raw_output='  raw overlength description\n', finished=True,
                                          prompt_tokens=25, generated_tokens=12))
        response['done_reason'] = 'length'
        self.assertFalse(self.local.extract_text('qwen', response)['finished'])
        response['message']['thinking'] = 'unexpected reasoning'
        with self.assertRaises(AssertionError):
            self.local.extract_text('qwen', response)

    def test_inference_and_release_keep_generation_routes(self):
        manifest = self.manifest()
        request = self.local.request_for('qwen', manifest['controls'], '', 'prompt',
                                         [dict(file='cad.png', sha256='hash')], ['pixels'])
        with patch.object(self.local, 'api', return_value=dict(ok=True)) as api:
            self.assertEqual(self.local.infer('qwen', request), dict(ok=True))
            api.assert_called_once_with('/api/chat', request)
            api.reset_mock()
            self.local.release('qwen', manifest)
            api.assert_called_once_with('/api/generate', dict(model=manifest['model'], keep_alive=0))
        with self.assertRaises(AssertionError):
            self.local.infer('qwen', dict(logprobs=True))


if __name__ == '__main__':
    unittest.main()
