"""Local Qwen/Gemma generation, extracted after likelihood-trial retirement.

Historical protocol hashes describe the earlier sources and remain unchanged.
Importing this helper does not contact a service or import an experiment driver.
"""
import copy
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'outputs/datasets/nist_task_board/shared_inputs/generation_models.json'
MODELS = ('qwen', 'gemma')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request('http://127.0.0.1:11434' + route, data,
                      {'Content-Type': 'application/json'})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def model_manifest(key, generation=True):
    assert key in MODELS
    assert generation is True, 'Only description/selection generation is retained'
    manifest = copy.deepcopy(read(SOURCE)[key])
    assert manifest['model_key'] == key and manifest['expected_backend_build'] is None
    assert manifest['controls']['logprobs'] is False and 'top_logprobs' not in manifest['controls']
    assert manifest['controls']['options']['num_predict'] == 1536
    return manifest


def runtime_for(manifest, folder):
    """Check the existing Ollama checkpoint/runtime before generation."""
    assert manifest['model_key'] in MODELS
    version = api('/api/version')['version']
    tag = next(t for t in api('/api/tags')['models'] if t['name'] == manifest['model'])
    assert version == manifest['controls']['expected_runtime_version']
    assert tag['digest'] == manifest['controls']['expected_model_digest']
    detail = dict(tag=tag, version=version, show=api('/api/show', dict(model=manifest['model'])))
    provenance = dict(model=manifest['model'], version=version, model_digest=tag['digest'])
    save(Path(folder) / 'runtime.json', dict(provenance=provenance,
                                           controls=manifest['controls'], detail=detail))
    return provenance


def request_for(key, controls, system, user, images, encoded=None):
    """Keep image metadata on disk and send encoded pixels only to the API."""
    assert key in MODELS
    assert controls.get('logprobs') is False and 'top_logprobs' not in controls
    wire = copy.deepcopy(controls)
    wire.pop('expected_model_digest', None)
    wire.pop('expected_runtime_version', None)
    assert images and all(set(image) == {'file', 'sha256'} for image in images)
    if encoded is not None:
        assert len(encoded) == len(images)
    message = dict(role='user', content=user, images=images if encoded is None else encoded)
    wire['messages'] = [dict(role='system', content=system), message]
    return wire


def infer(key, request):
    assert key in MODELS
    assert request.get('logprobs') is False and 'top_logprobs' not in request
    return api('/api/chat', request)


def extract_text(key, response):
    assert key in MODELS
    message = response['message']
    assert not message.get('thinking')
    return dict(raw_output=message.get('content') or '',
                finished=response['done'] and response.get('done_reason') == 'stop',
                prompt_tokens=response['prompt_eval_count'], generated_tokens=response['eval_count'])


def release(key, manifest):
    assert key in MODELS
    return api('/api/generate', dict(model=manifest['model'], keep_alive=0))


def source_files():
    return [Path(__file__), SOURCE]
