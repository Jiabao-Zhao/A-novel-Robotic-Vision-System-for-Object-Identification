"""Two independent, frozen RoboGround-derived description requests per image."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODES = ('keywords', 'phrases')
FIELDS = ('COLOR', 'SHAPE', 'MATERIAL')
HELPER = HERE / 'local_backend.py'
spec = importlib.util.spec_from_file_location('nist8_qwen_clip_local', HELPER)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_file(value):
    file = Path(value)
    return file if file.is_absolute() else REPO / file


def api_json_sha256(value):
    # The existing Ollama helper sends json.dumps(payload).encode().
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def verify(protocol):
    """Offline validation only; no service connection, model load or inference."""
    immutable = protocol['immutable_inputs']
    required = [HERE / 'inputs.json', Path(__file__),
                *[HERE / 'prompts' / (mode + '.txt') for mode in MODES]]
    frozen_paths = {resolve_file(file).resolve() for file in immutable}
    assert all(file.resolve() in frozen_paths for file in required), 'Missing frozen source/input/prompt hash'
    for file, expected in immutable.items():
        assert local.digest(resolve_file(file)) == expected, 'Changed frozen input: ' + file
    assert protocol.get('system', '') == '', 'Published prompts supply no system instruction'
    manifest = protocol['qwen']
    assert manifest == local.model_manifest('qwen', generation=True), 'Generation controls/checkpoint changed'
    assert manifest['model_key'] == 'qwen' and manifest['model'] == 'qwen3.5:9b'
    assert manifest['controls']['options']['temperature'] == 0
    assert manifest['controls']['logprobs'] is False
    file = resolve_file(protocol['inputs_file'])
    assert file.resolve() == (HERE / 'inputs.json').resolve()
    inputs = local.read(file)
    if isinstance(inputs, dict):
        inputs = inputs['inputs']
    assert isinstance(inputs, list) and inputs
    assert len({item['input_id'] for item in inputs}) == len(inputs)
    for item in inputs:
        assert item['kind'] in ('CAD', 'crop')
        assert re.fullmatch(r'[A-Za-z0-9_.-]+', item['input_id']), 'Unsafe request file identifier'
        assert local.digest(resolve_file(item['file'])) == item['sha256'], 'Changed input pixels'
    return inputs, manifest


def parse_fields(text):
    """Keep raw text separate; require exactly one nonempty field of each type."""
    fields = {}
    for name in FIELDS:
        found = re.findall(r'<' + name + r'\s*>(.*?)</' + name + r'\s*>',
                           text, flags=re.IGNORECASE | re.DOTALL)
        if len(found) != 1 or not found[0].strip() or '<' in found[0] or '>' in found[0]:
            return {}, 'missing_empty_duplicate_or_nested_' + name
        fields[name] = found[0].strip()
    return fields, None


def request_pair(item, mode, manifest):
    prompt = (HERE / 'prompts' / (mode + '.txt')).read_text(encoding='utf-8')
    image = dict(file=str(resolve_file(item['file']).resolve()), sha256=item['sha256'])
    encoded = base64.b64encode(resolve_file(item['file']).read_bytes()).decode()
    metadata = local.request_for('qwen', manifest['controls'], '', prompt, [image])
    wire = local.request_for('qwen', manifest['controls'], '', prompt, [image], [encoded])
    return metadata, wire


def parsed_record(stored, item, mode, manifest):
    result = dict(input_id=item['input_id'], mode=mode, status='unavailable',
                  fields={}, raw_output='', finished=False, reason=None,
                  prompt_tokens=None, generated_tokens=None,
                  duration_seconds=stored['duration_seconds'],
                  raw_file=stored['raw_file'])
    if stored.get('generation_error'):
        result['reason'] = 'runtime_error'
        return result
    response = stored['response']
    assert response['model'] == manifest['model'], 'Wrong response model'
    parsed = local.extract_text('qwen', response)
    result.update(parsed)
    if not parsed['raw_output'].strip():
        result['reason'] = 'empty_output'
    elif parsed['prompt_tokens'] >= manifest['context_capacity']:
        result['reason'] = 'context_capacity_reached'
    elif not parsed['finished']:
        result['reason'] = 'generation_truncated_or_not_finished'
    else:
        fields, reason = parse_fields(parsed['raw_output'])
        result.update(fields=fields, reason=reason,
                      status='ok' if reason is None else 'unavailable')
    return result


def validate_stored(stored, item, mode, metadata, wire_hash, protocol_sha256, provenance):
    assert stored['input_id'] == item['input_id'] and stored['mode'] == mode
    assert stored['request'] == metadata and stored['wire_sha256'] == wire_hash
    assert stored['protocol_sha256'] == protocol_sha256
    assert stored['runtime_provenance'] == provenance
    assert stored['image_count'] == 1 and stored['image_sha256'] == item['sha256']
    if stored.get('response') is not None:
        assert stored['response_sha256'] == api_json_sha256(stored['response'])


def run():
    protocol = local.read(HERE / 'protocol.json')
    inputs, manifest = verify(protocol)
    protocol_sha256 = local.digest(HERE / 'protocol.json')
    provenance = local.runtime_for(manifest, HERE / 'qwen/runtime')
    rows = []
    try:
        for item in inputs:
            for mode in MODES:
                metadata, wire = request_pair(item, mode, manifest)
                wire_hash = api_json_sha256(wire)
                file = HERE / 'qwen/raw' / mode / (item['input_id'] + '.json')
                if file.exists():
                    stored = local.read(file)
                else:
                    started = utc_now()
                    before = perf_counter()
                    response, error = None, None
                    try:
                        response = local.infer('qwen', wire)
                    except Exception as exc:
                        error = dict(type=type(exc).__name__, message=str(exc))
                    stored = dict(input_id=item['input_id'], mode=mode, request=metadata,
                                  wire_sha256=wire_hash, wire_serialization='json.dumps default UTF-8',
                                  response=response, generation_error=error,
                                  response_sha256=None if response is None else api_json_sha256(response),
                                  protocol_sha256=protocol_sha256, runtime_provenance=provenance,
                                  image_count=1, image_sha256=item['sha256'],
                                  started_utc=started, ended_utc=utc_now(),
                                  duration_seconds=perf_counter() - before,
                                  raw_file=file.relative_to(HERE).as_posix())
                    local.save(file, stored)
                validate_stored(stored, item, mode, metadata, wire_hash, protocol_sha256, provenance)
                row = parsed_record(stored, item, mode, manifest)
                rows.append(row)
                local.save(HERE / 'descriptions.json', rows)
                print(json.dumps(dict(completed=len(rows), total=2 * len(inputs),
                                      input_id=item['input_id'], mode=mode,
                                      status=row['status'], reason=row['reason'])), flush=True)
                if row['reason'] in ('runtime_error', 'empty_output', 'context_capacity_reached'):
                    raise RuntimeError('Generation halted: ' + row['reason'] + '; raw evidence retained')
        verify(protocol)
        assert local.digest(HERE / 'protocol.json') == protocol_sha256
        local.save(HERE / 'qwen/completion.json',
                   dict(completed_utc=utc_now(), protocol_sha256=protocol_sha256,
                        expected_requests=2 * len(inputs), saved_requests=len(rows),
                        ok=sum(row['status'] == 'ok' for row in rows),
                        unavailable=sum(row['status'] != 'ok' for row in rows),
                        descriptions_sha256=local.digest(HERE / 'descriptions.json')))
    finally:
        verify(protocol)
        assert local.digest(HERE / 'protocol.json') == protocol_sha256
        local.release('qwen', manifest)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'verify':
        items, _ = verify(local.read(HERE / 'protocol.json'))
        print(json.dumps(dict(verified_inputs=len(items), expected_requests=2 * len(items))))
    elif len(sys.argv) == 2 and sys.argv[1] == 'run':
        run()
    else:
        raise SystemExit('Usage: python run_qwen.py verify|run')
