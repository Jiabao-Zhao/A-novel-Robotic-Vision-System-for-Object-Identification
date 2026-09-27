"""Features-only ablation using the frozen family trial's local inference code."""
import argparse
import copy
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'family-association-ablation'
spec = importlib.util.spec_from_file_location('family_runner', SOURCE / 'run_trial.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
# Redirect only the imported module's output root; never edit the prior trial.
runner.HERE = HERE
MODELS = runner.MODELS
read, save, digest, table = runner.read, runner.save, runner.digest, runner.table
family, validate_inputs = runner.family, runner.validate_inputs
request_for, classify, collect = runner.request_for, runner.classify, runner.collect


def audit(manifest):
    baseline = read(SOURCE / 'manifest.json')
    assert manifest['controls'] == baseline['controls']
    assert manifest['crops'] == baseline['crops']
    assert manifest['names'] == baseline['names']
    assert manifest['models'] == baseline['models']
    forbidden = re.compile(r'\b(gears?|nuts?|pins?|USB|DSUB|connectors?|waterproof)\b', re.I)
    seen = {}
    for target, prompt in manifest['prompts'].items():
        text = prompt['system_prompt'] + '\n' + prompt['user_prompt']
        assert not forbidden.search(text), (target, forbidden.search(text))
        assert all(identity not in text for identity in manifest['names'])
        assert all(name.lower() not in text.lower() for name in manifest['names'].values())
        fam = family(target)
        if fam in seen:
            assert prompt == seen[fam]
        seen[fam] = prompt
        for description in manifest['descriptions'][fam]:
            assert '- ' + description in text
        assert text.count('A = target feature configuration supported') == 1
        assert text.count('B = target feature configuration contradicted') == 1
        assert text.count('C = insufficient visible evidence') == 1
    return dict(product_name_leakage=False, target_id_leakage=False,
                identical_size_family_prompts=True, controls_and_crops_identical=True)


def prepare():
    baseline = read(SOURCE / 'manifest.json')
    rows = [r for slug in MODELS.values() for r in read(SOURCE / slug / 'all_scores.json')]
    assert len(rows) == 1452
    descriptions = read(HERE / 'descriptions.json')
    system = (HERE / 'system_prompt.txt').read_text(encoding='utf-8')
    template = (HERE / 'user_prompt.txt').read_text(encoding='utf-8')
    prompts = {t: dict(system_prompt=system, user_prompt=template.format(
        descriptions='\n'.join('- ' + s for s in descriptions[family(t)]))) for t in baseline['names']}
    immutable = copy.deepcopy(baseline['immutable_inputs'])
    for p in SOURCE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            immutable[p.relative_to(ROOT).as_posix()] = digest(p)
    for name in ('descriptions.json', 'system_prompt.txt', 'user_prompt.txt'):
        immutable[(HERE / name).relative_to(ROOT).as_posix()] = digest(HERE / name)
    manifest = dict(condition='features_only', baseline_condition='family_association',
        baseline_directory=SOURCE.relative_to(ROOT).as_posix(),
        names=baseline['names'], crops=baseline['crops'], models=baseline['models'],
        controls=baseline['controls'], prompts=prompts, descriptions=descriptions,
        immutable_inputs=immutable, expected_calls_per_model=484, local_only=True,
        labels=dict(A='target feature configuration supported', B='target feature configuration contradicted',
                    C='insufficient visible evidence'),
        score='exp(raw log probability of token A); no normalization or calibration',
        threshold_before_collection=None, exclusions=baseline['exclusions'],
        fallback_policy=baseline['fallback_policy'],
        scope='Joint task/rules/descriptions ablation. Internal family names and target IDs are evaluation metadata only; no model-visible category labels. No new localization, views, or crop modifications.')
    audit(manifest)
    if (HERE / 'manifest.json').exists():
        assert read(HERE / 'manifest.json') == manifest, 'Frozen features condition changed.'
    validate_inputs(manifest)
    save(HERE / 'manifest.json', manifest)
    save(HERE / 'selected_prompts.json', prompts)
    save(HERE / 'prompt_audit.json', audit(manifest))
    table(HERE / 'baseline_scores', rows)
    print('Prepared 1452 local comparisons; family baseline and all inherited inputs protected.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'classify', 'collect', 'audit'])
    parser.add_argument('--model', choices=list(MODELS))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'audit':
        print(audit(read(HERE / 'manifest.json')))
        validate_inputs(read(HERE / 'manifest.json'))
    else:
        audit(read(HERE / 'manifest.json'))
        for model in ([args.model] if args.model else MODELS):
            (classify if args.action == 'classify' else collect)(model)
