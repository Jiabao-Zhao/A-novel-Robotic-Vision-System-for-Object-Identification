"""One-off finalization, gated on complete predictions and frozen execution evidence."""
import ast
from pathlib import Path
import subprocess
import sys
import time

if sys.flags.optimize:
    raise RuntimeError('Finalization controls require unoptimized Python')

from common import HERE, REPO, read, save, digest, path


def verify_execution_evidence(expected=None):
    receipt = HERE/'baselines/geometry_execution_evidence.json'
    anchor = HERE/'geometry/execution_evidence_lock.py'
    assignments = [node for node in ast.parse(anchor.read_text(encoding='utf-8')).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'EXECUTION_EVIDENCE']
    if len(assignments) != 1:
        raise RuntimeError('Missing unique literal execution-evidence anchor')
    pin = ast.literal_eval(assignments[0].value)
    expected_name = receipt.relative_to(REPO).as_posix()
    if not isinstance(pin, dict) or pin.get('file') != expected_name or pin.get('sha256') != digest(receipt):
        raise RuntimeError('Execution-evidence receipt differs from its literal Python anchor')
    value = read(receipt)
    if value.get('complete') is not True or not isinstance(value.get('files'), dict) or not value['files']:
        raise RuntimeError('Execution evidence must be a completed nonempty hash map')
    for name, signature in value['files'].items():
        file = path(name).resolve()
        if not file.is_relative_to(REPO.resolve()) or 'audit' in file.relative_to(REPO.resolve()).parts:
            raise RuntimeError('Execution evidence must use workspace artifacts outside accuracy-audit folders')
        if not isinstance(signature, str) or len(signature) != 64 or digest(file) != signature:
            raise RuntimeError('Execution evidence changed: '+name)
    state = dict(receipt_sha256=digest(receipt), anchor_sha256=digest(anchor), files=len(value['files']))
    if expected is not None and state != expected:
        raise RuntimeError('Execution-evidence anchor changed during finalization')
    return state


def main():
    if sys.argv[1:] != ['--after-complete-predictions']:
        raise RuntimeError('Use the explicit complete-predictions finalization flag')
    markers = [HERE/'geometry/queue_completion.json', HERE/'visual/queue_completion.json',
               HERE/'baselines/gpu_release.json', HERE/'baselines/geometry_execution_evidence.json',
               HERE/'geometry/execution_evidence_lock.py']
    folder = HERE/'finalization'
    while not all(file.exists() for file in markers):
        save(folder/'waiting.json', {'missing': [str(p.relative_to(HERE)) for p in markers if not p.exists()],
             'classification_accuracy_annotations_read': False})
        time.sleep(15)
    if read(HERE/'visual/queue_completion.json')['complete'] is not True or read(HERE/'baselines/completion.json')['complete'] is not True:
        raise RuntimeError('Prediction completion markers are invalid')
    execution_evidence = verify_execution_evidence()
    steps = [
        ('verify_before', 'evaluation/verify_protected.py', []),
        ('prediction_lock', 'evaluation/evaluate.py', ['lock']),
        ('accuracy_evaluation', 'evaluation/evaluate.py', ['evaluate', '--read-audit-after-go']),
        ('native_baseline_evaluation', 'baselines/evaluate_native.py', ['evaluate', '--read-audit-after-go']),
        ('render_report', 'evaluation/render_report.py', ['--after-go']),
        ('verify_after', 'evaluation/verify_protected.py', []),
    ]
    save(folder/'protocol.json', {'source_sha256': digest(__file__),
         'authorization': 'User authorized all six robustness experiments. Root conditionally authorizes accuracy annotation access only after complete prediction markers, frozen execution evidence and a successful verified 5000-query prediction lock.',
         'marker_sha256': {str(p.relative_to(HERE)): digest(p) for p in markers},
         'execution_evidence': execution_evidence,
         'sequence': [row[0] for row in steps]})
    for name, source, arguments in steps:
        print('START '+name, flush=True)
        save(folder/'progress.json', {'step': name})
        command = [sys.executable, '-u', str(HERE/source), *arguments]
        with (folder/(name+'.log')).open('w', encoding='utf-8') as stream:
            verify_execution_evidence(execution_evidence)
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        save(folder/(name+'.json'), {'returncode': result.returncode, 'source_sha256': digest(HERE/source)})
        if result.returncode:
            raise RuntimeError(name+' failed; inspect the preserved log.')
    save(folder/'completion.json', {'complete': True, 'prediction_lock_sha256': digest(HERE/'evaluation/prediction_lock.json'),
         'accuracy_completion_sha256': digest(HERE/'evaluation/results/completion.json'),
         'native_completion_sha256': digest(HERE/'evaluation/native_baselines/completion.json'),
         'report_completion_sha256': digest(HERE/'evaluation/report/completion.json'),
         'execution_evidence': verify_execution_evidence(execution_evidence)})
    print('Finalization complete; figures and interpretation still require root review.', flush=True)


if __name__ == '__main__':
    main()
