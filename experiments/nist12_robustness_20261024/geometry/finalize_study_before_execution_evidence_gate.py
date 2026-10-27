"""One-off finalization, gated on complete predictions and their verified lock."""
from pathlib import Path
import subprocess
import sys
import time
from common import HERE, read, save, digest


def main():
    assert sys.argv[1:] == ['--after-complete-predictions']
    markers = [HERE/'geometry/queue_completion.json', HERE/'visual/queue_completion.json',
               HERE/'baselines/gpu_release.json']
    folder = HERE/'finalization'
    while not all(file.exists() for file in markers):
        save(folder/'waiting.json', {'missing': [str(p.relative_to(HERE)) for p in markers if not p.exists()],
             'classification_accuracy_annotations_read': False})
        time.sleep(15)
    assert read(HERE/'visual/queue_completion.json')['complete']
    assert read(HERE/'baselines/completion.json')['complete']
    steps = [
        ('verify_before', 'evaluation/verify_protected.py', []),
        ('prediction_lock', 'evaluation/evaluate.py', ['lock']),
        ('accuracy_evaluation', 'evaluation/evaluate.py', ['evaluate', '--read-audit-after-go']),
        ('native_baseline_evaluation', 'baselines/evaluate_native.py', ['evaluate', '--read-audit-after-go']),
        ('render_report', 'evaluation/render_report.py', ['--after-go']),
        ('verify_after', 'evaluation/verify_protected.py', []),
    ]
    save(folder/'protocol.json', {'source_sha256': digest(__file__),
         'authorization': 'User authorized all six robustness experiments. Root conditionally authorizes accuracy annotation access only after complete prediction markers and a successful verified 5000-query prediction lock.',
         'marker_sha256': {str(p.relative_to(HERE)): digest(p) for p in markers},
         'sequence': [row[0] for row in steps]})
    for name, source, arguments in steps:
        print('START '+name, flush=True)
        save(folder/'progress.json', {'step': name})
        command = [sys.executable, '-u', str(HERE/source), *arguments]
        with (folder/(name+'.log')).open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        save(folder/(name+'.json'), {'returncode': result.returncode, 'source_sha256': digest(HERE/source)})
        assert result.returncode == 0, f'{name} failed; inspect the preserved log.'
    save(folder/'completion.json', {'complete': True, 'prediction_lock_sha256': digest(HERE/'evaluation/prediction_lock.json'),
         'accuracy_completion_sha256': digest(HERE/'evaluation/results/completion.json'),
         'native_completion_sha256': digest(HERE/'evaluation/native_baselines/completion.json'),
         'report_completion_sha256': digest(HERE/'evaluation/report/completion.json')})
    print('Finalization complete; figures and interpretation still require root review.', flush=True)


if __name__ == '__main__':
    main()
