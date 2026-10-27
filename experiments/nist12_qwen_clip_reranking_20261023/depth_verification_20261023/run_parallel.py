"""Eight-worker execution adapter; frozen fitting and scoring stay unchanged."""
import run_trial


if __name__ == '__main__':
    run_trial.check_sources()
    paths = sorted((run_trial.HERE / 'fits').glob('*.json'))
    completed = {}
    for path in paths:
        record = run_trial.read(path)
        assert record['protocol_sha256'] == run_trial.digest(run_trial.HERE / 'protocol.json')
        completed[path.name] = run_trial.digest(path)
    run_trial.write_once(run_trial.HERE / 'execution_adjustment.json', {
        'change': 'Resume the identical frozen fits with eight rather than four CPU workers.',
        'scoring_or_fitting_configuration_changed': False,
        'protocol_sha256': run_trial.digest(run_trial.HERE / 'protocol.json'),
        'adapter_sha256': run_trial.digest(__file__),
        'previous_workers': 4, 'workers': 8,
        'retained_completed_fit_sha256': completed,
        'evaluation_labels_read': False,
    })
    run_trial.WORKERS = 8
    run_trial.run()
