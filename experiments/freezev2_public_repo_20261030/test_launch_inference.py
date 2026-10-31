import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import launch_inference as launch
import suite_runner as suite


@pytest.fixture
def sealing(tmp_path):
    root=tmp_path/'repo';source=root/'experiments/study';source.mkdir(parents=True)
    for name in launch.FILES:(source/name).write_text('fixture '+name)
    output=root/'outputs/freezev2_public_repo_20261030/own_inference_v1'
    config={'public_revision':launch.ref.PUBLIC_REVISION,'datasets':list(suite.DATASETS),'settings':dict(suite.FIXED),
        'resources':{'cpu_threads':4,'inference_workers':1,'disk_reserve_GiB':15},
        'no_test_GT_tuning':True,'no_author_CSV_inference_inputs':True,'paths':{'output':str(output)}}
    path=source/'config.json';path.write_text(json.dumps(config))
    return source,output,path


def test_seal_copies_only_orchestration_and_reuses_identical_sources(sealing):
    source,output,path=sealing
    config,_,snapshot,saved=launch.seal(path,source)
    assert set(launch.ref.read(snapshot/'source_manifest.json')['files'])==set(launch.FILES)
    assert launch.ref.read(saved)==config
    assert launch.seal(path,source)[2]==snapshot
    assert not (snapshot/'run_bop.py').exists()


def test_seal_refuses_changed_source_or_configuration(sealing):
    source,_,path=sealing;launch.seal(path,source)
    (source/'suite_runner.py').write_text('semantic change')
    with pytest.raises(ValueError,match='versioned'):launch.seal(path,source)


def test_seal_refuses_modified_snapshot(sealing):
    source,output,path=sealing;launch.seal(path,source)
    (output/'source_snapshot/suite_runner.py').write_text('changed')
    with pytest.raises(ValueError,match='payload changed'):launch.seal(path,source)


def test_native_bridge_identity_requires_pid_creation_and_exact_command():
    class Gone(Exception):pass
    class Zombie(Exception):pass
    process=SimpleNamespace(create_time=lambda:10.,cmdline=lambda:['native','--supervise'])
    psutil=SimpleNamespace(Process=lambda pid:process,NoSuchProcess=Gone,ZombieProcess=Zombie)
    record={'pid':4,'create_time':10.,'actual_cmdline':['native','--supervise']}
    assert launch.exact_live_bridge(record,psutil)
    assert not launch.exact_live_bridge({**record,'create_time':11.},psutil)
    assert not launch.exact_live_bridge({**record,'actual_cmdline':['other']},psutil)


def test_access_denied_never_means_dead_bridge():
    class Gone(Exception):pass
    class Zombie(Exception):pass
    def process(pid):raise PermissionError('cannot inspect')
    psutil=SimpleNamespace(Process=process,NoSuchProcess=Gone,ZombieProcess=Zombie)
    with pytest.raises(PermissionError):launch.exact_live_bridge({'pid':4},psutil)


def test_lock_probe_inspects_real_linux_flock_without_gpu_launch(tmp_path):
    seen=[]
    def run(command,**kwargs):
        seen.append(command);return SimpleNamespace(returncode=1,stdout='',stderr='BlockingIOError')
    with pytest.raises(RuntimeError,match='surviving Linux'):launch.probe_suite({'paths':{'inference_python':'/venv/bin/python'}},tmp_path,run)
    assert 'fcntl.flock' in seen[0][-2] and 'suite.lock' in seen[0][-1]
    assert not any('run_bop' in token for token in seen[0])
