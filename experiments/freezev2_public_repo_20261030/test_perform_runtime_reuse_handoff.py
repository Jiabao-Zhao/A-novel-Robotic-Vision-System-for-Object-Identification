"""Offline tests: no controller is launched and no process is terminated."""
import ast
import json
import os
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
import pytest
import perform_runtime_reuse_handoff as handoff


def controller_functions(*names,**extra):
    tree=ast.parse(handoff.CONTROLLER)
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    namespace={'pathlib':__import__('pathlib'),'hashlib':__import__('hashlib'),'json':json,'os':os,'uuid':uuid,'time':time,**extra}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated-controller-functions','exec'),namespace)
    return namespace


def test_controller_syntax_and_nul_command_separation():
    tree=ast.parse(handoff.CONTROLLER);compile(tree,'controller','exec')
    args=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='rstrip' and n.args and isinstance(n.args[0],ast.Constant)]
    assert '\0' in args


def test_controller_atomic_write_makes_new_transition_directory(tmp_path):
    ns=controller_functions('write');path=tmp_path/'execution_revisions/transition.json'
    ns['write'](path,{'status':'ready_for_handoff'})
    assert json.loads(path.read_text())=={'status':'ready_for_handoff'}
    assert path.read_text().endswith('\n')


def test_controller_identity_reads_exact_current_proc():
    ns=controller_functions('identity');actual=ns['identity'](os.getpid())
    assert actual['pid']==os.getpid() and actual['start_ticks'].isdigit()
    assert all('\0' not in item for item in actual['actual_cmdline'])
    assert len(actual['actual_cmdline'])>=2


def test_controller_legacy_receipts_hashes_only_terminal(tmp_path):
    ns=controller_functions('sha','read','legacy_receipts',out=tmp_path)
    path=tmp_path/'datasets/lmo/targets/key/receipt.json';path.parent.mkdir(parents=True);path.write_text('{"status":"complete"}')
    expected={'datasets/lmo/targets/key/receipt.json':ns['sha'](path)}
    assert ns['legacy_receipts']()==expected
    path.write_text('{"status":"running"}')
    with pytest.raises(RuntimeError,match='pending target'):ns['legacy_receipts']()


def test_exact_native_identity_refuses_pid_reuse_and_cmdline_change():
    process=SimpleNamespace(create_time=lambda:12.5,cmdline=lambda:['python','supervisor'])
    ps=SimpleNamespace(Process=lambda pid:process)
    record={'pid':5,'create_time':12.5,'actual_cmdline':['python','supervisor']}
    assert handoff.native_identity(record,ps) is process
    for updated in [dict(record,create_time=13.5),dict(record,actual_cmdline=['python','other'])]:
        with pytest.raises(RuntimeError,match='identity differs'):handoff.native_identity(updated,ps)


def test_access_denied_does_not_authorize_termination_or_launch():
    def denied(pid):raise PermissionError('access denied')
    ps=SimpleNamespace(Process=denied,NoSuchProcess=LookupError,ZombieProcess=RuntimeError)
    record={'pid':5,'create_time':12.5,'actual_cmdline':[]}
    with pytest.raises(PermissionError):handoff.native_identity(record,ps)
    with pytest.raises(PermissionError):handoff.same_identity_live(record,ps)


def test_exited_old_identity_is_not_live():
    def gone(pid):raise LookupError('gone')
    ps=SimpleNamespace(Process=gone,NoSuchProcess=LookupError,ZombieProcess=RuntimeError)
    assert not handoff.same_identity_live({'pid':5},ps)


def test_timeout_before_readiness_performs_no_mutation(tmp_path):
    controller=SimpleNamespace(poll=lambda:None)
    with pytest.raises(TimeoutError):handoff.wait_file(tmp_path/'ready.json',controller,time.monotonic()-1)
    assert list(tmp_path.iterdir())==[]


def test_controller_has_real_flock_and_no_force_kill():
    assert 'fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)' in handoff.CONTROLLER
    assert 'fcntl.flock(suite_stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)' in handoff.CONTROLLER
    assert 'signal.SIGTERM' in handoff.CONTROLLER
    assert 'SIGKILL' not in handoff.CONTROLLER
    assert 'unlink' not in handoff.CONTROLLER