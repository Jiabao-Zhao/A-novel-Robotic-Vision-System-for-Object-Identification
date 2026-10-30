"""Linux ownership tests with sleeping Python children only; no GPU allocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from freezev2_repro.durable import worker_lock
from freezev2_repro.owned_process import run_owned, start_ticks


pytestmark = pytest.mark.skipif(os.name != "posix" or not Path("/proc/self/stat").exists(),
                                reason="Ownership requires the recorded Linux/WSL flock and /proc semantics")
SOURCE = Path(__file__).resolve().parents[1]


def _environment():
    result = os.environ.copy()
    result["PYTHONPATH"] = str(SOURCE) + os.pathsep + result.get("PYTHONPATH", "")
    return result


def _wait_json(path, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.02)
    pytest.fail(f"Timed out waiting for synthetic process marker {path}")


def _alive(pid, expected_ticks=None):
    try:
        data = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    # Orphaned zombies have released all process resources and file descriptors.
    return data[0] != "Z" and (expected_ticks is None or data[19] == str(expected_ticks))


def _locked(path):
    try:
        with worker_lock(path):
            return False
    except RuntimeError:
        return True


def _kill_identity(pid, ticks):
    if pid and _alive(pid, ticks):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_run_owned_normal_exit_and_releases_guard_lock(tmp_path):
    lock = tmp_path / "worker.lock"
    folder = tmp_path / "normal"
    heartbeats = []
    with worker_lock(lock) as stream:
        assert run_owned([sys.executable, "-c", "print('synthetic evaluator complete')"],
                         _environment(), folder, stream, heartbeats.append) == 0
        assert _locked(lock), "The worker itself must continue owning its lock"
    assert heartbeats and not _locked(lock)
    status = _wait_json(folder / "owned_process.json")
    assert not _alive(status["child_pid"], status["child_start_ticks"])
    assert not _alive(status["guard_pid"], status["guard_start_ticks"])
    assert "synthetic evaluator complete" in (folder / "execution.log").read_text()


def test_inherited_fd_holds_flock_after_parent_context_closes(tmp_path):
    lock = tmp_path / "worker.lock"
    marker = tmp_path / "inherited.json"
    child = None
    ticks = None
    try:
        with worker_lock(lock) as stream:
            code = ("import os,sys,json,time;from pathlib import Path;"
                    "os.fstat(int(sys.argv[1]));"
                    "Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid()}));time.sleep(1.5)")
            child = subprocess.Popen([sys.executable, "-c", code, str(stream.fileno()), str(marker)],
                                     pass_fds=(stream.fileno(),), env=_environment())
            ticks = start_ticks(child.pid)
            _wait_json(marker)
        assert child.poll() is None and _locked(lock), "Closing the original FD must not release its inherited flock"
        assert child.wait(timeout=5) == 0
        assert not _locked(lock)
    finally:
        if child is not None:
            _kill_identity(child.pid, ticks)
            child.wait(timeout=5)


def test_abrupt_owner_death_cleans_entire_child_group_before_unlock(tmp_path):
    """The leader exits on TERM; its descendant deliberately ignores TERM.

    A surviving descendant is a real overlap risk even after the evaluator's
    direct Popen leader has exited, so its lock must remain held until cleanup.
    """
    lock = tmp_path / "worker.lock"
    folder = tmp_path / "owned"
    marker = tmp_path / "grandchild.json"
    child_script = tmp_path / "synthetic_evaluator.py"
    grand_code = (
        "import os,sys,json,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19];"
        "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'ticks':ticks}));time.sleep(60)"
    )
    child_script.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{grand_code!r},sys.argv[1]])\n"
        "time.sleep(60)\n"
    )
    owner_code = (
        "import os,sys;from freezev2_repro.durable import worker_lock;"
        "from freezev2_repro.owned_process import run_owned;"
        "\nwith worker_lock(sys.argv[1]) as stream:\n"
        " run_owned([sys.executable,sys.argv[3],sys.argv[4]],os.environ.copy(),sys.argv[2],stream,lambda pid:None)\n"
    )
    owner = subprocess.Popen([sys.executable, "-c", owner_code, str(lock), str(folder),
                              str(child_script), str(marker)], env=_environment(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owner_ticks = start_ticks(owner.pid)
    status = grand = None
    try:
        status = _wait_json(folder / "owned_process.json")
        grand = _wait_json(marker)
        assert os.getpgid(grand["pid"]) == status["child_pid"]
        assert _locked(lock)
        owner.kill()
        owner.wait(timeout=5)  # Reap it so /proc cannot retain a zombie parent identity.
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            child_alive = _alive(status["child_pid"], status["child_start_ticks"])
            grand_alive = _alive(grand["pid"], grand["ticks"])
            held = _locked(lock)
            if child_alive or grand_alive:
                assert held, "Inherited lock released while an owned evaluator descendant was still live"
            elif not held:
                break
            time.sleep(.02)
        else:
            pytest.fail("Guard did not clean the synthetic child process group and release its lock within8s")
        assert not _alive(grand["pid"], grand["ticks"])
        assert not _locked(lock)
    finally:
        _kill_identity(owner.pid, owner_ticks)
        owner.wait(timeout=5)
        if status:
            _kill_identity(status["child_pid"], status["child_start_ticks"])
            _kill_identity(status["guard_pid"], status["guard_start_ticks"])
        if grand:
            _kill_identity(grand["pid"], grand["ticks"])
