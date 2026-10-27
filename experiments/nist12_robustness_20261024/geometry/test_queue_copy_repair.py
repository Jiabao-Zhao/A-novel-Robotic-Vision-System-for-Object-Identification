"""Focused control-copy failure and live-child preservation checks; no fits/models."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('queue_copy_under_test', HERE/'queue_cohorts.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)

class CopyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='queue_copy_test_', dir=HERE)
        self.folder = Path(self.directory.name).resolve()
        assert self.folder.is_relative_to(HERE.resolve())
        self.addCleanup(self.directory.cleanup)
        self.source, self.destination = self.folder/'source', self.folder/'destination'
        self.temporary = self.folder/'destination.part'
        self.value = b'new\x00receipt\xff\n'
        self.source.write_bytes(self.value)
        self.destination.write_bytes(b'old valid receipt')

    def preserved(self):
        self.assertEqual(self.source.read_bytes(), self.value)
        self.assertEqual(self.destination.read_bytes(), b'old valid receipt')

    def test_exact_bytes_then_atomic_replace(self):
        queue.copy_receipt(self.source, self.destination)
        self.assertEqual(self.destination.read_bytes(), self.value)
        self.assertEqual(self.source.read_bytes(), self.value)
        self.assertFalse(self.temporary.exists())

    def test_two_transient_errno61_reads_recover(self):
        original, attempts = Path.read_bytes, []
        def read(file):
            if file == self.source:
                attempts.append(file)
                if len(attempts) < 3: raise OSError(61, 'No data available')
            return original(file)
        with patch.object(Path, 'read_bytes', read), patch.object(queue.time, 'sleep') as sleep:
            queue.copy_receipt(self.source, self.destination)
            self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(self.destination.read_bytes(), self.value)

    def test_permanent_errno61_is_bounded_and_preserves_existing_part(self):
        original, attempts = Path.read_bytes, []
        self.temporary.write_bytes(b'earlier partial evidence')
        def read(file):
            if file == self.source:
                attempts.append(file)
                raise OSError(61, 'No data available')
            return original(file)
        with patch.object(Path, 'read_bytes', read), patch.object(queue.time, 'sleep'):
            with self.assertRaises(OSError): queue.copy_receipt(self.source, self.destination)
        self.assertEqual(len(attempts), 3)
        self.preserved()
        self.assertEqual(self.temporary.read_bytes(), b'earlier partial evidence')

    def test_other_read_errno_is_not_retried(self):
        original, attempts = Path.read_bytes, []
        def read(file):
            if file == self.source:
                attempts.append(file)
                raise OSError(2, 'Missing source')
            return original(file)
        with patch.object(Path, 'read_bytes', read), patch.object(queue.time, 'sleep') as sleep:
            with self.assertRaises(OSError): queue.copy_receipt(self.source, self.destination)
            sleep.assert_not_called()
        self.assertEqual(len(attempts), 1)
        self.preserved()

    def test_partial_write_failure_preserves_source_destination_and_temp(self):
        original = Path.write_bytes
        def write(file, value):
            if file == self.temporary:
                original(file, b'partial evidence')
                raise OSError(5, 'Write failed')
            return original(file, value)
        with patch.object(Path, 'write_bytes', write), patch.object(queue.time, 'sleep') as sleep:
            with self.assertRaises(OSError): queue.copy_receipt(self.source, self.destination)
            sleep.assert_not_called()
        self.preserved()
        self.assertEqual(self.temporary.read_bytes(), b'partial evidence')

    def test_hash_mismatch_cannot_replace_valid_destination(self):
        original = Path.read_bytes
        def read(file):
            return b'wrong bytes' if file == self.temporary else original(file)
        with patch.object(Path, 'read_bytes', read), patch.object(queue.time, 'sleep') as sleep:
            with self.assertRaisesRegex(OSError, 'hash mismatch'):
                queue.copy_receipt(self.source, self.destination)
            sleep.assert_not_called()
        self.preserved()
        self.assertEqual(self.temporary.read_bytes(), self.value)

    def test_replace_failure_keeps_verified_temp(self):
        original = Path.replace
        def replace(file, destination):
            if file == self.temporary: raise OSError(5, 'Replace failed')
            return original(file, destination)
        with patch.object(Path, 'replace', replace):
            with self.assertRaises(OSError): queue.copy_receipt(self.source, self.destination)
        self.preserved()
        self.assertEqual(self.temporary.read_bytes(), self.value)

class FakeChild:
    pid, returncode = 456789, None
    waited = False
    def poll(self): return None
    def wait(self):
        self.waited, self.returncode = True, 0
        return 0

class DrainTests(unittest.TestCase):
    def test_broken_stderr_cannot_skip_drain_or_replace_original_error(self):
        class BrokenStderr:
            def write(self, value): raise OSError(5, 'Closed stderr')
            def flush(self): raise OSError(5, 'Closed stderr')
        process, error = FakeChild(), OSError(61, 'Original receipt error')
        with patch.object(queue, 'preserve', side_effect=error), patch.object(queue, 'save', side_effect=OSError(5, 'Diagnostic failure')), patch.object(queue.sys, 'stderr', BrokenStderr()):
            with self.assertRaises(OSError) as captured:
                queue.preserve_while_running(process, HERE, HERE/'unused')
        self.assertTrue(process.waited)
        self.assertIs(captured.exception, error)

    def test_live_child_drains_before_original_copy_exception(self):
        process, records, error = FakeChild(), [], OSError(61, 'No data available')
        with patch.object(queue, 'preserve', side_effect=error), patch.object(queue, 'save', side_effect=lambda f,v: records.append(v)):
            with self.assertRaises(OSError) as captured:
                queue.preserve_while_running(process, HERE, HERE/'unused')
        self.assertIs(captured.exception, error)
        self.assertTrue(process.waited)
        self.assertEqual([row['child_drained'] for row in records], [False, True])

    def test_diagnostic_write_failure_still_drains_and_keeps_original_error(self):
        process, error = FakeChild(), OSError(61, 'Original receipt error')
        with patch.object(queue, 'preserve', side_effect=error), patch.object(queue, 'save', side_effect=OSError(5, 'Diagnostic failure')), patch.object(queue.sys, 'stderr', io.StringIO()):
            with self.assertRaises(OSError) as captured:
                queue.preserve_while_running(process, HERE, HERE/'unused')
        self.assertTrue(process.waited)
        self.assertIs(captured.exception, error)

if __name__ == '__main__':
    started, log = perf_counter(), io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    digest = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), passed=result.wasSuccessful(),
        tests_run=result.testsRun, seconds=perf_counter()-started,
        queue_source_sha256=digest(HERE/'queue_cohorts.py'), test_source_sha256=digest(Path(__file__)),
        original_queue_archive_sha256=digest(HERE/'queue_cohorts_before_copy_repair.py'),
        scientific_runner_sha256=digest(HERE/'run_depth.py'),
        actual_processes_started=0, models_loaded=0, fits_run=0, GPU_calls=0, labels_read=False,
        log=log.getvalue())
    (HERE/'queue_copy_repair_tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
