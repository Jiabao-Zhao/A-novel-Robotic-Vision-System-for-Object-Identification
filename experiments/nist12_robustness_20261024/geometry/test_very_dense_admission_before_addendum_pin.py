"""Synthetic resource controls only; no point clouds, fits or model execution."""
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import very_dense_admission as guard

OUT = Path(__file__).resolve().parent
NOW = datetime.now(timezone.utc)


class StopWait(Exception):
    pass


class Controls(unittest.TestCase):
    def setUp(self):
        self.expected = dict(manifest_sha256='a'*64, runner_sha256='b'*64,
            admission_guard_sha256='c'*64, first_cache_key='d'*64, first_observation_key='e'*64,
            first_observed_voxels=2_000_001, max_group_observed_voxels=3_375_380,
            workers=1, query_threads=8, conditional_authorization_sha256=guard.AUTH_SHA)
        self.value = dict(self.expected, root_execution_authorized=True, actual_windows_physical=True,
            measurement_source='Windows GlobalMemoryStatusEx', host_available_bytes=7*1024**3,
            utc=NOW.isoformat())

    def test_valid_exact_admission(self):
        guard.validate(self.value, self.expected, NOW)

    def test_insufficient_memory(self):
        self.value['host_available_bytes'] -= 1
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_bool_memory_is_not_byte_count(self):
        self.value['host_available_bytes'] = True
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_stale_sample(self):
        self.value['utc'] = (NOW-timedelta(seconds=121)).isoformat()
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_future_sample(self):
        self.value['utc'] = (NOW+timedelta(seconds=6)).isoformat()
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_false_authorization(self):
        self.value['root_execution_authorized'] = False
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_linux_memory_rejected(self):
        self.value['measurement_source'] = 'Linux MemAvailable'
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_each_identity_mismatch_rejected(self):
        for key in self.expected:
            value = dict(self.value); value[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError): guard.validate(value, self.expected, NOW)

    def test_no_physical_measurement_rejected(self):
        self.value['actual_windows_physical'] = False
        with self.assertRaises(ValueError): guard.validate(self.value, self.expected, NOW)

    def test_scientific_functions_and_cache_AST_unchanged(self):
        old = ast.parse((OUT/'run_depth_before_very_dense_admission.py').read_text())
        new = ast.parse((OUT/'run_depth_very_dense_admission_draft.py').read_text())
        old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old.body
                         if isinstance(n, ast.FunctionDef) and n.name != 'main'}
        new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new.body
                         if isinstance(n, ast.FunctionDef) and n.name != 'main'}
        self.assertEqual(old_functions, new_functions)
        for node in new.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VERY_DENSE_WORKERS' for t in node.targets):
                self.assertEqual(node.value.value, 1); node.value = ast.Constant(value=2)
            if isinstance(node, ast.FunctionDef) and node.name == 'main':
                loop = next(n for n in node.body if isinstance(n, ast.For) and ast.unparse(n.target) == '(stage, group, workers)')
                checks = [n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "stage == 'very_dense'"]
                self.assertEqual(len(checks), 1); loop.body.remove(checks[0])
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def exercise_wait(self, with_marker):
        temporary = tempfile.TemporaryDirectory(prefix='very_dense_guard_fixture_', dir=OUT)
        folder = Path(temporary.name).resolve()
        try:
            manifest, runner = folder/'manifest.json', folder/'runner.py'
            manifest.write_text('{}'); runner.write_text('# synthetic, never executed\n')
            (folder/'conditional_controller_authorization_20261025.json').write_bytes(
                (OUT/'conditional_controller_authorization_20261025.json').read_bytes())
            job = ('synthetic', 1, 0, 'd'*64, 'e'*64)
            value = dict(self.value, utc=datetime.now(timezone.utc).isoformat(),
                manifest_sha256=guard.digest(manifest), runner_sha256=guard.digest(runner),
                admission_guard_sha256=guard.digest(guard.__file__), max_group_observed_voxels=3_000_000)
            if with_marker: (folder/'very_dense_host_admission.json').write_text(json.dumps(value))
            with patch.object(guard, 'OUT', folder), patch.object(guard.time, 'sleep', side_effect=StopWait):
                if with_marker:
                    guard.wait_for_admission(folder, manifest, runner, job, 2_000_001, 3_000_000)
                    self.assertTrue((folder/'very_dense_admission_used.json').exists())
                else:
                    with self.assertRaises(StopWait):
                        guard.wait_for_admission(folder, manifest, runner, job, 2_000_001, 3_000_000)
                    waiting=json.loads((folder/'very_dense_admission_waiting.json').read_text())
                    self.assertIn('pending', waiting['status'])
                    self.assertFalse((folder/'very_dense_admission_used.json').exists())
        finally:
            if not folder.is_relative_to(OUT.resolve()) or not folder.name.startswith('very_dense_guard_fixture_'):
                raise RuntimeError('Unexpected fixture cleanup target')
            temporary.cleanup()

    def test_missing_marker_waits_without_pool_or_unavailable(self):
        self.exercise_wait(False)

    def test_valid_marker_releases_only_resource_guard(self):
        self.exercise_wait(True)


if __name__ == '__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    platform='WSL' if sys.platform.startswith('linux') else 'Windows'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(passed=result.wasSuccessful(), tests_run=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), source_sha256=digest(OUT/'very_dense_admission.py'),
        draft_runner_sha256=digest(OUT/'run_depth_very_dense_admission_draft.py'),
        old_runner_archive_sha256=digest(OUT/'run_depth_before_very_dense_admission.py'),
        tests_sha256=digest(Path(__file__)), actual_pool_or_fit_starts=0,
        actual_host_admission_marker_created=False, labels_read=False)
    (OUT/f'very_dense_admission_tests_{platform}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
