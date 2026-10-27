"""Technical control tests with synthetic files; no scientific child or labels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
SOURCE = OUT/'finalize_study_execution_evidence_draft.py'
sys.path.insert(0, str(OUT.parent))
spec = importlib.util.spec_from_file_location('finalizer_guard_draft', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Controls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='finalizer_guard_fixture_', dir=OUT)
        self.repo = Path(self.temporary.name).resolve()
        self.here = self.repo/'experiments/nist12_robustness_20261024'
        self.here.mkdir(parents=True)
        self.receipt = self.here/'baselines/geometry_execution_evidence.json'
        self.anchor = self.here/'geometry/execution_evidence_lock.py'
        self.artifact = self.here/'geometry/synthetic_stable_receipt.json'
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_text('{"synthetic":true}\n')
        self.originals = module.HERE, module.REPO, module.path
        module.HERE, module.REPO = self.here, self.repo
        module.path = lambda value: self.repo/value
        self.value = {'complete': True, 'files': {self.artifact.relative_to(self.repo).as_posix(): self.sha(self.artifact)}}
        self.freeze()

    def tearDown(self):
        module.HERE, module.REPO, module.path = self.originals
        if not self.repo.is_relative_to(OUT.resolve()) or not self.repo.name.startswith('finalizer_guard_fixture_'):
            raise RuntimeError('Unexpected synthetic-fixture cleanup path')
        self.temporary.cleanup()

    @staticmethod
    def sha(file):
        return hashlib.sha256(file.read_bytes()).hexdigest()

    def freeze(self):
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        self.receipt.write_text(json.dumps(self.value)+'\n')
        self.anchor.parent.mkdir(parents=True, exist_ok=True)
        pin = {'file': self.receipt.relative_to(self.repo).as_posix(), 'sha256': self.sha(self.receipt)}
        self.anchor.write_text('EXECUTION_EVIDENCE = '+repr(pin)+'\n')

    def markers(self):
        for name in ['geometry/queue_completion.json', 'visual/queue_completion.json',
                     'baselines/gpu_release.json', 'baselines/completion.json']:
            p = self.here/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{"complete":true}')
        for name in ['evaluation/verify_protected.py', 'evaluation/evaluate.py', 'baselines/evaluate_native.py',
                     'evaluation/render_report.py']:
            p = self.here/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic, never executed\n')
        for name in ['evaluation/prediction_lock.json', 'evaluation/results/completion.json',
                     'evaluation/native_baselines/completion.json', 'evaluation/report/completion.json']:
            p = self.here/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')

    def test_valid_completed_map(self):
        self.assertEqual(module.verify_execution_evidence()['files'], 1)

    def test_missing_receipt(self):
        self.receipt.unlink()
        with self.assertRaises(FileNotFoundError): module.verify_execution_evidence()

    def test_incomplete_receipt(self):
        self.value['complete'] = False; self.freeze()
        with self.assertRaises(RuntimeError): module.verify_execution_evidence()

    def test_empty_map(self):
        self.value['files'] = {}; self.freeze()
        with self.assertRaises(RuntimeError): module.verify_execution_evidence()

    def test_missing_artifact(self):
        self.artifact.unlink()
        with self.assertRaises(FileNotFoundError): module.verify_execution_evidence()

    def test_artifact_tampered(self):
        self.artifact.write_text('changed')
        with self.assertRaises(RuntimeError): module.verify_execution_evidence()

    def test_anchor_wrong_hash(self):
        self.anchor.write_text('EXECUTION_EVIDENCE = '+repr({'file':self.receipt.relative_to(self.repo).as_posix(),'sha256':'0'*64}))
        with self.assertRaises(RuntimeError): module.verify_execution_evidence()

    def test_nonliteral_anchor_rejected(self):
        self.anchor.write_text('EXECUTION_EVIDENCE = dict(file="x", sha256="x")')
        with self.assertRaises(ValueError): module.verify_execution_evidence()

    def test_duplicate_anchor_rejected(self):
        self.anchor.write_text(self.anchor.read_text()*2)
        with self.assertRaises(RuntimeError): module.verify_execution_evidence()

    def test_audit_path_rejected_before_hash(self):
        self.value['files'] = {'experiments/nist12_robustness_20261024/audit/missing.json':'0'*64}; self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'accuracy-audit'): module.verify_execution_evidence()

    def test_consistent_reanchoring_during_run_rejected(self):
        old = module.verify_execution_evidence()
        self.value['note'] = 'changed receipt'; self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'during finalization'): module.verify_execution_evidence(old)

    def test_every_scientific_child_preceded_by_guard(self):
        self.markers(); events=[]; original=module.verify_execution_evidence
        def verify(expected=None): events.append('verify'); return original(expected)
        def fake_run(command, **kwargs):
            self.assertEqual(events[-1], 'verify'); events.append(command[3:]); return SimpleNamespace(returncode=0)
        with patch.object(sys, 'argv', [str(SOURCE),'--after-complete-predictions']), patch.object(module,'verify_execution_evidence',side_effect=verify), patch.object(module.subprocess,'run',side_effect=fake_run):
            module.main()
        self.assertEqual(sum(isinstance(row,list) for row in events),6)

    def test_tamper_after_lock_blocks_accuracy_child(self):
        self.markers(); calls=[]
        def fake_run(command, **kwargs):
            calls.append(command)
            if command[3:] == ['lock']: self.artifact.write_text('changed after synthetic lock')
            return SimpleNamespace(returncode=0)
        with patch.object(sys,'argv',[str(SOURCE),'--after-complete-predictions']), patch.object(module.subprocess,'run',side_effect=fake_run):
            with self.assertRaises(RuntimeError): module.main()
        self.assertEqual(len(calls),2)
        self.assertFalse(any('--read-audit-after-go' in row for row in calls))

    def test_optimized_python_rejected_before_import_or_io(self):
        result = subprocess.run([sys.executable,'-O',str(SOURCE)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Finalization controls require unoptimized Python',result.stderr)


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    platform='WSL' if sys.platform.startswith('linux') else 'Windows'
    value={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),
           'errors':len(result.errors),'source_sha256':Controls.sha(SOURCE),'tests_sha256':Controls.sha(Path(__file__)),
           'old_source_archive_sha256':Controls.sha(OUT/'finalize_study_before_execution_evidence_gate.py'),
           'actual_scientific_child_processes':0,'technical_optimized_reject_probes':1,'labels_read':False}
    (OUT/f'finalizer_execution_evidence_tests_{platform}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value));sys.exit(0 if result.wasSuccessful() else 1)
