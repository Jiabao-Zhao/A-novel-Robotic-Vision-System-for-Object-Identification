"""Replace only the identified idle old finalizer after independent control review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import psutil

if sys.flags.optimize:
    raise RuntimeError('Finalizer activation controls require unoptimized Python')
OUT = Path(__file__).resolve().parent
TRIAL, REPO = OUT.parent, OUT.parents[2]
SOURCE = TRIAL/'finalize_study.py'
OLD_SHA = '0ee6dc3aa4e9d5ce2a3e39e8764be20875dec24c263c20658d5f699e981dd686'
NEW_SHA = '2f60db529726ee0c41945eb9477002eab7be0a20acc06cc0991dee6d9981cce5'
REVIEW = TRIAL/'evaluation/finalizer_execution_evidence_independent_review_20261025.json'
def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()

assert digest(REVIEW) == 'ba7ebd143c8055f10478a0a509a921974ff97bc96e49523d0b3dcf24ed027cdc'
review = json.loads(REVIEW.read_text())
assert review['finalizer_execution_evidence_source_review_passed'] is True
for name, signature in review['reviewed_sha256'].items():
    assert digest(REPO/name) == signature, name
assert digest(SOURCE) == digest(OUT/'finalize_study_before_execution_evidence_gate.py') == OLD_SHA
assert digest(OUT/'finalize_study_execution_evidence_draft.py') == NEW_SHA
assert json.loads((OUT/'gpu_required_gear_reservation.json').read_text())['held'] is False
assert not (OUT/'alternate_new_cads_ownership.json').exists()
assert not (OUT/'queue_completion.json').exists()
assert not (TRIAL/'evaluation/prediction_lock.json').exists()
assert not (TRIAL/'evaluation/audit_access.json').exists()
matches = []
for process in psutil.process_iter(['pid', 'cmdline']):
    try:
        command = process.info['cmdline'] or []
        candidates = [Path(arg) for arg in command if arg.lower().replace('\\', '/').split('/')[-1] == 'finalize_study.py']
        if any((arg if arg.is_absolute() else Path(process.cwd())/arg).resolve() == SOURCE.resolve() for arg in candidates):
            matches.append(process)
    except (psutil.Error, OSError):
        pass
assert len(matches) == 1
process = matches[0]
assert process.pid == 48432 and abs(process.create_time()-1792853536.5657597) < .000001
assert process.cmdline()[1:] == ['-u', 'experiments/nist12_robustness_20261024/finalize_study.py', '--after-complete-predictions']
assert process.children(recursive=True) == []
identity = dict(pid=process.pid, create_time=process.create_time(), argv=process.cmdline(), cwd=process.cwd(), children=[])
before = dict(utc=datetime.now(timezone.utc).isoformat(), old_identity=identity,
    old_source_sha256=OLD_SHA, review_sha256=digest(REVIEW),
    prediction_lock_exists=False, accuracy_access_exists=False, classification_children=0, labels_read=False)
before_file = OUT/'finalizer_verified_idle_before_replacement.json'
assert not before_file.exists()
before_file.write_text(json.dumps(before, indent=2)+'\n')
process.terminate()
exit_code = process.wait(timeout=10)
assert not process.is_running()
assert not (TRIAL/'evaluation/prediction_lock.json').exists()
assert not (TRIAL/'evaluation/audit_access.json').exists()
payload = (OUT/'finalize_study_execution_evidence_draft.py').read_bytes()
temporary = SOURCE.with_suffix('.py.part')
temporary.write_bytes(payload)
assert temporary.read_bytes() == payload and digest(temporary) == NEW_SHA
temporary.replace(SOURCE)
assert digest(SOURCE) == NEW_SHA
receipt = dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(Path(__file__)),
    root_conditional_authorization_sha256=digest(OUT/'conditional_controller_authorization_20261025.json'),
    old_identity=identity, old_process_actually_exited=True, old_exit_code=exit_code,
    archived_old_source='finalize_study_before_execution_evidence_gate.py', archived_old_source_sha256=OLD_SHA,
    activated_source_sha256=NEW_SHA, independent_review_sha256=digest(REVIEW),
    completed_evidence_receipt_created=False, literal_anchor_created=False,
    new_finalizer_launched=False, scientific_child_processes_launched=0, labels_read=False)
(OUT/'finalizer_execution_evidence_activation.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
