"""Completed cohort receipts survive a restart during another cohort."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import queue_cohorts as queue


class ReceiptTests(unittest.TestCase):
    def test_rejects_another_cohorts_globals_and_retains_durable_completion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder)
            manifest = output/'manifest.json'
            queue.save(manifest, {'scenes': [{'scene_id': 'nominal'}]})
            destination = output/'receipt'
            destination.mkdir()
            completion = {'manifest_sha256': queue.digest(manifest)}
            queue.save(destination/'completion.json', completion)
            queue.save(output/'pair_index.json', {'scene_pairs': {'other': []}})
            queue.save(output/'progress.json', {'other': True})
            with patch.object(queue, 'OUT', output):
                queue._INDEX_STATE.clear()
                queue._COPIED.clear()
                queue.preserve(destination, manifest)
                self.assertTrue(queue.durable_finished(destination, manifest))
                self.assertFalse((destination/'progress.json').exists())
                self.assertEqual(queue.read(destination/'completion.json'), completion)

    def test_preserves_matching_index_and_archives_attempt_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder)
            manifest = output/'manifest.json'
            queue.save(manifest, {'scenes': [{'scene_id': 'target'}]})
            queue.save(output/'pair_index.json', {'scene_pairs': {'target': []}})
            queue.save(output/'progress.json', {'done': 1})
            destination = output/'receipt'
            with patch.object(queue, 'OUT', output):
                queue._INDEX_STATE.clear()
                queue._COPIED.clear()
                queue.preserve(destination, manifest)
                self.assertEqual(queue.read(destination/'progress.json'), {'done': 1})
                queue.archive_attempt(destination)
                archived = list((destination/'attempts').glob('*/progress.json'))
                self.assertEqual(len(archived), 1)
                self.assertEqual(queue.read(archived[0]), {'done': 1})


if __name__ == '__main__':
    unittest.main()
