"""Synthetic error-body tests; no transport, model, runtime or dataset reads."""
import base64
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from urllib.error import HTTPError

SOURCE=Path(__file__).resolve().parents[2]/'run_visual.py'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('visual_HTTP_logging_test',SOURCE)
visual=importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


class ErrorLoggingTests(unittest.TestCase):
    def test_HTTP400_body_bytes_preserved(self):
        body=b'{"error":"synthetic context limit"}\xff'
        error=HTTPError('http://synthetic.invalid/api/chat',400,'Bad Request',{},io.BytesIO(body))
        actual=visual.generation_error_record(error)
        self.assertEqual(actual['http_status'],400)
        self.assertEqual(base64.b64decode(actual['http_response_body_base64']),body)
        self.assertEqual(actual['http_response_body_bytes'],len(body))
        self.assertIn('synthetic context limit',actual['http_response_body_utf8'])
        self.assertEqual(len(actual['diagnostic_source_sha256']),64)

    def test_non_HTTP_exception_keeps_original_record(self):
        self.assertEqual(visual.generation_error_record(ValueError('synthetic')),
                         dict(type='ValueError',message='synthetic'))

    def test_body_read_failure_does_not_hide_original_status(self):
        class BrokenBody:
            def read(self):raise OSError('synthetic unreadable body')
            def close(self):pass
        error=HTTPError('http://synthetic.invalid/api/chat',400,'Bad Request',{},BrokenBody())
        actual=visual.generation_error_record(error)
        self.assertEqual(actual['http_status'],400)
        self.assertEqual(actual['http_response_body_read_error']['type'],'OSError')


if __name__=='__main__':unittest.main()
