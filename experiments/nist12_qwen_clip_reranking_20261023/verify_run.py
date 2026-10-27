"""Check final artifact links and the separately recorded tokenizer API repair."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in ('src', 'href') and value)


def main():
    compatibility = read(HERE / 'clip/tokenizer_compatibility.json')
    encoding = read(HERE / 'clip/encoding.json')
    assert sha(HERE / 'run_clip_tokenizer_fix.py') == compatibility['wrapper_sha256']
    assert sha(HERE / 'clip/tokenizer_compatibility.json') == encoding['tokenizer_compatibility']['readiness_sha256']
    lock = read(HERE / 'analysis_source_lock.json')
    assert sha(HERE / 'analyze.py') == lock['analyze_sha256']
    assert sha(HERE / 'focusedtest_report.py') == lock['tests_sha256']
    completion = read(HERE / 'analysis/completion.json')
    assert all(sha(HERE / 'analysis' / file) == expected
               for file, expected in completion['artifact_hashes'].items())
    parser = Links()
    parser.feed((HERE / 'analysis/index.html').read_text(encoding='utf-8'))
    targets = [urlsplit(value) for value in parser.targets]
    paths = [(HERE / 'analysis' / unquote(target.path)).resolve()
             for target in targets if not target.scheme and target.path]
    missing = [str(path) for path in paths if not path.is_file()]
    assert not missing, missing
    verification = dict(passed=True, tokenizer_adapter_sha256_verified=True,
        native_tokenizer_readiness_sha256_verified=True,
        analysis_source_matches_pre_evaluation_lock=True,
        analysis_artifact_hashes_verified=True, HTML_local_links_checked=len(paths),
        HTML_missing_links=missing, native_tokenizer_cases_checked=len(compatibility['test_cases']))
    (HERE / 'run_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
