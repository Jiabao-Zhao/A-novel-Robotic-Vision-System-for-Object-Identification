"""Import the frozen prior geometry implementation without changing its source."""
from pathlib import Path
import hashlib
import importlib.util
import sys

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/geometry.py'
EXPECTED_SHA256 = 'bf3120854a2fe0682d3c9723b1e6786b1150bda74e783550d41eb83c9e80b4cc'


def load_native():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA256
    name = 'frozen_nist12_depth_geometry'
    if name not in sys.modules:
        sys.path.insert(0, str(REPO))
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
