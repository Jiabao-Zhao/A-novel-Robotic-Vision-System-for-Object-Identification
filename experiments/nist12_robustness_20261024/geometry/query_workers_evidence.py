"""Literal evidence hashes chained through the final dynamic Python-source lock."""
from pathlib import Path
from common import path, digest

EVIDENCE = {
    'experiments/nist12_robustness_20261024/evaluation/query_workers/installed_build.json': 'f04e46526ad5177a907c1133f31e150f7a391f461f87616796e0e27f44252785',
    'experiments/nist12_robustness_20261024/evaluation/query_workers/expanded.json': '97e32b96cf8d51f8fb284d5642f0ee29a2bf36fbe7699b86a06582b732c4792e',
    'experiments/nist12_robustness_20261024/evaluation/query_workers/four_process_throughput.json': '9e13da802f16b0e0b7c987fe1d30605cc10f863a4accf062ea83546b510e9e41',
    'experiments/nist12_robustness_20261024/geometry/query_workers_parity_protocol.json': '1794ec18ba98112aee3d51e8c5dae7fcb8c16499e425405c7bcf661f335bd3d7',
    'experiments/nist12_robustness_20261024/geometry/query_workers_parity_completion.json': 'fb7739c9d6a281045e706d2a96b8b8671d480ef4187dc8138e05254536db6af4',
    'experiments/nist12_robustness_20261024/geometry/query_proxy_adapter.json': '984522fc0185c5f50f5ffcc2214919b2cffa382b5d1d102ba4cc806bb7144f08',
    'experiments/nist12_robustness_20261024/geometry/query_workers.py': 'd60d69a6449208c839905234940300daa5be8deec3c4765ced99a8a395f62e07',
}
SCIPY_VERSION = '1.18.1'
NUMPY_VERSION = '2.2.6'
SCIPY_EXTENSION_SHA256 = '29316ed1bf1f2b1c4915740fcfe8014df17524a6537b707c1f1347b51c75ddf4'


def verify():
    import numpy as np
    import scipy
    import scipy.spatial._ckdtree as extension
    assert scipy.__version__ == SCIPY_VERSION
    assert np.__version__ == NUMPY_VERSION
    assert digest(Path(extension.__file__)) == SCIPY_EXTENSION_SHA256
    for file, expected in EVIDENCE.items():
        assert digest(path(file)) == expected, file
    return {'anchor_sha256': digest(__file__), 'receipt_count': len(EVIDENCE),
            'scipy_extension_sha256': SCIPY_EXTENSION_SHA256}
