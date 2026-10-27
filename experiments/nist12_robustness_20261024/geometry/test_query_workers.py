"""Threading leaves all nearest-neighbor distances and indices unchanged."""
import unittest
import numpy as np
from scipy.spatial import cKDTree
from query_workers import QueryWorkerTree


class QueryTests(unittest.TestCase):
    def test_full_arrays_including_exact_and_tied_neighbors(self):
        rng = np.random.default_rng(20261024)
        cad = np.vstack((rng.normal(size=(10000, 3)), np.zeros((2, 3))))
        points = np.vstack((rng.normal(size=(131072, 3)), cad[:100], np.zeros((3, 3))))
        tree = cKDTree(cad)
        serial = tree.query(points, workers=1)
        parallel = QueryWorkerTree(tree, 4).query(points, workers=1)
        for expected, actual in zip(serial, parallel):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
