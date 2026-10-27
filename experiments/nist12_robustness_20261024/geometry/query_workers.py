"""Execution-only cKDTree query threading; nearest-neighbor rows stay independent."""


class QueryWorkerTree:
    def __init__(self, tree, workers):
        self.tree = tree
        self.workers = int(workers)

    def query(self, points, *args, **kwargs):
        kwargs['workers'] = self.workers
        return self.tree.query(points, *args, **kwargs)


def install_for_native(native, workers):
    """Install only in a fresh dense-stage worker, without touching fit kernels."""
    original = native.prepare_cad

    def prepare(mesh):
        cad = original(mesh)
        cad.tree = QueryWorkerTree(cad.tree, workers)
        return cad

    native.prepare_cad = prepare
