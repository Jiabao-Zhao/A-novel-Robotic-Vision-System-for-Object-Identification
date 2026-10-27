"""Execute Fabrica's original pair-sparsity function, with its MIT license."""
import ast
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SOURCE=Path(__file__).with_name('vendor')/'fabrica/util_grasp.py'


def original_pair_sparsity(pairs,spacing_m):
    tree=ast.parse(SOURCE.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='filter_sparse_pairs')
    namespace={'np':np,'cKDTree':cKDTree}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),namespace)
    pairs=np.asarray(pairs,dtype=float).reshape(-1,2,3)
    if not len(pairs):
        return pairs
    return namespace['filter_sparse_pairs'](pairs,spacing_m)


def canonical_pairs(pairs):
    pairs=np.asarray(pairs,dtype=float).copy().reshape(-1,2,3)
    for i,pair in enumerate(pairs):
        if tuple(pair[1])<tuple(pair[0]):
            pairs[i]=pair[::-1]
    return pairs
