#%%

import numpy as np

# graph of n vertices x m edges

# adjacent matrix ndarray n x n
mat = np.array([
    [1, 0, 0],
    [0.5, 1, 0.5],
    [0.5, 0, 1],
])

# array of size n x 2
sampling_strategy = np.array([
    [1, 1],
    [0, 0],
    [1, 1],
])

class Node2Vec:
    def __init__(self, graph: np.ndarray, p: float, q: float) -> None:
        assert 0 <= p <= 1
        assert 0 <= q <= 1
        self.mat = graph

    def _normalize_mat(self):
        self.mat = np.array(
            list((r / np.sum(r)) for r in self.mat)
        )

    def _init_default_sampling_strategy(self):
        pass

def check_params(mat: np.ndarray, sampling_strategy: np.ndarray):
    assert len(mat.shape) == 2
    n, n = mat.shape
    assert sampling_strategy.shape == (n, 2)

def normalize_mat(mat: np.ndarray):
    for r in mat:
        pass
    pass

check_params(mat, sampling_strategy)