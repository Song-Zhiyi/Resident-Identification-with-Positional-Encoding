from functools import lru_cache

import numpy as np
import numpy.linalg as npl
from typing import Literal

def adj_mat_to_t_mat(adj_mat: np.ndarray, diag_weight: float=0.5, dis_method: Literal["inv"]="inv") -> np.ndarray:
    x, x = adj_mat.shape
    mat = adj_mat.copy()
    I = np.identity(x)

    mask = np.int32(mat != 0)
    # remove diagonal
    mat *= (1 - I)

    # ...

    if diag_weight is None:
        diag_weight = 1 / x

    if dis_method == "inv":
        mat = (1 / (mat + 1)) * mask
    elif dis_method == "minus":
        mat = (mat.sum(axis=1, keepdims=True) - mat) * mask
    else:
        raise ValueError(f"Unsupported distance method {dis_method}")
    # 1st normalize
    mat /= mat.sum(axis=1, keepdims=True)
    # add diagonal
    mat += (I * (diag_weight / (1-diag_weight)))

    # 2nd normalize
    mat /= mat.sum(axis=1, keepdims=True)
    return mat

class ResidentExtractor:
    def __init__(self, t_mat: np.ndarray, initial_state: list[int], threshold: float=0, max_it: int=20):
        self._t_mat = t_mat = t_mat.copy()
        self._threshold = threshold
        self._max_it = max_it
        self._state = initial_state
        self._n = 0

        @lru_cache()
        def t_mat_cached(i: int) -> np.ndarray:
            return npl.matrix_power(t_mat, i)

        self._t_mat_cached = t_mat_cached

    def set_state(self, rid: int, sid: int):
        self._state[rid] = sid

    def extract_once(self, current_sid: int):
        self._n += 1
        it = 1
        while True:
            if it > self._max_it:
                raise RuntimeError("Max iterations exceeded")

            tm = self._t_mat_cached(it)


            probs = np.fromiter(
                (tm[r, current_sid] for r in self._state),
                dtype=tm.dtype,
                count=len(self._state)
            )
            r = np.argmax(probs)

            if probs[r] > self._threshold:
                break

            it += 1

        prob_max = probs[r]
        max_idx = np.argwhere(probs == prob_max)
        if len(max_idx) > 1:
            print(f"HERE {self._n} {probs} {self._state} {current_sid}")

        self._state[r] = current_sid
        return r

