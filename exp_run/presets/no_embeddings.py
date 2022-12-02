from .. import lib
from ..lib import *

import numpy as np

def build_preset(exp_name: str, repeat: int=N_REPEATS):
    dat = lib.dat[X_names + [y_name]].dropna(axis=0)

    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)

    exps: list[Experiment] = []
    exps.append(
        Experiment(
            X=X, y=y, experiment_id=exp_name, embeddings=None, embeddings_name="none",
            data_name="myexp-full", n_repeats=repeat, chunk_size=200, n_epoch=100,
            drop_out=0.1, data_train_dup=8
        )
    )
    return exps



