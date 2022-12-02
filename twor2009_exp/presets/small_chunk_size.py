from .. import lib
from pathlib import Path

import numpy as np
from ..lib import X_names, y_name, Experiment, exp_embeddings

def build_preset(exp_name: str, repeat: int=lib.N_REPEATS):
    dat = lib.dat[X_names + [y_name]].dropna(axis=0)
    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)

    emb_path = Path("preprocessed/emb/kyoto-layout1/node2vec-700-1000-256-5")

    return [
        exp_embeddings(
            X=X, y=y, exp_name=exp_name, emb_path=emb_path,
            chunk_size=2000,
            data_name="twor2009-full-chunk-2000", n_repeats=repeat
        ),
        exp_embeddings(
            X=X, y=y, exp_name=exp_name, emb_path=emb_path,
            chunk_size=1000,
            data_name="twor2009-full-chunk-1000", n_repeats=repeat
        ),
        exp_embeddings(
            X=X, y=y, exp_name=exp_name, emb_path=emb_path,
            chunk_size=500,
            data_name="twor2009-full-chunk-500", n_repeats=repeat
        ),
        exp_embeddings(
            X=X, y=y, exp_name=exp_name, emb_path=emb_path,
            chunk_size=200,
            data_name="twor2009-full-chunk-200", n_repeats=repeat
        ),
    ]