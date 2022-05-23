from .. import lib
from ..lib import *

def build_preset(exp_name: str, repeat: int=N_REPEATS):
    dat = lib.dat[X_names + [y_name]].dropna(axis=0)
    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)

    return [
        Experiment(
            X=X, y=y, embeddings=None, experiment_id=exp_name, data_name="twor2009-full",
            embeddings_name="none", n_repeats=repeat
        )
    ]
