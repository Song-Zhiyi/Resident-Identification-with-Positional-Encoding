from .. import lib
from ..lib import *

def build_preset(exp_name: str, repeat: int=N_REPEATS):
    dat = lib.dat[X_names + [y_name]].dropna(axis=0)
    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)

    coord = lib.load_embedding_file("preprocessed/emb/kyoto-layout1/coord")
    cluster = lib.load_embedding_file("preprocessed/emb/kyoto-layout1/cluster")

    return [
        lib.Experiment(X=X, y=y, experiment_id=exp_name, embeddings=None, embeddings_name="none", data_name="twor2009-full", n_repeats=repeat,),
        lib.Experiment(X=X, y=y, experiment_id=exp_name, embeddings=coord, embeddings_name="coord", data_name="twor2009-full", n_repeats=repeat,),
        lib.Experiment(X=X, y=y, experiment_id=exp_name, embeddings=cluster, embeddings_name="cluster", data_name="twor2009-full", n_repeats=repeat,),
    ]

