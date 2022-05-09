from .. import lib
from ..lib import *

#TODO: change it back

def build_preset(exp_name: str, repeat: int=N_REPEATS):
    dat = lib.dat[X_names + [y_name]].dropna(axis=0)
    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)

    exps: list[Experiment] = []
    for emb_path in Path("preprocessed/emb/kyoto-layout1").iterdir():
        exps.append(
            exp_embeddings(X=X, y=y, exp_name=exp_name, emb_path=emb_path,
                           data_name="twor2009-full", n_repeats=repeat)
        )
    return exps
