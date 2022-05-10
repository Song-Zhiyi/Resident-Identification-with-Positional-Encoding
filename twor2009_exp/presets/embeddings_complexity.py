from .. import lib
from ..lib import *

def _test(X, y, exp_name, repeat, walk_n, walk_len, dimension, window_size):
    emb_path = Path(f"preprocessed/emb/kyoto-layout1/node2vec-{walk_n}-{walk_len}-{dimension}-{window_size}")
    print(emb_path)
    if not emb_path.exists():
        os.system((
            "python node2vec_train.py --target preprocessed/graph/kyoto-layout1-full-pruned-prob.pkl "
            f"-J 8 --name kyoto-layout1 --dimensions {dimension} --walks {walk_n} "
            f"--walk-length {walk_len} --window-size {window_size} --quiet"))

    return exp_embeddings(X=X, y=y, exp_name=exp_name, emb_path=emb_path,
                          data_name="twor2009-full", n_repeats=repeat)

_SETUPS = [
#   n,    len,  dim, w

# seq len
    #(2000, 700, 256, 5),
    #(1000, 700, 256, 5),
    (500, 700, 256, 5),
    (200, 700, 256, 5),
    (50, 700, 256, 5),
    (5, 700, 256, 5),

# window size
    (1000, 700, 256, 1),
    #(1000, 700, 256, 5),
    #(1000, 700, 256, 20),
    #(1000, 700, 256, 50),
    #(1000, 700, 256, 100),
    #(1000, 700, 256, 200),

# dimension size
    #(1000, 700, 256, 5),
    (1000, 700, 128, 5),
    (1000, 700, 64, 5),
    (1000, 700, 32, 5),

# walk n
    #(1000, 700, 256, 5),
    (1000, 400, 256, 5),
    (1000, 100, 256, 5),
    (1000, 20, 256, 5),
]

def build_preset(exp_name: str, repeat: int=N_REPEATS):
    dat = lib.dat[X_names + [y_name]].dropna(axis=0)
    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)

    exps: list[Experiment] = []
    for s in _SETUPS:
        exps.append(
            _test(X, y, exp_name, repeat, *s)
        )
    return exps
