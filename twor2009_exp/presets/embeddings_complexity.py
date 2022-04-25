from ..lib import *

def build_preset(exp_name: str, repeat: int=N_REPEATS):
    X = dat[X_names].to_numpy(dtype=np.float32)
    Y = dat[y_name].to_numpy(dtype=np.int64)
    return exp_node2vec_complexity(X, Y, exp_name=exp_name, data_name="twor2009-full", n_repeats=repeat)
