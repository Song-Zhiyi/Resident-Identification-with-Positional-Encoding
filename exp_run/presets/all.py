from .. import lib
from ..lib import *
from ..data_prep import g
from .. import data_prep

import numpy as np

from src.caching import pickle_load_from_file


def build_sensor_coord_emb():
    name_to_id = g['sensor_name_to_id']
    sensor_coord = g['sensor_coord']

    id_to_name = {i: n for n, i in name_to_id.items()}

    l = []
    last_i = -1
    for i, n in sorted(id_to_name.items()):
        if i - last_i != 1:
            raise Exception("Skipping index")
        else:
            last_i = i
        l.append(
            sensor_coord[n]
        )

    return np.array(l, dtype=np.float32)


def build_preset_from_dat(data: pd.DataFrame, exp_name: str, repeat: int):
    data: pd.DataFrame = data[X_names + [y_name]].dropna(axis=0)

    X = data[X_names].to_numpy(dtype=np.float32)
    y = data[y_name].to_numpy(dtype=np.int64)

    X_a, X_b = np.array_split(X, 2)
    y_a, y_b = np.array_split(y, 2)

    coord_emb = build_sensor_coord_emb()

    CHUNK_SIZE = 200
    EPOCH = 100
    DATA_TRAIN_DUP = 8
    DROP_OUT = None

    common_kws = dict(
        n_repeats=repeat, chunk_size=CHUNK_SIZE, n_epoch=EPOCH,
        drop_out=DROP_OUT, data_train_dup=DATA_TRAIN_DUP
    )

    def _data(data, label, name):
        exps = []
        exps += [
            Experiment(
                X=data, y=label,
                data_name=name,
                experiment_id=exp_name,
                embeddings=None, embeddings_name="none",
                **common_kws
            ),
            Experiment(
                X=data, y=label,
                data_name=name,
                experiment_id=exp_name,
                embeddings=coord_emb, embeddings_name="coord",
                **common_kws
            ),
        ]
        for emb_path in Path("preprocessed/emb/Exp").iterdir():
            emb = load_embedding_file(emb_path)
            exps.append(
                Experiment(
                    X=data, y=label,
                    data_name=name,
                    experiment_id=exp_name,
                    embeddings=emb, embeddings_name=emb_path.name,
                    **common_kws
                )
            )

        return exps

    exps: list[Experiment] = []
    exps += _data(X, y, "myexp-full")
    exps += _data(X_a, y_a, "myexp-half-a")
    exps += _data(X_b, y_b, "myexp-half-b")

    return exps


def build_preset(exp_name: str, repeat: int = N_REPEATS):
    return build_preset_from_dat(lib.dat, exp_name, repeat)