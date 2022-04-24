# %%
import logging
logging.basicConfig(format="%(levelname)s %(message)s", level="ERROR")

# %%
import src.data_preprocessing.twor2009
from src.models.models import Trainer, TrainMetadata, LSTMTagger, prepare_tensor_dataset
from src.caching import pickle_load_from_file, pickle_dump_to_file
from src.extractor import adj_mat_to_t_mat

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


dat = src.data_preprocessing.twor2009.get_data()
s = pickle_load_from_file("preprocessed/graph/twor2009_sensor_graph.pkl")
dat = dat.merge(
    pd.DataFrame(
        s['sensor_name_to_id'].items(), columns=["sensor", "sensor_id"]
    )
)
pickle_load_from_file("preprocessed/graph/kyoto-layout1-full-pruned-prob.pkl")

# %%

def save_emb(file_name: os.PathLike, emb_array: np.ndarray):
    with open(file_name, "wb") as fp:
        np.savez_compressed(fp, emb_array)

def build_coord_emb(graph_file: os.PathLike, savedir: os.PathLike, force: bool=False):
    if Path(savedir).exists() and not force:
        return

    s = pickle_load_from_file(graph_file)
    sensor_coord = s['sensor_coord']
    sensor_name_to_id = s['sensor_name_to_id']
    l = []
    id_to_name = {
        v: k for k, v in sensor_name_to_id.items()
    }
    for i in range(len(sensor_name_to_id)):
        try:
            coord = sensor_coord[id_to_name[i]]
        except KeyError:
            coord = np.array([0,0,0])

        l.append(coord)

    emb = np.array(l, dtype=np.float32)

    for i, r in enumerate(emb):
        assert (r == sensor_coord[id_to_name[i]]).all()

    save_emb(savedir, emb)

SENSOR_ROOM_MAP = {
    "M47": 0, "M48": 0, "M46": 0, "M49": 0, "M45": 0, "M50": 0, "M41": 0, "D04": 0, "M44": 0,

    "M42": 1, "M43": 1, "M27": 1, "M28": 1, "M29": 1,

    "M37": 2, "M38": 2, "M39": 2, "M40": 2, "D05": 2,

    "M41": 3, "D06": 3,

    "M30": 4, "M31": 4, "M32": 4, "M33": 4, "M34": 4, "M35": 4, "M36": 4, "D03": 4,

    "M01": 5, "M02": 5, "M03": 5, "M04": 5, "M05": 5, "M06": 5, "M07": 5, "M08": 5,
    "M09": 5, "M10": 5, "M11": 5, "M12": 5, "M13": 5, "M14": 5, "M15": 5, "D02": 5,

    "M21": 6, "M22": 6, "M23": 6, "M24": 6, "M25": 6, "M26": 6, "D01": 6,

    "M19": 7,

    "M20": 8,

    "M51": 9, "D11": 9,

    "M16": 10, "M17": 10, "M18": 10, "D08": 10, "D09": 10, "D10": 10,

    "D12": 11,
}

def build_cluster_emb(graph_file: os.PathLike, savedir: os.PathLike, force: bool=False):
    if Path(savedir).exists() and not force:
        return

    s = pickle_load_from_file(graph_file)
    sensor_name_to_id = s['sensor_name_to_id']
    l = []
    id_to_name = {
        v: k for k, v in sensor_name_to_id.items()
    }

    unknown_id = len(SENSOR_ROOM_MAP)

    for i in range(len(sensor_name_to_id)):
        try:
            cluster_id = SENSOR_ROOM_MAP[id_to_name[i]]
        except KeyError:
            cluster_id = unknown_id

        l.append(cluster_id)

    emb = np.array(l, dtype=np.float32)

    n, = emb.shape
    emb = emb.reshape((n, 1))

    for i, r in enumerate(emb):
        assert (r == SENSOR_ROOM_MAP[id_to_name[i]]).all()

    save_emb(savedir, emb)

build_coord_emb(
    "preprocessed/graph/twor2009_sensor_graph.pkl",
    "preprocessed/emb/kyoto-layout1/coord",
)

build_cluster_emb(
    "preprocessed/graph/twor2009_sensor_graph.pkl",
    "preprocessed/emb/kyoto-layout1/cluster",
)


# %%

def load_embedding_file(path):
    emb = np.load(path)
    return emb['arr_0']

dat = dat[dat['resident_id'] >= 0]
dat.sort_values(by="datetime", inplace=True)
X_names = ['weekday_sin', 'weekday_cos', 'dayofyear_sin', 'dayofyear_cos', 'secondofday_sin', 'secondofday_cos', 'read_bin', 'sensor_id']
y_name = 'resident_id'

dat = dat[X_names + [y_name]].dropna(axis=0)

dat[:4*3000]

X = dat[X_names].to_numpy(dtype=np.float32)
Y = dat[y_name].to_numpy(dtype=np.int64)

# %%


def concat_embeddings(dat: np.ndarray, embeddings: np.ndarray, keyidx: int=-1):
    result = []
    for row in dat:
        emb = embeddings[int(row[keyidx])]
        row = np.concatenate((row, emb))
        result.append(row)
    return np.array(result)

@dataclass
class TrainMetadata(object):
    data_name: str
    embeddings_name: str
    chunk_size: int
    n_epoch: int
    repeat: int = 0

    def safe_id(self):
        return f"{self.data_name}.{self.embeddings_name}.{self.repeat}"

class Experiment:
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            embeddings: np.ndarray,
            experiment_id: str,
            data_name: str,
            embeddings_name: str,
            keyidx: int=-1,
            chunk_size: int = 3000,
            n_repeats: int=1,
            n_epoch: int=50,
            validation_ratio: float=0.25,
            save_dir: os.PathLike="result",
            progress: bool=True,
            progress_pos: int=2,
        ):
            self.exp_save_dir = Path(save_dir).expanduser() / experiment_id
            self.logger = logging.getLogger("e.experiment")
            self.experiment_id = experiment_id
            self.metas = []
            for repeat_id in range(n_repeats):
                self.metas.append(
                    TrainMetadata(
                        data_name=data_name,
                        embeddings_name=embeddings_name,
                        chunk_size=chunk_size,
                        n_epoch = n_epoch,
                        repeat=repeat_id,
                    )
                )

            X = concat_embeddings(X, embeddings, keyidx)

            self.n, self.input_size = X.shape
            self.n_class = len(np.unique(y))
            self.dat_train, self.dat_val = prepare_tensor_dataset(X, y, chunk_size, validation_ratio)

            self.show_progress = progress
            self.progress_pos = progress_pos

    def _one_experiment(self, meta: TrainMetadata):
        self.logger.debug(f"train with input_size={self.input_size} n_class={self.n_class}")

        progress = tqdm(
            desc=meta.safe_id(),
            unit="epoch",
            total=meta.n_epoch,
            disable=not self.show_progress,
            position=self.progress_pos+1,
            leave=False,
        )
        with Trainer(
            model=LSTMTagger(input_size=self.input_size, hidden_size=256,
                             output_size=self.n_class, loss_function=nn.CrossEntropyLoss()),
            train_data=self.dat_train,
            val_data=self.dat_val,
            optimizer=optim.Adam,
            save_path=(self.exp_save_dir),
            train_meta=meta,
            train_id=meta.safe_id(),
        ) as trainer:
            trainer.do_train(meta.n_epoch, progress)
            return trainer.train_id

    def start(self):
        for meta in tqdm(
            self.metas,
            desc=self.experiment_id,
            disable=not self.show_progress,
            position=self.progress_pos,
            leave=False,
            unit="repeat",
        ):
            train_id = self._one_experiment(meta)



# %%
N_EPOCH=40
N_REPEATS=5

def exp_node2vec_complexity(
        X, Y,
        exp_name: str, data_name: str, n_epoch=N_EPOCH, n_repeats=N_REPEATS):

    exps: list[Experiment] = []
    for emb_path in Path("preprocessed/emb/kyoto-layout1").iterdir():
        if emb_path.name != "cluster":
            continue
        emb = load_embedding_file(emb_path)
        exp = Experiment(
            X=X, y=Y, embeddings=emb,
            experiment_id=exp_name,
            data_name=data_name,
            embeddings_name=emb_path.name,
            n_epoch=n_epoch,
            n_repeats=n_repeats,
            progress_pos=1
        )
        exps.append(exp)
    return exps

exps = exp_node2vec_complexity(X, Y, "Node2Vec-Complexity", "twor2009-full")

for e in tqdm(exps, desc="Experiment setup", unit="setup", leave=False):
    e.start()

# number of data, time ...
# different embeddings ...
# different embedding strategy


