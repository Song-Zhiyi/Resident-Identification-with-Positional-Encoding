from src.models.models import Trainer, TrainMetadata, LSTMTagger, prepare_tensor_dataset

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data_prep import dat, X_names, y_name

# default epoch
N_EPOCH=40
# default repeat
N_REPEATS=5

def load_embedding_file(path):
    emb = np.load(path)
    return emb['arr_0']

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
            n_repeats: int=N_REPEATS,
            n_epoch: int=N_EPOCH,
            validation_ratio: float=0.25,
            save_dir: os.PathLike="result",
            progress: bool=True,
            progress_pos: int=2,
            drop_out: float=None,
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

            if embeddings is not None:
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
            save_model=False,
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


def exp_embeddings(
        X, y,
        exp_name: str,
        emb_path: Path,
        data_name: str,
        n_epoch: int=N_EPOCH,
        n_repeats: int=N_REPEATS,
        chunk_size: int=3000,
        progress_pos: int=2,
        drop_out: float=None):
    emb = load_embedding_file(emb_path)
    exp = Experiment(
        X=X, y=y, embeddings=emb,
        experiment_id=exp_name,
        data_name=data_name,
        embeddings_name=emb_path.name,
        chunk_size=chunk_size,
        n_epoch=n_epoch,
        n_repeats=n_repeats,
        progress_pos=progress_pos,
        drop_out=drop_out,
    )
    return exp

def exp_node2vec_complexity(
        X, y,
        exp_name: str, data_name: str, n_epoch=N_EPOCH, n_repeats=N_REPEATS, progress_pos: int=2):

    exps: list[Experiment] = []
    for emb_path in Path("preprocessed/emb/kyoto-layout1").iterdir():
        exps.append(
            exp_embeddings(X=X, y=y, exp_name=exp_name, emb_path=emb_path,
                           data_name=data_name, n_epoch=n_epoch, n_repeats=n_repeats,
                           progress_pos=progress_pos)
        )
    return exps

def select_timeframe(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[
        df['datetime'].between(start, end)
    ]

def make_data(dat: pd.DataFrame):
    dat = dat[X_names + [y_name]].dropna(axis=0)
    X = dat[X_names].to_numpy(dtype=np.float32)
    y = dat[y_name].to_numpy(dtype=np.int64)
    return X, y


def start_preset(exps: list[Experiment]):
    for e in tqdm(exps, desc="Experiment setup", unit="setup", leave=False, position=1):
        e.start()

import importlib
import sys
def load_preset():
    result = {}
    spec = sys.modules[__name__].__spec__
    parent = spec.parent
    for preset_path in (Path(spec.origin).parent / "presets").iterdir():
        name = preset_path.stem
        if name.startswith("_"):
            continue
        mod = importlib.import_module(f"{parent}.presets.{name}")
        result[name] = mod
    return result