from __future__ import annotations
import logging
import time
import os
import pickle
import json

from pathlib import Path
from datetime import datetime
from typing import Callable
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sklearn.metrics as metrics
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def concat_v2v(all_data, sensor_idx, vv):
    result = []
    for row in all_data:
        v2v = vv[int(row[sensor_idx])]
        row = np.concatenate((row, v2v))
        result.append(row)
    return np.array(result)

def data_chunk(X, Y, chunk_size):
    n, n_feats = X.shape
    Xs = []
    Ys = []
    for i in range(n // chunk_size):
        i_start = i * chunk_size
        i_end = i_start + chunk_size

        Xs.append(X[i_start:i_end])
        Ys.append(Y[i_start:i_end])

    if n % chunk_size > chunk_size / 2:
        Xs.append(X[-chunk_size:])
        Ys.append(Y[-chunk_size:])

    return np.array(Xs), np.array(Ys)

def prepare_tensor_dataset(X: np.ndarray, Y: np.ndarray, chunk_size: int, validation_ratio: float, shuffle: bool=False):
    X_train, X_test, Y_train, Y_test = train_test_split(
        *data_chunk(X, Y, chunk_size), test_size=validation_ratio, shuffle=False)
    train_data = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_data = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(Y_test))

    return train_data, test_data

@dataclass
class TrainMetadata(object):
    data_name: str
    embeddings_name: str
    embeddings: np.ndarray
    chunk_size: int
    n_epoch: int
    repeats: int = 0

    def safe_id(self) -> str:
        return ""


from dataclasses import field
@dataclass
class EpochResult(object):
    epoch: int
    n_train: int
    n_val: int
    ns_start: int = 0
    ns_end: int = 0
    train_loss: float = 0
    val_loss: float = 0

    # weighted precision, recall, f1. weighted recall == accuracy
    train_precision: float = 0
    train_recall: float = 0
    train_f1: float = 0
    val_precision: float = 0
    val_recall: float = 0
    val_f1: float = 0

    train_x_true: list[np.ndarray] = field(default_factory=list)
    train_x_pred: list[np.ndarray] = field(default_factory=list)
    val_x_true: list[np.ndarray] = field(default_factory=list)
    val_x_pred: list[np.ndarray] = field(default_factory=list)
    _done: bool=False

    def asdict(self):
        d = asdict(self)
        del d["train_x_true"]
        del d["train_x_pred"]
        del d["val_x_true"]
        del d["val_x_pred"]
        del d["_done"]
        return d

    def update_metrics(self):
        if self._done:
            return
        self._done = True
        if len(self.train_x_true) > 0 and len(self.train_x_pred) > 0:
            self.train_precision, self.train_recall, self.train_f1, _ = (
                metrics.precision_recall_fscore_support(
                    np.concatenate(self.train_x_true),
                    np.concatenate(self.train_x_pred),
                    average="weighted"
                )
            )
        if len(self.val_x_true) > 0 and len(self.val_x_pred) > 0:
            self.val_precision, self.val_recall, self.val_f1, _ = (
                metrics.precision_recall_fscore_support(
                    np.concatenate(self.val_x_true),
                    np.concatenate(self.val_x_pred),
                    average="weighted"
                )
            )

    def start_profiling(self):
        self.ns_start = time.perf_counter_ns()

    def end_profiling(self):
        self.ns_end = time.perf_counter_ns()

    def __enter__(self):
        self.start_profiling()
        return self

    def __exit__(self, exc_type, exc_info, traceback):
        self.end_profiling()
        self.update_metrics()
        return exc_type is None

    def __str__(self):
        self.update_metrics()
        ns_delta = self.ns_end - self.ns_start
        n_train = self.n_train
        n_val = self.n_val
        train_loss = self.train_loss / n_train
        train_accuracy = self.train_recall
        val_loss = self.val_loss / n_val
        val_accuracy = self.val_recall
        epoch = self.epoch

        s = ns_delta // 1000000000
        ns = ns_delta % 1000000000

        return (
            f"{epoch=:>4} ({s}.{ns}s) "
            f"total_train_loss={self.train_loss:.4f} {train_loss=:.4f} {train_accuracy=:.4f} "
            f"total_val_loss={self.val_loss:.4f} {val_loss=:.4f} {val_accuracy=:.4f}"
        )

class LSTMTagger(nn.Module):
    def __init__(
            self, input_size: int, hidden_size: int, output_size: int,
            loss_function: Callable, bidirectional=True,
            dropout: float = 0, num_layers: int = 1,
            device=default_device):
        super().__init__()
        self.logger = logging.getLogger("e.models.LSTMTagger")
        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
            device=device,
        )

        if bidirectional:
            self._linear = nn.Linear(hidden_size * 2, output_size, device=device)
        else:
            self._linear = nn.Linear(hidden_size, output_size, device=device)

        self.loss_function = loss_function

    def forward(self, X):
        tmp, _ = self._lstm(X)
        tmp = self._linear(tmp)
        return tmp

def remove_pad(predict, expected, pad_val):
    if not any(expected == pad_val):
        return predict, expected
    idx = np.abs(expected - pad_val).argmin()
    return pred[:idx], expected[:idx]


class Trainer(object):
    @staticmethod
    def load_result_table(path: os.PathLike):
        path = Path(path).expanduser().absolute()
        with open(path, "rb") as fp:
            return pickle.load(fp)

    def __init__(self, model: LSTMTagger, train_data: TensorDataset, val_data: TensorDataset,
                 save_path: os.PathLike, optimizer=optim.Adam,
                 device=default_device, lr_scheduler=None, debug=1,
                 train_meta: TrainMetadata=None,
                 train_id=None):
        self.model = model
        self.model.to(default_device)
        self.train_data = train_data
        self.val_data = val_data
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer(params=model.parameters(), lr=0.001)
        self.device = device
        if train_id is None:
            train_id = str(datetime.now())
        self.train_id = train_id
        self.save_path = Path(save_path).expanduser().absolute() / self.train_id
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.train_meta = train_meta

        self.current_epoch = 0
        self.train_result: list[EpochResult] = []
        self.debug = debug

    def __enter__(self):
        return self

    def __exit__(self, *args):
        with open(self.save_path / "results.json", "w") as fp:
            json.dump(
                [r.asdict() for r in self.train_result], fp, indent=4
            )

        with open(self.save_path / "metadata.json", "w") as fp:
            json.dump(asdict(self.train_meta), fp, indent=4)

    def get_best_results(self, method: str="val_loss"):
        if method == "val_loss":
            return list(sorted(self.train_result, key=lambda r: r.val_loss))
        if method == "val_accuracy":
            return list(sorted(self.train_result, key=lambda r: r.val_accuracy))

    def save_best(self, method):
        bests = self.get_best_results(method=method)
        best = bests[0]
        save_path = self.save_path / f"epoch{best.epoch:04d}"
        if not save_path.exists():
            self.model.logger.info(
                (f"saveing currently best model (val_loss={best.val_loss/best.n_val:.4f} "
                f"val_acc={best.val_recall:.4f}) "
                f"of epoch {best.epoch} to {save_path!s}")
            )
            torch.save(best, save_path)

    def after_epoch(self, result):
        self.save_best("val_loss")

    def train(self, epoch: int):
        return self.do_train(epoch)

    def do_train(self, epoch, progress: tqdm=None):
        n_train = len(self.train_data)
        n_val = len(self.val_data)
        for i in range(self.current_epoch, self.current_epoch + epoch):
            with EpochResult(i, n_train, n_val) as result:
                result = self._one_epoch(result)
                self.train_result.append(result)
            self.model.logger.info(str(result))
            self.current_epoch += 1
            if progress is not None:
                progress.update()

            self.after_epoch(result)

    def _one_epoch(self, result: EpochResult):
        train_data = self.train_data
        val_data = self.val_data

        model = self.model
        optimizer = self.optimizer
        loss_function = self.model.loss_function
        device = self.device

        def update_train_metrics(expect, predict):
            result.train_x_pred.append(predict)
            result.train_x_true.append(expect)

        def update_val_metrics(expect, predict):
            result.val_x_pred.append(predict)
            result.val_x_true.append(expect)

        def func(model, data, update_metrics, is_train=True):
            total_loss = 0

            if is_train:
                model.train()
            else:
                model.eval()

            for dat, tag in DataLoader(data):
                dat = dat.to(device)
                tag = tag.to(device)

                if is_train:
                    optimizer.zero_grad()
                tag_space = model(dat)

                loss = loss_function(tag_space.view(-1, tag_space.shape[-1]), tag.view(-1))
                total_loss += loss.item()

                if is_train:
                    loss.backward()
                    optimizer.step()

                predict = torch.argmax(tag_space, -1).view(-1).cpu().numpy()
                expect = tag.view(-1).cpu().numpy()
                update_metrics(expect, predict)

            return total_loss

        train_data = self.train_data
        val_data = self.val_data

        result.train_loss = func(model, train_data, update_train_metrics, is_train=True)
        result.val_loss = func(model, val_data, update_val_metrics, is_train=False)

        return result
