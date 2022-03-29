import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from ..data_preprocessing.twor2009 import get_data

d = get_data()
print(d)

class TransformerModel(nn.Module):
    def __init__(self,
                 n_token: int,
                 dim_model: int,
                 n_head: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.2,
                 transformer_layers: int = 1) -> None:
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(dim_model, n_head, dim_feedforward, dropout),
            transformer_layers
        )

        self.encoder = nn.Embedding(n_token, dim_model)
        self.dim_model = dim_model
        self.decoder = nn.Linear(dim_model, n_token)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor):
        src_mask = gen_mask(src.size(0))
        src = self.encoder(src) * math.sqrt(self.dim_model)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


import functools
@functools.lru_cache(maxsize=20)
def gen_mask(x: int):
    return torch.triu(torch.ones(x, x) * -torch.inf, diagonal=1)

import numpy as np

# time, sensor_id, status, tag
toy_time_seq = np.array([
    [1, 1, 1, 0],
    [2, 0, 1, 1],
    [3, 1, 1, 0],
    [4, 0, 0, 1],
    [5, 1, 1, 0],
    [6, 1, 1, 0],
    [7, 1, 0, 0],
], dtype=np.int64)


toy_emb = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
], dtype=np.int64)


from torch.utils.data import TensorDataset, DataLoader

def col_i_is_label(mat: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    label = mat[:,i:i+1]
    data = np.concatenate((mat[:, :i], mat[:, i+1:]), axis=1)
    return data, label

X, Y = col_i_is_label(toy_time_seq, 3)

def concat_pos(mat: np.ndarray, idx: int, emb: np.ndarray):
    data, i = col_i_is_label(mat, idx)
    i = i.reshape(-1)
    return np.concatenate((data, emb[i]), axis=1)

X = concat_pos(X, 2, toy_emb)


X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

data = TensorDataset(X, Y)

model = TransformerModel(n_token=8, dim_model=6)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()

for x, y in DataLoader(data):
    v = model(x)

