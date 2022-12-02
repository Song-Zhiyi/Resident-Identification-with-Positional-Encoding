from .. import lib
from ..lib import N_REPEATS
from datetime import datetime
from pathlib import Path
import pandas as pd


def build_preset(exp_name: str, repeat: int=N_REPEATS):
    EMB_PATH = Path("preprocessed/emb/kyoto-layout1/node2vec-700-1000-256-5")

    def in_time(data_name: str, start: datetime=pd.Timestamp.min, end: datetime=pd.Timestamp.max):
        dat = lib.dat
        dat = dat[(
            (dat['datetime'] >= start)
            & (dat['datetime'] < end))
        ]
        X, y = lib.make_data(dat)
        return lib.exp_embeddings(
            X=X, y=y, exp_name=exp_name, emb_path=EMB_PATH,
            data_name=data_name, n_epoch=lib.N_EPOCH, n_repeats=repeat)

    return [
        in_time("twor2009-march", start=datetime(2009, 3, 1)),
        in_time("twor2009-feb", end=datetime(2009, 3, 1)),
        in_time("twor2009-10d-01", start=datetime(2009, 2, 2),
                end=datetime(2009, 2, 12)),
        in_time("twor2009-10d-02", start=datetime(2009, 2, 12),
                end=datetime(2009, 2, 22)),
        in_time("twor2009-10d-03", start=datetime(2009, 3, 1),
                end=datetime(2009, 3, 10)),
        in_time("twor2009-10d-04", start=datetime(2009, 3, 10),
                end=datetime(2009, 3, 20)),
        in_time("twor2009-10d-05", start=datetime(2009, 3, 20),
                end=datetime(2009, 3, 30)),
    ]
