#%%
import pandas as pd
from pathlib import Path
import re
import numpy as np
import os


from IPython.display import display

ARAS_DATA_HEADER = [
    "Ph1", "Ph2", "Ir1", "Fo1", "Fo2", "Di3", "Di4",
    "Ph3", "Ph4", "Ph5", "Ph6", "Co1", "Co2", "Co3",
    "So1", "So2", "Di1", "Di2", "Te1", "Fo3", "R1", "R2",
]

def _transform_generator(dat: np.ndarray):
    # index 20, 21 are annotations
    # index 22 is # of day
    # index 23 is # of second

    # yielding (5,) array [R1 activity, R2 activity, day, second, sensor id]


    for r in dat:
        for i in np.argwhere(r[:20]):
            yield np.append(r[20:], i)

def _load_single(p: Path):
    daynum = int(p.stem.split("_")[1])
    dat: np.ndarray = np.loadtxt(p, dtype=np.int32)
    nrow, ncol = dat.shape

    dat = np.concatenate(
        (
            dat,
            np.zeros((nrow, 1), dtype=np.int32) + daynum,
            np.arange(nrow, dtype=np.int32).reshape((nrow, 1)),
        ), axis=1)

    dat = np.array(list(_transform_generator(dat)))
    return dat


def _do_load_and_transform_data(data_path: Path):
    dat = []
    for f in data_path.iterdir():
        if f.suffix != ".txt":
            continue
        tab = _load_single(f)
        dat.append(tab)

    dat: np.ndarray = np.concatenate(dat)
    dat: pd.DataFrame = pd.DataFrame(dat, columns=["R1", "R2", "day", "second", "sid"])
    dat.sort_values(by=["day", "second"])
    return dat


#%%

def load_df(f: os.PathLike):
    return pd.read_feather(f)

def dump_df(f: os.PathLike, df: pd.DataFrame):
    df.to_feather(f, compression="zstd")


def get_data(force: bool=False,
             data_path="./data/Aras/House A",
             cache_path=".cache/data/aras-a-preprocessed.feather"):

    data_path = Path(data_path).expanduser().absolute()
    cache_path = Path(cache_path).expanduser().absolute()

    try:
        if not force:
            return load_df(cache_path)
    except OSError as e:
        print("Failed to load cache")

    dat = _do_load_and_transform_data(data_path)
    dump_df(cache_path, dat)
    return dat