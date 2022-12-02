import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import datetime as dt

def load_annotated(dir: str="experiment/data/annotated/") -> pd.DataFrame:
    dats = []
    for f in Path(dir).iterdir():
        if f.suffix != ".csv":
            continue
        dats.append(
            pd.read_csv(f, parse_dates=["last_update"])
        )
    dat = pd.concat(dats)
    dat = dat[['collected_at', 'sensor_name', 'last_update', 'trigger', 'annotation']]
    return dat

def prepare_data(dat: pd.DataFrame) -> pd.DataFrame:
    dat = dat.copy()
    dat = dat[~pd.isna(dat['annotation'])]
    dat = dat[dat['annotation'].str.endswith("r")]
    dat = dat[['last_update', 'sensor_name', 'annotation']]
    dat['annotation'] = dat['annotation'].apply(lambda x: int(x[0]))
    return dat

def cyclical_time(dat: pd.DataFrame, target_col: str):

    def apply_cyclical_time(df, col, max_t):
        ang = 2 * np.pi * df[col] / max_t
        df[col + "_sin"] = np.sin(ang)
        df[col + "_cos"] = np.cos(ang)

    def seconds_of_day(hour, minute, second):
        return hour * 3600 + minute * 60 + second

    dat_dt = dat[target_col].dt
    dat['year'] = dat_dt.year
    dat['dayofyear'] = dat_dt.dayofyear - 1
    dat['secondofday'] = seconds_of_day(dat_dt.hour, dat_dt.minute, dat_dt.second) - 1
    dat['weekday'] = dat_dt.weekday

    apply_cyclical_time(dat, 'weekday', 7)
    apply_cyclical_time(dat, 'dayofyear', 365)
    apply_cyclical_time(dat, 'secondofday', 86400)

    return dat

@dataclass()
class ChunkInfo:
    sensor_name: str
    annotation: int

    sampling_rate: dt.timedelta=dt.timedelta(seconds=300)

    chunks: list[list[int]] = field(default_factory=list)
    chunk_i: int = 0
    in_chunk: bool = False
    begin: Optional[dt.datetime] = None
    current: Optional[dt.datetime] = None

    @property
    def chunk(self):
        try:
            return self.chunks[self.chunk_i]
        except IndexError:
            self.chunks.append(list())
            return self.chunks[self.chunk_i]

    def begin_chunk(self, l: dt.datetime):
        self.begin = l
        self.current = l
        self.in_chunk = True

    def end_chunk(self):
        self.begin = None
        self.current = None
        self.in_chunk = False
        self.chunk_i += 1

    def without_sampling_interval(self, t: dt.datetime):
        return t - self.current > self.sampling_rate

    def check(self, index: int, sensor_name: str, last_update: dt.datetime):
        if self.in_chunk:
            if sensor_name == self.sensor_name:
                if self.without_sampling_interval(last_update):
                    self.end_chunk()
            else:
                self.end_chunk()
        else:
            if sensor_name == self.sensor_name:
                self.begin_chunk(last_update)

        self.chunk.append(index)

def down_sample_stationary_data(dat: pd.DataFrame, infos: dict[int, ChunkInfo]) -> pd.DataFrame:
    for i, row in dat.iterrows():
        r = row['annotation']
        s = row['sensor_name']
        l = row['last_update'].to_pydatetime()
        infos[r].check(i, s, l)

    index_to_remove = set()
    for i, info in infos.items():
        for chunk_i, chunk in enumerate(info.chunks):
            index_to_remove.update(chunk[1:])

    return dat.drop(index=index_to_remove)


def up_sampling(dat: pd.DataFrame, n: int) -> pd.DataFrame:
    dups: list[pd.DataFrame] = []
    offset = dt.timedelta(days=28)
    for i in range(n):
        dat_copy = dat.copy()
        dat_copy['last_update'] += (i * offset)
        dat_copy = cyclical_time(dat_copy, "last_update")
        dups.append(dat_copy)

    return pd.concat(dups)

from src.caching import pickle_load_from_file

dat = load_annotated()
dat = prepare_data(dat)

g = pickle_load_from_file("preprocessed/graph/exp-full-pruned-prob.pkl")


def apply_sensor_id(dat: pd.DataFrame, sensor_name_to_id: dict[str, int]):
    dat['sensor_id'] = dat['sensor_name'].apply(lambda x: sensor_name_to_id[x.upper()])
    return dat
dat = apply_sensor_id(dat, g['sensor_name_to_id'])
dat = cyclical_time(dat, "last_update")
dat.sort_values(by=["last_update"], inplace=True)
dat.reset_index(drop=True, inplace=True)
# fix annotation
dat['annotation'] -= 1
dat.loc[dat['annotation'] == 3, 'annotation'] -= 1

dat_nosampling = dat

dat = down_sample_stationary_data(dat_nosampling.copy(), infos={
    0: ChunkInfo("m6", 0, sampling_rate=dt.timedelta(seconds=300)),
    1: ChunkInfo("m7", 1, sampling_rate=dt.timedelta(seconds=200)),
    2: ChunkInfo("m4", 2, sampling_rate=dt.timedelta(seconds=100))
})


X_names = ["year", "weekday_sin", "weekday_cos", "dayofyear_sin",
           "dayofyear_cos", "secondofday_sin", "secondofday_cos", "sensor_id"]
y_name = "annotation"