#%%
import io
import os
import pickle

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from collections import OrderedDict, UserDict

DATA_ROOT = Path("data/twor.2010")
CACHE_ROOT = Path(".cache")

def _apply_cyclical_time(df, col, max_t):
    ang = 2 * np.pi * df[col] / max_t
    df[col + "_sin"] = np.sin(ang)
    df[col + "_cos"] = np.cos(ang)

def seconds_of_day(hour, minute, second):
    return hour * 3600 + minute * 60 + second

class Labeller(UserDict):
    def __init__(self, no_label="nolabel"):
        self._label_id = 0
        self.data = OrderedDict()

        self.no_label_name = no_label
        self.no_label_id = self[no_label]

    def __missing__(self, k: str):
        ret = self._label_id
        self._label_id += 1
        self.data[k] = ret
        return ret

    def __contains__(self, key: object) -> bool:
        if key not in self.data:
            self.__missing__(key)
        return True

def _parse_time(t):
    try:
        return datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        pass
    return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

from typing import TypedDict
class ResultDict(TypedDict):
    event_to_id: dict[str, int]
    sensor_to_id: dict[str, int]
    dat: pd.DataFrame

class Loader:
    def __init__(self,
                 base_dir: os.PathLike = DATA_ROOT,
                 cache_dir: os.PathLike = CACHE_ROOT,
                 debug: bool = False):
        self.base_dir = Path(base_dir).expanduser().absolute()
        self.cache_dir = Path(cache_dir).expanduser().absolute() / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sensor_label = Labeller("Unknown")
        self.event_label = Labeller("")
        self.debug = debug

    def __parse(self, fp: io.TextIOWrapper):
        r_to_int = {
            "ON": 1,
            "OPEN": 1,
            "PRESENT": 1,
            "OFF": 0,
            "CLOSE": 0,
            "ABSENT": 0,
        }
        change_to_int = {
            "begin": 1,
            "end": 0,
            None: None,
        }

        for line in fp:
            line = line.strip()
            parts = line.split()

            n_parts = len(parts)

            if n_parts == 4:
                tsd, tst, sid, r = parts
                sc = ""
                change = None
            elif n_parts == 6:
                tsd, tst, sid, r, sc, change = parts
            elif n_parts == 0:
                continue
            else:
                raise Exception(parts)

            change = change_to_int[change]

            try:
                r = r_to_int[r]
            except KeyError:
                pass

            ts = _parse_time(tsd + " " + tst)

            yield ts, sid, r, sc, change

    def __transform(self, gen):
        r1_to_yield = r1_state = ""
        r2_to_yield = r2_state = ""

        evl = self.event_label

        for ts, sid, r, sc, change in gen:
            sc: str

            if sc == "":
                pass
            elif sc.startswith("R1"):
                state = sc[3:]
                if change == 1:
                    r1_to_yield = r1_state = state
                elif change == 0:
                    r1_state = ""
                elif change is None:
                    r1_to_yield = r1_state = ""
                else:
                    raise RuntimeError(f"not recognizable {change}")
            elif sc.startswith("R2"):
                state = sc[3:]
                if change == 1:
                    r2_to_yield = r2_state = state
                elif change == 0:
                    r2_state = ""
                elif change is None:
                    r2_to_yield = r2_state = ""
                else:
                    raise RuntimeError(f"not recognizable {change}")
            else:
                raise NotImplementedError(f"{sc=}")

            r1_evid = evl[r1_to_yield]
            r2_evid = evl[r2_to_yield]

            yield ts, sid, r, sc, change, r1_evid, r2_evid


    def __parse_helper(self, f: Path):
        with open(f) as fp:
            yield from self.__transform(
                sorted(
                    self.__parse(fp)
                )
            )

    def __load_preprocess(self, f: Path):
        dat = pd.DataFrame(
            list(self.__parse_helper(f)),
            columns=["timestamp", "sensor_id", "sensor_state", "event", "change", "r1_state", "r2_state"],
        )
        dat = self.__preprocess(dat)
        dat.reset_index(drop=True, inplace=True)
        return ResultDict(
            event_to_id=self.event_label.data,
            sensor_to_id=self.sensor_label.data,
            dat=dat
        )

    def __drop_unused_sensors(self, dat: pd.DataFrame) -> pd.DataFrame:
        dat = dat[
            ~(dat['sensor_id'].str.startswith("L")
            & (dat['sensor_state'] != 0)
            & (dat['sensor_state'] != 1))
        ]
        dat = dat[~dat["sensor_id"].str.startswith("T")]
        dat = dat[~dat["sensor_id"].str.startswith("P")]

        if not self.debug:
            dat.drop(columns=["event", "change"], inplace=True)

        return dat

    def __transform_sensor_id(self, dat: pd.DataFrame) -> pd.DataFrame:
        dat['sensor_id'] = dat['sensor_id'].map(self.sensor_label.get)
        return dat

    def __transform_timestamp(self, dat: pd.DataFrame) -> pd.DataFrame:
        dat_dt = dat['timestamp'].dt
        dat['year'] = dat_dt.year
        dat['dayofyear'] = dat_dt.dayofyear - 1
        dat['secondofday'] = seconds_of_day(dat_dt.hour, dat_dt.minute, dat_dt.second)
        dat['weekday'] = dat_dt.weekday

        _apply_cyclical_time(dat, 'weekday', 7)
        _apply_cyclical_time(dat, 'dayofyear', 365)
        _apply_cyclical_time(dat, 'secondofday', 86400)
        return dat

    def __preprocess(self, dat: pd.DataFrame) -> pd.DataFrame:
        dat = self.__drop_unused_sensors(dat)
        dat = self.__transform_sensor_id(dat)
        dat = self.__transform_timestamp(dat)
        return dat

    def __cache_file_path(self):
        return (
            self.cache_dir / "twor2010-preprocess-data.feather",
            self.cache_dir / "twor2010-sensor-mapping.pkl",
        )

    def _dump_cache(self, result: ResultDict):
        data_path, sensor_map_path = self.__cache_file_path()
        with open(sensor_map_path, "wb") as fp:
            d = dict(result)
            d.pop("dat")
            pickle.dump(d, fp)
        result["dat"].to_feather(data_path, compression="zstd")
        return result

    def _load_cache(self):
        data_path, sensor_map_path = self.__cache_file_path()
        with open(sensor_map_path, "rb") as fp:
            ret: ResultDict = pickle.load(fp)
        ret["dat"] = pd.read_feather(data_path)
        return ret

    def load(self, force: bool=False) -> ResultDict:
        try:
            if not force:
                return self._load_cache()
        except Exception as e:
            print(f"load cache failed due to {type(e).__name__} {e!s}")

        data_file = self.base_dir / "data"
        return self._dump_cache(
            self.__load_preprocess(data_file)
        )


#%%


state_labeller = Labeller("")
r1_events_lab = Labeller(tuple())
r2_events_lab = Labeller(tuple())

def test():
    r_to_int = {
        "ON": 1,
        "OPEN": 1,
        "PRESENT": 1,
        "OFF": 0,
        "CLOSE": 0,
        "ABSENT": 0,
    }
    with open(Path("../..") / "data/twor.2010" / "data") as fp:
        both_no_event = 0

        stack = [
            OrderedDict(),
            OrderedDict(),
        ]

        last_ts = datetime.min

        for lineno, line in enumerate(sorted(fp)):
            line = line.strip()
            parts = line.split()

            n_parts = len(parts)

            if n_parts == 4:
                tsd, tst, sid, r = parts
                sc = ""
                change = None
            elif n_parts == 6:
                tsd, tst, sid, r, sc, change = parts
            elif n_parts == 0:
                continue
            else:
                raise Exception(parts)

            try:
                r = r_to_int[r]
            except KeyError:
                pass

            try:
                ts = _parse_time(tsd + " " + tst)
            except ValueError:
                print(lineno, line, "cannot parse time")
                continue

            if last_ts <= ts:
                last_ts = ts
            else:
                print(f"wrong order at {lineno} {last_ts} {ts}")

            state = state_labeller[sc[3:]]
            if sc.startswith("R1"):
                who = 0
            elif sc.startswith("R2"):
                who = 1
            else:
                who = None

            r1 = None
            r2 = None

            if who is not None:
                if change == "begin":
                    if state in stack[who]:
                        print(f"Event {sc} for already begin at lineno={stack[who][state]}, ignored")
                    else:
                        stack[who][state] = lineno
                elif change == "end":
                    r1 = tuple(stack[0])
                    r2 = tuple(stack[1])
                    if len(stack[who]) == 0:
                        print(f"at line {lineno}: {sc} end without begin, ignoring this event")
                    else:
                        stack[who].pop(state)
                else:
                    pass
                print(lineno, len(stack[0]), len(stack[1]), ts, sc, change)

            if r1 is None:
                r1 = tuple(stack[0])
            if r2 is None:
                r2 = tuple(stack[1])

            r1_nevent = len(r1)
            r2_nevent = len(r2)

            if r1_nevent + r2_nevent == 0:
                both_no_event += 1


            r1 = r1_events_lab[tuple(stack[0])]
            r2 = r2_events_lab[tuple(stack[1])]

            yield ts, sid, r, sc, change, r1_nevent, r2_nevent, r1, r2

        for i, s in enumerate(stack):
            if len(s) != 0:
                print(f"stack {i} is not empty {s!s}")


dat = pd.DataFrame(
    list(test())
)

#%%
print(state_labeller)
print(r1_events_lab)
print(r2_events_lab)

r1s = dat[7]
r2s = dat[8]
print(dat.shape)

only_one_has_event = dat[
    ((r1s == 0) & (r2s != 0)) |
    ((r1s != 0) & (r2s == 0))
]
both_have_event = dat[(r1s != 0) & (r2s != 0)]
both_no_event = dat[(r1s == 0) & (r2s == 0)]

print(f"{len(only_one_has_event)=}")
print(f"{len(both_have_event)=}")
print(f"{len(both_no_event)=}")

#%%

dat