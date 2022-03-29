#%%
from pathlib import Path

import numpy as np
import pandas as pd


#%%
PUCK_DIR = Path('data/PUCK')

from collections import OrderedDict, UserDict

class Labeller(UserDict):
    def __init__(self, no_label="nolabel"):
        self._label_id = 0
        self.data = OrderedDict()

    def __missing__(self, k: str):
        ret = self._label_id
        self._label_id += 1
        return ret

lab = Labeller()

class Loader:
    def __init__(self, base_dir: Path = PUCK_DIR):
        self.base_dir = base_dir
        self.annotated_dir = base_dir / 'Annotated (Env+Obj)'

    def __load_single(self, p: Path):
        rid = int(p.stem.removeprefix("Participant"))
        records = []
        with open(p) as fp:
            for line in fp:
                records.append(
                    line.strip().split("\t")[:3]
                )

        df = pd.DataFrame(records, columns=["timestamp", "sensor_name", "readings"])
        df['rid'] = rid
        return df

    def load_nocache(self):
        dfs: list[pd.DataFrame] = []
        for p in self.annotated_dir.iterdir():
            dfs.append(
                self.__load_single(p)
            )

        return pd.concat(dfs)

    def load(self):
        return self.load_nocache()


loader = Loader("../.." / PUCK_DIR)

df = loader.load()

#%%
df['readings'].value_counts()[:10]

df[df['readings'] == "START_INSTRUCT"]




