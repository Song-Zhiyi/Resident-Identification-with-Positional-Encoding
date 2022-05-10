import pandas as pd
import os
import json
from pathlib import Path
from collections import OrderedDict

def load_json(fname: os.PathLike):
    with open(fname, "r") as fp:
        return json.load(fp)

class ExperimentLoader:
    def __init__(self, root_dir: os.PathLike):
        self.root_dir = Path(root_dir).expanduser()
        self._results = self.__load_all_experiments()

    @property
    def results(self):
        return self._results

    def __load_all_experiments(self):
        results = []
        meta_keys = OrderedDict(
            experiment_id=None,
            train_id=None,
        )

        for exp_dir in self.root_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            for train_dir in exp_dir.iterdir():
                if not train_dir.is_dir():
                    continue

                try:
                    meta: dict = load_json(train_dir / "metadata.json")
                    meta_keys.update({k: None for k in meta})
                    result = pd.DataFrame(
                        load_json(train_dir / "results.json")
                    )
                except FileNotFoundError as e:
                    print(f"Skipping {train_dir!s} due to {e}")
                    continue

                for key, val in meta.items():
                    result[key] = val

                result['experiment_id'] = exp_dir.name
                result['train_id'] = train_dir.name

                results.append(result)

        ret: pd.DataFrame = pd.concat(results)

        other_column = []
        for c in ret.columns:
            if c not in meta_keys:
                other_column.append(c)

        ret = ret[list(meta_keys.keys()) + other_column]
        return ret

    def clean_saved_models(self, keep_best_n: int=3, *, dry_run: bool=True):
        for key, df in self.results.groupby(["experiment_id", "train_id"]):
            df: pd.DataFrame
            df = df.sort_values(by="val_loss")

            to_purge = frozenset(df["epoch"][keep_best_n:])

            saved_model_path: list[Path] = []
            for path in (self.root_dir / key[0] / key[1]).iterdir():
                if path.name.startswith("epoch"):
                    saved_model_path.append(path)

            for path in saved_model_path:
                epoch = int(path.name.removeprefix("epoch"))

                if epoch in to_purge:
                    if not dry_run:
                        path.unlink()
                    else:
                        print(f"Remove: {path!s}")

    def export(self, tarball_path: os.PathLike, with_model: bool=False):
        import tarfile

        with tarfile.open(tarball_path, "w:xz", preset=9) as tar:
            for exp_dir in self.root_dir.iterdir():
                for train_dir in exp_dir.iterdir():
                    tar.add(train_dir / "metadata.json")
                    tar.add(train_dir / "results.json")

                    if with_model:
                        for path in train_dir.iterdir():
                            if path.name.startswith("epoch"):
                                tar.add(path)
