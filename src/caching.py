import os
import pickle
from typing import Any
from pathlib import Path

class CacheManager:
    def __init__(self, dir=".cache") -> None:
        self._root = Path(dir).expanduser().absolute()

    def _restore_structure(self):
        (self._root / "graphs").mkdir()
        (self._root / "data").mkdir()

    def put_item(self, relpath, item):
        pass

def pickle_load_from_file(path: os.PathLike) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(fp)

def pickle_dump_to_file(path: os.PathLike, obj, ignore_failure: bool=True):
    try:
        with open(path, "wb") as fp:
            pickle.dump(obj, fp)
    except Exception as e:
        print(f"Failed to write cache to {path!s} due to {e!s}")
        if not ignore_failure:
            raise

    return obj


