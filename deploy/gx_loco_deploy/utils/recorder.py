from typing import List, Tuple

import pickle
from pathlib import Path
from gx_loco_deploy.types import Obs, Act


# change pickle to something more like dict, pickle relies on the class name
class Recoder:
    def __init__(self):
        self._buf = []

    def reset(self):
        self._buf = []

    def add(self, obs: Obs, act: Act):
        self._buf.append((obs, act))

    def save(self, filename: str):
        path = Path(filename)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._buf, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str) -> List[Tuple[Obs, Act]]:
        path = Path(filename)
        with path.open("rb") as f:
            buf = pickle.load(f)
        return buf
