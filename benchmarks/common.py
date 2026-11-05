"""The workload and the memory sampler every harness shares."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil

N_TENSORS = 8


def tensor_shape(model_mib: int) -> tuple:
    return ((model_mib * (1 << 20) // N_TENSORS) // 4,)


def initial_model(model_mib: int) -> dict:
    """Random, so no two chunks match and content-addressed transfer gets no free ride."""
    rng = np.random.default_rng(2026)
    return {
        f"t{i}": rng.random(tensor_shape(model_mib), dtype=np.float32) for i in range(N_TENSORS)
    }


def local_update(params: dict, site: int) -> dict:
    """Stands in for training. Costs one model's worth of arithmetic and memory,
    and every site's result differs from every other site's in every coordinate."""
    factor = np.float32(1 + 0.001 * (site + 1))
    return {k: np.asarray(v, dtype=np.float32) * factor for k, v in params.items()}


def expected_factor(sites: int, rounds: int) -> float:
    """The mean of the sites' factors, once per round."""
    return (1 + 0.001 * (sites + 1) / 2) ** rounds


def model_error(params: dict, model_mib: int, sites: int, rounds: int) -> float:
    """The largest distance of any coordinate from the right answer."""
    factor = expected_factor(sites, rounds)
    start = initial_model(model_mib)
    return max(
        float(np.abs(np.asarray(params[k], dtype=np.float64) - start[k].astype(np.float64) * factor).max())
        for k in start
    )


MODEL_TOLERANCE = 2e-5


def tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        total = root.memory_info().rss
        for child in root.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


class Sampler:
    """Peak resident memory of named process groups, each a list of root pids."""

    def __init__(self, groups: dict, every: float = 0.05):
        self.groups = groups
        self.every = every
        self.peak = {name: 0 for name in groups}
        self.peak_each = {name: 0 for name in groups}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            for name, pids in self.groups.items():
                sizes = [tree_rss(p) for p in list(pids)]
                self.peak[name] = max(self.peak[name], sum(sizes))
                self.peak_each[name] = max(self.peak_each[name], max(sizes, default=0))
            time.sleep(self.every)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def mib(self) -> dict:
        return {
            name: {
                "peak_total_mib": round(self.peak[name] / (1 << 20), 1),
                "peak_each_mib": round(self.peak_each[name] / (1 << 20), 1),
            }
            for name in self.groups
        }


def write_result(path, **fields) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2) + "\n")
    print(json.dumps(fields))
