"""Models are plain safetensors files, read without copying."""

import json
import struct
from pathlib import Path

import numpy as np

from . import _native

_DTYPES = {"F32": np.float32, "F64": np.float64}
_NAMES = {np.dtype(np.float32): "F32", np.dtype(np.float64): "F64"}


def layout_of(params: dict) -> list:
    layout = []
    for name, array in params.items():
        array = np.asarray(array)
        if array.dtype not in _NAMES:
            raise ValueError(f"{name}: only float32 and float64 tensors are supported")
        layout.append((name, _NAMES[array.dtype], [int(n) for n in array.shape]))
    return layout


def in_layout(params: dict, layout: list) -> list:
    arrays = []
    for name, dtype, shape in layout:
        if name not in params:
            raise ValueError(f"the trainer returned no tensor called {name}")
        array = np.asarray(params[name], dtype=_DTYPES[dtype], order="C")
        if list(array.shape) != shape:
            raise ValueError(f"{name} has shape {array.shape}, the plan says {shape}")
        arrays.append(array)
    return arrays


def save_model(path, params: dict) -> str:
    """Writes a model and returns its digest."""
    layout = layout_of(params)
    return _native.write_model(str(path), layout, in_layout(params, layout))


def load_model(path) -> dict:
    """Maps a model file into memory read-only."""
    path = Path(path)
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    start = 8 + size
    params = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        dtype = _DTYPES[entry["dtype"]]
        if begin == end:
            params[name] = np.empty(entry["shape"], dtype=dtype)
            continue
        params[name] = np.memmap(
            path, mode="r", dtype=dtype, offset=start + begin, shape=tuple(entry["shape"])
        )
    return params
