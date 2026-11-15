"""NumPy msgpack encoding compatible with the OpenPI wire format."""

from __future__ import annotations

from typing import Any
import math

import numpy as np


def _msgpack():
    try:
        import msgpack
    except ImportError as exc:
        raise ImportError("NumPy transport requires: pip install 'dobot-nexus[websocket]'") from exc
    return msgpack


def pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported dtype: {value.dtype}.")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _numeric_dtype(value):
    dtype = np.dtype(value)
    if dtype.kind not in 'biuf':
        raise ValueError(f"Unsupported transport dtype: {dtype}.")
    return dtype


def unpack_array(value: Any) -> Any:
    if b"__ndarray__" in value:
        dtype = _numeric_dtype(value[b"dtype"])
        shape, data = value[b"shape"], value[b"data"]
        if not isinstance(shape, (list, tuple)) or len(shape) > 8 or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError("Invalid array dimensions.")
        if not isinstance(data, bytes) or math.prod(shape)*dtype.itemsize != len(data):
            raise ValueError("Array shape does not match encoded bytes.")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if b"__npgeneric__" in value:
        return _numeric_dtype(value[b"dtype"]).type(value[b"data"])
    return value


def Packer(**kwargs):
    kwargs.setdefault("default", pack_array)
    return _msgpack().Packer(**kwargs)


def packb(value, **kwargs):
    kwargs.setdefault("default", pack_array)
    return _msgpack().packb(value, **kwargs)


def unpackb(value, **kwargs):
    kwargs.setdefault("object_hook", unpack_array)
    kwargs.setdefault("max_array_len", 1_000_000)
    kwargs.setdefault("max_map_len", 4096)
    kwargs.setdefault("max_str_len", 1024*1024)
    kwargs.setdefault("max_bin_len", 32*1024*1024)
    return _msgpack().unpackb(value, **kwargs)
