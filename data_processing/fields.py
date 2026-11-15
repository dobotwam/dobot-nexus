"""Typed scalar fields copied from recordings without algorithm-specific derivation."""

from __future__ import annotations

import math

import numpy as np


def extra_schema(fields, *, reserved):
    import pyarrow as pa

    types = {
        "bool": pa.bool_(), "float32": pa.float32(), "float64": pa.float64(),
        "int64": pa.int64(), "string": pa.string(),
    }
    if not isinstance(fields, dict):
        raise ValueError("extra_fields must map output column names to field definitions.")
    result = []
    for name, spec in fields.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Extra field names must be non-empty strings.")
        if name in reserved or name.startswith("observation.images.") or name in {"task", "prompt"}:
            raise ValueError(f"Extra field {name!r} conflicts with a dataset column.")
        if not isinstance(spec, dict) or set(spec) - {"source", "dtype", "required"}:
            raise ValueError(f"Extra field {name!r} accepts source, dtype and required only.")
        source = spec.get("source", name)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Extra field {name!r} requires a non-empty source key.")
        dtype = spec.get("dtype")
        if not isinstance(dtype, str) or dtype not in types:
            raise ValueError(f"Extra field {name!r} dtype must be one of {tuple(types)}.")
        required = spec.get("required", False)
        if type(required) is not bool:
            raise ValueError(f"Extra field {name!r} required must be a boolean.")
        result.append(pa.field(name, types[dtype], nullable=not required))
    return result


def extra_values(frame, fields):
    result = {}
    for name, spec in fields.items():
        source = spec.get("source", name)
        value = frame.get(source)
        if value is None:
            if spec.get("required", False):
                raise ValueError(f"Required extra field {name!r} is missing at source {source!r}.")
            result[name] = None
            continue
        dtype = spec["dtype"]
        if dtype == "bool":
            valid = isinstance(value, (bool, np.bool_))
            if valid:
                value = bool(value)
        elif dtype == "string":
            valid = isinstance(value, str)
        elif dtype == "int64":
            valid = (
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and -(2**63) <= value < 2**63
            )
            if valid:
                value = int(value)
        else:
            valid = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, (bool, np.bool_)
            )
            if valid:
                try:
                    value = float(value)
                    valid = math.isfinite(value) and abs(value) <= np.finfo(dtype).max
                except OverflowError:
                    valid = False
        if not valid:
            raise ValueError(f"Extra field {name!r} requires a finite {dtype} scalar.")
        result[name] = value
    return result
