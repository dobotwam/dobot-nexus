"""Small validation helpers shared by RECAP modules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path


def mapping(value, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def check_keys(value: Mapping, allowed, name: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {name} fields: {sorted(unknown)}.")


def integer(value, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def nonempty(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def path_value(value, name: str) -> Path:
    return Path(nonempty(str(value) if isinstance(value, Path) else value, name)).expanduser()
