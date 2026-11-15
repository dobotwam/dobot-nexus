from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def format_version(value: Any) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower().removeprefix("v")
    if normalized.startswith("2"):
        return "v2.1"
    if normalized.startswith("3"):
        return "v3.0"
    return str(value)


def load_info(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}.")
    with info_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"LeRobot metadata must be an object: {info_path}.")
    return dict(value)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {source}.")
    values: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}.")
            values.append(value)
    return values


def find_parquet_files(dataset_root: str | Path) -> list[Path]:
    data_root = Path(dataset_root).expanduser() / "data"
    paths = sorted(data_root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot Parquet files found under {data_root}.")
    return paths
