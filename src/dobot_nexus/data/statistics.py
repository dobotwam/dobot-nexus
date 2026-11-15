from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormStats:
    """Per-feature statistics using the OpenPI-compatible fields."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class RunningStats:
    """Bounded-memory statistics for vectors stored in the final dimension."""

    def __init__(self, *, quantile_bins: int = 5000) -> None:
        if quantile_bins <= 1:
            raise ValueError("quantile_bins must be greater than one.")
        self._quantile_bins = int(quantile_bins)
        self._count = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._minimum: np.ndarray | None = None
        self._maximum: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None

    @property
    def count(self) -> int:
        return self._count

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim < 1:
            raise ValueError(
                f"Statistics values must have at least one dimension, got {batch.shape}."
            )
        batch = batch.reshape(-1, batch.shape[-1])
        if batch.shape[0] == 0:
            return
        if not np.isfinite(batch).all():
            raise ValueError("Statistics values contain non-finite entries.")

        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        batch_minimum = batch.min(axis=0)
        batch_maximum = batch.max(axis=0)

        if self._count == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
            self._minimum = batch_minimum
            self._maximum = batch_maximum
            self._histograms = [
                np.zeros(self._quantile_bins, dtype=np.float64)
                for _ in range(batch.shape[1])
            ]
            self._bin_edges = [
                self._make_edges(batch_minimum[index], batch_maximum[index])
                for index in range(batch.shape[1])
            ]
            self._count = batch_count
            self._update_histograms(batch)
            return

        assert self._mean is not None
        assert self._m2 is not None
        assert self._minimum is not None
        assert self._maximum is not None
        if batch.shape[1] != self._mean.shape[0]:
            raise ValueError(
                "Statistics vector dimension changed from "
                f"{self._mean.shape[0]} to {batch.shape[1]}."
            )

        new_minimum = np.minimum(self._minimum, batch_minimum)
        new_maximum = np.maximum(self._maximum, batch_maximum)
        if np.any(new_minimum < self._minimum) or np.any(new_maximum > self._maximum):
            self._minimum = new_minimum
            self._maximum = new_maximum
            self._resize_histograms()

        total = self._count + batch_count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (batch_count / total)
        self._m2 = (
            self._m2
            + batch_m2
            + delta * delta * (self._count * batch_count / total)
        )
        self._count = total
        self._update_histograms(batch)

    def result(self) -> NormStats:
        if self._count == 0:
            raise ValueError("Cannot compute statistics for an empty dataset.")
        assert self._mean is not None
        assert self._m2 is not None
        variance = np.maximum(self._m2 / self._count, 0.0)
        q01, q99 = self._quantiles((0.01, 0.99))
        return NormStats(
            mean=self._mean.copy(),
            std=np.sqrt(variance),
            q01=q01,
            q99=q99,
        )

    def _make_edges(self, minimum: float, maximum: float) -> np.ndarray:
        if maximum <= minimum:
            padding = max(abs(minimum) * 1e-12, 1e-10)
            minimum -= padding
            maximum += padding
        return np.linspace(minimum, maximum, self._quantile_bins + 1, dtype=np.float64)

    def _resize_histograms(self) -> None:
        assert self._minimum is not None
        assert self._maximum is not None
        assert self._histograms is not None
        assert self._bin_edges is not None
        for index, histogram in enumerate(self._histograms):
            old_edges = self._bin_edges[index]
            old_centers = (old_edges[:-1] + old_edges[1:]) * 0.5
            new_edges = self._make_edges(self._minimum[index], self._maximum[index])
            resized, _ = np.histogram(old_centers, bins=new_edges, weights=histogram)
            self._histograms[index] = resized.astype(np.float64, copy=False)
            self._bin_edges[index] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        assert self._histograms is not None
        assert self._bin_edges is not None
        for index in range(batch.shape[1]):
            histogram, _ = np.histogram(batch[:, index], bins=self._bin_edges[index])
            self._histograms[index] += histogram

    def _quantiles(self, quantiles: tuple[float, ...]) -> tuple[np.ndarray, ...]:
        assert self._histograms is not None
        assert self._bin_edges is not None
        results: list[np.ndarray] = []
        for quantile in quantiles:
            target = min(max(float(quantile), 0.0), 1.0) * max(self._count - 1, 0)
            values: list[float] = []
            for histogram, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumulative = np.cumsum(histogram)
                bin_index = min(
                    int(np.searchsorted(cumulative, target + 1.0, side="left")),
                    histogram.size - 1,
                )
                values.append(float((edges[bin_index] + edges[bin_index + 1]) * 0.5))
            results.append(np.asarray(values, dtype=np.float64))
        return tuple(results)


def _as_stats(name: str, value: Any) -> NormStats:
    if not isinstance(value, Mapping) or not {"mean", "std"}.issubset(value):
        raise ValueError(f"Normalization entry {name!r} must contain mean and std.")
    mean = np.asarray(value["mean"], dtype=np.float64)
    std = np.asarray(value["std"], dtype=np.float64)
    q01 = np.asarray(value["q01"], dtype=np.float64) if value.get("q01") is not None else None
    q99 = np.asarray(value["q99"], dtype=np.float64) if value.get("q99") is not None else None
    arrays = [mean, std, *([q01] if q01 is not None else []), *([q99] if q99 is not None else [])]
    if mean.ndim < 1 or std.shape != mean.shape:
        raise ValueError(f"Normalization entry {name!r} has incompatible mean/std shapes.")
    if any(array.shape != mean.shape or not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Normalization entry {name!r} contains invalid statistics.")
    if np.any(std < 0):
        raise ValueError(f"Normalization entry {name!r} contains negative std values.")
    return NormStats(mean=mean, std=std, q01=q01, q99=q99)


def load_norm_stats(path: str | Path) -> dict[str, NormStats]:
    """Load bare LeRobot or wrapped OpenPI-compatible normalization statistics."""
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "norm_stats.json"
    if not source.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {source}.")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("norm_stats", payload) if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid normalization payload in {source}.")
    return {str(name): _as_stats(str(name), value) for name, value in raw.items()}


def _stats_payload(stats: Mapping[str, NormStats]) -> dict[str, Any]:
    norm_stats: dict[str, dict[str, list[float]]] = {}
    for name, value in stats.items():
        entry = {
            "mean": np.asarray(value.mean, dtype=np.float64).tolist(),
            "std": np.asarray(value.std, dtype=np.float64).tolist(),
        }
        if value.q01 is not None:
            entry["q01"] = np.asarray(value.q01, dtype=np.float64).tolist()
        if value.q99 is not None:
            entry["q99"] = np.asarray(value.q99, dtype=np.float64).tolist()
        norm_stats[str(name)] = entry
    return {"norm_stats": norm_stats}


def save_norm_stats(
    path: str | Path,
    stats: Mapping[str, NormStats],
    *,
    atomic: bool = False,
) -> Path:
    """Save statistics to a JSON file, optionally replacing it atomically."""
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".json":
        destination = destination / "norm_stats.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_stats_payload(stats), indent=2, sort_keys=True) + "\n"
    if not atomic:
        destination.write_text(serialized, encoding="utf-8")
        return destination.resolve()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.resolve()
