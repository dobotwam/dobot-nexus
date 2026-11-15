"""Internal RECAP records, independent of the public model contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, order=True)
class FrameKey:
    dataset_id: str
    episode_index: int
    frame_index: int


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int


@dataclass(frozen=True)
class ReturnScale:
    task_max_lengths: Mapping[int, int]
    c_fail_coef: float = 1.0

    def __post_init__(self):
        lengths = {int(k): int(v) for k, v in self.task_max_lengths.items()}
        if (
            not lengths
            or any(k < 0 or v < 1 for k, v in lengths.items())
            or not math.isfinite(self.c_fail_coef)
            or self.c_fail_coef < 0
        ):
            raise ValueError("Invalid Dobot per-task return scale.")
        object.__setattr__(self, "task_max_lengths", lengths)

    def denominator(self, task: int) -> float:
        return self.task_max_lengths[int(task)] * (1.0 + self.c_fail_coef)


@dataclass(frozen=True)
class EpisodeReturns:
    rewards: NDArray[np.float32]
    returns: NDArray[np.float32]


@dataclass(frozen=True)
class AdvantageLabels:
    advantages: NDArray[np.float32]
    thresholds: Mapping[int, float]
    positive: NDArray[np.bool_]


@dataclass(frozen=True)
class FrameIndex:
    dataset_id: str
    keys: tuple[FrameKey, ...]
    source_indices: NDArray[np.int64]
    episode_offsets: NDArray[np.int64]
    task_indices: NDArray[np.int64]
    episodes: tuple[EpisodeRecord, ...]
    dataset_signature: str


@dataclass(frozen=True)
class ReturnTable:
    index: FrameIndex
    rewards: NDArray[np.float32]
    returns: NDArray[np.float32]
    interventions: NDArray[np.bool_] | NDArray[np.object_] | None
    scale: ReturnScale
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyLabelTable:
    index: FrameIndex
    positive: NDArray[np.bool_]
    advantages: NDArray[np.float32]
    scale: ReturnScale
    manifest: Mapping[str, Any]

    def row_lookup(self) -> dict[FrameKey, int]:
        return {key: row for row, key in enumerate(self.index.keys)}


@dataclass(frozen=True)
class ValueTable:
    index: FrameIndex
    values: NDArray[np.float32]
    scale: ReturnScale
    checkpoint_id: str
