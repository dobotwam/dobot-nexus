"""One selection/assembly implementation for offline training and online serving."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .position import position_embeddings
from .types import MemoryConfig, MemorySnapshot

SingleFrame = Literal["oldest", "start", "current"]
DEFAULT_HISTORY_SINGLE_FRAME: SingleFrame = "oldest"


def validate_single_frame(value: SingleFrame) -> SingleFrame:
    if value not in ("oldest", "start", "current"):
        raise ValueError(f"single_frame must be 'oldest', 'start' or 'current', got {value!r}.")
    return value


def select_frames(
    current: int,
    count: int,
    *,
    segment_start: int = 0,
    window_frames: int | None = None,
    single_frame: SingleFrame = DEFAULT_HISTORY_SINGLE_FRAME,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute frame indices and offsets relative to the retained segment."""
    if not 0 <= segment_start <= current or count <= 0:
        raise ValueError("Invalid memory frame range or sample count.")
    start = segment_start
    if window_frames is not None:
        if window_frames <= 0:
            raise ValueError("window_frames must be positive.")
        start = max(start, current - window_frames + 1)
    validate_single_frame(single_frame)
    if count == 1:
        frame = current if single_frame == "current" else start
        return np.array([frame], dtype=np.int64), np.array([frame - start], dtype=np.int64)
    length = current - start + 1
    offsets = (
        np.arange(length, dtype=np.int64)
        if length <= count
        else np.linspace(0, length - 1, count, dtype=np.int32).astype(np.int64)
    )
    return offsets + start, offsets


def assemble_memory(
    features: ArrayLike, relative_steps: ArrayLike, config: MemoryConfig
) -> MemorySnapshot:
    features = np.asarray(features, dtype=np.float32)
    k = len(features)
    if features.shape != (k, 16, 2048) or not 0 < k <= config.sampled_frames:
        raise ValueError("Selected history features must be [1..sampled_frames,16,2048].")
    if len(relative_steps) != k:
        raise ValueError("Feature and position counts differ.")
    image = np.zeros((config.sampled_frames, 16, 2048), np.float32)
    position = np.zeros((config.sampled_frames, 16, 768), np.float32)
    mask = np.zeros((config.sampled_frames, 16), np.bool_)
    image[:k], position[:k], mask[:k] = features, position_embeddings(relative_steps), True
    return MemorySnapshot(
        image.reshape(config.budget, 2048),
        position.reshape(config.budget, 768),
        mask.reshape(config.budget),
    ).validate(config)
