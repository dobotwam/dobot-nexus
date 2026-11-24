"""One selection/assembly implementation for offline training and online serving."""

import numpy as np

from .position import position_embeddings
from .types import MemorySnapshot


def select_frames(current, count, *, segment_start=0, window_frames=None):
    if not 0 <= segment_start <= current or count <= 0:
        raise ValueError("Invalid memory frame range or sample count.")
    start = segment_start
    if window_frames is not None:
        if window_frames <= 0:
            raise ValueError("window_frames must be positive.")
        start = max(start, current - window_frames + 1)
    if count == 1:
        return np.array([current], dtype=np.int64), np.array([current - start], dtype=np.int64)
    length = current - start + 1
    offsets = (
        np.arange(length, dtype=np.int64)
        if length <= count
        else np.linspace(0, length - 1, count, dtype=np.int32).astype(np.int64)
    )
    return offsets + start, offsets


def assemble_memory(features, relative_steps, config):
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
