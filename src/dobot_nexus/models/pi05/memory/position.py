"""FrameSample 4x4 patch positions, independent of a tensor backend."""

import numpy as np


def position_embeddings(steps):
    steps = np.asarray(steps, dtype=np.int64)
    if steps.ndim != 1 or np.any(steps < 0):
        raise ValueError("Position steps must be a nonnegative vector.")
    omega = np.arange(128, dtype=np.float32) / np.float32(127)
    # Match XLA's correctly rounded float32 power; NumPy's SIMD float32
    # implementation can differ by an ulp, amplified by long time offsets.
    temporal_scale = np.float32(1) / np.power(10000.0, omega.astype(np.float64)).astype(np.float32)
    temporal = steps.astype(np.float32)[:, None] * temporal_scale
    yy, xx = np.mgrid[:4, :4]
    scale = np.float32(1) / np.power(1000.0, omega.astype(np.float64)).astype(np.float32)
    y = (4 * yy.flatten() + 2).astype(np.float32)[:, None] * scale
    x = (4 * xx.flatten() + 2).astype(np.float32)[:, None] * scale
    space = np.concatenate((np.sin(y), np.cos(y), np.sin(x), np.cos(x)), axis=-1)
    time = np.concatenate((np.sin(temporal), np.cos(temporal)), axis=-1)
    return np.concatenate(
        (
            np.broadcast_to(time[:, None], (len(steps), 16, 256)),
            np.broadcast_to(space, (len(steps), 16, 512)),
        ),
        axis=-1,
    ).astype(np.float32)
