from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch


@dataclass(frozen=True)
class MemorySnapshot:
    features: np.ndarray
    positions: np.ndarray
    mask: np.ndarray

    def validate(self, config, batch_size=None):
        prefix = () if batch_size is None else (batch_size,)
        for name, tail in (
            ("features", (config.budget, config.image_dim)),
            ("positions", (config.budget, config.position_dim)),
            ("mask", (config.budget,)),
        ):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != prefix + tail:
                raise ValueError(f"Memory {name} must have shape {prefix + tail}.")
            expected = np.bool_ if name == "mask" else np.float32
            if value.dtype != expected or not np.isfinite(value).all():
                raise ValueError(f"Memory {name} must be finite {expected}.")
        if not self.mask.any(axis=-1).all():
            raise ValueError(
                "Memory requires at least the current frame; empty history is unsupported."
            )
        return self

    @classmethod
    def stack(cls, values):
        return cls(
            *(
                np.stack([getattr(v, key) for v in values])
                for key in ("features", "positions", "mask")
            )
        )


@dataclass
class MemoryObservationBatch(ObservationBatch):
    memory: MemorySnapshot | None = None
