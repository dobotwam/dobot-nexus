from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch


@dataclass(frozen=True)
class MemoryConfig:
    """FrameSample layout, history policy and frozen vision asset identity."""

    schema_version: int = 1
    budget: int = 512
    window_frames: int | None = None
    max_frames: int = 8192
    camera: str = "base"
    tokens_per_image: int = 16
    image_dim: int = 2048
    position_dim: int = 768
    token_dim: int = 1024
    vision_checkpoint: str = ""
    vision_sha256: str = ""
    preprocessing: str = "openpi_jax_float_resize_v1"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported memory schema_version.")
        for name in ("budget", "max_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"memory.{name} must be a positive integer.")
        if self.window_frames is not None and (
            type(self.window_frames) is not int or self.window_frames <= 0
        ):
            raise ValueError("memory.window_frames must be null or a positive integer.")
        if (self.tokens_per_image, self.image_dim, self.position_dim, self.token_dim) != (
            16,
            2048,
            768,
            1024,
        ):
            raise ValueError(
                "FrameSample v1 requires 16 tokens, image=2048, position=768, token=1024."
            )
        if self.budget % self.tokens_per_image:
            raise ValueError("memory.budget must be divisible by tokens_per_image.")
        if not self.camera or self.preprocessing != "openpi_jax_float_resize_v1":
            raise ValueError("Specify a memory camera and the supported preprocessing recipe.")
        if self.vision_sha256 and (
            len(self.vision_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.vision_sha256)
        ):
            raise ValueError("memory.vision_sha256 must be a SHA256 content digest.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryConfig:
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown memory configuration fields: {sorted(unknown)}")
        return cls(**dict(value))

    @property
    def sampled_frames(self) -> int:
        return self.budget // self.tokens_per_image


class MemoryReference(NamedTuple):
    """Episode and frame indices for a lookup within one feature store."""

    episode_index: int
    frame_index: int


@dataclass(frozen=True)
class MemorySnapshot:
    features: np.ndarray
    positions: np.ndarray
    mask: np.ndarray

    def validate(self, config: MemoryConfig, batch_size: int | None = None) -> MemorySnapshot:
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
    def stack(cls, values: Sequence[MemorySnapshot]) -> MemorySnapshot:
        return cls(
            *(
                np.stack([getattr(v, key) for v in values])
                for key in ("features", "positions", "mask")
            )
        )


@dataclass
class MemoryObservationBatch(ObservationBatch):
    memory: MemorySnapshot | None = None
