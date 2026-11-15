from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dobot_nexus.data.statistics import NormStats

_NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True)
class ActionTransform:
    """Model-selected action representation implemented by the common data layer."""

    kind: str
    delta_mask: np.ndarray

    @classmethod
    def create(
        cls,
        kind: str,
        delta_mask: Any,
        *,
        action_dim: int,
    ) -> ActionTransform:
        normalized = str(kind).lower()
        if normalized not in {"absolute", "delta"}:
            raise ValueError("Action transform type must be absolute or delta.")
        if normalized == "delta" and delta_mask is None:
            raise ValueError("Delta actions require a delta_mask.")
        mask = (
            np.zeros(action_dim, dtype=np.bool_)
            if normalized == "absolute"
            else np.asarray(delta_mask)
        )
        if mask.dtype != np.bool_ or mask.shape != (action_dim,):
            raise ValueError(f"delta_mask must contain {action_dim} boolean values.")
        mask = np.ascontiguousarray(mask).copy()
        mask.setflags(write=False)
        return cls(kind=normalized, delta_mask=mask)


def apply_action_transform(
    actions: np.ndarray,
    state: np.ndarray,
    transform: ActionTransform,
) -> np.ndarray:
    result = np.asarray(actions).copy()
    if transform.kind == "delta":
        if state.shape[-1] != transform.delta_mask.shape[0]:
            raise ValueError("Delta actions require matching state and action dimensions.")
        reference = np.asarray(state)
        while reference.ndim < result.ndim:
            reference = np.expand_dims(reference, axis=-2)
        result[..., transform.delta_mask] -= reference[..., transform.delta_mask]
    return result


def invert_action_transform(
    actions: np.ndarray,
    state: np.ndarray,
    transform: ActionTransform,
) -> np.ndarray:
    result = np.asarray(actions).copy()
    if transform.kind == "delta":
        if state.shape[-1] != transform.delta_mask.shape[0]:
            raise ValueError("Delta actions require matching state and action dimensions.")
        reference = np.asarray(state)
        while reference.ndim < result.ndim:
            reference = np.expand_dims(reference, axis=-2)
        result[..., transform.delta_mask] += reference[..., transform.delta_mask]
    return result


def normalize(value: np.ndarray, stats: NormStats) -> np.ndarray:
    dimension = value.shape[-1]
    if stats.mean.shape[-1] < dimension or stats.std.shape[-1] < dimension:
        raise ValueError(
            f"Normalization stats dimension is smaller than input dimension {dimension}."
        )
    return (value - stats.mean[..., :dimension]) / (
        stats.std[..., :dimension] + _NORMALIZATION_EPSILON
    )


def unnormalize(value: np.ndarray, stats: NormStats) -> np.ndarray:
    dimension = value.shape[-1]
    if stats.mean.shape[-1] < dimension or stats.std.shape[-1] < dimension:
        raise ValueError(
            f"Normalization stats dimension is smaller than input dimension {dimension}."
        )
    return value * (stats.std[..., :dimension] + _NORMALIZATION_EPSILON) + stats.mean[
        ..., :dimension
    ]
