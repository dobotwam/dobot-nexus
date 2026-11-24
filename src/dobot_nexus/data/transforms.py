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
    return _transform_actions(actions, state, transform, np.subtract)


def invert_action_transform(
    actions: np.ndarray,
    state: np.ndarray,
    transform: ActionTransform,
) -> np.ndarray:
    return _transform_actions(actions, state, transform, np.add)


def _transform_actions(actions, state, transform, operation):
    if transform.kind != "delta":
        return np.asarray(actions).copy()
    actions = _delta_operand(actions)
    reference = _delta_operand(state)
    mask = transform.delta_mask
    if reference.shape[-1] != mask.size:
        raise ValueError("Delta actions require matching state and action dimensions.")
    while reference.ndim < actions.ndim:
        reference = np.expand_dims(reference, axis=-2)
    delta = operation(actions[..., mask], reference[..., mask])
    if delta.dtype == np.int64:
        # Each integer operand is within ±2**53, so their exact sum/difference fits in int64.
        floating = delta.astype(np.float64)
        if not np.array_equal(floating.astype(np.int64), delta):
            raise ValueError("Integer delta result has no exact float64 representation.")
        delta = floating
    # Mixed integer/float operations need the promoted result dtype even on assignment.
    dtype = delta.dtype if "i" in (actions.dtype.kind, reference.dtype.kind) else actions.dtype
    result = actions.astype(dtype)
    result[..., mask] = delta
    return result


def _delta_operand(value):
    value = np.asarray(value)
    if value.dtype.kind not in "biu":
        return _as_float(value)
    if value.dtype.itemsize >= 8:
        limit = 2**53
        if np.any(value > limit) or (value.dtype.kind == "i" and np.any(value < -limit)):
            raise ValueError("Delta operands exceed the exact float64 integer range ±2**53.")
    return value.astype(np.int64, copy=False)


def _finite_real(vector: np.ndarray) -> bool:
    if vector.dtype.kind not in "buif":
        return False
    # Statistics are serialized and loaded as float64, including extended-precision inputs.
    with np.errstate(over="ignore"):
        return bool(np.isfinite(np.asarray(vector, dtype=np.float64)).all())


def validate_norm_stats(
    stats: NormStats, dimension: int, *, mode: str = "zscore", name: str = "value"
) -> None:
    if mode not in {"zscore", "quantile"}:
        raise ValueError(f"{name} normalization must be zscore or quantile, got {mode!r}.")
    mean, std = np.asarray(stats.mean), np.asarray(stats.std)
    if mean.ndim < 1 or mean.shape != std.shape or mean.shape[-1] < dimension:
        raise ValueError(
            f"{name} mean/std must have matching shapes covering dimension {dimension}, "
            f"got {mean.shape} and {std.shape}."
        )
    if not _finite_real(mean) or not _finite_real(std) or np.any(std < 0):
        raise ValueError(f"{name} mean/std must be finite with std >= 0.")
    for quantile in (stats.q01, stats.q99):
        if quantile is not None:
            array = np.asarray(quantile)
            if array.shape != mean.shape or not _finite_real(array):
                raise ValueError(f"{name} q01/q99 must be finite and match mean.shape.")
    if mode == "quantile":
        low, high = stats.q01, stats.q99
        if low is None or high is None:
            raise ValueError(f"{name} quantile normalization requires q01/q99 statistics.")
        low, high = np.asarray(low), np.asarray(high)
        if low.ndim != 1 or np.any(high < low):
            raise ValueError(f"{name} q01/q99 must be 1D with q99 >= q01.")


def _as_float(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return value if np.issubdtype(value.dtype, np.floating) else value.astype(np.float64)


def _zscore_parameters(stats, shape):
    mean = np.asarray(stats.mean)[..., :shape[-1]]
    std = np.asarray(stats.std)[..., :shape[-1]]
    if np.broadcast_shapes(mean.shape, shape) != shape:
        raise ValueError(f"Z-score statistics {mean.shape} would expand input shape {shape}.")
    return mean, std


def _quantile_parameters(value, stats):
    dimension = value.shape[-1]
    low = _as_float(stats.q01)[:dimension]
    high = _as_float(stats.q99)[:dimension]
    dtype = np.result_type(value.dtype, low.dtype, high.dtype, np.float32)
    return (
        value.astype(dtype, copy=False),
        low.astype(dtype, copy=False),
        high.astype(dtype, copy=False),
    )


def normalize(value: np.ndarray, stats: NormStats, *, mode: str = "zscore") -> np.ndarray:
    value = _as_float(value)
    dimension = value.shape[-1]
    validate_norm_stats(stats, dimension, mode=mode)
    if mode == "quantile":
        value, low, high = _quantile_parameters(value, stats)
        return (value - low) / (high - low + _NORMALIZATION_EPSILON) * 2.0 - 1.0
    mean, std = _zscore_parameters(stats, value.shape)
    return (value - mean) / (std + _NORMALIZATION_EPSILON)


def unnormalize(value: np.ndarray, stats: NormStats, *, mode: str = "zscore") -> np.ndarray:
    value = _as_float(value)
    dimension = value.shape[-1]
    validate_norm_stats(stats, dimension, mode=mode)
    if mode == "quantile":
        value, low, high = _quantile_parameters(value, stats)
        return (value + 1.0) / 2.0 * (high - low + _NORMALIZATION_EPSILON) + low
    mean, std = _zscore_parameters(stats, value.shape)
    return value * (std + _NORMALIZATION_EPSILON) + mean
