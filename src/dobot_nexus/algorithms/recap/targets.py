"""Dobot ACP: task-scaled time-to-go, n-step TD and task-local quantiles."""

from __future__ import annotations

import numpy as np

from ._validation import integer, number
from .types import AdvantageLabels, EpisodeReturns, ReturnScale


def finite_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return array


def intervention_vector(value, *, length: int) -> np.ndarray:
    """Preserve unknown provenance while rejecting coerced numeric/string labels."""
    array = np.asarray(value, dtype=object)
    if array.shape != (length,) or any(
        item is not None and not isinstance(item, (bool, np.bool_)) for item in array
    ):
        raise ValueError("Interventions must contain one boolean or null per frame.")
    if all(item is not None for item in array):
        return np.asarray(array, dtype=np.bool_)
    return np.asarray([None if item is None else bool(item) for item in array], dtype=object)


def resolve_return_scale(index, c_fail_coef: float = 1.0) -> ReturnScale:
    coefficient = number(c_fail_coef, "returns.c_fail_coef")
    lengths = {}
    for episode, begin, end in zip(
        index.episodes, index.episode_offsets[:-1], index.episode_offsets[1:], strict=True
    ):
        tasks = np.unique(index.task_indices[begin:end])
        if len(tasks) != 1:
            raise ValueError("Dobot ACP requires one task per episode.")
        task = int(tasks[0])
        lengths[task] = max(lengths.get(task, 0), episode.length)
    return ReturnScale(lengths, coefficient)


def compute_episode_returns(length: int, success: bool, *, task: int, scale: ReturnScale):
    integer(length, "episode length")
    if not isinstance(success, (bool, np.bool_)):
        raise ValueError("Episode success must be boolean.")
    if length > scale.task_max_lengths[task]:
        raise ValueError("Episode exceeds its task's maximum length.")
    raw = -(float(length) - np.arange(length, dtype=np.float32) - 1.0)
    if not success:
        raw -= scale.c_fail_coef * scale.task_max_lengths[task]
    targets = np.clip(raw / scale.denominator(task), -1, 0).astype(np.float32)
    rewards = np.empty_like(targets)
    rewards[:-1] = targets[:-1] - targets[1:]
    rewards[-1] = targets[-1]
    return EpisodeReturns(rewards, targets)


def compute_n_step_advantages(rewards, predicted_values, episode_offsets, *, n_step: int):
    integer(n_step, "advantage.n_step")
    r = finite_vector(rewards, "rewards")
    v = finite_vector(predicted_values, "predicted_values")
    offsets = np.asarray(episode_offsets)
    if (
        v.shape != r.shape
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or offsets[-1] != len(r)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("Invalid episode offsets or reward/value lengths.")
    if v.min() < -1.000001 or v.max() > 0.000001:
        raise ValueError("Predicted values must be normalized to [-1,0].")
    result = np.empty_like(r)
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        steps = np.arange(end - begin)
        next_steps = np.minimum(steps + n_step, end - begin)
        cumulative = np.concatenate(([0.0], r[begin:end].cumsum()))
        bootstrap = np.concatenate((v[begin:end], [0.0]))[next_steps]
        result[begin:end] = cumulative[next_steps] - cumulative[steps] + bootstrap - v[begin:end]
    return result.astype(np.float32)


def label_advantages(
    advantages,
    task_indices,
    *,
    positive_ratio: float,
    interventions=None,
    force_intervention_positive: bool = True,
):
    values = finite_vector(advantages, "advantages")
    tasks = np.asarray(task_indices)
    ratio = number(positive_ratio, "positive_ratio")
    if not 0 < ratio <= 1 or tasks.shape != values.shape or tasks.dtype.kind not in "iu":
        raise ValueError("Expected positive_ratio in (0,1] and integer task indices per frame.")
    thresholds = {
        int(task): float(np.quantile(values[tasks == task], 1 - ratio, method="linear"))
        for task in np.unique(tasks)
    }
    positive = np.array([v >= thresholds[int(t)] for v, t in zip(values, tasks, strict=True)])
    if interventions is not None:
        human = intervention_vector(interventions, length=len(values))
        if force_intervention_positive:
            positive |= np.asarray([item is not None and bool(item) for item in human])
    return AdvantageLabels(values.astype(np.float32), thresholds, positive)
