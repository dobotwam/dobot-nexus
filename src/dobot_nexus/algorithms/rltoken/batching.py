"""Observation batching without importing a simulator or a model implementation."""

from __future__ import annotations

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch


def observation_row(observation, index):
    observation = ObservationBatch.from_value(observation)
    return ObservationBatch(
        {name: value[index : index + 1].copy() for name, value in observation.images.items()},
        observation.state[index : index + 1].copy(),
        observation.prompt[index : index + 1],
        observation.env_ids[index : index + 1] if observation.env_ids is not None else None,
        observation.timestamps[index : index + 1].copy()
        if observation.timestamps is not None
        else None,
    ).validate()


def stack_observations(observations, env_ids):
    observations = [ObservationBatch.from_value(value) for value in observations]
    if not observations or len(observations) != len(env_ids):
        raise ValueError("Observation rows must match the requested environment IDs.")
    cameras = set(observations[0].images)
    for observation, env_id in zip(observations, env_ids, strict=True):
        if observation.batch_size != 1 or set(observation.images) != cameras:
            raise ValueError("Batch rows must have size one and identical camera names.")
        if observation.env_ids is not None and observation.env_ids != [env_id]:
            raise ValueError("Observation env_ids disagree with the requested row order.")
    timestamps = [value.timestamps is not None for value in observations]
    if any(timestamps) and not all(timestamps):
        raise ValueError("Batch rows must agree on whether timestamps are provided.")
    return ObservationBatch(
        {name: np.concatenate([value.images[name] for value in observations]) for name in cameras},
        np.concatenate([value.state for value in observations]),
        [value.prompt[0] for value in observations],
        list(env_ids),
        np.concatenate([value.timestamps for value in observations]) if all(timestamps) else None,
    ).validate()


def environment_batch_size(config):
    kind = config.get("kind", "robot")
    batch_size = config.get("batch_size", 1)
    if kind not in {"robot", "simulation"}:
        raise ValueError("environment.kind must be robot or simulation.")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("environment.batch_size must be a positive integer.")
    if kind == "robot" and batch_size != 1:
        raise ValueError("Robot execution requires environment.batch_size=1.")
    return batch_size
