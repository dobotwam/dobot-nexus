"""Offline jobs: episode returns, complete value prediction, task-local policy labels."""

from __future__ import annotations

import logging

import numpy as np

from .annotations import check_return_job, object_digest, read_returns, write_labels, write_returns
from .data import build_frame_index, read_episode_outcomes, read_interventions
from .targets import (
    compute_episode_returns,
    compute_n_step_advantages,
    label_advantages,
    resolve_return_scale,
)
from .types import ValueTable

LOGGER = logging.getLogger(__name__)


def run_returns(config):
    index = build_frame_index(config.dataset)
    outcomes = read_episode_outcomes(
        config.dataset, index, dataset_type=config.dataset_type, source=config.outcome
    )
    interventions = read_interventions(config.dataset, index, field=config.intervention_field)
    scale = resolve_return_scale(index, config.returns.c_fail_coef)
    per_episode = [
        compute_episode_returns(
            e.length, outcomes[e.episode_index], task=int(index.task_indices[start]), scale=scale
        )
        for e, start in zip(index.episodes, index.episode_offsets[:-1], strict=True)
    ]
    rewards = np.concatenate([e.rewards for e in per_episode])
    returns = np.concatenate([e.returns for e in per_episode])
    result = write_returns(
        config.annotation_dir,
        index,
        rewards,
        returns,
        scale,
        outcome_signature=object_digest(sorted(outcomes.items())),
        config=config,
        interventions=interventions,
    )
    LOGGER.info(
        "returns frames=%s episodes=%s successes=%s scale=%s output=%s",
        len(index.keys),
        len(index.episodes),
        sum(outcomes.values()),
        scale.task_max_lengths,
        result,
    )
    return result


def predict_values(config, checkpoint) -> ValueTable:
    import torch

    from .value.checkpoint import load_value_checkpoint
    from .value.configuration import job_device
    from .value.dataset import ReCapValueDataset
    from .value.processing import forward_value

    device = job_device(config.labeling.device)
    index = build_frame_index(config.dataset)
    returned = read_returns(config.annotation_dir, expected_index=index)
    check_return_job(returned, config)
    loaded = load_value_checkpoint(checkpoint, device=device, config=config)
    if (
        loaded.scale != returned.scale
        or loaded.dataset_signature != index.dataset_signature
        or loaded.returns_id != returned.manifest["files"]["returns.parquet"]
    ):
        raise ValueError("Value checkpoint was trained with different data/returns/normalization.")
    dataset = ReCapValueDataset(config, index, returned)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.labeling.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.labeling.num_workers,
        collate_fn=loaded.processor.collate,
    )
    lookup = {key: row for row, key in enumerate(index.keys)}
    values = np.full(len(index.keys), np.nan, dtype=np.float32)
    seen = np.zeros(len(values), dtype=np.bool_)
    loaded.model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = loaded.processor.to_device(batch, device)
            prediction = forward_value(loaded.model, batch, with_targets=False)
            for key, value in zip(batch.keys, prediction.values.cpu().numpy(), strict=True):
                row = lookup.get(key)
                if row is None or seen[row]:
                    raise ValueError(f"Unexpected/duplicate value prediction for {key}.")
                values[row], seen[row] = value, True
    if not seen.all() or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite value predictions.")
    return ValueTable(index, values, loaded.scale, loaded.checkpoint_id)


def run_labeling(config, checkpoint):
    values = predict_values(config, checkpoint)
    returned = read_returns(config.annotation_dir, expected_index=values.index)
    advantages = compute_n_step_advantages(
        returned.rewards,
        values.values,
        values.index.episode_offsets,
        n_step=config.advantage.n_step,
    )
    labels = label_advantages(
        advantages,
        values.index.task_indices,
        positive_ratio=config.advantage.positive_ratio,
        interventions=returned.interventions,
        force_intervention_positive=config.advantage.force_intervention_positive,
    )
    result = write_labels(
        config.annotation_dir,
        index=values.index,
        predicted_values=values.values,
        labels=labels,
        value_checkpoint_id=values.checkpoint_id,
        config=config,
    )
    LOGGER.info(
        "labels frames=%s episodes=%s V=[%s,%s] A=[%s,%s] thresholds=%s positive=%s",
        len(values.index.keys),
        len(values.index.episodes),
        values.values.min(),
        values.values.max(),
        advantages.min(),
        advantages.max(),
        labels.thresholds,
        labels.positive.mean(),
    )
    return result
