"""Adapt RECAP labels and condition tokens to the Pi0.5 data pipeline."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from dobot_nexus.algorithms.recap.annotations import load_annotation_manifest, read_policy_labels
from dobot_nexus.algorithms.recap.conditioning import condition_prompt
from dobot_nexus.algorithms.recap.data import (
    _index_integer,
    build_frame_index,
    resolve_policy_dataset_spec,
)
from dobot_nexus.algorithms.recap.types import FrameKey

from .pipeline import Pi05Pipeline


def route_observation(batch, use_condition):
    observation = batch["observation"]
    tokens, mask = observation.tokenized_prompt, observation.tokenized_prompt_mask
    condition_tokens, condition_mask = batch["condition_tokens"], batch["condition_token_mask"]
    if (
        tokens.ndim != 2
        or tokens.dtype != torch.int64
        or mask.dtype != torch.bool
        or condition_tokens.shape != tokens.shape
        or condition_tokens.dtype != tokens.dtype
        or mask.shape != tokens.shape
        or condition_mask.shape != mask.shape
        or condition_mask.dtype != torch.bool
        or use_condition.dtype != torch.bool
        or use_condition.shape != (tokens.shape[0],)
        or len({v.device for v in (tokens, mask, condition_tokens, condition_mask, use_condition)})
        != 1
    ):
        raise ValueError("RECAP condition token/mask batch shapes, dtypes or devices disagree.")
    return dataclasses.replace(
        observation,
        tokenized_prompt=torch.where(use_condition[:, None], condition_tokens, tokens),
        tokenized_prompt_mask=torch.where(use_condition[:, None], condition_mask, mask),
    )


class ReCapPipeline(Pi05Pipeline):
    def __init__(self, root_config, model_config):
        super().__init__(root_config, model_config)
        self.labels = None
        self.label_rows = {}
        if root_config.get("task") == "train":
            directory = root_config.data.recap.annotations
            manifest = load_annotation_manifest(directory, stage="labels")
            if manifest["advantage_config"]["n_step"] != model_config.action_horizon:
                raise ValueError("RECAP advantage.n_step must match model.action_horizon.")
            spec = resolve_policy_dataset_spec(root_config, manifest)
            index = build_frame_index(spec)
            self.labels = read_policy_labels(directory, expected_index=index)
            self.label_rows = self.labels.row_lookup()

    def tokenize_prompt(self, prompt, normalized_state, *, strict=False):
        return super().tokenize_prompt(prompt, normalized_state, strict=True)

    def transform_frame(self, frame):
        if self.labels is None:
            raise RuntimeError("RECAP training labels have not been loaded.")
        key = FrameKey(
            self.labels.index.dataset_id,
            _index_integer(frame["episode_index"], "episode_index"),
            _index_integer(frame["frame_index"], "frame_index"),
        )
        if key not in self.label_rows:
            raise ValueError(f"No RECAP label for {key}.")
        positive = self.labels.positive[self.label_rows[key]]
        conditioned_prompt = condition_prompt(
            self._prompt(frame), "positive" if positive else "negative"
        )
        result = super().transform_frame(frame)
        tokens, mask = self.tokenize_prompt(
            conditioned_prompt, result["state"][: self.model_config.state_dim]
        )
        result.update(
            positive=positive,
            condition_tokens=tokens,
            condition_token_mask=mask,
        )
        return result

    def collate_items(self, items):
        result = super().collate_items(items)
        for name in ("positive", "condition_tokens", "condition_token_mask"):
            result[name] = torch.from_numpy(np.stack([item[name] for item in items]))
        return result
