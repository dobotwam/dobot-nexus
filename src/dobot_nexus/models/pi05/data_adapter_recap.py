from __future__ import annotations

import dataclasses

import torch

from dobot_nexus.algorithms.recap.conditioning import condition_prompt
from dobot_nexus.algorithms.recap.configuration import resolve_inference_config
from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.models.pi05.data_adapter_pi05 import Pi05DataAdapter

from .data.recap import ReCapPipeline


class ReCapDataAdapter(Pi05DataAdapter):
    def __init__(self, config, root_config, observation_type):
        self.config = config
        self.observation_type = observation_type
        self.pipeline = ReCapPipeline(root_config, config)
        self.condition = resolve_inference_config(root_config).condition

    def encode_train_batch(self, raw_batch, device):
        result = super().encode_train_batch(raw_batch, device)
        for key, dtype in (
            ("positive", torch.bool),
            ("condition_tokens", torch.int64),
            ("condition_token_mask", torch.bool),
        ):
            value = raw_batch.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != dtype:
                raise ValueError(f"RECAP batch {key} must have dtype {dtype}.")
            result[key] = value.to(device, non_blocking=True)
        if result["positive"].shape != (result["actions"].shape[0],):
            raise ValueError("RECAP positive must be bool[B].")
        return result

    def _condition_observation(self, observation):
        original = ObservationBatch.from_value(observation)
        return dataclasses.replace(
            original, prompt=[condition_prompt(task, self.condition) for task in original.prompt]
        )

    def encode_observation(self, observation):
        return super().encode_observation(self._condition_observation(observation))
