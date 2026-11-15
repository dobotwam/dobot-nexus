from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from dobot_nexus.core.contracts import ObservationBatch, PolicyOutput
from dobot_nexus.models.base import BaseDataAdapter
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.data import EncodedObservation, Pi05Pipeline


def _move_model_kwargs(value: Any, device: torch.device) -> Any:
    """Move extension tensors while retaining the argument tree and scalar values."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_model_kwargs(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_model_kwargs(item, device) for item in value)
    if isinstance(value, list):
        return [_move_model_kwargs(item, device) for item in value]
    return value


def _move_observation(observation: Any, device: torch.device) -> Any:
    def move(value: Any) -> Any:
        return (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )

    return dataclasses.replace(
        observation,
        images={key: move(value) for key, value in observation.images.items()},
        image_masks={key: move(value) for key, value in observation.image_masks.items()},
        state=move(observation.state),
        tokenized_prompt=move(observation.tokenized_prompt),
        tokenized_prompt_mask=move(observation.tokenized_prompt_mask),
        token_ar_mask=move(observation.token_ar_mask),
        token_loss_mask=move(observation.token_loss_mask),
        pcd_xyz=move(observation.pcd_xyz),
    )


class Pi05DataAdapter(BaseDataAdapter):
    """Stable Nexus observations to the private Pi0.5 leaf types."""

    def __init__(
        self,
        config: Pi05Config,
        root_config: Any,
        observation_type: type[Any],
    ) -> None:
        self.config = config
        self.observation_type = observation_type
        self.pipeline = Pi05Pipeline(root_config, config)

    def encode_train_batch(self, raw_batch: Any, device: torch.device) -> dict[str, Any]:
        if not isinstance(raw_batch, dict) or "actions" not in raw_batch:
            raise TypeError("Pi0.5 training batches must be dictionaries with actions.")
        observation = self.observation_type.from_dict(
            {
                "image": raw_batch["image"],
                "image_mask": raw_batch["image_mask"],
                "state": raw_batch["state"],
                "tokenized_prompt": raw_batch["tokenized_prompt"],
                "tokenized_prompt_mask": raw_batch["tokenized_prompt_mask"],
            }
        )
        return {
            "observation": _move_observation(observation, device),
            "actions": raw_batch["actions"].to(device, non_blocking=True),
        }

    def encode_observation(self, observation: ObservationBatch) -> EncodedObservation:
        observation = ObservationBatch.from_value(observation)
        if observation.state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"Expected state_dim={self.config.state_dim}, got {observation.state.shape[-1]}."
            )
        missing = sorted(set(self.config.camera_names) - set(observation.images))
        if missing:
            raise KeyError(f"Pi0.5 observation is missing cameras: {missing}.")
        encoded, raw_state = self.pipeline.transform_runtime(observation)
        return EncodedObservation(
            observation=self.observation_type.from_dict(encoded), raw_state=raw_state
        )

    def move_encoded(self, value: EncodedObservation, device: torch.device) -> EncodedObservation:
        return EncodedObservation(
            observation=_move_observation(value.observation, device),
            raw_state=value.raw_state,
            model_kwargs=_move_model_kwargs(value.model_kwargs, device),
        )

    def decode_actions(
        self,
        actions: Any,
        *,
        aux: dict[str, Any] | None = None,
    ) -> PolicyOutput:
        raw_aux = dict(aux or {})
        raw_state = raw_aux.pop("raw_state", None)
        if raw_state is None:
            raise ValueError("Pi0.5 action decoding requires the source robot state.")
        decoded = self.pipeline.decode_actions(actions, np.asarray(raw_state))
        timing = dict(raw_aux.pop("timing", {}))
        return PolicyOutput(actions=decoded, timing=timing, aux=raw_aux)
