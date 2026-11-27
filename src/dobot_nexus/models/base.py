from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dobot_nexus.core.contracts import (
    EncodedActionBatch,
    ObservationBatch,
    PolicyOutput,
    PolicySpec,
    RLFeatureBatch,
    SamplingOptions,
    TrainOutput,
)


class BaseDataAdapter(ABC):
    """Model-owned boundary between repository data and a concrete VLA implementation."""

    @abstractmethod
    def encode_train_batch(self, raw_batch: Any, device: torch.device) -> Any:
        raise NotImplementedError

    @abstractmethod
    def encode_observation(self, observation: ObservationBatch) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode_actions(self, actions: Any, *, aux: dict[str, Any] | None = None) -> PolicyOutput:
        raise NotImplementedError

    def encode_executed_actions(
        self, actions: Any, *, aux: dict[str, Any] | None = None
    ) -> EncodedActionBatch:
        raise NotImplementedError(f"{type(self).__name__} does not support RL action encoding.")


class BaseVLAModule(nn.Module, ABC):
    """Model contract used by the common trainer and inference policy.

    A model implementation may live inside or outside this repository. Point
    ``model.target`` at any class implementing this contract; no registry or
    repository-specific subclass layer is required.
    """

    adapter: BaseDataAdapter

    def extract_rl_features(
        self, observation: ObservationBatch, options: SamplingOptions
    ) -> RLFeatureBatch:
        """Extract detached features; vector rollout also needs per-row action_contexts."""
        raise NotImplementedError(f"{type(self).__name__} does not support RL feature extraction.")

    @classmethod
    @abstractmethod
    def from_config(cls, config, *, mode: str) -> BaseVLAModule:
        raise NotImplementedError

    @abstractmethod
    def build_train_dataloader(self, config):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any) -> TrainOutput:
        raise NotImplementedError

    def forward(self, batch: Any) -> TrainOutput:
        """Keep distributed wrappers on the normal nn.Module forward path."""
        return self.training_step(batch)

    @abstractmethod
    def predict_action_chunk(
        self,
        observation: ObservationBatch,
        options: SamplingOptions,
    ) -> PolicyOutput:
        raise NotImplementedError

    @property
    @abstractmethod
    def policy_spec(self) -> PolicySpec:
        raise NotImplementedError

    def get_param_groups(self, optim_config) -> Any:
        return self.parameters()

    def export_backend_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return a checkpoint view consumable by the configured backend.

        The default keeps the complete Nexus module state. Integrations whose
        native loader expects different key names can override this method.
        """
        return state_dict

    def save_nexus_weights(self) -> bool:
        """Whether inference needs the wrapper-shaped Nexus state dict.

        Models that restore the backend checkpoint inside ``from_config`` can
        return ``False`` and avoid storing the same tensors twice.
        """
        return True

    def export_deployment_artifacts(
        self, *, source_weights: Path, artifact_dir: Path
    ) -> dict[str, str]:
        """Create final deployment-only files and return manifest entries."""
        del source_weights, artifact_dir
        return {}

    def artifact_assets(self) -> dict[str, str]:
        """Map resolved config paths to small files that should travel with an artifact."""
        return {}

    def artifact_metadata(self) -> dict[str, Any]:
        metadata = self.policy_spec.to_dict()
        metadata["target"] = f"{type(self).__module__}:{type(self).__name__}"
        return metadata
