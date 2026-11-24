"""Extend Pi0 codecs, carrying history separately from base preprocessing."""

from __future__ import annotations

import numpy as np
import torch

from dobot_nexus.models.pi0.data.pipeline import EncodedObservation, Pi0Pipeline
from dobot_nexus.models.pi0.data_adapter_pi0 import Pi0DataAdapter
from dobot_nexus.modules.memory.feature_store import configured_cache_bytes

from .feature_store import FrameFeatureStore
from .types import MemoryObservationBatch, MemorySnapshot


class MemoryPipeline(Pi0Pipeline):
    def __init__(self, root_config, model_config, memory_config):
        super().__init__(root_config, model_config)
        self.memory_config = memory_config
        self.store = None
        if root_config.get("task") == "train":
            path = root_config.data.get("memory_features")
            identity = root_config.data.get("memory_dataset_id")
            if not path or not identity:
                raise ValueError(
                    "Memory training requires data.memory_features and data.memory_dataset_id."
                )
            self.store = FrameFeatureStore(
                path,
                memory_config,
                dataset_id=identity,
                image_field=self.image_fields[model_config.camera_mapping[memory_config.camera]],
                language_field=self.language_key,
                max_cached_episodes=root_config.data.get("memory_cache_episodes", 8),
                max_cached_bytes=configured_cache_bytes(root_config.data),
            )

    def transform_frame(self, frame):
        if self.store is None:
            raise RuntimeError("No training feature store configured.")
        history = self.store.lookup(int(frame["episode_index"]), int(frame["frame_index"]))
        result = super().transform_frame(frame)
        result["memory"] = history
        return result

    def collate_items(self, items):
        result = super().collate_items(items)
        result["memory"] = MemorySnapshot.stack([item["memory"] for item in items])
        return result


class MemoryDataAdapter(Pi0DataAdapter):
    def __init__(self, config, root_config, observation_type, memory_config):
        self.config, self.observation_type = config, observation_type
        self.memory_config = memory_config
        self.pipeline = MemoryPipeline(root_config, config, memory_config)

    def _torch_inputs(self, history, batch_size, device):
        history.validate(self.memory_config, batch_size)
        return tuple(
            torch.from_numpy(np.array(value, copy=True)).to(device)
            for value in (history.features, history.positions, history.mask)
        )

    def encode_train_batch(self, raw_batch, device):
        result = super().encode_train_batch(raw_batch, device)
        result["model_kwargs"] = {
            "memory_inputs": self._torch_inputs(
                raw_batch["memory"], len(raw_batch["actions"]), device
            )
        }
        return result

    def _history(self, observation):
        if not isinstance(observation, MemoryObservationBatch) or observation.memory is None:
            raise ValueError(
                "Memory inference requires a MemoryObservationBatch from MemorySession."
            )
        observation.validate()
        return observation.memory.validate(self.memory_config, observation.batch_size)

    def encode_observation(self, observation):
        history = self._history(observation)
        base = super().encode_observation(observation)
        return EncodedObservation(
            base.observation,
            base.raw_state,
            {"memory_inputs": self._torch_inputs(history, observation.batch_size, "cpu")},
        )
