from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.core.contracts import (
    ObservationBatch,
    PolicyOutput,
    PolicySpec,
    TrainOutput,
)
from dobot_nexus.models.base import BaseDataAdapter, BaseVLAModule


class InvalidModel:
    @classmethod
    def from_config(cls, config, *, mode):
        del config, mode
        return cls()


class DummyAdapter(BaseDataAdapter):
    def encode_train_batch(self, raw_batch, device):
        inputs, targets = raw_batch
        return inputs.to(device), targets.to(device)

    def encode_observation(self, observation):
        return ObservationBatch.from_value(observation)

    def decode_actions(self, actions, *, aux=None):
        return PolicyOutput(actions=actions, aux=aux or {})


class DummyVLAModule(BaseVLAModule):
    def __init__(self, config, *, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.adapter = DummyAdapter()

    @classmethod
    def from_config(cls, config, *, mode):
        model = cls(config, mode=mode)
        checkpoint = config.model.get("checkpoint")
        if mode == "infer" and checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
        return model

    def build_train_dataloader(self, config):
        del config
        inputs = torch.ones(4, 2)
        targets = torch.zeros(4, 2)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets), batch_size=2, shuffle=False
        )

    def training_step(self, batch):
        inputs, targets = batch
        prediction = inputs * self.scale
        loss = torch.nn.functional.mse_loss(prediction, targets)
        return TrainOutput(loss=loss, metrics={"dummy_loss": loss.detach()})

    def predict_action_chunk(self, observation, options):
        del options
        batch = observation.batch_size
        horizon = int(self.config.model.action_horizon)
        action_dim = int(self.config.model.action_dim)
        value = float(self.scale.detach().cpu())
        actions = np.full((batch, horizon, action_dim), value, dtype=np.float32)
        return PolicyOutput(actions=actions)

    @property
    def policy_spec(self):
        return PolicySpec(
            model_name="dummy",
            implementation="test",
            camera_names=("base",),
            state_dim=int(self.config.model.state_dim),
            action_dim=int(self.config.model.action_dim),
            action_horizon=int(self.config.model.action_horizon),
            image_size=(8, 8),
        )

    def artifact_assets(self):
        asset = self.config.model.get("asset")
        return {"model.asset": str(asset)} if asset else {}


class SelfLoadingDeploymentVLAModule(DummyVLAModule):
    """Test model that owns backend loading and a final deployment export."""

    def save_nexus_weights(self) -> bool:
        return False

    def export_deployment_artifacts(
        self, *, source_weights: Path, artifact_dir: Path
    ) -> dict[str, str]:
        self.exported_source = source_weights
        relative = Path("deployment/policy_info.txt")
        output = artifact_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"test-deployment")
        return {"policy_info": relative.as_posix()}


class ShuffledDummyVLAModule(DummyVLAModule):
    def build_train_dataloader(self, config):
        inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 10.0 + 1.0
        targets = torch.zeros_like(inputs)
        generator = torch.Generator().manual_seed(int(config.exp.get("seed", 0)))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets),
            batch_size=int(config.data.batch_size),
            shuffle=True,
            generator=generator,
        )


class StochasticDummyVLAModule(DummyVLAModule):
    """Random initialization/loss and deterministic indexed data for resume tests."""

    def __init__(self, config, *, mode):
        super().__init__(config, mode=mode)
        self.scale = torch.nn.Parameter(torch.rand(4))
        self.initial_scale = self.scale.detach().tolist()
        self.trace = []

    def build_train_dataloader(self, config):
        inputs = torch.arange(96, dtype=torch.float32).reshape(24, 4) / 100
        workers = int(config.data.get("num_workers", 0))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, torch.zeros_like(inputs)),
            batch_size=int(config.data.batch_size),
            shuffle=True,
            num_workers=workers,
            persistent_workers=workers > 0,
        )

    def training_step(self, batch):
        inputs, targets = batch
        noise = torch.randn_like(inputs)
        time = torch.distributions.Beta(1.5, 1.0).sample().item()
        python_draw, numpy_draw = random.random(), float(np.random.random())
        self.trace.append(
            {
                "samples": inputs[:, 0].detach().float().cpu().tolist(),
                "noise": noise.detach().float().cpu().tolist(),
                "time": time,
                "python": python_draw,
                "numpy": numpy_draw,
            }
        )
        prediction = inputs * self.scale + noise * 0.01 + (time + python_draw + numpy_draw) * 0.001
        loss = torch.nn.functional.mse_loss(prediction, targets)
        return TrainOutput(loss=loss, metrics={"dummy_loss": loss.detach()})

    def export_deployment_artifacts(self, *, source_weights, artifact_dir):
        # Export must not change the following training draw, on rank 0 or locally.
        random.random()
        np.random.random()
        torch.rand(3)
        return {}

    def predict_action_chunk(self, observation, options):
        del options
        values = self.scale.detach().cpu().numpy()[:int(self.config.model.action_dim)]
        actions = np.broadcast_to(values, (observation.batch_size,
                                          int(self.config.model.action_horizon), len(values)))
        return PolicyOutput(actions=actions.copy())
