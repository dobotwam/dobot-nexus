from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from dobot_nexus.core.config import clone_config
from dobot_nexus.models.pi05.configuration_recap import MODEL_TARGET
from tests.models.pi05.fakes import FakeObservation


class FakeCore(torch.nn.Module):
    def __init__(self, action_dim, horizon):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.action_dim, self.horizon = action_dim, horizon

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def compute_loss(self, observation, actions, train=True, rng=None):
        predicted = self.weight + observation.tokenized_prompt[:, 1, None, None] * 0.01
        return (predicted - actions).square().mean(-1)

    def sample_actions(self, observation, **kwargs):
        v = self.weight + observation.tokenized_prompt[:, 1] * 0.01
        return v[:, None, None].expand(-1, self.horizon, self.action_dim)

class FakeCoreConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create(self):
        return FakeCore(self.kwargs["action_dim"], self.kwargs["action_horizon"])

def fake_core_source(*args, **kwargs):
    return SimpleNamespace(
        Pi0Config=FakeCoreConfig,
        Observation=FakeObservation,
        core_dir=Path("fake-core"),
        revision="test",
        embedded=True,
    )

def policy_config(root: Path, offline):
    tokenizer = root / "tokenizer.model"
    tokenizer.write_bytes(b"fake-tokenizer")
    weights = root / "initial.pt"
    torch.save({"weight": torch.tensor(0.2)}, weights)
    return clone_config(
        {
            "schema_version": 1,
            "task": "train",
            "algorithm": "bc",
            "exp": {"name": "recap_test", "output_dir": str(root / "runs"), "seed": 42},
            "model": {
                "name": "pi05",
                "target": MODEL_TARGET,
                "checkpoint": str(weights),
                "state_dim": 2,
                "action_dim": 2,
                "action_horizon": 2,
                "precision": "fp32",
                "parameter_precision": "fp32",
                "recap": {"unconditional_prob": 0.0},
            },
            "data": {
                "path": str(offline.dataset.path),
                "format_version": "v2.1",
                "field_mapping": {"images": dict(offline.dataset.image_fields)},
                "tokenizer_path": str(tokenizer),
                "batch_size": 2,
                "num_workers": 0,
                "recap": {"annotations": str(offline.annotation_dir)},
            },
            "runtime": {"launcher": "local", "device": "cpu", "precision": "fp32"},
            "optim": {"name": "adamw", "lr": 0.01},
            "scheduler": {"name": "constant"},
            "train": {
                "max_steps": 1,
                "save_every": 1,
                "log_every": 1,
                "resume_limit": 0,
                "gradient_checkpointing": False,
                "grad_accum_steps": 1,
            },
        }
    )
